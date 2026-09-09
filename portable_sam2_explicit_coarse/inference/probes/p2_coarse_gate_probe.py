#!/usr/bin/env python3
"""First gate for the natural P2 coarse-to-dense route.

This is deliberately an *observation* probe: it never substitutes GT or
changes a tensor returned to the model.  On a frozen P2 checkpoint it records
class-aware IoU-matched proposals' natural raw/refined coarse masks and the
two canvases produced by the production renderer.  Statistics are pooled by
image and uncertainty is a paired **image** bootstrap, never an ROI bootstrap.

The gate is passed only when (1) natural refined coarse IoU and (2) the
renderer-delivered, proposal-support canvas IoU both have a positive 95% paired
bootstrap lower bound, and (3) the renderer actually changed support pixels.
This is a causal reachability gate, not a claim of final COCO mAP improvement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import box_iou

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from inference.infer_from_checkpoint import (  # noqa: E402
    _build_data_samples, _build_loader, _extract_instances_numpy,
    _load_checkpoint, _load_model_state, _register_and_build,
    _resolve_dataset_contract, _resolve_model_config, _select_state_dict,
    _snapshot, _write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--weights", choices=("model", "ema"), default="model")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--skip-parity-check", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _hash_update(hasher: "hashlib._Hash", value: Any) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    hasher.update(str((array.dtype.str, array.shape)).encode("ascii"))
    hasher.update(array.tobytes())


def _gt_cache(samples: Iterable[Any]) -> List[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    result = []
    for sample in samples:
        instances = getattr(sample, "gt_instances", None)
        if instances is None or len(getattr(instances, "bboxes", [])) == 0:
            result.append(None)
            continue
        result.append((
            instances.bboxes.detach().cpu(), instances.labels.detach().cpu(),
            instances.masks.to_tensor(dtype=torch.float32, device="cpu"),
        ))
    return result


def _roi_target_grid(mask: torch.Tensor, box: torch.Tensor, grid_hw: Tuple[int, int]) -> Optional[torch.Tensor]:
    """Same integer crop and nearest resize convention as the coarse target."""
    height, width = mask.shape[-2:]
    x1f, y1f, x2f, y2f = [float(value) for value in box]
    # Keep exactly the target quantisation used by get_coarse_targets:
    # floor top-left / ceil bottom-right, then clamp with >=1px extent.
    x1, y1, x2, y2 = math.floor(x1f), math.floor(y1f), math.ceil(x2f), math.ceil(y2f)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, max(x1 + 1, x2)), min(height, max(y1 + 1, y2))
    crop = mask[y1:y2, x1:x2]
    if crop.numel() == 0:
        return None
    return F.interpolate(crop[None, None].float(), size=grid_hw, mode="nearest")[0, 0] >= 0.5


def _matched_index(box: torch.Tensor, label: torch.Tensor, entry: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], threshold: float) -> Optional[int]:
    if entry is None:
        return None
    gt_boxes, gt_labels, _ = entry
    same_class = gt_labels == label
    if not bool(same_class.any()):
        return None
    ious = box_iou(box[None].float(), gt_boxes.float())[0]
    ious = torch.where(same_class, ious, torch.full_like(ious, -1.0))
    index = int(ious.argmax())
    return index if float(ious[index]) >= threshold else None


def _box_support(box: torch.Tensor, height: int, width: int, image_size: int, device: torch.device) -> torch.Tensor:
    """Exact floor/ceil paste domain of ``paste_roi_to_full_canvas``."""
    x1, y1, x2, y2 = (float(value) for value in box)
    x1i = int(max(0, min(width - 1, math.floor(x1 * width / image_size))))
    y1i = int(max(0, min(height - 1, math.floor(y1 * height / image_size))))
    x2i = int(max(x1i + 1, min(width, math.ceil(x2 * width / image_size))))
    y2i = int(max(y1i + 1, min(height, math.ceil(y2 * height / image_size))))
    support = torch.zeros((height, width), dtype=torch.bool, device=device)
    support[y1i:y2i, x1i:x2i] = True
    return support


def _counts(pred: torch.Tensor, target: torch.Tensor, support: Optional[torch.Tensor] = None) -> Dict[str, int]:
    if support is not None:
        pred, target = pred[support], target[support]
    inter = int((pred & target).sum())
    pred_sum, target_sum = int(pred.sum()), int(target.sum())
    return {"inter": inter, "pred": pred_sum, "target": target_sum, "union": pred_sum + target_sum - inter, "pixels": int(pred.numel())}


def _metric_from_counts(counts: Mapping[str, float], kind: str) -> float:
    if kind == "iou":
        return float(counts["inter"]) / max(float(counts["union"]), 1.0)
    if kind == "dice":
        return 2.0 * float(counts["inter"]) / max(float(counts["pred"] + counts["target"]), 1.0)
    raise ValueError(kind)


def _merge_counts(records: Iterable[Mapping[str, Any]], prefix: str) -> Dict[str, int]:
    return {key: sum(int(record[prefix + key]) for record in records) for key in ("inter", "pred", "target", "union", "pixels")}


def paired_image_bootstrap(records: List[Mapping[str, Any]], *, resamples: int, seed: int) -> Dict[str, Any]:
    """Paired bootstrap over images; ROI rows are never independently sampled."""
    if not records:
        raise RuntimeError("no images reached P2 gate statistics")
    groups = [("coarse", "coarse_raw_", "coarse_refined_"), ("canvas", "canvas_raw_", "canvas_refined_")]
    point = {}
    for name, raw, refined in groups:
        point[name] = {
            metric: _metric_from_counts(_merge_counts(records, raw), metric)
            for metric in ("iou", "dice")
        }
        point[name].update({
            "refined_" + metric: _metric_from_counts(_merge_counts(records, refined), metric)
            for metric in ("iou", "dice")
        })
        for metric in ("iou", "dice"):
            point[name]["delta_" + metric] = point[name]["refined_" + metric] - point[name][metric]
    point["canvas_changed_fraction"] = sum(int(row["canvas_changed_pixels"]) for row in records) / max(sum(int(row["canvas_support_pixels"]) for row in records), 1)
    if resamples == 0:
        return {"point": point, "resamples": 0, "ci": {}}
    rng, sample_count = np.random.default_rng(seed), len(records)
    deltas: Dict[str, List[float]] = defaultdict(list)
    for _ in range(resamples):
        sampled = [records[int(index)] for index in rng.integers(0, sample_count, size=sample_count)]
        for name, raw, refined in groups:
            for metric in ("iou", "dice"):
                delta = _metric_from_counts(_merge_counts(sampled, refined), metric) - _metric_from_counts(_merge_counts(sampled, raw), metric)
                deltas[f"{name}_delta_{metric}"].append(delta)
    ci = {key: {"low": float(np.quantile(values, .025)), "high": float(np.quantile(values, .975)), "p_gt_zero": float(np.mean(np.asarray(values) > 0))} for key, values in deltas.items()}
    return {"point": point, "resamples": resamples, "seed": seed, "ci": ci}


def _gate(bootstrap: Mapping[str, Any]) -> Dict[str, Any]:
    point, ci = bootstrap["point"], bootstrap["ci"]
    if not ci:
        return {"verdict": "not_decided", "reason": "resamples=0 is reproduction-only"}
    coarse_ok = ci["coarse_delta_iou"]["low"] > 0
    canvas_ok = ci["canvas_delta_iou"]["low"] > 0
    changed_ok = point["canvas_changed_fraction"] > 0
    passed = coarse_ok and canvas_ok and changed_ok
    return {"verdict": "pass" if passed else "fail", "coarse_iou_ci_positive": coarse_ok, "canvas_iou_ci_positive": canvas_ok, "canvas_changed": changed_ok, "rule": "both paired-image 95% IoU lower bounds > 0 and natural renderer support changes"}


def main() -> int:
    args = parse_args()
    if not 0 < args.match_iou <= 1 or args.resamples < 0:
        raise SystemExit("--match-iou must be in (0,1] and --resamples non-negative")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint, snapshot = _load_checkpoint(checkpoint_path), None
    snapshot = _snapshot(checkpoint)
    namespace = SimpleNamespace(checkpoint=str(checkpoint_path), config=None, split=args.split, data_root=None, ann_file=None, image_subdir=None, image_size=None, batch_size=None, sam2_repo=None, sam2_ckpt=None)
    model_config, config_source = _resolve_model_config(namespace, snapshot)
    model = _register_and_build(model_config)
    report = _load_model_state(model, _select_state_dict(checkpoint, args.weights), allow_nonstrict=False)
    if report["missing_keys"] or report["unexpected_keys"]:
        raise RuntimeError(f"strict checkpoint load failed: {report}")
    device = torch.device(args.device)
    model.to(device).eval()
    head = model.roi_head.mask_head
    if head.p2_boundary_refiner is None:
        raise SystemExit("first gate requires a natural P2BoundaryRefiner checkpoint")
    if not (head.use_shape_dense and head.explicit_use_dense_prompt and head.shape_injector is not None):
        raise SystemExit("first gate requires active ShapePriorInjector -> dense canvas consumption")
    # A state-dict can contain non-zero P2 weights while its production input
    # route is dead.  This deterministic synthetic call is only a component
    # liveness discriminator; it is never used in the gate metric.
    with torch.inference_mode():
        refiner = head.p2_boundary_refiner
        side = int(refiner.coarse_size)
        yy, xx = torch.meshgrid(torch.arange(side, device=device), torch.arange(side, device=device), indexing="ij")
        synthetic_raw = (((xx - side / 2).square() + (yy - side / 2).square()) < (side / 4) ** 2).float()[None, None] * 4 - 2
        synthetic = refiner(
            torch.randn(1, refiner.p2_in_channels, 256, 256, device=device),
            torch.tensor([[0., 0., 0., 1024., 1024.]], device=device), synthetic_raw,
        )
        synthetic_delta_abs = float(synthetic["delta_logits"].abs().mean())
    contract, loader = _resolve_dataset_contract(namespace, snapshot), None
    loader = _build_loader(contract, 0)
    out_dir = Path(args.output_dir).expanduser().resolve(); out_dir.mkdir(parents=True, exist_ok=True)

    def run(observe: bool) -> Dict[str, Any]:
        state: Dict[str, Any] = {"ids": None, "boxes": None, "labels": None, "gts": None, "label_queue": None, "cursor": 0, "image_ids": None, "actual_p2_abs_sum": 0.0, "actual_p2_count": 0, "residual_input_abs_sum": 0.0, "residual_input_count": 0, "residual_output_abs_sum": 0.0, "residual_output_count": 0, "actual_delta_abs_sum": 0.0, "actual_delta_count": 0}
        images: Dict[int, Dict[str, Any]] = {}
        detector_hash, output_hash = hashlib.sha256(), hashlib.sha256()
        orig_predict, orig_coarse = model.roi_head.predict_mask, head._forward_coarse_p2
        p_had, p_saved = "predict_mask" in model.roi_head.__dict__, model.roi_head.__dict__.get("predict_mask")
        c_had, c_saved = "_forward_coarse_p2" in head.__dict__, head.__dict__.get("_forward_coarse_p2")

        def ensure(image_id: int) -> Dict[str, Any]:
            if image_id not in images:
                row = {"image_id": image_id, "matched_rois": 0, "unmatched_rois": 0, "support_valid_rois": 0, "rejected_support_rois": 0, "canvas_changed_pixels": 0, "canvas_support_pixels": 0, "coarse_delta_abs_sum": 0.0, "coarse_pixels": 0, "canvas_delta_abs_sum": 0.0}
                for prefix in ("coarse_raw_", "coarse_refined_", "canvas_raw_", "canvas_refined_"):
                    row.update({prefix + key: 0 for key in ("inter", "pred", "target", "union", "pixels")})
                images[image_id] = row
            return images[image_id]

        def wrapped_predict(*fn_args: Any, **kwargs: Any) -> Any:
            results = kwargs.get("results_list", fn_args[2] if len(fn_args) > 2 else None)
            state["label_queue"] = torch.cat([x.labels.detach() for x in results], dim=0) if results else torch.zeros(0, dtype=torch.long)
            state["cursor"] = 0
            return orig_predict(*fn_args, **kwargs)

        def pre_hook(module: Any, fn_args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
            del module, fn_args
            boxes, ids, queue = kwargs.get("boxes"), kwargs.get("roi_img_ids"), state["label_queue"]
            state["boxes"], state["ids"] = (boxes.detach().cpu() if boxes is not None else None), (ids.detach().cpu() if ids is not None else None)
            if boxes is None or queue is None: return
            start, end = int(state["cursor"]), int(state["cursor"]) + int(boxes.shape[0])
            if end > int(queue.numel()): raise RuntimeError("proposal-label queue underflow")
            state["labels"], state["cursor"] = queue[start:end].detach().cpu(), end

        def wrapped_coarse(*fn_args: Any, **kwargs: Any) -> Any:
            output = orig_coarse(*fn_args, **kwargs)
            returned_logits, outputs, returned_canvas, _ = output
            # The first return slot is the *downstream refined* coarse mask by
            # contract.  Raw is intentionally retained only in coarse_outputs.
            # Do not compare that first slot to refined_logits: that silently
            # compares refined to itself and manufactures a zero-effect result.
            if returned_logits is None or outputs is None or outputs.get("raw_logits") is None or outputs.get("refined_logits") is None:
                raise RuntimeError("P2 checkpoint produced no natural raw/refined coarse outputs")
            raw, refined = outputs["raw_logits"], outputs["refined_logits"]
            boxes_device = fn_args[2] if len(fn_args) > 2 else kwargs["boxes"]
            raw_canvas, _, _ = head._shape_prior_to_prompt_mask(raw, boxes_device)
            # The production forward already rendered this exact tensor from
            # refined; assert rather than introducing a second semantic route.
            if not torch.equal(returned_canvas, head._shape_prior_to_prompt_mask(refined, boxes_device)[0]):
                raise RuntimeError("returned refined canvas differs from production renderer")
            refined_canvas = returned_canvas
            refined, boxes, ids, labels, gts = refined, state["boxes"], state["ids"], state["labels"], state["gts"]
            p2_feature = fn_args[4] if len(fn_args) > 4 else kwargs.get("p2_feature")
            if p2_feature is not None:
                state["actual_p2_abs_sum"] += float(p2_feature.abs().sum())
                state["actual_p2_count"] += int(p2_feature.numel())
            state["actual_delta_abs_sum"] += float((refined - raw).abs().sum())
            state["actual_delta_count"] += int(raw.numel())
            if boxes is None or ids is None or labels is None or gts is None or len(labels) != raw.shape[0]:
                raise RuntimeError("cannot establish aligned proposal/label/GT state")
            support_valid = outputs.get("support_valid")
            for i in range(raw.shape[0]):
                local_img = int(ids[i]); image_id = int(state["image_ids"][local_img]); row = ensure(image_id)
                entry = gts[local_img] if 0 <= local_img < len(gts) else None
                index = _matched_index(boxes[i], labels[i], entry, args.match_iou)
                if index is None: row["unmatched_rois"] += 1; continue
                target = _roi_target_grid(entry[2][index], boxes[i], tuple(raw.shape[-2:]))
                if target is None: row["unmatched_rois"] += 1; continue
                row["matched_rois"] += 1
                valid = bool(support_valid[i]) if support_valid is not None else False
                row["support_valid_rois" if valid else "rejected_support_rois"] += 1
                target = target.to(device=raw.device)
                row["coarse_delta_abs_sum"] += float((refined[i, 0] - raw[i, 0]).abs().sum())
                row["coarse_pixels"] += int(raw[i, 0].numel())
                for prefix, logits in (("coarse_raw_", raw[i, 0]), ("coarse_refined_", refined[i, 0])):
                    for key, value in _counts(logits >= 0, target).items(): row[prefix + key] += value
                canvas_target = F.interpolate(entry[2][index][None, None].to(device=raw.device), size=raw_canvas.shape[-2:], mode="nearest")[0, 0] >= .5
                support = _box_support(boxes[i], raw_canvas.shape[-2], raw_canvas.shape[-1], int(head.prompt_encoder_image_size), raw.device)
                for prefix, canvas in (("canvas_raw_", raw_canvas[i, 0]), ("canvas_refined_", refined_canvas[i, 0])):
                    for key, value in _counts(canvas >= 0, canvas_target, support).items(): row[prefix + key] += value
                row["canvas_changed_pixels"] += int(((raw_canvas[i, 0] - refined_canvas[i, 0]).abs() > 0)[support].sum())
                row["canvas_support_pixels"] += int(support.sum())
                row["canvas_delta_abs_sum"] += float((raw_canvas[i, 0] - refined_canvas[i, 0]).abs()[support].sum())
            return output  # observation only: exact production tensors continue downstream

        handle = residual_handle = residual_output_handle = None
        if observe:
            handle = head.register_forward_pre_hook(pre_hook, with_kwargs=True); model.roi_head.predict_mask = wrapped_predict; head._forward_coarse_p2 = wrapped_coarse
            def residual_pre_hook(module: Any, inputs: Tuple[Any, ...]) -> None:
                del module
                state["residual_input_abs_sum"] += float(inputs[0].abs().sum())
                state["residual_input_count"] += int(inputs[0].numel())
            residual_handle = head.p2_boundary_refiner.residual_head.register_forward_pre_hook(residual_pre_hook)
            def residual_output_hook(module: Any, inputs: Tuple[Any, ...], output: torch.Tensor) -> None:
                del module, inputs
                state["residual_output_abs_sum"] += float(output.abs().sum())
                state["residual_output_count"] += int(output.numel())
            residual_output_handle = head.p2_boundary_refiner.residual_head.register_forward_hook(residual_output_hook)
        try:
            with torch.inference_mode():
                for batch_index, batch in enumerate(loader):
                    if args.max_batches and batch_index >= args.max_batches: break
                    inputs, samples = _build_data_samples(batch, device)
                    processed = model.data_preprocessor({"inputs": inputs, "data_samples": samples}, training=False)
                    state["gts"] = _gt_cache(processed["data_samples"])
                    state["image_ids"] = [int(x.get("image_id", x.get("scene_id", batch_index * 1000 + j))) for j, x in enumerate(batch["img_metas"])]
                    outputs = model.predict(processed["inputs"], processed["data_samples"], rescale=False)
                    if observe and state["cursor"] != int(state["label_queue"].numel()): raise RuntimeError("proposal-label queue not consumed exactly")
                    for j, output in enumerate(outputs):
                        meta, shape = dict(batch["img_metas"][j]), batch["img_metas"][j]["img_shape"]
                        pred = output.pred_instances if hasattr(output, "pred_instances") else output
                        dt = _extract_instances_numpy(pred, shape); image_id = state["image_ids"][j]
                        _hash_update(detector_hash, np.asarray([image_id], dtype=np.int64))
                        for key in ("bboxes", "scores", "labels"):
                            _hash_update(detector_hash, dt[key]); _hash_update(output_hash, dt[key])
                        _hash_update(output_hash, dt["masks"])
        finally:
            if handle is not None:
                handle.remove()
                assert residual_handle is not None
                residual_handle.remove()
                assert residual_output_handle is not None
                residual_output_handle.remove()
                if p_had: model.roi_head.predict_mask = p_saved
                else: delattr(model.roi_head, "predict_mask")
                if c_had: head._forward_coarse_p2 = c_saved
                else: delattr(head, "_forward_coarse_p2")
        return {"detector_sha256": detector_hash.hexdigest(), "output_sha256": output_hash.hexdigest(), "route_liveness": {"p2_feature_abs_mean": state["actual_p2_abs_sum"] / max(state["actual_p2_count"], 1), "residual_head_input_abs_mean": state["residual_input_abs_sum"] / max(state["residual_input_count"], 1), "residual_head_output_abs_mean": state["residual_output_abs_sum"] / max(state["residual_output_count"], 1), "natural_delta_abs_mean": state["actual_delta_abs_sum"] / max(state["actual_delta_count"], 1)}, "image_records": list(images.values())}

    standard = None if args.skip_parity_check else run(False)
    observed = run(True)
    if standard is not None:
        if observed["detector_sha256"] != standard["detector_sha256"]: raise RuntimeError("detector changed under observation hook")
        if observed["output_sha256"] != standard["output_sha256"]: raise RuntimeError("observation hook changed final masks")
    records = sorted(observed.pop("image_records"), key=lambda x: x["image_id"])
    bootstrap = paired_image_bootstrap(records, resamples=args.resamples, seed=args.seed)
    telemetry = {
        "matched_rois": sum(int(row["matched_rois"]) for row in records),
        "support_valid_rois": sum(int(row["support_valid_rois"]) for row in records),
        "rejected_support_rois": sum(int(row["rejected_support_rois"]) for row in records),
        "mean_abs_natural_coarse_delta": sum(float(row["coarse_delta_abs_sum"]) for row in records) / max(sum(int(row["coarse_pixels"]) for row in records), 1),
        "mean_abs_renderer_canvas_delta_in_support": sum(float(row["canvas_delta_abs_sum"]) for row in records) / max(sum(int(row["canvas_support_pixels"]) for row in records), 1),
        "synthetic_component_delta_abs_mean": synthetic_delta_abs,
        **observed["route_liveness"],
    }
    summary = {"checkpoint": str(checkpoint_path), "weights": args.weights, "config_source": config_source, "strict_load": report, "dataset_contract": contract, "standard": standard, "observed": observed, "matched_images": len(records), "image_records": str(out_dir / "image_metrics.json"), "telemetry": telemetry, "bootstrap": bootstrap, "gate": _gate(bootstrap), "method": "natural P2 raw/refined logits; class-aware IoU-matched proposal rows; production canvas renderer; image-level paired bootstrap"}
    _write_json(out_dir / "image_metrics.json", records)
    _write_json(Path(args.output).expanduser().resolve() if args.output else out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
