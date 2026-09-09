#!/usr/bin/env python3
"""Frozen A0 coarse-to-mining Oracle.

This diagnostic preserves the original design's causal chain: a ROI-local
coarse mask is mined into 2P2N points and transformed into the dense canvas.
All A0 parameters, proposals, boxes, classes, and detector scores stay frozen.
For class-aware, IoU-matched proposals only, it substitutes an exact GT coarse
target at either one of the two *consumers*:

``raw``   raw coarse drives points and dense canvas (identity control)
``points`` GT coarse drives only ShapePointMiner
``dense``  GT coarse drives only _shape_prior_to_prompt_mask
``both``   GT coarse drives both consumers

It is an inference-only upper-bound probe, never a trainable method: GT does
not select proposals, alter detector output, or enter any model parameter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
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
    _build_data_samples,
    _build_loader,
    _extract_instances_numpy,
    _load_checkpoint,
    _load_model_state,
    _register_and_build,
    _resolve_dataset_contract,
    _resolve_model_config,
    _select_state_dict,
    _snapshot,
    _write_json,
)
from utils.coco_eval_utils import (  # noqa: E402
    VHR10_CATEGORIES,
    _mask_to_rle,
    _xyxy_to_xywh,
    build_coco_gt_and_dt,
    run_coco_eval,
)


VALID_MODES = ("raw", "points", "dense", "both")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--weights", choices=("model", "ema"), default="model")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--modes", nargs="+", choices=VALID_MODES, default=list(VALID_MODES))
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--logit-span", type=float, default=8.0)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--skip-parity-check", action="store_true")
    parser.add_argument("--canonical-dir", default=None,
                        help="Production A0 inference dir required for legacy contract drift.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _hash_update(hasher: "hashlib._Hash", value: Any) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    hasher.update(str((array.dtype.str, array.shape)).encode("ascii"))
    hasher.update(array.tobytes())


def _roi_target_grid(instance_mask: torch.Tensor, box: torch.Tensor, grid_hw: Tuple[int, int]) -> Optional[torch.Tensor]:
    """Training-equivalent crop then nearest resize into the coarse ROI grid."""
    height, width = instance_mask.shape[-2:]
    x1, y1, x2, y2 = [float(value) for value in box]
    x1 = max(0.0, min(x1, width - 1))
    y1 = max(0.0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    crop = instance_mask[int(y1):int(y2), int(x1):int(x2)]
    if crop.numel() == 0:
        return None
    return F.interpolate(
        crop[None, None].float(), size=grid_hw, mode="nearest"
    )[0, 0] >= 0.5


def _match_gt_index(
    proposal_box: torch.Tensor,
    proposal_label: torch.Tensor,
    entry: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    match_iou: float,
) -> Optional[int]:
    """Class-aware proposal-to-GT matching used in every Oracle cell."""
    if entry is None:
        return None
    gt_boxes, gt_labels, _ = entry
    same_class = gt_labels == proposal_label
    if not bool(same_class.any()):
        return None
    ious = box_iou(proposal_box[None].float(), gt_boxes.float())[0]
    ious = torch.where(same_class, ious, torch.full_like(ious, -1.0))
    index = int(ious.argmax())
    return index if float(ious[index]) >= match_iou else None


def build_gt_coarse_candidate(
    raw_logits: torch.Tensor,
    roi_img_ids: torch.Tensor,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    gt_by_image: List[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    *,
    match_iou: float,
    logit_span: float,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Return raw logits with only matched ROI rows replaced by signed GT logits."""
    if raw_logits.ndim != 4 or raw_logits.shape[1] != 1:
        raise RuntimeError(f"expected [N,1,H,W] coarse logits, got {tuple(raw_logits.shape)}")
    count = raw_logits.shape[0]
    if not (len(roi_img_ids) == len(boxes) == len(labels) == count):
        raise RuntimeError("coarse Oracle ROI metadata is not aligned with coarse rows")
    candidate = raw_logits.clone()
    stats = {"matched_rois": 0, "unmatched_rois": 0, "replaced_pixels": 0}
    grid_hw = (int(raw_logits.shape[-2]), int(raw_logits.shape[-1]))
    for index in range(count):
        image_index = int(roi_img_ids[index])
        entry = gt_by_image[image_index] if 0 <= image_index < len(gt_by_image) else None
        gt_index = _match_gt_index(boxes[index], labels[index], entry, match_iou)
        if gt_index is None:
            stats["unmatched_rois"] += 1
            continue
        assert entry is not None
        target = _roi_target_grid(entry[2][gt_index], boxes[index], grid_hw)
        if target is None:
            stats["unmatched_rois"] += 1
            continue
        target = target.to(device=raw_logits.device)
        candidate[index, 0] = torch.where(
            target,
            torch.full_like(candidate[index, 0], logit_span),
            torch.full_like(candidate[index, 0], -logit_span),
        )
        stats["matched_rois"] += 1
        stats["replaced_pixels"] += int(target.numel())
    return candidate, stats


def select_consumer_outputs(
    mode: str,
    raw_logits: torch.Tensor,
    coarse_outputs: Mapping[str, Any],
    raw_canvas: torch.Tensor,
    raw_valid: torch.Tensor,
    candidate_logits: torch.Tensor,
    candidate_canvas: torch.Tensor,
    candidate_valid: torch.Tensor,
) -> Tuple[torch.Tensor, Mapping[str, Any], torch.Tensor, torch.Tensor]:
    """Keep the four pre-registered interventions mutually exclusive.

    The unchanged ``coarse_outputs`` object is intentionally returned: inference
    has no coarse loss, and this makes the Oracle affect only the two named
    downstream consumers rather than any auxiliary side channel.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"unknown coarse-to-mining mode {mode!r}")
    return (
        candidate_logits if mode in ("points", "both") else raw_logits,
        coarse_outputs,
        candidate_canvas if mode in ("dense", "both") else raw_canvas,
        candidate_valid if mode in ("dense", "both") else raw_valid,
    )


def _gt_cache(data_samples: Iterable[Any]) -> List[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    cached = []
    for sample in data_samples:
        instances = getattr(sample, "gt_instances", None)
        if instances is None or len(getattr(instances, "bboxes", [])) == 0:
            cached.append(None)
            continue
        cached.append((
            instances.bboxes.detach().cpu(),
            instances.labels.detach().cpu(),
            instances.masks.to_tensor(dtype=torch.float32, device="cpu"),
        ))
    return cached


def _records(
    all_gt: List[dict], all_dt: List[dict], all_metas: List[dict]
) -> Tuple[List[dict], List[dict], List[dict]]:
    gt_records, dt_records, images = [], [], []
    for gt, dt, meta in zip(all_gt, all_dt, all_metas):
        image_id = int(meta.get("image_id", meta.get("scene_id", 0)))
        height, width = (int(value) for value in meta["img_shape"][:2])
        images.append({"image_id": image_id, "height": height, "width": width})
        for index, label in enumerate(np.asarray(gt["labels"])):
            gt_records.append({
                "image_id": image_id,
                "category_id": int(label) + 1,
                "bbox": [float(value) for value in _xyxy_to_xywh(gt["bboxes"][index])],
                "segmentation": _mask_to_rle(gt["masks"][index]),
                "iscrowd": 0,
            })
        for index, box in enumerate(np.asarray(dt["bboxes"])):
            x1, y1, x2, y2 = (float(value) for value in box)
            dt_records.append({
                "image_id": image_id,
                "category_id": int(dt["labels"][index]) + 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(dt["scores"][index]),
                "mask_score": float(dt.get("mask_scores", dt["scores"])[index]),
                "segmentation": _mask_to_rle(dt["masks"][index]),
            })
    return gt_records, dt_records, images


def main() -> int:
    args = parse_args()
    if args.logit_span <= 0:
        raise SystemExit("--logit-span must be positive")
    if not args.max_batches:
        if args.skip_parity_check:
            raise SystemExit("full coarse-to-mining Oracle refuses --skip-parity-check")
        if set(args.modes) != set(VALID_MODES) or len(args.modes) != len(VALID_MODES):
            raise SystemExit("full coarse-to-mining Oracle requires exactly raw points dense both")
        if not args.canonical_dir:
            raise SystemExit("full coarse-to-mining Oracle requires --canonical-dir")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = _load_checkpoint(checkpoint_path)
    snapshot = _snapshot(checkpoint)
    namespace = SimpleNamespace(
        checkpoint=str(checkpoint_path), config=None, split=args.split,
        data_root=None, ann_file=None, image_subdir=None, image_size=None,
        batch_size=None, sam2_repo=None, sam2_ckpt=None,
    )
    contract_drift = None
    try:
        model_config, config_source = _resolve_model_config(namespace, snapshot)
    except RuntimeError as exc:
        if "does not match its saved architecture contract" not in str(exc):
            raise
        model_config = snapshot["model_config"]
        config_source = "embedded_config_strict_state_only_legacy_contract_drift"
        contract_drift = str(exc)
    model = _register_and_build(model_config)
    report = _load_model_state(
        model, _select_state_dict(checkpoint, args.weights), allow_nonstrict=False
    )
    if report["missing_keys"] or report["unexpected_keys"]:
        raise RuntimeError(f"strict checkpoint load failed: {report}")
    device = torch.device(args.device)
    model.to(device).eval()
    head = model.roi_head.mask_head
    if head.p2_boundary_refiner is not None:
        raise RuntimeError("coarse-to-mining Oracle requires P2-off A0, not a P2 checkpoint")
    if not (head.prompt_sparse_mode == "shape_point" and head.shape_injector is not None):
        raise RuntimeError("Oracle requires ShapePriorInjector -> ShapePointMiner A0 wiring")
    if not (head.explicit_use_point_prompt and head.explicit_use_dense_prompt):
        raise RuntimeError("Oracle requires A0 points+box+dense prompt consumption")
    if not head.use_shape_dense or head.prompt_encoder is None:
        raise RuntimeError("Oracle requires an active frozen PromptEncoder dense path")
    if head.quality_head_enabled:
        raise RuntimeError("quality-head checkpoints are excluded: ranking would be a second intervention")

    contract = _resolve_dataset_contract(namespace, snapshot)
    loader = _build_loader(contract, 0)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    def run_pass(mode: str, *, unhooked: bool = False) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "ids": None, "boxes": None, "labels": None, "gts": None,
            "label_queue": None, "label_cursor": 0,
        }
        stats = {"matched_rois": 0, "unmatched_rois": 0, "replaced_pixels": 0}
        detector_hash, output_hash = hashlib.sha256(), hashlib.sha256()
        all_gt: List[dict] = []
        all_dt: List[dict] = []
        all_metas: List[dict] = []
        original_predict_mask = model.roi_head.predict_mask
        original_coarse = head._forward_coarse_p2
        predict_had_attr = "predict_mask" in model.roi_head.__dict__
        predict_saved_attr = model.roi_head.__dict__.get("predict_mask")
        coarse_had_attr = "_forward_coarse_p2" in head.__dict__
        coarse_saved_attr = head.__dict__.get("_forward_coarse_p2")

        def wrapped_predict_mask(*fn_args: Any, **fn_kwargs: Any) -> Any:
            results_list = fn_kwargs.get("results_list")
            if results_list is None:
                results_list = fn_args[2]
            state["label_queue"] = (
                torch.cat([result.labels.detach() for result in results_list], dim=0)
                if results_list else torch.zeros(0, dtype=torch.long)
            )
            state["label_cursor"] = 0
            return original_predict_mask(*fn_args, **fn_kwargs)

        def head_pre_hook(module: Any, fn_args: Tuple[Any, ...], kwargs: Dict[str, Any]):
            del module, fn_args
            boxes = kwargs.get("boxes")
            ids = kwargs.get("roi_img_ids")
            state["boxes"] = boxes.detach().cpu() if boxes is not None else None
            state["ids"] = ids.detach().cpu() if ids is not None else None
            queue = state["label_queue"]
            if boxes is None or queue is None:
                return
            start = int(state["label_cursor"])
            end = start + int(boxes.shape[0])
            if end > int(queue.numel()):
                raise RuntimeError("proposal-label queue underflow in coarse Oracle")
            state["labels"] = queue[start:end].detach().cpu()
            state["label_cursor"] = end

        def wrapped_coarse(*fn_args: Any, **fn_kwargs: Any) -> Any:
            output = original_coarse(*fn_args, **fn_kwargs)
            if mode == "raw":
                return output
            raw_logits, coarse_outputs, raw_canvas, raw_valid = output
            boxes = fn_args[2] if len(fn_args) > 2 else fn_kwargs.get("boxes")
            roi_img_ids = fn_args[3] if len(fn_args) > 3 else fn_kwargs.get("roi_img_ids")
            labels = state["labels"]
            if raw_logits is None or boxes is None or roi_img_ids is None or labels is None:
                raise RuntimeError("coarse Oracle cannot establish aligned coarse/ROI/label state")
            candidate, local = build_gt_coarse_candidate(
                raw_logits, roi_img_ids.detach().cpu(), boxes.detach().cpu(), labels,
                state["gts"], match_iou=args.match_iou, logit_span=args.logit_span,
            )
            for name, value in local.items():
                stats[name] += int(value)
            if mode in ("dense", "both"):
                candidate_canvas, candidate_valid, _ = head._shape_prior_to_prompt_mask(
                    candidate, boxes
                )
            else:
                candidate_canvas, candidate_valid = raw_canvas, raw_valid
            return select_consumer_outputs(
                mode, raw_logits, coarse_outputs, raw_canvas, raw_valid,
                candidate, candidate_canvas, candidate_valid,
            )

        handle = None
        if not unhooked:
            handle = head.register_forward_pre_hook(head_pre_hook, with_kwargs=True)
            model.roi_head.predict_mask = wrapped_predict_mask
            head._forward_coarse_p2 = wrapped_coarse
        try:
            with torch.inference_mode():
                for batch_index, batch in enumerate(loader):
                    if args.max_batches and batch_index >= args.max_batches:
                        break
                    images, data_samples = _build_data_samples(batch, device)
                    processed = model.data_preprocessor(
                        {"inputs": images, "data_samples": data_samples}, training=False
                    )
                    state["gts"] = _gt_cache(processed["data_samples"])
                    outputs = model.predict(
                        processed["inputs"], processed["data_samples"], rescale=False
                    )
                    if not unhooked and state["label_queue"] is not None:
                        if state["label_cursor"] != int(state["label_queue"].numel()):
                            raise RuntimeError("proposal-label queue was not consumed exactly")
                    for position, output in enumerate(outputs):
                        meta = dict(batch["img_metas"][position])
                        shape = meta["img_shape"]
                        gt = _extract_instances_numpy(processed["data_samples"][position].gt_instances, shape)
                        prediction = output.pred_instances if hasattr(output, "pred_instances") else output
                        dt = _extract_instances_numpy(prediction, shape)
                        image_id = int(meta.get("image_id", meta.get("scene_id", position)))
                        _hash_update(detector_hash, np.asarray([image_id], dtype=np.int64))
                        for key in ("bboxes", "scores", "labels"):
                            _hash_update(detector_hash, dt[key])
                            _hash_update(output_hash, dt[key])
                        _hash_update(output_hash, dt["masks"])
                        all_gt.append(gt)
                        all_dt.append(dt)
                        all_metas.append(meta)
        finally:
            if handle is not None:
                handle.remove()
                if predict_had_attr:
                    model.roi_head.predict_mask = predict_saved_attr
                else:
                    delattr(model.roi_head, "predict_mask")
                if coarse_had_attr:
                    head._forward_coarse_p2 = coarse_saved_attr
                else:
                    delattr(head, "_forward_coarse_p2")

        coco_gt, coco_dt = build_coco_gt_and_dt(
            all_gt, all_dt, all_metas, categories=VHR10_CATEGORIES
        )
        metrics = run_coco_eval(coco_gt, coco_dt, iou_type="segm")
        gt_records, dt_records, images = _records(all_gt, all_dt, all_metas)
        cell = {
            "mode": "standard" if unhooked else mode,
            "unhooked": bool(unhooked),
            "images": len(images),
            **stats,
            "segm/mAP": metrics["segm/mAP"],
            "segm/mAP_50": metrics["segm/mAP_50"],
            "segm/mAP_75": metrics["segm/mAP_75"],
            "segm/AR@100": metrics["segm/AR@100"],
            "detector_sha256": detector_hash.hexdigest(),
            "output_sha256": output_hash.hexdigest(),
        }
        artifact_dir = output_dir / cell["mode"]
        artifact_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "checkpoint": str(checkpoint_path), "weights": args.weights,
            # Keep the checkpoint's evaluated data contract in every Oracle
            # artifact.  ``bootstrap_paired_map.py`` uses this to select the
            # correct category set; omitting it silently collapses VHR-10 to a
            # single class even when the records carry ids 1..10.
            "dataset": dict(contract),
            "processed_images": len(images),
            "processed_image_ids": sorted(item["image_id"] for item in images),
            "ground_truth_instances": len(gt_records),
            "predicted_instances": len(dt_records), "coarse_to_mining_mode": cell["mode"],
            "frozen_weights": True, "match_iou": args.match_iou,
            "logit_span": args.logit_span, "detector_sha256": cell["detector_sha256"],
            "output_sha256": cell["output_sha256"], "metrics": metrics,
        }
        _write_json(artifact_dir / "gt_records.json", gt_records)
        _write_json(artifact_dir / "dt_records.json", dt_records)
        _write_json(artifact_dir / "images.json", images)
        _write_json(artifact_dir / "metrics.json", metrics)
        _write_json(artifact_dir / "run_manifest.json", manifest)
        cell["artifact_dir"] = str(artifact_dir)
        return cell

    standard = None if args.skip_parity_check else run_pass("raw", unhooked=True)
    if standard is not None and not args.max_batches:
        assert args.canonical_dir is not None  # enforced above; documents formal-run gate
        canonical_dir = Path(args.canonical_dir).expanduser().resolve()
        for filename in ("gt_records.json", "dt_records.json", "images.json"):
            canonical_value = json.loads((canonical_dir / filename).read_text())
            standard_value = json.loads((output_dir / "standard" / filename).read_text())
            if json.dumps(canonical_value, sort_keys=True, separators=(",", ":")) != json.dumps(standard_value, sort_keys=True, separators=(",", ":")):
                raise RuntimeError(
                    f"unhooked standard {filename} differs from canonical production A0 inference"
                )
        canonical_metrics = json.loads((canonical_dir / "metrics.json").read_text())
        standard_metrics = json.loads((output_dir / "standard" / "metrics.json").read_text())
        canonical_segm = {
            key: value for key, value in canonical_metrics.items() if key.startswith("segm/")
        }
        if canonical_segm != standard_metrics:
            raise RuntimeError("unhooked standard segm metrics differ from canonical production A0 inference")
        standard["canonical_records_exact"] = True
        standard["canonical_dir"] = str(canonical_dir)

    cells = []
    for mode in args.modes:
        cell = run_pass(mode)
        if standard is not None:
            if cell["detector_sha256"] != standard["detector_sha256"]:
                raise RuntimeError(f"detector changed under coarse-to-mining mode={mode}")
            if mode == "raw" and cell["output_sha256"] != standard["output_sha256"]:
                raise RuntimeError("raw hooked path is not bitwise identical to unhooked standard inference")
        cells.append(cell)
        print(json.dumps(cell, ensure_ascii=False), flush=True)

    raw_map = next((cell["segm/mAP"] for cell in cells if cell["mode"] == "raw"), None)
    summary = {
        "checkpoint": str(checkpoint_path), "config_source": config_source,
        "contract_drift": contract_drift, "strict_load": report,
        "dataset_contract": contract, "standard_inference": standard, "cells": cells,
        "delta_mAP_vs_raw": {
            cell["mode"]: (cell["segm/mAP"] - raw_map if raw_map is not None else None)
            for cell in cells
        },
        "note": (
            "Frozen A0 coarse-to-mining Oracle: GT replaces only class-aware IoU-matched "
            "ROI coarse rows at the designated existing point/dense consumer(s); all "
            "unmatched rows and every detector output remain unchanged."
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    _write_json(Path(args.output).expanduser().resolve() if args.output else output_dir / "summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
