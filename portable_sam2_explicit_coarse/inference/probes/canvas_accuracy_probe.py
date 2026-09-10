#!/usr/bin/env python
"""R3 pre-test: learned-canvas accuracy vs matched instance GT, plus the
canvas-content -> mAP oracle grid, in one frozen-weight probe.

Motivation (P2-v2 design discussion, 2026-09-07): the +0.0154 GT-canvas
intervention (WHU) proved the decoder pays for an accurate spatial prior,
but R3 (a dedicated canvas renderer) only pays if the DELIVERED canvas is
currently inaccurate.  This probe measures, on frozen weights:

  per canvas mode x gate setting (--canvas-modes x --gate-alphas):
    - canvas accuracy: per matched ROI (max box-IoU >= 0.5, same class,
      many-to-one by design), IoU(bin canvas, GT occupancy) restricted to
      the pasted bbox support on the PE canvas frame (2026-09-09 audit P0
      fix: outside_fill is background, never part of the prediction)
    - canvas intervention: matched ROIs' canvas logits replaced by GT
      occupancy encoded at +-logit_span
      * learned          - no rewrite (accuracy baseline; asserted bitwise
                           identical to an unhooked standard inference pass)
      * support_matched  - GT only inside the original bbox-paste support
      * full_gt          - GT everywhere, but NOTE: with
                           restrict_dense_prompt_to_box=True (this model) the
                           embedding-level box support is re-imposed after the
                           PE, so full_gt is "GT within the same support
                           restriction", NOT a geometry-unconstrained ceiling.
    - full-val segm mAP for the pass (every image scored, zero-dets kept)

Blend interpolation (learned*(1-r)+gt*r) was intentionally dropped in favour
of the discrete mode grid (docs/a0_two_site_oracle_design.md): no r=0.5
midpoint is produced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

MAINLINE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MAINLINE_ROOT))

from inference.infer_from_checkpoint import (  # noqa: E402
    _build_data_samples,
    _build_loader,
    _extract_instances_numpy,
    _load_checkpoint,
    _load_model_state,
    _register_and_build,
    _resolve_dataset_contract,
    _resolve_model_config,
    _resolve_sam2_repo,
    _select_state_dict,
    _snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--weights", choices=("model", "ema"), default="model")
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--ann-file", default=None,
                        help="Optional COCO annotation override for frozen train-split diagnostics.")
    parser.add_argument("--image-subdir", default=None,
                        help="Optional image subdirectory paired with --ann-file.")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--logit-span", type=float, default=None,
                        help="GT-occupancy logit magnitude. Default: derive "
                             "from the model's dense clamp_range (8.0 for the "
                             "current recipe) so the GT canvas matches the "
                             "learned canvas distribution; falls back to 4.0.")
    parser.add_argument(
        "--canvas-modes", nargs="+",
        choices=("learned", "support_matched", "full_gt"),
        default=["learned", "support_matched", "full_gt"],
        help="support_matched replaces GT only inside the original bbox-paste "
             "support; full_gt writes GT everywhere BUT the model's "
             "restrict_dense_prompt_to_box re-imposes the box support on the "
             "embedding after the PE, so full_gt is a GT ceiling within the "
             "same support restriction, not a geometry-unconstrained ceiling.",
    )
    parser.add_argument(
        "--gate-alphas", nargs="+", default=["current", "1.0"],
        help="Dense-gate settings per blend: 'current' keeps the checkpoint "
             "gate; a number in [0,1] forces that alpha.  Both current and "
             "1.0 are needed to separate canvas content from gate strength.",
    )
    parser.add_argument(
        "--skip-parity-check", action="store_true",
        help="Skip the extra standard-inference pass and strict output hashes. "
             "Use only for exploratory partial runs.",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-dir", default=None,
                        help="Directory for per-cell records and manifests.")
    parser.add_argument("--export-pre-prompt-features", action="store_true",
                        help="Export read-only per-proposal features from the "
                             "pre-PromptEncoder ROI/coarse/canvas path; requires "
                             "--output-dir and never changes model tensors.")
    parser.add_argument("--canonical-dir", default=None,
                        help="Required for legacy contract drift: production A0 "
                             "inference directory whose predictions.json anchors C0.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = _load_checkpoint(checkpoint_path)
    snapshot = _snapshot(checkpoint)
    ns = SimpleNamespace(
        checkpoint=str(checkpoint_path), config=None, split=args.split,
        data_root=None, ann_file=args.ann_file, image_subdir=args.image_subdir, image_size=None,
        batch_size=None, sam2_repo=None, sam2_ckpt=None,
    )
    contract_drift = None
    try:
        model_config, config_source = _resolve_model_config(ns, snapshot)
    except RuntimeError as exc:
        # A0 was saved before a later contract-normalisation change.  Its
        # embedded config is still usable only if the *full* state dict loads
        # strictly below; record this exceptional path rather than silently
        # treating the recomputed fingerprint as equivalent.
        if "does not match its saved architecture contract" not in str(exc):
            raise
        model_config = snapshot["model_config"]
        config_source = "embedded_config_strict_state_only_legacy_contract_drift"
        contract_drift = str(exc)
    ns.sam2_repo = _resolve_sam2_repo(ns, snapshot)
    model = _register_and_build(model_config)
    load_report = _load_model_state(
        model, _select_state_dict(checkpoint, args.weights), allow_nonstrict=False
    )
    if load_report["missing_keys"] or load_report["unexpected_keys"]:
        raise RuntimeError(f"strict load failed: {load_report}")
    if contract_drift:
        print("[contract-drift] embedded config used after strict state load: "
              + contract_drift, flush=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    model.eval()
    head = model.roi_head.mask_head
    if not head.explicit_use_dense_prompt:
        raise SystemExit("checkpoint does not consume a dense canvas")
    if head.shape_injector is None or head.prompt_encoder is None:
        raise SystemExit(
            "dense-canvas probe requires both ShapePriorInjector and PromptEncoder"
        )
    print(f"[startup] canvas probe ready: checkpoint={checkpoint_path.name} "
          f"device={device} (model loaded strict)", flush=True)
    if args.logit_span is None:
        _dense_cfg = getattr(head, "dense_prompt_cfg", None) or {}
        _clamp = _dense_cfg.get("clamp_range")
        if isinstance(_clamp, (tuple, list)) and len(_clamp) == 2:
            args.logit_span = float(max(abs(float(_clamp[0])), abs(float(_clamp[1]))))
        elif isinstance(_clamp, (int, float)):
            args.logit_span = float(_clamp)
        else:
            args.logit_span = 4.0
        print(f"[startup] logit_span derived from model clamp_range: "
              f"{args.logit_span}", flush=True)
    mask_hw = head._pe_mask_input_size()

    def gt_cache(data_samples):
        out = []
        for sample in data_samples:
            instances = getattr(sample, "gt_instances", None)
            if instances is None or len(getattr(instances, "bboxes", [])) == 0:
                out.append(None)
                continue
            out.append((
                instances.bboxes.detach().cpu(),
                instances.labels.detach().cpu(),
                instances.masks.to_tensor(dtype=torch.float32, device="cpu"),
            ))
        return out

    def gt_canvas_256(instance_mask):
        return F.interpolate(
            instance_mask[None, None].to(device=device), size=tuple(mask_hw),
            mode="nearest",
        )[0, 0]

    gate_specs = []
    for value in args.gate_alphas:
        if value == "current":
            gate_specs.append(("current", None))
            continue
        try:
            alpha = float(value)
        except ValueError as exc:
            raise SystemExit(f"invalid --gate-alphas value {value!r}") from exc
        if not 0.0 <= alpha <= 1.0:
            raise SystemExit(f"gate alpha must be in [0,1], got {alpha}")
        gate_specs.append((f"{alpha:g}", alpha))

    def _roi_target_grid(instance_mask, box, grid):
        """Match training's ROI crop -> nearest coarse-grid target semantics."""
        h, w = instance_mask.shape
        x1, y1, x2, y2 = [float(v) for v in box]
        x1, y1 = max(0.0, min(x1, w - 1)), max(0.0, min(y1, h - 1))
        x2, y2 = max(x1 + 1, min(x2, w)), max(y1 + 1, min(y2, h))
        crop = instance_mask[int(y1):int(y2), int(x1):int(x2)]
        if crop.numel() == 0:
            return None
        return F.interpolate(
            crop[None, None], size=(grid, grid), mode="nearest"
        )[0, 0] >= 0.5

    def _update_hash(hasher, value):
        array = np.ascontiguousarray(np.asarray(value))
        hasher.update(str((array.dtype.str, array.shape)).encode("ascii"))
        hasher.update(array.tobytes())

    def run_pass(canvas_mode, forced_alpha, *, standard_inference=False):
        state = {"ids": None, "boxes": None, "labels": None, "gts": None,
                 "label_queue": None, "label_cursor": 0, "canvas_hw": mask_hw,
                 "pre_prompt": None, "feature_rows": []}
        stats = {"roi_n": 0, "iou_sum": 0.0, "iou_vals": [],
                 "canvas_min": float("inf"), "canvas_max": float("-inf"),
                 "coarse_dice_sum": 0.0, "coarse_n": 0,
                 "matched": 0, "unmatched": 0}
        detector_hash = hashlib.sha256()
        output_hash = hashlib.sha256()

        def head_pre_hook(module, fn_args, kwargs):
            ids = kwargs.get("roi_img_ids")
            boxes = kwargs.get("boxes")
            state["ids"] = ids.detach().cpu() if ids is not None else None
            state["boxes"] = boxes.detach().cpu() if boxes is not None else None
            queue = state["label_queue"]
            if boxes is not None and queue is not None:
                start = int(state["label_cursor"])
                end = start + int(boxes.shape[0])
                if end > int(queue.numel()):
                    raise RuntimeError("proposal-label queue underflow in canvas probe")
                state["labels"] = queue[start:end].detach().cpu()
                state["label_cursor"] = end

        def record_coarse(output):
            # ``forward_roi`` is called directly by the mask head, so a module
            # forward hook would never see this output.
            raw = output[0].detach()
            ids, boxes, labels, gts = state["ids"], state["boxes"], state["labels"], state["gts"]
            if ids is None or boxes is None or labels is None or gts is None:
                return
            from torchvision.ops import box_iou
            for i in range(raw.shape[0]):
                img = int(ids[i]) if ids[i].numel() else -1
                entry = gts[img] if 0 <= img < len(gts) else None
                if entry is None:
                    continue
                bboxes, gt_labels, masks_t = entry
                same_class = gt_labels == labels[i]
                if not bool(same_class.any()):
                    continue
                ious = box_iou(boxes[i:i+1].float(), bboxes.float())[0]
                ious = torch.where(same_class, ious, torch.full_like(ious, -1.0))
                best = int(ious.argmax())
                if float(ious[best]) < args.match_iou:
                    continue
                grid = raw.shape[-1]
                tgt = _roi_target_grid(masks_t[best], boxes[i], grid)
                if tgt is None:
                    continue
                tgt = tgt.to(device=raw.device)
                pred = (raw[i, 0] >= 0)
                inter = float((pred & tgt).sum())
                union = float((pred | tgt).sum())
                if union > 0:
                    stats["coarse_dice_sum"] += 2 * inter / (inter + union + 1e-6)
                    stats["coarse_n"] += 1

        injector = head.shape_injector
        had_forward_roi_attr = "forward_roi" in injector.__dict__
        saved_forward_roi_attr = injector.__dict__.get("forward_roi")
        original_forward_roi = injector.forward_roi

        def wrapped_forward_roi(*fn_args, **fn_kwargs):
            output = original_forward_roi(*fn_args, **fn_kwargs)
            record_coarse(output)
            return output

        # Observational wrapper: all captured tensors already exist upstream of
        # PromptEncoder.  The original output object is returned unchanged.
        original_coarse = head._forward_coarse_p2
        had_coarse_attr = "_forward_coarse_p2" in head.__dict__
        saved_coarse_attr = head.__dict__.get("_forward_coarse_p2")

        def wrapped_coarse(*fn_args, **fn_kwargs):
            output = original_coarse(*fn_args, **fn_kwargs)
            if not args.export_pre_prompt_features:
                return output
            raw, _coarse_outputs, canvas, _valid = output
            x = fn_args[0] if fn_args else fn_kwargs.get("x")
            boxes, labels, ids = state["boxes"], state["labels"], state["ids"]
            if (raw is None or canvas is None or x is None or boxes is None
                    or labels is None or ids is None):
                raise RuntimeError("pre-PromptEncoder feature hook lacks active A0 tensors")
            if not (raw.shape[0] == canvas.shape[0] == x.shape[0] == boxes.shape[0]
                    == labels.shape[0] == ids.shape[0]):
                raise RuntimeError("pre-PromptEncoder feature rows are not aligned")
            state["pre_prompt"] = tuple(value.detach().cpu() for value in
                                         (x, raw, canvas, boxes, labels, ids))
            return output

        def pe_pre_hook(module, fn_args, kwargs):
            masks = kwargs.get("masks")
            ids, boxes, labels, gts = state["ids"], state["boxes"], state["labels"], state["gts"]
            if (standard_inference or masks is None
                    or ids is None or boxes is None or labels is None or gts is None):
                return fn_args, kwargs
            if len(ids) != masks.shape[0]:
                return fn_args, kwargs
            from torchvision.ops import box_iou
            canvas = masks.clone()
            for i in range(masks.shape[0]):
                img = int(ids[i]) if ids[i].numel() else -1
                entry = gts[img] if 0 <= img < len(gts) else None
                if entry is None:
                    stats["unmatched"] += 1
                    continue
                bboxes, gt_labels, masks_t = entry
                same_class = gt_labels == labels[i]
                if not bool(same_class.any()):
                    stats["unmatched"] += 1
                    continue
                ious = box_iou(boxes[i:i+1].float(), bboxes.float())[0]
                ious = torch.where(same_class, ious, torch.full_like(ious, -1.0))
                best = int(ious.argmax())
                if float(ious[best]) < args.match_iou:
                    stats["unmatched"] += 1
                    continue
                stats["matched"] += 1
                gt_occ = gt_canvas_256(masks_t[best])  # [256,256] on device
                learned = canvas[i, 0]
                # Audit P0 fix (2026-09-09): outside the pasted box the canvas
                # is outside_fill_logit (0.0), which the >=0 binarization would
                # otherwise count as foreground across the whole frame and turn
                # "IoU" into the GT occupancy fraction.  Both the accuracy stat
                # and the support rewrite use the true paste support, computed
                # with paste_roi_to_full_canvas's floor/ceil mapping.
                image_size = float(head.prompt_encoder_image_size)
                bx1, by1, bx2, by2 = [float(v) for v in boxes[i]]
                sx1 = max(0, min(mask_hw[1] - 1, int(np.floor(bx1 * mask_hw[1] / image_size))))
                sy1 = max(0, min(mask_hw[0] - 1, int(np.floor(by1 * mask_hw[0] / image_size))))
                sx2 = max(sx1 + 1, min(mask_hw[1], int(np.ceil(bx2 * mask_hw[1] / image_size))))
                sy2 = max(sy1 + 1, min(mask_hw[0], int(np.ceil(by2 * mask_hw[0] / image_size))))
                support = learned[sy1:sy2, sx1:sx2]
                pred_fg = support >= 0
                gt_fg = gt_occ[sy1:sy2, sx1:sx2] >= 0.5
                inter = float((pred_fg & gt_fg).sum())
                union = float((pred_fg | gt_fg).sum())
                if union > 0:
                    stats["iou_sum"] += inter / union
                    stats["iou_vals"].append(inter / union)
                    stats["roi_n"] += 1
                    stats["canvas_min"] = min(stats["canvas_min"], float(support.min()))
                    stats["canvas_max"] = max(stats["canvas_max"], float(support.max()))
                if canvas_mode == "learned":
                    continue
                gt_logits = gt_occ * args.logit_span - (1.0 - gt_occ) * args.logit_span
                if canvas_mode == "support_matched":
                    delivered = learned.clone()
                    delivered[sy1:sy2, sx1:sx2] = gt_logits[sy1:sy2, sx1:sx2]
                elif canvas_mode == "full_gt":
                    # NOTE: with restrict_dense_prompt_to_box=True the box
                    # support is re-imposed on the embedding after the PE, so
                    # this is a GT ceiling within the same support restriction
                    # (not geometry-unconstrained).
                    delivered = gt_logits
                else:
                    raise AssertionError(f"unexpected canvas mode {canvas_mode!r}")
                # the actual intervention on the consumed canvas
                canvas[i, 0] = delivered
            kwargs["masks"] = canvas
            return fn_args, kwargs

        original_predict_mask = model.roi_head.predict_mask
        had_predict_mask_attr = "predict_mask" in model.roi_head.__dict__
        saved_predict_mask_attr = model.roi_head.__dict__.get("predict_mask")

        def wrapped_predict_mask(*fn_args, **fn_kwargs):
            results_list = fn_kwargs.get("results_list")
            if results_list is None:
                # bound-method positional form: x, metas, results_list, ...
                results_list = fn_args[2]
            state["label_queue"] = torch.cat(
                [result.labels.detach() for result in results_list], dim=0
            ) if results_list else torch.zeros(0, dtype=torch.long)
            state["label_cursor"] = 0
            return original_predict_mask(*fn_args, **fn_kwargs)

        head_handle = head.register_forward_pre_hook(head_pre_hook, with_kwargs=True)
        model.roi_head.predict_mask = wrapped_predict_mask
        injector.forward_roi = wrapped_forward_roi
        head._forward_coarse_p2 = wrapped_coarse
        pe_handle = head.prompt_encoder.register_forward_pre_hook(
            pe_pre_hook, with_kwargs=True
        )
        had_alpha_attr = "_effective_shape_dense_alpha" in head.__dict__
        saved_alpha_attr = head.__dict__.get("_effective_shape_dense_alpha")
        original_alpha = head._effective_shape_dense_alpha
        if forced_alpha is not None:
            def forced_gate():
                current = original_alpha()
                if current is None:
                    return None
                return torch.full_like(current, forced_alpha)
            head._effective_shape_dense_alpha = forced_gate
        all_gt, all_dt, all_metas = [], [], []
        n_batches = 0
        try:
            with torch.inference_mode():
                for index, batch in enumerate(loader):
                    if args.max_batches and index >= args.max_batches:
                        break
                    imgs, data_samples = _build_data_samples(batch, device)
                    processed = model.data_preprocessor(
                        {"inputs": imgs, "data_samples": data_samples}, training=False
                    )
                    state["gts"] = gt_cache(processed["data_samples"])
                    outputs = model.predict(
                        processed["inputs"], processed["data_samples"], rescale=False
                    )
                    if state["label_queue"] is not None and state["label_cursor"] != int(state["label_queue"].numel()):
                        raise RuntimeError("proposal-label queue was not consumed exactly")
                    for position, output in enumerate(outputs):
                        meta = batch["img_metas"][position]
                        shape = meta["img_shape"]
                        gt = _extract_instances_numpy(
                            processed["data_samples"][position].gt_instances, shape
                        )
                        pred = (
                            output.pred_instances
                            if hasattr(output, "pred_instances") else output
                        )
                        dt = _extract_instances_numpy(pred, shape)
                        image_id = int(meta.get("image_id", meta.get("scene_id", position)))
                        if args.export_pre_prompt_features:
                            payload = state["pre_prompt"]
                            if payload is None:
                                raise RuntimeError("feature hook did not run before prediction")
                            x, raw, canvas, pboxes, plabels, pids = payload
                            take = pids.eq(position)
                            if int(take.sum()) != len(dt["bboxes"]):
                                raise RuntimeError("pre-PromptEncoder feature proposal count differs from output")
                            if not np.allclose(pboxes[take].numpy(), dt["bboxes"], rtol=0, atol=1e-4):
                                raise RuntimeError("pre-PromptEncoder feature boxes differ from output detections")
                            if not np.array_equal(plabels[take].numpy(), dt["labels"]):
                                raise RuntimeError("pre-PromptEncoder feature labels differ from output detections")
                            from inference.probes.dense_gate_readability import pre_prompt_features
                            scalar, full = pre_prompt_features(
                                x[take], raw[take], canvas[take], pboxes[take], tuple(shape[:2]),
                                plabels[take], torch.from_numpy(np.asarray(dt["scores"], dtype=np.float32)),
                                int(contract["num_classes"]),
                            )
                            state["feature_rows"].append((image_id, scalar.numpy(), full.numpy()))
                        _update_hash(detector_hash, np.asarray([image_id], dtype=np.int64))
                        for key in ("bboxes", "scores", "labels"):
                            _update_hash(detector_hash, dt[key])
                            _update_hash(output_hash, dt[key])
                        _update_hash(output_hash, dt["masks"])
                        all_gt.append(gt)
                        all_dt.append(dt)
                        all_metas.append(dict(meta))
                    n_batches += 1
        finally:
            head_handle.remove()
            if had_predict_mask_attr:
                model.roi_head.predict_mask = saved_predict_mask_attr
            else:
                delattr(model.roi_head, "predict_mask")
            pe_handle.remove()
            if had_forward_roi_attr:
                injector.forward_roi = saved_forward_roi_attr
            else:
                delattr(injector, "forward_roi")
            if had_coarse_attr:
                head._forward_coarse_p2 = saved_coarse_attr
            else:
                delattr(head, "_forward_coarse_p2")
            if forced_alpha is not None:
                if had_alpha_attr:
                    head._effective_shape_dense_alpha = saved_alpha_attr
                else:
                    delattr(head, "_effective_shape_dense_alpha")

        from utils.coco_eval_utils import VHR10_CATEGORIES, build_coco_gt_and_dt, run_coco_eval

        coco_gt, coco_dt = build_coco_gt_and_dt(
            all_gt, all_dt, all_metas, categories=VHR10_CATEGORIES
        )
        metrics = run_coco_eval(coco_gt, coco_dt, iou_type="segm")
        vals = np.asarray(stats["iou_vals"]) if stats["iou_vals"] else np.zeros(0)
        cell = {
            "canvas_mode": canvas_mode,
            "gate_alpha": "current" if forced_alpha is None else forced_alpha,
            "batches": n_batches,
            "images": len(all_metas),
            "matched_rois": stats["matched"],
            "unmatched_rois": stats["unmatched"],
            "canvas_iou_mean": float(vals.mean()) if vals.size else None,
            "canvas_iou_median": float(np.median(vals)) if vals.size else None,
            "canvas_iou_p25": float(np.percentile(vals, 25)) if vals.size else None,
            "canvas_support_min": (
                stats["canvas_min"] if stats["canvas_min"] != float("inf") else None
            ),
            "canvas_support_max": (
                stats["canvas_max"] if stats["canvas_max"] != float("-inf") else None
            ),
            "coarse_dice_mean": (
                stats["coarse_dice_sum"] / max(stats["coarse_n"], 1)
                if stats["coarse_n"] else None
            ),
            "segm/mAP": metrics.get("segm/mAP"),
            "segm/mAP_50": metrics.get("segm/mAP_50"),
            "segm/mAP_75": metrics.get("segm/mAP_75"),
            "segm/AR@100": metrics.get("segm/AR@100"),
            "detector_sha256": detector_hash.hexdigest(),
            "output_sha256": output_hash.hexdigest(),
        }
        if output_dir is not None:
            from inference.infer_from_checkpoint import _write_json
            from utils.coco_eval_utils import _mask_to_rle, _xyxy_to_xywh
            name = "standard" if standard_inference else (
                f"{canvas_mode}_gate_{cell['gate_alpha']}"
            )
            artifact_dir = output_dir / name
            artifact_dir.mkdir(parents=True, exist_ok=True)
            gt_records, dt_records, images = [], [], []
            for gt, dt, meta in zip(all_gt, all_dt, all_metas):
                image_id = int(meta.get("image_id", meta.get("scene_id", 0)))
                images.append({"image_id": image_id, "height": int(meta["img_shape"][0]),
                               "width": int(meta["img_shape"][1])})
                for j, label in enumerate(np.asarray(gt["labels"])):
                    gt_records.append({"image_id": image_id, "category_id": int(label) + 1,
                        "bbox": [float(v) for v in _xyxy_to_xywh(gt["bboxes"][j])],
                        "segmentation": _mask_to_rle(gt["masks"][j]), "iscrowd": 0})
                for j, box in enumerate(np.asarray(dt["bboxes"])):
                    x1, y1, x2, y2 = (float(v) for v in box)
                    dt_records.append({"image_id": image_id, "category_id": int(dt["labels"][j]) + 1,
                        "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(dt["scores"][j]),
                        "mask_score": float(dt.get("mask_scores", dt["scores"])[j]),
                        "segmentation": _mask_to_rle(dt["masks"][j])})
            manifest = {"checkpoint": str(checkpoint_path), "weights": args.weights,
                # Paired bootstrap selects its COCO category contract from this
                # field.  Keeping it avoids a silent one-class fallback.
                "dataset": dict(contract),
                "processed_images": len(images),
                "processed_image_ids": sorted(item["image_id"] for item in images),
                "ground_truth_instances": len(gt_records), "predicted_instances": len(dt_records),
                "canvas_mode": canvas_mode, "gate_alpha": cell["gate_alpha"],
                "detector_sha256": cell["detector_sha256"], "output_sha256": cell["output_sha256"],
                "metrics": metrics}
            _write_json(artifact_dir / "gt_records.json", gt_records)
            _write_json(artifact_dir / "dt_records.json", dt_records)
            _write_json(artifact_dir / "images.json", images)
            _write_json(artifact_dir / "metrics.json", metrics)
            _write_json(artifact_dir / "run_manifest.json", manifest)
            if args.export_pre_prompt_features:
                rows = state["feature_rows"]
                if len(rows) != len(all_dt):
                    raise RuntimeError("feature export image row count differs from predictions")
                image_ids = np.concatenate([
                    np.full(len(scalar), image_id, dtype=np.int64)
                    for image_id, scalar, _full in rows
                ]) if rows else np.zeros(0, dtype=np.int64)
                scalar = np.concatenate([value for _id, value, _full in rows], axis=0) if rows else np.zeros((0, 0), dtype=np.float32)
                full = np.concatenate([value for _id, _scalar, value in rows], axis=0) if rows else np.zeros((0, 0), dtype=np.float32)
                if len(image_ids) != len(dt_records) or scalar.shape[0] != len(dt_records):
                    raise RuntimeError("feature export rows do not map one-to-one to dt_records")
                np.savez_compressed(artifact_dir / "pre_prompt_features.npz",
                                    image_ids=image_ids, scalar=scalar, full=full)
                cell["pre_prompt_features"] = str(artifact_dir / "pre_prompt_features.npz")
            cell["artifact_dir"] = str(artifact_dir)
        return cell

    contract = _resolve_dataset_contract(ns, snapshot)
    loader = _build_loader(contract, 0)
    if head.quality_head_enabled:
        raise RuntimeError(
            "canvas oracle refuses quality-head checkpoints: canvas changes can "
            "alter mask-score ranking, which is a separate intervention"
        )
    output_dir = (
        Path(args.output_dir).expanduser().resolve() if args.output_dir else
        (Path(args.output).expanduser().resolve().with_suffix("") if args.output else None)
    )
    if args.export_pre_prompt_features and output_dir is None:
        raise RuntimeError("--export-pre-prompt-features requires --output-dir")
    standard = None
    if not args.skip_parity_check:
        standard = run_pass("learned", None, standard_inference=True)
        if contract_drift and not args.max_batches:
            if output_dir is None or not args.canonical_dir:
                raise RuntimeError(
                    "legacy checkpoint contract drift requires --output-dir and "
                    "--canonical-dir; refusing an unanchored Oracle conclusion"
                )
            canonical_dir = Path(args.canonical_dir).expanduser().resolve()
            canonical_dt = json.loads((canonical_dir / "dt_records.json").read_text())
            probe_dt = json.loads((output_dir / "standard" / "dt_records.json").read_text())
            canonical_metrics = json.loads((canonical_dir / "metrics.json").read_text())
            if json.dumps(canonical_dt, sort_keys=True, separators=(",", ":")) != json.dumps(probe_dt, sort_keys=True, separators=(",", ":")):
                raise RuntimeError("C0 dt_records differ from canonical production inference")
            if float(canonical_metrics["segm/mAP"]) != float(standard["segm/mAP"]):
                raise RuntimeError("C0 mAP differs from canonical production inference")
            standard["canonical_dir"] = str(canonical_dir)
            standard["canonical_records_exact"] = True
        elif contract_drift:
            standard["canonical_records_exact"] = None
        print(f"[standard] mAP={standard['segm/mAP']:.4f}", flush=True)

    results = []
    reference_detector = None
    for gate_label, forced_alpha in gate_specs:
        for canvas_mode in args.canvas_modes:
            cell = run_pass(canvas_mode, forced_alpha)
            if reference_detector is None:
                reference_detector = cell["detector_sha256"]
            elif cell["detector_sha256"] != reference_detector:
                raise RuntimeError(
                    "detector output changed under a canvas/gate-only intervention: "
                    f"canvas={canvas_mode}, gate={gate_label}"
                )
            if standard is not None:
                if cell["detector_sha256"] != standard["detector_sha256"]:
                    raise RuntimeError(
                        "detector output changed under a canvas-only intervention: "
                        f"canvas={canvas_mode}, gate={gate_label}"
                    )
                if canvas_mode == "learned" and forced_alpha is None:
                    if cell["output_sha256"] != standard["output_sha256"]:
                        raise RuntimeError(
                            "learned/current-gate does not reproduce "
                            "standard inference exactly"
                        )
            results.append(cell)
            print(f"[canvas={canvas_mode} gate={gate_label}] "
                  f"canvas_iou={cell['canvas_iou_mean']} "
                  f"coarse_dice={cell['coarse_dice_mean']} "
                  f"mAP={cell['segm/mAP']:.4f}", flush=True)

    base_map = next((
        c["segm/mAP"] for c in results
        if c["canvas_mode"] == "learned" and c["gate_alpha"] == "current"
    ), None)
    summary = {
        "checkpoint": str(checkpoint_path),
        "config_source": config_source,
        "contract_drift": contract_drift,
        "weights": args.weights,
        "match_iou": args.match_iou,
        "matching_protocol": "per-ROI argmax, same-class, many-to-one by design",
        "logit_span": args.logit_span,
        "canvas_binarization_logit": 0.0,
        "canvas_frame": list(mask_hw),
        "outside_fill_logit": (
            (getattr(head, "dense_prompt_cfg", None) or {}).get("outside_fill_logit")
        ),
        "restrict_dense_prompt_to_box": bool(
            getattr(head, "restrict_dense_prompt_to_box", False)
        ),
        "gate_alpha_current": (
            float(torch.sigmoid(torch.as_tensor(_raw)).item())
            if (_raw := getattr(head, "shape_dense_alpha_raw", None)) is not None
            and torch.is_tensor(_raw) else None
        ),
        "standard_inference": standard,
        "cells": results,
        "delta_mAP_vs_learned": {
            f"canvas={c['canvas_mode']},gate={c['gate_alpha']}": (
                c["segm/mAP"] - base_map if base_map is not None else None
            )
            for c in results
        },
        "note": (
            "Frozen-weight canvas oracle. learned/current is asserted bitwise "
            "identical to standard inference. canvas_iou is computed on the "
            "pasted bbox support only (outside_fill is background). "
            "support_matched preserves the original bbox support; full_gt "
            "writes GT everywhere but, under restrict_dense_prompt_to_box, the "
            "embedding-level box support is re-imposed after the PE, so it is "
            "a GT ceiling within the same support restriction, not an "
            "unconstrained one. No blend interpolation by design."
        ),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
