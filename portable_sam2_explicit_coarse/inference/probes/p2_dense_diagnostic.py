#!/usr/bin/env python
"""Frozen-weight diagnostic v2: P2 correction-vs-confidence, and whether the
decoder can consume an INSTANCE-ACCURATE dense canvas.

v2 fixes over v1 (audit-driven, 2026-09-06):
  1. Every image enters COCO scoring, including zero-detection images
     (misses count); real image ids + GT totals are recorded.
  2. Per-ROI matching to the highest-IoU GT INSTANCE (IoU>=0.5); unmatched
     ROIs are counted separately and KEEP the learned canvas during
     intervention (explicit policy).  v1 used the per-image GT union, which
     tested "image building occupancy", not instance-accurate prompts.
  3. Correct-pixel deltas are split by direction delta*(2*GT-1):
     toward (confidence increase) vs away (weakening); "large" is reported
     per direction, never direction-blind.

Grid mode additionally supports the consumption-end localisation matrix:
  canvas in {learned, matched-instance-GT} x alpha in {current, 0.5, 1.0}
  plus an alpha=0 (dense off) cell.  Per cell: full-eval segm metrics,
  dense-embedding norm, decoder mask-logit magnitude, and final-mask
  changes vs the baseline cell (same detections; only the mask head path
  differs).  Inference-time alpha changes shift the input distribution:
  localisation only, never a training proposal.

All interventions keep points, boxes and detections untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import box_iou

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
    _select_state_dict,
    _snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--weights", choices=("model", "ema"), default="model")
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-batches", type=int, default=0,
                        help="0 = full validation pass (subset for grid runs)")
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--logit-span", type=float, default=4.0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--skip-refiner-pass", action="store_true")
    parser.add_argument("--skip-grid", action="store_true")
    parser.add_argument("--grid-alphas", type=float, nargs="+",
                        default=[None, 0.5, 1.0, 0.0],
                        help="None = current learned gate; 0 disables dense")
    return parser.parse_args()


def load_model(args):
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = _load_checkpoint(checkpoint_path)
    snapshot = _snapshot(checkpoint)
    ns = SimpleNamespace(
        checkpoint=str(checkpoint_path), config=None, split=args.split,
        data_root=None, ann_file=None, image_subdir=None, image_size=None,
        batch_size=None, sam2_repo=None, sam2_ckpt=None,
    )
    model_config, config_source = _resolve_model_config(ns, snapshot)
    model = _register_and_build(model_config)
    load_report = _load_model_state(
        model, _select_state_dict(checkpoint, args.weights), allow_nonstrict=False
    )
    if load_report["missing_keys"] or load_report["unexpected_keys"]:
        raise RuntimeError(f"strict load failed: {load_report}")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    model.eval()
    return model, device, ns, snapshot, config_source, str(checkpoint_path)


def gt_cache(data_samples):
    """Per-image (boxes [M,4], masks [M,H,W]) tensors on CPU."""
    out = []
    for sample in data_samples:
        instances = getattr(sample, "gt_instances", None)
        if instances is None or len(getattr(instances, "bboxes", [])) == 0:
            out.append(None)
            continue
        masks = instances.masks.to_tensor(dtype=torch.float32, device="cpu")
        out.append((instances.bboxes.detach().cpu(), masks))
    return out


def match_instance(gt_entry, roi_box, iou_thr):
    """Return (matched full-image mask [H,W] or None, best IoU)."""
    if gt_entry is None:
        return None, 0.0
    boxes, masks = gt_entry
    ious = box_iou(roi_box[None].float(), boxes.float())[0]
    best = int(ious.argmax())
    if float(ious[best]) < iou_thr:
        return None, float(ious[best])
    return masks[best], float(ious[best])


def roi_target_grid(instance_mask, box, grid):
    """Instance mask cropped by the ROI box, nearest-resized to grid."""
    if instance_mask is None:
        return None
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


class Streaming:
    """v2 pixel counters: matched ROIs only, direction-split."""

    def __init__(self, cap, amp_fraction=0.5):
        self.cap = float(cap)
        self.amp_fraction = amp_fraction
        self.c = dict(
            matched_rois=0, unmatched_rois=0,
            support_total=0,
            correct_n=0, correct_toward=0, correct_away=0,
            correct_large_toward=0, correct_large_away=0,
            correct_toward_sum=0.0, correct_destroy=0,
            wrong_n=0, wrong_in_support=0, wrong_flippable=0,
            wrong_toward=0, wrong_flip_ok=0,
            unmatched_delta_abs_sum=0.0, unmatched_support_px=0,
        )

    def update(self, raw, delta, support, target, roi_matched):
        cap = self.cap
        large_thr = self.amp_fraction * cap
        refined = raw + delta
        raw_fg = raw >= 0.0
        refined_fg = refined >= 0.0
        target = target.bool()
        sup = support.bool()
        c = self.c

        if not roi_matched:
            c["unmatched_rois"] += int(raw.shape[0])
            px = int(sup.sum())
            c["unmatched_support_px"] += px
            if px:
                c["unmatched_delta_abs_sum"] += float(delta[sup].abs().sum())
            return
        c["matched_rois"] += int(raw.shape[0])
        c["support_total"] += int(sup.sum())

        toward_sign = torch.where(target, 1.0, -1.0).to(raw.dtype)
        toward_delta = delta * toward_sign  # >0 pushes toward GT label

        correct = (raw_fg == target)
        wrong = (raw_fg != target)

        cn = int((correct & sup).sum())
        c["correct_n"] += cn
        if cn:
            sel = correct & sup
            c["correct_toward"] += int((toward_delta[sel] > 0).sum())
            c["correct_away"] += int((toward_delta[sel] < 0).sum())
            c["correct_large_toward"] += int(
                ((toward_delta[sel] > 0) & (delta[sel].abs() >= large_thr)).sum()
            )
            c["correct_large_away"] += int(
                ((toward_delta[sel] < 0) & (delta[sel].abs() >= large_thr)).sum()
            )
            c["correct_toward_sum"] += float(toward_delta[sel].sum())
            c["correct_destroy"] += int(
                (correct & sup & (refined_fg != target)).sum()
            )

        c["wrong_n"] += int(wrong.sum())
        ws = wrong & sup
        wsn = int(ws.sum())
        c["wrong_in_support"] += wsn
        if wsn:
            c["wrong_flippable"] += int((raw[ws].abs() <= cap).sum())
            c["wrong_toward"] += int((toward_delta[ws] > 0).sum())
            c["wrong_flip_ok"] += int(
                (ws & (refined_fg == target)).sum()
            )

    def summarize(self):
        c = self.c
        cn = max(c["correct_n"], 1)
        wsn = max(c["wrong_in_support"], 1)
        wn = max(c["wrong_n"], 1)
        rois = max(c["matched_rois"], 1)
        return {
            **c,
            "correct_toward_fraction": c["correct_toward"] / cn,
            "correct_away_fraction": c["correct_away"] / cn,
            "correct_large_toward_fraction": c["correct_large_toward"] / cn,
            "correct_large_away_fraction": c["correct_large_away"] / cn,
            "correct_mean_toward_delta": c["correct_toward_sum"] / cn,
            "correct_destroy_fraction": c["correct_destroy"] / cn,
            "wrong_support_coverage": c["wrong_in_support"] / wn,
            "wrong_flippable_fraction": wsn and c["wrong_flippable"] / wsn,
            "wrong_toward_fraction": c["wrong_toward"] / wsn,
            "wrong_achieved_flip_fraction": c["wrong_flip_ok"] / wsn,
            "wrong_flip_all_wrong_fraction": c["wrong_flip_ok"] / wn,
            "unmatched_delta_abs_mean": c["unmatched_delta_abs_sum"]
            / max(c["unmatched_support_px"], 1),
            "mean_support_px_per_roi": c["support_total"] / rois,
        }


def run_refiner_pass(model, device, loader, max_batches, match_iou):
    head = model.roi_head.mask_head
    refiner = head.p2_boundary_refiner
    grid = refiner.coarse_size
    stream = Streaming(float(refiner.beta) * float(refiner.delta_logit_max))
    state = {"gts": None}

    def refiner_hook(module, args, kwargs, output):
        rois = kwargs["prompt_rois"].detach().cpu()
        raw = kwargs["raw_logits"].detach().float()
        delta = output["delta_logits"].detach().float()
        support = output["search_support"].detach().float()[:, 0]
        targets, matched = [], []
        for index in range(rois.shape[0]):
            image_id = int(rois[index, 0])
            entry = state["gts"][image_id] if state["gts"] and image_id < len(state["gts"]) else None
            mask, _iou = match_instance(entry, rois[index, 1:5], match_iou)
            target = roi_target_grid(mask, rois[index, 1:5], grid)
            if target is None:
                target = torch.zeros(grid, grid, dtype=torch.bool)
            targets.append(target)
            matched.append(mask is not None)
        target = torch.stack(targets).to(raw.device)
        matched_t = torch.tensor(matched, dtype=torch.bool)
        if matched_t.any():
            stream.update(raw[matched_t, 0], delta[matched_t, 0],
                          support[matched_t], target[matched_t], True)
        if (~matched_t).any():
            stream.update(raw[~matched_t, 0], delta[~matched_t, 0],
                          support[~matched_t], target[~matched_t], False)

    handle = refiner.register_forward_hook(refiner_hook, with_kwargs=True)
    n_batches = 0
    try:
        with torch.inference_mode():
            for index, batch in enumerate(loader):
                if max_batches and index >= max_batches:
                    break
                imgs, data_samples = _build_data_samples(batch, device)
                processed = model.data_preprocessor(
                    {"inputs": imgs, "data_samples": data_samples}, training=False
                )
                state["gts"] = gt_cache(processed["data_samples"])
                model.predict(
                    processed["inputs"], processed["data_samples"], rescale=False
                )
                n_batches += 1
    finally:
        handle.remove()
    return stream.summarize(), n_batches


class GridPass:
    """One predict pass with optional canvas/alpha interventions + telemetry.

    Evaluates EVERY image (zero-detection images included) and records
    dense-embedding / decoder-logit magnitudes plus final-mask changes
    against a stored baseline (same detections; only mask path differs).
    """

    def __init__(self, model, device, match_iou, logit_span):
        self.model = model
        self.device = device
        self.head = model.roi_head.mask_head
        self.match_iou = match_iou
        self.logit_span = logit_span
        self.mask_hw = self.head._pe_mask_input_size()
        self.state = {}
        self.stats = dict(
            dense_norm_sum=0.0, dense_norm_n=0,
            logit_abs_sum=0.0, logit_n=0,
            injected=0, kept_learned=0,
        )

    def _alpha_override(self, value):
        if value is None:
            self.head._effective_shape_dense_alpha = self._orig_alpha
        else:
            tensor = torch.tensor(float(value))
            self.head._effective_shape_dense_alpha = lambda: tensor

    def _canvas_hook(self, module, args, kwargs):
        masks = kwargs.get("masks")
        ids = self.state.get("roi_img_ids")
        if masks is None or not self.intervene or ids is None:
            return None
        if len(ids) != masks.shape[0]:
            return None
        canvas = masks.clone()
        for index in range(masks.shape[0]):
            entry = self.state["gts"][int(ids[index])]
            instance, _ = match_instance(
                entry, self.state["roi_boxes"][index], self.match_iou
            )
            if instance is None:
                self.stats["kept_learned"] += 1
                continue
            occ = F.interpolate(
                instance[None, None].to(device=masks.device, dtype=masks.dtype),
                size=tuple(self.mask_hw), mode="nearest",
            )[0, 0]
            canvas[index, 0] = torch.where(
                occ >= 0.5,
                torch.full_like(occ, self.logit_span),
                torch.full_like(occ, -self.logit_span),
            )
            self.stats["injected"] += 1
        kwargs["masks"] = canvas
        return args, kwargs

    def run(self, loader, max_batches, intervene, alpha,
            baseline_masks=None):
        self.intervene = intervene
        self._orig_alpha = self.head._effective_shape_dense_alpha
        head = self.head
        state = self.state
        state.clear()
        self.stats = dict(
            dense_norm_sum=0.0, dense_norm_n=0,
            logit_abs_sum=0.0, logit_n=0,
            injected=0, kept_learned=0,
        )

        def head_pre_hook(module, args, kwargs):
            ids = kwargs.get("roi_img_ids")
            if ids is None and len(args) > 3:
                ids = args[3]
            boxes = kwargs.get("boxes")
            if boxes is None and len(args) > 4:
                boxes = args[4]
            state["roi_img_ids"] = ids.detach().cpu() if ids is not None else None
            state["roi_boxes"] = boxes.detach().cpu() if boxes is not None else None

        def decoder_pre_hook(module, args):
            dense = args[3]
            self.stats["dense_norm_sum"] += float(
                dense.detach().float().flatten(1).norm(dim=1).sum()
            )
            self.stats["dense_norm_n"] += int(dense.shape[0])

        def decoder_post_hook(module, args, output):
            low = output[0].detach().float()
            self.stats["logit_abs_sum"] += float(low.abs().sum())
            self.stats["logit_n"] += int(low.numel())

        head_handle = head.register_forward_pre_hook(head_pre_hook, with_kwargs=True)
        pe_handle = head.prompt_encoder.register_forward_pre_hook(
            self._canvas_hook, with_kwargs=True
        )
        dec_pre = head.mask_decoder.register_forward_pre_hook(decoder_pre_hook)
        dec_post = head.mask_decoder.register_forward_hook(decoder_post_hook)
        self._alpha_override(alpha)

        all_gt, all_dt, all_metas = [], [], []
        image_ids, gt_total = [], 0
        masks_now = {}
        n_batches = 0
        try:
            with torch.inference_mode():
                for index, batch in enumerate(loader):
                    if max_batches and index >= max_batches:
                        break
                    imgs, data_samples = _build_data_samples(batch, device=self.device)
                    processed = self.model.data_preprocessor(
                        {"inputs": imgs, "data_samples": data_samples}, training=False
                    )
                    state["gts"] = gt_cache(processed["data_samples"])
                    outputs = self.model.predict(
                        processed["inputs"], processed["data_samples"], rescale=False
                    )
                    for position, output in enumerate(outputs):
                        meta = batch["img_metas"][position]
                        shape = meta["img_shape"]
                        image_id = int(meta.get("image_id", meta.get("scene_id", 0)))
                        image_ids.append(image_id)
                        gt = _extract_instances_numpy(
                            processed["data_samples"][position].gt_instances, shape
                        )
                        gt_total += int(len(gt["labels"]))
                        pred_instances = (
                            output.pred_instances if hasattr(output, "pred_instances")
                            else output
                        )
                        prediction = _extract_instances_numpy(pred_instances, shape)
                        # every image is scored, zero detections included
                        all_gt.append(gt)
                        all_dt.append(prediction)
                        all_metas.append({"img_shape": shape})
                        if baseline_masks is not None:
                            det_masks = np.asarray(prediction["masks"])
                            base = baseline_masks.get(image_id)
                            if base is not None:
                                changed, ious = _mask_diff(det_masks, base)
                                self.stats.setdefault("mask_changed", 0)
                                self.stats.setdefault("mask_compare", 0)
                                self.stats.setdefault("mask_iou_sum", 0.0)
                                self.stats["mask_changed"] += changed
                                self.stats["mask_compare"] += len(base)
                                self.stats["mask_iou_sum"] += ious
                    n_batches += 1
        finally:
            self._alpha_override(None)
            head_handle.remove()
            pe_handle.remove()
            dec_pre.remove()
            dec_post.remove()

        from utils.coco_eval_utils import build_coco_gt_and_dt, run_coco_eval

        coco_gt, coco_dt = build_coco_gt_and_dt(all_gt, all_dt, all_metas)
        metrics = run_coco_eval(coco_gt, coco_dt, iou_type="segm")
        keep = ("segm/mAP", "segm/mAP_50", "segm/AR@100")
        result = {
            "batches": n_batches,
            "images": len(image_ids),
            "unique_images": len(set(image_ids)),
            "gt_instances": gt_total,
            "images_with_dets": sum(1 for dt in all_dt if len(dt["scores"])),
            "metrics": {k: metrics[k] for k in keep if k in metrics},
            "dense_norm_mean": self.stats["dense_norm_sum"]
            / max(self.stats["dense_norm_n"], 1),
            "mask_logit_abs_mean": self.stats["logit_abs_sum"]
            / max(self.stats["logit_n"], 1),
            "injected": self.stats["injected"],
            "kept_learned": self.stats["kept_learned"],
        }
        if "mask_compare" in self.stats:
            compare = max(self.stats["mask_compare"], 1)
            result["mask_changed_fraction"] = self.stats["mask_changed"] / compare
            result["mask_mean_iou_vs_baseline"] = (
                self.stats["mask_iou_sum"] / compare
            )
        if baseline_masks is None and intervene is False and alpha is None:
            result["_baseline_masks"] = {
                image_id: _pack_masks(np.asarray(dt["masks"]))
                for image_id, dt in zip(image_ids, all_dt)
            }
        return result


def _pack_masks(det_masks, size=256):
    """Downsample bool masks to size x size and pack per image for diffing."""
    packed = []
    for mask in det_masks:
        tensor = torch.from_numpy(np.ascontiguousarray(mask)).float()[None, None]
        small = F.interpolate(tensor, size=(size, size), mode="area")[0, 0] >= 0.5
        packed.append(np.packbits(small.numpy().reshape(-1)))
    return packed


def _mask_diff(det_masks, base_packed, size=256):
    """Changed count and IoU sum of current masks vs packed baseline."""
    changed, iou_sum = 0, 0.0
    for index, mask in enumerate(det_masks):
        if index >= len(base_packed):
            break
        tensor = torch.from_numpy(np.ascontiguousarray(mask)).float()[None, None]
        small = (F.interpolate(tensor, size=(size, size), mode="area")[0, 0] >= 0.5)
        current = np.unpackbits(base_packed[index])[: size * size].reshape(size, size)
        current = current.astype(bool)
        inter = int((small.numpy() & current).sum())
        union = int((small.numpy() | current).sum())
        iou_sum += inter / max(union, 1)
        if union == 0:
            continue
        if inter != union:
            changed += 1
    return changed, iou_sum


def main() -> int:
    args = parse_args()
    model, device, ns, snapshot, config_source, ckpt_path = load_model(args)
    head = model.roi_head.mask_head
    if head.p2_boundary_refiner is None:
        raise SystemExit("checkpoint has no P2 boundary refiner")
    contract = _resolve_dataset_contract(ns, snapshot)
    loader = _build_loader(contract, 0)
    print(f"config_source={config_source} match_iou={args.match_iou}")

    summary = {
        "checkpoint": ckpt_path,
        "match_iou": args.match_iou,
        "logit_span": args.logit_span,
    }

    if not args.skip_refiner_pass:
        counts, batches = run_refiner_pass(
            model, device, loader, args.max_batches, args.match_iou
        )
        counts["batches"] = batches
        summary["pass_a_refiner"] = counts

    if not args.skip_grid:
        grid_pass = GridPass(model, device, args.match_iou, args.logit_span)
        cells = []
        for canvas in ("learned", "gt_instance"):
            for alpha in args.grid_alphas:
                if canvas == "learned" and alpha is None:
                    continue  # baseline cell defined once below
                if canvas == "learned" and alpha == 0.0:
                    continue  # alpha=0 disables dense regardless of canvas
                cells.append((canvas, alpha))
        # order: baseline first, then alphas on learned, then gt canvas cells
        ordered = [(None, None)] + [
            c for c in cells if c[0] == "learned"
        ] + [
            c for c in cells if c[0] == "gt_instance"
        ] + [("off", 0.0)]
        results = {}
        baseline_masks = None
        for canvas, alpha in ordered:
            key = (
                "baseline" if canvas is None
                else ("dense_off" if canvas == "off" else f"{canvas}_a{alpha}")
            )
            intervene = canvas in ("gt_instance",)
            res = grid_pass.run(
                loader, args.max_batches, intervene, alpha,
                baseline_masks=baseline_masks,
            )
            if canvas is None:
                baseline_masks = res.pop("_baseline_masks")
            else:
                res.pop("_baseline_masks", None)
            results[key] = res
            print(f"[grid] {key}: mAP={res['metrics'].get('segm/mAP'):.4f} "
                  f"dense_norm={res['dense_norm_mean']:.2f} "
                  f"logit={res['mask_logit_abs_mean']:.3f} "
                  f"changed={res.get('mask_changed_fraction', 0):.3f} "
                  f"inj={res['injected']} kept={res['kept_learned']}", flush=True)
        summary["grid"] = results

    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
