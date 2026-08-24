#!/usr/bin/env python
"""Measure P2BoundaryRefiner's coarse-prompt effect on a validation split.

For every predicted ROI (hooked from the refiner forward), match the best GT
instance by box IoU, crop the GT mask to the ROI box at the coarse 64x64 grid,
and compare sigmoid(raw) vs sigmoid(refined) masks against it:

  - mean coarse IoU before/after P2 (all ROIs, and support-valid ROIs only)
  - share of ROIs improved / unchanged / worsened
  - delta magnitude vs support validity

Run after the architecture env is set (see scripts/visualize_p2_checkpoint.sh).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

MAINLINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAINLINE_ROOT))

from inference.infer_from_checkpoint import (  # noqa: E402
    _build_data_samples,
    _build_loader,
    _load_checkpoint,
    _load_model_state,
    _register_and_build,
    _resolve_dataset_contract,
    _resolve_model_config,
    _resolve_sam2_repo,
    _select_state_dict,
    _snapshot,
)
from inference.visualize_p2_refiner import P2Capture  # noqa: E402

logger = logging.getLogger("p2_stats")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", choices=("validation", "test", "custom"), default="validation")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--ann-file", default=None)
    parser.add_argument("--image-subdir", default=None)
    parser.add_argument("--image-size", type=int, nargs=2, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=150, help="max images")
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--match-iou", type=float, default=0.3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--weights", choices=("model", "ema"), default="model")
    parser.add_argument("--sam2-repo", default=None)
    parser.add_argument("--sam2-ckpt", default=None)
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def _box_iou(boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(boxes[:, None, 0], gt_boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], gt_boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], gt_boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], gt_boxes[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area_b = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-6, None)


def _crop_gt_to_coarse(gt_mask: torch.Tensor, box: np.ndarray, size: int = 64) -> torch.Tensor:
    height, width = gt_mask.shape
    x1, y1, x2, y2 = [float(v) for v in box]
    x1i = int(np.floor(x1).clip(0, width - 1))
    y1i = int(np.floor(y1).clip(0, height - 1))
    x2i = int(np.ceil(x2).clip(x1i + 1, width))
    y2i = int(np.ceil(y2).clip(y1i + 1, height))
    crop = gt_mask[y1i:y2i, x1i:x2i][None, None].float()
    resized = F.interpolate(crop, size=(size, size), mode="bilinear", align_corners=False)
    return resized[0, 0]


def _mask_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    inter = (a & b).sum().item()
    union = (a | b).sum().item()
    return inter / union if union > 0 else float("nan")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = _load_checkpoint(checkpoint_path)
    snapshot = _snapshot(checkpoint)
    _resolve_sam2_repo(args, snapshot)
    model_config, _ = _resolve_model_config(args, snapshot)
    device = torch.device(
        "cuda:0" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    model = _register_and_build(model_config)
    _load_model_state(model, _select_state_dict(checkpoint, args.weights), allow_nonstrict=False)
    model.to(device)
    model.eval()
    refiner = getattr(model.roi_head.mask_head, "p2_boundary_refiner", None)
    if refiner is None:
        raise RuntimeError("This checkpoint has P2BoundaryRefiner disabled.")

    contract = _resolve_dataset_contract(args, snapshot)
    loader = _build_loader(contract, args.num_workers)
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else MAINLINE_ROOT / "logs" / "p2_coarse_stats"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict] = []
    processed = 0
    skipped = 0
    capture = P2Capture(refiner)
    try:
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader):
                if processed >= args.limit:
                    break
                imgs, data_samples = _build_data_samples(batch, device)
                processed_inputs = model.data_preprocessor(
                    {"inputs": imgs, "data_samples": data_samples}, training=False
                )
                capture.frames.clear()
                model.predict(processed_inputs["inputs"], processed_inputs["data_samples"], rescale=False)
                if not capture.frames:
                    continue
                frame = capture.frames[-1]
                rois = frame["rois"].numpy()
                raw = frame["raw"]
                refined = frame["refined"]
                support = frame["support"]
                for image_index in range(len(imgs)):
                    if skipped < args.skip:
                        skipped += 1
                        continue
                    if processed >= args.limit:
                        break
                    processed += 1
                    gt = data_samples[image_index].gt_instances
                    if len(getattr(gt, "bboxes", [])) == 0:
                        continue
                    gt_boxes = gt.bboxes.detach().cpu().numpy()
                    gt_masks = gt.masks.to_tensor(dtype=torch.float32, device="cpu")
                    sel = rois[:, 0] == image_index
                    image_rows = rois[sel]
                    global_indices = np.nonzero(sel)[0]
                    if len(image_rows) == 0:
                        continue
                    ious = _box_iou(image_rows[:, 1:5], gt_boxes)
                    best_gt = ious.argmax(axis=1)
                    best_iou = ious[np.arange(len(image_rows)), best_gt]
                    for j in range(len(image_rows)):
                        if best_iou[j] < args.match_iou:
                            continue
                        gi = int(best_gt[j])
                        target = _crop_gt_to_coarse(gt_masks[gi], image_rows[j][1:5]) >= 0.5
                        raw_mask = (torch.sigmoid(raw[global_indices[j]][0]) >= 0.5)
                        ref_mask = (torch.sigmoid(refined[global_indices[j]][0]) >= 0.5)
                        delta = frame["delta"][global_indices[j]]
                        records.append(
                            {
                                "raw_iou": _mask_iou(raw_mask, target),
                                "refined_iou": _mask_iou(ref_mask, target),
                                "support_valid": bool(frame["support_valid"][global_indices[j]]),
                                "delta_abs_mean": float(delta.abs().mean()),
                            }
                        )
    finally:
        capture.close()

    def summarize(rows: List[Dict], label: str) -> Dict:
        raw = np.array([r["raw_iou"] for r in rows], dtype=np.float64)
        ref = np.array([r["refined_iou"] for r in rows], dtype=np.float64)
        valid = np.isfinite(raw) & np.isfinite(ref)
        raw, ref = raw[valid], ref[valid]
        changed = np.abs(ref - raw) > 1e-6
        return {
            "label": label,
            "num_rois": int(len(raw)),
            "raw_iou_mean": float(raw.mean()),
            "refined_iou_mean": float(ref.mean()),
            "delta_mean": float((ref - raw).mean()),
            "improved": int((ref > raw + 1e-6).sum()),
            "worsened": int((ref < raw - 1e-6).sum()),
            "unchanged": int((~changed).sum()),
            "delta_abs_mean_of_residual": float(
                np.array([r["delta_abs_mean"] for r in rows], dtype=np.float64)[valid].mean()
            ),
        }

    all_summary = summarize(records, "all matched ROIs")
    valid_rows = [r for r in records if r["support_valid"]]
    valid_summary = summarize(valid_rows, "support-valid ROIs (residual active)")
    rejected_summary = summarize(
        [r for r in records if not r["support_valid"]], "support-rejected ROIs (delta=0)"
    )
    report = {
        "checkpoint": str(checkpoint_path),
        "images": processed,
        "match_iou_thr": args.match_iou,
        "summaries": [all_summary, valid_summary, rejected_summary],
    }
    out_path = out_dir / f"coarse_stats_{checkpoint_path.stem}.json"
    report["records"] = records
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    for summary in report["summaries"]:
        print(
            f"[{summary['label']}] n={summary['num_rois']} "
            f"raw={summary['raw_iou_mean']:.4f} refined={summary['refined_iou_mean']:.4f} "
            f"Δ={summary['delta_mean']:+.4f} | ↑{summary['improved']} ↓{summary['worsened']} "
            f"={summary['unchanged']} | mean|δ|={summary['delta_abs_mean_of_residual']:.4f}"
        )
    print(f"report: {out_path}")


if __name__ == "__main__":
    main()
