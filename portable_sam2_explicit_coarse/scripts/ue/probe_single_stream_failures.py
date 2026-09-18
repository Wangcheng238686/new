#!/usr/bin/env python3
"""Diagnose UE single-stream errors and whether calibrated UAV can cover them.

It reads a completed ``infer_from_checkpoint.py --dataset-format ue`` output;
it does not load a model or write into the dataset.  Ground-truth instances are
matched one-to-one to predictions by mask IoU, then each GT is assigned a
failure type and a multi-height UAV coverage profile.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--data-root", type=Path,
                   default="/data1/wangcheng/dataset/ue_coco_with_uav")
    p.add_argument("--split", choices=("train", "val", "test"), default="val")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--height-bins", default="0,4,7,10,15,22,32")
    p.add_argument("--extent-m", type=float, default=360.0)
    p.add_argument("--match-mask-iou", type=float, default=0.5)
    return p.parse_args()


def rle_iou(gt_rles, pred_rles):
    """Use the compiled pycocotools implementation; never materialise N×M masks."""
    if not gt_rles or not pred_rles:
        return np.zeros((len(gt_rles), len(pred_rles)), dtype=np.float32)
    return np.asarray(mask_utils.iou(gt_rles, pred_rles, [0] * len(pred_rles)), dtype=np.float32)


def box_iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, aw, ah = a.T; bx1, by1, bw, bh = b.T
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    ix1, iy1 = np.maximum(ax1[:, None], bx1), np.maximum(ay1[:, None], by1)
    ix2, iy2 = np.minimum(ax2[:, None], bx2), np.minimum(ay2[:, None], by2)
    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    return inter / np.maximum(aw[:, None] * ah[:, None] + bw * bh - inter, 1e-6)


def read_gt(json_path, image_shape):
    ann = json.loads(json_path.read_text())
    groups = defaultdict(list)
    for i, shape in enumerate(ann.get("shapes", [])):
        if shape.get("shape_type") == "polygon" and len(shape.get("points", [])) >= 3:
            groups[shape.get("group_id", i)].append(np.asarray(shape["points"], np.int32))
    h0, w0 = image_shape
    masks, boxes = [], []
    for polygons in groups.values():
        m = np.zeros((h0, w0), np.uint8); cv2.fillPoly(m, polygons, 1)
        ys, xs = np.where(m)
        if len(xs):
            masks.append(cv2.resize(m, (1024, 1024), interpolation=cv2.INTER_NEAREST).astype(bool))
            boxes.append([xs.min() * 2, ys.min() * 2, (xs.max() + 1) * 2, (ys.max() + 1) * 2])
    rles = []
    for mask in masks:
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        if isinstance(rle["counts"], bytes):
            rle["counts"] = rle["counts"].decode("ascii")
        rles.append(rle)
    return np.asarray(masks, dtype=bool), rles, np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def coverage_for_center(center_px, height_m, scene_center, K, E, size=(512, 512), extent_m=360.):
    # Current-code satellite mapping, in UE cm and W2C column-vector convention.
    u, v = center_px
    world = np.array([scene_center[0] + (0.5 - v / 1024.) * extent_m * 100,
                      scene_center[1] + (u / 1024. - 0.5) * extent_m * 100,
                      height_m * 100., 1.], dtype=np.float64)
    cam = np.einsum("vij,j->vi", E, world)
    pix = np.einsum("vij,vj->vi", K, cam[:, :3])
    uv = pix[:, :2] / np.maximum(pix[:, 2:3], 1e-6)
    valid = ((cam[:, 2] > 1e-4) & (uv[:, 0] >= 0) & (uv[:, 0] < size[0])
             & (uv[:, 1] >= 0) & (uv[:, 1] < size[1]))
    return int(valid.sum())


def area_bucket(area):
    return "small" if area < 32**2 else "medium" if area < 96**2 else "large"


def main():
    a = args(); a.out_dir.mkdir(parents=True, exist_ok=True)
    heights = [float(x) for x in a.height_bins.split(",")]
    predictions = json.loads(a.predictions.read_text())
    by_scene = defaultdict(list)
    for p in predictions:
        by_scene[Path(p["file_name"]).stem].append(p)
    centers = json.loads((a.data_root / "scene_centers.json").read_text())
    rows, aggregate = [], Counter()
    delta_by_bucket, failures_by_bucket = defaultdict(list), defaultdict(Counter)

    for sid, preds in sorted(by_scene.items()):
        sat = a.data_root / a.split / "satellite" / f"{sid}.png"
        image = cv2.imread(str(sat)); gt_masks, gt_rles, gt_xyxy = read_gt(sat.with_suffix(".json"), image.shape[:2])
        pred_rles = [p["segmentation"] for p in preds]
        pred_xywh = np.asarray([p["bbox"] for p in preds], dtype=np.float32).reshape(-1, 4)
        miou, biou = rle_iou(gt_rles, pred_rles), box_iou(
            np.c_[gt_xyxy[:, :2], gt_xyxy[:, 2:] - gt_xyxy[:, :2]], pred_xywh)
        pairs = []
        if len(gt_masks) and len(pred_rles):
            gi, pi = linear_sum_assignment(-miou)
            pairs = [(int(g), int(p)) for g, p in zip(gi, pi) if miou[g, p] >= a.match_mask_iou]
        matched_gt, matched_pred = {g for g, _ in pairs}, {p for _, p in pairs}
        meta = json.loads((a.data_root / a.split / "drone" / sid / "ue_render_metadata" / "camera_params.json").read_text())
        K, E, scene_center = np.asarray(meta["intrinsics"]), np.asarray(meta["extrinsics"]), np.asarray(centers[sid])
        pred_for_gt = dict(pairs)
        for gi in range(len(gt_masks)):
            mask = gt_masks[gi]; area, bucket = int(mask.sum()), area_bucket(int(mask.sum()))
            ys, xs = np.where(mask); center_px = (float(xs.mean()), float(ys.mean()))
            if gi in pred_for_gt:
                pi = pred_for_gt[gi]; m, b = float(miou[gi, pi]), float(biou[gi, pi])
                failure = "matched_good" if m >= .75 else "matched_boundary_error"
                delta_by_bucket[bucket].append(m - b)
            else:
                m = float(miou[gi].max()) if len(pred_rles) else 0.; b = float(biou[gi].max()) if len(pred_rles) else 0.
                failure = "missed" if m < .1 else "partial_or_merge"
            cov = {f"views_z{h:g}": coverage_for_center(center_px, h, scene_center, K, E, extent_m=a.extent_m) for h in heights}
            row = {"scene_id": sid, "gt_index": gi, "area_px": area, "area_bucket": bucket,
                   "failure": failure, "best_mask_iou": m, "best_bbox_iou": b,
                   "mask_minus_bbox_iou": m - b, **cov}
            rows.append(row); aggregate[failure] += 1; failures_by_bucket[bucket][failure] += 1
        for pi in range(len(pred_rles)):
            if pi not in matched_pred:
                aggregate["unmatched_predictions"] += 1

    fields = list(rows[0]) if rows else []
    with (a.out_dir / "instances.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    summary = {
        "prediction_file": str(a.predictions), "scenes": len(by_scene), "gt_instances": len(rows),
        "match_mask_iou": a.match_mask_iou, "height_bins_m": heights,
        "counts": dict(aggregate),
        "by_area_bucket": {k: dict(v) for k, v in failures_by_bucket.items()},
        "mean_mask_minus_bbox_iou_by_bucket": {k: float(np.mean(v)) if v else None for k, v in delta_by_bucket.items()},
        "coverage_of_missed_or_partial": {
            str(h): float(np.mean([r[f"views_z{h:g}"] >= 2 for r in rows if r["failure"] != "matched_good"]))
            for h in heights},
        "interpretation": "Coverage is geometry-only. It shows whether UAV evidence can be queried, not whether it is semantically useful."
    }
    (a.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
