#!/usr/bin/env python
"""Paired image-bootstrap CI for a COCO segm-mAP delta between two runs.

Phase-0 oracle verdicts need image-level uncertainty, not a single mAP.
Inputs:
  --baseline-dt   predictions.json from infer_from_checkpoint (baseline run)
  --treatment-dir probe output dir with dt_records.json + gt_records.json + images.json
Both runs must cover the SAME image set with the same annotation convention
(category_id = label+1, xywh boxes, COCO RLE segmentation).

Method: resample images with replacement (seeded), assign fresh unique ids per
occurrence, rebuild COCO gt/dt for BOTH runs on the identical resample, and
recompute segm mAP.  Reports the point estimates plus the percentile CI of the
paired delta (treatment - baseline).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dt", required=True)
    parser.add_argument("--treatment-dir", required=True)
    parser.add_argument("--resamples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-dets", type=int, nargs="+", default=None,
                        help="Optional custom maxDets, e.g. 1 10 100")
    return parser.parse_args()


def _load_records(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _group(records):
    grouped = defaultdict(lambda: {"bboxes": [], "labels": [], "rles": [], "scores": [], "mask_scores": []})
    for rec in records:
        g = grouped[int(rec["image_id"])]
        g["bboxes"].append(rec["bbox"])
        g["labels"].append(int(rec["category_id"]) - 1)
        g["rles"].append(rec["segmentation"])
        if "score" in rec:
            g["scores"].append(float(rec["score"]))
            g["mask_scores"].append(float(rec.get("mask_score", rec["score"])))
    return grouped


def _evaluate(resample, gt_by_img, base_by_img, treat_by_img, score_key, max_dets):
    from utils.coco_eval_utils import build_coco_gt_and_dt, run_coco_eval

    all_gt, all_dt_b, all_dt_t, metas = [], [], [], []
    for new_id, image_id in enumerate(resample, start=1):
        gt = gt_by_img.get(image_id)
        if gt is None:
            gt = {"bboxes": [], "labels": [], "rles": []}
        all_gt.append(gt)
        base = base_by_img.get(image_id, {"bboxes": [], "labels": [], "rles": [], "scores": [], "mask_scores": []})
        treat = treat_by_img.get(image_id, {"bboxes": [], "labels": [], "rles": [], "scores": [], "mask_scores": []})
        all_dt_b.append(base)
        all_dt_t.append(treat)
        metas.append({"img_shape": (image_id, 0)})  # placeholder; shapes unused via rles path

    # build_coco_gt_and_dt reads meta["img_shape"][:2] as (h, w) only for
    # mask fallback; with precomputed "rles" the values are unused.  Give the
    # resample a canonical 1024 shape to keep the contract explicit.
    metas = [{"img_shape": (1024, 1024)} for _ in resample]
    coco_gt, dt_b = build_coco_gt_and_dt(all_gt, all_dt_b, metas, score_key=score_key)
    metrics_b = run_coco_eval(coco_gt, dt_b, iou_type="segm", max_dets=max_dets)
    _, dt_t = build_coco_gt_and_dt(all_gt, all_dt_t, metas, score_key=score_key)
    metrics_t = run_coco_eval(coco_gt, dt_t, iou_type="segm", max_dets=max_dets)
    return metrics_b["segm/mAP"], metrics_t["segm/mAP"]


def main():
    args = parse_args()
    score_key = "scores"  # canonical contract: detector-score segm ranking

    base_records = _load_records(args.baseline_dt)
    treat_records = _load_records(Path(args.treatment_dir) / "dt_records.json")
    gt_records = _load_records(Path(args.treatment_dir) / "gt_records.json")
    images = _load_records(Path(args.treatment_dir) / "images.json")

    gt_by_img = _group(gt_records)
    base_by_img = _group(base_records)
    treat_by_img = _group(treat_records)
    image_ids = sorted({int(img["image_id"]) for img in images})
    if not image_ids:
        raise SystemExit("no images found")

    # Point estimates on the full sets.
    point_b, point_t = _evaluate(
        image_ids, gt_by_img, base_by_img, treat_by_img, score_key, args.max_dets
    )

    rng = random.Random(args.seed)
    deltas = []
    for index in range(args.resamples):
        resample = [rng.choice(image_ids) for _ in image_ids]
        mb, mt = _evaluate(resample, gt_by_img, base_by_img, treat_by_img, score_key, args.max_dets)
        deltas.append(mt - mb)
        if (index + 1) % 50 == 0:
            print(f"resample {index + 1}/{args.resamples}: rolling delta mean {np.mean(deltas):+.5f}", flush=True)

    deltas = np.asarray(deltas)
    payload = {
        "images": len(image_ids),
        "resamples": args.resamples,
        "seed": args.seed,
        "baseline_mAP": point_b,
        "treatment_mAP": point_t,
        "delta_point": point_t - point_b,
        "delta_bootstrap_mean": float(deltas.mean()),
        "delta_ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        "p_delta_gt_0": float((deltas > 0).mean()),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
