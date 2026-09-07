#!/usr/bin/env python
"""Paired image-bootstrap CI for a COCO segm-mAP delta between two runs.

Phase-0 oracle verdicts need image-level uncertainty, not a single mAP.
Inputs:
  --baseline-dt   predictions.json from infer_from_checkpoint (COCO records)
  --treatment-dir probe output dir with dt_records.json + gt_records.json +
                  images.json (oracle_p2_probe.py export)

Coordinate contract (single source of truth for this tool):
  All record files store COCO-style ``bbox`` in **xywh** format — the probe
  exports GT via ``_xyxy_to_xywh`` and DT via ``[x1, y1, x2-x1, y2-y1]``
  (oracle_p2_probe.py), and infer_from_checkpoint writes COCO-standard xywh.
  ``utils.coco_eval_utils.build_coco_gt_and_dt`` expects **xyxy** and converts
  to xywh itself.  This tool therefore converts xywh -> xyxy exactly once at
  grouping time; feeding xywh through unconverted would double-convert
  (w' = w - x1 < 0) and silently corrupt every metric.  The corruption is NOT
  limited to area-bucketed splits: COCOeval area-gates GT by ``area`` even for
  the "all" range ([0, 1e10]), so on the real Phase-0 export 3,711 / 15,926 GT
  instances acquired negative areas under the double conversion and were
  silently IGNORED — every pre-fix bootstrap number was computed on a biased
  ~77% GT subset.  The reference point estimates are the ORIGINAL runs' own
  metrics (correct in-mask areas); this fixed tool reproduces them exactly.

Execution-coverage contract:
  Absent prediction records are legitimate implicit zero detections, but a
  run that never EXECUTED an image must not be conflated with one that found
  nothing there.  ``--baseline-manifest`` (the baseline run's
  run_manifest.json) is therefore required: ``processed_image_ids`` (the
  exact executed ids; compat alias ``image_ids``), when present, must equal
  the treatment image set exactly — a mismatch fails and never falls back —
  while counts (``processed_images`` / ``ground_truth_instances``) stay as
  auxiliary checks; manifests without any id list fall back to verifying the
  recorded dataset listing (ann_file) against the treatment ids.

Method: resample images with replacement (seeded), assign fresh unique ids
per occurrence, rebuild COCO gt/dt for BOTH runs on the identical resample,
and recompute segm mAP.  Reports the point estimates plus the percentile CI
of the paired delta (treatment - baseline).  ``--resamples 0`` skips the
resample loop and only rebuilds the point estimates (reproduction check
against the runs' own metrics.json).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dt", required=True)
    parser.add_argument("--treatment-dir", required=True)
    parser.add_argument("--baseline-manifest", required=True,
                        help="run_manifest.json of the baseline inference run; "
                        "proves the baseline executed the same image universe")
    parser.add_argument("--resamples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-dets", type=int, nargs="+", default=None,
                        help="Optional custom maxDets, e.g. 1 10 100")
    return parser.parse_args()


def _load_records(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_bbox_xywh(bbox, image_id, width, height, source):
    """Reject anything that is not a finite, in-bounds, positive-area xywh."""
    if len(bbox) != 4:
        raise SystemExit(
            f"{source}: image {image_id} bbox has {len(bbox)} coords, expected 4"
        )
    x, y, w, h = (float(v) for v in bbox)
    if not all(np.isfinite([x, y, w, h])):
        raise SystemExit(f"{source}: image {image_id} bbox has non-finite coords")
    # xywh invariants; a violated one usually means an xyxy array was fed in
    # (coord2 > coord1 would still look "finite", so say so in the message).
    if w <= 0 or h <= 0:
        raise SystemExit(
            f"{source}: image {image_id} bbox {bbox} has non-positive w/h — "
            "records must be COCO xywh (w=bbox[2]>0, h=bbox[3]>0); an xyxy "
            "array produces negative 'widths' here"
        )
    if x < 0 or y < 0:
        raise SystemExit(f"{source}: image {image_id} bbox has negative origin")
    if width > 0 and height > 0 and (x + w > width + 1e-3 or y + h > height + 1e-3):
        raise SystemExit(
            f"{source}: image {image_id} bbox {bbox} exceeds image bounds "
            f"{width}x{height}"
        )
    return x, y, w, h


def _group(records, image_sizes, source):
    """Group COCO xywh records into per-image xyxy arrays for the builder."""
    grouped = defaultdict(
        lambda: {"bboxes": [], "labels": [], "rles": [], "scores": [], "mask_scores": []}
    )
    for rec in records:
        image_id = int(rec["image_id"])
        size = image_sizes.get(image_id)
        if size is None:
            raise SystemExit(
                f"{source}: image {image_id} absent from images.json; refusing "
                "to invent image bounds for validation"
            )
        x, y, w, h = _validate_bbox_xywh(
            rec["bbox"], image_id, size[1], size[0], source
        )
        g = grouped[image_id]
        # Single xywh -> xyxy conversion point; build_coco_gt_and_dt converts
        # xyxy back to xywh for COCO exactly once.
        g["bboxes"].append([x, y, x + w, y + h])
        g["labels"].append(int(rec["category_id"]) - 1)
        g["rles"].append(rec["segmentation"])
        if "score" in rec:
            g["scores"].append(float(rec["score"]))
            g["mask_scores"].append(float(rec.get("mask_score", rec["score"])))
    return grouped


def _require_same_images(sources):
    """Paired bootstrap is only defined when both runs cover the same set."""
    names = list(sources)
    reference = set(sources[names[0]])
    for name in names[1:]:
        other = set(sources[name])
        if other != reference:
            missing = sorted(reference - other)[:10]
            extra = sorted(other - reference)[:10]
            raise SystemExit(
                f"paired image sets differ: {names[0]} has {len(reference)} "
                f"images, {name} has {len(other)}; missing_from_{name}="
                f"{missing} extra_in_{name}={extra}. A missing image must not "
                "silently evaluate as zero predictions."
            )


def _require_subset(name, ids, reference_ids):
    """Records are flat detection lists: an image with zero detections is
    ABSENT, which is a legitimate implicit zero — but an id outside the
    evaluated universe (images.json) means the runs compared different sets
    and must hard-fail.  Coverage is reported in the payload so "zero
    predictions" is always a checked, visible decision."""
    extra = sorted(set(ids) - set(reference_ids))[:10]
    if extra:
        raise SystemExit(
            f"{name} contains images absent from images.json: {extra}"
        )


def _validate_baseline_execution(manifest, image_ids, n_gt_records):
    """Prove the baseline executed the SAME image identities, not just as many.

    Prediction records may legitimately be absent for images where the
    baseline found nothing; missing execution may NOT.  Equal counts alone
    are never sufficient — two disjoint universes of the same size would pass
    — so identity evidence is mandatory, in order of strength:
      1. manifest["processed_image_ids"] (compat alias: "image_ids"): exact
         set equality with the treatment ids.  Presence COMMITS to this
         channel — a mismatch fails, never falling back to channel 2.
      2. manifest["dataset"] run parameters (``ann_file`` resolved against
         ``data_root``): the recorded dataset listing is loaded from disk and
         its image-id set must equal the treatment ids — verifiable
         independently of the run, for historical manifests without ids.
    ``processed_images`` / ``ground_truth_instances`` stay as auxiliary
    consistency checks.  Without either identity channel the validation
    FAILS; it never reports coverage verified on counts alone.
    Returns the identity-evidence description for the payload.
    """
    if not isinstance(manifest, dict):
        raise SystemExit("baseline manifest must be a JSON object (run_manifest.json)")
    # Strong channel: the execution id list.  infer_from_checkpoint writes
    # ``processed_image_ids``; ``image_ids`` stays as a compatible alias for
    # manifests authored before the field existed.  When an execution id list
    # is PRESENT it decides: a mismatch hard-fails and must never fall back
    # to the weaker dataset-listing channel.
    manifest_ids = manifest.get("processed_image_ids")
    id_field = "processed_image_ids"
    if manifest_ids is None:
        manifest_ids = manifest.get("image_ids")
        id_field = "image_ids"
    if manifest_ids is not None:
        _require_same_images({
            "images.json": image_ids,
            f"baseline_manifest.{id_field}": [int(v) for v in manifest_ids],
        })
        evidence = f"manifest_{id_field}"
    else:
        dataset = manifest.get("dataset") or {}
        ann_ref = dataset.get("ann_file")
        if not ann_ref:
            raise SystemExit(
                "cannot verify baseline image identity: manifest has neither "
                "image_ids nor dataset.ann_file; counts alone are not evidence"
            )
        ann_path = Path(ann_ref)
        if not ann_path.is_absolute():
            data_root = dataset.get("data_root")
            if not data_root:
                raise SystemExit(
                    "cannot verify baseline image identity: relative ann_file "
                    "without data_root in the manifest dataset block"
                )
            ann_path = Path(data_root) / ann_ref
        if not ann_path.is_file():
            raise SystemExit(
                f"cannot verify baseline image identity: recorded ann_file not "
                f"found on disk: {ann_path}"
            )
        with ann_path.open("r", encoding="utf-8") as handle:
            ann = json.load(handle)
        ann_ids = sorted({int(img["id"]) for img in ann.get("images", [])})
        _require_same_images({
            "images.json": image_ids,
            f"ann_file({ann_path.name})": ann_ids,
        })
        evidence = f"dataset_ann_file({ann_path})"
    processed = manifest.get("processed_images")
    if processed is None or int(processed) != len(image_ids):
        raise SystemExit(
            f"baseline execution coverage mismatch: manifest processed_images="
            f"{processed} vs treatment images.json {len(image_ids)}; images the "
            "baseline never processed must not be assumed zero-detection"
        )
    gt_total = manifest.get("ground_truth_instances")
    if gt_total is not None and int(gt_total) != int(n_gt_records):
        raise SystemExit(
            f"gt_records count {n_gt_records} != baseline manifest "
            f"ground_truth_instances {gt_total}: records do not belong to the "
            "manifest's run"
        )
    return evidence


def _evaluate(
    resample, gt_by_img, base_by_img, treat_by_img, score_key, max_dets, image_sizes,
    categories=None,
):
    from utils.coco_eval_utils import build_coco_gt_and_dt, run_coco_eval

    all_gt, all_dt_b, all_dt_t, metas = [], [], [], []
    # A resample contains images WITH replacement; enumerate assigns each
    # occurrence a fresh unique COCO image id so duplicate occurrences are
    # independent images (both their gt and dt are duplicated), which is the
    # standard image-bootstrap semantics.
    for new_id, image_id in enumerate(resample, start=1):
        size = image_sizes.get(image_id, (0, 0))
        gt = gt_by_img.get(image_id)
        if gt is None:
            gt = {"bboxes": [], "labels": [], "rles": []}
        gt = {
            "bboxes": np.asarray(gt["bboxes"], dtype=np.float32).reshape(-1, 4),
            "labels": np.asarray(gt["labels"], dtype=np.int64),
            "rles": list(gt["rles"]),
        }
        all_gt.append(gt)
        base = base_by_img.get(image_id)
        treat = treat_by_img.get(image_id)

        def _as_dt(rec):
            if rec is None:
                rec = {"bboxes": [], "labels": [], "rles": [], "scores": [], "mask_scores": []}
            return {
                "bboxes": np.asarray(rec["bboxes"], dtype=np.float32).reshape(-1, 4),
                "labels": np.asarray(rec["labels"], dtype=np.int64),
                "rles": list(rec["rles"]),
                "scores": np.asarray(rec["scores"], dtype=np.float32),
                "mask_scores": np.asarray(rec["mask_scores"], dtype=np.float32),
            }

        all_dt_b.append(_as_dt(base))
        all_dt_t.append(_as_dt(treat))
        metas.append({"img_shape": (int(size[0]), int(size[1]))})
    coco_gt, dt_b = build_coco_gt_and_dt(
        all_gt, all_dt_b, metas, score_key=score_key, categories=categories
    )
    metrics_b = run_coco_eval(coco_gt, dt_b, iou_type="segm", max_dets=max_dets)
    _, dt_t = build_coco_gt_and_dt(
        all_gt, all_dt_t, metas, score_key=score_key, categories=categories
    )
    metrics_t = run_coco_eval(coco_gt, dt_t, iou_type="segm", max_dets=max_dets)
    return metrics_b["segm/mAP"], metrics_t["segm/mAP"]


def main():
    args = parse_args()
    score_key = "scores"  # canonical contract: detector-score segm ranking

    base_records = _load_records(args.baseline_dt)
    treat_records = _load_records(Path(args.treatment_dir) / "dt_records.json")
    gt_records = _load_records(Path(args.treatment_dir) / "gt_records.json")
    images = _load_records(Path(args.treatment_dir) / "images.json")
    baseline_manifest = _load_records(args.baseline_manifest)
    num_classes = int((baseline_manifest.get("dataset") or {}).get("num_classes", 1))
    if num_classes not in {1, 10}:
        raise SystemExit(f"unsupported bootstrap class contract: num_classes={num_classes}")
    categories = None
    if num_classes == 10:
        from utils.coco_eval_utils import VHR10_CATEGORIES
        categories = VHR10_CATEGORIES

    # Keep exporter order for the point estimate.  COCO ranks equal-score
    # detections deterministically by insertion order, so sorting image ids
    # here can change AP despite identical records.  Bootstrap resamples are
    # still seeded over this same image universe.
    image_ids = [int(img["image_id"]) for img in images]
    if len(set(image_ids)) != len(image_ids):
        raise SystemExit("images.json contains duplicate image_id entries")
    image_sizes = {
        int(img["image_id"]): (int(img["height"]), int(img["width"]))
        for img in images
    }
    if not image_sizes:
        raise SystemExit("no images found in images.json")
    # All three record sources must live inside the evaluated universe
    # (images.json).  Absent images are implicit zero-gt / zero-prediction
    # entries; coverage is reported below so that decision is explicit.
    image_id_list = list(image_sizes)
    _require_subset("gt_records", [int(r["image_id"]) for r in gt_records], image_id_list)
    _require_subset("baseline_dt", [int(r["image_id"]) for r in base_records], image_id_list)
    _require_subset("treatment_dt", [int(r["image_id"]) for r in treat_records], image_id_list)
    # Execution coverage: absent RECORDS are zeros, absent EXECUTION is not —
    # image-identity evidence (manifest ids or the recorded dataset listing)
    # is mandatory; counts alone never verify coverage.
    identity_evidence = _validate_baseline_execution(
        baseline_manifest, image_ids, len(gt_records)
    )

    gt_by_img = _group(gt_records, image_sizes, "gt_records")
    base_by_img = _group(base_records, image_sizes, "baseline_dt")
    treat_by_img = _group(treat_records, image_sizes, "treatment_dt")

    # Point estimates on the full sets; the treatment value must reproduce the
    # probe's own metrics.json segm/mAP, the baseline value the infer run's.
    point_b, point_t = _evaluate(
        image_ids, gt_by_img, base_by_img, treat_by_img, score_key, args.max_dets,
        image_sizes, categories,
    )

    rng = random.Random(args.seed)
    deltas = []
    for index in range(args.resamples):
        resample = [rng.choice(image_ids) for _ in image_ids]
        mb, mt = _evaluate(
            resample, gt_by_img, base_by_img, treat_by_img, score_key,
            args.max_dets, image_sizes, categories,
        )
        deltas.append(mt - mb)
        if (index + 1) % 50 == 0:
            print(f"resample {index + 1}/{args.resamples}: rolling delta mean {np.mean(deltas):+.5f}", flush=True)

    payload = {
        "images": len(image_ids),
        "images_with_gt": len(gt_by_img),
        "images_with_baseline_dt": len(base_by_img),
        "images_with_treatment_dt": len(treat_by_img),
        "baseline_manifest": str(args.baseline_manifest),
        "baseline_processed_images": int(baseline_manifest.get("processed_images", -1)),
        "identity_evidence": identity_evidence,
        "num_classes": num_classes,
        "category_ids": [int(item["id"]) for item in (categories or [{"id": 1}])],
        "resamples": len(deltas),
        "seed": args.seed,
        "baseline_mAP": point_b,
        "treatment_mAP": point_t,
        "delta_point": point_t - point_b,
    }
    if deltas:
        deltas_arr = np.asarray(deltas)
        payload.update({
            "delta_bootstrap_mean": float(deltas_arr.mean()),
            "delta_ci95": [float(np.percentile(deltas_arr, 2.5)),
                            float(np.percentile(deltas_arr, 97.5))],
            "p_delta_gt_0": float((deltas_arr > 0).mean()),
            # Raw per-resample deltas: parallel shards merge by concatenating
            # these lists before the pooled percentile CI is computed.
            "deltas": [float(v) for v in deltas_arr],
        })
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
