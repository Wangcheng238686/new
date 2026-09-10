#!/usr/bin/env python3
"""Post-hoc frozen Oracle for *instance-selective* dense-prompt trust.

This tool consumes two complete outputs of the *same frozen forward pass*:
the delivered learned canvas under the checkpoint's current dense gate and the
same learned canvas with only ``alpha`` forced to 1.  It never hooks a model
and never feeds GT to a forward pass.  GT is used afterwards solely to select,
for each class/box matched detection, the candidate mask with greater mask IoU.

The resulting ``oracle`` records are an upper bound for a per-instance gate,
not a deployable result.  A per-image-count-matched random selector is emitted
as the required control: it answers whether an apparent oracle gain is merely
caused by turning the gate on for K detections.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _xywh_iou(a, b):
    ax1, ay1, aw, ah = (float(x) for x in a)
    bx1, by1, bw, bh = (float(x) for x in b)
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _rle_iou(a, b):
    """Mask IoU for JSON-safe COCO RLEs; normalise string counts for C API."""
    from pycocotools import mask as mask_util
    def normalise(rle):
        out = dict(rle)
        if isinstance(out.get("counts"), str):
            out["counts"] = out["counts"].encode("ascii")
        return out
    return float(mask_util.iou([normalise(a)], [normalise(b)], [0])[0, 0])


def _immutable_key(record):
    """Fields that a gate-only intervention must leave exactly unchanged."""
    return (
        int(record["image_id"]), int(record["category_id"]),
        tuple(float(x) for x in record["bbox"]), float(record["score"]),
        float(record.get("mask_score", record["score"])),
    )


def _match_gt(record, gt_by_image, match_iou):
    candidates = [g for g in gt_by_image[int(record["image_id"])]
                  if int(g["category_id"]) == int(record["category_id"])]
    if not candidates:
        return None
    best = max(candidates, key=lambda g: _xywh_iou(record["bbox"], g["bbox"]))
    return best if _xywh_iou(record["bbox"], best["bbox"]) >= match_iou else None


def make_selectors(baseline, forced, gt_records, match_iou, random_trials, seed):
    """Return oracle/random selection indices and audit rows.

    ``baseline`` and ``forced`` retain exporter order: equal detector scores can
    make that order AP-relevant, so this function never sorts detections.
    """
    if len(baseline) != len(forced):
        raise ValueError(f"dt length differs: current={len(baseline)} forced={len(forced)}")
    for index, (raw, full) in enumerate(zip(baseline, forced)):
        if _immutable_key(raw) != _immutable_key(full):
            raise ValueError(f"non-mask detection field changed at dt index {index}")
    gt_by_image = defaultdict(list)
    for row in gt_records:
        gt_by_image[int(row["image_id"])].append(row)

    oracle, matched_by_image, audit = set(), defaultdict(list), []
    for index, (raw, full) in enumerate(zip(baseline, forced)):
        gt = _match_gt(raw, gt_by_image, match_iou)
        row = {"dt_index": index, "image_id": int(raw["image_id"]), "matched": gt is not None}
        if gt is not None:
            raw_iou, full_iou = _rle_iou(raw["segmentation"], gt["segmentation"]), _rle_iou(full["segmentation"], gt["segmentation"])
            choose_forced = full_iou > raw_iou  # exact ties stay baseline deterministically
            row.update(raw_mask_iou=raw_iou, forced_mask_iou=full_iou, choose_forced=choose_forced)
            matched_by_image[row["image_id"]].append(index)
            if choose_forced:
                oracle.add(index)
        audit.append(row)

    rng = np.random.default_rng(seed)
    random_sets = []
    for _ in range(random_trials):
        selected = set()
        for image_id, candidates in matched_by_image.items():
            count = sum(index in oracle for index in candidates)
            if count:
                selected.update(int(x) for x in rng.choice(candidates, size=count, replace=False))
        if len(selected) != len(oracle):
            raise RuntimeError("random selector did not preserve oracle selected count")
        random_sets.append(selected)
    return oracle, random_sets, audit


def _mixed_records(baseline, forced, selected):
    out = copy.deepcopy(baseline)
    for index in selected:
        out[index]["segmentation"] = copy.deepcopy(forced[index]["segmentation"])
    return out


def _write_arm(out_dir, name, dt_records, template_dir, manifest_extra):
    out = out_dir / name
    out.mkdir(parents=True, exist_ok=False)
    for filename in ("gt_records.json", "images.json"):
        (out / filename).write_text((template_dir / filename).read_text(encoding="utf-8"), encoding="utf-8")
    (out / "dt_records.json").write_text(json.dumps(dt_records) + "\n", encoding="utf-8")
    manifest = json.loads((template_dir / "run_manifest.json").read_text(encoding="utf-8"))
    manifest.update(manifest_extra)
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out


def _point_map(gt_records, dt_records, images, num_classes):
    # Reuse the canonical evaluator through the fixed coordinate conversion in
    # bootstrap_paired_map; this keeps the Oracle and CI contracts identical.
    spec = importlib.util.spec_from_file_location(
        "_local_bootstrap_paired_map", ROOT / "tools" / "bootstrap_paired_map.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load repository bootstrap_paired_map.py")
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    from utils.coco_eval_utils import VHR10_CATEGORIES
    sizes = {int(x["image_id"]): (int(x["height"]), int(x["width"])) for x in images}
    ids = [int(x["image_id"]) for x in images]
    categories = VHR10_CATEGORIES if int(num_classes) == 10 else None
    gt = bootstrap._group(gt_records, sizes, "gt_records")
    dt = bootstrap._group(dt_records, sizes, "dt_records")
    value, _ = bootstrap._evaluate(ids, gt, dt, dt, "scores", None, sizes, categories)
    return float(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-dir", required=True, help="learned/current canvas probe arm")
    parser.add_argument("--forced-dir", required=True, help="same learned canvas with forced alpha=1 arm")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--contract-manifest", default=None,
                        help="For legacy canvas artifacts lacking dataset.num_classes: "
                             "same-checkpoint, same-image-ID manifest providing only "
                             "the dataset contract for new output records.")
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--random-trials", type=int, default=32)
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()
    if not 0 < args.match_iou <= 1 or args.random_trials < 1:
        raise SystemExit("--match-iou must be in (0,1], --random-trials >= 1")
    current, forced, out = (Path(x).resolve() for x in (args.current_dir, args.forced_dir, args.output_dir))
    if out.exists():
        raise SystemExit(f"output exists: {out}")
    base_dt = json.loads((current / "dt_records.json").read_text())
    forced_dt = json.loads((forced / "dt_records.json").read_text())
    gt = json.loads((current / "gt_records.json").read_text())
    images = json.loads((current / "images.json").read_text())
    current_manifest = json.loads((current / "run_manifest.json").read_text())
    forced_manifest = json.loads((forced / "run_manifest.json").read_text())
    if current_manifest.get("processed_image_ids") != forced_manifest.get("processed_image_ids"):
        raise SystemExit("current/forced processed image identities differ")
    if current_manifest.get("dataset") != forced_manifest.get("dataset"):
        raise SystemExit("current/forced dataset contracts differ")
    contract = current_manifest.get("dataset") or {}
    contract_source = "current_manifest"
    if not contract.get("num_classes"):
        if not args.contract_manifest:
            raise SystemExit("legacy current manifest lacks dataset contract; pass --contract-manifest")
        reference = json.loads(Path(args.contract_manifest).read_text(encoding="utf-8"))
        if reference.get("checkpoint") != current_manifest.get("checkpoint"):
            raise SystemExit("contract manifest checkpoint differs from current canvas artifact")
        if reference.get("processed_image_ids") != current_manifest.get("processed_image_ids"):
            raise SystemExit("contract manifest image identities differ from current canvas artifact")
        if int(reference.get("processed_images", -1)) != int(current_manifest.get("processed_images", -2)):
            raise SystemExit("contract manifest processed image count differs from current canvas artifact")
        contract = reference.get("dataset") or {}
        contract_source = str(Path(args.contract_manifest).resolve())
    num_classes = int(contract.get("num_classes", 0))
    if num_classes not in (1, 10):
        raise SystemExit("dataset.num_classes must be an explicit supported contract")
    oracle, random_sets, audit = make_selectors(base_dt, forced_dt, gt, args.match_iou, args.random_trials, args.seed)
    out.mkdir(parents=True)
    oracle_dt = _mixed_records(base_dt, forced_dt, oracle)
    # Never edit legacy input artifacts.  Upgrade only emitted manifests after
    # the checkpoint/image-identity checks above.
    template = out / "_template"
    template.mkdir()
    upgraded_manifest = dict(current_manifest, dataset=dict(contract), selector_contract_source=contract_source)
    for filename, value in (("gt_records.json", gt), ("images.json", images),
                            ("run_manifest.json", upgraded_manifest)):
        (template / filename).write_text(json.dumps(value, indent=2) + "\n")
    oracle_dir = _write_arm(out, "oracle", oracle_dt, template, dict(
        selector="gt_mask_iou_oracle", selected_detections=len(oracle), match_iou=args.match_iou,
        source_current=str(current), source_forced=str(forced)))
    random_maps = []
    for index, selected in enumerate(random_sets):
        records = _mixed_records(base_dt, forced_dt, selected)
        _write_arm(out, f"random_{index:02d}", records, template, dict(
            selector="per_image_count_matched_random", selected_detections=len(selected),
            random_seed=args.seed, random_trial=index, match_iou=args.match_iou,
            source_current=str(current), source_forced=str(forced)))
        random_maps.append(_point_map(gt, records, images, num_classes))
    summary = dict(
        protocol="post-hoc GT only selector; no GT in forward; unmatched detections stay current",
        match_iou=args.match_iou, random_trials=args.random_trials, random_seed=args.seed,
        detections=len(base_dt), matched_detections=sum(x["matched"] for x in audit),
        oracle_selected=len(oracle), current_mAP=_point_map(gt, base_dt, images, num_classes),
        forced_mAP=_point_map(gt, forced_dt, images, num_classes), oracle_mAP=_point_map(gt, oracle_dt, images, num_classes),
        random_mAP_mean=float(np.mean(random_maps)), random_mAP_p025=float(np.percentile(random_maps, 2.5)),
        random_mAP_p975=float(np.percentile(random_maps, 97.5)), random_mAP=random_maps,
        oracle_dir=str(oracle_dir), contract_source=contract_source,
    )
    for file in template.iterdir():
        file.unlink()
    template.rmdir()
    (out / "selector_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
