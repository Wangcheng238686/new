#!/usr/bin/env python3
"""Read-only B1 diagnosis for small failures that already have an RPN proposal.

The target set is fixed by the archived B0 scan.  B1 only repeats normal
deployment inference, then asks whether an aligned final box reaches the GT
but its mask fails, or whether the post-RPN final box is absent/mislocalized.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

os.environ.setdefault("SAM2_CKPT", "/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from data.ue_dual_stream_dataset import UEDualStreamDataset
from inference.infer_from_checkpoint import (
    _load_checkpoint, _load_model_state, _register_and_build, _resolve_model_config,
    _resolve_sam2_repo, _select_state_dict, _snapshot,
)
from probe_small_rpn_bottleneck import box_iou_xyxy, gt_from_item, masks_from_prediction, normal_predict, rle_iou


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--b0-dir", type=Path, required=True)
    parser.add_argument("--data-root", default="/data1/wangcheng/dataset/ue_coco_with_uav")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sam2-ckpt", default=os.environ["SAM2_CKPT"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-target-scenes", type=int, default=0, help="0 runs every B0 target scene.")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260916)
    parser.add_argument("--experiment-id", default="MHBEV-DIAG-20260915-B1")
    return parser.parse_args()


def typed_hash(value: Any) -> str:
    if hasattr(value, "to_ndarray"):
        value = value.to_ndarray()
    elif hasattr(value, "masks"):
        value = value.masks
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous().numpy()
    array = np.ascontiguousarray(value)
    payload = f"dtype={array.dtype};shape={tuple(array.shape)};".encode() + array.tobytes()
    return hashlib.sha256(payload).hexdigest()


def output_digest(output) -> dict[str, str]:
    instance = output[0].pred_instances
    return {name: typed_hash(getattr(instance, name)) for name in ("bboxes", "scores", "labels", "masks")}


def final_boxes(output) -> np.ndarray:
    boxes = output[0].pred_instances.bboxes
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().float().cpu().numpy()
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def failure_type(mask_iou: np.ndarray) -> tuple[list[tuple[int, int]], np.ndarray, dict[int, int]]:
    from scipy.optimize import linear_sum_assignment
    pairs = linear_sum_assignment(-mask_iou) if mask_iou.shape[0] and mask_iou.shape[1] else ([], [])
    matches = {int(g): int(p) for g, p in zip(*pairs) if mask_iou[g, p] >= .5}
    best = mask_iou.max(axis=1) if mask_iou.shape[1] else np.zeros(mask_iou.shape[0], dtype=np.float32)
    return list(zip(*pairs)), best, matches


def classify_final(mask_iou: np.ndarray, gi: int, max_mask_iou: float, matches: dict[int, int]) -> str:
    if gi in matches:
        return "matched_good" if float(mask_iou[gi, matches[gi]]) >= .75 else "matched_boundary_error"
    return "missed" if max_mask_iou < .1 else "partial_or_merge"


def bootstrap(per_scene: list[dict], samples: int, seed: int) -> dict:
    values = np.asarray([[r["mask_only"], r["box_lost"]] for r in per_scene], dtype=np.int64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    sums = values[draws].sum(axis=1)
    total = sums.sum(axis=1)
    valid = total > 0
    diff = (sums[valid, 0] - sums[valid, 1]) / total[valid]
    if not len(diff):
        raise RuntimeError("bootstrap has no target-containing draws")
    return {"samples_requested": samples, "samples_valid": int(len(diff)), "seed": seed,
            "difference_mask_only_minus_box_lost": float(np.mean(diff)),
            "ci95_percentile": [float(x) for x in np.percentile(diff, [2.5, 97.5])]}


def main() -> None:
    args = arguments()
    b0_summary_path, b0_rows_path = args.b0_dir / "summary.json", args.b0_dir / "instances.csv"
    if not b0_summary_path.is_file() or not b0_rows_path.is_file():
        raise FileNotFoundError("B1 requires archived B0 summary.json and instances.csv")
    b0 = json.loads(b0_summary_path.read_text())
    if not (b0["data"].get("complete") and all(b0["invariants"].values()) and b0["data"].get("scenes_processed") == 259 and b0["data"].get("gt_instances") == 5937):
        raise RuntimeError("B0 lineage is incomplete or has failed invariants")
    if Path(b0["checkpoint"]).resolve() != Path(args.checkpoint).resolve():
        raise RuntimeError("B1 checkpoint must exactly equal B0 checkpoint")
    if (Path(b0["data"].get("root", "")).resolve() != Path(args.data_root).resolve() or
            b0["data"].get("split") != "val" or b0["contract"].get("input_size") != [1024, 1024]):
        raise RuntimeError("B0 data/split/input contract does not match B1")
    b0_rows = list(csv.DictReader(b0_rows_path.open()))
    if len(b0_rows) != 5937 or len({r["scene_id"] for r in b0_rows}) != 259:
        raise RuntimeError("B0 CSV does not satisfy 259/5937 contract")
    targets = [r for r in b0_rows if r["area_bucket"] == "small" and
               r["final_failure"] in ("missed", "partial_or_merge") and r["rpn_recall_iou05"] == "True"]
    if len(targets) != 120 or len({(r["scene_id"], int(r["gt_index"])) for r in targets}) != 120:
        raise RuntimeError(f"B0 target membership mismatch: expected 120, got {len(targets)}")
    by_scene: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    for row in targets:
        if any(int(row[f"views_{name}"]) not in range(6) for name in ("correct", "xy_plus_80m", "scene_permuted")):
            raise RuntimeError(f"B0 geometry view count is outside [0,5] for {row['scene_id']}/{row['gt_index']}")
        by_scene[row["scene_id"]][int(row["gt_index"])] = row

    if not Path(args.sam2_ckpt).is_file():
        raise FileNotFoundError(args.sam2_ckpt)
    device = torch.device("cuda:0" if args.device == "cuda" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(20260916); torch.backends.cudnn.benchmark = False
    checkpoint = _load_checkpoint(Path(args.checkpoint)); snapshot = _snapshot(checkpoint)
    loader_args = SimpleNamespace(config=None, sam2_ckpt=args.sam2_ckpt, sam2_repo=None)
    _resolve_sam2_repo(loader_args, snapshot)
    model_config, source = _resolve_model_config(loader_args, snapshot)
    model = _register_and_build(model_config).to(device).eval()
    load_report = _load_model_state(model, _select_state_dict(checkpoint, "model"), False)
    if load_report["missing_keys"] or load_report["unexpected_keys"]:
        raise RuntimeError(f"strict model loading failed: {load_report}")

    dataset = UEDualStreamDataset(args.data_root, "val", image_size=(1024, 1024), load_uav=False)
    index_by_id = {sid: index for index, sid in enumerate(dataset.ids)}
    target_scenes = [sid for sid in dataset.ids if sid in by_scene]
    if len(index_by_id) != 259 or len(target_scenes) != len(by_scene):
        raise RuntimeError("B1 dataset scene IDs do not exactly cover B0 target scenes")
    n = len(target_scenes) if args.max_target_scenes == 0 else min(args.max_target_scenes, len(target_scenes))
    target_scenes = target_scenes[:n]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, parity, per_scene = [], [], []
    with torch.inference_mode():
        for sid in target_scenes:
            item = dataset[index_by_id[sid]]
            before, after = normal_predict(model, item, device), normal_predict(model, item, device)
            parity.append({"scene_id": sid, "prediction_equal": output_digest(before) == output_digest(after),
                           "target_count": len(by_scene[sid])})
            gt_masks, gt_rles, gt_boxes = gt_from_item(item)
            pred_rles, pred_boxes = masks_from_prediction(before), final_boxes(before)
            mask_iou = rle_iou(gt_rles, pred_rles)
            _, best_mask, matches = failure_type(mask_iou)
            box_iou = box_iou_xyxy(gt_boxes, pred_boxes)
            best_box = box_iou.max(axis=1) if len(pred_boxes) else np.zeros(len(gt_boxes), dtype=np.float32)
            mask_only = box_lost = 0
            for gi, b0_row in by_scene[sid].items():
                recomputed = classify_final(mask_iou, gi, float(best_mask[gi]), matches)
                if recomputed != b0_row["final_failure"] or best_mask[gi] >= .5:
                    raise RuntimeError(f"B0 target drift for {sid}/{gi}: {b0_row['final_failure']} vs {recomputed}")
                kind = "mask_only_failure" if best_box[gi] >= .5 else "post_rpn_box_lost_or_mislocalized"
                mask_only += kind == "mask_only_failure"; box_lost += kind == "post_rpn_box_lost_or_mislocalized"
                rows.append({"scene_id": sid, "gt_index": gi, "b0_final_failure": b0_row["final_failure"],
                             "max_final_mask_iou": float(best_mask[gi]), "max_final_box_iou": float(best_box[gi]),
                             "final_detection_count": int(len(pred_boxes)), "category": kind,
                             "views_correct": int(b0_row["views_correct"]),
                             "views_xy_plus_80m": int(b0_row["views_xy_plus_80m"]),
                             "views_scene_permuted": int(b0_row["views_scene_permuted"])})
            per_scene.append({"scene_id": sid, "mask_only": int(mask_only), "box_lost": int(box_lost)})

    def frac(kind): return float(np.mean([r["category"] == kind for r in rows])) if rows else None
    def coverage(kind, field):
        records = [r for r in rows if r["category"] == kind]
        covered = sum(r[field] >= 2 for r in records)
        return {"fraction": covered / len(records) if records else None, "covered": int(covered), "total": len(records)}
    data_complete = n == len(by_scene) and len(rows) == 120
    # The image-paired bootstrap universe is every official val image, not just
    # target-containing images.  Unprocessed target scenes are also zero here
    # during a smoke, which is harmless because smoke is forced INVALID.
    processed_stats = {row["scene_id"]: row for row in per_scene}
    bootstrap_per_scene = [processed_stats.get(sid, {"scene_id": sid, "mask_only": 0, "box_lost": 0})
                           for sid in dataset.ids]
    boot = bootstrap(bootstrap_per_scene, args.bootstrap_samples, args.bootstrap_seed)
    mask_fraction, box_fraction = frac("mask_only_failure"), frac("post_rpn_box_lost_or_mislocalized")
    ci_lo, ci_hi = boot["ci95_percentile"]
    if not data_complete or not all(r["prediction_equal"] for r in parity):
        decision = "INVALID"
    elif mask_fraction >= .60 and ci_lo > 0:
        decision = "ROI_LOCAL_MASK"
    elif box_fraction >= .60 and ci_hi < 0:
        decision = "POST_RPN_ROI_BOX"
    else:
        decision = "MIXED"
    summary = {"experiment_id": args.experiment_id, "arm": "B1-Frozen-PostRPN-Diagnosis", "status": "probe_only_no_training",
               "checkpoint": str(Path(args.checkpoint).resolve()), "model_config_source": source,
               "b0_lineage": {"dir": str(args.b0_dir.resolve()), "target_count_expected": 120,
                              "b0_complete": b0["data"]["complete"], "b0_invariants": b0["invariants"]},
               "data": {"split": "val", "bootstrap_scene_universe": len(dataset), "target_scenes_expected": len(by_scene), "target_scenes_processed": n,
                        "targets_expected": 120, "targets_processed": len(rows), "complete": data_complete},
               "invariants": {"processed_target_membership_exact": len(rows) == sum(len(by_scene[s]) for s in target_scenes),
                              "full_target_coverage": data_complete,
                              "prediction_parity_bitwise": all(r["prediction_equal"] for r in parity),
                              "b0_final_failure_recomputed": True,
                              "detector_uav_or_sham_intervention": "none"},
               "metrics": {"mask_only_fraction": mask_fraction, "post_rpn_box_lost_fraction": box_fraction,
                           "inherited_two_view_coverage": {kind: {name: coverage(kind, f"views_{name}") for name in ("correct", "xy_plus_80m", "scene_permuted")}
                                                             for kind in ("mask_only_failure", "post_rpn_box_lost_or_mislocalized")}},
               "bootstrap": boot, "decision": decision,
               "decision_rule": "ROI_LOCAL_MASK iff mask_only>=.60 and CI lower>0; POST_RPN_ROI_BOX iff box_lost>=.60 and CI upper<0; otherwise MIXED; failed lineage/data/parity is INVALID.",
               "interpretation": "B1 locates the post-RPN outcome stage only; it does not identify a causal UAV mechanism or permit training.",
               "parity": parity, "per_scene": per_scene}
    fields = list(rows[0]) if rows else []
    with (args.output_dir / "targets.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    (args.output_dir / "bootstrap.json").write_text(json.dumps(boot, indent=2) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "run_manifest.json").write_text(json.dumps({k: summary[k] for k in ("experiment_id", "arm", "status", "checkpoint", "b0_lineage", "data", "invariants", "decision")}, indent=2) + "\n")
    print(json.dumps({"output": str(args.output_dir / "summary.json"), "decision": decision,
                      "metrics": summary["metrics"], "invariants": summary["invariants"]}, indent=2))


if __name__ == "__main__":
    main()
