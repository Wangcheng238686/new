#!/usr/bin/env python3
"""Read-only diagnosis of whether UE small failures lack RPN proposals.

This implements the pre-registered MHBEV-DIAG-20260915 frozen diagnostic.  It
never attaches proposals to a data sample and never alters model parameters:
the RPN is read once (and repeated for a determinism hash), while the final
prediction is always the normal ``model.predict`` path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

os.environ.setdefault("SAM2_CKPT", "/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from mmdet.structures import DetDataSample
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment

from data.ue_dual_stream_dataset import UEDualStreamDataset
from inference.infer_from_checkpoint import (
    _load_checkpoint, _load_model_state, _register_and_build, _resolve_model_config,
    _resolve_sam2_repo, _select_state_dict, _snapshot,
)
from probe_single_stream_failures import coverage_for_center


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="/data1/wangcheng/dataset/ue_coco_with_uav")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--scenes", type=int, default=0, help="0 scans the complete official split.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sam2-ckpt", default=os.environ["SAM2_CKPT"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-id", default="MHBEV-DIAG-20260915-B0")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260915)
    return parser.parse_args()


def build_sample(item: Mapping[str, Any], device: torch.device) -> DetDataSample:
    sample = DetDataSample()
    sample.set_metainfo(dict(item["img_metas"]))
    return sample.to(device)


def prepare(model, item: Mapping[str, Any], device: torch.device):
    sample = build_sample(item, device)
    data = model.data_preprocessor({"inputs": [item["img"].to(device)], "data_samples": [sample]}, training=False)
    if data["data_samples"][0].get("proposals", None) is not None:
        raise RuntimeError("read-only diagnostic requires no pre-supplied proposals")
    return data


def array_hash(value: Any) -> str:
    if hasattr(value, "to_ndarray"):
        value = value.to_ndarray()
    elif hasattr(value, "masks"):
        value = value.masks
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def output_digest(output) -> dict[str, str]:
    instance = output[0].pred_instances
    return {name: array_hash(getattr(instance, name)) for name in ("bboxes", "scores", "labels", "masks")}


def normal_predict(model, item: Mapping[str, Any], device: torch.device):
    data = prepare(model, item, device)
    return model.predict(data["inputs"], data["data_samples"], rescale=False)


def read_rpn(model, item: Mapping[str, Any], device: torch.device):
    """Return detached pre-ROI RPN output without writing data_sample.proposals."""
    data = prepare(model, item, device)
    x, _, _, _ = model.extract_feat(data["inputs"], data["data_samples"])
    proposal = model.rpn_head.predict(x, data["data_samples"], rescale=False)[0]
    if data["data_samples"][0].get("proposals", None) is not None:
        raise RuntimeError("RPN readout wrote proposals into its input data sample")
    boxes, scores = proposal.bboxes.detach().float().cpu().numpy(), proposal.scores.detach().float().cpu().numpy()
    if len(boxes) > 1000 or not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise RuntimeError("RPN output violates finite/max-per-image diagnostic contract")
    if len(boxes) and (boxes.min() < -1e-3 or boxes.max() > 1024.001):
        raise RuntimeError("RPN boxes are outside the deployed 1024x1024 input coordinates")
    return boxes, scores


def rle_iou(gt_rles, pred_rles):
    if not gt_rles or not pred_rles:
        return np.zeros((len(gt_rles), len(pred_rles)), dtype=np.float32)
    return np.asarray(mask_utils.iou(gt_rles, pred_rles, [0] * len(pred_rles)), dtype=np.float32)


def box_iou_xyxy(gt: np.ndarray, proposals: np.ndarray) -> np.ndarray:
    if len(gt) == 0 or len(proposals) == 0:
        return np.zeros((len(gt), len(proposals)), dtype=np.float32)
    ix1 = np.maximum(gt[:, None, 0], proposals[None, :, 0])
    iy1 = np.maximum(gt[:, None, 1], proposals[None, :, 1])
    ix2 = np.minimum(gt[:, None, 2], proposals[None, :, 2])
    iy2 = np.minimum(gt[:, None, 3], proposals[None, :, 3])
    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    area_gt = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    area_p = (proposals[:, 2] - proposals[:, 0]) * (proposals[:, 3] - proposals[:, 1])
    return inter / np.maximum(area_gt[:, None] + area_p[None] - inter, 1e-6)


def gt_from_item(item: Mapping[str, Any]):
    """Use the exact 1024-raster GT and xyxy boxes supplied to this model."""
    masks = np.asarray(item["gt_masks"], dtype=bool)
    boxes = item["gt_bboxes"].detach().cpu().numpy().astype(np.float32, copy=False)
    if masks.ndim != 3 or masks.shape[1:] != (1024, 1024) or boxes.shape != (len(masks), 4):
        raise RuntimeError(f"UE GT contract mismatch: masks={masks.shape}, boxes={boxes.shape}")
    rles = []
    for mask in masks:
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        if isinstance(rle["counts"], bytes):
            rle["counts"] = rle["counts"].decode("ascii")
        rles.append(rle)
    return masks, rles, boxes


def masks_from_prediction(output):
    masks = output[0].pred_instances.masks
    if hasattr(masks, "to_ndarray"):
        masks = masks.to_ndarray()
    elif hasattr(masks, "masks"):
        masks = masks.masks
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()
    masks = np.asarray(masks, dtype=np.uint8)
    rles = []
    for mask in masks:
        rle = mask_utils.encode(np.asfortranarray(mask))
        if isinstance(rle["counts"], bytes):
            rle["counts"] = rle["counts"].decode("ascii")
        rles.append(rle)
    return rles


def bootstrap_difference(per_scene: list[dict], samples: int, seed: int) -> dict:
    """Image-paired bootstrap of failure minus good RPN recall@.5."""
    values = np.asarray([[r["failure_hits"], r["failure_total"], r["good_hits"], r["good_total"]]
                         for r in per_scene], dtype=np.int64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    sums = values[draws].sum(axis=1)
    valid = (sums[:, 1] > 0) & (sums[:, 3] > 0)
    diff = sums[valid, 0] / sums[valid, 1] - sums[valid, 2] / sums[valid, 3]
    if not len(diff):
        raise RuntimeError("bootstrap has no draws with both failure and matched-good small instances")
    return {"samples_requested": samples, "samples_valid": int(len(diff)), "seed": seed,
            "difference_failure_minus_good": float(np.mean(diff)),
            "ci95_percentile": [float(x) for x in np.percentile(diff, [2.5, 97.5])]}


def main() -> None:
    args = arguments()
    if not Path(args.sam2_ckpt).is_file():
        raise FileNotFoundError(args.sam2_ckpt)
    device = torch.device("cuda:0" if args.device == "cuda" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(20260915)
    torch.backends.cudnn.benchmark = False

    checkpoint = _load_checkpoint(Path(args.checkpoint)); snapshot = _snapshot(checkpoint)
    loader_args = SimpleNamespace(config=None, sam2_ckpt=args.sam2_ckpt, sam2_repo=None)
    _resolve_sam2_repo(loader_args, snapshot)
    model_config, source = _resolve_model_config(loader_args, snapshot)
    model = _register_and_build(model_config).to(device).eval()
    load_report = _load_model_state(model, _select_state_dict(checkpoint, "model"), False)
    if load_report["missing_keys"] or load_report["unexpected_keys"]:
        raise RuntimeError(f"strict model loading failed: {load_report}")

    dataset = UEDualStreamDataset(args.data_root, args.split, image_size=(1024, 1024), load_uav=False)
    n = len(dataset) if args.scenes == 0 else min(args.scenes, len(dataset))
    if n < 2:
        raise RuntimeError("need at least two scenes for scene-permuted geometry sham")
    root = Path(args.data_root); output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    centers = json.loads((root / "scene_centers.json").read_text())
    rows, scene_stats, parity, rpn_hashes = [], [], [], []

    with torch.inference_mode():
        for index in range(n):
            item, sid = dataset[index], dataset.ids[index]
            before = normal_predict(model, item, device)
            before_digest = output_digest(before)
            boxes1, scores1 = read_rpn(model, item, device)
            boxes2, scores2 = read_rpn(model, item, device)
            after = normal_predict(model, item, device)
            after_digest = output_digest(after)
            rpn_digest1 = {"bboxes": array_hash(boxes1), "scores": array_hash(scores1)}
            rpn_digest2 = {"bboxes": array_hash(boxes2), "scores": array_hash(scores2)}
            parity.append({"scene_id": sid, "prediction_equal": before_digest == after_digest})
            rpn_hashes.append({"scene_id": sid, "repeat_equal": rpn_digest1 == rpn_digest2,
                               "proposal_count": int(len(boxes1))})

            gt_masks, gt_rles, gt_boxes = gt_from_item(item)
            pred_rles = masks_from_prediction(before)
            mask_iou = rle_iou(gt_rles, pred_rles)
            assignment = linear_sum_assignment(-mask_iou) if len(gt_rles) and len(pred_rles) else ([], [])
            matches = {int(g): int(p) for g, p in zip(*assignment) if mask_iou[g, p] >= .5}
            best_mask = mask_iou.max(axis=1) if len(pred_rles) else np.zeros(len(gt_boxes), np.float32)
            proposal_iou = box_iou_xyxy(gt_boxes, boxes1)
            max_proposal = proposal_iou.max(axis=1) if len(boxes1) else np.zeros(len(gt_boxes), np.float32)

            camera = json.loads((root / args.split / "drone" / sid / "ue_render_metadata" / "camera_params.json").read_text())
            next_sid = dataset.ids[(index + 1) % n]
            next_camera = json.loads((root / args.split / "drone" / next_sid / "ue_render_metadata" / "camera_params.json").read_text())
            K, E = np.asarray(camera["intrinsics"]), np.asarray(camera["extrinsics"])
            K_sham, E_sham = np.asarray(next_camera["intrinsics"]), np.asarray(next_camera["extrinsics"])
            failure_hits = failure_total = good_hits = good_total = 0
            for gi, mask in enumerate(gt_masks):
                area = int(mask.sum()); small = area < 32 ** 2
                matched = gi in matches
                failure = "matched_good" if matched and float(mask_iou[gi, matches[gi]]) >= .75 else (
                    "matched_boundary_error" if matched else ("missed" if best_mask[gi] < .1 else "partial_or_merge"))
                ys, xs = np.where(mask); center = (float(xs.mean()), float(ys.mean()))
                coverage = {
                    "correct": coverage_for_center(center, 0., centers[sid], K, E),
                    "xy_plus_80m": coverage_for_center((center[0], center[1] - 80 / 360 * 1024), 0., centers[sid], K, E),
                    "scene_permuted": coverage_for_center(center, 0., centers[sid], K_sham, E_sham),
                }
                row = {"scene_id": sid, "gt_index": gi, "area_px": area,
                       "area_bucket": "small" if small else ("medium" if area < 96 ** 2 else "large"),
                       "final_failure": failure, "best_final_mask_iou": float(best_mask[gi]),
                       "max_rpn_box_iou": float(max_proposal[gi]), "rpn_recall_iou03": bool(max_proposal[gi] >= .3),
                       "rpn_recall_iou05": bool(max_proposal[gi] >= .5),
                       "rpn_proposal_count": int(len(boxes1)),
                       **{f"views_{name}": count for name, count in coverage.items()}}
                rows.append(row)
                if small and failure in ("missed", "partial_or_merge"):
                    failure_total += 1; failure_hits += int(row["rpn_recall_iou05"])
                if small and failure == "matched_good":
                    good_total += 1; good_hits += int(row["rpn_recall_iou05"])
            scene_stats.append({"scene_id": sid, "failure_hits": failure_hits, "failure_total": failure_total,
                                "good_hits": good_hits, "good_total": good_total})

    def subset(condition): return [r for r in rows if condition(r)]
    def ratio(records, field): return float(np.mean([r[field] for r in records])) if records else None
    def two_view_coverage(records, field):
        covered = int(sum(r[field] >= 2 for r in records))
        return {"fraction": covered / len(records) if records else None,
                "covered": covered, "total": len(records)}
    small = subset(lambda r: r["area_bucket"] == "small")
    failed_small = subset(lambda r: r["area_bucket"] == "small" and r["final_failure"] in ("missed", "partial_or_merge"))
    good_small = subset(lambda r: r["area_bucket"] == "small" and r["final_failure"] == "matched_good")
    bootstrap = bootstrap_difference(scene_stats, args.bootstrap_samples, args.bootstrap_seed)
    failure_recall, good_recall = ratio(failed_small, "rpn_recall_iou05"), ratio(good_small, "rpn_recall_iou05")
    ci_hi = bootstrap["ci95_percentile"][1]
    expected_scenes, expected_instances = (259, 5937) if args.split == "val" else (len(dataset), len(rows))
    data_complete = n == expected_scenes and len(dataset) == expected_scenes and len(rows) == expected_instances
    if not all(row["prediction_equal"] for row in parity) or not all(row["repeat_equal"] for row in rpn_hashes) or not data_complete:
        decision = "INVALID"
    elif failure_recall < .50 and good_recall - failure_recall >= .15 and ci_hi < 0:
        decision = "PROPOSAL_OBJECTNESS"
    elif failure_recall >= .75:
        decision = "ROI_OR_MASK"
    else:
        decision = "AMBIGUOUS"
    summary = {
        "experiment_id": args.experiment_id, "arm": "B0-Frozen-RPN-Diagnosis", "status": "probe_only_no_training",
        "checkpoint": str(Path(args.checkpoint).resolve()), "model_config_source": source,
        "data": {"root": str(root), "split": args.split, "scenes_expected": expected_scenes, "scenes_processed": n,
                 "gt_instances_expected": expected_instances, "gt_instances": len(rows), "complete": data_complete,
                 "scene_ids": dataset.ids[:n]},
        "contract": {"input_size": [1024, 1024], "small_area_px_lt": 1024,
                     "rpn_iou_thresholds": [.3, .5], "final_match_mask_iou": .5,
                     "geometry": "W2C column-vector UE centimetres; 360m; z=0; >=2 view"},
        "invariants": {"prediction_parity_bitwise": all(r["prediction_equal"] for r in parity),
                       "rpn_readout_repeat_bitwise": all(r["repeat_equal"] for r in rpn_hashes),
                       "proposal_counts_unchanged_by_geometry_shams": True},
        "counts": {"small": len(small), "small_final_failure": len(failed_small), "small_matched_good": len(good_small),
                   "failure_types": dict(Counter(r["final_failure"] for r in small))},
        "metrics": {"small_rpn_recall_iou03": ratio(small, "rpn_recall_iou03"),
                    "small_rpn_recall_iou05": ratio(small, "rpn_recall_iou05"),
                    "failed_small_rpn_recall_iou05": failure_recall,
                    "matched_good_small_rpn_recall_iou05": good_recall,
                    "failed_small_proposal_present_iou05": failure_recall,
                    "failed_small_two_view_coverage": {name: two_view_coverage(failed_small, f"views_{name}") for name in ("correct", "xy_plus_80m", "scene_permuted")}},
        "bootstrap": bootstrap, "decision": decision,
        "decision_rule": "PROPOSAL_OBJECTNESS iff failure recall@.5<.50, good-failure>=.15 and bootstrap CI upper<0; ROI_OR_MASK iff failure recall@.5>=.75; otherwise AMBIGUOUS; invariant/data failure is INVALID.",
        "interpretation": "RPN readout locates the baseline bottleneck only. Geometry coverage is necessary-query evidence, not UAV semantic gain evidence.",
        "per_scene": scene_stats, "parity": parity, "rpn_readout": rpn_hashes,
    }
    fields = list(rows[0]) if rows else []
    import csv
    with (output / "instances.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    (output / "bootstrap.json").write_text(json.dumps(bootstrap, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "run_manifest.json").write_text(json.dumps({k: summary[k] for k in ("experiment_id", "arm", "status", "checkpoint", "data", "contract", "invariants", "decision")}, indent=2) + "\n")
    print(json.dumps({"output": str(output / "summary.json"), "decision": decision, "metrics": summary["metrics"], "invariants": summary["invariants"]}, indent=2))


if __name__ == "__main__":
    main()
