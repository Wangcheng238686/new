#!/usr/bin/env python3
"""Read-only B2 diagnosis: refinement versus ROI score/NMS loss for B1 targets."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("SAM2_CKPT", "/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from mmdet.structures.bbox import bbox2roi

from data.ue_dual_stream_dataset import UEDualStreamDataset
from inference.infer_from_checkpoint import (
    _load_checkpoint, _load_model_state, _register_and_build, _resolve_model_config,
    _resolve_sam2_repo, _select_state_dict, _snapshot,
)
from probe_small_postrpn_bottleneck import output_digest
from probe_small_rpn_bottleneck import (
    box_iou_xyxy, gt_from_item, masks_from_prediction, normal_predict, prepare, read_rpn, rle_iou,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--b1-dir", type=Path, required=True)
    parser.add_argument("--data-root", default="/data1/wangcheng/dataset/ue_coco_with_uav")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sam2-ckpt", default=os.environ["SAM2_CKPT"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-target-scenes", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260917)
    parser.add_argument("--experiment-id", default="MHBEV-DIAG-20260915-B2")
    return parser.parse_args()


def as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def bbox_readout(model, item, device: torch.device):
    """Deployment-equivalent RPN+bbox forward, stopped strictly before NMS."""
    data = prepare(model, item, device)
    x_base, _, _, _ = model.extract_feat(data["inputs"], data["data_samples"])
    # Deployment order in RSPrompterAnchor.predict: RPN consumes the raw neck
    # features.  Only roi_head.predict subsequently applies extra_pe.
    rpn = model.rpn_head.predict(x_base, data["data_samples"], rescale=False)[0]
    # Must reproduce RSPrompterAnchorRoIPromptHead.predict() exactly before
    # predict_bbox; create a new tuple so the passive readout cannot mutate x_base.
    x_roi = x_base
    if hasattr(model.roi_head, "extra_pe"):
        bs, _, h, w = x_base[0].shape
        mask_pe = torch.zeros((bs, h, w), device=x_base[0].device, dtype=torch.bool)
        image_feats_pe = model.roi_head.extra_pe(mask_pe)
        x_roi = tuple(feat + F.interpolate(image_feats_pe, size=feat.shape[-2:], mode="bilinear", align_corners=False)
                      for feat in x_base)
    if data["data_samples"][0].get("proposals", None) is not None:
        raise RuntimeError("B2 readout wrote proposals to data sample")
    rois = bbox2roi([rpn.bboxes])
    if len(rois) != len(rpn.bboxes):
        raise RuntimeError("RPN/ROI cardinality changed before B2 bbox readout")
    bbox = model.roi_head._bbox_forward(x_roi, rois)
    cls_score, bbox_pred = bbox["cls_score"], bbox["bbox_pred"]
    if isinstance(cls_score, list):
        cls_score = sum(cls_score) / float(len(cls_score))
    scores = F.softmax(cls_score.float(), dim=1)
    if scores.ndim != 2 or scores.shape[1] != 2 or len(scores) != len(rpn.bboxes):
        raise RuntimeError(f"unexpected single-class ROI cls score shape {tuple(scores.shape)}")
    decoded = model.roi_head.bbox_head.bbox_coder.decode(
        rois[:, 1:].float(), bbox_pred, data["data_samples"][0].metainfo["img_shape"])
    decoded = as_numpy(decoded)
    if decoded.shape != (len(rpn.bboxes), 4) or not np.isfinite(decoded).all():
        raise RuntimeError(f"decoded DeltaXYWH boxes violate contract: {decoded.shape}")
    return as_numpy(rpn.bboxes), as_numpy(rpn.scores), decoded, as_numpy(scores[:, 0])


def final_failure(mask_iou: np.ndarray, gi: int) -> float:
    return float(mask_iou[gi].max()) if mask_iou.shape[1] else 0.0


def bootstrap(per_scene: list[dict], samples: int, seed: int) -> dict:
    keys = ("bbox_refinement_lost", "roi_score_rejected", "nms_or_rank_loss")
    values = np.asarray([[row[key] for key in keys] for row in per_scene], dtype=np.int64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    sums = values[draws].sum(axis=1)
    total = sums.sum(axis=1)
    valid = total > 0
    fractions = sums[valid] / total[valid, None]
    winner = fractions.argmax(axis=1)
    diff = fractions[np.arange(len(fractions)), winner] - (1.0 - fractions[np.arange(len(fractions)), winner])
    # The winner must be fixed from the observed full sample, not selected per resample.
    observed = values.sum(axis=0)
    observed_winner = int(observed.argmax())
    fixed_diff = fractions[:, observed_winner] - (1.0 - fractions[:, observed_winner])
    return {"samples_requested": samples, "samples_valid": int(valid.sum()), "seed": seed,
            "categories": list(keys), "observed_winner": keys[observed_winner],
            "winner_minus_others_mean": float(np.mean(fixed_diff)),
            "ci95_percentile": [float(x) for x in np.percentile(fixed_diff, [2.5, 97.5])]} 


def main() -> None:
    args = arguments()
    b1_summary_file, b1_targets_file = args.b1_dir / "summary.json", args.b1_dir / "targets.csv"
    if not b1_summary_file.is_file() or not b1_targets_file.is_file():
        raise FileNotFoundError("B2 needs B1 summary.json and targets.csv")
    b1 = json.loads(b1_summary_file.read_text())
    if not (b1["data"].get("complete") and all(bool(v) for v in b1["invariants"].values()) and b1["decision"] == "POST_RPN_ROI_BOX"):
        raise RuntimeError("B1 lineage is incomplete, invalid, or not the required B1 decision")
    if Path(b1["checkpoint"]).resolve() != Path(args.checkpoint).resolve():
        raise RuntimeError("B2 checkpoint must exactly equal B1 checkpoint")
    b0_dir = Path(b1["b0_lineage"]["dir"])
    b0 = json.loads((b0_dir / "summary.json").read_text())
    if not (Path(b0["checkpoint"]).resolve() == Path(args.checkpoint).resolve() and b0["data"].get("complete") and
            b0["data"].get("scenes_processed") == 259 and b0["data"].get("gt_instances") == 5937 and
            b0["contract"].get("input_size") == [1024, 1024] and all(bool(v) for v in b0["invariants"].values())):
        raise RuntimeError("B0 lineage does not satisfy B2 fail-closed contract")
    if Path(b0["data"]["root"]).resolve() != Path(args.data_root).resolve() or b0["data"]["split"] != "val":
        raise RuntimeError("B0 root/split does not match B2")
    if not (b1["data"].get("target_scenes_expected") == 94 and b1["data"].get("targets_expected") == 120 and
            b1["data"].get("target_scenes_processed") == 94 and b1["data"].get("targets_processed") == 120):
        raise RuntimeError("B1 target-scene/target-count lineage is not exact")
    b1_rows = list(csv.DictReader(b1_targets_file.open()))
    targets = [r for r in b1_rows if r["category"] == "post_rpn_box_lost_or_mislocalized"]
    if len(b1_rows) != 120 or len(targets) != 101 or len({(r["scene_id"], int(r["gt_index"])) for r in targets}) != 101:
        raise RuntimeError("B1 target membership must be exactly 101 post-RPN box-lost rows")
    by_scene: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    for row in targets:
        by_scene[row["scene_id"]][int(row["gt_index"])] = row

    if not Path(args.sam2_ckpt).is_file():
        raise FileNotFoundError(args.sam2_ckpt)
    device = torch.device("cuda:0" if args.device == "cuda" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(20260917); torch.backends.cudnn.benchmark = False
    checkpoint = _load_checkpoint(Path(args.checkpoint)); snapshot = _snapshot(checkpoint)
    loader_args = SimpleNamespace(config=None, sam2_ckpt=args.sam2_ckpt, sam2_repo=None)
    _resolve_sam2_repo(loader_args, snapshot)
    model_config, source = _resolve_model_config(loader_args, snapshot)
    model = _register_and_build(model_config).to(device).eval()
    report = _load_model_state(model, _select_state_dict(checkpoint, "model"), False)
    if report["missing_keys"] or report["unexpected_keys"]:
        raise RuntimeError(f"strict model loading failed: {report}")
    if type(model.roi_head.bbox_head).__name__ != "Shared2FCBBoxHead" or type(model.roi_head.bbox_head.bbox_coder).__name__ != "DeltaXYWHBBoxCoder":
        raise RuntimeError("B2 is pre-registered only for Shared2FCBBoxHead + DeltaXYWHBBoxCoder")
    score_thr = float(model.roi_head.test_cfg.score_thr)
    live_test_cfg = json.loads(json.dumps(dict(model.roi_head.test_cfg), default=str))
    live_bbox_contract = {"bbox_head_type": type(model.roi_head.bbox_head).__name__,
                          "bbox_coder_type": type(model.roi_head.bbox_head.bbox_coder).__name__}

    dataset = UEDualStreamDataset(args.data_root, "val", image_size=(1024, 1024), load_uav=False)
    index_by_id = {sid: i for i, sid in enumerate(dataset.ids)}
    target_scenes = [sid for sid in dataset.ids if sid in by_scene]
    if len(dataset) != 259 or len(target_scenes) != len(by_scene):
        raise RuntimeError("dataset does not exactly cover B2 target scenes")
    n = len(target_scenes) if args.max_target_scenes == 0 else min(args.max_target_scenes, len(target_scenes))
    target_scenes = target_scenes[:n]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, parity, per_scene = [], [], []
    with torch.inference_mode():
        for sid in target_scenes:
            item = dataset[index_by_id[sid]]
            before = normal_predict(model, item, device)
            proposal_boxes, proposal_scores, decoded_boxes, fg_scores = bbox_readout(model, item, device)
            if len(proposal_boxes) != 1000:
                raise RuntimeError(f"B2 requires deployed 1000 RPN proposals, got {len(proposal_boxes)} for {sid}")
            after = normal_predict(model, item, device)
            parity.append({"scene_id": sid, "prediction_equal": output_digest(before) == output_digest(after),
                           "target_count": len(by_scene[sid]), "rpn_proposal_count": int(len(proposal_boxes))})
            gt_masks, gt_rles, gt_boxes = gt_from_item(item)
            mask_iou = rle_iou(gt_rles, masks_from_prediction(before))
            final_boxes = before[0].pred_instances.bboxes.detach().float().cpu().numpy()
            final_iou = box_iou_xyxy(gt_boxes, final_boxes)
            counts = {"bbox_refinement_lost": 0, "roi_score_rejected": 0, "nms_or_rank_loss": 0}
            for gi, b1_row in by_scene[sid].items():
                pre_iou = box_iou_xyxy(gt_boxes[gi:gi + 1], proposal_boxes)[0]
                pstar = int(pre_iou.argmax())
                if pre_iou[pstar] < .5:
                    raise RuntimeError(f"B0/B2 P* drift for {sid}/{gi}: {pre_iou[pstar]}")
                decoded_iou = box_iou_xyxy(gt_boxes[gi:gi + 1], decoded_boxes)[0]
                aligned = decoded_iou >= .5
                max_final_iou = float(final_iou[gi].max()) if len(final_boxes) else 0.0
                if final_failure(mask_iou, gi) >= .5 or max_final_iou >= .5:
                    raise RuntimeError(f"B1 final failure drift for {sid}/{gi}")
                if not aligned.any():
                    category = "bbox_refinement_lost"; aligned_best_score = None
                else:
                    aligned_best_score = float(fg_scores[aligned].max())
                    category = "roi_score_rejected" if aligned_best_score <= score_thr else "nms_or_rank_loss"
                counts[category] += 1
                rows.append({"scene_id": sid, "gt_index": gi, "b1_category": b1_row["category"],
                             "pstar_index": pstar, "pstar_pre_rpn_iou": float(pre_iou[pstar]),
                             "pstar_decoded_iou": float(decoded_iou[pstar]),
                             "pstar_raw_fg_score": float(fg_scores[pstar]), "pstar_raw_score_margin": float(fg_scores[pstar] - score_thr),
                             "max_decoded_iou": float(decoded_iou.max()),
                             "aligned_decoded_count": int(aligned.sum()), "aligned_best_fg_score": aligned_best_score,
                             "aligned_best_score_margin": None if aligned_best_score is None else float(aligned_best_score - score_thr),
                             "score_thr": score_thr, "max_final_box_iou": max_final_iou,
                             "max_final_mask_iou": final_failure(mask_iou, gi), "category": category,
                             "views_correct": int(b1_row["views_correct"]), "views_xy_plus_80m": int(b1_row["views_xy_plus_80m"]),
                             "views_scene_permuted": int(b1_row["views_scene_permuted"])})
            per_scene.append({"scene_id": sid, **counts})

    categories = ("bbox_refinement_lost", "roi_score_rejected", "nms_or_rank_loss")
    def fraction(category): return float(np.mean([r["category"] == category for r in rows])) if rows else None
    data_complete = n == len(by_scene) and len(rows) == 101
    processed = {row["scene_id"]: row for row in per_scene}
    boot_rows = [processed.get(sid, {"scene_id": sid, **{key: 0 for key in categories}}) for sid in dataset.ids]
    boot = bootstrap(boot_rows, args.bootstrap_samples, args.bootstrap_seed)
    fractions = {key: fraction(key) for key in categories}
    winner = max(fractions, key=fractions.get)
    ci_lo, ci_hi = boot["ci95_percentile"]
    if not data_complete or not all(r["prediction_equal"] for r in parity):
        decision = "INVALID"
    elif fractions[winner] >= .60 and ci_lo > 0:
        decision = {"bbox_refinement_lost": "BBOX_REFINEMENT", "roi_score_rejected": "ROI_SCORE", "nms_or_rank_loss": "NMS_OR_RANK"}[winner]
    else:
        decision = "MIXED"
    summary = {"experiment_id": args.experiment_id, "arm": "B2-Frozen-PostRPN-Mechanism", "status": "probe_only_no_training",
               "checkpoint": str(Path(args.checkpoint).resolve()), "model_config_source": source,
               "b1_lineage": {"dir": str(args.b1_dir.resolve()), "decision": b1["decision"], "b0_dir": str(b0_dir), "target_count_expected": 101},
               "data": {"split": "val", "bootstrap_scene_universe": len(dataset), "target_scenes_expected": len(by_scene), "target_scenes_processed": n,
                        "targets_expected": 101, "targets_processed": len(rows), "complete": data_complete},
               "live_roi_test_cfg": live_test_cfg,
               "live_bbox_contract": live_bbox_contract,
               "invariants": {"b1_target_membership_exact": len(rows) == sum(len(by_scene[s]) for s in target_scenes),
                              "full_target_coverage": data_complete, "prediction_parity_bitwise": all(r["prediction_equal"] for r in parity),
                              "pstar_pre_iou_ge05": all(r["pstar_pre_rpn_iou"] >= .5 for r in rows),
                              "b1_final_failure_recomputed": all(r["max_final_box_iou"] < .5 and r["max_final_mask_iou"] < .5 for r in rows),
                              "detector_uav_or_sham_intervention": "none"},
               "metrics": {"fractions": fractions, "pstar_pre_iou_mean": float(np.mean([r["pstar_pre_rpn_iou"] for r in rows])),
                           "pstar_decoded_iou_mean": float(np.mean([r["pstar_decoded_iou"] for r in rows])),
                           "max_decoded_iou_mean": float(np.mean([r["max_decoded_iou"] for r in rows]))},
               "bootstrap": boot, "decision": decision,
               "decision_rule": "Largest category must be >=.60 and its fixed-category minus all-other bootstrap CI lower>0; otherwise MIXED; failed lineage/data/parity/P* is INVALID.",
               "interpretation": "B2 localizes the deployed post-RPN mechanism only; it is not causal evidence for UAV or authorization to train.",
               "parity": parity, "per_scene": per_scene}
    fields = list(rows[0]) if rows else []
    with (args.output_dir / "targets.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    (args.output_dir / "bootstrap.json").write_text(json.dumps(boot, indent=2) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "run_manifest.json").write_text(json.dumps({key: summary[key] for key in ("experiment_id", "arm", "status", "checkpoint", "b1_lineage", "data", "live_bbox_contract", "invariants", "decision")}, indent=2) + "\n")
    print(json.dumps({"output": str(args.output_dir / "summary.json"), "decision": decision, "metrics": summary["metrics"], "invariants": summary["invariants"]}, indent=2))


if __name__ == "__main__":
    main()
