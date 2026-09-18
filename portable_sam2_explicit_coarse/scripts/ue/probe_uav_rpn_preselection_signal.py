#!/usr/bin/env python3
"""C1 frozen UAV signal probe for dense RPN preselection candidates.

No detector/RPN/UAV parameter or proposal is modified.  This is deliberately
not an AP or proposal-recall intervention: it asks whether frozen UAV content
can rank an already-existing raw dense RPN candidate above a score-matched
background candidate on B0's RPN-miss targets.
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
from typing import Any

os.environ.setdefault("SAM2_CKPT", "/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data.ue_dual_stream_dataset import UEDualStreamDataset
from inference.infer_from_checkpoint import (
    _load_checkpoint, _load_model_state, _register_and_build, _resolve_model_config,
    _resolve_sam2_repo, _select_state_dict, _snapshot,
)
from probe_a1_frozen import encode_p2, make_uav_encoder
from probe_small_postrpn_bottleneck import output_digest
from probe_small_rpn_bottleneck import array_hash, box_iou_xyxy, gt_from_item, normal_predict, prepare, read_rpn
from probe_uav_roi_score_signal import model_param_hash, uav_descriptor

TRAIN_RATIO = 0.1797752808988764
TRAIN_SEED = 44
TARGETS = 107
CONDITIONS = ("correct", "xy", "perm")
VAL_FIELDS = [
    "scene_id", "gt_index", "pair_identity_sha256", "b0_final_failure",
    "b0_max_rpn_box_iou", "post_nms_replay_max_iou", "raw_geometry_available",
    "pos_candidate", "neg_candidate", "pos_box_xyxy", "neg_box_xyxy", "pos_raw_objectness",
    "neg_raw_objectness", "positive_label", "negative_label", "raw_pos_iou", "score_gap",
    "score_match", "common_valid", "evaluation_status",
] + [f"{c}_pair_identity_sha256" for c in CONDITIONS] + [field for c in CONDITIONS for field in (
    f"{c}_pos_min_support", f"{c}_pos_valid_points", f"{c}_neg_min_support", f"{c}_neg_valid_points",
    f"{c}_pos_readout_score", f"{c}_neg_readout_score", f"{c}_win",
)]


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--b0-dir", type=Path, required=True)
    p.add_argument("--data-root", default="/data1/wangcheng/dataset/ue_coco_with_uav")
    p.add_argument("--device", default="cuda")
    p.add_argument("--sam2-ckpt", default=os.environ["SAM2_CKPT"])
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-train-scenes", type=int, default=0)
    p.add_argument("--max-val-scenes", type=int, default=0)
    p.add_argument("--bootstrap-samples", type=int, default=10_000)
    p.add_argument("--bootstrap-seed", type=int, default=20260919)
    p.add_argument("--experiment-id", default="MHBEV-DIAG-20260915-C1")
    return p.parse_args()


def sha_record(record: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def dense_raw_rpn(model, item: dict, device: torch.device):
    """Decode every deployed dense RPN anchor before score filtering/NMS, read-only."""
    data = prepare(model, item, device)
    x, _, _, _ = model.extract_feat(data["inputs"], data["data_samples"])
    cls_scores, bbox_preds = model.rpn_head(x)
    sizes = [tensor.shape[-2:] for tensor in cls_scores]
    priors = model.rpn_head.prior_generator.grid_priors(sizes, dtype=cls_scores[0].dtype, device=device)
    all_boxes, all_scores, layouts = [], [], []
    img_shape = data["data_samples"][0].metainfo["img_shape"]
    for cls, delta, prior in zip(cls_scores, bbox_preds, priors):
        cls = cls[0].permute(1, 2, 0).reshape(-1, model.rpn_head.cls_out_channels)
        if model.rpn_head.use_sigmoid_cls:
            score = cls.sigmoid().reshape(-1)
        else:
            score = cls.softmax(-1)[:, :-1].reshape(-1)
        dim = model.rpn_head.bbox_coder.encode_size
        delta = delta[0].permute(1, 2, 0).reshape(-1, dim)
        box = model.rpn_head.bbox_coder.decode(prior, delta, max_shape=img_shape)
        if not (len(prior) == len(delta) == len(score) == len(box)):
            raise RuntimeError(f"dense RPN anchor cardinality mismatch: priors={len(prior)}, delta={len(delta)}, score={len(score)}, boxes={len(box)}")
        all_boxes.append(box.detach().float().cpu().numpy())
        all_scores.append(score.detach().float().cpu().numpy())
        layouts.append({"prior_count": int(len(prior)), "delta_count": int(len(delta)), "score_count": int(len(score)), "decoded_count": int(len(box))})
    if data["data_samples"][0].get("proposals", None) is not None:
        raise RuntimeError("dense raw readout wrote proposals")
    boxes, scores = np.concatenate(all_boxes), np.concatenate(all_scores)
    if not (np.isfinite(boxes).all() and np.isfinite(scores).all() and len(boxes) == len(scores)):
        raise RuntimeError("dense RPN output contract failed")
    return boxes, scores, layouts


def pair_from_dense(boxes: np.ndarray, scores: np.ndarray, gt_boxes: np.ndarray, gi: int):
    ious = box_iou_xyxy(gt_boxes, boxes)
    pos = int(ious[gi].argmax())
    pos_iou = float(ious[gi, pos])
    if pos_iou < .5:
        return None
    background = ious.max(axis=0) < .1
    if not background.any():
        return None
    bg = np.flatnonzero(background)
    neg = int(bg[np.abs(scores[bg] - scores[pos]).argmin()])
    return pos, neg, pos_iou, float(abs(scores[pos] - scores[neg]))


def identity(scene_id: str, gi: int, pos: int, neg: int, boxes: np.ndarray, scores: np.ndarray):
    record = {"scene_id": scene_id, "gt_index": int(gi), "pos_candidate": int(pos), "neg_candidate": int(neg),
              "pos_box_xyxy": [float(x) for x in boxes[pos]], "neg_box_xyxy": [float(x) for x in boxes[neg]],
              "pos_raw_objectness": float(scores[pos]), "neg_raw_objectness": float(scores[neg]),
              "positive_label": 1, "negative_label": 0}
    return sha_record(record), record


def boot(per_scene: list[dict], samples: int, seed: int):
    values = np.asarray([[r["correct"], r["xy"], r["perm"], r["pairs"]] for r in per_scene], dtype=np.int64)
    rng = np.random.default_rng(seed); draw = rng.integers(0, len(values), size=(samples, len(values)))
    sums = values[draw].sum(axis=1); valid = sums[:, 3] > 0
    if not valid.any():
        return {"samples_requested": samples, "samples_valid": 0, "seed": seed,
                "correct_minus_xy_ci95": [None, None], "correct_minus_permuted_ci95": [None, None]}
    rates = [sums[valid, i] / sums[valid, 3] for i in range(3)]
    return {"samples_requested": samples, "samples_valid": int(valid.sum()), "seed": seed,
            "correct_minus_xy_ci95": [float(x) for x in np.percentile(rates[0] - rates[1], [2.5, 97.5])],
            "correct_minus_permuted_ci95": [float(x) for x in np.percentile(rates[0] - rates[2], [2.5, 97.5])]}


def empty_val_row(record: dict[str, Any], b0: dict[str, str], replay_iou: float | None, available: bool,
                  raw_iou: float | None, gap: float | None):
    out = {**record, "b0_final_failure": b0["final_failure"], "b0_max_rpn_box_iou": float(b0["max_rpn_box_iou"]),
           "post_nms_replay_max_iou": replay_iou, "raw_geometry_available": available,
           "raw_pos_iou": raw_iou, "score_gap": gap, "score_match": bool(gap is not None and gap <= .02),
           "common_valid": False, "evaluation_status": "raw_geometry_unavailable"}
    for c in CONDITIONS:
        out[f"{c}_pair_identity_sha256"] = record.get("pair_identity_sha256")
        for suffix in ("pos_min_support", "pos_valid_points", "neg_min_support", "neg_valid_points",
                       "pos_readout_score", "neg_readout_score", "win"):
            out[f"{c}_{suffix}"] = None
    return out


def main():
    a = arguments(); dev = torch.device("cuda:0" if a.device == "cuda" else a.device)
    b0 = json.loads((a.b0_dir / "summary.json").read_text())
    b0_rows = list(csv.DictReader((a.b0_dir / "instances.csv").open()))
    if not (b0["data"]["complete"] and b0["data"]["scenes_processed"] == 259 and b0["data"]["gt_instances"] == 5937 and
            all(b0["invariants"].values()) and Path(b0["checkpoint"]).resolve() == Path(a.checkpoint).resolve()):
        raise RuntimeError("B0 lineage contract failed")
    targets = [r for r in b0_rows if r["area_bucket"] == "small" and r["final_failure"] in ("missed", "partial_or_merge") and r["rpn_recall_iou05"] == "False"]
    target_by_scene: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    for row in targets: target_by_scene[row["scene_id"]][int(row["gt_index"])] = row
    if len(targets) != TARGETS or sum(len(x) for x in target_by_scene.values()) != TARGETS:
        raise RuntimeError("B0 target identity/count contract failed")
    checkpoint = _load_checkpoint(Path(a.checkpoint)); snap = _snapshot(checkpoint)
    la = SimpleNamespace(config=None, sam2_ckpt=a.sam2_ckpt, sam2_repo=None); _resolve_sam2_repo(la, snap)
    cfg, source = _resolve_model_config(la, snap); model = _register_and_build(cfg).to(dev).eval()
    report = _load_model_state(model, _select_state_dict(checkpoint, "model"), False)
    if report["missing_keys"] or report["unexpected_keys"]: raise RuntimeError(f"strict model load failed: {report}")
    detector_hash_before = model_param_hash(model); encoder = make_uav_encoder(a, dev); encoder_hash_before = model_param_hash(encoder)
    train = UEDualStreamDataset(a.data_root, "train", image_size=(1024, 1024), train_subset=TRAIN_RATIO, seed=TRAIN_SEED, load_uav=True)
    val = UEDualStreamDataset(a.data_root, "val", image_size=(1024, 1024), load_uav=True)
    if len(train) != 128 or len(val) != 259 or set(train.ids) & set(val.ids): raise RuntimeError("C1 split contract failed")
    train_n = len(train) if not a.max_train_scenes else min(a.max_train_scenes, len(train))
    val_ids = [sid for sid in val.ids if sid in target_by_scene]
    val_n = len(val_ids) if not a.max_val_scenes else min(a.max_val_scenes, len(val_ids)); val_ids = val_ids[:val_n]
    a.output_dir.mkdir(parents=True, exist_ok=True)
    train_x, train_y, train_rows, parity_train, rpn_parity, dense_layout = [], [], [], [], [], []
    with torch.inference_mode():
        for index in range(train_n):
            item, sid = train[index], train.ids[index]; before = normal_predict(model, item, dev)
            rpn_before_boxes, rpn_before_scores = read_rpn(model, item, dev)
            boxes, scores, layouts = dense_raw_rpn(model, item, dev); dense_layout.extend({"scene_id": sid, "split": "train", "level": level, **layout} for level, layout in enumerate(layouts))
            fmap = encode_p2(encoder, item["drone_images"].to(dev)); masks, _, gt = gt_from_item(item)
            for gi, mask in enumerate(masks):
                if int(mask.sum()) >= 1024: continue
                pair = pair_from_dense(boxes, scores, gt, gi)
                if pair is None: continue
                pos, neg, pos_iou, gap = pair; sha, rec = identity(sid, gi, pos, neg, boxes, scores)
                dp, sp = uav_descriptor(fmap, boxes[pos], item, dev); dn, sn = uav_descriptor(fmap, boxes[neg], item, dev)
                if dp is None or dn is None: continue
                train_x.extend((dp, dn)); train_y.extend((1, 0)); train_rows.append({**rec, "pair_identity_sha256": sha, "raw_pos_iou": pos_iou, "score_gap": gap,
                    "correct_pos_min_support": sp["min_support"], "correct_pos_valid_points": sp["valid_points"], "correct_neg_min_support": sn["min_support"], "correct_neg_valid_points": sn["valid_points"]})
            rpn_after_boxes, rpn_after_scores = read_rpn(model, item, dev); after = normal_predict(model, item, dev)
            parity_train.append({"scene_id": sid, "prediction_equal": output_digest(before) == output_digest(after)})
            rpn_parity.append({"scene_id": sid, "split": "train", "rpn_equal": array_hash(rpn_before_boxes) == array_hash(rpn_after_boxes) and array_hash(rpn_before_scores) == array_hash(rpn_after_scores)})
        if len(train_rows) < 20 or len(set(train_y)) != 2: raise RuntimeError(f"insufficient C1 train pairs: {len(train_rows)}")
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1., class_weight="balanced", max_iter=1000, random_state=20260919))
        clf.fit(np.asarray(train_x), np.asarray(train_y))
        val_rows, parity_val, scene_stats = [], [], {}
        val_index = {sid: i for i, sid in enumerate(val.ids)}
        for sid in val_ids:
            i, item = val_index[sid], val[val_index[sid]]; before = normal_predict(model, item, dev)
            post_boxes, post_scores = read_rpn(model, item, dev); boxes, scores, layouts = dense_raw_rpn(model, item, dev)
            dense_layout.extend({"scene_id": sid, "split": "val", "level": level, **layout} for level, layout in enumerate(layouts))
            fmap = encode_p2(encoder, item["drone_images"].to(dev)); perm_fmap = encode_p2(encoder, val[(i + 1) % len(val)]["drone_images"].to(dev))
            masks, _, gt = gt_from_item(item); stat = {"scene_id": sid, "correct": 0, "xy": 0, "perm": 0, "pairs": 0}
            for gi, b0row in target_by_scene[sid].items():
                replay_iou = float(box_iou_xyxy(gt[gi:gi + 1], post_boxes).max()) if len(post_boxes) else 0.
                if abs(replay_iou - float(b0row["max_rpn_box_iou"])) > 1e-5 or replay_iou >= .5:
                    raise RuntimeError(f"B0 post-NMS lineage drift {sid}/{gi}")
                pair = pair_from_dense(boxes, scores, gt, gi)
                if pair is None:
                    placeholder = {"scene_id": sid, "gt_index": gi, "pos_candidate": None, "neg_candidate": None,
                                   "pos_box_xyxy": None, "neg_box_xyxy": None, "pos_raw_objectness": None, "neg_raw_objectness": None,
                                   "positive_label": 1, "negative_label": 0, "pair_identity_sha256": None}
                    val_rows.append(empty_val_row(placeholder, b0row, replay_iou, False, None, None)); continue
                pos, neg, pos_iou, gap = pair; sha, rec = identity(sid, gi, pos, neg, boxes, scores)
                out = empty_val_row(rec | {"pair_identity_sha256": sha}, b0row, replay_iou, True, pos_iou, gap)
                if gap > .02:
                    out["evaluation_status"] = "score_match_rejected"; val_rows.append(out); continue
                desc = {}
                for name, fmap_i, shift in (("correct", fmap, 0.), ("xy", fmap, 80.), ("perm", perm_fmap, 0.)):
                    p, ps = uav_descriptor(fmap_i, boxes[pos], item, dev, shift); n, ns = uav_descriptor(fmap_i, boxes[neg], item, dev, shift)
                    desc[name] = (p, n)
                    out[f"{name}_pos_min_support"], out[f"{name}_pos_valid_points"] = ps["min_support"], ps["valid_points"]
                    out[f"{name}_neg_min_support"], out[f"{name}_neg_valid_points"] = ns["min_support"], ns["valid_points"]
                common = all(p is not None and n is not None for p, n in desc.values()); out["common_valid"] = common
                if common:
                    for name, (p, n) in desc.items():
                        ps, ns = float(clf.predict_proba(p[None])[:, 1][0]), float(clf.predict_proba(n[None])[:, 1][0])
                        out[f"{name}_pos_readout_score"], out[f"{name}_neg_readout_score"] = ps, ns; out[f"{name}_win"] = int(ps > ns); stat[name] += out[f"{name}_win"]
                    stat["pairs"] += 1; out["evaluation_status"] = "included"
                else: out["evaluation_status"] = "descriptor_invalid"
                val_rows.append(out)
            post_after_boxes, post_after_scores = read_rpn(model, item, dev); after = normal_predict(model, item, dev)
            parity_val.append({"scene_id": sid, "prediction_equal": output_digest(before) == output_digest(after)})
            rpn_parity.append({"scene_id": sid, "split": "val", "rpn_equal": array_hash(post_boxes) == array_hash(post_after_boxes) and array_hash(post_scores) == array_hash(post_after_scores)})
            scene_stats[sid] = stat
    detector_hash_after, encoder_hash_after = model_param_hash(model), model_param_hash(encoder)
    train_complete, val_complete = train_n == len(train), val_n == len(target_by_scene); complete = train_complete and val_complete
    ledger = {key: sum(r["evaluation_status"] == key for r in val_rows) for key in ("included", "descriptor_invalid", "score_match_rejected", "raw_geometry_unavailable")}
    if sum(ledger.values()) != len(val_rows) or len(val_rows) != sum(len(target_by_scene[s]) for s in val_ids): raise RuntimeError("C1 ledger incomplete")
    per_scene = [scene_stats.get(s, {"scene_id": s, "correct": 0, "xy": 0, "perm": 0, "pairs": 0}) for s in val.ids]
    total, wins = sum(x["pairs"] for x in per_scene), {k: sum(x[k] for x in per_scene) for k in ("correct", "xy", "perm")}; bootstrap = boot(per_scene, a.bootstrap_samples, a.bootstrap_seed)
    coverage = total / TARGETS if val_complete else None; invalid = []
    if not train_complete: invalid.append("train_subset_incomplete")
    if not val_complete: invalid.append("validation_target_coverage_incomplete")
    if not all(x["prediction_equal"] for x in parity_train + parity_val): invalid.append("detector_parity_failed")
    if not all(x["rpn_equal"] for x in rpn_parity): invalid.append("post_nms_rpn_parity_failed")
    if detector_hash_before != detector_hash_after: invalid.append("detector_hash_changed")
    if encoder_hash_before != encoder_hash_after: invalid.append("encoder_hash_changed")
    condition_identity_fixed = all(r.get("pair_identity_sha256") is None or all(r.get(f"{c}_pair_identity_sha256") == r.get("pair_identity_sha256") for c in CONDITIONS) for r in val_rows)
    if not condition_identity_fixed: invalid.append("condition_pair_identity_failed")
    decision = "INVALID"
    if complete and not invalid:
        if total and coverage >= .80 and wins["correct"] / total >= .60 and bootstrap["correct_minus_xy_ci95"][0] > .05 and bootstrap["correct_minus_permuted_ci95"][0] > .05: decision = "FROZEN_UAV_RPN_PRESELECTION_SIGNAL"
        else: decision = "NO_FROZEN_SIGNAL"
    raw_available = len(val_rows) - ledger["raw_geometry_unavailable"]
    score_match_eligible = ledger["included"] + ledger["descriptor_invalid"]
    summary = {"experiment_id": a.experiment_id, "arm": "C1-Frozen-UAV-RPN-Preselection-Readout", "status": "probe_only_no_detector_training",
        "checkpoint": str(Path(a.checkpoint).resolve()), "model_config_source": source,
        "b0_lineage": {"dir": str(a.b0_dir.resolve()), "targets_expected": TARGETS, "parent_final_failure_small": 227},
        "data": {"train_subset_ratio": TRAIN_RATIO, "train_seed": TRAIN_SEED, "train_scenes_expected": 128, "train_scenes_processed": train_n, "train_complete": train_complete, "val_scene_universe": 259, "target_scenes_expected": len(target_by_scene), "target_scenes_processed": val_n, "val_pairs_expected": TARGETS, "val_pairs_common_valid": total, "val_complete": val_complete, "complete": complete},
        "contract": {"candidate": "all dense decoded RPN anchors before score_thr/nms_pre/NMS", "descriptor": "frozen SAM2 finest FPN; 3x3 decoded raw-RPN box grid; z0/360m/W2C/fp32; >=2 views every point", "conditions": ["correct", "XY+80m", "scene_permuted"], "linear_readout": "StandardScaler+LogisticRegression(C=1,class_weight=balanced,max_iter=1000,random_state=20260919)"},
        "invariants": {"train_val_scene_disjoint": not bool(set(train.ids) & set(val.ids)), "detector_parity_bitwise": all(x["prediction_equal"] for x in parity_train + parity_val), "post_nms_rpn_parity_bitwise": all(x["rpn_equal"] for x in rpn_parity), "dense_anchor_cardinality": all(x["prior_count"] == x["delta_count"] == x["score_count"] == x["decoded_count"] for x in dense_layout), "condition_pair_identity_fixed": condition_identity_fixed, "detector_hash_unchanged": detector_hash_before == detector_hash_after, "encoder_hash_unchanged": encoder_hash_before == encoder_hash_after, "no_detector_uav_or_sham_intervention": True, "b0_target_lineage_fail_closed": True},
        "metrics": {"common_pair_coverage": coverage, "wins": wins, "win_rates": {k: wins[k] / total if total else None for k in wins}, "train_pairs": len(train_rows), "train_class_balance": {"positive": len(train_rows), "negative": len(train_rows)}, "raw_geometry_available": {"count": raw_available, "denominator": len(val_rows), "fraction": raw_available / len(val_rows) if val_rows else None}, "score_match_eligible": {"count": score_match_eligible, "denominator": len(val_rows), "fraction": score_match_eligible / len(val_rows) if val_rows else None}, "val_ledger": ledger, "mean_abs_raw_objectness_gap": float(np.mean([float(r["score_gap"]) for r in val_rows if r["score_gap"] not in (None, "")])) if val_rows else None},
        "bootstrap": bootstrap, "invalid_reasons": invalid, "decision": decision,
        "decision_rule": "FROZEN_UAV_RPN_PRESELECTION_SIGNAL iff complete 128-train/107-target validation, coverage>=.80, correct win>=.60, and both correct-sham bootstrap CI lower >+.05; else NO_FROZEN_SIGNAL; failed lineage/split/parity/hash/completeness is INVALID.",
        "interpretation": "A frozen dense-RPN readout probes preselection targetness only; it is neither a proposal-recall intervention nor AP evidence.", "per_scene": per_scene, "parity": {"train": parity_train, "val": parity_val, "post_nms_rpn": rpn_parity}, "dense_layout": dense_layout}
    with (a.output_dir / "val_pairs.csv").open("w", newline="") as f: w = csv.DictWriter(f, fieldnames=VAL_FIELDS); w.writeheader(); w.writerows(val_rows)
    train_fields = sorted({k for row in train_rows for k in row})
    with (a.output_dir / "train_pairs.csv").open("w", newline="") as f: w = csv.DictWriter(f, fieldnames=train_fields); w.writeheader(); w.writerows(train_rows)
    with (a.output_dir / "dense_layout.csv").open("w", newline="") as f: w = csv.DictWriter(f, fieldnames=["scene_id", "split", "level", "prior_count", "delta_count", "score_count", "decoded_count"]); w.writeheader(); w.writerows(dense_layout)
    (a.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n"); (a.output_dir / "bootstrap.json").write_text(json.dumps(bootstrap, indent=2) + "\n")
    (a.output_dir / "run_manifest.json").write_text(json.dumps({k: summary[k] for k in ("experiment_id", "arm", "status", "checkpoint", "b0_lineage", "data", "contract", "invariants", "decision")}, indent=2) + "\n")
    print(json.dumps({"output": str(a.output_dir / "summary.json"), "decision": decision, "metrics": summary["metrics"], "invariants": summary["invariants"]}, indent=2))


if __name__ == "__main__": main()
