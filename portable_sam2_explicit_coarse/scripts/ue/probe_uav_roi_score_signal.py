#!/usr/bin/env python3
"""B3 frozen UAV ROI-score signal probe; detector and UAV encoder stay frozen."""
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
import torch.nn.functional as F
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
from probe_small_postrpn_mechanism import bbox_readout
from probe_small_rpn_bottleneck import box_iou_xyxy, gt_from_item, normal_predict

TRAIN_RATIO = 0.1797752808988764
TRAIN_SEED = 44
CONDITIONS = ("correct", "xy", "perm")
VAL_PAIR_FIELDS = [
    "scene_id", "gt_index", "pair_identity_sha256", "pos_candidate", "neg_candidate",
    "pos_box_xyxy", "neg_box_xyxy", "pos_raw_fg_score", "neg_raw_fg_score",
    "positive_label", "negative_label", "b2_category", "b2_pstar_index",
    "b2_pstar_pre_rpn_iou", "b2_pstar_decoded_iou", "b3_pos_max_decoded_iou",
    "score_gap", "score_match", "common_valid", "evaluation_status",
] + [field for condition in CONDITIONS for field in (
    f"{condition}_pos_min_support", f"{condition}_pos_valid_points",
    f"{condition}_neg_min_support", f"{condition}_neg_valid_points",
    f"{condition}_pos_readout_score", f"{condition}_neg_readout_score", f"{condition}_win",
)]


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--b2-dir", type=Path, required=True)
    p.add_argument("--data-root", default="/data1/wangcheng/dataset/ue_coco_with_uav")
    p.add_argument("--device", default="cuda")
    p.add_argument("--sam2-ckpt", default=os.environ["SAM2_CKPT"])
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-train-scenes", type=int, default=0)
    p.add_argument("--max-val-scenes", type=int, default=0)
    p.add_argument("--bootstrap-samples", type=int, default=10_000)
    p.add_argument("--bootstrap-seed", type=int, default=20260918)
    p.add_argument("--experiment-id", default="MHBEV-DIAG-20260915-B3")
    return p.parse_args()


def model_param_hash(model) -> str:
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode()); h.update(str(tuple(value.shape)).encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def candidate_pair(decoded: np.ndarray, fg: np.ndarray, gt_boxes: np.ndarray, gi: int):
    """Positive aligned decoded candidate + score-matched true-background negative."""
    iou = box_iou_xyxy(gt_boxes, decoded)
    pos = int(iou[gi].argmax())
    if iou[gi, pos] < .5:
        return None
    bg = iou.max(axis=0) < .1
    if not bg.any():
        return None
    candidates = np.flatnonzero(bg)
    neg = int(candidates[np.abs(fg[candidates] - fg[pos]).argmin()])
    return pos, neg, float(iou[gi, pos]), float(abs(fg[pos] - fg[neg]))


def pair_identity(scene_id: str, gt_index: int, pos: int, neg: int, decoded: np.ndarray,
                  fg: np.ndarray) -> tuple[str, dict[str, Any]]:
    """Immutable satellite-side identity shared by all UAV/sham conditions."""
    record = {
        "scene_id": scene_id, "gt_index": int(gt_index), "pos_candidate": int(pos),
        "neg_candidate": int(neg), "pos_box_xyxy": [float(x) for x in decoded[pos]],
        "neg_box_xyxy": [float(x) for x in decoded[neg]], "pos_raw_fg_score": float(fg[pos]),
        "neg_raw_fg_score": float(fg[neg]), "positive_label": 1, "negative_label": 0,
    }
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest(), record


def close(actual: float, expected: str, name: str, tol: float = 1e-5):
    if abs(actual - float(expected)) > tol:
        raise RuntimeError(f"B2 lineage drift in {name}: {actual} != {expected}")


def check_b2_target(row: dict[str, str], proposal_boxes: np.ndarray, decoded: np.ndarray,
                    fg: np.ndarray, gt_boxes: np.ndarray, gi: int, pos: int, pos_iou: float):
    """Fail closed on B2's recorded target classification and P* lineage.

    B2 P* is an RPN-stage record; B3's positive is deliberately the best decoded box.
    Both are preserved and checked, rather than silently treating them as the same candidate.
    """
    if row["category"] != "roi_score_rejected" or row["b1_category"] != "post_rpn_box_lost_or_mislocalized":
        raise RuntimeError("B2 target category lineage drift")
    if float(row["score_thr"]) != .05:
        raise RuntimeError("B2 score threshold contract drift")
    pstar = int(row["pstar_index"])
    if not 0 <= pstar < len(proposal_boxes):
        raise RuntimeError("B2 P* index out of current proposal range")
    proposal_iou = float(box_iou_xyxy(gt_boxes[gi:gi + 1], proposal_boxes)[0, pstar])
    decoded_iou = float(box_iou_xyxy(gt_boxes[gi:gi + 1], decoded)[0, pstar])
    close(proposal_iou, row["pstar_pre_rpn_iou"], "pstar_pre_rpn_iou")
    close(decoded_iou, row["pstar_decoded_iou"], "pstar_decoded_iou")
    close(float(fg[pstar]), row["pstar_raw_fg_score"], "pstar_raw_fg_score")
    close(pos_iou, row["max_decoded_iou"], "max_decoded_iou")
    if float(fg[pos]) > float(row["score_thr"]) + 1e-7:
        raise RuntimeError("B3 maximum-decoded positive no longer ROI-score rejected")
    if float(row["aligned_best_fg_score"]) > float(row["score_thr"]) + 1e-7:
        raise RuntimeError("B2 ROI-score predicate malformed")
    return pstar, proposal_iou, decoded_iou


def points_for_box(box: np.ndarray, center: torch.Tensor, device: torch.device, shift_x_m: float = 0.) -> torch.Tensor:
    x1, y1, x2, y2 = [float(x) for x in box]
    t = torch.tensor((1 / 6, 3 / 6, 5 / 6), device=device, dtype=torch.float32)
    vv, uu = torch.meshgrid(y1 + (y2 - y1) * t, x1 + (x2 - x1) * t, indexing="ij")
    u, v = uu.reshape(-1), vv.reshape(-1)
    return torch.stack((center[0] + (0.5 - v / 1024.) * 36000. + shift_x_m * 100.,
                        center[1] + (u / 1024. - .5) * 36000.,
                        torch.zeros_like(u), torch.ones_like(u)), dim=-1)


def uav_descriptor(fmap: torch.Tensor, box: np.ndarray, item: dict, device: torch.device, shift_x_m: float = 0.):
    """Return 256-D frozen multi-view 3x3 descriptor or None if any point lacks 2 views."""
    points = points_for_box(box, item["scene_center"].to(device), device, shift_x_m)
    K, E = item["intrinsics"].to(device), item["extrinsics"].to(device)
    with torch.autocast(device_type=device.type, enabled=False):
        camera = torch.einsum("vij,qj->vqi", E.float(), points.float())
        pixel = torch.einsum("vij,vqj->vqi", K.float(), camera[..., :3])
        uv = pixel[..., :2] / pixel[..., 2:3].clamp_min(1e-6)
    valid = ((camera[..., 2] > 1e-4) & (uv[..., 0] >= 0) & (uv[..., 0] < 512) &
             (uv[..., 1] >= 0) & (uv[..., 1] < 512))
    support = valid.sum(0)
    if not bool((support >= 2).all()):
        return None, {"min_support": int(support.min().cpu()), "valid_points": int((support >= 2).sum().cpu())}
    grid = ((uv / 512.) * 2. - 1.).to(fmap.dtype)
    sample = F.grid_sample(fmap, grid[:, None], mode="bilinear", padding_mode="zeros", align_corners=False)
    sample = sample[:, :, 0].transpose(1, 2)  # [V,Q,C]
    fused = (sample * valid[..., None].to(sample.dtype)).sum(0) / support[:, None].to(sample.dtype)
    return fused.mean(0).detach().float().cpu().numpy(), {"min_support": int(support.min().cpu()), "valid_points": 9}


def bootstrap(per_scene, samples, seed):
    values = np.asarray([[x["correct"], x["xy"], x["perm"], x["pairs"]] for x in per_scene], dtype=np.int64)
    rng = np.random.default_rng(seed); draws = rng.integers(0, len(values), size=(samples, len(values)))
    sums = values[draws].sum(axis=1); valid = sums[:, 3] > 0
    if not bool(valid.any()):
        return {"samples_requested": samples, "samples_valid": 0, "seed": seed,
                "correct_minus_xy_ci95": [None, None],
                "correct_minus_permuted_ci95": [None, None]}
    correct, xy, perm = (sums[valid, i] / sums[valid, 3] for i in range(3))
    return {"samples_requested": samples, "samples_valid": int(valid.sum()), "seed": seed,
            "correct_minus_xy_ci95": [float(x) for x in np.percentile(correct - xy, [2.5, 97.5])],
            "correct_minus_permuted_ci95": [float(x) for x in np.percentile(correct - perm, [2.5, 97.5])]}


def main():
    a = arguments(); dev = torch.device("cuda:0" if a.device == "cuda" else a.device)
    b2 = json.loads((a.b2_dir / "summary.json").read_text()); rows_b2 = list(csv.DictReader((a.b2_dir / "targets.csv").open()))
    if not (b2["decision"] == "ROI_SCORE" and b2["data"]["complete"] and all(bool(v) for v in b2["invariants"].values()) and
            b2["data"]["targets_processed"] == 101 and b2["live_bbox_contract"] == {"bbox_head_type": "Shared2FCBBoxHead", "bbox_coder_type": "DeltaXYWHBBoxCoder"}):
        raise RuntimeError("B2 lineage contract failed")
    if Path(b2["checkpoint"]).resolve() != Path(a.checkpoint).resolve(): raise RuntimeError("checkpoint differs from B2")
    targets = [r for r in rows_b2 if r["category"] == "roi_score_rejected"]
    if len(targets) != 87: raise RuntimeError(f"expected B2's 87 ROI-score targets, got {len(targets)}")
    target_by_scene = defaultdict(dict)
    for r in targets: target_by_scene[r["scene_id"]][int(r["gt_index"])] = r
    # B2's 84/101 is the complete post-RPN parent population.  Its 87
    # ROI-score-rejected targets are carried by 74 (not 84) distinct scenes.
    if sum(len(scene_targets) for scene_targets in target_by_scene.values()) != 87 or len(target_by_scene) != 74:
        raise RuntimeError("B2 ROI-score target identity is not the expected 87 targets / 74 scenes")
    if not Path(a.sam2_ckpt).is_file(): raise FileNotFoundError(a.sam2_ckpt)
    checkpoint = _load_checkpoint(Path(a.checkpoint)); snap = _snapshot(checkpoint)
    la = SimpleNamespace(config=None, sam2_ckpt=a.sam2_ckpt, sam2_repo=None); _resolve_sam2_repo(la, snap)
    cfg, source = _resolve_model_config(la, snap); model = _register_and_build(cfg).to(dev).eval()
    report = _load_model_state(model, _select_state_dict(checkpoint, "model"), False)
    if report["missing_keys"] or report["unexpected_keys"]: raise RuntimeError(f"strict model load failed: {report}")
    if type(model.roi_head.bbox_head).__name__ != "Shared2FCBBoxHead" or type(model.roi_head.bbox_head.bbox_coder).__name__ != "DeltaXYWHBBoxCoder":
        raise RuntimeError("B3 requires B2 live bbox contract")
    encoder = make_uav_encoder(a, dev); encoder_hash_before = model_param_hash(encoder)
    train = UEDualStreamDataset(a.data_root, "train", image_size=(1024, 1024), train_subset=TRAIN_RATIO, seed=TRAIN_SEED, load_uav=True)
    val = UEDualStreamDataset(a.data_root, "val", image_size=(1024, 1024), load_uav=True)
    if len(train) != 128 or len(val) != 259 or set(train.ids) & set(val.ids): raise RuntimeError("B3 train/val split contract failed")
    train_n = len(train) if a.max_train_scenes == 0 else min(a.max_train_scenes, len(train)); val_ids = [s for s in val.ids if s in target_by_scene]
    val_n = len(val_ids) if a.max_val_scenes == 0 else min(a.max_val_scenes, len(val_ids)); val_ids = val_ids[:val_n]
    a.output_dir.mkdir(parents=True, exist_ok=True)
    train_x, train_y, train_rows, parity_train = [], [], [], []
    with torch.inference_mode():
        for idx in range(train_n):
            item, sid = train[idx], train.ids[idx]; before = normal_predict(model, item, dev)
            _, _, decoded, fg = bbox_readout(model, item, dev)
            fmap = encode_p2(encoder, item["drone_images"].to(dev)); gt_masks, _, gt_boxes = gt_from_item(item)
            for gi, mask in enumerate(gt_masks):
                if int(mask.sum()) >= 1024: continue
                pair = candidate_pair(decoded, fg, gt_boxes, gi)
                if pair is None: continue
                pos, neg, pos_iou, gap = pair
                identity_sha, identity = pair_identity(sid, gi, pos, neg, decoded, fg)
                dp, sp = uav_descriptor(fmap, decoded[pos], item, dev); dn, sn = uav_descriptor(fmap, decoded[neg], item, dev)
                if dp is None or dn is None: continue
                train_x.extend((dp, dn)); train_y.extend((1, 0)); train_rows.append({**identity, "pair_identity_sha256": identity_sha,
                    "score_gap":gap,"pos_iou":pos_iou,"correct_pos_min_support":sp["min_support"],"correct_pos_valid_points":sp["valid_points"],
                    "correct_neg_min_support":sn["min_support"],"correct_neg_valid_points":sn["valid_points"]})
            after = normal_predict(model, item, dev)
            parity_train.append({"scene_id": sid, "prediction_equal": output_digest(before) == output_digest(after)})
        if len(set(train_y)) != 2 or len(train_rows) < 20: raise RuntimeError(f"insufficient frozen-readout train pairs: {len(train_rows)}")
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1., class_weight="balanced", max_iter=1000, random_state=20260918))
        clf.fit(np.asarray(train_x), np.asarray(train_y))
        val_rows, parity_val, per_processed = [], [], {}
        val_index = {sid:i for i,sid in enumerate(val.ids)}
        for sid in val_ids:
            i, item = val_index[sid], val[val_index[sid]]; before = normal_predict(model, item, dev)
            proposal_boxes, _, decoded, fg = bbox_readout(model, item, dev)
            fmap = encode_p2(encoder, item["drone_images"].to(dev)); next_item = val[(i+1)%len(val)]; perm_fmap = encode_p2(encoder, next_item["drone_images"].to(dev))
            gt_masks, _, gt_boxes = gt_from_item(item); stat={"scene_id":sid,"correct":0,"xy":0,"perm":0,"pairs":0}
            for gi, b2_row in target_by_scene[sid].items():
                pair=candidate_pair(decoded,fg,gt_boxes,gi)
                if pair is None: raise RuntimeError(f"B2 candidate drift {sid}/{gi}")
                pos,neg,pos_iou,gap=pair
                pstar, pstar_rpn_iou, pstar_decoded_iou = check_b2_target(b2_row, proposal_boxes, decoded, fg, gt_boxes, gi, pos, pos_iou)
                identity_sha, identity = pair_identity(sid, gi, pos, neg, decoded, fg)
                out={**identity, "pair_identity_sha256":identity_sha,"b2_category":b2_row["category"],"b2_pstar_index":pstar,
                     "b2_pstar_pre_rpn_iou":pstar_rpn_iou,"b2_pstar_decoded_iou":pstar_decoded_iou,
                     "b3_pos_max_decoded_iou":pos_iou,"score_gap":gap,"score_match":gap<=.02,
                     "common_valid":False,"evaluation_status":"score_match_rejected" if gap>.02 else "descriptor_pending"}
                for name in CONDITIONS:
                    for suffix in ("pos_min_support", "pos_valid_points", "neg_min_support", "neg_valid_points",
                                   "pos_readout_score", "neg_readout_score", "win"):
                        out[f"{name}_{suffix}"] = None
                if gap > .02:
                    val_rows.append(out)
                    continue
                desc, supports = {}, {}
                for name,fm,shift in (("correct",fmap,0.),("xy",fmap,80.),("perm",perm_fmap,0.)):
                    p,ps=uav_descriptor(fm,decoded[pos],item,dev,shift); n,ns=uav_descriptor(fm,decoded[neg],item,dev,shift)
                    desc[name]=(p,n); supports[name]=(ps,ns)
                    out[f"{name}_pos_min_support"], out[f"{name}_pos_valid_points"] = ps["min_support"], ps["valid_points"]
                    out[f"{name}_neg_min_support"], out[f"{name}_neg_valid_points"] = ns["min_support"], ns["valid_points"]
                common=all(p is not None and n is not None for p,n in desc.values())
                out["common_valid"]=common
                if common:
                    for name,(p,nv) in desc.items():
                        pscore=float(clf.predict_proba(p[None])[:,1][0]); nscore=float(clf.predict_proba(nv[None])[:,1][0])
                        win=int(pscore > nscore); out[f"{name}_pos_readout_score"]=pscore; out[f"{name}_neg_readout_score"]=nscore; out[f"{name}_win"]=win; stat[name]+=win
                    stat["pairs"]+=1
                    out["evaluation_status"]="included"
                else:
                    out["evaluation_status"]="descriptor_invalid"
                val_rows.append(out)
            per_processed[sid]=stat
            after = normal_predict(model, item, dev)
            parity_val.append({"scene_id":sid,"prediction_equal":output_digest(before)==output_digest(after)})
    encoder_hash_after=model_param_hash(encoder)
    train_complete = train_n == len(train)
    val_complete = val_n == len(target_by_scene)
    complete = train_complete and val_complete
    per_scene=[per_processed.get(s,{"scene_id":s,"correct":0,"xy":0,"perm":0,"pairs":0}) for s in val.ids]
    total=sum(x["pairs"] for x in per_scene); wins={k:sum(x[k] for x in per_scene) for k in ("correct","xy","perm")}
    boot=bootstrap(per_scene,a.bootstrap_samples,a.bootstrap_seed)
    coverage=total/87 if val_complete else None
    status_counts = {name: sum(r["evaluation_status"] == name for r in val_rows)
                     for name in ("included", "score_match_rejected", "descriptor_invalid")}
    if sum(status_counts.values()) != len(val_rows):
        raise RuntimeError("B3 evaluation ledger is not a complete partition")
    pair_hashes = [r["pair_identity_sha256"] for r in val_rows]
    condition_identity_fixed = len(pair_hashes) == len(set((r["scene_id"], r["gt_index"], r["pair_identity_sha256"]) for r in val_rows))
    if not condition_identity_fixed:
        raise RuntimeError("B3 pair identity collision or mutation")
    invalid_reasons = []
    if not train_complete: invalid_reasons.append("train_subset_incomplete")
    if not val_complete: invalid_reasons.append("validation_target_coverage_incomplete")
    if not all(x["prediction_equal"] for x in parity_train+parity_val): invalid_reasons.append("detector_parity_failed")
    if encoder_hash_before != encoder_hash_after: invalid_reasons.append("encoder_hash_changed")
    decision="INVALID"
    if complete and not invalid_reasons:
        if total and coverage>=.80 and wins["correct"]/total>=.60 and boot["correct_minus_xy_ci95"][0]>.05 and boot["correct_minus_permuted_ci95"][0]>.05: decision="FROZEN_UAV_ROI_SCORE_SIGNAL"
        else: decision="NO_FROZEN_SIGNAL"
    summary={"experiment_id":a.experiment_id,"arm":"B3-Frozen-UAV-ROIScore-Readout","status":"probe_only_no_detector_training","checkpoint":str(Path(a.checkpoint).resolve()),"model_config_source":source,
             "b2_lineage":{"dir":str(a.b2_dir.resolve()),"decision":b2["decision"],"targets_expected":87},
             "data":{"train_subset_ratio":TRAIN_RATIO,"train_seed":TRAIN_SEED,"train_scenes_expected":128,"train_scenes_processed":train_n,"train_complete":train_complete,"val_scene_universe":259,"target_scenes_expected":len(target_by_scene),"target_scenes_processed":val_n,"val_pairs_expected":87,"val_pairs_common_valid":total,"val_complete":val_complete,"complete":complete},
             "contract":{"descriptor":"frozen SAM2 finest FPN; 3x3 decoded ROI grid; z0/360m/W2C/fp32; >=2 views every point","conditions":["correct","XY+80m","scene_permuted"],"linear_readout":"StandardScaler+LogisticRegression(C=1,class_weight=balanced,max_iter=1000,random_state=20260918)"},
             "invariants":{"train_val_scene_disjoint":not bool(set(train.ids)&set(val.ids)),"detector_parity_bitwise":all(x["prediction_equal"] for x in parity_train+parity_val),"encoder_hash_unchanged":encoder_hash_before==encoder_hash_after,"no_detector_uav_or_sham_intervention":True,"condition_pair_identity_fixed":condition_identity_fixed,"b2_target_lineage_fail_closed":True},
             "metrics":{"common_pair_coverage":coverage,"wins":wins,"win_rates":{k:wins[k]/total if total else None for k in wins},"mean_abs_satellite_score_gap":float(np.mean([r["score_gap"] for r in val_rows if r["common_valid"]])) if total else None,"train_pairs":len(train_rows,
             ),"val_ledger":{"targets_seen":len(val_rows),"targets_expected_processed":sum(len(target_by_scene[s]) for s in val_ids),"by_status":status_counts,"score_gap_all":{"min":float(np.min([r["score_gap"] for r in val_rows])) if val_rows else None,"median":float(np.median([r["score_gap"] for r in val_rows])) if val_rows else None,"max":float(np.max([r["score_gap"] for r in val_rows])) if val_rows else None}}},"bootstrap":boot,"invalid_reasons":invalid_reasons,"decision":decision,
             "decision_rule":"FROZEN_UAV_ROI_SCORE_SIGNAL iff complete 128-train/87-target validation, coverage>=.80, correct win>=.60, and both correct-sham bootstrap CI lower >+.05; else NO_FROZEN_SIGNAL; any incomplete split, failed lineage/split/parity/hash is INVALID.","interpretation":"A frozen linear readout probes conditional UAV targetness information only, not detector AP or causal end-to-end gain.","per_scene":per_scene,"parity":{"train":parity_train,"val":parity_val}}
    with (a.output_dir/"val_pairs.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=VAL_PAIR_FIELDS); w.writeheader(); w.writerows(val_rows)
    with (a.output_dir/"train_pairs.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=sorted({key for row in train_rows for key in row})); w.writeheader(); w.writerows(train_rows)
    (a.output_dir/"bootstrap.json").write_text(json.dumps(boot,indent=2)+"\n"); (a.output_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (a.output_dir/"run_manifest.json").write_text(json.dumps({k:summary[k] for k in ("experiment_id","arm","status","checkpoint","b2_lineage","data","contract","invariants","decision")},indent=2)+"\n")
    print(json.dumps({"output":str(a.output_dir/"summary.json"),"decision":decision,"metrics":summary["metrics"],"invariants":summary["invariants"]},indent=2))

if __name__ == "__main__": main()
