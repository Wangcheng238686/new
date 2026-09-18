#!/usr/bin/env python3
"""D1: read-only attribution of B0 RPN-miss targets to a deployed RPN stage."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("SAM2_CKPT", "/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from data.ue_dual_stream_dataset import UEDualStreamDataset
from inference.infer_from_checkpoint import (_load_checkpoint, _load_model_state, _register_and_build,
    _resolve_model_config, _resolve_sam2_repo, _select_state_dict, _snapshot)
from probe_small_postrpn_bottleneck import output_digest
from probe_small_rpn_bottleneck import (array_hash, box_iou_xyxy, gt_from_item, normal_predict, prepare, read_rpn)

TARGETS = 107
STAGES = ("raw_geometry_unavailable", "score_nms_pre_rejected", "nms_suppressed", "max_per_img_truncated")


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--b0-dir", type=Path, required=True)
    p.add_argument("--data-root", default="/data1/wangcheng/dataset/ue_coco_with_uav")
    p.add_argument("--device", default="cuda")
    p.add_argument("--sam2-ckpt", default=os.environ["SAM2_CKPT"])
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-target-scenes", type=int, default=0)
    p.add_argument("--bootstrap-samples", type=int, default=10_000)
    p.add_argument("--bootstrap-seed", type=int, default=20260920)
    p.add_argument("--experiment-id", default="MHBEV-DIAG-20260917-D1")
    return p.parse_args()


def instance_arrays(instance):
    return (instance.bboxes.detach().float().cpu().numpy(), instance.scores.detach().float().cpu().numpy())


def stage_hash(boxes, scores):
    def typed(value):
        value = np.ascontiguousarray(value)
        h = hashlib.sha256()
        h.update(str(value.dtype).encode()); h.update(repr(value.shape).encode()); h.update(value.tobytes())
        return h.hexdigest()
    return {"boxes": typed(boxes), "scores": typed(scores)}


def tensor_hashes(values):
    return [stage_hash(x.detach().float().cpu().numpy(), np.empty((0,), dtype=np.float32))["boxes"] for x in values]


def max_iou(gt, boxes):
    return float(box_iou_xyxy(gt[None], boxes).max()) if len(boxes) else 0.


def dense_raw(head, cls_scores, bbox_preds, img_shape, device):
    """All anchors before score filtering/nms_pre/NMS, with per-level cardinality."""
    sizes = [x.shape[-2:] for x in cls_scores]
    priors = head.prior_generator.grid_priors(sizes, dtype=cls_scores[0].dtype, device=device)
    boxes, scores, layout = [], [], []
    for level, (cls, delta, prior) in enumerate(zip(cls_scores, bbox_preds, priors)):
        logits = cls[0].permute(1, 2, 0).reshape(-1, head.cls_out_channels)
        score = logits.sigmoid().reshape(-1) if head.use_sigmoid_cls else logits.softmax(-1)[:, :-1].reshape(-1)
        delta = delta[0].permute(1, 2, 0).reshape(-1, head.bbox_coder.encode_size)
        decoded = head.bbox_coder.decode(prior, delta, max_shape=img_shape)
        if not (len(prior) == len(delta) == len(score) == len(decoded)):
            raise RuntimeError(f"level {level} dense cardinality mismatch")
        boxes.append(decoded.detach().float().cpu().numpy()); scores.append(score.detach().float().cpu().numpy())
        layout.append({"level": level, "prior_count": len(prior), "delta_count": len(delta), "score_count": len(score), "decoded_count": len(decoded)})
    return np.concatenate(boxes), np.concatenate(scores), layout, tensor_hashes(priors)


def deployed_pre_nms(head, cls_scores, bbox_preds, img_shape, device):
    """Exactly replay RPN's per-level sigmoid and ``nms_pre`` selection.

    RPNHead does not expose a no-NMS result (it asserts ``with_nms=True``),
    so this is the pre-postprocess state reconstructed from its implementation.
    """
    priors = head.prior_generator.grid_priors(
        [x.shape[-2:] for x in cls_scores], dtype=cls_scores[0].dtype,
        device=device)
    nms_pre = head.test_cfg.get("nms_pre", -1)
    boxes, scores = [], []
    for cls, delta, prior in zip(cls_scores, bbox_preds, priors):
        logits = cls[0].permute(1, 2, 0).reshape(-1, head.cls_out_channels)
        score = logits.sigmoid() if head.use_sigmoid_cls else logits.softmax(-1)[:, :-1]
        # RPNHead uses squeeze here; this experiment's one-foreground-class
        # contract therefore yields a one-dimensional ranking vector.
        score = score.squeeze()
        delta = delta[0].permute(1, 2, 0).reshape(-1, head.bbox_coder.encode_size)
        if 0 < nms_pre < score.shape[0]:
            ranked, inds = score.sort(descending=True)
            score, delta, prior = ranked[:nms_pre], delta[inds[:nms_pre]], prior[inds[:nms_pre]]
        boxes.append(head.bbox_coder.decode(prior, delta, max_shape=img_shape))
        scores.append(score)
    boxes, scores = torch.cat(boxes), torch.cat(scores)
    min_size = head.test_cfg.get("min_bbox_size", -1)
    if min_size >= 0:
        valid = ((boxes[:, 2] - boxes[:, 0]) > min_size) & ((boxes[:, 3] - boxes[:, 1]) > min_size)
        boxes, scores = boxes[valid], scores[valid]
    return boxes.detach().float().cpu().numpy(), scores.detach().float().cpu().numpy()


def assert_live_contract(live):
    nms = live.get("nms", {})
    resolved = {"nms_pre": int(live.get("nms_pre", -1)), "max_per_img": int(live.get("max_per_img", -1)),
                "min_bbox_size": float(live.get("min_bbox_size", -1)), "score_thr_resolved": float(live.get("score_thr", 0)),
                "nms_type": nms.get("type"), "nms_iou_threshold": float(nms.get("iou_threshold", -1))}
    expected = {"nms_pre": 1000, "max_per_img": 1000, "min_bbox_size": 0., "score_thr_resolved": 0.,
                "nms_type": "nms", "nms_iou_threshold": .7}
    if resolved != expected: raise RuntimeError(f"unexpected live RPN contract: {resolved}")
    return resolved


def staged_readout(model, item, device):
    """Read all four nested stage outputs without writing proposals."""
    data = prepare(model, item, device); sample = data["data_samples"][0]
    x, _, _, _ = model.extract_feat(data["inputs"], data["data_samples"])
    head = model.rpn_head; cls_scores, bbox_preds = head(x)
    raw_boxes, raw_scores, layout, anchor_hashes = dense_raw(head, cls_scores, bbox_preds, sample.metainfo["img_shape"], device)
    live = copy.deepcopy(head.test_cfg)
    resolved = assert_live_contract(live)
    pre_boxes, pre_scores = deployed_pre_nms(head, cls_scores, bbox_preds, sample.metainfo["img_shape"], device)
    # ``-1`` is not unlimited in this mmdet version: InstanceData[:-1]
    # removes one proposal.  A bound above the selected pre-NMS cardinality
    # preserves every candidate after live post-processing/NMS.
    unlimited_cfg = copy.deepcopy(live); unlimited_cfg["max_per_img"] = int(len(pre_boxes) + 1)
    unlimited = head.predict_by_feat(cls_scores, bbox_preds, batch_img_metas=[sample.metainfo], cfg=unlimited_cfg, rescale=False, with_nms=True)[0]
    final = head.predict_by_feat(cls_scores, bbox_preds, batch_img_metas=[sample.metainfo], cfg=live, rescale=False, with_nms=True)[0]
    no_proposal_write = sample.get("proposals", None) is None
    if not no_proposal_write:
        raise RuntimeError("D1 readout wrote proposals")
    unlimited_boxes, unlimited_scores = instance_arrays(unlimited); final_boxes, final_scores = instance_arrays(final)
    if len(unlimited_boxes) >= unlimited_cfg["max_per_img"]: raise RuntimeError("unlimited RPN cell may have truncated")
    if len(final.labels) and not bool((final.labels == 0).all()): raise RuntimeError("unexpected nonzero RPN labels")
    configs = {"raw": {"activation": "sigmoid" if head.use_sigmoid_cls else "softmax", "nms_pre": None, "nms": None, "max_per_img": None},
               "pre_nms": {"activation": "sigmoid" if head.use_sigmoid_cls else "softmax", "nms_pre": resolved["nms_pre"], "min_bbox_size": resolved["min_bbox_size"], "nms": None, "max_per_img": None},
               "nms_unlimited": {**resolved, "max_per_img": unlimited_cfg["max_per_img"]}, "final": resolved}
    return {"raw": (raw_boxes, raw_scores), "pre_nms": (pre_boxes, pre_scores), "nms_unlimited": (unlimited_boxes, unlimited_scores), "final": (final_boxes, final_scores), "layout": layout, "live_cfg": live, "resolved": resolved, "anchor_hashes": anchor_hashes, "feature_hashes": tensor_hashes(x), "configs": configs, "unlimited_bound": unlimited_cfg["max_per_img"], "unlimited_count": len(unlimited_boxes), "unlimited_nontruncated": len(unlimited_boxes) < unlimited_cfg["max_per_img"], "no_proposal_write": no_proposal_write}


def category(values):
    raw, pre, unlimited, final = (values[k] for k in ("raw", "pre_nms", "nms_unlimited", "final"))
    if raw < .5: return "raw_geometry_unavailable"
    if pre < .5: return "score_nms_pre_rejected"
    if unlimited < .5: return "nms_suppressed"
    if final < .5: return "max_per_img_truncated"
    raise RuntimeError("B0 RPN-miss has aligned deployed-final candidate")


def bootstrap(per_scene, samples, seed):
    values = np.asarray([[x[s] for s in STAGES] for x in per_scene], dtype=np.int64)
    rng = np.random.default_rng(seed); draws = rng.integers(0, len(values), size=(samples, len(values)))
    sums = values[draws].sum(axis=1); total = sums.sum(axis=1)
    if not (total > 0).all(): raise RuntimeError("invalid D1 bootstrap draw")
    proportions = sums / total[:, None]; major = proportions.argmax(axis=1)
    # The reported gap uses the observed major category fixed before resampling.
    observed = values.sum(axis=0); major_idx = int(observed.argmax())
    gap = proportions[:, major_idx] - (1 - proportions[:, major_idx])
    return {"samples_requested": samples, "samples_valid": int(len(gap)), "seed": seed,
            "observed_major_category": STAGES[major_idx], "major_minus_rest_ci95": [float(x) for x in np.percentile(gap, [2.5, 97.5])],
            "per_draw_major_counts": {STAGES[i]: int((major == i).sum()) for i in range(len(STAGES))}}


def main():
    a = arguments(); dev = torch.device("cuda:0" if a.device == "cuda" else a.device)
    b0 = json.loads((a.b0_dir / "summary.json").read_text()); rows_b0 = list(csv.DictReader((a.b0_dir / "instances.csv").open()))
    if not (b0["data"]["complete"] and b0["data"]["scenes_processed"] == 259 and b0["data"]["gt_instances"] == 5937 and all(b0["invariants"].values()) and Path(b0["checkpoint"]).resolve() == Path(a.checkpoint).resolve()):
        raise RuntimeError("B0 lineage contract failed")
    targets = [r for r in rows_b0 if r["area_bucket"] == "small" and r["final_failure"] in ("missed", "partial_or_merge") and r["rpn_recall_iou05"] == "False"]
    by_scene = defaultdict(dict)
    for row in targets: by_scene[row["scene_id"]][int(row["gt_index"])] = row
    if len(targets) != TARGETS or sum(len(v) for v in by_scene.values()) != TARGETS: raise RuntimeError("B0 D1 target count/identity failed")
    checkpoint = _load_checkpoint(Path(a.checkpoint)); snap = _snapshot(checkpoint); la = SimpleNamespace(config=None, sam2_ckpt=a.sam2_ckpt, sam2_repo=None)
    _resolve_sam2_repo(la, snap); cfg, source = _resolve_model_config(la, snap); model = _register_and_build(cfg).to(dev).eval()
    report = _load_model_state(model, _select_state_dict(checkpoint, "model"), False)
    if report["missing_keys"] or report["unexpected_keys"]: raise RuntimeError(f"strict model load failed: {report}")
    dataset = UEDualStreamDataset(a.data_root, "val", image_size=(1024, 1024), load_uav=False)
    if len(dataset) != 259: raise RuntimeError("D1 official val split contract failed")
    ids = [sid for sid in dataset.ids if sid in by_scene]; n = len(ids) if not a.max_target_scenes else min(a.max_target_scenes, len(ids)); ids = ids[:n]; index = {sid: i for i, sid in enumerate(dataset.ids)}
    a.output_dir.mkdir(parents=True, exist_ok=True); target_rows=[]; parity=[]; repeats=[]; layout_rows=[]; stage_hashes=[]; per_processed={}; live_contract=None
    with torch.inference_mode():
        for sid in ids:
            item = dataset[index[sid]]; before = normal_predict(model, item, dev); deployed_before_boxes, deployed_before_scores = read_rpn(model, item, dev)
            first = staged_readout(model, item, dev); second = staged_readout(model, item, dev)
            deployed_after_boxes, deployed_after_scores = read_rpn(model, item, dev); after = normal_predict(model, item, dev)
            if live_contract is None: live_contract = first["resolved"]
            elif first["resolved"] != live_contract: raise RuntimeError("live RPN config drift between scenes")
            before_rpn, after_rpn = stage_hash(deployed_before_boxes, deployed_before_scores), stage_hash(deployed_after_boxes, deployed_after_scores)
            final_rpn = stage_hash(*first["final"])
            parity.append({"scene_id": sid, "prediction_equal": output_digest(before) == output_digest(after), "deployed_rpn_equal": before_rpn == after_rpn, "staged_final_equal_live": final_rpn == before_rpn, "normal_before": output_digest(before), "normal_after": output_digest(after), "deployed_before": before_rpn, "deployed_after": after_rpn, "staged_final": final_rpn})
            repeats.append({"scene_id": sid, "raw_equal": stage_hash(*first["raw"]) == stage_hash(*second["raw"]), "pre_nms_equal": stage_hash(*first["pre_nms"]) == stage_hash(*second["pre_nms"]), "final_equal": stage_hash(*first["final"]) == stage_hash(*second["final"])})
            stage_hashes.append({"scene_id": sid, "feature_hashes": first["feature_hashes"], "anchor_hashes": first["anchor_hashes"], "raw": stage_hash(*first["raw"]), "pre_nms": stage_hash(*first["pre_nms"]), "nms_unlimited": stage_hash(*first["nms_unlimited"]), "final": final_rpn, "configs": first["configs"], "unlimited_bound": first["unlimited_bound"], "unlimited_count": first["unlimited_count"], "unlimited_nontruncated": first["unlimited_nontruncated"], "no_proposal_write": first["no_proposal_write"]})
            layout_rows.extend({"scene_id":sid, **x} for x in first["layout"])
            masks, _, gt = gt_from_item(item); stat={"scene_id": sid, **{stage:0 for stage in STAGES}}
            for gi, b0row in by_scene[sid].items():
                stage_iou = {name: max_iou(gt[gi], first[name][0]) for name in ("raw", "pre_nms", "nms_unlimited", "final")}
                replay = max_iou(gt[gi], deployed_before_boxes)
                if abs(replay - float(b0row["max_rpn_box_iou"])) > 1e-5 or abs(stage_iou["final"] - replay) > 1e-5: raise RuntimeError(f"B0/deployed-final replay drift {sid}/{gi}")
                if not (stage_iou["raw"] + 1e-6 >= stage_iou["pre_nms"] and stage_iou["pre_nms"] + 1e-6 >= stage_iou["nms_unlimited"] and stage_iou["nms_unlimited"] + 1e-6 >= stage_iou["final"]): raise RuntimeError(f"stage nesting monotonicity failed {sid}/{gi}")
                kind = category(stage_iou); stat[kind] += 1
                target_rows.append({"scene_id":sid,"gt_index":gi,"b0_final_failure":b0row["final_failure"],"b0_max_rpn_box_iou":float(b0row["max_rpn_box_iou"]),"deployed_replay_max_iou":replay,"raw_max_iou":stage_iou["raw"],"pre_nms_max_iou":stage_iou["pre_nms"],"nms_unlimited_max_iou":stage_iou["nms_unlimited"],"final_max_iou":stage_iou["final"],"category":kind,"raw_count":len(first["raw"][0]),"pre_nms_count":len(first["pre_nms"][0]),"nms_unlimited_count":len(first["nms_unlimited"][0]),"final_count":len(first["final"][0])})
            per_processed[sid]=stat
    complete = n == len(by_scene) and len(target_rows) == TARGETS
    per_scene=[per_processed.get(s,{"scene_id":s,**{stage:0 for stage in STAGES}}) for s in dataset.ids]
    counts={stage:sum(x[stage] for x in per_scene) for stage in STAGES}; total=sum(counts.values())
    # A deliberately capped smoke has zero-filled unprocessed scenes and hence
    # cannot support the registered 259-scene bootstrap.  It must still emit
    # the single expected coverage INVALID after all local contracts ran.
    boot=bootstrap(per_scene,a.bootstrap_samples,a.bootstrap_seed) if total and complete else None
    major=max(counts,key=counts.get); invalid=[]
    if not complete: invalid.append("validation_target_coverage_incomplete")
    if not all(x["prediction_equal"] and x["deployed_rpn_equal"] and x["staged_final_equal_live"] for x in parity): invalid.append("prediction_or_deployed_rpn_parity_failed")
    if not all(all(x.values()) for x in repeats): invalid.append("raw_or_pre_nms_repeat_failed")
    if not all(x["prior_count"] == x["delta_count"] == x["score_count"] == x["decoded_count"] for x in layout_rows): invalid.append("dense_cardinality_failed")
    if not all(x["unlimited_nontruncated"] and x["no_proposal_write"] for x in stage_hashes): invalid.append("unlimited_or_no_write_contract_failed")
    decision="INVALID"
    if not invalid:
        share=counts[major]/total
        decision=("D1_"+major.upper()) if share>=.60 and boot["major_minus_rest_ci95"][0]>0 else "D1_MIXED"
    summary={"experiment_id":a.experiment_id,"arm":"D1-Frozen-Satellite-RPN-Stage-Attribution","status":"probe_only_no_training","checkpoint":str(Path(a.checkpoint).resolve()),"model_config_source":source,"b0_lineage":{"dir":str(a.b0_dir.resolve()),"targets_expected":TARGETS},"data":{"val_scene_universe":259,"target_scenes_expected":len(by_scene),"target_scenes_processed":n,"targets_expected":TARGETS,"targets_processed":len(target_rows),"complete":complete},"live_rpn_contract":live_contract,"invariants":{"prediction_parity_bitwise":all(x["prediction_equal"] for x in parity),"deployed_rpn_parity_bitwise":all(x["deployed_rpn_equal"] for x in parity),"staged_final_equals_live_bitwise":all(x["staged_final_equal_live"] for x in parity),"raw_repeat_bitwise":all(x["raw_equal"] for x in repeats),"pre_nms_repeat_bitwise":all(x["pre_nms_equal"] for x in repeats),"final_repeat_bitwise":all(x["final_equal"] for x in repeats),"dense_anchor_cardinality":all(x["prior_count"] == x["delta_count"] == x["score_count"] == x["decoded_count"] for x in layout_rows),"b0_lineage_fail_closed":True,"no_proposal_write":True},"metrics":{"counts":counts,"fractions":{k:v/total if total else None for k,v in counts.items()},"raw_align_rate":(total-counts["raw_geometry_unavailable"])/total if total else None,"pre_nms_align_rate":(counts["nms_suppressed"]+counts["max_per_img_truncated"])/total if total else None,"nms_unlimited_align_rate":counts["max_per_img_truncated"]/total if total else None},"bootstrap":boot,"invalid_reasons":invalid,"decision":decision,"decision_rule":"D1_<STAGE> iff complete/invariants pass, largest stage>=.60 and its image-paired bootstrap major-minus-rest CI lower>0; else D1_MIXED; any contract failure is INVALID.","interpretation":"D1 attributes a fixed baseline RPN failure slice; it neither changes proposals nor proves an intervention/AP gain.","per_scene":per_scene,"parity":parity,"repeats":repeats,"stage_hashes":stage_hashes}
    fields=list(target_rows[0]) if target_rows else []
    with (a.output_dir/"targets.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(target_rows)
    with (a.output_dir/"dense_layout.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=["scene_id","level","prior_count","delta_count","score_count","decoded_count"]);w.writeheader();w.writerows(layout_rows)
    (a.output_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n");(a.output_dir/"bootstrap.json").write_text(json.dumps(boot,indent=2)+"\n")
    (a.output_dir/"run_manifest.json").write_text(json.dumps({k:summary[k] for k in ("experiment_id","arm","status","checkpoint","b0_lineage","data","live_rpn_contract","invariants","decision")},indent=2)+"\n")
    print(json.dumps({"output":str(a.output_dir/"summary.json"),"decision":decision,"metrics":summary["metrics"],"invariants":summary["invariants"]},indent=2))


if __name__=="__main__": main()
