#!/usr/bin/env python3
"""D2: frozen rank-profile probe for D1's per-level RPN preselection losses."""
from __future__ import annotations

import argparse, csv, hashlib, json, os, sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("SAM2_CKPT", "/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt")
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from data.ue_dual_stream_dataset import UEDualStreamDataset
from inference.infer_from_checkpoint import (_load_checkpoint,_load_model_state,_register_and_build,_resolve_model_config,_resolve_sam2_repo,_select_state_dict,_snapshot)
from probe_small_postrpn_bottleneck import output_digest as typed_normal_digest
from probe_small_rpn_bottleneck import box_iou_xyxy,gt_from_item,normal_predict,prepare,read_rpn
from probe_rpn_stage_attribution import category as d1_category, staged_readout

MAIN="score_nms_pre_rejected"; CATS=("raw_geometry_unavailable",MAIN,"nms_suppressed","max_per_img_truncated"); CAPS=(1250,1500,2000,4000)

def args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--checkpoint",required=True);p.add_argument("--b0-dir",type=Path,required=True);p.add_argument("--d1-dir",type=Path,required=True);p.add_argument("--data-root",default="/data1/wangcheng/dataset/ue_coco_with_uav");p.add_argument("--device",default="cuda");p.add_argument("--sam2-ckpt",default=os.environ["SAM2_CKPT"]);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--max-target-scenes",type=int,default=0);p.add_argument("--scene-id",default=None,help="Capped-smoke only: one explicit D1 target scene.");p.add_argument("--bootstrap-samples",type=int,default=10000);p.add_argument("--bootstrap-seed",type=int,default=20260921);p.add_argument("--experiment-id",default="MHBEV-DIAG-20260918-D2");return p.parse_args()

def digest(x):
 if isinstance(x,torch.Tensor): x=x.detach().float().cpu().numpy()
 x=np.ascontiguousarray(x);h=hashlib.sha256();h.update(str(x.dtype).encode());h.update(repr(x.shape).encode());h.update(x.tobytes());return h.hexdigest()
def pair_hash(boxes,scores): return {"boxes":digest(boxes),"scores":digest(scores)}
def max_iou(gt,boxes): return float(box_iou_xyxy(gt[None],boxes).max()) if len(boxes) else 0.

def assert_live(head):
 c=head.test_cfg;n=c.get("nms",{}); got={"nms_pre":int(c.get("nms_pre",-1)),"max_per_img":int(c.get("max_per_img",-1)),"min_bbox_size":float(c.get("min_bbox_size",-1)),"score_thr_resolved":float(c.get("score_thr",0)),"nms_type":n.get("type"),"nms_iou_threshold":float(n.get("iou_threshold",-1))}; exp={"nms_pre":1000,"max_per_img":1000,"min_bbox_size":0.,"score_thr_resolved":0.,"nms_type":"nms","nms_iou_threshold":.7}
 if got!=exp: raise RuntimeError(f"unexpected live RPN contract: {got}")
 return got

def rank_readout(model,item,dev):
 data=prepare(model,item,dev); sample=data["data_samples"][0];x,_,_,_=model.extract_feat(data["inputs"],data["data_samples"]);head=model.rpn_head; cls,deltas=head(x); contract=assert_live(head)
 priors=head.prior_generator.grid_priors([v.shape[-2:] for v in cls],dtype=cls[0].dtype,device=dev); levels=[]
 for level,(logit,delta,prior) in enumerate(zip(cls,deltas,priors)):
  score=logit[0].permute(1,2,0).reshape(-1,head.cls_out_channels);score=score.sigmoid() if head.use_sigmoid_cls else score.softmax(-1)[:,:-1];score=score.squeeze();delta=delta[0].permute(1,2,0).reshape(-1,head.bbox_coder.encode_size);boxes=head.bbox_coder.decode(prior,delta,max_shape=sample.metainfo["img_shape"])
  order=score.sort(descending=True).indices;ranks=torch.empty_like(order);ranks[order]=torch.arange(1,len(order)+1,device=dev)
  levels.append({"level":level,"boxes":boxes.detach().float().cpu().numpy(),"scores":score.detach().float().cpu().numpy(),"ranks":ranks.detach().cpu().numpy(),"hash":{"prior":digest(prior),"boxes":digest(boxes),"scores":digest(score),"ranks":digest(ranks)},"count":len(prior)})
 if sample.get("proposals",None) is not None: raise RuntimeError("D2 readout wrote proposals")
 return levels,{"features":[digest(v) for v in x],"levels":[v["hash"] for v in levels],"contract":contract}

def bootstrap(scene_stats,samples,seed):
 arr=np.asarray(scene_stats,dtype=np.int64);rng=np.random.default_rng(seed); vals=[]; redraws=0
 while len(vals)<samples:
  draw=arr[rng.integers(0,len(arr),len(arr))]; den=draw[:,1].sum()
  if not den: redraws+=1;continue
  vals.append(draw[:,0].sum()/den)
 v=np.asarray(vals);return {"samples_requested":samples,"samples_valid":len(v),"seed":seed,"redraws_zero_denominator":redraws,"cap":2000,"ci95":[float(x) for x in np.percentile(v,[2.5,97.5])]}

def main():
 a=args();dev=torch.device("cuda:0" if a.device=="cuda" else a.device); d1=json.loads((a.d1_dir/"summary.json").read_text()); rows=list(csv.DictReader((a.d1_dir/"targets.csv").open())); b0=json.loads((a.b0_dir/"summary.json").read_text()); b0rows=list(csv.DictReader((a.b0_dir/"instances.csv").open()))
 if not(d1["decision"]=="D1_MIXED" and d1["data"]["complete"] and all(d1["invariants"].values()) and Path(d1["checkpoint"]).resolve()==Path(a.checkpoint).resolve()): raise RuntimeError("D1 lineage contract failed")
 if not(b0["data"]["complete"] and b0["data"]["scenes_processed"]==259 and b0["data"]["gt_instances"]==5937 and all(b0["invariants"].values()) and Path(b0["checkpoint"]).resolve()==Path(a.checkpoint).resolve()):raise RuntimeError("B0 lineage contract failed")
 b0map={(r["scene_id"],int(r["gt_index"])):r for r in b0rows if r["area_bucket"]=="small" and r["final_failure"] in ("missed","partial_or_merge") and r["rpn_recall_iou05"]=="False"};b0targets=set(b0map)
 if len(b0targets)!=107:raise RuntimeError("B0 target count failed")
 by_scene=defaultdict(dict);d1keys=[]
 for r in rows: d1keys.append((r["scene_id"],int(r["gt_index"])));by_scene[r["scene_id"]][int(r["gt_index"])]=r
 counts=Counter(r["category"] for r in rows)
 if len(rows)!=107 or len(set(d1keys))!=107 or set(d1keys)!=b0targets or set(counts)!={*CATS} or counts!=Counter({"raw_geometry_unavailable":23,MAIN:59,"nms_suppressed":2,"max_per_img_truncated":23}): raise RuntimeError(f"D1 target/category contract failed: {counts}")
 ckpt=_load_checkpoint(Path(a.checkpoint));snap=_snapshot(ckpt);la=SimpleNamespace(config=None,sam2_ckpt=a.sam2_ckpt,sam2_repo=None);_resolve_sam2_repo(la,snap);cfg,source=_resolve_model_config(la,snap);model=_register_and_build(cfg).to(dev).eval();report=_load_model_state(model,_select_state_dict(ckpt,"model"),False)
 if report["missing_keys"] or report["unexpected_keys"]:raise RuntimeError(f"strict load failed: {report}")
 ds=UEDualStreamDataset(a.data_root,"val",image_size=(1024,1024),load_uav=False)
 if len(ds)!=259:raise RuntimeError("official val contract failed")
 ids=[s for s in ds.ids if s in by_scene]
 if a.scene_id is not None:
  if a.scene_id not in by_scene: raise RuntimeError("--scene-id is not a D1 target scene")
  ids=[a.scene_id]
 else: ids=ids[:(a.max_target_scenes or len(ids))]
 index={s:i for i,s in enumerate(ds.ids)};a.output_dir.mkdir(parents=True,exist_ok=True)
 out=[]; scenes=[]; telemetry=[]; invalid=[]
 with torch.inference_mode():
  for sid in ids:
   item=ds[index[sid]];before=normal_predict(model,item,dev); live_b=read_rpn(model,item,dev); stage=staged_readout(model,item,dev); first,h1=rank_readout(model,item,dev);second,h2=rank_readout(model,item,dev);live_a=read_rpn(model,item,dev);after=normal_predict(model,item,dev)
   if h1!=h2: raise RuntimeError(f"rank readout repeat hash drift {sid}")
   if typed_normal_digest(before)!=typed_normal_digest(after) or pair_hash(*live_b)!=pair_hash(*live_a) or pair_hash(*stage["final"])!=pair_hash(*live_b): raise RuntimeError(f"prediction/RPN/staged-final parity drift {sid}")
   _,_,gt=gt_from_item(item); stat={"scene_id":sid,"recovered":0,"main":0}
   for gi,r in by_scene[sid].items():
    candidates=[]
    for level in first:
     inds=np.flatnonzero(box_iou_xyxy(gt[gi][None],level["boxes"])[0]>=.5)
     candidates += [(int(level["ranks"][i]),level["level"],float(level["scores"][i])) for i in inds]
    minrank=min((x[0] for x in candidates),default=None); cat=r["category"]; replay=max_iou(gt[gi],live_b[0])
    expected={"raw":float(r["raw_max_iou"]),"pre_nms":float(r["pre_nms_max_iou"]),"nms_unlimited":float(r["nms_unlimited_max_iou"]),"final":float(r["final_max_iou"])}; observed={name:max_iou(gt[gi],stage[name][0]) for name in expected}; observed_cat=d1_category(observed)
    b0_iou=float(b0map[(sid,gi)]["max_rpn_box_iou"])
    if abs(replay-b0_iou)>1e-5 or abs(replay-float(r["deployed_replay_max_iou"]))>1e-5 or any(abs(observed[k]-expected[k])>1e-5 for k in expected) or observed_cat!=cat:raise RuntimeError(f"B0/D1 stage replay/category drift {sid}/{gi}")
    if cat=="raw_geometry_unavailable" and minrank is not None:raise RuntimeError(f"raw control violates no-positive {sid}/{gi}")
    if cat==MAIN and not(minrank is not None and minrank>1000):raise RuntimeError(f"main rank separation drift {sid}/{gi}")
    if cat in ("nms_suppressed","max_per_img_truncated") and not(minrank is not None and minrank<=1000):raise RuntimeError(f"selected control drift {sid}/{gi}")
    recovered={str(cap):(minrank is not None and minrank<=cap) for cap in CAPS}
    if cat==MAIN:stat["main"]+=1;stat["recovered"]+=int(recovered["2000"])
    out.append({"scene_id":sid,"gt_index":gi,"d1_category":cat,"recomputed_d1_category":observed_cat,"b0_max_rpn_box_iou":b0_iou,"d1_deployed_replay_max_iou":float(r["deployed_replay_max_iou"]),"d2_deployed_replay_max_iou":replay,**{f"d1_{name}_max_iou":expected[name] for name in expected},**{f"d2_{name}_max_iou":observed[name] for name in expected},"min_positive_rank":minrank,"positive_levels":";".join(map(str,sorted(set(x[1] for x in candidates)))),"positive_count":len(candidates),"best_positive_score":max((x[2] for x in candidates),default=None),"rank_margin_to_1000":None if minrank is None else minrank-1000,**{f"recovered_cap_{cap}":recovered[str(cap)] for cap in CAPS}})
   scenes.append(stat);telemetry.append({"scene_id":sid,"normal_before":typed_normal_digest(before),"normal_after":typed_normal_digest(after),"live_before":pair_hash(*live_b),"live_after":pair_hash(*live_a),"staged_final":pair_hash(*stage["final"]),"readout":h1})
 complete=len(ids)==len(by_scene) and len(out)==107
 main=[x for x in out if x["d1_category"]==MAIN]; recovery={str(cap):sum(x[f"recovered_cap_{cap}"] for x in main)/len(main) for cap in CAPS} if main else {}; boot=bootstrap([[x["recovered"],x["main"]] for x in scenes],a.bootstrap_samples,a.bootstrap_seed) if complete else None
 controls={"raw_no_positive":{"pass":sum(x["min_positive_rank"] is None for x in out if x["d1_category"]=="raw_geometry_unavailable"),"total":23},"selected_rank_le_1000":{"pass":sum(x["min_positive_rank"] is not None and x["min_positive_rank"]<=1000 for x in out if x["d1_category"] in ("nms_suppressed","max_per_img_truncated")),"total":25}}
 if not complete:invalid.append("validation_target_coverage_incomplete")
 decision="INVALID" if invalid else ("D2_BOUNDARY_RECOVERABLE" if recovery["2000"]>=.70 and boot["ci95"][0]>=.50 else "D2_BROAD_RANK_DEFICIT")
 summary={"experiment_id":a.experiment_id,"arm":"D2-Frozen-RPN-Rank-Profile","status":"probe_only_no_training","checkpoint":str(Path(a.checkpoint).resolve()),"model_config_source":source,"b0_lineage":{"dir":str(a.b0_dir.resolve()),"keys":107},"d1_lineage":{"dir":str(a.d1_dir.resolve()),"decision":d1["decision"],"counts":dict(counts)},"data":{"val_scene_universe":259,"target_scenes_expected":len(by_scene),"target_scenes_processed":len(ids),"targets_expected":107,"targets_processed":len(out),"main_targets_expected":59,"main_targets_processed":len(main),"complete":complete},"live_rpn_contract":telemetry[0]["readout"]["contract"] if telemetry else None,"invariants":{"normal_prediction_parity_typed":all(x["normal_before"]==x["normal_after"] for x in telemetry),"deployed_rpn_parity_typed":all(x["live_before"]==x["live_after"] for x in telemetry),"staged_final_equals_live_typed":all(x["staged_final"]==x["live_before"] for x in telemetry),"readout_repeat_typed_hash":len(telemetry)==len(ids),"d1_replay_and_rank_separation_fail_closed":True,"no_proposal_write":True},"metrics":{"recovery_by_cap":recovery,"main_min_rank_median":float(np.median([x["min_positive_rank"] for x in main])) if main else None,"main_min_rank_p90":float(np.percentile([x["min_positive_rank"] for x in main],90)) if main else None,"main_level_counts":dict(Counter(v for x in main for v in x["positive_levels"].split(";") if v)),"controls":controls},"bootstrap":boot,"invalid_reasons":invalid,"decision":decision,"decision_rule":"D2_BOUNDARY_RECOVERABLE iff complete/invariants, cap2000 recovery>=.70 and 85-scene paired bootstrap CI lower>=.50; else D2_BROAD_RANK_DEFICIT; contract failure INVALID.","interpretation":"D2 reads frozen candidate ranks; it does not alter proposals or demonstrate AP/compute benefit.","per_scene":scenes,"telemetry":telemetry}
 with (a.output_dir/"targets.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(out[0]) if out else []);w.writeheader();w.writerows(out)
 (a.output_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n");(a.output_dir/"bootstrap.json").write_text(json.dumps(boot,indent=2)+"\n");(a.output_dir/"run_manifest.json").write_text(json.dumps({k:summary[k] for k in ("experiment_id","arm","checkpoint","b0_lineage","d1_lineage","data","live_rpn_contract","invariants","decision")},indent=2)+"\n");print(json.dumps({"decision":decision,"metrics":summary["metrics"],"invariants":summary["invariants"]},indent=2))
if __name__=="__main__":main()
