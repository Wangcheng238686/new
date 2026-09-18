#!/usr/bin/env python3
"""D3: frozen satellite-FPN targetness readout on D1 low-ranked RPN positives."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,sys
from collections import Counter,defaultdict
from pathlib import Path
from types import SimpleNamespace
os.environ.setdefault("SAM2_CKPT","/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt")
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from data.ue_dual_stream_dataset import UEDualStreamDataset
from inference.infer_from_checkpoint import (_load_checkpoint,_load_model_state,_register_and_build,_resolve_model_config,_resolve_sam2_repo,_select_state_dict,_snapshot)
from probe_small_postrpn_bottleneck import output_digest
from probe_small_rpn_bottleneck import box_iou_xyxy,gt_from_item,normal_predict,prepare,read_rpn
from probe_rpn_stage_attribution import category as d1_category,staged_readout

MAIN="score_nms_pre_rejected"; TRAIN_RATIO=.1797752808988764; TRAIN_SEED=44
def args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--checkpoint",required=True);p.add_argument("--b0-dir",type=Path,required=True);p.add_argument("--d1-dir",type=Path,required=True);p.add_argument("--data-root",default="/data1/wangcheng/dataset/ue_coco_with_uav");p.add_argument("--device",default="cuda");p.add_argument("--sam2-ckpt",default=os.environ["SAM2_CKPT"]);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--max-train-scenes",type=int,default=0);p.add_argument("--max-val-scenes",type=int,default=0);p.add_argument("--scene-id",default=None);p.add_argument("--bootstrap-samples",type=int,default=10000);p.add_argument("--bootstrap-seed",type=int,default=20260924);p.add_argument("--experiment-id",default="MHBEV-DIAG-20260918-D3");return p.parse_args()
def dh(x):
 if isinstance(x,torch.Tensor):x=x.detach().float().cpu().numpy()
 x=np.ascontiguousarray(x);h=hashlib.sha256();h.update(str(x.dtype).encode());h.update(repr(x.shape).encode());h.update(x.tobytes());return h.hexdigest()
def pairhash(rec):return hashlib.sha256(json.dumps(rec,sort_keys=True,separators=(",",":"),default=float).encode()).hexdigest()
def pairh(b,s):return {"boxes":dh(b),"scores":dh(s)}
def maxi(gt,b):return float(box_iou_xyxy(gt[None],b).max()) if len(b) else 0.
def dense(model,item,dev):
 d=prepare(model,item,dev);sample=d["data_samples"][0];x,_,_,_=model.extract_feat(d["inputs"],d["data_samples"]);head=model.rpn_head;cls,deltas=head(x);priors=head.prior_generator.grid_priors([v.shape[-2:] for v in cls],dtype=cls[0].dtype,device=dev);outs=[]
 for lev,(c,de,p,f) in enumerate(zip(cls,deltas,priors,x)):
  h,w=c.shape[-2:];a=len(p)//(h*w)
  if a*h*w!=len(p) or c.shape[1]!=a*head.cls_out_channels or de.shape[1]!=a*head.bbox_coder.encode_size or tuple(f.shape[-2:])!=(h,w) or f.shape[1]!=head.in_channels:raise RuntimeError("anchor-grid/FPN mapping contract failed")
  # AnchorGenerator keeps base_anchors on CPU; its sparse-index API therefore
  # requires CPU indices even when it materializes the resulting priors on GPU.
  probe=torch.arange(len(p),device="cpu");sparse=head.prior_generator.sparse_priors(probe,(h,w),lev,dtype=p.dtype,device=dev)
  if not torch.equal(sparse,p) or dh(sparse)!=dh(p):raise RuntimeError("anchor index-to-cell sparse/grid order mismatch")
  score=c[0].permute(1,2,0).reshape(-1,head.cls_out_channels);score=score.sigmoid() if head.use_sigmoid_cls else score.softmax(-1)[:,:-1];score=score.reshape(-1);de=de[0].permute(1,2,0).reshape(-1,head.bbox_coder.encode_size);box=head.bbox_coder.decode(p,de,max_shape=sample.metainfo["img_shape"])
  idx=torch.arange(len(p),device=dev);cell=idx//a;desc=f[0,:,cell//w,cell%w].transpose(0,1)
  if not(len(score)==len(box)==len(desc)==len(p)):raise RuntimeError("dense descriptor cardinality failed")
  outs.append({"level":lev,"boxes":box.detach().float().cpu().numpy(),"scores":score.detach().float().cpu().numpy(),"desc":desc.detach().float().cpu().numpy(),"anchors_per_cell":a,"feature_w":w,"hash":{"boxes":dh(box),"scores":dh(score),"desc":dh(desc),"prior":dh(p),"sparse_prior":dh(sparse)},"layout":{"level":lev,"anchors":len(p),"feature_h":h,"feature_w":w,"feature_channels":f.shape[1],"anchors_per_cell":a,"cls_channels":c.shape[1],"delta_channels":de.shape[1]}})
 if sample.get("proposals",None) is not None:raise RuntimeError("D3 readout wrote proposals")
 return outs,{"features":[dh(v) for v in x],"levels":[o["hash"] for o in outs],"layouts":[o["layout"] for o in outs]}
def pairs(levels,gt,gi,sid):
 allious=[]
 for o in levels:allious.append(box_iou_xyxy(gt,o["boxes"]))
 pos=[]
 for li,o in enumerate(levels):
  ids=np.flatnonzero(allious[li][gi]>=.5)
  pos += [(float(o["scores"][j]),li,int(j),float(allious[li][gi,j])) for j in ids]
 if not pos:return None
 ps,li,pi,piou=max(pos);o=levels[li];bg=np.flatnonzero(allious[li].max(axis=0)<.1)
 if not len(bg):return None
 ni=int(bg[np.abs(o["scores"][bg]-ps).argmin()]);gap=float(abs(o["scores"][ni]-ps))
 rec={"scene_id":sid,"gt_index":int(gi),"level":li,"pos_index":pi,"neg_index":ni,"pos_cell_index":pi//o["anchors_per_cell"],"neg_cell_index":ni//o["anchors_per_cell"],"pos_cell_row":pi//o["anchors_per_cell"]//o["feature_w"],"pos_cell_col":pi//o["anchors_per_cell"]%o["feature_w"],"neg_cell_row":ni//o["anchors_per_cell"]//o["feature_w"],"neg_cell_col":ni//o["anchors_per_cell"]%o["feature_w"],"pos_box":o["boxes"][pi].tolist(),"neg_box":o["boxes"][ni].tolist(),"pos_score":ps,"neg_score":float(o["scores"][ni])}
 return o["desc"][pi],o["desc"][ni],piou,gap,pairhash(rec),rec
def boot(stats,n,seed):
 v=np.asarray(stats,dtype=np.int64);rng=np.random.default_rng(seed);draw=rng.integers(0,len(v),size=(n,len(v)));s=v[draw].sum(1);ok=s[:,3]>0;diff=s[ok,0]/s[ok,3]-s[ok,1]/s[ok,3];raw=s[ok,2]/s[ok,3]-.5;return {"samples_requested":n,"samples_valid":int(ok.sum()),"seed":seed,"correct_minus_sham_ci95":[float(x) for x in np.percentile(diff,[2.5,97.5])],"raw_score_minus_tie_ci95":[float(x) for x in np.percentile(raw,[2.5,97.5])]}
def main():
 a=args();dev=torch.device("cuda:0" if a.device=="cuda" else a.device);d1=json.loads((a.d1_dir/"summary.json").read_text());d1rows=list(csv.DictReader((a.d1_dir/"targets.csv").open()));b0=json.loads((a.b0_dir/"summary.json").read_text());b0rows=list(csv.DictReader((a.b0_dir/"instances.csv").open()))
 if not(d1["data"]["complete"] and all(d1["invariants"].values()) and d1["decision"]=="D1_MIXED" and b0["data"]["complete"] and all(b0["invariants"].values()) and Path(d1["checkpoint"]).resolve()==Path(a.checkpoint).resolve()==Path(b0["checkpoint"]).resolve()):raise RuntimeError("B0/D1 lineage failed")
 b0map={(r["scene_id"],int(r["gt_index"])):r for r in b0rows if r["area_bucket"]=="small" and r["final_failure"] in ("missed","partial_or_merge") and r["rpn_recall_iou05"]=="False"}; mainrows=[r for r in d1rows if r["category"]==MAIN];keys=[(r["scene_id"],int(r["gt_index"])) for r in mainrows]
 if len(b0map)!=107 or len(mainrows)!=59 or len(set(keys))!=59 or not set(keys)<=set(b0map):raise RuntimeError("D3 target lineage/count failed")
 byscene=defaultdict(dict)
 for r in mainrows:byscene[r["scene_id"]][int(r["gt_index"])]=r
 all_d1_scenes=sorted({r["scene_id"] for r in d1rows})
 if len(all_d1_scenes)!=85:raise RuntimeError("D1 85-scene bootstrap contract failed")
 ck=_load_checkpoint(Path(a.checkpoint));snap=_snapshot(ck);la=SimpleNamespace(config=None,sam2_ckpt=a.sam2_ckpt,sam2_repo=None);_resolve_sam2_repo(la,snap);cfg,source=_resolve_model_config(la,snap);model=_register_and_build(cfg).to(dev).eval();rep=_load_model_state(model,_select_state_dict(ck,"model"),False)
 if rep["missing_keys"] or rep["unexpected_keys"]:raise RuntimeError("strict load failed")
 tr=UEDualStreamDataset(a.data_root,"train",image_size=(1024,1024),train_subset=TRAIN_RATIO,seed=TRAIN_SEED,load_uav=False);va=UEDualStreamDataset(a.data_root,"val",image_size=(1024,1024),load_uav=False)
 if len(tr)!=128 or len(va)!=259 or set(tr.ids)&set(va.ids):raise RuntimeError("D3 split contract failed")
 trainx=[];trainy=[];trainrows=[];tele=[]
 with torch.inference_mode():
  for i in range(len(tr) if not a.max_train_scenes else a.max_train_scenes):
   item=tr[i];sid=tr.ids[i];before=normal_predict(model,item,dev);rb=read_rpn(model,item,dev);lv,h1=dense(model,item,dev);lv2,h2=dense(model,item,dev);ra=read_rpn(model,item,dev);after=normal_predict(model,item,dev)
   if h1!=h2 or output_digest(before)!=output_digest(after) or pairh(*rb)!=pairh(*ra):raise RuntimeError(f"train parity/hash drift {sid}")
   masks,_,gt=gt_from_item(item)
   for gi,m in enumerate(masks):
    if int(m.sum())>=1024:continue
    q=pairs(lv,gt,gi,sid)
    if q is None:continue
    p,n,piou,gap,sha,rec=q
    trainx += [p,n];trainy += [1,0];trainrows.append({"scene_id":sid,"gt_index":gi,"pair_sha256":sha,"pos_iou":piou,"score_gap":gap,**rec})
   tele.append({"scene_id":sid,"split":"train","normal_before":output_digest(before),"normal_after":output_digest(after),"rpn_before":pairh(*rb),"rpn_after":pairh(*ra),"readout":h1,"repeat_equal":h1==h2,"no_proposal_write":True})
  if len(trainrows)<20:raise RuntimeError(f"insufficient training pairs {len(trainrows)}")
  clf=make_pipeline(StandardScaler(),LogisticRegression(C=1.,class_weight="balanced",max_iter=1000,random_state=20260922));clf.fit(np.asarray(trainx),np.asarray(trainy));trainy=np.asarray(trainy);shamy=trainy[np.random.default_rng(20260923).permutation(len(trainy))]
  if np.array_equal(trainy,shamy):raise RuntimeError("sham permutation identity")
  sham=make_pipeline(StandardScaler(),LogisticRegression(C=1.,class_weight="balanced",max_iter=1000,random_state=20260922));sham.fit(np.asarray(trainx),shamy);sham_meta={"seed":20260923,"original_label_sha256":dh(trainy),"permuted_label_sha256":dh(shamy),"original_counts":dict(Counter(map(int,trainy)),),"permuted_counts":dict(Counter(map(int,shamy)))}
  out=[];stats=[];idx={s:i for i,s in enumerate(va.ids)};ids=[s for s in va.ids if s in byscene]
  if a.scene_id is not None:
   if a.scene_id not in byscene:raise RuntimeError("--scene-id must contain a D3 main target")
   ids=[a.scene_id]
  else:ids=ids[:(a.max_val_scenes or len(ids))]
  for sid in ids:
   item=va[idx[sid]];before=normal_predict(model,item,dev);rb=read_rpn(model,item,dev);stage=staged_readout(model,item,dev);lv,h1=dense(model,item,dev);lv2,h2=dense(model,item,dev);ra=read_rpn(model,item,dev);after=normal_predict(model,item,dev)
   if h1!=h2 or output_digest(before)!=output_digest(after) or pairh(*rb)!=pairh(*ra) or pairh(*stage["final"])!=pairh(*rb):raise RuntimeError(f"val parity/hash drift {sid}")
   masks,_,gt=gt_from_item(item);st={"scene_id":sid,"correct":0,"sham":0,"raw":0,"pairs":0}
   for gi,r in byscene[sid].items():
    obs={n:maxi(gt[gi],stage[n][0]) for n in ("raw","pre_nms","nms_unlimited","final")};exp={"raw":float(r["raw_max_iou"]),"pre_nms":float(r["pre_nms_max_iou"]),"nms_unlimited":float(r["nms_unlimited_max_iou"]),"final":float(r["final_max_iou"])};b0iou=float(b0map[(sid,gi)]["max_rpn_box_iou"]);replay=maxi(gt[gi],rb[0])
    if d1_category(obs)!=MAIN or any(abs(obs[k]-exp[k])>1e-5 for k in exp) or abs(replay-b0iou)>1e-5 or abs(replay-float(r["deployed_replay_max_iou"]))>1e-5:raise RuntimeError(f"D1/B0 replay drift {sid}/{gi}")
    q=pairs(lv,gt,gi,sid);row={"scene_id":sid,"gt_index":gi,"valid":False,"b0_max_rpn_iou":b0iou,"d1_replay_iou":float(r["deployed_replay_max_iou"]),"d3_replay_iou":replay}
    if q is not None:
     p,n,piou,gap,sha,rec=q;row.update({"pair_sha256":sha,"pos_iou":piou,"score_gap":gap,**rec})
     if gap<=.02:
      cp=clf.predict_proba(np.asarray([p,n]))[:,1];sp=sham.predict_proba(np.asarray([p,n]))[:,1];win=bool(cp[0]>cp[1]);swin=bool(sp[0]>sp[1]);rawwin=bool(rec["pos_score"]>rec["neg_score"]);row.update({"valid":True,"correct_win":win,"sham_win":swin,"raw_score_win":rawwin,"correct_pos_prob":float(cp[0]),"correct_neg_prob":float(cp[1]),"sham_pos_prob":float(sp[0]),"sham_neg_prob":float(sp[1])});st["pairs"]+=1;st["correct"]+=int(win);st["sham"]+=int(swin);st["raw"]+=int(rawwin)
    out.append(row)
   stats.append(st);tele.append({"scene_id":sid,"split":"val","normal_before":output_digest(before),"normal_after":output_digest(after),"rpn_before":pairh(*rb),"rpn_after":pairh(*ra),"staged_final":pairh(*stage["final"]),"readout":h1,"repeat_equal":h1==h2,"no_proposal_write":True})
 complete=len(ids)==len(byscene) and len(out)==59;valid=[r for r in out if r["valid"]];statmap={x["scene_id"]:x for x in stats};allstats=[[statmap.get(s,{"correct":0,"sham":0,"raw":0,"pairs":0})[k] for k in ("correct","sham","raw","pairs")] for s in all_d1_scenes];boot=boot(allstats,a.bootstrap_samples,a.bootstrap_seed) if complete and valid else None;acc=sum(r["correct_win"] for r in valid)/len(valid) if valid else None;shacc=sum(r["sham_win"] for r in valid)/len(valid) if valid else None;rawacc=sum(r["raw_score_win"] for r in valid)/len(valid) if valid else None;invalid=[]
 if not complete:invalid.append("validation_target_coverage_incomplete")
 if complete and not valid:invalid.append("no_valid_score_matched_pairs")
 if complete and boot is not None and boot["raw_score_minus_tie_ci95"][1]>.15:invalid.append("raw_score_match_control_failed")
 inv={"normal_prediction_parity_typed":all(x["normal_before"]==x["normal_after"] for x in tele),"deployed_rpn_parity_typed":all(x["rpn_before"]==x["rpn_after"] for x in tele),"repeat_feature_descriptor_hash":all(x["repeat_equal"] for x in tele),"d1_b0_replay_and_category_fail_closed":True,"no_proposal_write":all(x["no_proposal_write"] for x in tele)}
 decision="INVALID" if invalid else ("D3_FROZEN_SAT_OBJECTNESS_SIGNAL" if len(valid)>=48 and acc>=.65 and boot["correct_minus_sham_ci95"][0]>0 else "D3_NO_FROZEN_SAT_OBJECTNESS_SIGNAL")
 summ={"experiment_id":a.experiment_id,"arm":"D3-Frozen-Satellite-RPN-Objectness-Readout","status":"probe_only_no_training","checkpoint":str(Path(a.checkpoint).resolve()),"model_config_source":source,"b0_dir":str(a.b0_dir.resolve()),"d1_dir":str(a.d1_dir.resolve()),"sham":sham_meta,"data":{"train_scenes":len(tr) if not a.max_train_scenes else a.max_train_scenes,"train_pairs":len(trainrows),"bootstrap_scene_universe":85,"val_target_scenes_expected":len(byscene),"val_target_scenes_processed":len(ids),"targets_expected":59,"targets_processed":len(out),"valid_pairs":len(valid),"complete":complete},"invariants":inv,"metrics":{"correct_win_rate":acc,"sham_win_rate":shacc,"raw_score_win_rate":rawacc,"coverage":len(valid)/59,"mean_score_gap":float(np.mean([r["score_gap"] for r in valid])) if valid else None,"level_counts":dict(Counter(str(r["level"]) for r in valid))},"bootstrap":boot,"invalid_reasons":invalid,"decision":decision,"decision_rule":"D3_FROZEN_SAT_OBJECTNESS_SIGNAL iff complete/invariants, score-match CI upper<=.15, coverage>=48/59, correct win>=.65 and correct-minus-sham CI lower>0; else NO_SIGNAL; failures INVALID.","per_scene":stats,"bootstrap_per_scene":allstats,"telemetry":tele}
 a.output_dir.mkdir(parents=True,exist_ok=True)
 for name,data in (("train_pairs.csv",trainrows),("val_pairs.csv",out)):
  with (a.output_dir/name).open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(data[0]) if data else []);w.writeheader();w.writerows(data)
 layouts=[{"scene_id":x["scene_id"],"split":x["split"],**z} for x in tele for z in x["readout"]["layouts"]]
 with (a.output_dir/"descriptor_layout.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(layouts[0]) if layouts else []);w.writeheader();w.writerows(layouts)
 (a.output_dir/"summary.json").write_text(json.dumps(summ,indent=2)+"\n");(a.output_dir/"bootstrap.json").write_text(json.dumps(boot,indent=2)+"\n");(a.output_dir/"run_manifest.json").write_text(json.dumps({k:summ[k] for k in ("experiment_id","arm","checkpoint","b0_dir","d1_dir","data","invariants","decision")},indent=2)+"\n");print(json.dumps({"decision":decision,"metrics":summ["metrics"],"invariants":inv},indent=2))
if __name__=="__main__":main()
