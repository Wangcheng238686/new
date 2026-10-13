#!/usr/bin/env python3
"""P8: frozen conditional-ROI DBEV quality-information probe; Governance O."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[k]="1"
os.environ.setdefault("EXPLICIT_PROMPT_MODE","points_box_dense"); os.environ.setdefault("P2_BOUNDARY_REFINER_ENABLED","1")
os.environ.setdefault("SAM_IMAGE_EMBED_STRIDE","16"); os.environ.setdefault("SHAPE_DENSE_TRANSFORM","raw_logits")
os.environ.setdefault("SHAPE_DENSE_DETACH","1"); os.environ.setdefault("FINAL_MASK_COORDINATE_MODE","roi_local")
os.environ.setdefault("SAM2_CKPT","/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt"); os.environ.setdefault("UE_DUAL_INJECT_EMBEDDING","1")
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import cv2
from mmengine.config import Config
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmdet.registry import MODELS
from mmdet.structures.mask import BitmapMasks
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score
import mmdet.models  # noqa
import rsprompter  # noqa
from data.ue_dual_stream_dataset import UEDualStreamDataset
from probe_fusion_boundary_confirmation import (attach_dual_sample, frozen_features,
    has_non_degenerate_boundary_label, parameter_hash, typed_hash)
if "DetDataPreprocessor" not in MMENGINE_MODELS: MMENGINE_MODELS.register_module(name="DetDataPreprocessor",module=DetDataPreprocessor)
FIT_N, SEED, BOOT=80,20260931,10000

def parse():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--out",type=Path,required=True); p.add_argument("--device",default="cuda")
 p.add_argument("--single-ckpt",type=Path,default=Path("/data1/wangcheng/checkpoint/worktrees_dual-stream-dev_work_dirs/ue/ue_single_c5v2_control_tr128_va259_s44_r1/best_model_epoch86.pth")); p.add_argument("--dual-ckpt",type=Path,default=Path("/data1/wangcheng/checkpoint/worktrees_dual-stream-dev_work_dirs/ue/ue_dual_c5v2_p2p3_emb_tr128_va259_s44_r1/best_model_epoch85.pth")); p.add_argument("--config",type=Path,default=Path("configs/ue_dual_stream_p2p3_sparse.py")); p.add_argument("--data-root",type=Path,default=Path("/data1/wangcheng/dataset/ue_coco_with_uav")); p.add_argument("--target",choices=("iou","boundary_f1"),default="iou"); p.add_argument("--experiment-id",default="MHBEV-FUS-20260920-P8"); return p.parse_args()
def h(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def pred_digest(o):
 x=o[0].pred_instances; return {k:typed_hash(getattr(x,k)) for k in ("bboxes","scores","labels","masks")}
def iou(a,b): return float(np.logical_and(a,b).sum()/max(np.logical_or(a,b).sum(),1))
def boundary_f1(a,b,tol=3):
 k=np.ones((3,3),np.uint8); edge=lambda x:(x.astype(np.uint8)^cv2.erode(x.astype(np.uint8),k))>0
 pa,ga=edge(a),edge(b); pd=cv2.dilate(pa.astype(np.uint8),np.ones((2*tol+1,)*2,np.uint8))>0; gd=cv2.dilate(ga.astype(np.uint8),np.ones((2*tol+1,)*2,np.uint8))>0
 p=(pa&gd).sum()/max(pa.sum(),1); r=(ga&pd).sum()/max(ga.sum(),1); return float(2*p*r/max(p+r,1e-9))
def pear(y,t): return float(np.corrcoef(y,t)[0,1]) if len(t)>=2 and np.std(y)>0 and np.std(t)>0 else 0.0
def boot(x):
 r=np.random.RandomState(SEED); z=x[r.randint(0,len(x),(BOOT,len(x)))].mean(1); return [float(v) for v in np.percentile(z,[2.5,97.5])]

@torch.no_grad()
def rows(model,item,dev,target_name):
 # Prediction is exactly deployed and untouched; DBEV extraction is a separate read-only pass.
 sm=attach_dual_sample(item,dev); data=model.data_preprocessor({"inputs":[item["img"].to(dev)],"data_samples":[sm]},training=False); inp=data["inputs"]; inp=inp[0] if isinstance(inp,(list,tuple)) else inp
 if data["data_samples"][0].get("proposals",None) is not None: raise RuntimeError("P8 received supplied proposals")
 out=model.predict(inp,data["data_samples"],rescale=False)
 if data["data_samples"][0].get("proposals",None) is not None: raise RuntimeError("P8 predict wrote proposals")
 ins=out[0].pred_instances; feat=frozen_features(model,item,dev); sat=feat["sat"].reshape(128,128,512); cor=feat["correct"].reshape(128,128,1024); mis=feat["mis"].reshape(128,128,1024)
 ans=[]
 if len(ins)==0: return ans,pred_digest(out),h(feat["correct"])
 masks=ins.masks.to_ndarray() if isinstance(ins.masks,BitmapMasks) else ins.masks.detach().cpu().numpy(); boxes=ins.bboxes.detach().cpu().numpy(); scores=ins.scores.detach().cpu().numpy()
 for pm,box,score in zip(masks,boxes,scores):
  if score<.5: continue
  x1,y1,x2,y2=np.clip(np.round(box*.125).astype(int),0,128); x2=max(x2,x1+1); y2=max(y2,y1+1)
  best=max(item["gt_masks"],key=lambda g:iou(pm,g),default=None)
  target=0. if best is None else (boundary_f1(pm,best) if target_name=="boundary_f1" else iou(pm,best))
  ans.append((sat[y1:y2,x1:x2].mean((0,1)),cor[y1:y2,x1:x2].mean((0,1)),mis[y1:y2,x1:x2].mean((0,1)),target))
 return ans,pred_digest(out),h(feat["correct"])

def main():
 a=parse(); out=a.out.resolve(); out.mkdir(parents=True,exist_ok=False); dev=torch.device(a.device); torch.manual_seed(SEED); np.random.seed(SEED)
 cfg=Config.fromfile(str(a.config)); model=MODELS.build(cfg.model); d=torch.load(a.dual_ckpt,map_location="cpu",weights_only=False); d=d.get("model",d.get("state_dict",d)); z=model.load_state_dict(d,strict=False)
 if z.missing_keys or z.unexpected_keys: raise RuntimeError("dual checkpoint contract")
 s=torch.load(a.single_ckpt,map_location="cpu",weights_only=False); s=s.get("model",s.get("state_dict",s)); z=model.load_state_dict(s,strict=False); bad=[x for x in z.missing_keys if not x.startswith(("drone_encoder.","geometry_bev.","p2p3_gate.","embedding_gate."))]
 if bad or z.unexpected_keys: raise RuntimeError("single overlay contract")
 model=model.to(dev).eval(); before=parameter_hash(model); tr=UEDualStreamDataset(a.data_root,"train",image_size=(1024,1024),train_subset=.1797752808988764,seed=44,load_uav=True); va=UEDualStreamDataset(a.data_root,"val",image_size=(1024,1024),load_uav=True)
 xs={"sat":[],"correct":[],"mis":[]}; yy=[]; fit=[]
 for i,sid in enumerate(tr.ids):
  # Match the registered E5/P8 train-valid eligibility before feature extraction:
  # frozen_features constructs a P3 boundary label and correctly rejects degenerate unions.
  if not has_non_degenerate_boundary_label(tr[i]["gt_masks"]): continue
  r,_,_=rows(model,tr[i],dev,a.target)
  if r: 
   for s,c,m,t in r: xs["sat"].append(s); xs["correct"].append(np.r_[s,c]); xs["mis"].append(np.r_[s,m]); yy.append(t)
   fit.append(sid)
  if len(fit)>=FIT_N: break
 if len(fit)!=FIT_N: raise RuntimeError("P8 could not form 80 train ROI scenes")
 arms={};
 for k in xs: arms[k]=(StandardScaler().fit(np.asarray(xs[k])),None)
 y=np.asarray(yy)
 for k,(sc,_) in arms.items(): arms[k]=(sc,Ridge(alpha=1.,solver="lsqr",tol=1e-10,max_iter=20000).fit(sc.transform(np.asarray(xs[k])),y))
 per={k:[] for k in arms}; parity=[]; repeat=[]; counts=[]; global_pred={k:[] for k in arms}; global_target=[]
 for i,sid in enumerate(va.ids):
  r,dg,fh=rows(model,va[i],dev,a.target); parity.append({"scene_id":sid,"digest":dg}); rr,_,fh2=rows(model,va[i],dev,a.target)
  if fh!=fh2: raise RuntimeError(f"feature repeat hash failed {sid}")
  repeat.append(fh); counts.append(len(r)); target=np.asarray([q[3] for q in r])
  for k,(sc,reg) in arms.items():
   x=np.asarray([q[0] if k=="sat" else np.r_[q[0],q[1] if k=="correct" else q[2]] for q in r]) if r else np.empty((0,512 if k=="sat" else 1536))
   pred=reg.predict(sc.transform(x)) if len(r) else np.empty(0); per[k].append(pear(pred,target) if len(r) else 0.); global_pred[k].extend(pred.tolist())
  global_target.extend(target.tolist())
 if parameter_hash(model)!=before: raise RuntimeError("model state changed")
 gt=np.asarray(global_target); bad=(gt<.60).astype(int)
 risk={k:dict(auroc=float(roc_auc_score(bad,-np.asarray(v))) if bad.min()!=bad.max() else None,auprc=float(average_precision_score(bad,-np.asarray(v))) if bad.min()!=bad.max() else None) for k,v in global_pred.items()}
 result={"manifest":{"experiment":a.experiment_id,"target":a.target,"seed":SEED,"fit_scene_ids":fit,"state_hash":before,"val_count":len(va)},"mean_r":{k:float(np.mean(v)) for k,v in per.items()},"risk_metrics":{"bad_threshold":.60,"bad_prevalence":float(bad.mean()),"arms":risk},"gates":{"correct_minus_sat_ci95":boot(np.asarray(per["correct"])-np.asarray(per["sat"])),"correct_minus_mis_ci95":boot(np.asarray(per["correct"])-np.asarray(per["mis"])),"parity_count":len(parity),"repeat_hash_count":len(repeat),"state_unchanged":True},"per_scene_r":per,"roi_counts":counts,"parity":parity}
 (out/"summary.json").write_text(json.dumps(result,indent=1)); print(json.dumps({k:result[k] for k in ("mean_r","gates")},indent=2),flush=True)
if __name__=="__main__": main()
