#!/usr/bin/env python3
"""P10 multi-height P3 DBEV information probe; Governance Q."""
from __future__ import annotations
import argparse,json,os,sys,hashlib
from pathlib import Path
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="1"
os.environ.setdefault("EXPLICIT_PROMPT_MODE","points_box_dense");os.environ.setdefault("P2_BOUNDARY_REFINER_ENABLED","1");os.environ.setdefault("SAM_IMAGE_EMBED_STRIDE","16");os.environ.setdefault("SHAPE_DENSE_TRANSFORM","raw_logits");os.environ.setdefault("SHAPE_DENSE_DETACH","1");os.environ.setdefault("FINAL_MASK_COORDINATE_MODE","roi_local");os.environ.setdefault("SAM2_CKPT","/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt");os.environ.setdefault("UE_DUAL_INJECT_EMBEDDING","1")
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,torch,torch.nn.functional as F
from mmengine.config import Config
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmdet.registry import MODELS
import mmdet.models,rsprompter # noqa
from data.ue_dual_stream_dataset import UEDualStreamDataset
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from probe_fusion_boundary_confirmation import (frozen_features,world_grid,rotate_points_90,has_non_degenerate_boundary_label,parameter_hash)
if "DetDataPreprocessor" not in MMENGINE_MODELS:MMENGINE_MODELS.register_module(name="DetDataPreprocessor",module=DetDataPreprocessor)
H=(0.,4.,7.,10.,15.,22.,32.); SEED=20260932; FIT=80; PER=1500; BOOT=10000
def args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--out",type=Path,required=True);p.add_argument("--device",default="cuda");p.add_argument("--heights",type=float,nargs="+",default=list(H));p.add_argument("--aggregation",choices=("mean","consistency_select"),default="mean");p.add_argument("--seed",type=int,default=SEED);p.add_argument("--experiment-id",default="MHBEV-FUS-20260920-P10");p.add_argument("--single",type=Path,default=Path("/data1/wangcheng/checkpoint/worktrees_dual-stream-dev_work_dirs/ue/ue_single_c5v2_control_tr128_va259_s44_r1/best_model_epoch86.pth"));p.add_argument("--dual",type=Path,default=Path("/data1/wangcheng/checkpoint/worktrees_dual-stream-dev_work_dirs/ue/ue_dual_c5v2_p2p3_emb_tr128_va259_s44_r1/best_model_epoch85.pth"));p.add_argument("--config",type=Path,default=Path("configs/ue_dual_stream_p2p3_sparse.py"));p.add_argument("--data",type=Path,default=Path("/data1/wangcheng/dataset/ue_coco_with_uav"));return p.parse_args()
def boot(x):
 r=np.random.RandomState(SEED);z=x[r.randint(0,len(x),(BOOT,len(x)))].mean(1);return [float(v) for v in np.percentile(z,[2.5,97.5])]
def pear(a,b):return float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else 0.
@torch.no_grad()
def mh(model,item,dev,aggregation):
 drone=item["drone_images"][None].to(dev);drone=(drone-drone.new_tensor([.485,.456,.406])[None,None,:,None,None])/drone.new_tensor([.229,.224,.225])[None,None,:,None,None]
 _,fpn=model.drone_encoder(drone.reshape(5,*drone.shape[2:]));fpn=tuple(sorted(tuple(fpn[-4:]),key=lambda x:x.shape[-2]*x.shape[-1],reverse=True)); intr=item["intrinsics"].to(dev).float();ext=item["extrinsics"].to(dev).float();cen=item["scene_center"].to(dev).float()
 base=world_grid(128,128,cen,dev); outs=[]; covers=[]
 for rotated in (False,True):
  hs=[]; cs=[]; best=None; bestscore=None; bestidx=None
  for z in H:
   pts=base.clone();pts[:,2]=z*100.;pts=rotate_points_90(pts,cen) if rotated else pts; per=[];valids=[]
   for v in range(5):
    cam=pts@ext[v].T;uv=(cam[:,:3]@intr[v].T)[:,:2]/cam[:,2:3].clamp_min(1e-6);valid=(cam[:,2]>1e-4)&(uv[:,0]>=0)&(uv[:,0]<512)&(uv[:,1]>=0)&(uv[:,1]<512);grid=uv/512*2-1;grid[~valid]=0.;per.append(torch.cat([F.grid_sample(feat[v:v+1].float(),grid[None,:,None,:],align_corners=False)[0,:,:,0] for feat in fpn]));valids.append(valid)
   desc=torch.stack(per);valid=torch.stack(valids);count=valid.sum(0);avg=(desc*valid[:,None]).sum(0)/count.clamp_min(1)[None];norm=desc/desc.norm(dim=1,keepdim=True).clamp_min(1e-6);s=(norm*valid[:,None]).sum(0);cons=(s.square().sum(0)-count)/(count*(count-1)).clamp_min(1);cons[count<2]=-float("inf");cs.append(float((count>=2).float().mean()))
   if aggregation=="mean":hs.append(avg.T)
   else:
    if bestscore is None:bestscore=cons;best=avg;bestidx=torch.zeros_like(count,dtype=torch.int64)
    else:
     take=cons>bestscore;best[:,take]=avg[:,take];bestscore[take]=cons[take];bestidx[take]=len(cs)-1
  feat=torch.stack(hs).mean(0) if aggregation=="mean" else best.T; meta={"support":cs,"height_hist":torch.bincount(bestidx,minlength=len(H)).cpu().tolist() if aggregation!="mean" else None}
  outs.append((feat.cpu().numpy().astype("float32"),meta))
 return outs[0],outs[1]
def main():
 global H,SEED
 a=args();H=tuple(a.heights);SEED=a.seed;out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);dev=torch.device(a.device);torch.manual_seed(SEED);np.random.seed(SEED);cfg=Config.fromfile(str(a.config));model=MODELS.build(cfg.model);d=torch.load(a.dual,map_location="cpu",weights_only=False);d=d.get("model",d.get("state_dict",d));z=model.load_state_dict(d,strict=False);assert not z.missing_keys and not z.unexpected_keys;s=torch.load(a.single,map_location="cpu",weights_only=False);s=s.get("model",s.get("state_dict",s));z=model.load_state_dict(s,strict=False);assert not [x for x in z.missing_keys if not x.startswith(("drone_encoder.","geometry_bev.","p2p3_gate.","embedding_gate."))] and not z.unexpected_keys;model=model.to(dev).eval();state=parameter_hash(model);tr=UEDualStreamDataset(a.data,"train",image_size=(1024,1024),train_subset=.1797752808988764,seed=44,load_uav=True);va=UEDualStreamDataset(a.data,"val",image_size=(1024,1024),load_uav=True);X={k:[] for k in ("h0","mh","mis")};Y=[];ids=[]
 for i,sid in enumerate(tr.ids):
  if not has_non_degenerate_boundary_label(tr[i]["gt_masks"]):continue
  f=frozen_features(model,tr[i],dev);(m,_),(mis,_)=mh(model,tr[i],dev,a.aggregation);q=np.random.RandomState(SEED+i).choice(128*128,PER,False);X["h0"].append(np.c_[f["sat"][q],f["correct"][q]]);X["mh"].append(np.c_[f["sat"][q],m[q]]);X["mis"].append(np.c_[f["sat"][q],mis[q]]);Y.append(f["label"][q]);ids.append(sid)
  if len(ids)==FIT:break
 sc={k:StandardScaler().fit(np.concatenate(v)) for k,v in X.items()};y=np.concatenate(Y);rg={k:Ridge(alpha=1.,solver="lsqr",tol=1e-10,max_iter=20000).fit(sc[k].transform(np.concatenate(X[k])),y) for k in X};per={k:[] for k in X};tele=[]
 for i,sid in enumerate(va.ids):
  f=frozen_features(model,va[i],dev);(m,c),(mis,cm)=mh(model,va[i],dev,a.aggregation);arr={"h0":np.c_[f["sat"],f["correct"]],"mh":np.c_[f["sat"],m],"mis":np.c_[f["sat"],mis]};[per[k].append(pear(rg[k].predict(sc[k].transform(arr[k])),f["label"])) for k in X];tele.append({"scene_id":sid,"h_select":c,"mis_select":cm})
 if parameter_hash(model)!=state:raise RuntimeError("state changed")
 r={"manifest":{"experiment":a.experiment_id,"aggregation":a.aggregation,"heights_m":H,"fit_ids":ids,"state_hash":state},"mean_r":{k:float(np.mean(v)) for k,v in per.items()},"gates":{"mh_minus_h0_ci95":boot(np.array(per["mh"])-np.array(per["h0"])),"mh_minus_mis_ci95":boot(np.array(per["mh"])-np.array(per["mis"])),"state_unchanged":True},"per_scene_r":per,"telemetry":tele};(out/"summary.json").write_text(json.dumps(r,indent=1));print(json.dumps({"mean_r":r["mean_r"],"gates":r["gates"]},indent=2))
if __name__=="__main__":main()
