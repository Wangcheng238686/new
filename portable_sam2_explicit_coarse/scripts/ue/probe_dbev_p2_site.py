#!/usr/bin/env python3
"""P6: frozen direct-DBEV injection-site probe (registered in Governance M).

The train-only ridge turns SAT-P3+direct-DBEV into a boundary field.  At val it
is injected after the frozen neck's P2, before RPN/ROI/decoder.  No val label is
used to form the field.  Arms are zero/correct/90-degree-rotated DBEV.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
os.environ.setdefault("EXPLICIT_PROMPT_MODE", "points_box_dense")
os.environ.setdefault("P2_BOUNDARY_REFINER_ENABLED", "1")
os.environ.setdefault("SAM_IMAGE_EMBED_STRIDE", "16")
os.environ.setdefault("SHAPE_DENSE_TRANSFORM", "raw_logits")
os.environ.setdefault("SHAPE_DENSE_DETACH", "1")
os.environ.setdefault("FINAL_MASK_COORDINATE_MODE", "roi_local")
os.environ.setdefault("SAM2_CKPT", "/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt")
os.environ.setdefault("UE_DUAL_INJECT_EMBEDDING", "1")
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.structures.mask import BitmapMasks
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import mmdet.models  # noqa
import rsprompter  # noqa
from data.ue_dual_stream_dataset import UEDualStreamDataset
from probe_fusion_boundary_confirmation import (attach_dual_sample, frozen_features,
    has_non_degenerate_boundary_label, parameter_hash, typed_hash)
if "DetDataPreprocessor" not in MMENGINE_MODELS:
    MMENGINE_MODELS.register_module(name="DetDataPreprocessor", module=DetDataPreprocessor)

SEED, FIT_N, FIT_PER_SCENE, CALIB_N, BOOT_N = 20260929, 80, 1500, 20, 10000
ALPHA = 0.5

def args_():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--single-ckpt", type=Path, default=Path("work_dirs/ue/ue_single_c5v2_control_tr128_va259_s44_r1/best_model_epoch86.pth"))
    p.add_argument("--dual-ckpt", type=Path, default=Path("work_dirs/ue/ue_dual_c5v2_p2p3_emb_tr128_va259_s44_r1/best_model_epoch85.pth"))
    p.add_argument("--config", type=Path, default=Path("configs/ue_dual_stream_p2p3_sparse.py"))
    p.add_argument("--data-root", type=Path, default=Path("/data1/wangcheng/dataset/ue_coco_with_uav"))
    p.add_argument("--out", type=Path, required=True); p.add_argument("--device", default="cuda")
    p.add_argument("--field-source", choices=("sat_dbev", "dbev_only"), default="sat_dbev")
    p.add_argument("--experiment-id", default="MHBEV-FUS-20260919-P6")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--parity-scenes", type=int, default=4)
    return p.parse_args()

def union_f1(pred, gt, tol=3):
    ker=np.ones((3,3),np.uint8); b=lambda x:(x.astype(np.uint8)^cv2.erode(x.astype(np.uint8),ker))>0
    pb,gb=b(pred),b(gt); pd=cv2.dilate(pb.astype(np.uint8),np.ones((2*tol+1,)*2,np.uint8))>0; gd=cv2.dilate(gb.astype(np.uint8),np.ones((2*tol+1,)*2,np.uint8))>0
    pr=(pb&gd).sum()/max(pb.sum(),1); re=(gb&pd).sum()/max(gb.sum(),1)
    return float(2*pr*re/max(pr+re,1e-9))

def rle(m):
    x=maskUtils.encode(np.asfortranarray(m.astype(np.uint8))); x["counts"]=x["counts"].decode(); return x
def digest(out):
    ins=out[0].pred_instances
    return {k:typed_hash(getattr(ins,k)) for k in ("bboxes","scores","labels","masks")}
def boot(x):
    rng=np.random.RandomState(SEED); ind=rng.randint(0,len(x),(BOOT_N,len(x))); means=x[ind].mean(1)
    return [float(np.percentile(means,2.5)),float(np.percentile(means,97.5))]

def main():
    global SEED
    a=args_(); SEED=a.seed; out=a.out.resolve(); out.mkdir(parents=True,exist_ok=True)
    # Some launchers atomically materialize an empty artifact directory.  Permit
    # that case, but use an exclusive lock so a second process can never share it.
    if any(out.iterdir()):
      raise RuntimeError(f"refuse non-empty P6 artifact directory: {out}")
    (out / ".running").open("x").close(); dev=torch.device(a.device)
    torch.manual_seed(SEED); np.random.seed(SEED)
    cfg=Config.fromfile(str(a.config)); model=MODELS.build(cfg.model)
    d=torch.load(a.dual_ckpt,map_location="cpu",weights_only=False); d=d.get("model",d.get("state_dict",d)); z=model.load_state_dict(d,strict=False)
    if z.missing_keys or z.unexpected_keys: raise RuntimeError(f"dual strict contract {z}")
    s=torch.load(a.single_ckpt,map_location="cpu",weights_only=False); s=s.get("model",s.get("state_dict",s)); z=model.load_state_dict(s,strict=False)
    bad=[k for k in z.missing_keys if not k.startswith(("drone_encoder.","geometry_bev.","p2p3_gate.","embedding_gate."))]
    if bad or z.unexpected_keys: raise RuntimeError(f"single overlay contract bad={bad} unexpected={z.unexpected_keys}")
    model=model.to(dev).eval(); state0=parameter_hash(model)
    tr=UEDualStreamDataset(a.data_root,"train",image_size=(1024,1024),train_subset=.1797752808988764,seed=44,load_uav=True)
    va=UEDualStreamDataset(a.data_root,"val",image_size=(1024,1024),load_uav=True)
    if len(tr)!=128 or len(va)!=259: raise RuntimeError("fixed split cardinality failed")
    # Fit exactly once on train.  The P7 source deliberately omits SAT so that
    # correct and rotated DBEV do not share a satellite-derived boundary field.
    xs=[]; ys=[]; fit_ids=[]
    with torch.no_grad():
      for i,sid in enumerate(tr.ids):
        it=tr[i]
        if not has_non_degenerate_boundary_label(it["gt_masks"]): continue
        f=frozen_features(model,it,dev); rng=np.random.RandomState(SEED+i); q=rng.choice(128*128,FIT_PER_SCENE,replace=False)
        source=f["correct"][q] if a.field_source=="dbev_only" else np.concatenate((f["sat"][q],f["correct"][q]),1)
        xs.append(source); ys.append(f["label"][q]); fit_ids.append(sid)
        if len(fit_ids)==FIT_N: break
    if len(fit_ids)!=FIT_N: raise RuntimeError("could not form registered fit set")
    scaler=StandardScaler(); ridge=Ridge(alpha=1.,solver="lsqr",tol=1e-10,max_iter=20000).fit(scaler.fit_transform(np.concatenate(xs)),np.concatenate(ys))
    # train-only P2 channel signs, same P5 construction.
    signs=[]
    for i in range(CALIB_N):
      it=tr[i]; sample=attach_dual_sample(it,dev); dat=model.data_preprocessor({"inputs":[it["img"].to(dev)],"data_samples":[sample]},training=False); inp=dat["inputs"]; inp=inp[0] if isinstance(inp,(tuple,list)) else inp
      p2=model.neck(model.backbone(inp)[1])[0][0].float().reshape(512,-1).detach().cpu().numpy()
      # E5's registered readout label lives at P3=128²; P6 signs are for
      # P2=256², so resample the train-only label before channel correlation.
      y128=frozen_features(model,it,dev)["label"].reshape(128,128)
      y=cv2.resize(y128,(256,256),interpolation=cv2.INTER_LINEAR).reshape(-1)
      p2-=p2.mean(1,keepdims=True); signs.append((p2@y)/(np.linalg.norm(p2,axis=1)*np.linalg.norm(y)+1e-9))
    signs=np.sign(np.mean(signs,0)).astype("float32"); signs[signs==0]=1.; signs=torch.from_numpy(signs).to(dev)[None,:,None,None]
    active={"kind":"zero","field":None,"norm":[]}; original=model.neck.forward
    def patched(hidden):
      x=original(hidden)
      if active["field"] is None: return x
      p2=x[0].float(); field=active["field"].to(p2); sigma=p2.flatten(2).std(2).mean()
      delta=ALPHA*sigma*field*signs; active["norm"].append(float(delta.norm()/p2.norm().clamp_min(1e-9)))
      return (p2.add(delta).to(x[0].dtype),*x[1:])
    model.neck.forward=patched
    # Registered zero-arm parity: the hook exists but strict zero injection must
    # reproduce the untouched deployed forward on fixed validation scenes.
    zero_parity=[]
    if not 1 <= a.parity_scenes <= len(va): raise ValueError("parity-scenes must be within val")
    for i in range(a.parity_scenes):
      it=va[i]; sm=attach_dual_sample(it,dev); dat=model.data_preprocessor({"inputs":[it["img"].to(dev)],"data_samples":[sm]},training=False); inp=dat["inputs"]; inp=inp[0] if isinstance(inp,(tuple,list)) else inp
      model.neck.forward=original
      with torch.no_grad(): ref=digest(model.predict(inp,dat["data_samples"],rescale=False))
      model.neck.forward=patched
      sm=attach_dual_sample(it,dev); dat=model.data_preprocessor({"inputs":[it["img"].to(dev)],"data_samples":[sm]},training=False); inp=dat["inputs"]; inp=inp[0] if isinstance(inp,(tuple,list)) else inp
      with torch.no_grad(): got=digest(model.predict(inp,dat["data_samples"],rescale=False))
      if ref!=got: raise RuntimeError(f"zero-injection parity failed on {va.ids[i]}")
      zero_parity.append({"scene_id":va.ids[i],"equal":True})
    images=[]; anns=[]; aid=1
    for i in range(len(va)):
      it=va[i]; images.append(dict(id=i,file_name=va.ids[i],height=1024,width=1024))
      for m in it["gt_masks"]:
        yy,xx=np.where(m); anns.append(dict(id=aid,image_id=i,category_id=1,segmentation=rle(m),bbox=[float(xx.min()),float(yy.min()),float(xx.max()-xx.min()+1),float(yy.max()-yy.min()+1)],area=int(m.sum()),iscrowd=0)); aid+=1
    (out/"gt.json").write_text(json.dumps(dict(images=images,annotations=anns,categories=[dict(id=1,name="building")]))); gt=COCO(str(out/"gt.json"))
    result={"manifest":{"experiment":a.experiment_id,"field_source":a.field_source,"alpha":ALPHA,"seed":SEED,"parity_scenes":a.parity_scenes,"fit_ids":fit_ids,"state_hash":state0,"single":str(a.single_ckpt),"dual":str(a.dual_ckpt)},"arms":{}}
    zero_digests=[]
    for kind in ("zero","correct","mis"):
      active.update(kind=kind,field=None,norm=[]); b=[]; seg=[]; box=[]; bf=[]; xors=[]
      for i in range(len(va)):
        it=va[i]
        if kind!="zero":
          f=frozen_features(model,it,dev); db=f["correct"] if kind=="correct" else f["mis"]
          source=db if a.field_source=="dbev_only" else np.concatenate((f["sat"],db),1)
          q=ridge.predict(scaler.transform(source)).reshape(128,128).astype("float32"); q=(q-q.mean())/(q.std()+1e-6)
          active["field"]=F.interpolate(torch.from_numpy(q)[None,None].to(dev),size=(256,256),mode="bilinear",align_corners=False)
        sample=attach_dual_sample(it,dev); dat=model.data_preprocessor({"inputs":[it["img"].to(dev)],"data_samples":[sample]},training=False); inp=dat["inputs"]; inp=inp[0] if isinstance(inp,(tuple,list)) else inp
        with torch.no_grad(): o=model.predict(inp,dat["data_samples"],rescale=False)
        dg=digest(o)
        if kind=="zero": zero_digests.append(dg)
        elif i<4 and dg==zero_digests[i]: raise RuntimeError(f"{kind} field did not alter first-four prediction {i}")
        ins=o[0].pred_instances; pm=np.zeros((1024,1024),bool); gm=np.zeros((1024,1024),bool)
        for m in it["gt_masks"]: gm|=m.astype(bool)
        if len(ins):
          ms=ins.masks.to_ndarray() if isinstance(ins.masks,BitmapMasks) else ins.masks.cpu().numpy(); keep=ins.scores.detach().cpu().numpy()>=.5; pm=ms[keep].any(0) if keep.any() else pm
          for mm,bb,sc in zip(ms,ins.bboxes.detach().cpu().numpy(),ins.scores.detach().cpu().numpy()):
            seg.append(dict(image_id=i,category_id=1,segmentation=rle(mm),score=float(sc)))
            box.append(dict(image_id=i,category_id=1,bbox=[float(bb[0]),float(bb[1]),float(bb[2]-bb[0]),float(bb[3]-bb[1])],score=float(sc)))
        bf.append(union_f1(pm,gm)); xors.append(float((pm ^ (pm if kind=="zero" else pm)).mean()))
      def ev(data,typ):
        dt=gt.loadRes(data); e=COCOeval(gt,dt,typ); e.params.imgIds=list(range(len(va))); e.evaluate();e.accumulate();e.summarize(); return [float(x) for x in e.stats[:3]]
      result["arms"][kind]=dict(bf1=bf,mean_bf1=float(np.mean(bf)),segm=ev(seg,"segm"),bbox=ev(box,"bbox"),p2_over_base=float(np.mean(active["norm"])) if active["norm"] else 0.,prediction_count=len(seg))
      print(kind,result["arms"][kind]["mean_bf1"],result["arms"][kind]["segm"],result["arms"][kind]["bbox"],flush=True)
    if parameter_hash(model)!=state0: raise RuntimeError("model state mutated")
    z,c,m=(np.array(result["arms"][x]["bf1"]) for x in ("zero","correct","mis")); result["gates"]=dict(correct_minus_zero_ci95=boot(c-z),correct_minus_mis_ci95=boot(c-m),zero_parity=zero_parity,zero_prediction_hashes=zero_digests[:4],state_unchanged=True)
    (out/"summary.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result["gates"],indent=2),flush=True)
if __name__=="__main__": main()
