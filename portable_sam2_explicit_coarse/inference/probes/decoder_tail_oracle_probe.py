"""Frozen A0 decoder-tail PointRend-style oracle.

It never adds prompt tokens or reruns SAM2.  On the native single-mask grid it
selects only the K lowest-|logit| cells, using no GT, then (for class-aware
box-IoU matched ROIs only) writes the GT signed answer at those cells.  This
is an upper-bound diagnostic for a future point-tail classifier, not a model.
"""
from __future__ import annotations

import argparse, hashlib, json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import box_iou

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from inference.infer_from_checkpoint import (_build_data_samples, _build_loader,
    _extract_instances_numpy, _load_checkpoint, _load_model_state,
    _register_and_build, _resolve_dataset_contract, _resolve_model_config,
    _select_state_dict, _snapshot)
from utils.coco_eval_utils import VHR10_CATEGORIES, build_coco_gt_and_dt, run_coco_eval

def args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True); p.add_argument('--device',default='cuda:0')
    p.add_argument('--ks',type=int,nargs='+',default=[0,64,256,1024])
    p.add_argument('--match-iou',type=float,default=.5); p.add_argument('--logit-span',type=float,default=8.)
    p.add_argument('--max-batches',type=int,default=0); p.add_argument('--output',required=True)
    return p.parse_args()

def hupdate(h,x):
    a=np.ascontiguousarray(np.asarray(x)); h.update(str((a.dtype.str,a.shape)).encode()); h.update(a.tobytes())

def main():
    a=args(); ckpt_path=Path(a.checkpoint).resolve(); ckpt=_load_checkpoint(ckpt_path); snap=_snapshot(ckpt)
    ns=SimpleNamespace(checkpoint=str(ckpt_path),config=None,split='validation',data_root=None,ann_file=None,image_subdir=None,image_size=None,batch_size=None,sam2_repo=None,sam2_ckpt=None)
    drift=None
    try: cfg, source=_resolve_model_config(ns,snap)
    except RuntimeError as e:
        if 'does not match its saved architecture contract' not in str(e): raise
        cfg=snap['model_config']; source='embedded_config_strict_state_only_legacy_contract_drift'; drift=str(e)
    model=_register_and_build(cfg); report=_load_model_state(model,_select_state_dict(ckpt,'model'),allow_nonstrict=False)
    if report['missing_keys'] or report['unexpected_keys']: raise RuntimeError(report)
    device=torch.device(a.device); model.to(device).eval(); head=model.roi_head.mask_head
    if (head.quality_head_enabled or head.final_mask_coordinate_mode != 'full_image'
            or not head.class_agnostic or head.roi_sam_enabled):
        raise RuntimeError('tail oracle requires A0 full_image, class-agnostic, non-ROI-SAM, quality-head-off contract')
    loader=_build_loader(_resolve_dataset_contract(ns,snap),0)

    def cache(samples):
        out=[]
        for s in samples:
            ins=getattr(s,'gt_instances',None)
            out.append(None if ins is None or len(getattr(ins,'bboxes',[]))==0 else (ins.bboxes.cpu(),ins.labels.cpu(),ins.masks.to_tensor(dtype=torch.float32,device='cpu')))
        return out

    def run(k, intervene=True):
        state={'gts':None,'ids':None,'boxes':None,'labels':None,'queue':None,'cursor':0}
        stat={'matched':0,'unmatched':0,'selected':0,'errors':0,'total_rois':0}; dh=hashlib.sha256(); oh=hashlib.sha256(); gts=[]; dts=[]; metas=[]
        if not intervene:
            # Truly unpatched standard path: this is the only valid C0.
            all_gt=[]; all_dt=[]; all_meta=[]
            with torch.inference_mode():
                for bi,batch in enumerate(loader):
                    if a.max_batches and bi>=a.max_batches: break
                    imgs,samples=_build_data_samples(batch,device); proc=model.data_preprocessor({'inputs':imgs,'data_samples':samples},training=False); outs=model.predict(proc['inputs'],proc['data_samples'],rescale=False)
                    for n,o in enumerate(outs):
                        meta=batch['img_metas'][n]; all_meta.append(dict(meta)); all_gt.append(_extract_instances_numpy(proc['data_samples'][n].gt_instances,meta['img_shape'])); all_dt.append(_extract_instances_numpy(o.pred_instances if hasattr(o,'pred_instances') else o,meta['img_shape']))
            cg,cd=build_coco_gt_and_dt(all_gt,all_dt,all_meta,categories=VHR10_CATEGORIES); m=run_coco_eval(cg,cd,'segm')
            return {'k_per_roi':0,'images':len(all_meta),'segm/mAP':m['segm/mAP'],'segm/mAP_50':m['segm/mAP_50'],'segm/mAP_75':m['segm/mAP_75'],'segm/AR@100':m['segm/AR@100'],'detector_sha256':'unavailable','output_sha256':'unavailable','unhooked':True}
        original_pm=model.roi_head.predict_mask; had_pm='predict_mask' in model.roi_head.__dict__; saved_pm=model.roi_head.__dict__.get('predict_mask')
        def pm(*fa,**kw):
            rs=kw.get('results_list',fa[2]); state['queue']=torch.cat([r.labels.detach() for r in rs]) if rs else torch.zeros(0,dtype=torch.long); state['cursor']=0
            return original_pm(*fa,**kw)
        def pre(mod,fa,kw):
            boxes=kw.get('boxes'); ids=kw.get('roi_img_ids'); state['boxes']=boxes.detach().cpu() if boxes is not None else None; state['ids']=ids.detach().cpu() if ids is not None else None
            if boxes is None or state['queue'] is None: return
            e=state['cursor']+len(boxes)
            if e>len(state['queue']): raise RuntimeError('label queue underflow')
            state['labels']=state['queue'][state['cursor']:e].cpu(); state['cursor']=e
        def tail(mod,fa,kw,out):
            masks,iou,tokens=out; boxes,ids,labels=state['boxes'],state['ids'],state['labels']
            if k==0: return out
            if boxes is None or ids is None or labels is None or len(masks)!=len(boxes): raise RuntimeError('tail ROI state misalignment')
            revised=masks.clone(); H,W=masks.shape[-2:]
            for i in range(len(masks)):
                stat['total_rois']+=1; image=int(ids[i]); entry=state['gts'][image] if state['gts'] is not None and image<len(state['gts']) else None
                if entry is None: stat['unmatched']+=1; continue
                gb,gl,gm=entry; same=gl==labels[i]
                if not bool(same.any()): stat['unmatched']+=1; continue
                q=box_iou(boxes[i:i+1].float(),gb.float())[0]; q=torch.where(same,q,torch.full_like(q,-1.)); j=int(q.argmax())
                if float(q[j])<a.match_iou: stat['unmatched']+=1; continue
                stat['matched']+=1; gt=F.interpolate(gm[j:j+1,None].to(device),size=(H,W),mode='nearest')[0,0]>=.5
                z=masks[i,0]; n=min(k,z.numel()); idx=z.abs().flatten().topk(n,largest=False).indices
                target=gt.flatten()[idx]; before=(z.flatten()[idx]>=0); stat['errors']+=int((before!=target).sum()); stat['selected']+=n
                flat=revised[i,0].flatten(); flat[idx]=torch.where(target,torch.full_like(flat[idx],a.logit_span),torch.full_like(flat[idx],-a.logit_span))
            return revised,iou,tokens
        hp=head.register_forward_pre_hook(pre,with_kwargs=True); hd=head.mask_decoder.register_forward_hook(tail,with_kwargs=True); model.roi_head.predict_mask=pm
        try:
            with torch.inference_mode():
                for bi,batch in enumerate(loader):
                    if a.max_batches and bi>=a.max_batches: break
                    imgs,samples=_build_data_samples(batch,device); proc=model.data_preprocessor({'inputs':imgs,'data_samples':samples},training=False); state['gts']=cache(proc['data_samples'])
                    outputs=model.predict(proc['inputs'],proc['data_samples'],rescale=False)
                    if state['cursor']!=len(state['queue']): raise RuntimeError('label queue not fully consumed')
                    for n,o in enumerate(outputs):
                        meta=batch['img_metas'][n]; shape=meta['img_shape']; gt=_extract_instances_numpy(proc['data_samples'][n].gt_instances,shape); pred=_extract_instances_numpy(o.pred_instances if hasattr(o,'pred_instances') else o,shape); iid=int(meta.get('image_id',meta.get('scene_id',n)))
                        hupdate(dh,np.array([iid],np.int64)); [hupdate(dh,pred[x]) for x in ('bboxes','scores','labels')]; [hupdate(oh,pred[x]) for x in ('bboxes','scores','labels','masks')]
                        gts.append(gt);dts.append(pred);metas.append(dict(meta))
        finally:
            hp.remove();hd.remove();
            if had_pm:model.roi_head.predict_mask=saved_pm
            else:delattr(model.roi_head,'predict_mask')
        cg,cd=build_coco_gt_and_dt(gts,dts,metas,categories=VHR10_CATEGORIES); met=run_coco_eval(cg,cd,'segm')
        return {'k_per_roi':k,'images':len(metas),'segm/mAP':met['segm/mAP'],'segm/mAP_50':met['segm/mAP_50'],'segm/mAP_75':met['segm/mAP_75'],'segm/AR@100':met['segm/AR@100'],'detector_sha256':dh.hexdigest(),'output_sha256':oh.hexdigest(),**stat}
    standard=run(0,intervene=False); cells=[]
    for k in a.ks:
        c=run(k)
        if k==0 and float(c['segm/mAP']) != float(standard['segm/mAP']): raise RuntimeError('K=0 hook path differs from unhooked C0')
        cells.append(c); print(json.dumps(c),flush=True)
    result={'checkpoint':str(ckpt_path),'config_source':source,'contract_drift':drift,'strict_load':report,'standard':standard,'cells':cells,'note':'GT only writes labels at z-only uncertainty-selected native cells; no prompt/token/ranking change.'}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__': main()
