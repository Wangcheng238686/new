"""Trusted SAT prefix + low-priority independent hypotheses, fixed val pilot.

Compares real geometry, constant fields, and SAT's own rejected RoI boxes.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='2'
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from mmcv.ops import batched_nms
from mmengine.structures import InstanceData
from mmdet.structures.bbox import bbox_overlaps
from ue_geometry_proposal_consume_20261024 import arrays,PILOT,REF,d

OUT=Path('/data/wangcheng/checkpoint/ue_geometry_score_band_20261024')


def main():
    p=argparse.ArgumentParser();p.add_argument('--content',choices=['geometry','constant','satellite'],required=True);p.add_argument('--gate',action='store_true');a=p.parse_args()
    dest=OUT/a.content/('gate' if a.gate else 'val');dest.mkdir(parents=True,exist_ok=True)
    if not a.gate:assert json.loads((OUT/a.content/'gate/identity.json').read_text())['complete']
    torch.cuda.set_per_process_memory_fraction(.23);d.common.seed(44);model,_=d.build()
    model.rpn_head.test_cfg.nms_pre=2000
    reference=json.loads(REF.read_text())
    proposals=json.loads((PILOT/a.content/'val_records.json').read_text()) if a.content!='satellite' else None
    ds=d.common.MinimalSatelliteDataset(d.DATA,'val',augment=False)
    rows=[];gt_all=[];dt_all=[];metas=[]
    with torch.inference_mode():
        for i in range(2 if a.gate else len(ds)):
            ref=reference[i];assert ref['image_id']==ds.samples[i]['image_id']
            batch=d.common.rtmdet_collate_fn([ds[i]]);sample=d.common.processed(model,batch)
            f=model.extract_feat(sample['inputs']);meta=sample['data_samples'][0].metainfo
            cls,reg=model.rpn_head(f[0]);props=model.rpn_head.predict_by_feat(cls,reg,batch_img_metas=[meta],rescale=False)
            head=model.roi_head;xx=head._inject_extra_pe(f[0]);cfg=head.test_cfg
            sat=head.predict_bbox(xx,[meta],props,cfg,rescale=False)[0]
            for k in ('bboxes','scores','labels'):
                v=getattr(sat,k);assert torch.equal(v,torch.tensor(ref['dt'][k],device='cuda',dtype=v.dtype))
            assert bool((sat.scores>cfg.score_thr).all()) and cfg.score_thr>0
            gt=d.common.gt_arrays(sample['data_samples'][0].gt_instances);assert d.common.serialized(gt)==ref['gt']
            if proposals is not None:
                candidate=proposals[i];assert candidate['image_id']==ref['image_id']
                boxes=torch.tensor(candidate['boxes'],device='cuda',dtype=torch.float32).reshape(-1,4)
                confidence=torch.tensor(candidate['scores'],device='cuda',dtype=torch.float32)
            else:
                # A strong SAT-only control: entire original post-RPN pool,
                # including boxes rejected by foreground threshold/NMS.
                boxes=props[0].bboxes
                rois=torch.cat((boxes.new_zeros((len(boxes),1)),boxes),1)
                br=head._bbox_forward(xx,rois)
                raw=head.bbox_head._predict_by_feat_single(rois,br['cls_score'],br['bbox_pred'],meta,False,None)
                boxes=raw.bboxes;confidence=raw.scores[:,0]
            proposed=len(boxes)
            valid=torch.isfinite(boxes).all(1)&(boxes[:,2:]>boxes[:,:2]).all(1)&torch.isfinite(confidence)&(confidence>0)
            if len(sat) and len(boxes):valid &= bbox_overlaps(boxes,sat.bboxes).max(1).values<=cfg.nms.iou_threshold
            boxes,confidence=boxes[valid],confidence[valid]
            after_novelty=len(boxes)
            if len(boxes):
                dets,_=batched_nms(boxes,confidence,torch.zeros(len(boxes),device='cuda',dtype=torch.long),cfg.nms)
                dets=dets[:min(250,max(0,250-len(sat)))]
                boxes,confidence=dets[:,:4],dets[:,4]
            scores=cfg.score_thr*confidence.clamp(0,1)
            assert not len(scores) or bool((scores<=cfg.score_thr).all())
            dt=arrays(ref['dt'])
            if len(boxes):
                inst=InstanceData(bboxes=boxes,scores=scores,labels=torch.zeros(len(boxes),device='cuda',dtype=torch.long))
                pred=head.predict_mask(xx,[meta],[inst],rescale=True,image_embeddings=f[1],image_positional_embeddings=f[2],high_res_features=f[3])[0]
                extra=d.common.gt_arrays(pred)
                for k in ('bboxes','labels','scores'):dt[k]=np.concatenate((dt[k],extra[k]))
                dt['rles']+=extra['rles'];dt['mask_count']+=extra['mask_count'];dt['mask_fill_sum']+=extra['mask_fill_sum']
            assert len(dt['scores'])<=250
            for k in ('bboxes','scores','labels'):assert np.array_equal(dt[k][:len(sat)],arrays(ref['dt'])[k])
            assert dt['rles'][:len(sat)]==ref['dt']['rles']
            gt_all.append(gt);dt_all.append(dt);metas.append(batch['img_metas'][0])
            rows.append(dict(image_id=ref['image_id'],gt=d.common.serialized(gt),dt=d.common.serialized(dt),admission=dict(proposed=proposed,after_novelty=after_novelty,accepted=len(boxes),original=len(sat))))
            d.common.dump(dest/'progress.json',dict(images=i+1,pid=__import__('os').getpid()))
            if (i+1)%25==0:print('EVAL',a.content,i+1,flush=True)
    assert d.common.state_digest(model)==d.FROZEN_SHA
    if not a.gate:
        cg,cd=d.common.build_coco_gt_and_dt(gt_all,dt_all,metas,score_key='scores');metrics={}
        for kind in ('segm','bbox'):metrics.update(d.common.run_coco_eval(cg,cd,kind,max_dets=[1,10,250]))
        original=json.loads(REF.with_name('metrics.json').read_text())
        for k in ['segm/mAP','bbox/mAP']:assert metrics[k]>=original[k]-1e-12,'Trusted-prefix nondecrease violated'
        d.common.dump(dest/'metrics.json',metrics);print('METRICS',json.dumps(metrics),flush=True)
    d.common.dump(dest/'records.json',rows)
    d.common.dump(dest/'identity.json',dict(complete=True,images=len(rows),content=a.content,split='val',fp32=True,max_dets=250,
        source_sha256=d.common.file_sha(__file__),baseline_sha256=d.common.file_sha(REF),
        original_satellite_detections_and_masks_exact=True,satellite_state_unchanged=True,
        original_score_threshold=float(cfg.score_thr),score_rule='extra_score=original_threshold*head_confidence; originals all > threshold',
        peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
        scope='Conservative trusted-prefix hypothesis supplement. AP nondecrease follows from original rank/matches preserved, not itself UAV evidence. '
              'Real-vs-constant and real-vs-SAT-only controls required. SAT-only has a larger input pool, up to same final250 budget. '
              'No frozen SAT classifier veto on geometry; geometry boxes unregressed; no test evaluation.'))


if __name__=='__main__':main()
