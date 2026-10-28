"""Frozen SAT consumes independent geometry boxes; fixed val 2x2, no training.

Preserves every original SAT detection/mask and admits only nonoverlapping extras.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='2'
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from mmcv.ops import batched_nms
from mmengine.structures import InstanceData
from mmdet.structures.bbox import bbox_overlaps
import ue_detection_attrition_probe_20261023 as d

PILOT=Path('/data/wangcheng/checkpoint/ue_geometry_proposal_pilot_20261024')
OUT=Path('/data/wangcheng/checkpoint/ue_geometry_proposal_consume_20261024')
REF=Path('/data/wangcheng/checkpoint/ue_rpn_geometry_20261023/val/budget/records.json')


def arrays(row):
    out=copy.deepcopy(row)
    for k in ('bboxes','labels','scores'):
        if k in out:out[k]=np.asarray(out[k],dtype=np.int64 if k=='labels' else np.float32)
    out['bboxes']=out['bboxes'].reshape(-1,4)
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--content',choices=['geometry','constant'],required=True)
    p.add_argument('--boxes',choices=['raw','regressed'],required=True);p.add_argument('--gate',action='store_true')
    args=p.parse_args()
    dest=OUT/f'{args.content}_{args.boxes}'/('gate' if args.gate else 'val');dest.mkdir(parents=True,exist_ok=True)
    if not args.gate:assert json.loads((OUT/f'{args.content}_{args.boxes}'/'gate/identity.json').read_text())['complete']
    torch.cuda.set_per_process_memory_fraction(.23);d.common.seed(44)
    model,_=d.build();model.rpn_head.test_cfg.nms_pre=2000
    reference=json.loads(REF.read_text());proposals=json.loads((PILOT/args.content/'val_records.json').read_text())
    result=json.loads((PILOT/args.content/'result.json').read_text())
    assert result['images']==459 and result['checkpoint_sha256']==d.common.file_sha(PILOT/args.content/'last.pth')
    ds=d.common.MinimalSatelliteDataset(d.DATA,'val',augment=False)
    nms=model.roi_head.test_cfg.nms
    assert nms.type=='nms';threshold=nms.iou_threshold
    rows=[];all_gt=[];all_dt=[];metas=[]
    with torch.inference_mode():
        for index in range(2 if args.gate else len(ds)):
            baseline=reference[index];candidate=proposals[index]
            assert baseline['image_id']==candidate['image_id']==ds.samples[index]['image_id']
            batch=d.common.rtmdet_collate_fn([ds[index]]);sample=d.common.processed(model,batch)
            features=model.extract_feat(sample['inputs']);meta=sample['data_samples'][0].metainfo
            cls,reg=model.rpn_head(features[0])
            props=model.rpn_head.predict_by_feat(cls,reg,batch_img_metas=[meta],rescale=False)
            head=model.roi_head;xx=head._inject_extra_pe(features[0])
            sat=head.predict_bbox(xx,[meta],props,head.test_cfg,rescale=False)[0]
            for key in ('bboxes','scores','labels'):
                value=getattr(sat,key);expected=torch.tensor(baseline['dt'][key],device='cuda',dtype=value.dtype)
                assert torch.equal(value,expected),(index,key,'original SAT detection differs')
            gt=d.common.gt_arrays(sample['data_samples'][0].gt_instances)
            assert d.common.serialized(gt)==baseline['gt'],'GT differs'
            boxes=torch.tensor(candidate['boxes'],device='cuda',dtype=torch.float32).reshape(-1,4)
            input_count=len(boxes)
            if len(boxes):
                rois=torch.cat((boxes.new_zeros((len(boxes),1)),boxes),1)
                br=head._bbox_forward(xx,rois)
                decoded=head.bbox_head._predict_by_feat_single(rois,br['cls_score'],br['bbox_pred'],meta,False,None)
                scores=decoded.scores[:,0]
                if args.boxes=='regressed':boxes=decoded.bboxes
                valid=(scores>head.test_cfg.score_thr)&torch.isfinite(boxes).all(1)&(boxes[:,2:]>boxes[:,:2]).all(1)
                if len(sat):valid &= bbox_overlaps(boxes,sat.bboxes).max(1).values<=threshold
                boxes,scores=boxes[valid],scores[valid]
            else:scores=boxes.new_empty(0)
            before_nms=len(boxes)
            if len(boxes):
                dets,_=batched_nms(boxes,scores,torch.zeros_like(scores,dtype=torch.long),nms)
                dets=dets[:max(0,250-len(sat))];boxes,scores=dets[:,:4],dets[:,4]
            extra=InstanceData(bboxes=boxes,scores=scores,labels=torch.zeros(len(boxes),device='cuda',dtype=torch.long))
            dt=arrays(baseline['dt'])
            if len(boxes):
                pred=head.predict_mask(xx,[meta],[extra],rescale=True,image_embeddings=features[1],
                    image_positional_embeddings=features[2],high_res_features=features[3])[0]
                added=d.common.gt_arrays(pred)
                for key in ('bboxes','scores','labels'):dt[key]=np.concatenate((dt[key],added[key]),axis=0)
                dt['rles']=dt['rles']+added['rles'];dt['mask_count']+=added['mask_count'];dt['mask_fill_sum']+=added['mask_fill_sum']
            assert len(dt['scores'])<=250 and len(dt['scores'])>=len(sat)
            for key in ('bboxes','scores','labels'):assert np.array_equal(dt[key][:len(sat)],arrays(baseline['dt'])[key])
            assert dt['rles'][:len(sat)]==baseline['dt']['rles']
            all_gt.append(gt);all_dt.append(dt);metas.append(batch['img_metas'][0])
            rows.append(dict(image_id=baseline['image_id'],gt=d.common.serialized(gt),dt=d.common.serialized(dt),
                admission=dict(proposed=input_count,after_score_and_sat_overlap=before_nms,accepted=len(boxes),original=len(sat))))
            d.common.dump(dest/'progress.json',dict(images=index+1,pid=os.getpid()))
            if (index+1)%25==0:print('EVAL',args.content,args.boxes,index+1,flush=True)
    assert d.common.state_digest(model)==d.FROZEN_SHA
    if not args.gate:
        cg,cd=d.common.build_coco_gt_and_dt(all_gt,all_dt,metas,score_key='scores');metrics={}
        for kind in ('segm','bbox'):metrics.update(d.common.run_coco_eval(cg,cd,kind,max_dets=[1,10,250]))
        d.common.dump(dest/'metrics.json',metrics);print('METRICS',json.dumps(metrics),flush=True)
    d.common.dump(dest/'records.json',rows)
    d.common.dump(dest/'identity.json',dict(complete=True,images=len(rows),split='val',content=args.content,boxes=args.boxes,
        original_satellite_detections_exact=True,original_masks_reused_exact=True,satellite_state_unchanged=True,
        source_sha256=d.common.file_sha(__file__),baseline_records_sha256=d.common.file_sha(REF),
        proposal_records_sha256=d.common.file_sha(PILOT/args.content/'val_records.json'),
        proposal_checkpoint_sha256=result['checkpoint_sha256'],base_sha256=d.common.file_sha(d.BASE),
        max_dets=250,fp32=True,nms_pre=2000,original_nms_iou=float(threshold),
        peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
        scope='Frozen direct classifier/mask consumption; original SAT detections preserved first. '
              'Extra candidates use original ROI foreground classifier, score threshold and self-NMS; '
              'reject overlap with original SAT above original NMS threshold, remaining final250 slots only. '
              'Raw/regressed arms isolate additional original bbox regression, both use identical original ROI classifier. '
              'No training/test evaluation/threshold search.'))


if __name__=='__main__':main()
