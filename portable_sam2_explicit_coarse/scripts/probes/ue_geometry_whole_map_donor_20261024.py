"""Unpaired complete native map, fixed real-trained checkpoint; no fitting."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='2'
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch
from mmcv.ops import batched_nms
from mmengine.structures import InstanceData
from mmdet.structures.bbox import bbox_overlaps
from pycocotools import mask as mu
import ue_detection_attrition_probe_20261023 as d
import ue_geometry_proposal_pilot_20261024 as pilot

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
from ue_geometry_conditional_complement_20261024 import trusted_score

OUT=Path('/data/wangcheng/checkpoint/ue_geometry_whole_map_donor_20261024')
CACHED=Path('/data/wangcheng/checkpoint/ue_geometry_conditional_complement_20261024')
RPN=Path('/data/wangcheng/checkpoint/ue_rpn_geometry_20261023')


def arrays(row):
    x=copy.deepcopy(row)
    x['bboxes']=np.asarray(x['bboxes'],np.float32).reshape(-1,4)
    x['labels']=np.asarray(x['labels'],np.int64)
    x['scores']=np.asarray(x['scores'],np.float64)
    return x


def merge(base,extra):
    x=copy.deepcopy(base)
    for key in ('bboxes','labels','scores'):x[key]=np.concatenate((x[key],extra[key]))
    x['rles']+=extra['rles'];x['mask_count']+=extra['mask_count'];x['mask_fill_sum']+=extra['mask_fill_sum']
    return x


def hypotheses(boxes,confidence,original,cfg):
    valid=torch.isfinite(boxes).all(1)&(boxes[:,2:]>boxes[:,:2]).all(1)&torch.isfinite(confidence)&(confidence>0)
    if len(boxes) and len(original):valid &= bbox_overlaps(boxes,original).max(1).values<=cfg.nms.iou_threshold
    boxes,confidence=boxes[valid],confidence[valid]
    if len(boxes):
        dets,_=batched_nms(boxes,confidence,torch.zeros(len(boxes),device='cuda',dtype=torch.long),cfg.nms)
        dets=dets[:max(0,250-len(original))];boxes,confidence=dets[:,:4],dets[:,4]
    return boxes,confidence


def masks(head,xx,f,meta,boxes,scores):
    if not len(boxes):return dict(bboxes=np.zeros((0,4),np.float32),labels=np.zeros(0,np.int64),scores=np.zeros(0,np.float32),rles=[],mask_count=0,mask_fill_sum=0.)
    inst=InstanceData(bboxes=boxes,scores=scores,labels=torch.zeros(len(boxes),device='cuda',dtype=torch.long))
    pred=head.predict_mask(xx,[meta],[inst],rescale=True,image_embeddings=f[1],image_positional_embeddings=f[2],high_res_features=f[3])[0]
    return d.common.gt_arrays(pred)


def main():
    global OUT
    p=argparse.ArgumentParser();p.add_argument('--content',choices=['geometry','constant'],required=True);p.add_argument('--gate',action='store_true')
    p.add_argument('--pilot-root',type=Path,default=pilot.OUT);p.add_argument('--out-root',type=Path,default=OUT);p.add_argument('--cache-reference',type=Path)
    p.add_argument('--donor-offset',type=int,default=229)
    a=p.parse_args();OUT=a.out_root
    split='val' if a.gate else 'test'
    out=OUT/a.content/('gate' if a.gate else 'test');out.mkdir(parents=True,exist_ok=True)
    if not a.gate:assert json.loads((OUT/a.content/'gate/identity.json').read_text())['complete']
    torch.cuda.set_per_process_memory_fraction(.23);d.common.seed(44);model,_=d.build()
    model.rpn_head.test_cfg.nms_pre=2000
    head=model.roi_head;cfg=head.test_cfg
    checkpoint=a.pilot_root/a.content/'last.pth'
    cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert cp['epoch']==10 and cp['step']==2670 and cp['content']==a.content
    assert cp['protocol'].get('model_source_sha256',cp['protocol']['source_sha256'])==d.common.file_sha(pilot.__file__)
    net=pilot.Net().cuda();net.load_state_dict(cp['model'],strict=True);net.eval()
    means=torch.tensor(json.loads(pilot.CONSTANTS.read_text())['mean_fields'],device='cuda')[None,:,None,None]
    ds=d.common.MinimalSatelliteDataset(d.DATA,split,augment=False)
    ref=json.loads((RPN/split/'budget/records.json').read_text())
    cached=json.loads((a.cache_reference/f'{a.content}_records.json').read_text()) if a.gate and a.cache_reference else None
    prefix_rows=[];rows=[];metas=[];counts=[]
    with torch.inference_mode():
        for i in range(2 if a.gate else len(ds)):
            baseline=ref[i];assert baseline['image_id']==ds.samples[i]['image_id']
            batch=d.common.rtmdet_collate_fn([ds[i]]);sample=d.common.processed(model,batch)
            f=model.extract_feat(sample['inputs']);meta=sample['data_samples'][0].metainfo;xx=head._inject_extra_pe(f[0])
            cls,reg=model.rpn_head(f[0]);props=model.rpn_head.predict_by_feat(cls,reg,batch_img_metas=[meta],rescale=False)
            sat=head.predict_bbox(xx,[meta],props,cfg,rescale=False)[0]
            for key in ('bboxes','scores','labels'):
                value=getattr(sat,key);assert torch.equal(value,torch.tensor(baseline['dt'][key],device='cuda',dtype=value.dtype))
            gt=d.common.gt_arrays(sample['data_samples'][0].gt_instances);assert d.common.serialized(gt)==baseline['gt']
            boxes=props[0].bboxes;rois=torch.cat((boxes.new_zeros((len(boxes),1)),boxes),1)
            br=head._bbox_forward(xx,rois);raw=head.bbox_head._predict_by_feat_single(rois,br['cls_score'],br['bbox_pred'],meta,False,None)
            sb,sc=hypotheses(raw.bboxes,raw.scores[:,0],sat.bboxes,cfg)
            prefix=merge(arrays(baseline['dt']),masks(head,xx,f,meta,sb,cfg.score_thr*sc.clamp(0,1)))
            prefix_rows.append(dict(image_id=baseline['image_id'],gt=baseline['gt'],dt=d.common.serialized(prefix)))
            folder=d.common.point_folder(ds,(i+a.donor_offset)%len(ds),d.DATA)
            with np.load(pilot.CACHE/f'{folder.name}.npz') as z:
                x=torch.from_numpy(np.concatenate((z['fields'].astype(np.float32),z['support'][None].astype(np.float32)))).cuda()[None]
            gb,gc=pilot.decode(net(pilot.content(x,a.content,means)),x[:,4:5])
            gb,gc=hypotheses(gb,gc,sat.bboxes,cfg)
            # Reproduce the original full geometry mask batch before selecting
            # the subset, avoiding batch-shape changes relative to cached val.
            extra=masks(head,xx,f,meta,gb,cfg.score_thr*gc.clamp(0,1))
            trusted=torch.tensor(prefix['bboxes'],device='cuda',dtype=torch.float32)
            keep=bbox_overlaps(gb,trusted).max(1).values<=cfg.nms.iou_threshold if len(gb) and len(trusted) else torch.ones(len(gb),device='cuda',dtype=torch.bool)
            ids=torch.where(keep)[0][:max(0,250-len(trusted))].cpu().tolist()
            combined=copy.deepcopy(prefix);combined['scores']=np.asarray(trusted_score(prefix['scores']),np.float64)
            added=dict(bboxes=extra['bboxes'][ids],labels=extra['labels'][ids],scores=extra['scores'][ids].astype(np.float64)/.05*.25,
                rles=[extra['rles'][j] for j in ids],mask_count=len(ids),mask_fill_sum=float(np.sum(mu.area([extra['rles'][j] for j in ids]))/512**2) if ids else 0.)
            combined=merge(combined,added);assert len(combined['scores'])<=250
            row=dict(image_id=baseline['image_id'],gt=baseline['gt'],dt=d.common.serialized(combined),
                admission=dict(original=len(sat),satellite_supplement=len(sb),geometry_cached_equivalent=len(gb),geometry_accepted=len(ids),input_point_id=int(folder.name)))
            if a.gate and cached is not None:
                expected=cached[i]['dt'];actual=row['dt']
                assert actual['bboxes']==expected['bboxes'] and actual['labels']==expected['labels']
                assert actual['rles']==expected['rles'],'Online/cached mask mismatch'
                assert np.allclose(actual['scores'],expected['scores'],rtol=0,atol=1e-12),'Online/cached score mismatch'
            rows.append(row);metas.append(batch['img_metas'][0]);counts.append(row['admission'])
            d.common.dump(out/'progress.json',dict(images=i+1,pid=os.getpid(),split=split))
            if (i+1)%25==0:print('EVAL',a.content,split,i+1,flush=True)
    assert d.common.state_digest(model)==d.FROZEN_SHA
    for name,data in [('prefix',prefix_rows),('combined',rows)]:
        d.common.dump(out/f'{name}_records.json',data)
        if not a.gate:
            cg,cd=d.common.build_coco_gt_and_dt([arrays(r['gt']) if 'scores' in r['gt'] else {**r['gt'],'bboxes':np.asarray(r['gt']['bboxes'],np.float32),'labels':np.asarray(r['gt']['labels'],np.int64)} for r in data],[arrays(r['dt']) for r in data],metas,score_key='scores')
            metrics={}
            for kind in ('segm','bbox'):metrics.update(d.common.run_coco_eval(cg,cd,kind,max_dets=[1,10,250]))
            d.common.dump(out/f'{name}_metrics.json',metrics);print('METRICS',a.content,name,metrics['segm/mAP'],metrics['bbox/mAP'],flush=True)
    d.common.dump(out/'identity.json',dict(complete=True,images=len(rows),split=split,content=a.content,donor_offset=a.donor_offset,checkpoint_sha256=d.common.file_sha(checkpoint),
        source_sha256=d.common.file_sha(__file__),proposal_source_sha256=d.common.file_sha(pilot.__file__),
        base_sha256=d.common.file_sha(d.BASE),original_reference_sha256=d.common.file_sha(RPN/split/'budget/records.json'),
        satellite_state_unchanged=True,online_cached_equivalence_gate=bool(a.gate and cached is not None),original_prediction_gate=a.gate,FP32=True,max_dets=250,
        peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
        scope='Locked E10 direct geometry proposals on strong SAT2000+rejected-RoI prefix. Prefix confidence monotonically recoded, original rank and masks retained. '
              'Test descriptive development evaluation, single training seed, not independent unseen benchmark. No test selection or new fitting.'))


if __name__=='__main__':main()
