"""Confirm the val-positive direct geometry branch on fixed test; no refitting."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='2'
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from ue_geometry_conditional_online_20261024 import d,pilot,arrays,merge,hypotheses,masks,RPN

OUT=Path('/data/wangcheng/checkpoint/ue_geometry_branch_test_20261024')
VAL=Path('/data/wangcheng/checkpoint/ue_geometry_score_band_20261024')


def main():
    p=argparse.ArgumentParser();p.add_argument('--content',choices=['geometry','constant'],required=True);p.add_argument('--gate',action='store_true');a=p.parse_args()
    split='val' if a.gate else 'test';out=OUT/a.content/('gate' if a.gate else 'test');out.mkdir(parents=True,exist_ok=True)
    if not a.gate:assert json.loads((OUT/a.content/'gate/identity.json').read_text())['complete']
    torch.cuda.set_per_process_memory_fraction(.23);d.common.seed(44);model,_=d.build();model.rpn_head.test_cfg.nms_pre=2000
    head=model.roi_head;cfg=head.test_cfg
    cp=torch.load(pilot.OUT/a.content/'last.pth',map_location='cpu',weights_only=False)
    assert cp['epoch']==10 and cp['step']==2670 and cp['content']==a.content
    assert cp['protocol']['source_sha256']==d.common.file_sha(pilot.__file__)
    net=pilot.Net().cuda();net.load_state_dict(cp['model']);net.eval()
    means=torch.tensor(json.loads(pilot.CONSTANTS.read_text())['mean_fields'],device='cuda')[None,:,None,None]
    ref=json.loads((RPN/split/'budget/records.json').read_text());ds=d.common.MinimalSatelliteDataset(d.DATA,split,augment=False)
    expected=json.loads((VAL/a.content/'val/records.json').read_text()) if a.gate else None
    rows=[];metas=[]
    with torch.inference_mode():
        for i in range(2 if a.gate else len(ds)):
            r=ref[i];assert r['image_id']==ds.samples[i]['image_id']
            batch=d.common.rtmdet_collate_fn([ds[i]]);sample=d.common.processed(model,batch)
            f=model.extract_feat(sample['inputs']);meta=sample['data_samples'][0].metainfo;xx=head._inject_extra_pe(f[0])
            cls,reg=model.rpn_head(f[0]);props=model.rpn_head.predict_by_feat(cls,reg,batch_img_metas=[meta],rescale=False)
            sat=head.predict_bbox(xx,[meta],props,cfg,rescale=False)[0]
            for k in ('bboxes','scores','labels'):
                v=getattr(sat,k);assert torch.equal(v,torch.tensor(r['dt'][k],device='cuda',dtype=v.dtype))
            gt=d.common.gt_arrays(sample['data_samples'][0].gt_instances);assert d.common.serialized(gt)==r['gt']
            folder=d.common.point_folder(ds,i,d.DATA)
            with np.load(pilot.CACHE/f'{folder.name}.npz') as z:
                x=torch.from_numpy(np.concatenate((z['fields'].astype(np.float32),z['support'][None].astype(np.float32)))).cuda()[None]
            boxes,confidence=pilot.decode(net(pilot.content(x,a.content,means)),x[:,4:5])
            boxes,confidence=hypotheses(boxes,confidence,sat.bboxes,cfg)
            dt=merge(arrays(r['dt']),masks(head,xx,f,meta,boxes,cfg.score_thr*confidence.clamp(0,1)))
            assert len(dt['scores'])<=250 and (dt['scores'][:len(sat)]>cfg.score_thr).all() and (dt['scores'][len(sat):]<=cfg.score_thr).all()
            record=dict(image_id=r['image_id'],gt=r['gt'],dt=d.common.serialized(dt),admission=dict(original=len(sat),accepted=len(boxes)))
            if a.gate:
                for k in ('bboxes','labels','rles','scores'):assert record['dt'][k]==expected[i]['dt'][k],('val reproduction',i,k)
            rows.append(record);metas.append(batch['img_metas'][0]);d.common.dump(out/'progress.json',dict(images=i+1,pid=os.getpid()))
            if (i+1)%25==0:print('EVAL',a.content,split,i+1,flush=True)
    assert d.common.state_digest(model)==d.FROZEN_SHA
    d.common.dump(out/'records.json',rows)
    if not a.gate:
        gg=[{**r['gt'],'bboxes':np.asarray(r['gt']['bboxes'],np.float32),'labels':np.asarray(r['gt']['labels'],np.int64)} for r in rows]
        cg,cd=d.common.build_coco_gt_and_dt(gg,[arrays(r['dt']) for r in rows],metas,score_key='scores');metrics={}
        for kind in ('segm','bbox'):metrics.update(d.common.run_coco_eval(cg,cd,kind,max_dets=[1,10,250]))
        d.common.dump(out/'metrics.json',metrics);print('METRICS',a.content,metrics['segm/mAP'],metrics['bbox/mAP'],flush=True)
    d.common.dump(out/'identity.json',dict(complete=True,images=len(rows),split=split,content=a.content,FP32=True,max_dets=250,
        source_sha256=d.common.file_sha(__file__),shared_source_sha256=d.common.file_sha(Path(__file__).with_name('ue_geometry_conditional_online_20261024.py')),
        checkpoint_sha256=d.common.file_sha(pilot.OUT/a.content/'last.pth'),satellite_state_unchanged=True,
        cached_val_exact=a.gate,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
        scope='Frozen E10 standalone geometry score-band branch, paired controls; repeatedly exposed v1 test, single training seed. '
              'Original SAT predictions unchanged; AP nondecrease not sufficient for content attribution.'))


if __name__=='__main__':main()
