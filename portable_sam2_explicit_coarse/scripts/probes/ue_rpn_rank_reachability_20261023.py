"""Fixed train-only bound on which original candidates a bounded score residual can rescue."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import json
import numpy as np
import torch
from mmdet.structures.bbox import bbox_overlaps
import ue_detection_attrition_probe_20261023 as d


def main():
    d.common.seed(44);model,_=d.build();ds=d.common.MinimalSatelliteDataset(d.DATA,'train',augment=False)
    order=np.argsort([len(s['annotations']) for s in ds.samples],kind='stable')
    indices=[int(order[round(q*(len(ds)-1))]) for q in (.125,.375,.625,.875)]
    rows=[]
    with torch.inference_mode():
        for i in indices:
            p=d.common.processed(model,d.common.rtmdet_collate_fn([ds[i]]));features=model.extract_feat(p['inputs'])
            cls,reg=model.rpn_head(features[0]);rpn=model.rpn_head
            priors=rpn.prior_generator.grid_priors([x.shape[-2:] for x in cls],dtype=cls[0].dtype,device='cuda')
            gt=p['data_samples'][0].gt_instances.bboxes;gaps=[]
            for c,b,prior in zip(cls,reg,priors):
                logits=c[0].permute(1,2,0).reshape(-1)
                decoded=rpn.bbox_coder.decode(prior,b[0].permute(1,2,0).reshape(-1,4),max_shape=p['data_samples'][0].metainfo['img_shape'])
                # Use the same sigmoid ordering as the deployed RPN.
                ranking=logits.sigmoid().sort(descending=True)[1]
                cutoff=logits[ranking[min(1000,len(logits))-1]]
                best=torch.full((len(gt),),-float('inf'),device='cuda')
                for start in range(0,len(logits),2048):
                    ov=bbox_overlaps(gt,decoded[start:start+2048])
                    scores=logits[start:start+2048][None].expand(len(gt),-1)
                    best=torch.maximum(best,scores.masked_fill(ov<.5,-float('inf')).max(1).values)
                # A level with <=1000 anchors has no truncation and is reachable.
                gap=cutoff-best if len(logits)>1000 else torch.where(torch.isfinite(best),torch.zeros_like(best),torch.full_like(best,float('inf')))
                gaps.append(gap)
            gap=torch.stack(gaps).min(0).values.cpu().numpy()
            rows.append(dict(image_id=ds.samples[i]['image_id'],index=i,gt=len(gt),
                minimum_level_logit_gap=[float(x) if np.isfinite(x) else None for x in gap],
                no_good_decoded_anchor=int((~np.isfinite(gap)).sum()),
                bound_impossible=int((np.isfinite(gap)&(gap>4)).sum()),
                bound_not_excluded=int((np.isfinite(gap)&(gap<=4)).sum())))
    assert d.common.state_digest(model)==d.FROZEN_SHA
    out=Path('/data/wangcheng/checkpoint/ue_rpn_geometry_20261023/rank');out.mkdir(exist_ok=True)
    d.common.dump(out/'reachability.json',dict(complete=True,split='train',quantiles=[.125,.375,.625,.875],indices=indices,
        fixed_residual_bound_per_anchor=2,exclusion_gap=4,rows=rows,optimizer_steps=0,
        scope='Sufficient impossibility bound for original decoded anchor entering per-level top1000; ignores support and limited shared head capacity, so not_excluded is not an achievability guarantee',
        satellite_state_unchanged=True,base_sha256=d.common.file_sha(d.BASE),script_sha256=d.common.file_sha(__file__)))
if __name__=='__main__':main()
