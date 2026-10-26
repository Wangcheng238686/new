"""Small train/val candidate evidence bridge, not a final neural/AP experiment.

Fixed density samples; decoded-box target; matched table learners; no test access.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F
from mmdet.structures.bbox import bbox_overlaps
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
import ue_detection_attrition_probe_20261023 as d

OUT=Path('/data/wangcheng/checkpoint/ue_candidate_quality_bridge_20261024')
CACHE=Path('/data/wangcheng/cache/ue_v1_native_geometry')
CONSTANTS=Path('/data/wangcheng/checkpoint/ue_geometry_mechanism_20261022/common_constants.json')
ARMS=('constant_region','geometry_center','geometry_region')


def extract():
    OUT.mkdir(exist_ok=True)
    manifest=dict(source_sha256=d.common.file_sha(__file__),base_sha256=d.common.file_sha(d.BASE),
        archive_hashes=d.common.archive_identity(d.DATA), train_images=64,val_images=32,
        selection='Equally spaced mid density quantiles, stable GT-count sort',
        descriptor='frozen RPN hidden random orthogonal-unconstrained projection32 + channel LN; logit,width/512,height/512,level/4; geometry4x3x3; support3x3',
        pool='per level original sigmoid top2000, incl all when fewer; original regressed boxes; per-level select1000',
        geometry='nearest 3x3 decoded box interior grid at 1/6,3/6,5/6, no hole fill; center repeats same descriptor9x',
        target='decoded max IoU >=.5 positive; <.3 negative; [.3,.5) ignored during fitting only',
        fit='HistGradientBoosting max_iter100 max_leaf_nodes15 min_samples_leaf50 l2=1 lr=.1 early_stopping=False seed44; max2048 positives+2048 negatives/image, deterministic',
        scope='Train/val diagnostic, not final AP or stable UAV gain. No test access. Frozen random satellite projection may lose useful features. Constant retains true geometric support.',
        constants_sha256=d.common.file_sha(CONSTANTS))
    d.common.dump(OUT/'protocol.json',manifest)
    torch.cuda.set_per_process_memory_fraction(.23)
    d.common.seed(44);model,_=d.build()
    rng=torch.Generator().manual_seed(45)
    projection=torch.randn(32,model.rpn_head.rpn_cls.in_channels,generator=rng).cuda()/model.rpn_head.rpn_cls.in_channels**.5
    hidden=[]
    handle=model.rpn_head.rpn_cls.register_forward_pre_hook(lambda module,args:hidden.append(args[0].detach()))
    rows=[]
    with torch.inference_mode():
        for split,count in [('train',64),('val',32)]:
            ds=d.common.MinimalSatelliteDataset(d.DATA,split,augment=False)
            order=np.argsort([len(s['annotations']) for s in ds.samples],kind='stable')
            indices=[int(order[round((j+.5)/count*(len(ds)-1))]) for j in range(count)]
            for index in indices:
                folder=d.common.point_folder(ds,index,d.DATA)
                cache=CACHE/f'{folder.name}.npz'
                with np.load(cache) as z:
                    field=torch.from_numpy(z['fields'].astype(np.float32)).cuda()[None]
                    support=torch.from_numpy(z['support'].astype(np.float32)).cuda()[None,None]
                p=d.common.processed(model,d.common.rtmdet_collate_fn([ds[index]]))
                gt=p['data_samples'][0].gt_instances.bboxes
                meta=p['data_samples'][0].metainfo
                features=model.extract_feat(p['inputs']);hidden.clear()
                cls,reg=model.rpn_head(features[0]);rpn=model.rpn_head
                priors=rpn.prior_generator.grid_priors([c.shape[-2:] for c in cls],device='cuda')
                descriptors=[];center_descriptors=[];boxes=[];levels=[];original=[]
                for lev,(c,b,prior,hid) in enumerate(zip(cls,reg,priors,hidden)):
                    logits=c[0].permute(1,2,0).reshape(-1)
                    ids=logits.sigmoid().sort(descending=True)[1][:2000]
                    box=rpn.bbox_coder.decode(prior[ids],b[0].permute(1,2,0).reshape(-1,4)[ids],max_shape=meta['img_shape'])
                    q=F.conv2d(hid,projection[:,:,None,None])
                    q=F.layer_norm(q.permute(0,2,3,1).float(),(32,)).reshape(-1,32)[ids//c.shape[1]]
                    frac=(torch.arange(3,device='cuda')+.5)/3
                    yy,xx=torch.meshgrid(frac,frac,indexing='ij')
                    def sample_geometry(fractions):
                        uv=box[:,:2,None]+(box[:,2:]-box[:,:2])[:,:,None]*fractions[None]
                        grid=((uv.permute(0,2,1)+.5)/256-1)[None]
                        geo=F.grid_sample(field,grid,mode='nearest',align_corners=False)[0].permute(1,0,2)
                        sup=F.grid_sample(support,grid,mode='nearest',align_corners=False)[0].permute(1,0,2)
                        return torch.cat((geo.reshape(len(box),36),sup.reshape(len(box),9)),1)
                    region=sample_geometry(torch.stack((xx.flatten(),yy.flatten())))
                    center=sample_geometry(torch.full((2,9),.5,device='cuda'))
                    base=torch.cat((q,logits[ids,None],(box[:,2:]-box[:,:2])/512,torch.full((len(box),1),lev/4,device='cuda')),1)
                    descriptors.append(torch.cat((base,region),1));center_descriptors.append(torch.cat((base,center),1))
                    boxes.append(box);levels.append(torch.full((len(box),),lev,device='cuda'));original.append(logits[ids])
                box=torch.cat(boxes);iou=bbox_overlaps(box,gt)
                path=OUT/f'{split}_{folder.name}.npz'
                np.savez_compressed(path,x=torch.cat(descriptors).cpu().numpy(),xc=torch.cat(center_descriptors).cpu().numpy(),
                    boxes=box.cpu().numpy(),levels=torch.cat(levels).cpu().numpy(),logits=torch.cat(original).cpu().numpy(),
                    iou=iou.cpu().numpy(),gt=gt.cpu().numpy())
                rows.append(dict(split=split,index=index,image_id=ds.samples[index]['image_id'],file=path.name,
                    cache_sha256=d.common.file_sha(cache),sha256=d.common.file_sha(path),gt=len(gt)))
                d.common.dump(OUT/'progress.json',dict(stage='extract',images=len(rows),pid=os.getpid()))
                print('EXTRACT',split,index,flush=True)
    handle.remove();assert d.common.state_digest(model)==d.FROZEN_SHA
    d.common.dump(OUT/'manifest.json',dict(protocol=manifest,rows=rows,satellite_state_unchanged=True,optimizer_steps=0))


def transform(z,arm,means):
    x=z['xc' if arm=='geometry_center' else 'x'].copy()
    if arm=='constant_region':
        x[:,36:72]=(means[None,:,None]*x[:,None,72:81]).reshape(len(x),36)
    return x


def selected(scores,levels,k=1000):
    return np.concatenate([np.flatnonzero(levels==l)[np.argsort(-scores[levels==l],kind='stable')[:k]] for l in np.unique(levels)])


def fit():
    from joblib import dump
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert manifest['protocol']['source_sha256']==d.common.file_sha(__file__)
    means=np.array(json.loads(CONSTANTS.read_text())['mean_fields'],np.float32)
    train=[r for r in manifest['rows'] if r['split']=='train']
    val=[r for r in manifest['rows'] if r['split']=='val']
    indices={};ys=[]
    for row in train:
        with np.load(OUT/row['file']) as z:
            best=z['iou'].max(1)
            rng=np.random.default_rng(44+row['image_id'])
            a=np.flatnonzero(best>=.5);b=np.flatnonzero(best<.3)
            ids=np.r_[rng.permutation(a)[:2048],rng.permutation(b)[:2048]]
            indices[row['file']]=ids;ys.append((best[ids]>=.5).astype(np.int64))
    y=np.concatenate(ys)
    result={};per_image={}
    for arm in ARMS:
        xx=[]
        for row in train:
            with np.load(OUT/row['file']) as z:xx.append(transform(z,arm,means)[indices[row['file']]])
        x=np.concatenate(xx)
        clf=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,min_samples_leaf=50,l2_regularization=1,learning_rate=.1,early_stopping=False,random_state=44)
        clf.fit(x,y);dump(clf,OUT/f'{arm}.joblib')
        truths=[];predictions=[];counts=dict(gt=0,original_top1000=0,pool2000=0,reranked_top1000=0,recovered=0,lost=0)
        images=[]
        for row in val:
            with np.load(OUT/row['file']) as z:
                score=clf.predict_proba(transform(z,arm,means))[:,1]
                iou=z['iou'];levels=z['levels']
                # Pool order is the original torch sigmoid-sort order. Sorting
                # logits here would alter ties from saturated sigmoid scores.
                old=np.concatenate([np.flatnonzero(levels==l)[:1000] for l in np.unique(levels)])
                new=selected(score,levels)
                a=iou[old].max(0)>=.5;b=iou[new].max(0)>=.5;c=iou.max(0)>=.5
                eligible=(iou.max(1)>=.5)|(iou.max(1)<.3)
                truths.extend((iou.max(1)[eligible]>=.5).tolist());predictions.extend(score[eligible].tolist())
                record=dict(image_id=row['image_id'],gt=len(a),original_top1000=int(a.sum()),pool2000=int(c.sum()),
                    reranked_top1000=int(b.sum()),recovered=int((b&~a).sum()),lost=int((a&~b).sum()))
                images.append(record)
                for key in counts:counts[key]+=record[key]
        counts.update(candidate_roc_auc=float(roc_auc_score(truths,predictions)),candidate_average_precision=float(average_precision_score(truths,predictions)))
        result[arm]=counts;per_image[arm]=images
        print('FIT',arm,json.dumps(counts),flush=True)
        d.common.dump(OUT/'progress.json',dict(stage='fit',arms_complete=len(result),pid=os.getpid()))
    d.common.dump(OUT/'per_image.json',per_image)
    d.common.dump(OUT/'summary.json',dict(complete=True,result=result,training_rows=len(y),training_positive=int(y.sum()),
        source_sha256=d.common.file_sha(__file__),manifest_sha256=d.common.file_sha(OUT/'manifest.json'),
        scope='Fixed64train32val feature/target feasibility screen. Any-GT coverage only before RPN NMS; '
              'not one-to-one recall, full-split estimate, final AP, or evidence of stable UAV gain. '
              'No test inference or test-based selection. Real-vs-constant differences need full evaluation and repeatability.'))


if __name__=='__main__':
    {'extract':extract,'fit':fit}[sys.argv[1]]()
