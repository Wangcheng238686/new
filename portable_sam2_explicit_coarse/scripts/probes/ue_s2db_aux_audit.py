"""Offline auxiliary IoU and main/aux gradient diagnostics; no optimizer steps."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ue_s2db_probe as run
import numpy as np
import torch
from torch.nn import functional as F


def gradients(args):
    cp=torch.load(args.adapter,map_location='cpu',weights_only=False)
    for key in ('read','fusion','content','aux_weight'):setattr(args,key,cp['contract']['args'][key])
    assert args.aux_weight==.1
    model,adapter,_=run.make_model(args);adapter.load_state_dict(cp['adapter'],strict=True);del cp
    before=run.state_digest(model)
    ds=run.MinimalSatelliteDataset(args.data_root,'train',augment=False)
    order=np.argsort([len(s['annotations']) for s in ds.samples],kind='stable')
    indices=[int(order[round(q*(len(ds)-1))]) for q in (.125,.375,.625,.875)]
    # Evaluation mode removes dropout noise from this fixed diagnostic only.
    adapter.eval();rows=[]
    names=[n for n,p in adapter.named_parameters() if p.requires_grad and
        (n.startswith(('encoder.','source.','key.','value.','geom_key.','geom_value.','offset.')))]
    params=[dict(adapter.named_parameters())[n] for n in names]
    for i in indices:
        run.seed(44);p=run.processed(model,run.rtmdet_collate_fn([ds[i]]),True)
        data=run.payload(run.point_folder(ds,i,args.data_root),args.cache,'constant' if args.content=='constant' else 'correct')
        features=run.bind(model,adapter,p,data)
        with run.cached_extract(model,features):losses=model.loss(p['inputs'],p['data_samples'])
        main=sum((v for v in losses.values() if torch.is_tensor(v) and v.requires_grad),adapter.alpha*0)
        aux,count=adapter.auxiliary_loss(run.target_union(p))
        gm=torch.autograd.grad(main,params,allow_unused=True,retain_graph=True)
        ga=torch.autograd.grad(aux,params,allow_unused=True)
        m2=a2=dot=0.
        for x,y in zip(gm,ga):
            if x is not None:m2+=float(x.double().square().sum())
            if y is not None:a2+=float(y.double().square().sum())
            if x is not None and y is not None:dot+=float((x.double()*y.double()).sum())
        rows.append(dict(image_id=ds.samples[i]['image_id'],main_norm=m2**.5,aux_unweighted_norm=a2**.5,
            aux_weighted_norm=.1*a2**.5,cosine=dot/(m2*a2)**.5 if m2>0 and a2>0 else None,
            aux_valid_pixels=int(count),main_loss=float(main.detach()),aux_bce=float(aux.detach())))
        adapter.clear()
    assert run.state_digest(model)==before and all(p.grad is None for p in model.parameters())
    run.dump(args.out/'gradients.json',dict(epoch=torch.load(args.adapter,map_location='cpu',weights_only=False)['epoch'],
        fixed_train_quantiles=[.125,.375,.625,.875],augmentation=False,dropout=False,precision='fp32',
        scaled=False,clipped=False,shared_parameters=names,samples=rows,model_state_unchanged=True,
        checkpoint_sha256=run.file_sha(args.adapter)))


def aux_iou(args):
    cp=torch.load(args.adapter,map_location='cpu',weights_only=False)
    for key in ('read','fusion','content','aux_weight'):setattr(args,key,cp['contract']['args'][key])
    assert args.aux_weight==.1
    model,adapter,_=run.make_model(args);adapter.load_state_dict(cp['adapter'],strict=True);del cp
    model.eval();adapter.eval()
    ds=run.MinimalSatelliteDataset(args.data_root,args.eval_split)
    inter=union=count=0;rows=[]
    with torch.inference_mode():
        for i in range(len(ds)):
            p=run.processed(model,run.rtmdet_collate_fn([ds[i]]))
            data=run.payload(run.point_folder(ds,i,args.data_root),args.cache,'constant' if args.content=='constant' else 'correct')
            run.bind(model,adapter,p,data)
            z=F.interpolate(adapter.aux_logits.float(),(512,512),mode='bilinear',align_corners=False)[0,0]
            valid=adapter.support.bool();g=run.target_union(p).bool();pred=z>=0
            it=int((g&pred&valid).sum());un=int(((g|pred)&valid).sum());n=int(valid.sum())
            inter+=it;union+=un;count+=n
            rows.append(dict(image_id=ds.samples[i]['image_id'],intersection=it,union=un,valid_pixels=n))
            adapter.clear()
    run.dump(args.out/'aux_iou.json',dict(split=args.eval_split,images=len(ds),intersection=inter,union=union,
        valid_pixels=count,IoU=inter/union if union else None,threshold_logit=0,
        dependency='satellite-query-conditioned UAV prediction',rows=rows,
        checkpoint_sha256=run.file_sha(args.adapter)))

if __name__=='__main__':
    p=run.parser();p._actions[1].choices=['gradients','aux_iou'];a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);run.seed(44)
    if a.action=='gradients':gradients(a)
    else:aux_iou(a)
