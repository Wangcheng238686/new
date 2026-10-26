"""Direct instance-supervised geometry proposals; standalone train/val pilot.

No satellite weights, RGB, existing detector mutation, or test evaluation.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='2'
import argparse
import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from mmdet.structures.bbox import bbox_overlaps
import ue_geometry_mechanism_probe as g

OUT=Path('/data/wangcheng/checkpoint/ue_geometry_proposal_pilot_20261024')
CACHE=Path('/data/wangcheng/cache/ue_v1_native_geometry')
CONSTANTS=Path('/data/wangcheng/checkpoint/ue_geometry_mechanism_20261022/common_constants.json')


class Data(Dataset):
    def __init__(self,split):
        a=json.loads((g.DATA/'annotations'/f'{split}.json').read_text())
        self.ids=json.loads((g.DATA/'splits'/f'{split}.json').read_text())
        anns={i:[] for i in self.ids}
        for row in a['annotations']:anns[row['image_id']].append(row)
        self.boxes=[]
        for i in self.ids:
            boxes=np.asarray([r['bbox'] for r in anns[i]],dtype=np.float32).reshape(-1,4)
            boxes[:,2:]+=boxes[:,:2]
            self.boxes.append(boxes)

    def __len__(self):return len(self.ids)

    def __getitem__(self,index):
        ident=self.ids[index]
        with np.load(CACHE/f'{ident:06d}.npz') as z:
            x=np.concatenate((z['fields'].astype(np.float32),z['support'][None].astype(np.float32)))
        heat=np.zeros((1,128,128),np.float32)
        reg=np.zeros((4,128,128),np.float32);valid=np.zeros((1,128,128),np.float32)
        boxes=self.boxes[index]
        # Last (smallest-area) instance wins regression for a shared center cell.
        order=np.argsort(-np.prod(boxes[:,2:]-boxes[:,:2],axis=1),kind='stable')
        for box in boxes[order]:
            wh=(box[2:]-box[:2])/4
            center=(box[:2]+box[2:])/8
            ix,iy=np.clip(np.floor(center).astype(int),0,127)
            sigma=float(np.clip(min(wh)/6,1,4));radius=int(math.ceil(3*sigma))
            x0,x1=max(0,ix-radius),min(128,ix+radius+1)
            y0,y1=max(0,iy-radius),min(128,iy+radius+1)
            yy,xx=np.mgrid[y0:y1,x0:x1]
            gauss=np.exp(-((xx-ix)**2+(yy-iy)**2)/(2*sigma**2))
            heat[0,y0:y1,x0:x1]=np.maximum(heat[0,y0:y1,x0:x1],gauss)
            reg[:2,iy,ix]=center-[ix,iy]
            reg[2:,iy,ix]=np.log(np.maximum(wh,1e-3))
            valid[0,iy,ix]=1
        return dict(x=torch.from_numpy(x),heat=torch.from_numpy(heat),reg=torch.from_numpy(reg),valid=torch.from_numpy(valid),image_id=ident,index=index)


def block(cin,cout,stride=1):
    return nn.Sequential(nn.Conv2d(cin,cout,3,stride,padding=1),nn.GroupNorm(4,cout),nn.GELU(),
                         nn.Conv2d(cout,cout,3,padding=1),nn.GroupNorm(4,cout),nn.GELU())


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.a=block(5,16,2);self.b=block(16,32,2);self.c=block(32,64,2);self.d=block(64,64,2)
        self.up=block(160,32)
        self.heat=nn.Conv2d(32,1,1);self.reg=nn.Conv2d(32,4,1)
        nn.init.constant_(self.heat.bias,-2.19)

    def forward(self,x):
        b=self.b(self.a(x));c=self.c(b);d=self.d(c)
        z=self.up(torch.cat((b,F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),
            F.interpolate(d,size=b.shape[-2:],mode='bilinear',align_corners=False)),1))
        return self.heat(z),self.reg(z)


def loss_fn(outputs,batch):
    heat,reg=outputs
    prob=heat.float().sigmoid().clamp(1e-4,1-1e-4)
    target=batch['heat'];positive=(target==1).float();negative=(target<1).float()
    denom=batch['valid'].sum().clamp_min(1)
    focal=-(prob.log()*(1-prob)**2*positive+(1-prob).log()*prob**2*(1-target)**4*negative).sum()/denom
    error=F.smooth_l1_loss(reg.float(),batch['reg'],reduction='none')*batch['valid']
    offset=error[:,:2].sum()/denom;size=error[:,2:].sum()/denom
    return focal+offset+.1*size,dict(heat=float(focal.detach()),offset=float(offset.detach()),size=float(size.detach()))


def content(x,kind,means):
    if kind=='constant':return torch.cat((means*x[:,4:5],x[:,4:5]),1)
    return x


def decode(outputs,support):
    heat,reg=outputs
    score=heat.float().sigmoid()
    local=score==F.max_pool2d(score,3,1,1)
    # Any native support within the corresponding local 12x12 neighborhood.
    known=F.max_pool2d(F.max_pool2d(support,4,4),3,1,1)>0
    score=score.masked_fill(~local|~known,-1)
    vals,ids=score.flatten().topk(250)
    keep=vals>=0;vals,ids=vals[keep],ids[keep]
    r=reg[0].reshape(4,-1)[:,ids].T.float()
    center=torch.stack((ids%128,ids//128),1).float()+r[:,:2].clamp(0,1)
    wh=r[:,2:].clamp(math.log(.25),math.log(128)).exp()
    boxes=torch.cat((center-wh/2,center+wh/2),1).mul(4).clamp(0,512)
    return boxes,vals


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['init','gate','train']);p.add_argument('--content',choices=['geometry','constant'],default='geometry');p.add_argument('--resume',type=Path)
    args=p.parse_args();OUT.mkdir(exist_ok=True)
    g.seed(45);net=Net()
    if args.mode=='init':
        init=OUT/'common_init.pth';assert not init.exists()
        g.atomic_save(dict(model=net.state_dict(),fingerprint=g.tensor_digest(net.state_dict())),init)
        g.dump(OUT/'protocol.json',dict(epochs=10,updates=2670,batch=8,accum=1,train2142_drop_last6=True,
            optimizer='AdamW lr1e-3 wd.01 warmup50 cosine_min.1 clip.5 AMP scale512',seed=44,epoch_seed='1044000+epoch',
            target='128 grid Gaussian instance centers; sigma clamp(min(box_wh_grid)/6,1,4); smoothL1 offset + .1 log-wh; focal center normalized by unique center cells',
            input='native512 four geometric fields + support, no RGB/satellite features',
            collision='same grid center uses smallest-box regression; center denominator unique cells',
            selection='E10 last only',eval='all459 val FP32; top250 local heat peaks; same known-support mask in both arms; no test access',
            content_control='train-only field means times unchanged support; common complete initialization',
            source_sha256=g.file_sha(__file__),constants_sha256=g.file_sha(CONSTANTS),archive_hashes=g.archive_identity(g.DATA),
            scope='Standalone proposal learnability pilot; no SAT weights loaded or updated, no final AP claim. A matched constant arm retains actual UAV support.'))
        return
    torch.cuda.set_per_process_memory_fraction(.23)
    cp=torch.load(OUT/'common_init.pth',map_location='cpu',weights_only=False)
    net.load_state_dict(cp['model']);assert g.tensor_digest(net.state_dict())==cp['fingerprint'];net.cuda()
    means=torch.tensor(json.loads(CONSTANTS.read_text())['mean_fields'],device='cuda')[None,:,None,None]
    if args.mode=='gate':
        ds=Data('train');b=next(iter(DataLoader(ds,batch_size=2)))
        b={k:v.cuda() if torch.is_tensor(v) else v for k,v in b.items()}
        initial=copy.deepcopy(net.state_dict())
        with torch.autocast('cuda',dtype=torch.float16):out=net(b['x']);loss,_=loss_fn(out,b)
        assert out[0].shape==(2,1,128,128) and torch.isfinite(loss)
        loss.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in net.parameters())
        net.eval();single=(out[0][:1].float(),out[1][:1].float())
        boxes,scores=decode(single,b['x'][:1,4:5]);assert len(boxes)<=250 and torch.isfinite(boxes).all()
        empty,_=decode(single,torch.zeros_like(b['x'][:1,4:5]));assert len(empty)==0
        const=content(b['x'],'constant',means);assert torch.equal(const[:,4:],b['x'][:,4:])
        assert torch.equal(const[:,:4][~b['x'][:,4:5].bool().expand(-1,4,-1,-1)],torch.zeros_like(const[:,:4][~b['x'][:,4:5].bool().expand(-1,4,-1,-1)]))
        assert all(torch.equal(initial[k],v) for k,v in net.state_dict().items())
        g.dump(OUT/'gate.json',dict(passed=True,AMP_finite_gradients=True,unknown_proposals_empty=True,constant_support_exact=True,
            initial_fingerprint=cp['fingerprint'],peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20))
        print('GATE PASS',flush=True);return
    assert json.loads((OUT/'gate.json').read_text())['passed']
    arm=OUT/args.content;arm.mkdir(exist_ok=True)
    assert not (arm/'last.pth').exists() or args.resume
    protocol=json.loads((OUT/'protocol.json').read_text());assert protocol['source_sha256']==g.file_sha(__file__)
    g.seed(44);gen=torch.Generator().manual_seed(44)
    optimizer=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',init_scale=512)
    step=0;start_epoch=1
    if args.resume:
        old=torch.load(args.resume,map_location='cpu',weights_only=False)
        assert old['content']==args.content and old['protocol']==protocol
        net.load_state_dict(old['model']);optimizer.load_state_dict(old['optimizer']);scaler.load_state_dict(old['scaler'])
        step=old['step'];start_epoch=old['epoch']+1;g.restore_rng(old['rng_state'],gen)
    ds=Data('train');assert len(ds)==2142
    loader=DataLoader(ds,batch_size=8,shuffle=True,drop_last=True,num_workers=2,generator=gen)
    for epoch in range(start_epoch,11):
        g.seed(1044000+epoch);gen.manual_seed(1044000+epoch)
        net.train();ids=[];total=0;parts={k:0. for k in ('heat','offset','size')};began=time.time()
        for b in loader:
            ids.extend(b['image_id'].tolist());b={k:v.cuda(non_blocking=True) for k,v in b.items()}
            x=content(b['x'],args.content,means);optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):loss,components=loss_fn(net(x),b)
            assert torch.isfinite(loss)
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(net.parameters(),.5);assert torch.isfinite(norm)
            step+=1;lr=1e-3*min(1,step/50)*(.1+.9*.5*(1+math.cos(math.pi*step/2670)))
            for group in optimizer.param_groups:group['lr']=lr
            scaler.step(optimizer);scaler.update();total+=float(loss.detach())
            for key in parts:parts[key]+=components[key]
        assert len(ids)==2136 and step==epoch*267
        g.dump(arm/f'epoch{epoch}.json',dict(epoch=epoch,step=step,mean_batch_loss=total/267,
            mean_batch_components={k:v/267 for k,v in parts.items()},sample_order=ids,seconds=time.time()-began,amp_scale=scaler.get_scale()))
        ck=dict(model=net.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),epoch=epoch,step=step,
            protocol=protocol,content=args.content,rng_state=g.rng_state(gen),initial_fingerprint=cp['fingerprint'])
        g.atomic_save(ck,arm/'last.pth')
        g.dump(arm/'progress.json',dict(stage='train',epoch=epoch,step=step,pid=os.getpid()))
        print('EPOCH',args.content,epoch,total/267,time.time()-began,flush=True)
    net.eval();val=Data('val');rows=[]
    totals={k:0 for k in ['gt','any50@50','any50@100','any50@250','any75@250']}
    with torch.inference_mode():
        for index in range(len(val)):
            b=val[index];x=b['x'][None].cuda();boxes,scores=decode(net(content(x,args.content,means)),x[:,4:5])
            gt=torch.tensor(val.boxes[index],device='cuda');ov=bbox_overlaps(gt,boxes)
            counts=dict(gt=len(gt))
            for k in (50,100,250):counts[f'any50@{k}']=int((ov[:,:k].max(1).values>=.5).sum()) if len(boxes) else 0
            counts['any75@250']=int((ov.max(1).values>=.75).sum()) if len(boxes) else 0
            for key in totals:totals[key]+=counts[key]
            rows.append(dict(image_id=b['image_id'],counts=counts,boxes=boxes.cpu().tolist(),scores=scores.cpu().tolist()))
    g.dump(arm/'val_records.json',rows)
    g.dump(arm/'result.json',dict(complete=True,images=len(val),totals=totals,checkpoint_sha256=g.file_sha(arm/'last.pth'),
        source_sha256=g.file_sha(__file__),initial_fingerprint=cp['fingerprint'],
        scope='Standalone proposal any-box IoU recall, no matching exclusivity/NMS/AP. Constant shares real support. Test untouched.'))
    print('RESULT',args.content,totals,flush=True)


if __name__=='__main__':main()
