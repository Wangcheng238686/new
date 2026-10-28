"""Independent seed repeat of the locked geometry instance proposal pilot."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='2'
import argparse
import json
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
import ue_geometry_proposal_pilot_20261024 as p


def main():
    cli=argparse.ArgumentParser();cli.add_argument('mode',choices=['init','train'])
    cli.add_argument('--root',type=Path,required=True);cli.add_argument('--seed',type=int,default=45)
    cli.add_argument('--content',choices=['geometry','constant'],default='geometry');cli.add_argument('--resume',type=Path)
    args=cli.parse_args();root=args.root;root.mkdir(parents=True,exist_ok=True)
    p.g.seed(args.seed+1);net=p.Net()
    if args.mode=='init':
        assert not (root/'common_init.pth').exists()
        p.g.atomic_save(dict(model=net.state_dict(),fingerprint=p.g.tensor_digest(net.state_dict())),root/'common_init.pth')
        original=json.loads((p.OUT/'protocol.json').read_text())
        protocol={**original,'seed':args.seed,'initialization_seed':args.seed+1,
            'epoch_seed':f'{1_000_000+1000*args.seed}+epoch',
            'source_sha256':p.g.file_sha(__file__),'training_source_sha256':p.g.file_sha(__file__),
            'model_source_sha256':p.g.file_sha(p.__file__),
            'repeat_of':str(p.OUT),'changes':'Independent initialization/data seed only; architecture/loss/budget/optimizer/content rules fixed',
            'selection':'fixed E10 last; no intermediate selection; no test data decoded in training'}
        p.g.dump(root/'protocol.json',protocol);return
    protocol=json.loads((root/'protocol.json').read_text())
    assert protocol['source_sha256']==p.g.file_sha(__file__) and protocol['model_source_sha256']==p.g.file_sha(p.__file__)
    assert protocol['seed']==args.seed
    torch.cuda.set_per_process_memory_fraction(.23)
    init=torch.load(root/'common_init.pth',map_location='cpu',weights_only=False)
    net.load_state_dict(init['model']);assert p.g.tensor_digest(net.state_dict())==init['fingerprint'];net.cuda()
    p.g.seed(args.seed);gen=torch.Generator().manual_seed(args.seed)
    means=torch.tensor(json.loads(p.CONSTANTS.read_text())['mean_fields'],device='cuda')[None,:,None,None]
    ds=p.Data('train');assert len(ds)==2142
    loader=DataLoader(ds,batch_size=8,shuffle=True,drop_last=True,num_workers=2,generator=gen)
    optimizer=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',init_scale=512)
    arm=root/args.content;arm.mkdir(exist_ok=True);assert not (arm/'last.pth').exists() or args.resume
    first=1;step=0
    if args.resume:
        old=torch.load(args.resume,map_location='cpu',weights_only=False)
        assert old['protocol']==protocol and old['content']==args.content
        net.load_state_dict(old['model']);optimizer.load_state_dict(old['optimizer']);scaler.load_state_dict(old['scaler'])
        first=old['epoch']+1;step=old['step'];p.g.restore_rng(old['rng_state'],gen)
    for epoch in range(first,11):
        seed=1_000_000+1000*args.seed+epoch;p.g.seed(seed);gen.manual_seed(seed)
        net.train();ids=[];total=0;components={key:0. for key in ['heat','offset','size']};began=time.time()
        for batch in loader:
            ids.extend(batch['image_id'].tolist());batch={k:v.cuda(non_blocking=True) for k,v in batch.items()}
            x=p.content(batch['x'],args.content,means);optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):loss,parts=p.loss_fn(net(x),batch)
            assert torch.isfinite(loss)
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(net.parameters(),.5);assert torch.isfinite(norm)
            step+=1;lr=1e-3*min(1,step/50)*(.1+.9*.5*(1+math.cos(math.pi*step/2670)))
            for group in optimizer.param_groups:group['lr']=lr
            scaler.step(optimizer);scaler.update();total+=float(loss.detach())
            for key in parts:components[key]+=parts[key]
        assert step==epoch*267 and len(ids)==2136
        p.g.dump(arm/f'epoch{epoch}.json',dict(epoch=epoch,step=step,mean_batch_loss=total/267,
            mean_batch_components={k:v/267 for k,v in components.items()},sample_order=ids,
            seconds=time.time()-began,amp_scale=scaler.get_scale(),epoch_seed=seed))
        checkpoint=dict(model=net.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),epoch=epoch,step=step,
            content=args.content,protocol=protocol,rng_state=p.g.rng_state(gen),initial_fingerprint=init['fingerprint'])
        p.g.atomic_save(checkpoint,arm/'last.pth')
        if epoch in [1,3,10]:p.g.atomic_save(checkpoint,arm/f'epoch{epoch}.pth')
        p.g.dump(arm/'progress.json',dict(epoch=epoch,step=step,pid=os.getpid()))
        print('EPOCH',args.content,epoch,total/267,time.time()-began,flush=True)


if __name__=='__main__':main()
