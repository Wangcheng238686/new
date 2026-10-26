"""Native geometry vs matched zero-content residual, using the same sealed base/loss."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/probes'),str(ROOT)]
import ue_v1_adapter as common
import torch
from rsprompter.uav.native_geometry_probe import NativeGeometryAdapter,prepare_native,VERSION


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','smoke','train','eval'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=Path('/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021'))
    p.add_argument('--base',type=Path,default=Path('/data/wangcheng/checkpoint/ue_minimal_single_20261021_EXP-0017/checkpoints/best_model_epoch73.pth'))
    p.add_argument('--cache',type=Path,default=Path('/data/wangcheng/cache/ue_v1_native_geometry'))
    p.add_argument('--content',choices=['geometry','zero'],default='geometry');p.add_argument('--adapter',type=Path)
    p.add_argument('--resume-from',type=Path);p.add_argument('--reseed-from-epoch',type=int,default=0)
    p.add_argument('--epochs',type=int,default=10);p.add_argument('--accum',type=int,default=8)
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--seed',type=int,default=44)
    p.add_argument('--max-images',type=int,default=0)
    p.add_argument('--eval-split',choices=['val','test'],default='val')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);a.cache.mkdir(parents=True,exist_ok=True)
    if a.action!='eval' and a.eval_split!='val':raise ValueError('eval-split is evaluation-only; training always uses train')
    common.seed(a.seed)
    a.route='highres';a.source_mode='fine';a.appearance='correct';a.mode='correct'
    if a.action=='prepare':
        folders=[]
        for split in ('train','val'):
            ds=common.MinimalSatelliteDataset(a.data_root,split)
            folders.extend(common.point_folder(ds,i,a.data_root) for i in range(len(ds)))
        with ThreadPoolExecutor(max_workers=4) as pool:
            for i,v in enumerate(pool.map(lambda f:prepare_native(f,a.cache),folders)):
                if i%100==0:print('PREPARED',i,len(folders),str(v),flush=True)
        common.dump(a.out/'geometry_complete.json',dict(version=VERSION,points=len(folders)));return
    cp=None
    if a.adapter:
        cp=torch.load(a.adapter,map_location='cpu',weights_only=False)
        a.content=cp['contract']['args']['content']
    model,unused,snapshot=common.make_model(a)
    for handle in unused.handles:handle.remove()
    del unused;torch.cuda.empty_cache()
    # Same native adapter initialization for geometry and zero-content controls.
    common.seed(a.seed+1)
    adapter=NativeGeometryAdapter().cuda();adapter.attach(model)
    common.seed(a.seed)  # keep training sampling seed44 after matched seed45 adapter initialization
    print('NATIVE_ACTIVE_TRAINABLE_PARAMETERS',sum(p.numel() for p in adapter.parameters() if p.requires_grad), 'RGB encoder discarded',flush=True)
    if cp:
        assert cp['contract']['base_sha256']==common.file_sha(a.base)
        adapter.load_state_dict(cp['adapter'],strict=True)
        del cp
    def payload(folder,cache,appearance='correct',donor=None):
        path=prepare_native(folder,cache)
        with common.np.load(path) as data:
            support=torch.from_numpy(data['support'].astype('float32'))[None,None].cuda()
            fields=torch.from_numpy(data['fields'].astype('float32'))[None].cuda()
        if a.content=='zero':fields.zero_()
        return None,support,fields
    common.payload=payload
    original_contract=common.contract
    def contract(args,snap):
        c=original_contract(args,snap)
        c.update(probe_family='native_geometry',geometry_version=VERSION,
                 native_source_sha256=common.file_sha(ROOT/'rsprompter/uav/native_geometry_probe.py'),
                 native_runner_sha256=common.file_sha(__file__),rgb_encoder_used=False,content=a.content,
                 trainable_parameters={n:p.numel() for n,p in adapter.named_parameters() if p.requires_grad})
        return c
    common.contract=contract
    common.dump(a.out/f'contract_{a.action}.json',contract(a,snapshot))
    for path in (Path(__file__),ROOT/'rsprompter/uav/native_geometry_probe.py'):
        (a.out/(path.name+'.source')).write_text(path.read_text())
    if a.action=='train':
        common.train(a,model,adapter,common.MinimalSatelliteDataset(a.data_root,'train',augment=True),snapshot)
    elif a.action=='eval':
        assert a.adapter
        common.evaluate(a,model,adapter,common.MinimalSatelliteDataset(a.data_root,a.eval_split),'correct',a.out/'correct')
    else:
        ds=common.MinimalSatelliteDataset(a.data_root,'train');batch=common.rtmdet_collate_fn([ds[0]])
        sample=common.processed(model,batch);data=payload(common.point_folder(ds,0,a.data_root),a.cache)
        before=common.state_digest(model);adapter.enabled=False
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            base=model.predict(sample['inputs'],sample['data_samples'],rescale=True)[0].pred_instances
            features=common.bind(model,adapter,sample,data);adapter.enabled=True
            with common.cached_extract(model,features):replay=model.predict(sample['inputs'],sample['data_samples'],rescale=True)[0].pred_instances
        assert torch.equal(base.masks,replay.masks) and torch.equal(base.bboxes,replay.bboxes) and torch.equal(base.scores,replay.scores)
        opt=torch.optim.AdamW(adapter.parameters(),lr=a.lr);scaler=torch.amp.GradScaler('cuda',init_scale=512)
        rows=[];adapter.train()
        for step in range(3):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                features=common.bind(model,adapter,sample,data)
                with common.cached_extract(model,features):losses=model.loss(sample['inputs'],sample['data_samples'])
                loss=sum(v for v in losses.values() if torch.is_tensor(v) and v.requires_grad)
            scaler.scale(loss).backward();scaler.unscale_(opt)
            g={k:float(p.grad.float().norm()) for k,p in adapter.named_parameters() if p.grad is not None}
            assert all(common.np.isfinite(v) for v in g.values())
            norm=torch.nn.utils.clip_grad_norm_(adapter.parameters(),.5);assert torch.isfinite(norm)
            scaler.step(opt);scaler.update();adapter.context=None
            rows.append(dict(step=step,loss=float(loss.detach()),grads=g,stats=dict(adapter.stats)))
        assert rows[0]['grads']['alpha']>0 and rows[-1]['grads']['mixer.0.weight']>0
        assert common.state_digest(model)==before and all(p.grad is None for p in model.parameters())
        adapter.eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            empty=(None,torch.zeros_like(data[1]),torch.zeros_like(data[2]))
            features=common.bind(model,adapter,sample,empty)
            with common.cached_extract(model,features):replay=model.predict(sample['inputs'],sample['data_samples'],rescale=True)[0].pred_instances
        assert torch.equal(base.masks,replay.masks)
        common.dump(a.out/'smoke.json',dict(zero_init_exact=True,no_support_exact=True,satellite_unchanged=True,
            mixer_gradient_nonzero=True,steps=rows,peak_MiB=torch.cuda.max_memory_allocated()/2**20))
    print('COMPLETE',a.action,a.content,flush=True)


if __name__=='__main__':main()
