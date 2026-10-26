"""Engineering gates for the independent S2DB-inspired cohort, never AP tuning."""
import copy
import json
import random
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ue_s2db_probe as run
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from rsprompter.uav.s2db_adapter import S2DBAdapter
from rsprompter.uav.s2db_geometry import load_depth,observed_support,prepare_point


class RandomDataset(Dataset):
    def __len__(self):return 35
    def __getitem__(self,i):return i,float(torch.rand(())),float(np.random.rand()),random.random()


def loader_epoch(gen):
    loader=DataLoader(RandomDataset(),batch_size=1,shuffle=True,num_workers=2,generator=gen)
    return [tuple(float(v) for v in batch) for batch in loader]


def cpu_gate(out):
    run.seed(44)
    a=S2DBAdapter.__new__(S2DBAdapter);nn.Module.__init__(a)
    a.key=nn.Linear(32,32);a.value=nn.Linear(32,32)
    a.geom_key=nn.Linear(6,32,bias=False);a.geom_value=nn.Linear(6,32,bias=False)
    a.offset=nn.Sequential(nn.Linear(70,32),nn.GELU(),nn.Linear(32,2))
    nn.init.zeros_(a.offset[-1].weight);nn.init.zeros_(a.offset[-1].bias)
    a.output=nn.Sequential(nn.Linear(32,32),nn.LayerNorm(32,elementwise_affine=False))
    a.joint=nn.Sequential(nn.Conv2d(64,32,3,padding=1),nn.GELU(),nn.Conv2d(32,32,1))
    a.observe=True;a.observation={};a.content='correct';a.read='fixed'
    ix=torch.full((16384,8),-1,dtype=torch.long)
    ix[0,0]=0;ix[5,0]=16384+129;ix[6,0]=4*16384+16383
    geom=torch.zeros((16384,8,6));f=torch.nn.functional.layer_norm(torch.randn(5*16384,32),(32,))
    q=torch.randn(1,32,128,128)
    a.anchor=torch.zeros(16384,8,6);a.anchor[...,2]=10
    a.anchor[0,0,:2]=torch.tensor([1.,1.]);a.anchor[5,0,:2]=torch.tensor([5.,5.]);a.anchor[6,0,:2]=torch.tensor([511.,511.])
    a.depth=torch.full((5,512,512),10.)
    fixed=a._candidates(f,ix,geom,q);a.read='local'
    local=a._candidates(f,ix,geom,q)
    assert all(torch.equal(x,y) for x,y in zip(fixed,local)), 'zero offset gather equivalence'
    with torch.no_grad():a.offset[-1].bias.fill_(.9)
    a._candidates(f,ix,geom,q)
    assert a.observation['fallback']>=1,'edge must fall back'
    a.depth.fill_(float('nan'));fallback=a._candidates(f,ix,geom,q)
    assert all(torch.equal(x,y) for x,y in zip(fixed,fallback)), 'holes must fall back'
    a.depth.fill_(50.);a._candidates(f,ix,geom,q)
    assert a.observation['depth_jump_accepted']>0,'valid discontinuity must stay'
    boxes=torch.tensor([[0.,0.,512.,512.],[-2.,-2.,8.,8.],[508.,508.,516.,516.]])
    for fusion in ('normalized','concat'):
        a.fusion=fusion
        empty=torch.full_like(ix,-1)
        d=a._chunk(q,*fixed,empty,boxes,8)
        assert torch.equal(d,torch.zeros_like(d)),'no support after normalization'
        assert torch.isfinite(a._chunk(q,*fixed,ix,boxes,8)).all()
    # Simulate an E1 boundary: restore the independent generator as well as all
    # parent streams. Worker random augmentations and sample order must replay.
    gen=torch.Generator().manual_seed(44);loader_epoch(gen)
    state=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),loader_generator=gen.get_state())
    expected=loader_epoch(gen)
    other=torch.Generator().manual_seed(987)
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch']);other.set_state(state['loader_generator'])
    replay=loader_epoch(other)
    assert expected==replay
    # The actual training dataset must also replay both ordering and image
    # augmentation, not merely a synthetic sampler.
    ds=run.MinimalSatelliteDataset(run.DATA,'train',augment=True)
    def actual_prefix(generator):
        loader=DataLoader(ds,batch_size=1,shuffle=True,num_workers=2,generator=generator,collate_fn=run.rtmdet_collate_fn)
        rows=[]
        for i,b in enumerate(loader):
            rows.append((b['img_metas'][0]['filename'],b['imgs'].clone()))
            if i==3:break
        return rows
    sample_state=other.get_state();first=actual_prefix(other)
    restored=torch.Generator();restored.set_state(sample_state);second=actual_prefix(restored)
    assert all(x[0]==y[0] and torch.equal(x[1],y[1]) for x,y in zip(first,second))
    report=dict(actual_train_augmentation_resume_prefix_exact=True,actual_train_prefix_images=len(first),zero_offset_gather_exact=True,invalid_depth_fallback=True,bounds_fallback=True,
        valid_depth_jump_retained=True,unsupported_concat_and_normalized_exact=True,roi_edges_finite=True,
        dataloader_resume_order_and_worker_augmentation_exact=True,images=len(replay),
        note='CPU synthetic gate only; full GPU logits, AMP, frozen SHA and resource gates separate')
    report['source_hashes']={p:run.file_sha(run.ROOT/p) for p in run.SOURCES+['scripts/probes/ue_s2db_gate.py']}
    run.dump(Path(out)/'cpu_gate.json',report)
    return report


def gpu_gate(args,model,adapter):
    ds=run.MinimalSatelliteDataset(args.data_root,'train')
    order=np.argsort([len(s['annotations']) for s in ds.samples]);i=int(order[len(order)//2])
    batch=run.rtmdet_collate_fn([ds[i]]);p=run.processed(model,batch,True)
    folder=run.point_folder(ds,i,args.data_root);data=run.payload(folder,args.cache)
    before=run.state_digest(model);assert before==run.FROZEN_SHA
    gen=torch.Generator().manual_seed(44);saved_rng=run.rng_state(gen)
    def draws():return (random.random(),np.random.rand(),torch.rand(3),torch.rand(3,device='cuda'),torch.randperm(9,generator=gen))
    first=draws();run.restore_rng(saved_rng,gen);second=draws()
    assert first[:2]==second[:2] and all(torch.equal(x,y) for x,y in zip(first[2:],second[2:]))
    adapter.eval();adapter.enabled=False;capture=[]
    handle=model.roi_head.mask_head.register_forward_hook(lambda m,a,o:capture.append(o.mask_logits.detach().clone()))
    precision_reports={}
    for amp in (False,True):
        capture.clear()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16,enabled=amp):
            baseline=model.predict(p['inputs'],p['data_samples'],rescale=True)[0].pred_instances
            logits=list(capture);capture.clear();adapter.enabled=True
            features=run.bind(model,adapter,p,data)
            with run.cached_extract(model,features):pred=model.predict(p['inputs'],p['data_samples'],rescale=True)[0].pred_instances
        assert len(logits)==len(capture) and all(torch.equal(x,y) for x,y in zip(logits,capture))
        assert torch.equal(baseline.masks,pred.masks)
        assert torch.equal(baseline.bboxes,pred.bboxes) and torch.equal(baseline.scores,pred.scores)
        precision_reports['amp' if amp else 'fp32']=True
        adapter.clear();adapter.enabled=False
    adapter.enabled=True;adapter.observe=True
    with torch.no_grad():
        features=run.bind(model,adapter,p,data)
        f,ix,g,_=adapter.context;q=adapter.prepared[0]
        old=adapter.read;adapter.read='fixed';fixed=adapter._candidates(f,ix,g,q)
        adapter.read='local';local=adapter._candidates(f,ix,g,q)
        assert all(torch.equal(x,y) for x,y in zip(fixed,local)), 'GPU zero offset unequal'
        adapter.read=old;adapter.clear();adapter.alpha.fill_(.01)
    # Unsupported output identity at nonzero alpha, including raw logits.
    adapter.enabled=False;capture.clear()
    with torch.no_grad():
        baseline=model.predict(p['inputs'],p['data_samples'],rescale=True)[0].pred_instances
        baseline_logits=list(capture);capture.clear();adapter.enabled=True
        empty=list(data);empty[1]=torch.full_like(data[1],-1);empty[5]=torch.zeros_like(data[5])
        features=run.bind(model,adapter,p,tuple(empty))
        with run.cached_extract(model,features):pred=model.predict(p['inputs'],p['data_samples'],rescale=True)[0].pred_instances
    assert len(baseline_logits)==len(capture) and all(torch.equal(x,y) for x,y in zip(baseline_logits,capture))
    assert torch.equal(baseline.masks,pred.masks)
    handle.remove();adapter.clear();adapter.train()
    opt=torch.optim.AdamW([p for p in adapter.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',init_scale=512);steps=[]
    for step in range(3):
        run.seed(44+step);opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            features=run.bind(model,adapter,p,data)
            with run.cached_extract(model,features):losses=model.loss(p['inputs'],p['data_samples'])
            main=sum((v for v in losses.values() if torch.is_tensor(v) and v.requires_grad),adapter.alpha*0)
            aux=adapter.alpha*0
            if adapter.aux_weight:
                aux,_=adapter.auxiliary_loss(run.target_union(p))
                qgrads=torch.autograd.grad(aux,tuple(adapter.query.parameters()),retain_graph=True,allow_unused=True)
                assert all(v is None or torch.count_nonzero(v)==0 for v in qgrads),'aux query gradient leaked'
            loss=main+adapter.aux_weight*aux
        scaler.scale(loss).backward();scaler.unscale_(opt)
        grads={k:float(v.grad.float().norm()) for k,v in adapter.named_parameters() if v.grad is not None}
        assert np.isfinite(list(grads.values())).all()
        norm=torch.nn.utils.clip_grad_norm_(adapter.parameters(),.5);assert torch.isfinite(norm)
        scaler.step(opt);scaler.update();adapter.clear();steps.append(grads)
    assert any('lora_' in k and v>0 for k,v in steps[-1].items())
    if adapter.read=='local':assert any(k.startswith('offset.') and v>0 for k,v in steps[-1].items())
    if adapter.fusion=='concat':assert any(k.startswith('joint.') and v>0 for k,v in steps[-1].items())
    assert all(v.grad is None for v in model.parameters());assert run.state_digest(model)==before
    # Formal FP32 instance path with exactly 250 RoIs, retaining production chunk32.
    adapter.eval();torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        features=run.bind(model,adapter,p,data);x,embedding,pe,high=features
        boxes=torch.tensor([a['bbox'] for a in ds.samples[i]['annotations']],device='cuda',dtype=torch.float32)
        boxes[:,2:]+=boxes[:,:2];boxes=boxes.repeat((250+len(boxes)-1)//len(boxes),1)[:250]
        rois=torch.cat((torch.zeros(250,1,device='cuda'),boxes),1)
        outputs=[]
        for chunk in rois.split(32):
            outputs.append(model.roi_head._mask_forward(x,chunk,image_embeddings=embedding,
                image_positional_embeddings=pe,high_res_features=high)['mask_preds'])
        assert sum(len(v) for v in outputs)==250 and all(torch.isfinite(v).all() for v in outputs)
        peak=torch.cuda.max_memory_allocated()/2**20
    adapter.clear()
    assert run.state_digest(model)==before
    run.dump(args.out/'gpu_gate.json',dict(zero_alpha_logits_exact=precision_reports,
        zero_offset_gpu_exact=True,all_rng_streams_replay_exact=True,no_support_nonzero_alpha_logits_exact=True,
        satellite_state_sha256=before,satellite_frozen_parameters_buffers_exact=True,
        lora_offset_fusion_gradient_reachable=True,aux_query_gradient_blocked=bool(adapter.aux_weight),
        gradient_steps=steps,fp32_resource_rois=250,fp32_resource_peak_MiB=peak,
        gate_read=args.read,gate_fusion=args.fusion,gate_aux_weight=args.aux_weight,image_id=ds.samples[i]['image_id'],
        source_hashes={p:run.file_sha(run.ROOT/p) for p in run.SOURCES+['scripts/probes/ue_s2db_gate.py']}))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    print(json.dumps(cpu_gate(args.out)),flush=True)
