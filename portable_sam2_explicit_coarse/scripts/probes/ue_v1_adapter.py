"""Self-contained v1 frozen-base screening runner, launched by ue_rgbd.sh adapter.

Training reads train only. Explicit evaluation can read val or test after the
2026-10-22 user-authorized protocol amendment. Production contracts stay intact.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'inference'), str(ROOT/'train')]
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
os.environ.setdefault('SAM2_REPO', str(ROOT.parent/'sam2'))
os.environ['RSPROMPTER_LIGHT_IMPORT'] = '0'
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from rsprompter.uav.local_adapter_probe import LocalUAVAdapter, prepare_point, file_sha, YAWS, GEOMETRY_VERSION
from data.ue_minimal_dataset import MinimalSatelliteDataset, archive_identity
from data.loader import rtmdet_collate_fn
from eval_records import _build_data_samples
from model_rebuild import _register_and_build, _load_model_state
from gt_samples import _extract_instances_numpy
from utils.coco_eval_utils import build_coco_gt_and_dt, run_coco_eval, _masks_to_rles


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2)+'\n')
    tmp.replace(path)


def seed(value):
    torch.manual_seed(value); np.random.seed(value); random.seed(value)
    torch.set_num_threads(2); cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_model(args):
    cp = torch.load(args.base, map_location='cpu', weights_only=False)
    snapshot = cp['config_snapshot']
    if archive_identity(args.data_root) != snapshot['data_config']['archive_hashes']:
        raise ValueError('v1 data does not match sealed base identity')
    model = _register_and_build(snapshot['model_config'])
    report = _load_model_state(model, cp['model'], allow_nonstrict=False)
    assert not report['missing_keys'] and not report['unexpected_keys']
    for p in model.parameters(): p.requires_grad_(False)
    model.eval().cuda()
    if args.source_mode == 'unit':
        assert args.route == 'highres'
        from rsprompter.uav.local_adapter_variants import UnitLocalUAVAdapter
        adapter = UnitLocalUAVAdapter(snapshot['runtime_config']['sam2_checkpoint']).cuda()
    elif args.route == 'highres' and args.source_mode == 'fine':
        adapter = LocalUAVAdapter(snapshot['runtime_config']['sam2_checkpoint']).cuda()
    else:
        from rsprompter.uav.local_adapter_variants import VariantLocalUAVAdapter
        adapter = VariantLocalUAVAdapter(snapshot['runtime_config']['sam2_checkpoint'],args.route,args.source_mode).cuda()
    adapter.attach(model)
    del cp
    return model, adapter, snapshot


def point_folder(ds, i, root):
    # COCO names encode archive point IDs; do not assume position == ID.
    sample = ds.samples[i]
    path = Path(sample['img_path']) if 'img_path' in sample else None
    if path is None:
        path = Path(sample['file_name']) if 'file_name' in sample else None
    if path is not None:
        if not path.is_absolute(): path = Path(root)/path
        return path.parent.parent
    return Path(root)/'points'/f"{sample['image_id']:06d}"


def payload(folder, cache, appearance='correct', donor=None):
    name = prepare_point(folder, cache)
    with np.load(name) as a:
        index = torch.from_numpy(a['index'].astype(np.int64)).cuda()
        geom = torch.from_numpy(a['geom'].astype(np.float32)).cuda()
    if donor is not None: folder = donor
    rgb = []
    for yaw in YAWS:
        a = cv2.imread(str(folder/f'uav/yaw_{yaw:03d}/rgb.png'))
        assert a is not None and a.shape == (512,512,3)
        rgb.append(a[:,:,::-1].transpose(2,0,1).copy())
    rgb = torch.from_numpy(np.stack(rgb)).float().cuda()
    if appearance == 'constant':
        rgb[:] = torch.tensor([123.675,116.28,103.53],device='cuda').view(1,3,1,1)
    return rgb, index, geom


def processed(model, batch, training=False):
    imgs, samples = _build_data_samples(batch, torch.device('cuda'))
    return model.data_preprocessor(dict(inputs=imgs,data_samples=samples),training=training)


def bind(model, adapter, p, data):
    # Baseline has no trainable input path; no_grad saves memory and ensures frozen detector.
    with torch.no_grad():
        features = model.extract_feat(p['inputs'])
    x, embedding, pe, high = features
    assert high is not None and high[0].shape[-2:] == (128,128)
    adapter.encode(*data, high[0])
    return features


@contextlib.contextmanager
def cached_extract(model, features):
    # Avoid executing the satellite backbone twice; temporary instance override only.
    original = model.extract_feat
    model.extract_feat = lambda inputs: features
    try: yield
    finally: model.extract_feat = original


def state_digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def gt_arrays(inst):
    return _extract_instances_numpy(inst,(512,512),encode_masks=True)


def serialized(a):
    return {k: v.tolist() if isinstance(v,np.ndarray) else v for k,v in a.items()}


def evaluate(args, model, adapter, ds, mode, out):
    seed(args.seed)
    model.eval(); adapter.eval()
    adapter.enabled = mode != 'disabled'
    all_gt, all_dt, metas, records = [], [], [], []
    loader = DataLoader(ds,batch_size=1,shuffle=False,num_workers=2,collate_fn=rtmdet_collate_fn)
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            if args.max_images and i >= args.max_images: break
            p = processed(model,batch)
            if adapter.enabled:
                folder = point_folder(ds,i,args.data_root)
                donor = point_folder(ds,(i+1)%len(ds),args.data_root) if mode == 'donor' else None
                data = payload(folder,args.cache,args.appearance if mode != 'constant' else 'constant',donor)
                features = bind(model,adapter,p,data)
            else:
                features = model.extract_feat(p['inputs']); adapter.context = None
            with cached_extract(model,features):
                result = model.predict(p['inputs'],p['data_samples'],rescale=True)[0].pred_instances
            target = gt_arrays(p['data_samples'][0].gt_instances)
            pred = gt_arrays(result)
            all_gt.append(target); all_dt.append(pred); metas.append(batch['img_metas'][0])
            records.append(dict(image_id=ds.samples[i]['image_id'],gt=serialized(target),dt=serialized(pred),
                                stats=adapter.stats if adapter.enabled else {}))
            adapter.context = None
            if (i+1)%25 == 0: print('EVAL',mode,i+1,flush=True)
    cg, cd = build_coco_gt_and_dt(all_gt,all_dt,metas,score_key='scores')
    metrics = {}
    for kind in ('segm','bbox'): metrics.update(run_coco_eval(cg,cd,kind,max_dets=[1,10,250]))
    dump(out/'metrics.json',metrics); dump(out/'records.json',records)
    dump(out/'evaluation_identity.json',dict(split=getattr(args,'eval_split','val'),images=len(records),
        base_sha256=file_sha(args.base),adapter_path=str(args.adapter) if args.adapter else None,
        adapter_sha256=file_sha(args.adapter) if args.adapter else None,mode=mode,
        score_key='scores',max_dets=[1,10,250],inference_precision='fp32',
        runner_sha256=file_sha(__file__)))
    print('METRICS',mode,json.dumps(metrics),flush=True)
    return metrics


def contract(args, snapshot):
    diff = os.popen('git diff HEAD').read()
    return dict(base=str(args.base),base_sha256=file_sha(args.base),data_root=str(args.data_root),
        archive_hashes=archive_identity(args.data_root),base_config_snapshot=snapshot,
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        geometry_version=GEOMETRY_VERSION,script_sha256=file_sha(__file__),
        adapter_sha256=file_sha(ROOT/'rsprompter/uav/local_adapter_probe.py'),
        variants_sha256=file_sha(ROOT/'rsprompter/uav/local_adapter_variants.py'),
        git_commit=os.popen('git rev-parse HEAD').read().strip(),git_status=os.popen('git status --porcelain').read(),
        git_diff_sha256=hashlib.sha256(diff.encode()).hexdigest(),scope='v1 train-only adaptation; explicit fixed-checkpoint val/test evaluation',
        checkpoint_selection='last',optimizer='AdamW',effective_batch=args.accum,
        satellite_frozen=True,ema=False)


def smoke(args,model,adapter,ds):
    # Pick median-count train image without consulting validation outcomes.
    order = np.argsort([len(s['annotations']) for s in ds.samples])
    i = int(order[len(order)//2]); batch = rtmdet_collate_fn([ds[i]])
    p = processed(model,batch,True)
    folder = point_folder(ds,i,args.data_root)
    data = payload(folder,args.cache,args.appearance)
    before = state_digest(model)
    adapter.eval(); adapter.enabled = False
    captured=[]
    capture_handle=model.roi_head.mask_head.register_forward_hook(lambda m,a,o:captured.append(o.mask_logits.detach().clone()))
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        baseline = model.predict(p['inputs'],p['data_samples'],rescale=True)[0].pred_instances
        baseline_logits=list(captured);captured.clear()
        features = bind(model,adapter,p,data)
        adapter.enabled = True
        with cached_extract(model,features):
            replay = model.predict(p['inputs'],p['data_samples'],rescale=True)[0].pred_instances
    assert len(baseline_logits)==len(captured) and all(torch.equal(a,b) for a,b in zip(baseline_logits,captured)), 'Zero-gain logits changed'
    capture_handle.remove()
    assert torch.equal(baseline.masks,replay.masks)
    assert torch.equal(baseline.bboxes,replay.bboxes) and torch.equal(baseline.scores,replay.scores)
    opt = torch.optim.AdamW([p for p in adapter.parameters() if p.requires_grad],lr=args.lr,weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda',init_scale=512)
    rows=[]
    adapter.train()
    for step in range(3):
        seed(args.seed+step); opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            features = bind(model,adapter,p,data)
            with cached_extract(model,features): losses = model.loss(p['inputs'],p['data_samples'])
            differentiable = {k: v for k,v in losses.items() if torch.is_tensor(v) and v.requires_grad}
            loss = sum(differentiable.values())
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        grads = {k:float(v.grad.float().norm()) for k,v in adapter.named_parameters() if v.grad is not None}
        assert np.isfinite(list(grads.values())).all()
        norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(),.5)
        assert torch.isfinite(norm)
        scaler.step(opt); scaler.update()
        rows.append(dict(step=step,loss=float(loss.detach()),grads=grads,stats=dict(adapter.stats)))
        adapter.context=None
        print('SMOKE_STEP',step,float(loss.detach()),float(norm),float(adapter.alpha),flush=True)
    assert rows[0]['grads']['alpha'] > 0
    assert any('lora_' in k and v > 0 for k,v in rows[-1]['grads'].items())
    assert all(p.grad is None for p in model.parameters())
    assert state_digest(model) == before
    # Explicit no-candidate identity at nonzero alpha, in addition to zero-init replay.
    adapter.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        empty = (data[0],torch.full_like(data[1],-1),torch.zeros_like(data[2]))
        features=bind(model,adapter,p,empty)
        with cached_extract(model,features): no_support=model.predict(p['inputs'],p['data_samples'],rescale=True)[0].pred_instances
    assert torch.equal(baseline.masks,no_support.masks)
    report=dict(zero_init_logits_exact=True,zero_init_replay_exact=True,no_candidate_replay_exact=True,satellite_state_unchanged=True,
        frozen_parameters_no_grad=True,lora_gradient_nonzero=True,steps=rows,
        peak_MiB=torch.cuda.max_memory_allocated()/2**20,image_id=ds.samples[i]['image_id'])
    dump(args.out/'smoke.json',report)


def train(args,model,adapter,ds,snapshot):
    assert not (args.out/'last.pth').exists(), 'Refuse overwriting existing trained adapter'
    base_digest=state_digest(model)
    initial={k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()}
    optimizer=torch.optim.AdamW([p for p in adapter.parameters() if p.requires_grad],lr=args.lr,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',init_scale=512)
    updates_per_epoch=len(ds)//args.accum
    total=updates_per_epoch*args.epochs;step=0;start_epoch=0
    frozen_contract=contract(args,snapshot)
    resume=getattr(args,'resume_from',None)
    reseed_from=int(getattr(args,'reseed_from_epoch',0))
    if resume:
        parent=torch.load(resume,map_location='cpu',weights_only=False)
        assert parent['contract']['base_sha256']==frozen_contract['base_sha256']
        assert parent['contract']['archive_hashes']==frozen_contract['archive_hashes']
        saved=parent['contract']['args']
        for key,default in [('epochs',10),('accum',8),('seed',44),('lr',1e-4),
                            ('route','highres'),('source_mode','fine'),('appearance','correct')]:
            assert saved.get(key,default)==getattr(args,key,default),(key,'resume identity mismatch')
        if 'content' in saved:assert saved['content']==args.content
        start_epoch=int(parent['epoch']);step=int(parent['step'])
        assert step==start_epoch*updates_per_epoch
        adapter.load_state_dict(parent['adapter'],strict=True)
        optimizer.load_state_dict(parent['optimizer']);scaler.load_state_dict(parent['scaler'])
        if not parent.get('rng_state'):
            assert reseed_from==start_epoch+1,'Legacy resume requires explicit deterministic new epoch stream'
        else:
            rng=parent['rng_state'];random.setstate(rng['python']);np.random.set_state(rng['numpy'])
            torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state_all(rng['cuda'])
        frozen_contract['recovery']=dict(parent_checkpoint=str(resume),parent_sha256=file_sha(resume),
            parent_epoch=start_epoch,parent_step=step,parent_contract=parent['contract'],
            exact_parent_rng_available=bool(parent.get('rng_state')),reseed_from_epoch=reseed_from,
            epoch_seed_formula='1000000+seed*1000+epoch',initial_parameter_reference='fresh initialization before resume load')
        del parent
    frozen_contract['satellite_state_sha256']=base_digest
    dump(args.out/'frozen_training_contract.json',frozen_contract)
    (args.out/'ue_v1_adapter.source.py').write_text(Path(__file__).read_text())
    (args.out/'local_adapter_probe.source.py').write_text((ROOT/'rsprompter/uav/local_adapter_probe.py').read_text())
    (args.out/'local_adapter_variants.source.py').write_text((ROOT/'rsprompter/uav/local_adapter_variants.py').read_text())
    gen=torch.Generator().manual_seed(args.seed)
    def loader_for(generator):
        return DataLoader(ds,batch_size=1,shuffle=True,num_workers=2,collate_fn=rtmdet_collate_fn,generator=generator)
    loader=loader_for(gen)
    history=args.out/'train.jsonl'
    if resume:
        old=[json.loads(v) for v in history.read_text().splitlines()]
        assert len(old)==step and [v['update'] for v in old]==list(range(1,step+1))
    else:assert not history.exists(),'Refuse overwriting prior training history'
    model.eval();adapter.train()
    dump(args.out/'progress.json',dict(stage='train',epoch=start_epoch,update=step,pid=os.getpid()))
    with history.open('a' if resume else 'w') as log:
        for epoch in range(start_epoch+1,args.epochs+1):
            epoch_seed=None
            if reseed_from and epoch>=reseed_from:
                epoch_seed=1000000+args.seed*1000+epoch
                seed(epoch_seed);gen=torch.Generator().manual_seed(epoch_seed);loader=loader_for(gen)
            optimizer.zero_grad(set_to_none=True);start=time.time();summary=[]
            for i,batch in enumerate(loader):
                if i>=updates_per_epoch*args.accum:break
                p=processed(model,batch,True)
                folder=Path(batch['img_metas'][0]['filename']).parent.parent
                data=payload(folder,args.cache,args.appearance)
                with torch.autocast('cuda',dtype=torch.float16):
                    features=bind(model,adapter,p,data)
                    with cached_extract(model,features):losses=model.loss(p['inputs'],p['data_samples'])
                    loss=sum(v for v in losses.values() if torch.is_tensor(v) and v.requires_grad)
                assert torch.isfinite(loss),'Nonfinite loss'
                scaler.scale(loss/args.accum).backward()
                stats=dict(adapter.stats);adapter.context=None
                if (i+1)%args.accum==0:
                    step+=1
                    lr=args.lr*min(1.,step/50)*(.1+.9*.5*(1+np.cos(np.pi*step/total)))
                    for group in optimizer.param_groups:group['lr']=lr
                    scaler.unscale_(optimizer)
                    norm=torch.nn.utils.clip_grad_norm_(adapter.parameters(),.5)
                    assert torch.isfinite(norm),'Nonfinite gradient'
                    scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
                    row=dict(epoch=epoch,update=step,images=i+1,loss=float(loss.detach()),lr=lr,
                        grad_norm=float(norm),amp_scale=scaler.get_scale(),latest_forward=stats,
                        epoch_seed=epoch_seed,recovery_parent_epoch=start_epoch if resume else None)
                    log.write(json.dumps(row)+'\n');log.flush();summary.append(row['loss'])
                    if step%10==0:print('TRAIN',json.dumps(row),flush=True)
            assert state_digest(model)==base_digest,'Frozen base changed'
            print('EPOCH',epoch,'seconds',time.time()-start,'loss',np.mean(summary),flush=True)
            rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                     cuda=torch.cuda.get_rng_state_all(),loader_generator=gen.get_state())
            path=args.out/f'epoch{epoch}.pth';tmp=path.with_suffix('.pth.tmp')
            torch.save(dict(adapter=adapter.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                epoch=epoch,step=step,contract=frozen_contract,rng_state=rng),tmp);tmp.replace(path)
            dump(args.out/'progress.json',dict(stage='train',epoch=epoch,update=step,pid=os.getpid()))
    assert step==total and state_digest(model)==base_digest
    delta={k:float((v.detach().cpu()-initial[k]).float().norm()) for k,v in adapter.state_dict().items()}
    dump(args.out/'parameter_changes.json',dict(satellite_unchanged=True,satellite_state_sha256=base_digest,
        adapter_changes=delta,initial_reference='fresh initialization before resume load',recovery_parent_epoch=start_epoch))
    tmp=args.out/'last.pth.tmp'
    torch.save(dict(adapter=adapter.state_dict(),epoch=args.epochs,step=step,contract=frozen_contract),tmp)
    tmp.replace(args.out/'last.pth')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['prepare','smoke','train','eval'])
    p.add_argument('--data-root',type=Path,default=Path('/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021'))
    p.add_argument('--base',type=Path,default=Path('/data/wangcheng/checkpoint/ue_minimal_single_20261021_EXP-0017/checkpoints/best_model_epoch73.pth'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cache',type=Path,default=Path('/data/wangcheng/cache/ue_v1_local_adapter_geometry'))
    p.add_argument('--appearance',choices=['correct','constant'],default='correct')
    p.add_argument('--route',choices=['highres','early'],default='highres')
    p.add_argument('--source-mode',choices=['fine','contextual','unit'],default='fine')
    p.add_argument('--mode',choices=['correct','constant','donor','disabled'],default='correct')
    p.add_argument('--adapter',type=Path)
    p.add_argument('--resume-from',type=Path);p.add_argument('--reseed-from-epoch',type=int,default=0)
    p.add_argument('--epochs',type=int,default=10);p.add_argument('--accum',type=int,default=8)
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--seed',type=int,default=44)
    p.add_argument('--max-images',type=int,default=0)
    p.add_argument('--eval-split',choices=['val','test'],default='val')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);args.cache.mkdir(parents=True,exist_ok=True)
    if args.action!='eval' and args.eval_split!='val':raise ValueError('eval-split is evaluation-only; training always uses train')
    seed(args.seed)
    if args.action=='prepare':
        ds=[MinimalSatelliteDataset(args.data_root,s) for s in ('train','val')]
        folders=[point_folder(d,i,args.data_root) for d in ds for i in range(len(d))]
        with ThreadPoolExecutor(max_workers=6) as pool:
            for i,path in enumerate(pool.map(lambda f:prepare_point(f,args.cache),folders)):
                if i%100==0:print('PREPARED',i,len(folders),path,flush=True)
        dump(args.out/'geometry_complete.json',dict(points=len(folders),version=GEOMETRY_VERSION,cache=str(args.cache)))
        return
    adapter_checkpoint=None
    if args.adapter:
        adapter_checkpoint=torch.load(args.adapter,map_location='cpu',weights_only=False)
        saved_args=adapter_checkpoint['contract']['args']
        args.route=saved_args.get('route','highres')
        args.source_mode=saved_args.get('source_mode','fine')
        args.appearance=saved_args.get('appearance','correct')
    model,adapter,snapshot=make_model(args)
    dump(args.out/f'contract_{args.action}_{args.mode}.json',contract(args,snapshot))
    if args.adapter:
        cp=adapter_checkpoint
        assert cp['contract']['base_sha256']==file_sha(args.base)
        adapter.load_state_dict(cp['adapter'],strict=True)
    if args.action=='smoke':smoke(args,model,adapter,MinimalSatelliteDataset(args.data_root,'train'))
    elif args.action=='train':train(args,model,adapter,MinimalSatelliteDataset(args.data_root,'train',augment=True),snapshot)
    else:
        if args.mode!='disabled' and not args.adapter:raise ValueError('Evaluation requires trained adapter')
        evaluate(args,model,adapter,MinimalSatelliteDataset(args.data_root,args.eval_split),args.mode,args.out/args.mode)
    print('COMPLETE',args.action,flush=True)


if __name__=='__main__':main()
