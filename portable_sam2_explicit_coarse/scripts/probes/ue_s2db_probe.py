"""Independent fixed-contract S2DB-inspired factorial runner. Never edits old runs."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parent))
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import ue_v1_adapter as common
from ue_v1_adapter import (ROOT,seed,dump,file_sha,MinimalSatelliteDataset,archive_identity,
    point_folder,processed,cached_extract,state_digest,gt_arrays,serialized,rtmdet_collate_fn,
    _register_and_build,_load_model_state,build_coco_gt_and_dt,run_coco_eval,YAWS)
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from pycocotools import mask as mu
from rsprompter.uav.s2db_adapter import S2DBAdapter
from rsprompter.uav.s2db_geometry import prepare_point,load_depth,observed_support,VERSION

DATA=Path('/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021')
BASE=Path('/data/wangcheng/checkpoint/ue_minimal_single_20261021_EXP-0017/checkpoints/best_model_epoch73.pth')
SOURCES=['scripts/probes/ue_s2db_probe.py','rsprompter/uav/s2db_adapter.py','rsprompter/uav/s2db_geometry.py',
         'scripts/probes/ue_v1_adapter.py','rsprompter/uav/local_adapter_probe.py','rsprompter/uav/local_adapter_variants.py',
         'data/ue_minimal_dataset.py','data/whu_instance_dataset.py']
FROZEN_SHA='c6921c30416595fb994facd3d4d927408be9bdd636399fd669cfbe5968329536'


def tensor_digest(state,trainable_only=None):
    h=hashlib.sha256()
    for name,tensor in state.items():
        if trainable_only is not None and name not in trainable_only:continue
        h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def make_model(args):
    cp=torch.load(args.base,map_location='cpu',weights_only=False);snapshot=cp['config_snapshot']
    assert archive_identity(args.data_root)==snapshot['data_config']['archive_hashes']
    model=_register_and_build(snapshot['model_config'])
    report=_load_model_state(model,cp['model'],allow_nonstrict=False)
    assert not report['missing_keys'] and not report['unexpected_keys']
    for p in model.parameters():p.requires_grad_(False)
    model.eval().cuda()
    adapter=S2DBAdapter(snapshot['runtime_config']['sam2_checkpoint'],args.read,args.fusion,args.content,args.aux_weight).cuda()
    if args.init:
        init=torch.load(args.init,map_location='cpu',weights_only=False)
        adapter.load_state_dict(init['adapter'],strict=True)
        assert init['fingerprint']==tensor_digest(adapter.state_dict())
    adapter.attach(model)
    return model,adapter,snapshot


def payload(folder,cache,appearance='correct',donor=None):
    path=prepare_point(folder,cache)
    with np.load(path) as a:
        index=a['index'].astype(np.int64);geom=a['geom'].astype(np.float32);anchor=a['anchor']
    support=observed_support(index,anchor)
    rgbfolder=folder if donor is None else donor
    rgb=[]
    for yaw in YAWS:
        image=cv2.imread(str(rgbfolder/f'uav/yaw_{yaw:03d}/rgb.png'))
        assert image is not None and image.shape==(512,512,3)
        rgb.append(image[:,:,::-1].transpose(2,0,1).copy())
    rgb=np.stack(rgb).astype(np.float32)
    if appearance=='constant':rgb[:]=np.array([123.675,116.28,103.53],np.float32)[None,:,None,None]
    return tuple(torch.from_numpy(v).cuda() for v in (rgb,index,geom,anchor.astype(np.float32),load_depth(folder),support))


def bind(model,adapter,p,data):
    with torch.no_grad():features=model.extract_feat(p['inputs'])
    adapter.encode(*data,features[3][0])
    return features


def target_union(p):
    masks=gt_arrays(p['data_samples'][0].gt_instances)['rles']
    if not len(masks):return torch.zeros((512,512),device='cuda')
    return torch.from_numpy(mu.decode(masks).any(-1).astype(np.float32)).cuda()


def contract(args,snapshot,adapter):
    return dict(study='s2db_mechanism_v1',args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        base_sha256=file_sha(args.base),archive_hashes=archive_identity(args.data_root),
        geometry_version=VERSION,source_hashes={p:file_sha(ROOT/p) for p in SOURCES},
        initial_fingerprint=tensor_digest(adapter.state_dict()),satellite_state_sha256=FROZEN_SHA,
        init_file_sha256=file_sha(args.init) if args.init else None,base_config_snapshot=snapshot,
        optimizer='AdamW',weight_decay=.01,clip=.5,seed=44,epoch_seed_from=5,
        epoch_seed_formula='1000000+seed*1000+epoch',batch=1,accum=8,epochs=10,updates=2670,
        lr=1e-4,warmup=50,cosine_min_factor=.1,amp=True,ema=False,selection='E10 last',
        eval_precision='fp32',max_dets=[1,10,250],satellite_frozen=True,
        aux_query_detached=True,aux_valid_denominator='unique satellite pixels of retained anchors',
        test_primary=True,init_extra_rng_isolated=True)


def rng_state(gen):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all(),loader_generator=gen.get_state())


def restore_rng(rng,gen):
    random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'])
    torch.cuda.set_rng_state_all(rng['cuda']);gen.set_state(rng['loader_generator'])


def train(args,model,adapter,ds,snapshot):
    assert args.init and len(ds)==2142
    assert not (args.out/'last.pth').exists()
    seed(44)
    base_sha=state_digest(model);assert base_sha==FROZEN_SHA
    initial={k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()}
    optimizer=torch.optim.AdamW([p for p in adapter.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',init_scale=512)
    gen=torch.Generator().manual_seed(44)
    c=contract(args,snapshot,adapter);start_epoch=step=0
    if args.resume_from:
        parent=torch.load(args.resume_from,map_location='cpu',weights_only=False)
        for key in ('base_sha256','archive_hashes','source_hashes','initial_fingerprint','init_file_sha256'):
            assert parent['contract'][key]==c[key],key
        for key in ('read','fusion','content','aux_weight'):
            assert parent['contract']['args'][key]==c['args'][key]
        adapter.load_state_dict(parent['adapter'],strict=True)
        optimizer.load_state_dict(parent['optimizer']);scaler.load_state_dict(parent['scaler'])
        start_epoch,step=parent['epoch'],parent['step'];assert step==267*start_epoch
        restore_rng(parent['rng_state'],gen)
        c['resume_parent']=dict(path=str(args.resume_from),sha256=file_sha(args.resume_from),epoch=start_epoch,step=step)
        del parent
    history=args.out/'train.jsonl'
    if args.resume_from:
        rows=[json.loads(s) for s in history.read_text().splitlines()]
        assert len(rows)==step and [r['update'] for r in rows]==list(range(1,step+1))
    else:assert not history.exists()
    dump(args.out/'frozen_training_contract.json',c)
    for path in SOURCES:
        dest=args.out/'source'/path;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((ROOT/path).read_bytes())
    model.eval();adapter.train()
    with history.open('a' if args.resume_from else 'w') as log:
        for epoch in range(start_epoch+1,11):
            epoch_seed=None
            if epoch>=5:
                epoch_seed=1044000+epoch;seed(epoch_seed);gen.manual_seed(epoch_seed)
            loader=DataLoader(ds,batch_size=1,shuffle=True,num_workers=2,collate_fn=rtmdet_collate_fn,generator=gen)
            optimizer.zero_grad(set_to_none=True);started=time.time();ids=[];aux_sum=0.;aux_pixels=0;main_sum=0.;updates=[]
            for i,batch in enumerate(loader):
                if i>=2136:break
                folder=Path(batch['img_metas'][0]['filename']).parent.parent;ids.append(folder.name)
                p=processed(model,batch,True)
                data=payload(folder,args.cache,'constant' if args.content=='constant' else 'correct')
                with torch.autocast('cuda',dtype=torch.float16):
                    features=bind(model,adapter,p,data)
                    with cached_extract(model,features):losses=model.loss(p['inputs'],p['data_samples'])
                    main=sum((v for v in losses.values() if torch.is_tensor(v) and v.requires_grad),adapter.alpha*0)
                    aux=adapter.alpha*0;count=0
                    if args.aux_weight:
                        aux,count=adapter.auxiliary_loss(target_union(p))
                    total=main+args.aux_weight*aux
                assert torch.isfinite(total)
                scaler.scale(total/8).backward()
                stats=dict(adapter.stats);adapter.clear()
                main_sum+=float(main.detach());aux_sum+=float(aux.detach());aux_pixels+=int(count)
                if (i+1)%8==0:
                    step+=1
                    lr=1e-4*min(1.,step/50)*(.1+.9*.5*(1+np.cos(np.pi*step/2670)))
                    for group in optimizer.param_groups:group['lr']=lr
                    scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(adapter.parameters(),.5)
                    assert torch.isfinite(norm)
                    scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
                    row=dict(epoch=epoch,update=step,images=i+1,loss=float(main.detach()),lr=lr,
                        grad_norm=float(norm),amp_scale=scaler.get_scale(),latest_forward=stats,epoch_seed=epoch_seed)
                    log.write(json.dumps(row)+'\n');log.flush();updates.append(row)
                    if step%10==0:print('TRAIN',json.dumps(row),flush=True)
            assert len(ids)==2136 and step==267*epoch
            assert all(p.grad is None for p in model.parameters())
            assert state_digest(model)==base_sha
            dump(args.out/f'epoch{epoch}_summary.json',dict(epoch=epoch,update=step,seconds=time.time()-started,
                microbatches=len(ids),main_loss_microbatch_mean=main_sum/len(ids),
                aux_bce_microbatch_mean=aux_sum/len(ids),aux_valid_pixel_sum=aux_pixels,
                sample_order=ids,sample_order_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                satellite_state_sha256=base_sha))
            cp=dict(adapter=adapter.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                epoch=epoch,step=step,contract=c,rng_state=rng_state(gen))
            path=args.out/f'epoch{epoch}.pth';tmp=path.with_suffix('.pth.tmp');torch.save(cp,tmp);tmp.replace(path)
            dump(args.out/'progress.json',dict(stage='train',epoch=epoch,update=step,pid=os.getpid()))
            print('EPOCH',epoch,time.time()-started,flush=True)
    delta={k:float((v.detach().cpu()-initial[k]).float().norm()) for k,v in adapter.state_dict().items()}
    dump(args.out/'parameter_changes.json',dict(satellite_unchanged=True,satellite_state_sha256=base_sha,adapter_changes=delta))
    tmp=args.out/'last.pth.tmp';torch.save(dict(adapter=adapter.state_dict(),epoch=10,step=2670,contract=c),tmp);tmp.replace(args.out/'last.pth')


def evaluate(args,model,adapter,ds,mode,out):
    seed(44);model.eval();adapter.eval();adapter.enabled=mode!='disabled';adapter.observe=True
    adapter.aux_weight=0.  # Primary instance inference removes the training-only auxiliary path.
    gt=[];dt=[];metas=[];records=[]
    loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=2,collate_fn=rtmdet_collate_fn)
    with torch.inference_mode():
        for i,batch in enumerate(loader):
            if args.max_images and i>=args.max_images:break
            p=processed(model,batch)
            if adapter.enabled:
                folder=point_folder(ds,i,args.data_root)
                donor=point_folder(ds,(i+1)%len(ds),args.data_root) if mode=='donor' else None
                appearance='constant' if mode=='constant' else 'correct'
                features=bind(model,adapter,p,payload(folder,args.cache,appearance,donor))
            else:features=model.extract_feat(p['inputs']);adapter.context=None
            with cached_extract(model,features):result=model.predict(p['inputs'],p['data_samples'],rescale=True)[0].pred_instances
            g,d=gt_arrays(p['data_samples'][0].gt_instances),gt_arrays(result)
            gt.append(g);dt.append(d);metas.append(batch['img_metas'][0])
            records.append(dict(image_id=ds.samples[i]['image_id'],gt=serialized(g),dt=serialized(d),
                stats=dict(adapter.stats) if adapter.enabled else {},read_observation=dict(adapter.observation)))
            adapter.clear()
            if (i+1)%25==0:print('EVAL',mode,i+1,flush=True)
    cg,cd=build_coco_gt_and_dt(gt,dt,metas,score_key='scores');metrics={}
    for kind in ('segm','bbox'):metrics.update(run_coco_eval(cg,cd,kind,max_dets=[1,10,250]))
    dump(out/'records.json',records);dump(out/'metrics.json',metrics)
    dump(out/'evaluation_identity.json',dict(split=args.eval_split,images=len(records),mode=mode,
        base_sha256=file_sha(args.base),adapter_path=str(args.adapter),adapter_sha256=file_sha(args.adapter) if args.adapter else None,
        score_key='scores',max_dets=[1,10,250],inference_precision='fp32',
        source_hashes={p:file_sha(ROOT/p) for p in SOURCES},donor_preserves_geometry=True))


def parser():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','init','train','eval','gate'])
    p.add_argument('--data-root',type=Path,default=DATA);p.add_argument('--base',type=Path,default=BASE)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cache',type=Path,default=Path('/data/wangcheng/cache/ue_v1_s2db_geometry_v2'))
    p.add_argument('--init',type=Path);p.add_argument('--adapter',type=Path);p.add_argument('--resume-from',type=Path)
    p.add_argument('--read',choices=['fixed','local'],default='fixed')
    p.add_argument('--fusion',choices=['normalized','concat'],default='normalized')
    p.add_argument('--content',choices=['correct','constant','zero'],default='correct')
    p.add_argument('--aux-weight',type=float,choices=[0.,.1],default=0.)
    p.add_argument('--eval-split',choices=['val','test'],default='val')
    p.add_argument('--mode',choices=['correct','constant','donor','disabled'],default='correct')
    p.add_argument('--max-images',type=int,default=0)
    return p


def main():
    args=parser().parse_args();args.out.mkdir(parents=True,exist_ok=True);args.cache.mkdir(parents=True,exist_ok=True);seed(44)
    if args.action=='prepare':
        assert args.eval_split=='val'
        folders=[point_folder(d,i,args.data_root) for d in [MinimalSatelliteDataset(args.data_root,s) for s in ('train','val')] for i in range(len(d))]
        with ThreadPoolExecutor(max_workers=4) as pool:
            for i,path in enumerate(pool.map(lambda f:prepare_point(f,args.cache),folders)):
                if i%100==0:print('PREPARED',i,len(folders),path,flush=True)
        dump(args.out/'geometry_complete.json',dict(points=len(folders),version=VERSION,legacy_index_geom_exact=True))
        return
    if args.adapter:
        cp=torch.load(args.adapter,map_location='cpu',weights_only=False)
        for key in ('read','fusion','content','aux_weight'):setattr(args,key,cp['contract']['args'][key])
    model,adapter,snapshot=make_model(args)
    if args.action=='init':
        assert not (args.out/'common_init.pth').exists()
        state={k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()}
        torch.save(dict(adapter=state,fingerprint=tensor_digest(state),seed=44),args.out/'common_init.pth')
        dump(args.out/'initialization.json',dict(fingerprint=tensor_digest(state),state_keys=list(state),
            source_hashes={p:file_sha(ROOT/p) for p in SOURCES}))
    elif args.action=='train':train(args,model,adapter,MinimalSatelliteDataset(args.data_root,'train',augment=True),snapshot)
    elif args.action=='eval':
        if args.mode!='disabled':
            assert args.adapter;assert cp['epoch']==10 and cp['step']==2670
            assert cp['contract']['base_sha256']==file_sha(args.base)
            adapter.load_state_dict(cp['adapter'],strict=True)
        evaluate(args,model,adapter,MinimalSatelliteDataset(args.data_root,args.eval_split),args.mode,args.out/args.mode)
    else:
        from ue_s2db_gate import gpu_gate
        gpu_gate(args,model,adapter)
    print('COMPLETE',args.action,flush=True)

if __name__=='__main__':main()
