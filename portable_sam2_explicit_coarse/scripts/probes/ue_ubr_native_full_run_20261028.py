"""Isolated native A/B/C full100 runner; fresh foundation, validation-best test.

Reuses sealed native model, actual candidate consumer and synchronized AMP update
math. No short-study checkpoint is a permitted initializer. One arm per root
subdirectory, four ranks per experiment; owner schedules the three experiments.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
WORK=Path(__file__).resolve().parents[2];REPO=WORK.parent
sys.path[:0]=[str(REPO/'sam2'),str(WORK),str(WORK/'scripts/probes')]
import ue_ubr_native_run_20261027 as native
engine=native.engine;core=engine.core;previous=engine.previous
CONFIG=WORK/'configs/ablations/ue_g_ubr_native_full_protocol_20261028.json'
PROTOCOL=json.loads(CONFIG.read_text())
ROOT=Path('/data/wangcheng/checkpoint/ue_g_ubr_native_full100_20261028')
RETAIN=(1,10,25,50,75,100)


def read(path):return json.loads(Path(path).read_text())
def dump(path,value):core.C.dump(path,value)
def dataset_digest(ds):
    # JSON canonicalization is explicit so the producer need not construct a GPU model.
    return hashlib.sha256(json.dumps(ds.samples,sort_keys=True,default=str).encode()).hexdigest()


def prepare_orders(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'prepare_orders.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _prepare_orders_locked(root)


def _prepare_orders_locked(root):
    """CPU-only deterministic common orders; existing bytes must match exactly."""
    root=Path(root);folder=root/'orders';folder.mkdir(parents=True,exist_ok=True)
    ds=core.C.MinimalSatelliteDataset(core.previous.DATA,'train',augment=False)
    assert len(ds)==2142
    manifest=dict(algorithm='torch.randperm(seed=44+epoch)',dataset_samples_sha256=dataset_digest(ds),epochs={})
    def immutable(path,value):
        content=(json.dumps(value,indent=2,sort_keys=True)+'\n').encode()
        if path.exists():
            if path.read_bytes()!=content:raise RuntimeError('Existing order artifact differs: '+str(path))
        else:
            # O_EXCL prevents racing producers from silently replacing another order.
            try:
                with path.open('xb') as stream:stream.write(content)
            except FileExistsError:
                if path.read_bytes()!=content:raise RuntimeError('Concurrent order producer conflict')
    for epoch in range(1,101):
        indices=torch.randperm(2142,generator=torch.Generator().manual_seed(44+epoch))[:2136].tolist()
        row=dict(epoch=epoch,seed=44+epoch,dataset_indices=indices,sample_order=[int(ds.samples[i]['image_id']) for i in indices])
        path=folder/('epoch%d.json'%epoch);immutable(path,row)
        manifest['epochs'][str(epoch)]=dict(path=path.name,sha256=core.C.file_sha(path))
    immutable(folder/'manifest.json',manifest)
    return manifest


def verify_orders(root,ds):
    manifest_path=root/'orders/manifest.json';manifest=read(manifest_path)
    assert len(ds)==2142 and manifest['algorithm']=='torch.randperm(seed=44+epoch)'
    assert manifest['dataset_samples_sha256']==dataset_digest(ds)
    assert set(manifest['epochs'])==set(map(str,range(1,101)))
    hashes={str(manifest_path.resolve()):core.C.file_sha(manifest_path)}
    for epoch in range(1,101):
        item=manifest['epochs'][str(epoch)];path=root/'orders'/item['path']
        assert path.resolve().parent==(root/'orders').resolve(),'Order path must be direct child'
        assert core.C.file_sha(path)==item['sha256']
        row=read(path);indices=torch.randperm(2142,generator=torch.Generator().manual_seed(44+epoch))[:2136].tolist()
        ids=[int(ds.samples[i]['image_id']) for i in indices]
        assert row['epoch']==epoch and row['seed']==44+epoch and row['dataset_indices']==indices and row['sample_order']==ids
        assert len(set(ids))==2136
        hashes[str(path.resolve())]=item['sha256']
    return hashes


def epoch_loader(rt,ds,root,epoch):
    row=read(root/'orders'/read(root/'orders/manifest.json')['epochs'][str(epoch)]['path'])
    indices=row['dataset_indices'];rank=dist.get_rank()
    local=[indices[start+rank*2+offset] for start in range(0,2136,8) for offset in range(2)]
    return DataLoader(ds,batch_size=1,sampler=local,num_workers=2,generator=rt.generator,collate_fn=core.C.rtmdet_collate_fn),row['sample_order']


def source_inventory():
    import sam2
    assert (REPO/'sam2') in Path(sam2.__file__).resolve().parents,'SAM2 editable install escaped pinned worktree'
    # The editable installation must never silently supply another checkout.
    for name,module in tuple(sys.modules.items()):
        path=getattr(module,'__file__',None)
        if path and (name=='sam2' or name.startswith('sam2.')):
            assert REPO in Path(path).resolve().parents,('SAM2 source escaped runtime',name,path)
    result=native.source_inventory()
    for path in (Path(__file__).resolve(),CONFIG):result[str(path.relative_to(REPO))]=core.C.file_sha(path)
    return dict(sorted(result.items()))


def identity(rt,args,sources,ds):
    return dict(arm=args.arm,epochs=100,purpose=PROTOCOL['purpose'],protocol=PROTOCOL,allocator_cap_mib=args.allocator_cap_mib,
        source_hashes=sources,initialization={k:v for k,v in rt.initialization_audit.items() if k!='arm'},
        native_evidence_port={'A_nohead':'absent','B_zero':'zero RGB+geometry','C_correct':'correct RGB+geometry'}[args.arm],
        data_identity=core.C.archive_identity(core.previous.DATA),input_cache_manifest_sha256=core.C.file_sha(core.OUT/'input_cache_sha256.json'),
        measured_cache='/data/wangcheng/cache/ue_v1_native_geometry',anchor_cache='/data/wangcheng/cache/ue_v1_s2db_geometry_v2',
        dataset_samples_sha256=dataset_digest(ds),source_order_sha256=verify_orders(args.root,ds),
        engineering_receipt_sha256=native.verify_receipt(args.engineering_receipt,sources))


def save(rt,path,epoch,step,frozen,cpu,best):
    local=previous.rank_rng(rt);states=[None]*4 if dist.get_rank()==0 else None
    dist.gather_object(local,states,dst=0,group=cpu)
    if dist.get_rank()==0:
        core.previous.atomic_save(dict(model=rt.bare.state_dict(),optimizer=rt.optimizer.state_dict(),scaler=rt.scaler.state_dict(),rank_rng=states,
            epoch=epoch,step=step,contract=frozen,world_size=4,checkpoint_phase='after_validation' if epoch else 'gate_no_validation',validation_performed=epoch>0,gate_only=epoch==0,best=best),path)
    dist.barrier()


def restore(rt,path,frozen,training):
    cp=torch.load(path,map_location='cpu',weights_only=False)
    assert cp['contract']==frozen and cp['contract']['arm']==frozen['arm'] and cp['world_size']==4
    assert len(cp['rank_rng'])==4
    if cp.get('gate_only'):
        assert cp['epoch']==cp['step']==0 and cp['checkpoint_phase']=='gate_no_validation' and not cp['validation_performed']
    else:assert cp['checkpoint_phase']=='after_validation' and cp['validation_performed'] and cp['epoch']>0
    assert cp['step']==cp['epoch']*267 and 0<=cp['epoch']<=100
    assert set(cp['best'])=={'epoch','score','metric'} and cp['best']['metric']=='segm/mAP'
    rt.bare.load_state_dict(cp['model'],strict=True);rt.set_epoch(max(1,cp['epoch']))
    if training:
        rt.optimizer.load_state_dict(cp['optimizer']);rt.scaler.load_state_dict(cp['scaler']);previous.restore_rank_rng(rt,cp['rank_rng'][dist.get_rank()])
    return cp


def atomic_copy(source,target):
    temporary=target.with_suffix(target.suffix+'.tmp');shutil.copyfile(source,temporary);os.replace(temporary,target)


def train(rt,args,cpu,frozen,ds):
    folder=args.root/args.arm;epoch=step=0;best=dict(epoch=0,score=None,metric='segm/mAP')
    if args.resume:
        assert args.resume.resolve()==(folder/'last.pth').resolve(),'Resume only this arm audited last.pth'
        cp=restore(rt,args.resume,frozen,True);assert not cp.get('gate_only');epoch,step,best=cp['epoch'],cp['step'],cp['best']
        assert read(folder/'contract.json')==frozen
        if dist.get_rank()==0:
            # Recover interruption between atomic last save and derived copies.
            if best['epoch']==epoch:atomic_copy(folder/'last.pth',folder/'best_model.pth')
            if epoch in RETAIN:atomic_copy(folder/'last.pth',folder/('epoch%d.pth'%epoch))
            dump(folder/'best.json',best)
        dist.barrier()
    elif dist.get_rank()==0 and any(folder.iterdir()):raise RuntimeError('Arm already has artifacts; explicit --resume required')
    dist.barrier()
    if dist.get_rank()==0:dump(folder/'contract.json',frozen)
    for epoch in range(epoch+1,101):
        engine.verify_sources(frozen['source_hashes']);engine.verify_orders(frozen)
        rt.network.train();rt.set_epoch(epoch);loader,reference=epoch_loader(rt,ds,args.root,epoch)
        seen=[];batches=[];total=0.;retries=0;began=time.monotonic()
        for batch in loader:
            ident=int(Path(batch['img_metas'][0]['filename']).parent.parent.name);seen.append(ident);batches.append((batch,ident))
            if len(batches)==2:
                value=previous.accumulate(rt,batches,step+1);step+=1;total+=value['loss_sum_global'];retries+=value['retries'];batches=[]
                if dist.get_rank()==0 and step%25==0:
                    dump(folder/'progress.json',dict(epoch=epoch,update=step,last_global_microbatch_mean=value['loss_sum_global']/8,
                        preclip_norm=value['preclip_norm'],AMP_retries_epoch=retries,world_size=4,effective_batch=8))
                    print('NATIVE FULL UPDATE',args.arm,epoch,step,flush=True)
        assert not batches and len(seen)==len(set(seen))==534 and step==epoch*267
        all_seen=[None]*4;dist.all_gather_object(all_seen,seen,group=cpu)
        assert [v for block in range(267) for rank in range(4) for v in all_seen[rank][block*2:block*2+2]]==reference
        metrics=native.evaluate(rt,args.arm,'val',folder/('validation_epoch%d'%epoch),cpu)
        score=float(metrics['segm/mAP']);assert np.isfinite(score)
        improved=best['score'] is None or score>best['score']
        if improved:best=dict(epoch=epoch,score=score,metric='segm/mAP')
        save(rt,folder/'last.pth',epoch,step,frozen,cpu,best)
        if dist.get_rank()==0:
            if improved:atomic_copy(folder/'last.pth',folder/'best_model.pth')
            if epoch in RETAIN:atomic_copy(folder/'last.pth',folder/('epoch%d.pth'%epoch))
            dump(folder/'best.json',best)
            dump(folder/('epoch%d.json'%epoch),dict(epoch=epoch,step=step,sample_order=reference,rank_sample_order=all_seen,world_size=4,
                mean_microbatch_loss=total/2136,AMP_retries_epoch=retries,validation=metrics,best=best,seconds=time.monotonic()-began))
        dist.barrier();engine.verify_sources(frozen['source_hashes'])
    if dist.get_rank()==0:
        dump(folder/'training_complete.json',dict(complete=True,epoch=100,updates=26700,world_size=4,arm=args.arm,contract=frozen,best=best,
            checkpoint_sha256=core.C.file_sha(folder/'last.pth'),last_checkpoint_sha256=core.C.file_sha(folder/'last.pth'),best_checkpoint_sha256=core.C.file_sha(folder/'best_model.pth')))
    dist.barrier()


def resume_gate(rt,args,cpu,frozen):
    folder=args.root/'resume_gate'/args.arm
    if dist.get_rank()==0:assert not folder.exists(),'Preserve existing resume gate'

    if dist.get_rank()==0:folder.mkdir(parents=True)
    dist.barrier();rt.set_epoch(1)
    parameter=next(p for p in rt.bare.parameters() if p.requires_grad)
    rt.optimizer.state[parameter]=dict(step=torch.tensor(0.),exp_avg=torch.zeros_like(parameter),exp_avg_sq=torch.zeros_like(parameter))
    before=core.C.state_digest(rt.bare);scaler=rt.scaler.state_dict();path=folder/'restore_gate.pth'
    best=dict(epoch=0,score=None,metric='segm/mAP');save(rt,path,0,0,frozen,cpu,best)
    expected=engine.rng_draw(rt);engine.rng_draw(rt)
    with torch.no_grad():parameter.add_(1);rt.optimizer.state[parameter]['exp_avg'].add_(1)
    rt.scaler.load_state_dict(dict(scaler,scale=scaler['scale']/2));cp=restore(rt,path,frozen,True)
    actual=engine.rng_draw(rt);checks={k:actual[k]==v for k,v in expected.items()}
    passed=all(checks.values()) and core.C.state_digest(rt.bare)==before and rt.scaler.state_dict()==scaler and not torch.count_nonzero(rt.optimizer.state[parameter]['exp_avg']) and cp['best']==best
    rows=[None]*4 if dist.get_rank()==0 else None
    dist.gather_object(dict(rank=dist.get_rank(),passed=bool(passed),rng_next_five_exact=checks),rows,dst=0,group=cpu)
    if dist.get_rank()==0:dump(folder/'complete.json',dict(passed=all(r['passed'] for r in rows),world_size=4,ranks=rows,contract=frozen,checkpoint_sha256=core.C.file_sha(path),scope='synthetic optimizer moments; restore mechanics not next-update trajectory'))
    assert passed


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['plan','train','eval','gate-resume'])
    p.add_argument('--arm',choices=PROTOCOL['arms'],required=True);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--epochs',type=int,choices=[100],default=100);p.add_argument('--engineering-receipt',type=Path,required=True)
    p.add_argument('--allocator-cap-mib',type=int,default=0);p.add_argument('--resume',type=Path);p.add_argument('--split',choices=['val','test'],default='test');args=p.parse_args()
    if args.action=='plan':print(json.dumps(dict(protocol=PROTOCOL,root=str(args.root),arm=args.arm,orders=str(args.root/'orders/manifest.json')),indent=2));return
    native.configure();engine.PROTOCOL=PROTOCOL;engine.source_inventory=source_inventory;engine.validate_protocol()
    # Shared numerical runtime remains original100cosine. Only orchestration differs.
    cpu=engine.setup(args.allocator_cap_mib);lock=None
    try:
        if dist.get_rank()==0:
            (args.root/'locks').mkdir(parents=True,exist_ok=True);lock=(args.root/'locks'/(args.arm+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        dist.barrier();rt=native.runtime(args.arm,args.action=='train');ds=core.C.MinimalSatelliteDataset(core.previous.DATA,'train',augment=True)
        sources=source_inventory();frozen=json.loads(json.dumps(identity(rt,args,sources,ds)))
        commit=native.freeze_root(args,frozen) if dist.get_rank()==0 else None;box=[commit];dist.broadcast_object_list(box,src=0,group=cpu);frozen['git_commit']=box[0]
        engine.verify_sources(sources);dist.barrier()
        launches=[None]*4 if dist.get_rank()==0 else None
        dist.gather_object(dict(rank=dist.get_rank(),local_rank=int(os.environ['LOCAL_RANK']),pid=os.getpid(),
            start_ticks=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()[19]),launches,dst=0,group=cpu)
        if dist.get_rank()==0:dump(args.root/'rank_launches'/('%s_%s.json'%(args.action,args.arm)),
            dict(world_size=4,ranks=launches,source_hashes=sources,git_commit=box[0]))
        if args.action=='train':

            if dist.get_rank()==0:(args.root/args.arm).mkdir(parents=True,exist_ok=True)
            dist.barrier();train(rt,args,cpu,frozen,ds)
        elif args.action=='gate-resume':resume_gate(rt,args,cpu,frozen)
        else:
            assert args.resume is None,'Evaluation uses own validation-selected best, not user-supplied checkpoint'
            if args.split=='test':
                for arm in PROTOCOL['arms']:
                    done=read(args.root/arm/'training_complete.json');assert done['complete'] and done['epoch']==100 and done['updates']==26700 and done['arm']==arm
            done=read(args.root/args.arm/'training_complete.json');assert done['contract']==frozen
            path=args.root/args.arm/'best_model.pth';assert core.C.file_sha(path)==done['best_checkpoint_sha256']
            cp=restore(rt,path,frozen,False);assert cp['epoch']==done['best']['epoch'] and cp['best']==done['best']
            out=args.root/'evaluation'/args.split/args.arm
            assert not out.exists(),'Refuse overwrite of evaluation artifacts'
            native.evaluate(rt,args.arm,args.split,out,cpu,done['best_checkpoint_sha256'])
        dist.barrier()
    finally:
        if lock:lock.close()
        dist.destroy_process_group()

if __name__=='__main__':main()
