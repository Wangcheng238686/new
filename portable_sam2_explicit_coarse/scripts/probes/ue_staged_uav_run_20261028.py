"""Four-rank staged SUTC/modality/view runner; explicit recipe, no auto launch.

Reuses frozen full100 update/epoch/save/restore mathematics process-locally.
All recipes call the SUTC objective (including S/T), never the old unconditional
UAV-instance objective. Gates are engineering-only and never initializers.
"""
import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import MethodType, SimpleNamespace
from unittest.mock import patch
import traceback

WORK=Path(__file__).resolve().parents[2];REPO=WORK.parent
sys.path[:0]=[str(REPO/'sam2'),str(WORK),str(WORK/'scripts/probes'),str(WORK/'scripts/experiments')]
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import ue_ubr_native_full_run_20261028 as full
import ue_ubr_native_sutc_runtime_20261028 as sutc
from ue_staged_uav_matrix import canonical_sha, audit_matrix
import ue_staged_uav_control as controller
from ue_ubr_native_resource_gate_20261027 import memory
from ue_ubr_native_gate_20261027 import grad_group
core=full.core;engine=full.engine;previous=full.previous


def read(p):return json.loads(Path(p).read_text())
def sha(p):return core.C.file_sha(Path(p))
def dump(p,v):core.C.dump(Path(p),v)


def validate_manifest(root,recipe):
    m=read(root/'launch_manifest.json')
    if Path(m['root']).resolve()!=root.resolve() or Path(m['runtime_source']).resolve()!=REPO:
        raise RuntimeError('Run only inside this manifest pinned worktree/root')
    for name,digest in m['source_sha256'].items():
        if sha(REPO/name)!=digest:raise RuntimeError('Frozen source drift '+name)
    protocol=read(WORK/m['protocol_path'])
    if canonical_sha(protocol)!=m['protocol_sha256'] or protocol!=m['protocol']:
        raise RuntimeError('Protocol identity changed')
    if sha(WORK/m['matrix_path'])!=m['matrix_sha256']:raise RuntimeError('Matrix changed')
    matrix=read(WORK/m['matrix_path'])
    audit=audit_matrix(matrix,m.get('base_lock',{}).get('base_arm') if m.get('base_lock') else None)
    if m['recipes']!=audit['recipes']:raise RuntimeError('Resolved recipe drift')
    if recipe not in m['recipes']:raise RuntimeError('Later stages require explicit base lock')
    r=m['recipes'][recipe]
    if canonical_sha({k:v for k,v in r.items() if k!='recipe_sha256'})!=r['recipe_sha256']:
        raise RuntimeError('Recipe digest mismatch')
    if sha(root/'orders/manifest.json')!=m['order_manifest_sha256']:raise RuntimeError('Order manifest changed')
    if protocol['epochs']!=100 or protocol['total_updates']!=26700 or protocol['world_size']!=4:
        raise RuntimeError('Unexpected full training budget')
    # Direct torchrun receives the same source/selection/cache checks as wrapper.
    controller.validate(m)
    controller._verify_cache_receipt(m,recipe,deep=False)
    return m,r


def verify_loaded_sources(sources):
    for name,digest in sources.items():
        if sha(REPO/name)!=digest:raise RuntimeError('Frozen source drift '+name)
    import sam2
    if REPO/'sam2' not in Path(sam2.__file__).resolve().parents:
        raise RuntimeError('SAM2 escaped pinned worktree')
    for name,module in tuple(sys.modules.items()):
        path=getattr(module,'__file__',None)
        if not path:continue
        path=Path(path).resolve()
        if path.suffix=='.py' and REPO in path.parents and str(path.relative_to(REPO)) not in sources:
            raise RuntimeError('Imported local source outside frozen inventory '+str(path))
        if (name=='sam2' or name.startswith('sam2.')) and REPO not in path.parents:
            raise RuntimeError('External SAM2 import '+name)


def marker_identity(m,r):
    return dict(recipe=r['name'],recipe_sha256=r['recipe_sha256'],protocol_sha256=m['protocol_sha256'],
                source_sha256=m['source_sha256'],world_size=4)


def build_runtime(m,r,training):
    rt=sutc.build_runtime(r['arm'],uav_view_ids=r['uav_view_ids'],subset_cache_root=r['subset_cache_root'],
                          rgb_content=r['rgb_content'],geometry_content=r['geometry_content'])
    rt.bare=rt.network
    if training:
        rt.network=DistributedDataParallel(rt.bare,device_ids=[torch.cuda.current_device()],
            output_device=torch.cuda.current_device(),find_unused_parameters=True,broadcast_buffers=False)
    rt.generator=torch.Generator().manual_seed(m['protocol']['seed']+dist.get_rank())
    rt.update=MethodType(sutc.base.Runtime.update,rt)
    core.C.seed(m['protocol']['seed']+dist.get_rank())
    return rt


def initial_fingerprint(rt):
    p=rt.joint.uav
    rgb={k:v for k,v in p.state_dict().items() if k.startswith(('rgb_encoder.','rgb_project.')) or k in ('rgb_mean','rgb_std')} if isinstance(p,torch.nn.Module) else None
    return dict(satellite_sha256=core.C.state_digest(rt.satellite),
        head_sha256=core.C.state_digest(rt.satellite.roi_head.mask_head.native_refiner),
        rgb_sha256=core.previous.tensor_digest(rgb) if rgb is not None else None,
        full_model_sha256=core.C.state_digest(rt.bare),audit=rt.initialization_audit)


def configure(m):
    # Process-local substitutions only. No editing or importing code from a live
    # ABC runtime tree. These bindings are also used during resume/resource gate.
    full.PROTOCOL=m['protocol'];engine.PROTOCOL=m['protocol']
    previous.objective=lambda rt,sample,ident:rt.objective(sample,ident)
    engine.source_inventory=lambda:m['source_sha256']
    engine.verify_sources=verify_loaded_sources
    full.native.evaluate=evaluate
    engine.validate_protocol()


def contract(rt,args,m,r,ds,gate_sha=None):
    return dict(arm=r['name'],recipe=r,protocol=m['protocol'],epochs=100,git_commit=m['git_commit'],
        source_hashes=m['source_sha256'],recipe_sha256=r['recipe_sha256'],protocol_sha256=m['protocol_sha256'],
        initialization=rt.initialization_audit,data_identity=core.C.archive_identity(core.previous.DATA),
        dataset_samples_sha256=full.dataset_digest(ds),source_order_sha256=full.verify_orders(args.root,ds),
        measured_cache_manifest_sha256=sha(core.OUT/'input_cache_sha256.json'),
        subset_cache_manifest_sha256=rt.initialization_audit.get('subset_cache_manifest_sha256'),
        cache_receipt_sha256=m['cache_receipts'][r['name']]['sha256'],
        engineering_receipt_sha256=gate_sha)


@torch.inference_mode()
def evaluate(rt,arm,split,out,cpu,checkpoint_sha=None):
    if torch.is_autocast_enabled():raise RuntimeError('Evaluation requires FP32')
    rt.bare.eval();ds=core.C.MinimalSatelliteDataset(core.previous.DATA,split)
    if len(ds)!=459:raise RuntimeError('Full split required')
    records=[];metas=[]
    for index in range(dist.get_rank(),len(ds),4):
        ident=int(ds.samples[index]['image_id']);batch=core.C.rtmdet_collate_fn([ds[index]])
        sample=core.C.processed(rt.satellite,batch)
        try:prefix,combined,accepted=rt.inference(sample,ident)
        finally:rt.clear()
        records.append(dict(index=index,image_id=ident,gt=core.C.serialized(core.C.gt_arrays(sample['data_samples'][0].gt_instances)),
            dt=core.C.serialized(combined),original_path=core.C.serialized(prefix),geometry_accepted=accepted))
        metas.append((index,batch['img_metas'][0]))
        if len(records)%10==0:dump(out/('rank%d_progress.json'%dist.get_rank()),dict(images=len(records),split=split,recipe=arm))
    gathered=[None]*4 if dist.get_rank()==0 else None
    dist.gather_object((records,metas),gathered,dst=0,group=cpu);metrics=None
    if dist.get_rank()==0:
        rows=sorted([row for rec,_ in gathered for row in rec],key=lambda x:x['index'])
        assert [row.pop('index') for row in rows]==list(range(459))
        metadata=[meta for _,meta in sorted([row for _,part in gathered for row in part],key=lambda x:x[0])]
        gt=[{**row['gt'],'bboxes':np.asarray(row['gt']['bboxes'],np.float32).reshape(-1,4),'labels':np.asarray(row['gt']['labels'],np.int64)} for row in rows]
        pred=[core.consumer.arrays(row['dt']) for row in rows]
        cg,cd=core.C.build_coco_gt_and_dt(gt,pred,metadata,score_key='scores');metrics={}
        for kind in ('segm','bbox'):metrics.update(core.C.run_coco_eval(cg,cd,kind,max_dets=[1,10,250]))
        metrics={k.replace('AR@100','AR@250'):v for k,v in metrics.items()}
        dump(out/'records.json',rows);dump(out/'metrics.json',metrics)
        dump(out/'complete.json',dict(complete=True,split=split,images=459,precision='FP32',arm=arm,
            checkpoint_sha256=checkpoint_sha,world_size=4,scope='full_split_not_selection' if split=='test' else 'full_validation'))
    box=[metrics];dist.broadcast_object_list(box,src=0,group=cpu);return box[0]


@torch.no_grad()
def forced_masks(rt,batch,ident,*,zero_identity=False):
    rt.bare.eval();sample=core.C.processed(rt.satellite,batch,False)
    payload=rt.payload(ident);rt.joint.prepare_uav(**payload)
    features=rt.satellite.extract_feat(sample['inputs']);head=rt.satellite.roi_head
    x=head._inject_extra_pe(features[0]);meta=sample['data_samples'][0].metainfo
    boxes=torch.tensor([[16.25,20.5,83.75,93.25]],device='cuda').repeat(250,1)
    scores=torch.full((250,),.1,device='cuda');rt.shim.calls=0;rt.shim.observe=True
    comparisons=[]
    def compare_all_chunks(module,inputs,output):
        comparisons.append(dict(instances=len(output),exact=bool(torch.equal(inputs[1],output))))
    handle=head.mask_head.native_refiner.register_forward_hook(compare_all_chunks)
    try:
        result=core.consumer.masks(head,x,features,meta,boxes,scores)
        obs=rt.shim.observation
        assert len(result['scores'])==250 and rt.shim.calls==8
        assert torch.isfinite(obs['refined']).all()
        assert sum(row['instances'] for row in comparisons)==250 and len(comparisons)==8
        if zero_identity:assert all(row['exact'] for row in comparisons)
        return dict(instances=250,decoder_calls=rt.shim.calls,forced_box_fixture=True,
                    zero_native_logits_exact=all(row['exact'] for row in comparisons),memory=memory())
    finally:
        handle.remove();rt.shim.observe=False;rt.shim.observation={};rt.clear()


def gate(rt,args,cpu,m,r,ds,frozen):
    # This directory is separate from formal arm checkpoints. Every retry gets
    # its own engineering artifacts; a passed gate never seeds a training arm.
    attempt=args.root/'gates'/r['name']/('attempt_'+str(time.time_ns()) if dist.get_rank()==0 else 'pending')
    box=[str(attempt) if dist.get_rank()==0 else None];dist.broadcast_object_list(box,src=0,group=cpu);attempt=Path(box[0])
    if dist.get_rank()==0:attempt.mkdir(parents=True,exist_ok=False)
    dist.barrier()
    report=dict(rank=dist.get_rank(),**marker_identity(m,r),scope='engineering_only_no_AP_no_task_warm_start',updates=[])
    try:
        report['initialization']=initial_fingerprint(rt)
        frozen_before=core.previous.tensor_digest({n:p for n,p in rt.bare.named_parameters() if not p.requires_grad})
        order=sorted(range(len(ds)),key=lambda i:len(ds.samples[i]['annotations']),reverse=True)
        ids=[order[dist.get_rank()],order[dist.get_rank()+4]]
        batches=[(core.C.rtmdet_collate_fn([ds[i]]),int(ds.samples[i]['image_id'])) for i in ids]
        torch.cuda.reset_peak_memory_stats()
        report['zero_identity250']=forced_masks(rt,*batches[0],zero_identity=True)
        rt.network.train();rt.set_epoch(1)
        original_processed=core.C.processed
        for step in (1,2,3):
            def processed(model,batch,training=False):
                sample=original_processed(model,batch,training)
                if step==1 and dist.get_rank()==3:
                    sample=deepcopy(sample)
                    sample['data_samples'][0].gt_instances=sample['data_samples'][0].gt_instances[:0]
                return sample
            with patch.object(core.C,'processed',side_effect=processed):
                result=previous.accumulate(rt,batches,step)
            report['updates'].append(dict(step=step,empty_GT_rank3=step==1,**result,memory=memory()))
        # Original main-mask objective only; retain zero graph for other losses.
        sample=core.C.processed(rt.satellite,batches[0][0],True)
        rt.optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            losses=rt.losses(sample,batches[0][1]);terms=[]
            for name,value in losses.items():
                if 'loss' not in name:continue
                for v in value if isinstance(value,(tuple,list)) else [value]:
                    if torch.is_tensor(v) and v.requires_grad:terms.append(v*(1 if name=='loss_mask' else 0))
            loss=sum(terms)
        scale=rt.scaler.get_scale();rt.scaler.scale(loss).backward();rt.scaler.unscale_(rt.optimizer)
        report['main_mask_gradient']=dict(native=grad_group(rt.bare,'native_refiner'),hidden=grad_group(rt.bare,'native_refiner.input_proj'),
            UAV_RGB=grad_group(rt.bare,'rgb_encoder'),loss_terms=sorted(k for k in losses if 'loss' in k))
        assert report['main_mask_gradient']['native']['norm']>0 and report['main_mask_gradient']['hidden']['norm']>0
        assert ('loss_uav_instances' in losses)==rt.switches.upstream
        assert not any('ubr' in k or 'decoder_tail' in k for k in losses)
        rt.scaler.update(new_scale=scale);rt.optimizer.zero_grad(set_to_none=True);rt.clear()
        report['learned_inference250']=forced_masks(rt,*batches[0])
        state=core.C.state_digest(rt.bare);states=[None]*4;dist.all_gather_object(states,state,group=cpu)
        assert len(set(states))==1
        report['replicas_identical']=True
        assert frozen_before==core.previous.tensor_digest({n:p for n,p in rt.bare.named_parameters() if not p.requires_grad})
        report['frozen_parameters_unchanged']=True
        # Synthetic restore mechanics covers model, optimizer, AMP and all five
        # RNG families. Actual next-update reproducibility is not inferred.
        restore_args=SimpleNamespace(root=attempt,arm=r['name'])
        full.resume_gate(rt,restore_args,cpu,frozen)
        report['restore_gate']=read(attempt/'resume_gate'/r['name']/'complete.json')
        assert report['restore_gate']['passed']
        report['memory']=memory();report['passed']=True
        verify_loaded_sources(m['source_sha256'])
        dump(attempt/('rank%d.json'%dist.get_rank()),report)
        rows=[None]*4;dist.all_gather_object(rows,report,group=cpu)
        if dist.get_rank()==0:
            mem=[sample['memory'] for row in rows for sample in row['updates']]+[row['memory'] for row in rows]
            complete=dict(passed=True,**marker_identity(m,r),ranks=rows,attempt=str(attempt),
                max_reserved_mib=max(x['peak_reserved_mib'] for x in mem),
                max_resident_mib=max(x['nvidia_resident_mib'] for x in mem),contract=frozen,
                initialization=rows[0]['initialization'],scope='real_four_rank_three_global8_updates_and_forced250_no_AP')
            dump(args.root/'gates'/r['name']/'complete.json',complete)
    except Exception:
        report['failed']=traceback.format_exc();dump(attempt/('rank%d_failed.json'%dist.get_rank()),report);raise


def require_gate(root,m,r):
    path=root/'gates'/r['name']/'complete.json';g=read(path)
    if g.get('passed') is not True or any(g.get(k)!=v for k,v in marker_identity(m,r).items()):
        raise RuntimeError('Matching passed four-rank gate required')
    if len(g.get('ranks',[]))!=4 or not all(row.get('passed') for row in g['ranks']):
        raise RuntimeError('Gate rank coverage incomplete')
    return g,sha(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['gate','train','eval']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--recipe',required=True);p.add_argument('--allocator-cap-mib',type=int,default=0);p.add_argument('--resume',type=Path)
    args=p.parse_args();args.root=args.root.resolve();args.arm=args.recipe
    m,r=validate_manifest(args.root,args.recipe);configure(m)
    stage=next(stage for stage,names in m['stages'].items() if args.recipe in names)
    if args.action=='eval' and stage=='structure' and not m.get('base_lock'):
        raise RuntimeError('Lock follow-up base before exposing structure test results')
    g=gate_sha=None
    if args.action!='gate':g,gate_sha=require_gate(args.root,m,r)
    if args.action=='train':controller.verify_initialization_family(m,args.recipe)
    if args.action!='train' and args.resume:raise ValueError('Only training can resume')
    cpu=engine.setup(args.allocator_cap_mib);lock=None
    try:
        if dist.get_rank()==0:
            (args.root/'locks').mkdir(parents=True,exist_ok=True);lock=(args.root/'locks'/(args.recipe+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        dist.barrier()
        if dist.get_rank()==0:controller._verify_cache_receipt(m,args.recipe,deep=True)
        dist.barrier()
        original_verify_orders=engine.verify_orders
        def verify_order_and_cache(frozen):
            original_verify_orders(frozen)
            controller._verify_cache_receipt(m,args.recipe,deep=False)
        engine.verify_orders=verify_order_and_cache
        rt=build_runtime(m,r,args.action in ('gate','train'))
        ds=core.C.MinimalSatelliteDataset(core.previous.DATA,'train',augment=args.action=='train')
        frozen=json.loads(json.dumps(contract(rt,args,m,r,ds,gate_sha)))
        verify_loaded_sources(m['source_sha256'])
        if g is not None:
            if initial_fingerprint(rt)!=g['initialization']:
                raise RuntimeError('Fresh training initialization differs from engineering gate')
            gate_contract=dict(g['contract']);current=dict(frozen)
            gate_contract.pop('engineering_receipt_sha256');current.pop('engineering_receipt_sha256')
            if gate_contract!=current:
                raise RuntimeError('Data/order/cache or scientific contract differs from gate')
        launches=[None]*4 if dist.get_rank()==0 else None
        dist.gather_object(dict(rank=dist.get_rank(),local_rank=int(os.environ['LOCAL_RANK']),pid=os.getpid(),
            start_ticks=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()[19]),launches,dst=0,group=cpu)
        if dist.get_rank()==0:dump(args.root/'rank_launches'/(args.action+'_'+args.recipe+'.json'),dict(**marker_identity(m,r),ranks=launches))
        if args.action=='gate':gate(rt,args,cpu,m,r,ds,frozen)
        elif args.action=='train':
            if dist.get_rank()==0:(args.root/args.recipe).mkdir(parents=True,exist_ok=True)
            dist.barrier();full.train(rt,args,cpu,frozen,ds)
            if dist.get_rank()==0:
                path=args.root/args.recipe/'training_complete.json';done=read(path);done.update(marker_identity(m,r));dump(path,done)
        else:
            # Test waits for every recipe in its stage, not just its own arm.
            stage=next(stage for stage,names in m['stages'].items() if args.recipe in names)
            for name in m['stages'][stage]:
                done=read(args.root/name/'training_complete.json')
                assert done['complete'] and done['epoch']==100 and done['updates']==26700
                assert done['recipe_sha256']==m['recipes'][name]['recipe_sha256']
            done=read(args.root/args.recipe/'training_complete.json');assert done['contract']==frozen
            path=args.root/args.recipe/'best_model.pth';assert sha(path)==done['best_checkpoint_sha256']
            cp=full.restore(rt,path,frozen,False);assert cp['epoch']==done['best']['epoch']
            out=args.root/'evaluation/test'/args.recipe
            if (out/'complete.json').exists():raise RuntimeError('Refuse completed evaluation overwrite')
            evaluate(rt,args.recipe,'test',out,cpu,sha(path))
            if dist.get_rank()==0:
                d=read(out/'complete.json');d.update(marker_identity(m,r));dump(out/'complete.json',d)
        dist.barrier()
    finally:
        if lock:lock.close()
        dist.destroy_process_group()


if __name__=='__main__':main()
