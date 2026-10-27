"""Native same-pass G/UBR short study; sealed old engine reused process-locally.

Every arm is four ranks with original effectivebatch8, fixed E10, no task warm
start. Original256/.25 RoI sampler and512 mask target retained. Old source files
are never edited. All native evaluation uses the actual original candidate and
supplementary-mask consumers through the newly installed decoder shim.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import numpy as np
WORK=Path(__file__).resolve().parents[2];REPO=WORK.parent
sys.path[:0]=[str(WORK),str(WORK/'scripts/probes')]
import ue_ubr_short_run_20261027 as engine
import ue_ubr_native_runtime_20261027 as native
CONFIG=WORK/'configs/ablations/ue_g_ubr_native_protocol_20261027.json'
ROOT=Path('/data/wangcheng/checkpoint/ue_g_ubr_native_short_20261027')
PROTOCOL=json.loads(CONFIG.read_text())
CRITICAL=['rsprompter/uav/native_roi_refiner.py','rsprompter/uav/ubr_native_integration.py','scripts/probes/ue_ubr_native_runtime_20261027.py',
          'rsprompter/uav/candidate_detail_interaction.py','rsprompter/decoder/sam2_decoder.py','rsprompter/cspm/mask_head_targets.py',
          '../sam2/sam2/modeling/sam/mask_decoder.py','../sam2/sam2/modeling/sam/prompt_encoder.py','../sam2/sam2/modeling/sam/transformer.py']
_BASE_IDENTITY=engine.identity


def runtime(arm,training):
    rt=native.build_runtime(arm)
    rt.bare=rt.network
    if training:
        rt.network=DistributedDataParallel(rt.bare,device_ids=[torch.cuda.current_device()],output_device=torch.cuda.current_device(),find_unused_parameters=True,broadcast_buffers=False)
    rt.generator=torch.Generator().manual_seed(PROTOCOL['seed']+dist.get_rank())
    engine.core.C.seed(PROTOCOL['seed']+dist.get_rank())
    return rt


def normalize_sources(receipt):
    result={}
    for value in [receipt]+list(receipt.get('ranks',[])):
        for key in ('source_sha256','sources_after','source_hashes'):
            for name,digest in value.get(key,{}).items():
                path=Path(name)
                full=path.resolve() if path.is_absolute() else (REPO/path).resolve() if (REPO/path).is_file() else (WORK/path).resolve()
                if REPO not in full.parents or not full.is_file():raise RuntimeError('Unresolved source receipt '+name)
                relative=str(full.relative_to(REPO))
                if relative in result and result[relative]!=digest:raise RuntimeError('Source receipt conflict '+relative)
                result[relative]=digest
    return result


def verify_receipt(path,sources):
    receipt=json.loads(path.read_text())
    if receipt.get('passed') is not True or receipt.get('world_size')!=4:raise RuntimeError('Passed actual four-rank native engineering gate required')
    ranks=receipt.get('ranks',[])
    if len(ranks)!=4 or any(not r.get('passed') for r in ranks):raise RuntimeError('All actual native gate ranks must pass')
    if any(r['initialization']['main_mask_size']!=[512,512] for r in ranks):raise RuntimeError('Gate target resolution changed')
    for r in ranks:
        if not r['parity']['raw_logits_exact'] or not r['parity']['binary_exact']:raise RuntimeError('Native zero identity missing')
        if r['inference250']['decoder_calls']!=8 or not r['inference250']['refined_logits_changed']:raise RuntimeError('Formal250 same-pass consumer missing')
        if not r['main_mask_only_gradient']['hidden']['norm']>0:raise RuntimeError('Main mask native hidden gradient missing')
    declared=normalize_sources(receipt)
    for name in CRITICAL:
        relative=str((WORK/name).resolve().relative_to(REPO))
        if declared.get(relative)!=sources.get(relative):raise RuntimeError('Native gate critical source absent/changed '+relative)
    for name,digest in declared.items():
        if engine.core.C.file_sha(REPO/name)!=digest:raise RuntimeError('Native gate source changed '+name)
    return engine.core.C.file_sha(path)


def source_inventory():
    paths={Path(__file__).resolve(),CONFIG}
    for module in tuple(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            path=Path(name).resolve()
            if path.is_file() and path.suffix=='.py' and REPO in path.parents:paths.add(path)
    paths.update((WORK/name).resolve() for name in CRITICAL)
    paths.update(WORK/name for name in engine.previous.hashes())
    return {str(path.relative_to(REPO)):engine.core.C.file_sha(path) for path in sorted(paths)}


def identity(rt,args,sources):
    result=_BASE_IDENTITY(rt,args,sources)
    # Arm is recorded at the contract's outer level. A's absence of a tail must
    # not make otherwise identical shared root initialization receipts conflict.
    result['initialization']={key:value for key,value in rt.initialization_audit.items() if key!='arm'}
    result['native_evidence_port']={'A_nohead':'absent','B_zero':'zero RGB+geometry','C_correct':'correct RGB+geometry'}[args.arm]
    return result


def freeze_root(args,frozen):
    """Byte-exact source pin, arm-independent root, owner commit stable after docs."""
    import fcntl
    import subprocess
    import time
    shared={key:value for key,value in frozen.items() if key not in ('arm','native_evidence_port','allocator_cap_mib')}
    args.root.mkdir(parents=True,exist_ok=True)
    with (args.root/'study.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=args.root/'study_contract.json';requested=os.environ.get('UE_UBR_NATIVE_SOURCE_COMMIT')
        if path.exists():
            receipt=json.loads(path.read_text())
            if receipt['shared']!=shared:raise RuntimeError('Native root binds a different source/input/protocol contract')
            commit=receipt['git_commit']
            if requested and requested!=commit:raise RuntimeError('Owner source commit differs from study')
        else:commit=requested or subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
        for name,expected in shared['source_hashes'].items():
            blob=subprocess.check_output(['git','show',commit+':'+name],cwd=REPO)
            if hashlib.sha256(blob).hexdigest()!=expected:raise RuntimeError('Commit exact native source before launch: '+name)
        if not path.exists():
            engine.core.C.dump(path,dict(shared=shared,git_commit=commit,created=time.time()))
            for name in shared['source_hashes']:
                target=args.root/'frozen_source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((REPO/name).read_bytes())
    return commit




@torch.inference_mode()
def evaluate(rt,arm,split,out,cpu,checkpoint_sha=None):
    if torch.is_autocast_enabled():raise RuntimeError('FP32 full evaluation required')
    rt.bare.eval();ds=engine.core.C.MinimalSatelliteDataset(engine.core.previous.DATA,split)
    assert len(ds)==459
    rows,metas=[],[]
    for index in range(dist.get_rank(),len(ds),4):
        ident=ds.samples[index]['image_id'];batch=engine.core.C.rtmdet_collate_fn([ds[index]])
        sample=engine.core.C.processed(rt.satellite,batch);payload=rt.payload(ident);rt.joint.prepare_uav(**payload)
        features=rt.satellite.extract_feat(sample['inputs'])
        prefix,combined,accepted=engine.core.inference_outputs(rt.satellite,rt.joint.uav,sample,features,payload['geometry'])
        rows.append(dict(index=index,image_id=ident,gt=engine.core.C.serialized(engine.core.C.gt_arrays(sample['data_samples'][0].gt_instances)),
            dt=engine.core.C.serialized(combined),original_path=engine.core.C.serialized(prefix),geometry_accepted=accepted))
        metas.append((index,batch['img_metas'][0]));rt.clear()
        if (index//4+1)%10==0:
            engine.core.C.dump(out/('rank%d_progress.json'%dist.get_rank()),dict(images=index//4+1,split=split,arm=arm,world_size=4))
    gathered=[None]*4 if dist.get_rank()==0 else None;dist.gather_object((rows,metas),gathered,dst=0,group=cpu);metrics=None
    if dist.get_rank()==0:
        records=sorted([r for part,_ in gathered for r in part],key=lambda r:r['index']);assert [r.pop('index') for r in records]==list(range(459))
        assert len({r['image_id'] for r in records})==459
        metadata=[m for _,m in sorted([m for _,part in gathered for m in part],key=lambda v:v[0])]
        gt=[{**r['gt'],'bboxes':np.asarray(r['gt']['bboxes'],np.float32).reshape(-1,4),'labels':np.asarray(r['gt']['labels'],np.int64)} for r in records]
        pred=[engine.core.consumer.arrays(r['dt']) for r in records];cg,cd=engine.core.C.build_coco_gt_and_dt(gt,pred,metadata,score_key='scores');metrics={}
        for kind in ('segm','bbox'):metrics.update(engine.core.C.run_coco_eval(cg,cd,kind,max_dets=[1,10,250]))
        metrics={k.replace('AR@100','AR@250'):v for k,v in metrics.items()}
        engine.core.C.dump(out/'records.json',records);engine.core.C.dump(out/'metrics.json',metrics)
        engine.core.C.dump(out/'complete.json',dict(complete=True,split=split,images=459,world_size=4,precision='FP32',arm=arm,
            checkpoint_sha256=checkpoint_sha,consumer='original inference_outputs/native shim same-pass',scope='development_test_not_selection' if split=='test' else 'full_validation'))
    box=[metrics];dist.broadcast_object_list(box,src=0,group=cpu);return box[0]


def configure():
    # Explicit bindings affect THIS process only, leaving old running source/argv
    # contracts immutable. Save/restore/train/accumulation/RNG remain old engine.
    engine.__doc__=__doc__
    engine.CONFIG,engine.ROOT,engine.PROTOCOL=CONFIG,ROOT,PROTOCOL
    engine.CRITICAL=CRITICAL
    engine.runtime=runtime;engine.source_inventory=source_inventory
    engine.verify_receipt=verify_receipt;engine.identity=identity
    engine.freeze_root=freeze_root;engine.evaluate=evaluate


def main():
    configure();engine.main()


if __name__=='__main__':main()
