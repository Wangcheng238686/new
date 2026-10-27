"""Source-pinned four-GPU-per-arm native study owner; measured per-arm reservations."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
WORK=Path(__file__).resolve().parents[2];REPO=WORK.parent
sys.path[:0]=[str(WORK),str(WORK/'scripts/probes')]
import ue_ubr_short_owner_20261027 as old
import ue_ubr_native_run_20261027 as runner
RUN=Path(runner.__file__).resolve();PYTHON=old.PYTHON
ARMS=('A_nohead','B_zero','C_correct');GPUS=(0,1,2,3)
sha,atomic,proc_identity,git=old.sha,old.atomic,old.proc_identity,old.git


def prepare_sources(receipt):
    source=runner.source_inventory()
    for name,digest in runner.normalize_sources(receipt).items():
        if sha(REPO/name)!=digest:raise RuntimeError('Engineering source drift '+name)
        if name in source and source[name]!=digest:raise RuntimeError('Conflicting source '+name)
        source[name]=digest
    for path in (Path(__file__).resolve(),Path(old.__file__).resolve(),RUN,runner.CONFIG):
        source[str(path.relative_to(REPO))]=sha(path)
    runner.verify_receipt(_ENGINEERING_PATH,source)
    commit=git('rev-parse','HEAD').decode().strip()
    import hashlib
    for name,digest in source.items():
        if hashlib.sha256(git('show',commit+':'+name)).hexdigest()!=digest:
            raise RuntimeError('Commit exact source before launch '+name)
    return commit,source


def resources(jobs,reservations,margin):
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    cards={}
    for line in raw.splitlines():
        index,uuid,free=[x.strip() for x in line.split(',')]
        if int(index) in GPUS:cards[int(index)]=dict(uuid=uuid,actual_free_mib=int(free),admissible_free_mib=int(free)-margin)
    if set(cards)!=set(GPUS):raise RuntimeError('Four physical GPU indices0..3 required')
    usage={}
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    for line in raw.splitlines():
        fields=[x.strip() for x in line.split(',')]
        if len(fields)==3 and fields[1].isdigit() and fields[2].isdigit():usage[(fields[0],int(fields[1]))]=int(fields[2])
    for job in jobs.values():
        if job['status']!='running':continue
        ranks=old.rank_records(job['rank_launch_record'],job['launcher']['start_ticks']);job['ranks']=ranks
        for index,card in cards.items():
            pid=next((int(r['pid']) for r in ranks if int(r['local_rank'])==index and r['currently_live']),None)
            card['admissible_free_mib']-=max(0,reservations[job['arm']]-usage.get((card['uuid'],pid),0))
    return cards


def command(args,action,arm):
    return [PYTHON,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=4',str(RUN),action,
        '--arm',arm,'--root',str(args.root),'--epochs','10','--engineering-receipt',str(args.engineering_receipt),
        '--allocator-cap-mib',str(args.allocator_caps[arm]),'--split','test' if action=='eval' else 'val']


def run(args):
    args.root.mkdir(parents=True,exist_ok=True)
    lock=(args.root/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (args.root/'owner_state.json').exists():raise RuntimeError('Existing owner state requires explicit recovery audit')
    receipt=json.loads(args.engineering_receipt.read_text());commit,source=prepare_sources(receipt)
    orders={str(runner.engine.base.ROOT/'rgbd'/('epoch%d.json'%e)):sha(runner.engine.base.ROOT/'rgbd'/('epoch%d.json'%e)) for e in range(1,11)}
    # Resource observations are frozen input provenance, not model-selection metrics.
    resource_digest=sha(args.resource_receipt)
    resume=args.root/'resume_gate'/'C_correct'/'complete.json'
    gate=json.loads(resume.read_text())
    if not gate.get('passed') or gate.get('world_size')!=4:raise RuntimeError('Native four-rank restore gate required')
    contract=gate['contract']
    if contract['git_commit']!=commit or contract['arm']!='C_correct' or contract['allocator_cap_mib']!=args.allocator_caps['C_correct']:
        raise RuntimeError('Native restore gate source/arm/cap mismatch')
    if contract['engineering_receipt_sha256']!=sha(args.engineering_receipt):raise RuntimeError('Restore gate engineering receipt mismatch')
    if any(source.get(name)!=digest for name,digest in contract['source_hashes'].items()):raise RuntimeError('Restore gate source outside owner closure')
    resource=json.loads(args.resource_receipt.read_text())
    if not resource.get('passed') or resource.get('world_size')!=4:raise RuntimeError('Passed four-rank per-arm resource receipt required')
    for arm in ARMS:
        peak=resource['arms'][arm]['max_reserved_mib']
        resident=resource['arms'][arm]['max_resident_mib']
        if args.reservations[arm]<max(peak+512,resident+32):raise RuntimeError('Reservation below measured process residency '+arm)
        if args.allocator_caps[arm]<peak:raise RuntimeError('Allocator cap below measured peak '+arm)
    jobs={}
    for action in ('train','eval'):
        for arm in ARMS:
            name=action+'_'+arm
            marker=args.root/arm/'training_complete.json' if action=='train' else args.root/'evaluation'/'test'/arm/'complete.json'
            jobs[name]=dict(status='pending',action=action,arm=arm,argv=command(args,action,arm),marker=str(marker),
                dependencies=[] if action=='train' else ['train_'+a for a in ARMS],
                rank_launch_record=str(args.root/'rank_launches'/('%s_%s_%s.json'%(action,arm,'val' if action=='train' else 'test'))))
    owner=proc_identity(os.getpid())
    manifest=dict(owner=owner,git_commit=commit,source_sha256=source,engineering_receipt=str(args.engineering_receipt),
        engineering_sha256=sha(args.engineering_receipt),resource_receipt=str(args.resource_receipt),resource_sha256=resource_digest,
        resume_gate_sha256=sha(resume),source_order_sha256=orders,protocol=runner.PROTOCOL,GPUs=list(GPUS),
        world_size_per_experiment=4,per_arm_reservation_mib=args.reservations,allocator_caps=args.allocator_caps,
        all_card_margin_mib=args.margin_mib,max_concurrent_groups=3,evaluation_concurrency=1,
        schedule='three concurrent four-rank training groups when all-card headroom admits; fixedE10 fulltest serially after all training')
    atomic(args.root/'owner_manifest.json',manifest);atomic(args.root/'owner_job_plan.json',jobs)
    for name in source:
        target=args.root/'owner_frozen_source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/name,target)
    old.RUN=RUN
    children={};streams={};failures=[]
    while True:
        for name,child in list(children.items()):
            rc=child.poll()
            if rc is None:continue
            job=jobs[name];streams.pop(name).close();del children[name]
            try:
                accepted,reason=old.validate_marker(job,source,commit) if rc==0 else (False,'nonzero child returncode')
                if accepted and job['action']=='eval':
                    result=json.loads(Path(job['marker']).read_text());trained=json.loads((args.root/job['arm']/'training_complete.json').read_text())
                    accepted=result['checkpoint_sha256']==trained['checkpoint_sha256'];reason=None if accepted else 'Not fixedE10 checkpoint'
            except Exception as exc:accepted,reason=False,repr(exc)
            job.update(status='complete' if accepted else 'failed',returncode=rc,failure_reason=reason,completed=time.time())
            if not accepted:failures.append(name+': '+str(reason))
            print('END',name,rc,job['status'],flush=True)
        if not failures:
            try:
                if any(sha(REPO/n)!=h for n,h in source.items()):raise RuntimeError('Pinned source drift')
                if any(sha(n)!=h for n,h in orders.items()):raise RuntimeError('Sample order drift')
                if sha(args.engineering_receipt)!=manifest['engineering_sha256'] or sha(args.resource_receipt)!=resource_digest or sha(resume)!=manifest['resume_gate_sha256']:raise RuntimeError('Engineering/resource/restore receipt drift')
                if json.loads((args.root/'study_contract.json').read_text())['git_commit']!=commit:raise RuntimeError('Study git commit mismatch')
            except Exception as exc:failures.append(repr(exc))
        try:cards=resources(jobs,args.reservations,args.margin_mib)
        except Exception as exc:failures.append('Resource audit '+repr(exc));cards={}
        if not failures:
            for name,job in jobs.items():
                if job['status']!='pending' or any(jobs[d]['status']!='complete' for d in job['dependencies']):continue
                if len(children)>=3:break
                if job['action']=='eval' and children:continue
                reserve=args.reservations[job['arm']]
                if any(cards[g]['admissible_free_mib']<reserve for g in GPUS):continue
                log=args.root/'logs'/(name+'.log');log.parent.mkdir(exist_ok=True);stream=log.open('x')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',UE_UBR_NATIVE_SOURCE_COMMIT=commit)
                child=subprocess.Popen(job['argv'],cwd=WORK,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                launcher=proc_identity(child.pid) or dict(pid=child.pid,start_ticks='0',argv=job['argv'])
                children[name]=child;streams[name]=stream
                job.update(status='running',launcher=launcher,launched=time.time(),log=str(log),GPU_indices=list(GPUS),ranks=[])
                atomic(args.root/'owner_launches'/(name+'.json'),job)
                for card in cards.values():card['admissible_free_mib']-=reserve
                print('START',name,'four GPUs',child.pid,flush=True)
        complete=all(j['status']=='complete' for j in jobs.values())
        atomic(args.root/'owner_state.json',dict(stage='complete' if complete else 'failure_requires_audit' if failures else 'running',owner=owner,
            updated=time.time(),jobs=jobs,failures=failures,GPU_headroom=cards,running_groups=len(children),source_git_commit=commit))
        if complete:
            atomic(args.root/'owner_complete.json',dict(complete=True,jobs=len(jobs),git_commit=commit,no_automatic_claim_of_AP_gain=True));return
        if failures and not children:raise RuntimeError(str(failures))
        time.sleep(15)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--engineering-receipt',type=Path,required=True)
    p.add_argument('--resource-receipt',type=Path,required=True)
    p.add_argument('--reservations',type=json.loads,required=True,help='Measured per-arm whole-process GPU MiB incl context/slack')
    p.add_argument('--allocator-caps',type=json.loads,required=True,help='Per-arm torch allocator caps in MiB')
    p.add_argument('--margin-mib',type=int,default=1024);p.add_argument('--detach',action='store_true')
    args=p.parse_args();args.root=args.root.resolve();args.engineering_receipt=args.engineering_receipt.resolve(strict=True);args.resource_receipt=args.resource_receipt.resolve(strict=True)
    for mapping in (args.reservations,args.allocator_caps):
        if set(mapping)!=set(ARMS) or any(not isinstance(v,int) or v<=0 for v in mapping.values()):p.error('positive per-arm MiB required')
    if args.margin_mib<512:p.error('at least512MiB all-card remaining margin required')
    global _ENGINEERING_PATH
    _ENGINEERING_PATH=args.engineering_receipt
    if args.detach:
        args.root.mkdir(parents=True,exist_ok=True)
        argv=[PYTHON,'-u',str(Path(__file__).resolve()),*[v for v in sys.argv[1:] if v!='--detach']]
        with (args.root/'owner.log').open('a') as stream:
            child=subprocess.Popen(argv,cwd=WORK,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(owner=proc_identity(child.pid),log=str(args.root/'owner.log'))));return
    run(args)


if __name__=='__main__':main()
