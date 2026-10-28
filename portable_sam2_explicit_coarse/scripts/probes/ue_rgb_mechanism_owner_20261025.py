"""Non-preemptive, gate-controlled co-resident mechanism prefix jobs."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from ue_rgb_mechanism_run_20261025 import ROOT, WORK, VARIANTS, hashes, base

PYTHON='/data/wangcheng/envs/cvt2/bin/python'
RUN=WORK/'scripts/probes/ue_rgb_mechanism_run_20261025.py'


def main():
    ROOT.mkdir(exist_ok=True)
    lock=(ROOT/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'state.json').exists(), 'Refuse duplicate owner'
    source=hashes()
    driver_names=['scripts/probes/ue_rgb_mechanism_owner_20261025.py',
        'scripts/probes/ue_rgb_mechanism_analysis_20261025.py',
        'scripts/probes/ue_rgb_mechanism_satellite_reference_20261025.py',
        'scripts/probes/ue_joint_encoding_fast_analysis.py',
        'scripts/probes/ue_joint_encoding_analysis.py']
    drivers={n:base.core.C.file_sha(WORK/n) for n in driver_names}
    for v in VARIANTS:
        g=json.loads((ROOT/'gates'/v/'gpu_gate.json').read_text())
        assert g['passed'] and g['source_hashes']==source
        assert g['peak_reserved_MiB']<=7800
    cpu=json.loads((ROOT/'cpu_gate.json').read_text());assert cpu['passed'] and cpu['source_hashes']==source
    jobs={}
    def add(key,command,artifact,requires=(),gpu=True):
        jobs[key]=dict(command=command,artifact=str(artifact),dependencies=list(requires),gpu=gpu,status='pending')
    for v in VARIANTS:
        for c in ('correct','constant'):
            key=v+'_'+c
            common=[PYTHON,'-u',str(RUN)]
            options=['--variant',v,'--content',c]
            add(key,common+['train']+options,ROOT/v/c/'training_complete.json')
            for mode in (['matched','donor','constant'] if c=='correct' else ['matched']):
                add(key+'_'+mode,common+['eval']+options+['--mode',mode],
                    ROOT/'evaluation'/v/c/mode/'complete.json',[key])
    for c in ('correct','constant'):
        add('baseline_'+c,[PYTHON,'-u',str(RUN),'eval','--variant','baseline','--content',c],
            ROOT/'evaluation'/'baseline'/c/'matched'/'complete.json')
    add('baseline_satellite',[PYTHON,'-u',str(WORK/'scripts/probes/ue_rgb_mechanism_satellite_reference_20261025.py')],
        ROOT/'evaluation'/'satellite'/'correct'/'matched'/'complete.json')
    evaluation_keys=[k for k,j in jobs.items() if 'evaluation' in j['artifact']]
    add('analysis',[PYTHON,'-u',str(WORK/'scripts/probes/ue_rgb_mechanism_analysis_20261025.py')],
        ROOT/'analysis'/'summary.json',evaluation_keys,gpu=False)
    manifest=dict(variants=VARIANTS,trainings=6,epochs=10,updates=2670,
        initialization_file=str(base.ROOT/'common_init.pth'),
        common_init_sha256=base.core.C.file_sha(base.ROOT/'common_init.pth'),
        source_hashes=source,driver_hashes=drivers,allocator_cap_MiB=7800,minimum_free_MiB=11000,
        maximum_owned_jobs_per_gpu=1,CPU_analysis_concurrency=1,
        comparison='fixed E10 val, paired correct/constant and content interaction; original and combined paths',
        note='No source or state of original four-arm training modified')
    base.core.C.dump(ROOT/'manifest.json',manifest);base.core.C.dump(ROOT/'job_plan.json',jobs)
    state=dict(stage='running',owner_pid=os.getpid(),jobs=jobs)
    children,streams={},{}
    while True:
        for key,child in list(children.items()):
            rc=child.poll()
            if rc is None:continue
            streams.pop(key).close();del children[key]
            jobs[key].update(returncode=rc,status='complete' if rc==0 and Path(jobs[key]['artifact']).exists() else 'failed')
            print('END',key,rc,jobs[key]['status'],flush=True)
        if any(j['status']=='failed' for j in jobs.values()):state['stage']='failure_requires_audit'
        else:
            assert hashes()==source, 'Mechanism source changed'
            assert all(base.core.C.file_sha(WORK/n)==sha for n,sha in drivers.items()), 'Driver source changed'
            rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
            free={int(r.split(',')[0]):int(r.split(',')[1]) for r in rows.splitlines()}
            busy={jobs[k].get('gpu_index') for k in children if jobs[k]['gpu']}
            for key,j in jobs.items():
                if j['status']!='pending' or any(jobs[d]['status']!='complete' for d in j['dependencies']):continue
                env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
                if j['gpu']:
                    available=[g for g in free if free[g]>=11000 and g not in busy]
                    if not available:continue
                    gpu=max(available,key=lambda g:free[g]);busy.add(gpu)
                    env['CUDA_VISIBLE_DEVICES']=str(gpu);j.update(gpu_index=gpu,admission_free_MiB=free[gpu])
                else:env['CUDA_VISIBLE_DEVICES']=''
                log=ROOT/'logs'/(key+'.log');log.parent.mkdir(exist_ok=True)
                stream=log.open('w')
                child=subprocess.Popen(j['command'],cwd=WORK,env=env,stdin=subprocess.DEVNULL,
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                j.update(status='running',pid=child.pid,log=str(log));children[key]=child;streams[key]=stream
                print('START',key,j.get('gpu_index'),child.pid,flush=True)
        if all(j['status']=='complete' for j in jobs.values()):state['stage']='complete'
        base.core.C.dump(ROOT/'state.json',state)
        if state['stage']=='complete' or (state['stage']=='failure_requires_audit' and not children):return
        time.sleep(15)


if __name__=='__main__':main()
