"""Two content-paired Hiera joint-encoding arms; independent non-preemptive GPU slots."""
import argparse
import fcntl
import math
import os
from pathlib import Path
import subprocess
import time
from ue_geometry_mechanism_schedule import read,sha,dump,proc_identity,gpu_snapshot

ROOT = Path(__file__).resolve().parents[2]
PYTHON = '/data/wangcheng/envs/cvt2/bin/python'
INIT = Path('/data/wangcheng/checkpoint/ue_joint_encoding_20261023/common_init.pth')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
    lock=(r/'owner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert read(r/'cpu_gate.json')['passed'] and read(r/'wiring_gate.json')['passed']
    gate=read(r/'gate/gpu_gate.json');contract=read(r/'gate/gate_contract.json')
    assert contract['direction']=='J1' and contract['init_file_sha256']==sha(INIT)
    assert contract['args']['allocator_limit_mib']==5500
    assert max(gate['training_amp_peak']['reserved_MiB'],gate['fp32_250roi_peak']['reserved_MiB'])<5500
    assert gate['resource_rois']==250 and gate['satellite_parameters_buffers_exact'] and gate['satellite_no_grad']
    for v in gate['identity'].values(): assert v['zero_projection_logits_exact'] and v['no_support_nonzero_projection_logits_exact']
    hashes=gate['source_hashes'].copy()
    for n in ['scripts/probes/ue_joint_encoding_schedule.py','scripts/probes/ue_joint_encoding_bounded_worker.py','scripts/probes/ue_joint_encoding_cpu_gate.py','scripts/probes/ue_joint_encoding_wiring_gate.py','scripts/probes/ue_joint_encoding_analysis.py']:
        hashes[n]=sha(ROOT/n)
    assert all(sha(ROOT/n)==v for n,v in hashes.items())
    manifest=dict(study='J1_joint_encoding_content_pair_v1',source_hashes=hashes,gate_sha256=sha(r/'gate/gpu_gate.json'),
        cpu_gate_sha256=sha(r/'cpu_gate.json'),wiring_gate_sha256=sha(r/'wiring_gate.json'),init_sha256=sha(INIT),epochs=10,updates=2670,selection='E10 last',allowed_gpus=[0,1,2,3],
        maximum_owned_gpu_jobs_per_gpu=1,minimum_free_mib=5500+1536,allocator_ceiling_mib=5500,external_headroom_mib=1536,
        resource_policy='hard allocator bound same as successful GPU gate; minimum free=ceiling+1536MiB',
        measured_peak_reserved_mib=max(gate['training_amp_peak']['reserved_MiB'],gate['fp32_250roi_peak']['reserved_MiB']),
        no_ema=True,epoch_seed_formula='1044000+epoch',reference='frozen satellite; J1 changes feature extraction, proposals and scores',
        inference_modes='correct constant donor are RGB-only interventions; target geometry is retained')
    if (r/'manifest.json').exists():assert read(r/'manifest.json')==manifest
    else:dump(r/'manifest.json',manifest)
    jobs={}
    def add(key,cmd,artifact,deps=(),gpu=True,priority=10,need=None):
        jobs[key]=dict(key=key,cmd=cmd,artifact=str(artifact),dependencies=list(deps),gpu_required=gpu,
            priority=priority,status='pending',required_free_mib=need or manifest['minimum_free_mib'])
    probe=[PYTHON,str(ROOT/'scripts/probes/ue_joint_encoding_bounded_worker.py'),'--limit-mib','5500']
    common=['--direction','J1','--init',str(INIT)]
    add('baseline',probe+['eval']+common+['--content','correct','--mode','disabled','--out',str(r/'test/baseline')],r/'test/baseline/disabled/evaluation_identity.json',priority=5)
    for content in ['correct','constant']:
        train='train_'+content;folder=r/content
        add(train,probe+['train']+common+['--content',content,'--out',str(folder)],folder/'last.pth',priority=0)
        for split,modes in [('test',['correct','constant','donor'] if content=='correct' else ['constant']),('val',[content])]:
            for mode in modes:
                key=f'{split}_{content}_{mode}';out=r/split/content
                add(key,probe+['eval']+common+['--content',content,'--adapter',str(folder/'last.pth'),'--mode',mode,'--eval-split',split,'--out',str(out)],out/mode/'evaluation_identity.json',[train],priority=2 if mode==content and split=='test' else 4)
    def compare(key,left,right,deps,priority=7):
        out=r/'analysis'/(key+'.json')
        add(key,[PYTHON,str(ROOT/'scripts/probes/ue_joint_encoding_analysis.py'),'--baseline',str(left),'--candidate',str(right),'--output',str(out),'--split','test','--bootstrap','500'],out,deps,gpu=False,priority=priority)
    correct=r/'test/correct/correct/records.json';const=r/'test/constant/constant/records.json';base=r/'test/baseline/disabled/records.json'
    compare('content',const,correct,['test_correct_correct','test_constant_constant'],priority=1)
    compare('net_correct',base,correct,['baseline','test_correct_correct'])
    compare('net_constant',base,const,['baseline','test_constant_constant'])
    for mode in ['constant','donor']:
        compare('dependency_'+mode,r/'test/correct'/mode/'records.json',correct,['test_correct_'+mode,'test_correct_correct'])
    (r/'analysis').mkdir(exist_ok=True)
    dump(r/'job_plan.json',jobs)
    assert not (r/'state.json').exists(),'Existing owner state requires explicit audit; refuse duplicate launch'
    state=dict(owner_pid=os.getpid(),owner_identity=proc_identity(os.getpid()),stage='running',jobs=jobs)
    children={};streams={}
    while True:
        for key,child in list(children.items()):
            rc=child.poll()
            if rc is None:continue
            streams.pop(key).close();del children[key];j=jobs[key]
            j.update(returncode=rc,status='complete' if rc==0 and Path(j['artifact']).is_file() else 'failed')
            print('END',key,rc,j['status'],flush=True)
        if any(j['status']=='failed' for j in jobs.values()):
            state['stage']='failure_requires_audit';dump(r/'state.json',state)
            if not children:return
            time.sleep(15);continue
        if all(j['status']=='complete' for j in jobs.values()):
            state['stage']='complete';dump(r/'state.json',state);return
        assert all(sha(ROOT/n)==v for n,v in hashes.items()),'Frozen numerical source changed'
        busy={j['gpu'] for j in jobs.values() if j['status']=='running' and j['gpu_required']}
        cpu=sum(j['status']=='running' and not j['gpu_required'] for j in jobs.values())
        ready=sorted((j for j in jobs.values() if j['status']=='pending' and all(jobs[d]['status']=='complete' for d in j['dependencies'])),key=lambda j:j['priority'])
        for j in ready:
            gpu=None
            if j['gpu_required']:
                snapshot=gpu_snapshot()
                eligible=[d for d in snapshot['devices'] if d['gpu'] in (0,1,2,3) and d['gpu'] not in busy
                    and d['total_mib']-d['used_mib']>=j['required_free_mib']]
                if not eligible:continue
                gpu=max(eligible,key=lambda d:d['total_mib']-d['used_mib'])['gpu'];busy.add(gpu)
                j['admission']=dict(snapshot=snapshot,required_free_mib=j['required_free_mib'])
            elif cpu>=1:continue
            else:cpu+=1
            log=r/'logs'/(j['key']+'.log');log.parent.mkdir(exist_ok=True);stream=log.open('ab')
            env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
            if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
            child=subprocess.Popen(j['cmd'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            children[j['key']]=child;streams[j['key']]=stream
            j.update(status='running',gpu=gpu,pid=child.pid,process_identity=proc_identity(child.pid),log=str(log))
            print('START',j['key'],gpu,child.pid,flush=True)
            dump(r/'state.json',state)
        dump(r/'state.json',state);time.sleep(15)


if __name__=='__main__':main()
