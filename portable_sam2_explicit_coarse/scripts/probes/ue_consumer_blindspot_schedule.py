"""Four content-paired arms for confirmed consumer wiring gaps, GPU0/1 only."""
import argparse
import fcntl
import math
import os
from pathlib import Path
import subprocess
import time
from ue_geometry_mechanism_schedule import read,sha,dump,proc_identity,gpu_snapshot

ROOT=Path(__file__).resolve().parents[2]
PYTHON='/data/wangcheng/envs/cvt2/bin/python'
CHECKPOINT=Path('/data/wangcheng/checkpoint')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;r.mkdir(exist_ok=True)
    lock=(r/'owner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    studies={
        'roi':dict(root=CHECKPOINT/'ue_udpr_point_roi_20261023',runner='ue_udpr_point_roi_probe.py',direction='S1'),
        'shared':dict(root=CHECKPOINT/'ue_post_activation_shared_20261023',runner='ue_post_activation_shared_probe.py',direction='N2')}
    hashes={};gates={}
    for name,c in studies.items():
        folder=c['root'];g=read(folder/'gate/gpu_gate.json');contract=read(folder/'gate/gate_contract.json')
        assert contract['direction']==c['direction'] and contract['init_file_sha256']==sha(folder/'common_init.pth')
        assert g['resource_rois']==250 and g['satellite_parameters_buffers_exact'] and g['satellite_no_grad']
        for item in g['identity'].values():assert item['zero_alpha_logits_exact'] and item['no_support_nonzero_alpha_logits_exact']
        specific=folder/('evaluation_gate.json' if name=='roi' else 'consumer_gate.json');assert read(specific)['passed']
        if name=='roi':assert read(folder/'cpu_gate.json')['translation_scale_and_target_crop_verified']
        hashes.update(g['source_hashes'])
        peak=max(g['training_amp_peak']['reserved_MiB'],g['fp32_250roi_peak']['reserved_MiB'])
        gates[name]=dict(peak_reserved_mib=peak,required_free_mib=3072+math.ceil(1.25*peak),
            gpu_gate_sha256=sha(folder/'gate/gpu_gate.json'),consumer_gate_sha256=sha(specific),init_sha256=sha(folder/'common_init.pth'))
    for n in ['scripts/probes/ue_consumer_blindspot_schedule.py','scripts/probes/ue_udpr_point_roi_cpu_gate.py',
              'scripts/probes/ue_udpr_point_roi_evaluation_gate.py','scripts/probes/ue_post_activation_shared_gate.py',
              'scripts/probes/ue_udpr_point_summary.py','scripts/probes/ue_v1_adapter_analysis.py']:
        hashes[n]=sha(ROOT/n)
    assert all(sha(ROOT/n)==h for n,h in hashes.items())
    manifest=dict(studies={k:{kk:str(vv) for kk,vv in c.items()} for k,c in studies.items()},source_hashes=hashes,gates=gates,
        allowed_gpus=[0,1],maximum_owned_jobs_per_gpu=2,epochs=10,updates=2670,
        selection='E10 last',training_content=['correct','constant'],no_ema=True,
        admission='Each job requires 3GiB + 1.25x measured max reserved; at most two owned jobs/device; external owners not interrupted')
    assert not (r/'state.json').exists(),'Existing state requires explicit audit'
    dump(r/'manifest.json',manifest)
    for name in hashes:
        target=r/'frozen_source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    jobs={}
    def add(key,cmd,artifact,study,deps=(),gpu=True,priority=5):
        jobs[key]=dict(key=key,cmd=cmd,artifact=str(artifact),study=study,dependencies=list(deps),gpu_required=gpu,priority=priority,status='pending')
    for name,c in studies.items():
        folder=c['root'];(folder/'analysis').mkdir(exist_ok=True)
        probe=[PYTHON,str(ROOT/'scripts/probes'/c['runner'])];common=['--direction',c['direction'],'--init',str(folder/'common_init.pth')]
        add(name+'_baseline',probe+['eval']+common+['--content','correct','--mode','disabled','--out',str(folder/'test/baseline')],folder/'test/baseline/disabled/evaluation_identity.json',name,priority=8)
        for content in ['correct','constant']:
            train=name+'_train_'+content
            add(train,probe+['train']+common+['--content',content,'--out',str(folder/content)],folder/content/'last.pth',name,priority=0 if name=='roi' else 1)
            for split,modes in [('test',['correct','constant','donor'] if content=='correct' else ['constant']),('val',[content])]:
                for mode in modes:
                    key=f'{name}_{split}_{content}_{mode}'
                    add(key,probe+['eval']+common+['--content',content,'--adapter',str(folder/content/'last.pth'),'--mode',mode,'--eval-split',split,'--out',str(folder/split/content)],folder/split/content/mode/'evaluation_identity.json',name,[train],priority=3 if split=='test' and content==mode else 6)
        def compare(key,left,right,deps,priority=7):
            out=folder/'analysis'/(key+'.json')
            add(name+'_analysis_'+key,[PYTHON,str(ROOT/'scripts/probes/ue_v1_adapter_analysis.py'),'--baseline',str(left),'--candidate',str(right),'--output',str(out),'--split','test','--bootstrap','500'],out,name,deps,gpu=False,priority=priority)
        correct=folder/'test/correct/correct/records.json';const=folder/'test/constant/constant/records.json';base=folder/'test/baseline/disabled/records.json'
        compare('content',const,correct,[name+'_test_correct_correct',name+'_test_constant_constant'],priority=2)
        compare('net_correct',base,correct,[name+'_baseline',name+'_test_correct_correct'])
        compare('net_constant',base,const,[name+'_baseline',name+'_test_constant_constant'])
        for mode in ['constant','donor']:
            compare('dependency_'+mode,folder/'test/correct'/mode/'records.json',correct,[name+'_test_correct_'+mode,name+'_test_correct_correct'])
    dump(r/'job_plan.json',jobs)
    state=dict(stage='running',owner_pid=os.getpid(),owner_identity=proc_identity(os.getpid()),jobs=jobs);children={};streams={}
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
        if all(j['status']=='complete' for j in jobs.values()):state['stage']='complete';dump(r/'state.json',state);return
        assert all(sha(ROOT/n)==h for n,h in hashes.items()),'Frozen source changed'
        owned={g:sum(j['status']=='running' and j.get('gpu')==g for j in jobs.values()) for g in [0,1]}
        cpu=sum(j['status']=='running' and not j['gpu_required'] for j in jobs.values())
        ready=sorted((j for j in jobs.values() if j['status']=='pending' and all(jobs[d]['status']=='complete' for d in j['dependencies'])),key=lambda j:j['priority'])
        for j in ready:
            gpu=None
            if j['gpu_required']:
                snapshot=gpu_snapshot();need=gates[j['study']]['required_free_mib']
                eligible=[d for d in snapshot['devices'] if d['gpu'] in owned and owned[d['gpu']]<2 and d['total_mib']-d['used_mib']>=need]
                if not eligible:continue
                gpu=min(eligible,key=lambda d:(owned[d['gpu']],d['used_mib']))['gpu'];owned[gpu]+=1
                j['admission']=dict(snapshot=snapshot,required_free_mib=need)
            elif cpu>=1:continue
            else:cpu+=1
            log=r/'logs'/(j['key']+'.log');log.parent.mkdir(exist_ok=True);stream=log.open('ab')
            env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
            if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
            child=subprocess.Popen(j['cmd'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            children[j['key']]=child;streams[j['key']]=stream
            j.update(status='running',gpu=gpu,pid=child.pid,process_identity=proc_identity(child.pid),log=str(log))
            dump(r/'state.json',state);print('START',j['key'],gpu,child.pid,flush=True)
        dump(r/'state.json',state);time.sleep(15)


if __name__=='__main__':main()
