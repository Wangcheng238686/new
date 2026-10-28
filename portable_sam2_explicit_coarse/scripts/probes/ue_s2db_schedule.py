"""Durable non-preemptive owner of the predeclared 13-run mechanism cohort.

Old experiment owners retain their device reservations, even in short idle gaps.
On child failure the job is marked failed; no AP-driven early stop or retuning.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import fcntl

ROOT=Path(__file__).resolve().parents[2]
PY='/data/wangcheng/envs/cvt2/bin/python'
OLD=Path('/data/wangcheng/checkpoint/ue_v1_adapter_probe_20261021')
CACHE='/data/wangcheng/cache/ue_v1_s2db_geometry_v2'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,a):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(a,indent=2)+'\n');tmp.replace(p)
def alive(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False


def old_reserved():
    reserved=set()
    try:
        u=read(OLD/'unit_state.json')
        for a in u['arms']:
            if a['stage'] not in ('done','analysis','content_analysis','failed'):reserved.add(a['gpu'])
        t=read(OLD/'test/test_state.json')
        if any(a['stage'] not in ('complete','failed') for a in t['evaluations']):reserved.add(1)
        r=read(OLD/'run_state.json')
        if r['stage'] not in ('analysis_complete','done','failed'):reserved.add(2)
        c=read(OLD/'contextual_control_state.json')
        if c['stage'] not in ('analysis_complete','done','failed'):reserved.add(3)
    except (FileNotFoundError,KeyError):return {0,1,2,3}
    return reserved


def idle_gpus():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    return [int(line.split(',')[0]) for line in text.splitlines() if int(line.split(',')[1])<200]


def declared_arms():
    arms=[]
    for n,read_mode,fusion in [('A','fixed','normalized'),('B','local','normalized'),('C','fixed','concat'),('D','local','concat')]:
        arms.append(dict(name=n,read=read_mode,fusion=fusion,content='correct',aux_weight=0.,stage=1))
    for n in 'ABCD':
        original=next(a for a in arms if a['name']==n)
        arms.append(dict(original,name=n+'_constant',content='constant'))
    for n in 'CD':
        original=next(a for a in arms if a['name']==n)
        arms.append(dict(original,name=n+'_zero',content='zero'))
    for c,suffix in [('correct',''),('constant','_constant'),('zero','_zero')]:
        arms.append(dict(name='D_aux'+suffix,read='local',fusion='concat',content=c,aux_weight=.1,stage=2))
    return arms


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;r.mkdir(parents=True,exist_ok=True)
    lock=(r/'owner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source_files=['scripts/probes/ue_s2db_probe.py','scripts/probes/ue_s2db_gate.py','scripts/probes/ue_s2db_analysis.py',
        'scripts/probes/ue_s2db_aux_audit.py','scripts/probes/ue_s2db_cache_audit.py','scripts/probes/ue_s2db_schedule.py',
        'scripts/probes/ue_s2db_collect.py','rsprompter/uav/s2db_adapter.py','rsprompter/uav/s2db_geometry.py',
        'scripts/probes/ue_v1_adapter.py','rsprompter/uav/local_adapter_probe.py','rsprompter/uav/local_adapter_variants.py',
        'data/ue_minimal_dataset.py','data/whu_instance_dataset.py','scripts/probes/ue_v1_adapter_analysis.py']
    source_files += ['utils/coco_eval_utils.py','inference/model_rebuild.py','inference/eval_records.py',
        'train/gt_samples.py','data/loader.py','rsprompter/ckpt_utils.py','rsprompter/architecture_contract.py']
    for package in ('backbone','cspm','decoder','detection'):
        source_files += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'rsprompter'/package).glob('*.py'))]
    source_files=list(dict.fromkeys(source_files))
    manifest=dict(study='s2db_mechanism_v1',registered_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),
        arms=declared_arms(),stage1_training_count=10,stage2_extra_training_count=3,
        stage2_structure='D fixed before new test',init='same complete common_init.pth for every arm',
        primary_split='test',selection='E10 last',bootstrap=500,bootstrap_seed=20261021,
        source_hashes={f:sha(ROOT/f) for f in source_files},
        protocol_sha256=sha(ROOT/'docs/dual_stream/UE_S2DB_MECHANISM_PROTOCOL_20261022.md'),
        base_sha256='dd9106f4afe8ba72936316d7efa9621ca0c05d24836fa28c72ea322bed84ace1',
        observed_coverage_bins=[.5,.8],density_bins=[50,100],no_test_driven_changes=True)
    if (r/'manifest.json').exists():
        saved=read(r/'manifest.json');assert saved['source_hashes']==manifest['source_hashes'];manifest=saved
    else:
        dump(r/'manifest.json',manifest)
        for path in source_files:
            target=r/'frozen_source'/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/path).read_bytes())
    state=read(r/'state.json') if (r/'state.json').exists() else dict(stage='engineering',jobs={})
    state['owner_pid']=os.getpid();children={};streams={}
    def save():dump(r/'state.json',state)
    def add(key,cmd,gpu=False,priority=10,dependencies=(),artifact=None):
        if key not in state['jobs']:
            state['jobs'][key]=dict(key=key,cmd=cmd,gpu_required=gpu,priority=priority,dependencies=list(dependencies),
                status='complete' if artifact and Path(artifact).exists() else 'pending',artifact=str(artifact) if artifact else None)
    probe=[PY,str(ROOT/'scripts/probes/ue_s2db_probe.py')]
    analysis=[PY,str(ROOT/'scripts/probes/ue_s2db_analysis.py')]
    add('cache_audit_trainval',[PY,str(ROOT/'scripts/probes/ue_s2db_cache_audit.py'),'--split','trainval','--out',str(r/'cache_audit_trainval.json')],
        priority=0,artifact=r/'cache_audit_trainval.json')
    add('init',probe+['init','--out',str(r)],True,0,artifact=r/'common_init.pth')
    gatekeys=[]
    for n,rd,fu,aux in [('A','fixed','normalized',0),('B','local','normalized',0),('C','fixed','concat',0),('D','local','concat',0),('D_aux','local','concat',.1)]:
        key='gate_'+n;gatekeys.append(key)
        add(key,probe+['gate','--read',rd,'--fusion',fu,'--aux-weight',str(float(aux)),
            '--init',str(r/'common_init.pth'),'--out',str(r/'gates'/n)],True,1,['init'],r/'gates'/n/'gpu_gate.json')
    add('baseline_test',probe+['eval','--mode','disabled','--eval-split','test','--init',str(r/'common_init.pth'),
        '--out',str(r/'test/baseline')],True,8,gatekeys,r/'test/baseline/disabled/evaluation_identity.json')
    add('baseline_metrics',analysis+['metrics','--records',str(r/'test/baseline/disabled/records.json'),
        '--out',str(r/'analysis/baseline')],False,8,['baseline_test'],r/'analysis/baseline/metrics.json')
    trainkeys=[];stage1reviewdeps=[]
    for arm in manifest['arms']:
        name=arm['name'];folder=r/name;tk='train_'+name;trainkeys.append(tk)
        traincmd=probe+['train','--out',str(folder),'--init',str(r/'common_init.pth')]
        for k in ('read','fusion','content','aux_weight'):traincmd+=['--'+k.replace('_','-'),str(arm[k])]
        deps=gatekeys+['cache_audit_trainval']
        if arm['stage']==2:deps+=['stage1_review']
        add(tk,traincmd,True,5 if name in list('ABCD') else 30,deps,folder/'last.pth')
        for split,modes in [('val',['constant' if arm['content']=='constant' else 'correct']),('test',['correct','constant','donor'])]:
            for mode in modes:
                key=f'eval_{split}_{name}_{mode}'
                add(key,probe+['eval','--adapter',str(folder/'last.pth'),'--mode',mode,'--eval-split',split,
                    '--out',str(r/split/name)],True,15,[tk],r/split/name/mode/'evaluation_identity.json')
        primary='constant' if arm['content']=='constant' else 'correct'
        ek=f'eval_test_{name}_{primary}';mk='metrics_'+name
        add(mk,analysis+['metrics','--records',str(r/'test'/name/primary/'records.json'),'--out',str(r/'analysis'/name)],
            False,10,[ek],r/'analysis'/name/'metrics.json')
        add('mechanisms_'+name,analysis+['mechanisms','--baseline',str(r/'test/baseline/disabled/records.json'),
            '--candidate',str(r/'test'/name/primary/'records.json'),'--cache',CACHE,'--out',str(r/'analysis'/name/'mechanisms.json')],
            False,12,['baseline_test',ek],r/'analysis'/name/'mechanisms.json')
        # Full metric point reports for both non-primary RGB interventions, no
        # additional training selection. Their bootstrap is unnecessary for contrasts.
        for mode in ('correct','constant','donor'):
            if mode==primary:continue
            add('intervention_metrics_'+name+'_'+mode,analysis+['metrics','--bootstrap','0',
                '--records',str(r/'test'/name/mode/'records.json'),'--out',str(r/'analysis'/name/mode)],
                False,13,[f'eval_test_{name}_{mode}'],r/'analysis'/name/mode/'metrics.json')
        for mode in ('constant','donor'):
            add('content_changes_'+name+'_'+mode,analysis+['changes',
                '--first',str(r/'test'/name/'correct/records.json'),'--second',str(r/'test'/name/mode/'records.json'),
                '--out',str(r/'analysis'/name/(mode+'_prediction_changes.json'))],False,14,
                [f'eval_test_{name}_correct',f'eval_test_{name}_{mode}'],r/'analysis'/name/(mode+'_prediction_changes.json'))
        if arm['stage']==1:
            stage1reviewdeps+=['metrics_'+name,'mechanisms_'+name]
            stage1reviewdeps += [f'eval_test_{name}_{mode}' for mode in ('correct','constant','donor')]
            stage1reviewdeps += ['intervention_metrics_'+name+'_'+mode for mode in ('correct','constant','donor') if mode!=primary]
            stage1reviewdeps += ['content_changes_'+name+'_'+mode for mode in ('constant','donor')]
        if arm['aux_weight']:
            for epoch in (1,3,10):
                key=f'gradient_{name}_E{epoch}'
                add(key,[PY,str(ROOT/'scripts/probes/ue_s2db_aux_audit.py'),'gradients','--adapter',str(folder/f'epoch{epoch}.pth'),
                    '--out',str(r/'offline'/name/f'E{epoch}')],True,20,[tk],r/'offline'/name/f'E{epoch}'/'gradients.json')
            add('aux_iou_'+name,[PY,str(ROOT/'scripts/probes/ue_s2db_aux_audit.py'),'aux_iou','--adapter',str(folder/'last.pth'),
                '--eval-split','test','--out',str(r/'offline'/name/'test')],True,20,[tk],r/'offline'/name/'test/aux_iou.json')
    add('stage1_review',[PY,str(ROOT/'scripts/probes/ue_s2db_collect.py'),'stage1','--root',str(r)],False,10,
        stage1reviewdeps+['baseline_metrics'],r/'stage1_mechanism_review.json')
    add('cache_audit_test',[PY,str(ROOT/'scripts/probes/ue_s2db_cache_audit.py'),'--split','test','--out',str(r/'cache_audit_test.json')],
        False,20,['baseline_test'],r/'cache_audit_test.json')
    add('contrasts',analysis+['contrasts','--root',str(r)],False,20,
        ['baseline_metrics']+['metrics_'+a['name'] for a in manifest['arms']],r/'contrasts.json')
    prefinal=list(state['jobs'])
    add('collect',[PY,str(ROOT/'scripts/probes/ue_s2db_collect.py'),'final','--root',str(r)],False,100,prefinal,r/'complete_audit_summary.json')
    # Adopt live children after owner restart; completed artifact is required,
    # never infer success from disappearance alone.
    for job in state['jobs'].values():
        if job['status']=='running' and not alive(job['pid']):
            job['status']='complete' if job['artifact'] and Path(job['artifact']).exists() else 'failed'
    save()
    while True:
        for key,proc in list(children.items()):
            rc=proc.poll()
            if rc is None:continue
            streams.pop(key).close();del children[key]
            job=state['jobs'][key]
            job.update(returncode=rc,status='complete' if rc==0 and (not job['artifact'] or Path(job['artifact']).exists()) else 'failed')
            print('END',key,job['status'],rc,flush=True);save()
        # Restart-adopted process completion.
        for key,job in state['jobs'].items():
            if job['status']=='running' and key not in children and not alive(job['pid']):
                job['status']='complete' if job['artifact'] and Path(job['artifact']).exists() else 'failed';save()
        if all(j['status']=='complete' for j in state['jobs'].values()):
            state['stage']='complete';save();return
        failures=[k for k,j in state['jobs'].items() if j['status']=='failed']
        if failures:
            state.update(stage='failure_requires_repair',failed_jobs=failures);save()
            # Running siblings finish; never kill existing/other jobs.
            if not any(j['status']=='running' for j in state['jobs'].values()):return
            time.sleep(15);continue
        changed=[f for f,digest in manifest['source_hashes'].items() if sha(ROOT/f)!=digest]
        assert not changed,('Frozen source changed',changed)
        reserved=old_reserved();running=[j for j in state['jobs'].values() if j['status']=='running']
        busy={j.get('gpu') for j in running if j['gpu_required']}
        available=[g for g in idle_gpus() if g not in reserved and g not in busy]
        cpu_running=sum(not j['gpu_required'] for j in running)
        ready=[j for j in state['jobs'].values() if j['status']=='pending' and all(state['jobs'][k]['status']=='complete' for k in j['dependencies'])]
        ready.sort(key=lambda j:(j['priority'],list(state['jobs']).index(j['key'])))
        for j in ready:
            if j['key']=='cache_audit_trainval' and not (r/'geometry_complete.json').exists():continue
            if not (r/'cpu_gate.json').exists():continue
            if j['gpu_required']:
                if not available:continue
                gpu=available.pop(0)
            else:
                if cpu_running>=2:continue
                gpu=None;cpu_running+=1
            log=r/'logs'/f"{j['key']}.log";log.parent.mkdir(parents=True,exist_ok=True);stream=log.open('ab')
            env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
            if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
            child=subprocess.Popen(j['cmd'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            children[j['key']]=child;streams[j['key']]=stream
            j.update(status='running',pid=child.pid,gpu=gpu,log=str(log),started_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()))
            print('START',j['key'],gpu,child.pid,flush=True);save()
        state.update(stage='running',old_reserved_gpus=sorted(reserved));save();time.sleep(15)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if '--root' in __import__('sys').argv:
            argv=__import__('sys').argv;r=Path(argv[argv.index('--root')+1])
            if (r/'state.json').exists():
                state=read(r/'state.json');state.update(stage='owner_failed',owner_error=repr(exc));dump(r/'state.json',state)
        raise
