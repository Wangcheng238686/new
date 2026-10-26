"""Fixed ten-arm test evaluation, separate from train/validation and optimizer state."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
PYTHON='/data/wangcheng/envs/cvt2/bin/python'


def dump(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    a=parser.parse_args();r=a.root;t=r/'test';t.mkdir(exist_ok=True)
    assert not (t/'test_state.json').exists(),'Refuse duplicate test owner'
    arms=[dict(name=n,folder=n,family='nativegeo' if n.startswith('native_') else 'adapter',
               modes=['correct','constant','donor'] if n in ('correct','early','contextual','unit') else ['correct'])
          for n in ('correct','geometry','early','contextual','native_geometry','native_zero','unit','unit_geometry')]
    arms += [dict(name=n+'_control',folder=n+'/control',family='adapter',modes=['correct']) for n in ('early','contextual')]
    evaluations=[dict(key='baseline__disabled',name='baseline',mode='disabled',family='adapter',checkpoint=None,
                      out=str(t/'baseline'),stage='waiting')]
    for arm in arms:
        for mode in arm['modes']:
            evaluations.append(dict(key=arm['name']+'__'+mode,name=arm['name'],mode=mode,family=arm['family'],
                checkpoint=str(r/arm['folder']/'last.pth'),out=str(t/arm['name']),stage='waiting'))
    base=t/'baseline/disabled/records.json'
    analyses=[dict(key=arm['name']+'__baseline',baseline=str(base),candidate=str(t/arm['name']/'correct/records.json'),
                   output=str(t/arm['name']/'paired_analysis.json'),stage='waiting') for arm in arms]
    pairs=[('correct','geometry'),('early','early_control'),('contextual','contextual_control'),
           ('unit','unit_geometry'),('native_geometry','native_zero')]
    analyses += [dict(key=first+'__content',baseline=str(t/second/'correct/records.json'),
                     candidate=str(t/first/'correct/records.json'),output=str(t/first/'content_paired_analysis.json'),
                     stage='waiting') for first,second in pairs]
    plan=dict(registered_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),arms=arms,
              evaluation_count=len(evaluations),comparison_count=len(analyses),gpu=1,cpu_analysis_parallelism=3,
              split='test',selection='epoch10 last',bootstrap_samples=500,bootstrap_seed=20261021,
              score_key='scores',max_dets=[1,10,250],precision='fp32',
              protocol_sha256=sha(ROOT/'docs/dual_stream/UE_V1_FROZEN_ADAPTER_PROTOCOL_20261021.md'),
              analysis_sha256=sha(ROOT/'scripts/probes/ue_v1_adapter_analysis.py'),
              training_changes_from_test=False)
    assert len(arms)==10 and len(evaluations)==19 and len(analyses)==15
    dump(t/'frozen_test_plan.json',plan)
    state=dict(stage='running',owner_pid=os.getpid(),plan=plan,evaluations=evaluations,analyses=analyses)
    handles={};streams={}
    def save():dump(t/'test_state.json',state)
    def start(task,command,gpu=False):
        log=t/(task['key']+'.log');f=log.open('w')
        env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        if gpu:env['CUDA_VISIBLE_DEVICES']='1'
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,
                               stderr=subprocess.STDOUT,start_new_session=True)
        handles[task['key']]=child;streams[task['key']]=f
        task.update(stage='running',pid=child.pid,log=str(log));save();print('START',task['key'],child.pid,flush=True)
    def check_evaluation(task):
        path=Path(task['out'])/task['mode'];identity=json.loads((path/'evaluation_identity.json').read_text())
        assert identity['split']=='test' and identity['images']==459 and identity['inference_precision']=='fp32'
        assert identity['score_key']=='scores' and identity['max_dets']==[1,10,250]
        if task['checkpoint']:assert identity['adapter_sha256']==sha(Path(task['checkpoint']))
        rows=json.loads((path/'records.json').read_text());assert len(rows)==459
        if task['name']!='baseline':
            reference=json.loads(base.read_text())
            for b,d in zip(reference,rows):
                assert b['image_id']==d['image_id'] and b['gt']==d['gt']
                assert all(b['dt'][k]==d['dt'][k] for k in ('bboxes','scores','labels'))
        else:
            ids=set(json.loads((Path('/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021')/'splits/test.json').read_text()))
            assert {v['image_id'] for v in rows}==ids
        task['identity']=identity
    wrapper=str(ROOT/'scripts/ue_rgbd.sh');analysis=str(ROOT/'scripts/probes/ue_v1_adapter_analysis.py')
    try:
        save()
        while True:
            for task in evaluations+analyses:
                if task['stage']!='running':continue
                code=handles[task['key']].poll()
                if code is None:continue
                streams[task['key']].close()
                if code:raise RuntimeError(f"{task['key']} exited {code}")
                if task in evaluations:check_evaluation(task)
                else:
                    result=json.loads(Path(task['output']).read_text())
                    assert result['images']==459 and result['evaluation_split']=='test'
                    task['result']=result
                task['stage']='complete';save();print('DONE',task['key'],flush=True)
            if not any(v['stage']=='running' for v in evaluations):
                for task in evaluations:
                    if task['stage']!='waiting':continue
                    if task['name']!='baseline' and evaluations[0]['stage']!='complete':continue
                    if task['checkpoint'] and not Path(task['checkpoint']).exists():continue
                    # Keep dense test inference on an otherwise unused GPU.
                    used=int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
                    if used>2048:break
                    command=['bash',wrapper,task['family'],'eval','--eval-split','test','--out',task['out']]
                    if task['family']=='adapter':command+=['--mode',task['mode']]
                    if task['checkpoint']:command+=['--adapter',task['checkpoint']]
                    start(task,command,gpu=True);break
            for task in analyses:
                if sum(v['stage']=='running' for v in analyses)>=3:break
                if task['stage']!='waiting':continue
                dependencies=[v for v in evaluations if str(Path(v['out'])/v['mode']/'records.json') in (task['baseline'],task['candidate'])]
                if len(dependencies)!=2 or not all(v['stage']=='complete' for v in dependencies):continue
                start(task,[PYTHON,analysis,'--split','test','--baseline',task['baseline'],
                            '--candidate',task['candidate'],'--output',task['output']])
            if all(v['stage']=='complete' for v in evaluations+analyses):break
            time.sleep(20)
        state['stage']='analysis_complete';save()
    except BaseException as exc:
        state.update(stage='failed',error=str(exc));save();raise


if __name__=='__main__':main()
