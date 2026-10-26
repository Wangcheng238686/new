"""Detached recovery owner for all six authorized probes and their final comparisons."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
PYTHON='/data/wangcheng/envs/cvt2/bin/python'


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
    wrapper=str(ROOT/'scripts/ue_rgbd.sh');analysis=str(ROOT/'scripts/probes/ue_v1_adapter_analysis.py')
    arms=[]
    for name,gpu in [('correct',0),('geometry',1),('early',2),('contextual',3),('native_geometry',1),('native_zero',3)]:
        plan=json.loads((r/name/'recovery_plan.json').read_text())
        old=json.loads((r/name/'interrupted_20261022_0423/frozen_training_contract.json').read_text())
        native=name.startswith('native_');args=old['args']
        arms.append(dict(name=name,gpu=gpu,family='nativegeo' if native else 'adapter',
            route=args.get('route','highres'),source=args.get('source_mode','fine'),
            appearance=args.get('appearance','correct'),content=args.get('content'),
            prefix=plan['epoch'],parent=plan['path'],stage='pending'))
    state=dict(stage='recovering',owner_pid=os.getpid(),arms=arms,
        rationale='all probe processes disappeared near 04:23; no Python traceback/new kernel OOM observed; exact signal unknown',
        rng_policy='RGB-D common epoch4 + deterministic epochs5..10; native common epoch9 + deterministic epoch10')
    handles={};streams={};pairs={}
    def dump(path,value):
        tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
    def save():
        dump(r/'recovery_state.json',state)
        rgb=[v for v in arms if v['family']=='adapter'];native=[v for v in arms if v['family']=='nativegeo']
        for filename,group,pair in [('run_state.json',rgb,'rgb_content'),('native_state.json',native,'native_content')]:
            done=all(v['stage']=='done' for v in group) and pairs.get(pair,{}).get('stage')=='done'
            dump(r/filename,dict(stage='analysis_complete' if done else 'recovering',owner_pid=os.getpid(),arms=group,
                 pairs=pairs,rng_policy=state['rng_policy'],remaining=[] if done else ['training/evaluation/paired analysis']))
    def configuration(arm,constant=False):
        if arm['family']=='nativegeo':return ['--content',arm['content']]
        return ['--route',arm['route'],'--source-mode',arm['source'],'--appearance','constant' if constant else arm['appearance']]
    def spawn(arm,stage,command,gpu=None):
        label=arm['name']+'_'+stage;log=r/f'{label}.log';f=log.open('w')
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(arm['gpu'] if gpu is None else gpu),PYTHONUNBUFFERED='1')
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        handles[arm['name']]=child;streams[arm['name']]=f
        arm.update(stage=stage,pid=child.pid,log=str(log));save();print('START',label,child.pid,flush=True)
    def ev(arm,mode='correct',control=False):
        out=r/arm['name'];cp=out/('control/last.pth' if control else 'last.pth')
        command=['bash',wrapper,arm['family'],'eval','--adapter',str(cp)]
        if arm['family']=='adapter':command+=['--mode',mode]
        return command+['--out',str(out/('control_evaluation' if control else 'evaluation'))]
    def compare(arm,base,candidate,filename):
        return [PYTHON,analysis,'--baseline',str(base),'--candidate',str(candidate),'--output',str(r/arm['name']/filename)]
    try:
        for arm in arms:
            command=['bash',wrapper,arm['family'],'train',*configuration(arm),'--resume-from',arm['parent'],
                     '--reseed-from-epoch',str(arm['prefix']+1),'--out',str(r/arm['name'])]
            spawn(arm,'recovery_train',command)
        save()
        while True:
            for arm in arms:
                stage=arm['stage'];name=arm['name'];out=r/name
                if stage in ('done','failed'):continue
                rc=handles[name].poll()
                if rc is None:continue
                streams[name].close()
                if rc:
                    arm.update(stage='failed',returncode=rc);save();print('FAILED',name,stage,rc,flush=True);continue
                print('END',name,stage,flush=True)
                if stage=='recovery_train':
                    assert (out/'last.pth').is_file() and (out/'parameter_changes.json').is_file()
                    spawn(arm,'recovery_eval_correct',ev(arm))
                elif stage=='recovery_eval_correct':
                    if arm['family']=='adapter' and arm['appearance']=='correct':spawn(arm,'recovery_eval_constant',ev(arm,'constant'))
                    else:spawn(arm,'recovery_analysis',compare(arm,r/'evaluation/disabled/records.json',out/'evaluation/correct/records.json','paired_analysis.json'))
                elif stage=='recovery_eval_constant':spawn(arm,'recovery_eval_donor',ev(arm,'donor'))
                elif stage=='recovery_eval_donor':spawn(arm,'recovery_analysis',compare(arm,r/'evaluation/disabled/records.json',out/'evaluation/correct/records.json','paired_analysis.json'))
                elif stage=='recovery_analysis':
                    result=json.loads((out/'paired_analysis.json').read_text());arm.update(result=result['metrics'],gate=result['gate'])
                    if name in ('early','contextual') and result['gate']['direction_positive']:
                        command=['bash',wrapper,'adapter','train',*configuration(arm,True),'--reseed-from-epoch','5','--out',str(out/'control')]
                        spawn(arm,'recovery_control_train',command)
                    else:arm['stage']='done';save()
                elif stage=='recovery_control_train':spawn(arm,'recovery_control_eval',ev(arm,control=True))
                elif stage=='recovery_control_eval':spawn(arm,'recovery_control_analysis',compare(arm,out/'control_evaluation/correct/records.json',out/'evaluation/correct/records.json','appearance_paired_analysis.json'))
                elif stage=='recovery_control_analysis':arm['stage']='done';save()
            for key,first,second,filename in [('native_content','native_geometry','native_zero','content_paired_analysis.json'),
                                             ('rgb_content','correct','geometry','appearance_paired_analysis.json')]:
                deps=[next(v for v in arms if v['name']==n) for n in (first,second)]
                if key not in pairs and all(v['stage']=='done' for v in deps):
                    command=compare(deps[0],r/second/'evaluation/correct/records.json',r/first/'evaluation/correct/records.json',filename)
                    log=(r/f'recovery_pair_{key}.log').open('w')
                    child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    handles[key]=child;streams[key]=log;pairs[key]=dict(stage='running',pid=child.pid);save()
                elif key in pairs and pairs[key]['stage']=='running':
                    rc=handles[key].poll()
                    if rc is not None:
                        streams[key].close();pairs[key].update(stage='done' if rc==0 else 'failed',returncode=rc);save()
                elif key not in pairs and any(v['stage']=='failed' for v in deps):pairs[key]=dict(stage='failed',error='dependency failed');save()
            terminal=all(v['stage'] in ('done','failed') for v in arms)
            pairs_terminal=len(pairs)==2 and all(v['stage'] in ('done','failed') for v in pairs.values())
            if terminal and pairs_terminal:break
            time.sleep(30)
        state['stage']='analysis_complete' if all(v['stage']=='done' for v in arms) and all(v['stage']=='done' for v in pairs.values()) else 'partial_failure'
        save()
    except BaseException as exc:
        state.update(stage='failed',error=str(exc));save();raise


if __name__=='__main__':main()
