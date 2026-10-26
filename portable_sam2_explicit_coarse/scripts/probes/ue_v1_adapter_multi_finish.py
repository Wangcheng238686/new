"""Parallel fixed-budget probe owner. Existing highres training is adopted, not restarted."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
PYTHON='/data/wangcheng/envs/cvt2/bin/python'


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--existing-pid',type=int,required=True);a=p.parse_args()
    wrapper=str(ROOT/'scripts/ue_rgbd.sh');analysis=str(ROOT/'scripts/probes/ue_v1_adapter_analysis.py')
    baseline=a.root/'evaluation/disabled/records.json'
    assert baseline.is_file()
    arms=[dict(name='correct',gpu=0,route='highres',source='fine',appearance='correct',stage='training',pid=a.existing_pid,external=True),
          dict(name='geometry',gpu=1,route='highres',source='fine',appearance='constant',smoke='smoke_geometry',stage='ready'),
          dict(name='early',gpu=2,route='early',source='fine',appearance='correct',smoke='smoke_early',stage='ready'),
          dict(name='contextual',gpu=3,route='highres',source='contextual',appearance='correct',smoke='smoke_contextual',stage='ready')]
    os.kill(a.existing_pid,0)
    handles={};logs={}
    state=dict(stage='parallel_probe',supervisor_pid=os.getpid(),arms=arms,selection='epoch10_last',evaluation_precision='FP32')
    def save():
        tmp=a.root/'run_state.json.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(a.root/'run_state.json')
    def config(arm):
        return ['--route',arm['route'],'--source-mode',arm['source'],'--appearance',arm['appearance']]
    def start(arm,stage,command):
        label=arm['name']+'_'+stage
        path=a.root/f'{label}.log';logs[arm['name']]=path.open('w')
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(arm['gpu']),PYTHONUNBUFFERED='1')
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=logs[arm['name']],stderr=subprocess.STDOUT)
        handles[arm['name']]=proc
        arm.update(stage=stage,pid=proc.pid,external=False,log=str(path))
        save();print('START',label,proc.pid,flush=True)
    def ev(arm,mode,control=False):
        out=a.root/arm['name']/('control_evaluation' if control else 'evaluation')
        cp=a.root/arm['name']/('control/last.pth' if control else 'last.pth')
        return ['bash',wrapper,'adapter','eval','--mode',mode,'--adapter',str(cp),'--out',str(out)]
    def compare(arm,base,candidate,filename):
        return [PYTHON,analysis,'--baseline',str(base),'--candidate',str(candidate),'--output',str(a.root/arm['name']/filename)]
    try:
        for arm in arms[1:]:
            report=json.loads((a.root/arm['smoke']/'smoke.json').read_text())
            assert all(report[k] for k in ('zero_init_replay_exact','no_candidate_replay_exact','satellite_state_unchanged','lora_gradient_nonzero'))
            start(arm,'training',['bash',wrapper,'adapter','train',*config(arm),'--out',str(a.root/arm['name'])])
        save()
        while any(arm['stage'] not in ('done','failed') for arm in arms):
            for arm in arms:
                stage=arm['stage'];name=arm['name'];out=a.root/name
                if stage in ('done','failed'):continue
                if arm['external']:
                    try:os.kill(arm['pid'],0);continue
                    except ProcessLookupError:rc=None
                else:
                    rc=handles[name].poll()
                    if rc is None:continue
                    logs[name].close()
                    if rc:
                        arm.update(stage='failed',returncode=rc);save();continue
                print('END',name,stage,rc,flush=True)
                if stage=='training':
                    if not (out/'last.pth').is_file() or not (out/'parameter_changes.json').is_file():
                        arm.update(stage='failed',error='Training owner exited without required final artifacts');save();continue
                    start(arm,'eval_correct',ev(arm,'correct'))
                elif stage=='eval_correct':
                    if arm['appearance']=='constant':
                        start(arm,'analysis',compare(arm,baseline,out/'evaluation/correct/records.json','paired_analysis.json'))
                    else:start(arm,'eval_constant',ev(arm,'constant'))
                elif stage=='eval_constant':start(arm,'eval_donor',ev(arm,'donor'))
                elif stage=='eval_donor':start(arm,'analysis',compare(arm,baseline,out/'evaluation/correct/records.json','paired_analysis.json'))
                elif stage=='analysis':
                    result=json.loads((out/'paired_analysis.json').read_text())
                    arm['result']=result['metrics'];arm['gate']=result['gate']
                    if name in ('early','contextual') and result['gate']['direction_positive']:
                        command=['bash',wrapper,'adapter','train','--route',arm['route'],'--source-mode',arm['source'],
                                 '--appearance','constant','--out',str(out/'control')]
                        start(arm,'control_training',command)
                    else:arm['stage']='done';save()
                elif stage=='control_training':start(arm,'control_eval',ev(arm,'correct',True))
                elif stage=='control_eval':
                    start(arm,'control_analysis',compare(arm,out/'control_evaluation/correct/records.json',out/'evaluation/correct/records.json','appearance_paired_analysis.json'))
                elif stage=='control_analysis':
                    arm['appearance_result']=json.loads((out/'appearance_paired_analysis.json').read_text())
                    arm['stage']='done';save()
            time.sleep(30)
        if any(arm['stage']=='failed' for arm in arms):
            state['stage']='partial_failure';save();return
        # The geometry control is already independently trained in parallel for the original highres/fine route.
        arm=arms[0]
        start(arm,'appearance_analysis',compare(arm,a.root/'geometry/evaluation/correct/records.json',a.root/'correct/evaluation/correct/records.json','appearance_paired_analysis.json'))
        rc=handles['correct'].wait();logs['correct'].close()
        if rc:raise RuntimeError(f'appearance analysis rc={rc}')
        arm['stage']='done';state['stage']='analysis_complete';state['remaining']=['root audit and complete evidence report'];save()
    except BaseException as exc:
        state.update(stage='failed',error=str(exc));save();raise


if __name__=='__main__':main()
