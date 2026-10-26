"""Detached owner for the normalized feature pair and final content comparison."""
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
    arms=[dict(name='unit',gpu=0,appearance='correct'),dict(name='unit_geometry',gpu=2,appearance='constant')]
    state=dict(stage='training',owner_pid=os.getpid(),arms=arms);handles={};logs={}
    def save():
        tmp=r/'unit_state.json.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(r/'unit_state.json')
    def spawn(arm,stage,command):
        path=r/f"{arm['name']}_{stage}.log";f=path.open('w')
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(arm['gpu']),PYTHONUNBUFFERED='1')
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        handles[arm['name']]=child;logs[arm['name']]=f;arm.update(stage=stage,pid=child.pid,log=str(path));save()
        print('START',arm['name'],stage,child.pid,flush=True)
    def evaluate(arm,mode):
        return ['bash',wrapper,'adapter','eval','--mode',mode,'--adapter',str(r/arm['name']/'last.pth'),'--out',str(r/arm['name']/'evaluation')]
    def compare(arm,base,filename):
        return [PYTHON,analysis,'--baseline',str(base),'--candidate',str(r/arm['name']/'evaluation/correct/records.json'),
                '--output',str(r/arm['name']/filename)]
    try:
        for arm in arms:
            smoke_name='smoke_'+arm['name'];s=json.loads((r/smoke_name/'smoke.json').read_text())
            assert all(s[k] for k in ('zero_init_logits_exact','zero_init_replay_exact','no_candidate_replay_exact','satellite_state_unchanged','lora_gradient_nonzero'))
            assert all(row['stats']['source_delta_rms']<=1.01 for row in s['steps'])
            spawn(arm,'train',['bash',wrapper,'adapter','train','--source-mode','unit','--appearance',arm['appearance'],
                 '--reseed-from-epoch','5','--out',str(r/arm['name'])])
        while any(arm['stage'] not in ('done','failed') for arm in arms):
            for arm in arms:
                name=arm['name'];stage=arm['stage']
                if stage in ('done','failed'):continue
                rc=handles[name].poll()
                if rc is None:continue
                logs[name].close()
                if rc:arm.update(stage='failed',returncode=rc);save();continue
                if stage=='train':spawn(arm,'eval_correct',evaluate(arm,'correct'))
                elif stage=='eval_correct':
                    if arm['appearance']=='correct':spawn(arm,'eval_constant',evaluate(arm,'constant'))
                    else:spawn(arm,'analysis',compare(arm,r/'evaluation/disabled/records.json','paired_analysis.json'))
                elif stage=='eval_constant':spawn(arm,'eval_donor',evaluate(arm,'donor'))
                elif stage=='eval_donor':spawn(arm,'analysis',compare(arm,r/'evaluation/disabled/records.json','paired_analysis.json'))
                elif stage=='analysis':arm['stage']='done';save()
            time.sleep(30)
        if any(arm['stage']=='failed' for arm in arms):state['stage']='partial_failure';save();return
        arm=arms[0];spawn(arm,'content_analysis',compare(arm,r/'unit_geometry/evaluation/correct/records.json','appearance_paired_analysis.json'))
        rc=handles['unit'].wait();logs['unit'].close()
        if rc:raise RuntimeError(f'unit content comparison rc={rc}')
        arm['stage']='done';state['stage']='analysis_complete';save()
    except BaseException as exc:state.update(stage='failed',error=str(exc));save();raise


if __name__=='__main__':main()
