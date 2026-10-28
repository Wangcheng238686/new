"""Run the native geometry pair in parallel; serialize post-training evaluation on GPU1."""
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
    arms=[dict(name='native_geometry',content='geometry',gpu=1),dict(name='native_zero',content='zero',gpu=3)]
    state=dict(stage='training',owner_pid=os.getpid(),arms=arms);handles={};logs={}
    def save():
        tmp=r/'native_state.json.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(r/'native_state.json')
    def spawn(arm,stage,command,gpu=None):
        logfile=r/f"{arm['name']}_{stage}.log";f=logfile.open('w')
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(arm['gpu'] if gpu is None else gpu),PYTHONUNBUFFERED='1')
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        handles[arm['name']]=proc;logs[arm['name']]=f
        arm.update(stage=stage,pid=proc.pid,log=str(logfile));save();print('START',arm['name'],stage,proc.pid,flush=True)
        return proc
    def wait(arm):
        rc=handles[arm['name']].wait();logs[arm['name']].close()
        if rc:raise RuntimeError(f"{arm['name']} {arm['stage']} rc={rc}")
    try:
        for arm in arms:
            report=json.loads((r/f"smoke_{arm['name']}"/'smoke.json').read_text())
            assert all(report[k] for k in ('zero_init_exact','no_support_exact','satellite_unchanged','mixer_gradient_nonzero'))
            spawn(arm,'train',['bash',wrapper,'nativegeo','train','--content',arm['content'],'--out',str(r/arm['name'])])
        while any(arm['stage']=='train' for arm in arms):
            for arm in arms:
                if arm['stage']!='train':continue
                rc=handles[arm['name']].poll()
                if rc is None:continue
                logs[arm['name']].close()
                if rc:raise RuntimeError(f"{arm['name']} train rc={rc}")
                assert (r/arm['name']/'last.pth').is_file()
                arm['stage']='trained';save();print('TRAINED',arm['name'],flush=True)
            time.sleep(30)
        state['stage']='evaluation';save()
        for arm in arms:
            spawn(arm,'eval',['bash',wrapper,'nativegeo','eval','--adapter',str(r/arm['name']/'last.pth'),
                             '--out',str(r/arm['name']/'evaluation')],gpu=1);wait(arm)
            arm['stage']='evaluated';save()
        state['stage']='analysis';save()
        for arm in arms:
            spawn(arm,'analysis',[PYTHON,analysis,'--baseline',str(r/'evaluation/disabled/records.json'),
                '--candidate',str(r/arm['name']/'evaluation/correct/records.json'),'--output',str(r/arm['name']/'paired_analysis.json')],gpu=1)
        for arm in arms:wait(arm);arm['stage']='analyzed';save()
        arm=arms[0]
        spawn(arm,'content_analysis',[PYTHON,analysis,'--baseline',str(r/'native_zero/evaluation/correct/records.json'),
            '--candidate',str(r/'native_geometry/evaluation/correct/records.json'),'--output',str(r/'native_geometry/content_paired_analysis.json')],gpu=1)
        wait(arm);arm['stage']='analyzed'
        state['stage']='analysis_complete';save()
    except BaseException as exc:
        state.update(stage='failed',error=str(exc));save();raise


if __name__=='__main__':main()
