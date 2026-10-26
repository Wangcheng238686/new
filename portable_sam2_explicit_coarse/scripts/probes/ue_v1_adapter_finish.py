"""Finish the authorized fixed-budget probe after its verified training owner exits."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
PYTHON='/data/wangcheng/envs/cvt2/bin/python'


def main():
    p=argparse.ArgumentParser();p.add_argument('--train-pid',type=int,required=True)
    p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    state=json.loads((a.root/'run_state.json').read_text())
    state['supervisor_pid']=os.getpid()
    def save(**extra):
        state.update(extra);tmp=a.root/'run_state.json.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(a.root/'run_state.json')
    def run(label,args):
        save(stage=label)
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1')
        with (a.root/f'{label}.log').open('w') as log:
            child=subprocess.Popen(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            save(child_pid=child.pid,child_log=str(a.root/f'{label}.log'))
            rc=child.wait()
        if rc:raise RuntimeError(f'{label} failed rc={rc}')
    try:
        os.kill(a.train_pid,0)  # must be a confirmed live owner when supervision starts
        save(stage='training')
        while True:
            try:os.kill(a.train_pid,0)
            except ProcessLookupError:break
            time.sleep(45)
        cp=a.root/'correct/last.pth'
        if not cp.is_file():raise RuntimeError('Training owner ended without final checkpoint')
        wrapper=str(ROOT/'scripts/ue_rgbd.sh')
        for mode in ('correct','constant','donor'):
            run(f'eval_{mode}',['bash',wrapper,'adapter','eval','--mode',mode,'--adapter',str(cp),'--out',str(a.root/'evaluation')])
        base=a.root/'evaluation/disabled/records.json'
        real=a.root/'evaluation/correct/records.json'
        analysis=str(ROOT/'scripts/probes/ue_v1_adapter_analysis.py')
        run('analysis',[PYTHON,analysis,'--baseline',str(base),'--candidate',str(real),'--output',str(a.root/'paired_analysis.json')])
        result=json.loads((a.root/'paired_analysis.json').read_text())
        if result['gate']['direction_positive']:
            run('train_constant',['bash',wrapper,'adapter','train','--appearance','constant','--out',str(a.root/'trained_constant')])
            run('eval_trained_constant',['bash',wrapper,'adapter','eval','--appearance','constant','--mode','correct','--adapter',str(a.root/'trained_constant/last.pth'),'--out',str(a.root/'constant_evaluation')])
            run('analysis_trained_constant',[PYTHON,analysis,'--baseline',str(a.root/'constant_evaluation/correct/records.json'),'--candidate',str(real),'--output',str(a.root/'appearance_paired_analysis.json')])
        save(stage='analysis_complete',remaining=['root evidence review and final report'])
    except BaseException as exc:
        save(stage='failed',error=str(exc));raise


if __name__=='__main__':main()
