"""Complete the unconditional contextual RGB control after the test-primary amendment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
PYTHON='/data/wangcheng/envs/cvt2/bin/python'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    a=parser.parse_args();r=a.root;out=r/'contextual/control'
    assert not out.exists(),'Refuse replacing an existing control'
    state=dict(stage='starting',owner_pid=os.getpid(),gpu=3,
               reason='User changed the primary endpoint to test; all RGB routes need matched controls before test results are read')
    def save():
        p=r/'contextual_control_state.json';tmp=p.with_suffix('.tmp')
        tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(p)
    def run(stage,cmd):
        path=r/f'contextual_unconditional_{stage}.log'
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='3',PYTHONUNBUFFERED='1')
        with path.open('w') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state.update(stage=stage,pid=child.pid,log=str(path));save()
            print('START',stage,child.pid,flush=True)
            while child.poll() is None:time.sleep(20)
            if child.returncode:raise RuntimeError(f'{stage}: exit {child.returncode}')
    wrapper=str(ROOT/'scripts/ue_rgbd.sh');analysis=str(ROOT/'scripts/probes/ue_v1_adapter_analysis.py')
    try:
        run('smoke',['bash',wrapper,'adapter','smoke','--source-mode','contextual','--appearance','constant',
                     '--out',str(r/'smoke_contextual_constant')])
        smoke=json.loads((r/'smoke_contextual_constant/smoke.json').read_text())
        assert all(smoke[k] for k in ('zero_init_logits_exact','zero_init_replay_exact',
                   'no_candidate_replay_exact','satellite_state_unchanged','lora_gradient_nonzero'))
        run('train',['bash',wrapper,'adapter','train','--source-mode','contextual','--appearance','constant',
                     '--reseed-from-epoch','5','--out',str(out)])
        run('eval',['bash',wrapper,'adapter','eval','--adapter',str(out/'last.pth'),
                    '--out',str(r/'contextual/control_evaluation')])
        run('analysis',[PYTHON,analysis,'--baseline',str(r/'contextual/control_evaluation/correct/records.json'),
                        '--candidate',str(r/'contextual/evaluation/correct/records.json'),
                        '--output',str(r/'contextual/appearance_paired_analysis.json')])
        state['stage']='analysis_complete';save()
    except BaseException as exc:
        state.update(stage='failed',error=str(exc));save();raise


if __name__=='__main__':main()
