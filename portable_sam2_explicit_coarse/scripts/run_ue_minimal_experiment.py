"""Single sequential owner: train on train/val, seal val-best, then evaluate test once."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True)
    p.add_argument('--run-root',required=True);args=p.parse_args()
    out=Path(args.run_root).resolve();out.mkdir(parents=True,exist_ok=False)
    state=dict(stage='preflight',pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),
        data_root=str(Path(args.data_root).resolve()),run_root=str(out),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    def save():
        tmp=out/'run_state.json.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(out/'run_state.json')
    def run(command,stage):
        state.update(stage=stage);save()
        log=out/f'{stage}_pid{os.getpid()}.log'
        with log.open('w') as f:
            proc=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env)
            state.update(child_pid=proc.pid,log=str(log));save();rc=proc.wait()
        if rc:raise RuntimeError(f'{stage} returncode={rc}; see {log}')
    env=os.environ.copy();env.update(NPROC_PER_NODE='4',CUDA_VISIBLE_DEVICES='0,1,2,3',UE_MINIMAL_SMOKE='0')
    try:
        dirty=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
        if dirty:raise ValueError('Full run requires clean committed code')
        state['git_dirty']=False;save()
        # Verify actual bytes of the immutable archive, not just its manifest.
        with (out/'archive_integrity.log').open('w') as f:
            subprocess.run(['sha256sum','--quiet','-c','SHA256SUMS.txt'],cwd=args.data_root,
                           stdout=f,stderr=subprocess.STDOUT,check=True)
        state['all_archive_checksums_verified']=True;save()
        wrapper=str(ROOT/'scripts/ue_rgbd.sh')
        run(['bash',wrapper,'minimal','train','--data-root',args.data_root,
             '--checkpoint-dir',str(out/'checkpoints')],'train')
        checkpoint=(out/'checkpoints/best_model.pth').resolve(strict=True)
        # best_model is maintained by validation segm AP only; never inspect test for selection.
        h=hashlib.sha256()
        with checkpoint.open('rb') as f:
            for block in iter(lambda:f.read(2**20),b''):h.update(block)
        state.update(selected_checkpoint=str(checkpoint),selected_sha256=h.hexdigest(),
            selection='highest_validation_segm_AP250',weights='model')
        (out/'SEALED_SATELLITE.json').write_text(json.dumps(state,indent=2)+'\n');save()
        run(['bash',wrapper,'minimal','infer','--checkpoint',str(checkpoint),'--weights','model',
            '--split','test','--data-root',args.data_root,'--batch-size','1','--device','cuda:0',
            '--score-thr','0','--export-bootstrap-records','--output-dir',str(out/'test')],'test')
        state.update(stage='complete',finished_at=datetime.now(timezone.utc).isoformat());save()
    except BaseException as exc:
        state.update(stage='failed',error=str(exc),finished_at=datetime.now(timezone.utc).isoformat());save()
        raise

if __name__=='__main__':main()
