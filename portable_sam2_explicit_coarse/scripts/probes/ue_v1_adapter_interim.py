"""Fixed epoch1/3 diagnostic validations; never selects/stops training checkpoints."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from probe_gpu_lease import acquire,release

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--family',choices=['rgbd','native','unit'],default='rgbd');a=p.parse_args();r=a.root
    arms={'rgbd':[('correct',0),('geometry',1),('early',2),('contextual',3)],'native':[('native_geometry',1),('native_zero',3)],'unit':[('unit',0),('unit_geometry',2)]}[a.family]
    state_name={'rgbd':'interim_state.json','native':'native_interim_state.json','unit':'unit_interim_state.json'}[a.family]
    base=json.loads((r/'evaluation/disabled/records.json').read_text())
    baseline=json.loads((r/'evaluation/disabled/metrics.json').read_text())
    tasks={(name,e):dict(stage='waiting',gpu=2,training_gpu=gpu) for name,gpu in arms for e in (1,3)}
    previous=r/state_name
    if previous.is_file():
        saved=json.loads(previous.read_text())
        for (name,epoch),task in tasks.items():
            old=saved.get(f'{name}_e{epoch}',{})
            if old.get('stage')=='complete':task.update(old)
            assert old.get('stage')!='running','Drain existing evaluations before restarting interim owner'
    lease=r/'interim_gpu2.lock'
    handles={};logs={}
    def save():
        path=r/state_name;tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps({f'{n}_e{e}':v for (n,e),v in tasks.items()},indent=2)+'\n');tmp.replace(path)
    while any(t['stage'] not in ('complete','failed') for t in tasks.values()):
        for (name,epoch),task in tasks.items():
            key=(name,epoch);out=r/name/f'interim_epoch{epoch}'
            if task['stage']=='waiting':
                if any(v['stage']=='running' and v['gpu']==task['gpu'] for v in tasks.values()):continue
                progress=r/name/'progress.json'
                if not progress.is_file() or json.loads(progress.read_text())['epoch']<epoch:continue
                if not acquire(lease):continue
                cp=r/name/f'epoch{epoch}.pth'
                assert cp.is_file()
                logfile=r/f'interim_{name}_e{epoch}.log';logs[key]=logfile.open('w')
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(task['gpu']),PYTHONUNBUFFERED='1')
                command=['bash',str(ROOT/'scripts/ue_rgbd.sh'),'nativegeo' if a.family=='native' else 'adapter','eval']
                if a.family!='native':command+=['--mode','correct']
                command+=['--adapter',str(cp),'--out',str(out)]
                proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=logs[key],stderr=subprocess.STDOUT)
                handles[key]=proc;task.update(stage='running',pid=proc.pid,log=str(logfile));save()
                print('START',name,epoch,proc.pid,flush=True)
            elif task['stage']=='running':
                rc=handles[key].poll()
                if rc is None:continue
                logs[key].close()
                release(lease)
                if rc:task.update(stage='failed',returncode=rc);save();continue
                rows=json.loads((out/'correct/records.json').read_text())
                assert len(rows)==len(base)==459
                for b,c in zip(base,rows):
                    assert b['image_id']==c['image_id'] and b['gt']==c['gt']
                    assert all(b['dt'][k]==c['dt'][k] for k in ('bboxes','scores','labels'))
                metrics=json.loads((out/'correct/metrics.json').read_text())
                task.update(stage='complete',metrics=metrics,mask_delta=metrics['segm/mAP']-baseline['segm/mAP'],
                            detector_gt_unchanged=True,scope='fixed-budget diagnostic, not checkpoint selection')
                print('RESULT',name,epoch,task['mask_delta'],flush=True);save()
        time.sleep(30)
    save()


if __name__=='__main__':main()
