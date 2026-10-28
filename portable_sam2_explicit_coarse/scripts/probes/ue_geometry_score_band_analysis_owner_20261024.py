"""Bounded paired analysis after fixed score-band val predictions complete."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/data/wangcheng/checkpoint/ue_geometry_score_band_20261024')
BASE=Path('/data/wangcheng/checkpoint/ue_rpn_geometry_20261023/val/budget/records.json')


def main():
    out=ROOT/'analysis';out.mkdir(exist_ok=True)
    running={};done=[]
    comparisons=[('geometry_sat',BASE,ROOT/'geometry/val/records.json'),
        ('geometry_constant',ROOT/'constant/val/records.json',ROOT/'geometry/val/records.json'),
        ('geometry_satellite_supplement',ROOT/'satellite/val/records.json',ROOT/'geometry/val/records.json'),
        ('constant_sat',BASE,ROOT/'constant/val/records.json'),
        ('satellite_supplement_sat',BASE,ROOT/'satellite/val/records.json')]
    jobs=json.loads((ROOT/'launch.json').read_text())
    while len(done)<len(comparisons):
        for key,(p,log) in list(running.items()):
            rc=p.poll()
            if rc is not None:
                log.close();assert rc==0,(key,rc)
                done.append(key);del running[key]
        for key,baseline,candidate in comparisons:
            if len(running)>=2:break
            if key in done or key in running:continue
            required=[baseline,candidate]
            ready=all(p.exists() and (p==BASE or p.with_name('identity.json').exists()) for p in required)
            if not ready:continue
            log=open(out/f'{key}.log','a')
            env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
            argv=['/data/wangcheng/envs/cvt2/bin/python','-u','scripts/probes/ue_joint_encoding_fast_analysis.py',
                '--baseline',str(baseline),'--candidate',str(candidate),'--output',str(out/f'{key}.json'),'--split','val','--bootstrap','500']
            p=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env)
            running[key]=(p,log)
        for job in jobs:
            if not (ROOT/job['content']/'val/identity.json').exists():
                try:os.kill(job['pid'],0)
                except ProcessLookupError:raise RuntimeError(f"Evaluation exited without identity: {job}")
        (out/'state.json').write_text(json.dumps(dict(done=done,running={k:p.pid for k,(p,_) in running.items()},pid=os.getpid()),indent=2))
        if len(done)<len(comparisons):time.sleep(10)
    (out/'complete.json').write_text(json.dumps(dict(complete=True,comparisons=done)))


if __name__=='__main__':main()
