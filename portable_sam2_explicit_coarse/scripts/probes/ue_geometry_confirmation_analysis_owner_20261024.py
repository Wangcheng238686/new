"""Fixed five test comparisons, triggered only by completed frozen evaluations."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

DIRECT=Path('/data/wangcheng/checkpoint/ue_geometry_branch_test_20261024')
COND=Path('/data/wangcheng/checkpoint/ue_geometry_conditional_online_20261024')
BASE=Path('/data/wangcheng/checkpoint/ue_rpn_geometry_20261023/test/budget/records.json')
OUT=Path('/data/wangcheng/checkpoint/ue_geometry_confirmation_analysis_20261024')


def main():
    OUT.mkdir(exist_ok=True)
    dg=DIRECT/'geometry/test/records.json';dk=DIRECT/'constant/test/records.json'
    cg=COND/'geometry/test/combined_records.json';ck=COND/'constant/test/combined_records.json';prefix=COND/'geometry/test/prefix_records.json'
    jobs=[('direct_net',BASE,dg),('direct_content',dk,dg),('strongSAT_net',BASE,prefix),
          ('conditional_net',prefix,cg),('conditional_content',ck,cg)]
    done=[];running={};prefix_audited=False
    while len(done)<len(jobs):
        for key,(proc,log) in list(running.items()):
            rc=proc.poll()
            if rc is not None:
                log.close();assert rc==0,(key,rc)
                done.append(key);del running[key]
        for key,b,c in jobs:
            if key in done or key in running:continue
            ready=all(p.exists() and (p==BASE or p.with_name('identity.json').exists()) for p in [b,c])
            if not ready:continue
            log=open(OUT/f'{key}.log','a');env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
            argv=['/data/wangcheng/envs/cvt2/bin/python','-u','scripts/probes/ue_joint_encoding_fast_analysis.py',
                '--baseline',str(b),'--candidate',str(c),'--output',str(OUT/f'{key}.json'),'--split','test','--bootstrap','500']
            proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env);running[key]=(proc,log)
        left=COND/'geometry/test/prefix_records.json';right=COND/'constant/test/prefix_records.json'
        if not prefix_audited and all(p.with_name('identity.json').exists() for p in [left,right]):
            assert left.read_bytes()==right.read_bytes(),'G/K satellite prefixes differ'
            audit=dict(passed=True,prefix_records_exact=True,sha256=hashlib.sha256(left.read_bytes()).hexdigest())
            (OUT/'prefix_audit.json').write_text(json.dumps(audit,indent=2));prefix_audited=True
        for root in [DIRECT,COND]:
            for job in json.loads((root/'launch.json').read_text()):
                if not (root/job['content']/'test/identity.json').exists():
                    try:os.kill(job['pid'],0)
                    except ProcessLookupError:raise RuntimeError(f"Evaluation failed without result: {root} {job}")
        (OUT/'state.json').write_text(json.dumps(dict(done=done,running={k:p.pid for k,(p,_) in running.items()},pid=os.getpid(),prefix_audited=prefix_audited),indent=2))
        if len(done)<len(jobs):time.sleep(10)
    assert prefix_audited
    (OUT/'complete.json').write_text(json.dumps(dict(complete=True,comparisons=done)))


if __name__=='__main__':main()
