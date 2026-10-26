"""Locked seed45 training -> original-prediction gates -> test -> two analyses."""
import json
import os
from pathlib import Path
import subprocess
import time

import torch

TRAIN=Path('/data/wangcheng/checkpoint/ue_geometry_proposal_repeat_seed45_20261024')
OUT=Path('/data/wangcheng/checkpoint/ue_geometry_conditional_repeat_seed45_20261024')
REFERENCE=Path('/data/wangcheng/checkpoint/ue_geometry_conditional_online_20261024/geometry/test/prefix_records.json')


def main():
    OUT.mkdir(exist_ok=True)
    regression=Path('/data/wangcheng/checkpoint/ue_geometry_configured_regression_gate_20261024/geometry/gate/identity.json')
    assert json.loads(regression.read_text())['online_cached_equivalence_gate']
    initial=torch.load(TRAIN/'common_init.pth',map_location='cpu',weights_only=False)['fingerprint']
    old=torch.load('/data/wangcheng/checkpoint/ue_geometry_proposal_pilot_20261024/common_init.pth',map_location='cpu',weights_only=False)['fingerprint']
    assert initial!=old,'Repeat initialization must differ from seed44'
    training=json.loads((TRAIN/'launch.json').read_text())
    while True:
        ready=True
        for arm in ['geometry','constant']:
            p=TRAIN/arm/'last.pth'
            if not p.exists():ready=False;continue
            cp=torch.load(p,map_location='cpu',weights_only=False)
            if cp['epoch']!=10:ready=False;continue
            assert cp['step']==2670 and cp['initial_fingerprint']==initial and cp['protocol']['seed']==45
        if ready:break
        for job in training:
            p=TRAIN/job['content']/'progress.json'
            if p.exists() and json.loads(p.read_text())['epoch']==10:continue
            try:os.kill(job['pid'],0)
            except ProcessLookupError:raise RuntimeError(f'Training ended early: {job}')
        time.sleep(10)
    for epoch in range(1,11):
        a,b=[json.loads((TRAIN/arm/f'epoch{epoch}.json').read_text()) for arm in ['geometry','constant']]
        assert a['sample_order']==b['sample_order'] and len(a['sample_order'])==2136
        assert a['step']==b['step']==epoch*267 and a['epoch_seed']==b['epoch_seed']==1045000+epoch
    (OUT/'training_audit.json').write_text(json.dumps(dict(passed=True,epochs=10,updates=2670,initial_fingerprint=initial,
        different_from_seed44=True,paired_sample_order_exact=True,only_changes='initialization46, data45, epoch1045000+e'),indent=2))
    workers={}
    for arm in ['geometry','constant']:
        argv=['/data/wangcheng/envs/cvt2/bin/python','-u','scripts/probes/ue_geometry_conditional_configured_20261024.py',
              '--content',arm,'--pilot-root',str(TRAIN),'--out-root',str(OUT)]
        code='import subprocess\nsubprocess.run('+repr(argv+['--gate'])+',check=True)\nsubprocess.run('+repr(argv)+',check=True)\n'
        log=open(OUT/f'{arm}.log','a');env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
        proc=subprocess.Popen(['/data/wangcheng/envs/cvt2/bin/python','-u','-c',code],stdout=log,stderr=subprocess.STDOUT,env=env)
        workers[arm]=(proc,log)
    (OUT/'launch.json').write_text(json.dumps([{ 'content':a,'pid':p.pid,'gpu':0} for a,(p,_) in workers.items()],indent=2))
    while not all((OUT/arm/'test/identity.json').exists() for arm in workers):
        for arm,(p,_) in workers.items():
            if (OUT/arm/'test/identity.json').exists():continue
            assert p.poll() is None or p.returncode==0,(arm,p.returncode)
            if p.poll()==0:assert (OUT/arm/'test/identity.json').exists(),arm
        time.sleep(10)
    for arm,(p,log) in workers.items():
        assert p.wait()==0;log.close()
        assert (OUT/arm/'test/prefix_records.json').read_bytes()==REFERENCE.read_bytes(),(arm,'strongSAT prefix changed across seeds')
    (OUT/'prefix_audit.json').write_text(json.dumps(dict(passed=True,exact_across_content_and_seed=True)))
    analysis=OUT/'analysis';analysis.mkdir(exist_ok=True)
    comparisons=[('net',OUT/'geometry/test/prefix_records.json',OUT/'geometry/test/combined_records.json'),
                 ('content',OUT/'constant/test/combined_records.json',OUT/'geometry/test/combined_records.json')]
    processes=[]
    for key,b,c in comparisons:
        log=open(analysis/f'{key}.log','a');env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        argv=['/data/wangcheng/envs/cvt2/bin/python','-u','scripts/probes/ue_joint_encoding_fast_analysis.py','--baseline',str(b),'--candidate',str(c),'--output',str(analysis/f'{key}.json'),'--split','test','--bootstrap','500']
        p=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env);processes.append((key,p,log))
    (OUT/'analysis_launch.json').write_text(json.dumps([{ 'comparison':k,'pid':p.pid} for k,p,_ in processes]))
    for key,p,log in processes:
        assert p.wait()==0,key;log.close()
    (OUT/'complete.json').write_text(json.dumps(dict(complete=True,seed=45,paired_training=True,test459=True,comparisons=['net','content'])))


if __name__=='__main__':main()
