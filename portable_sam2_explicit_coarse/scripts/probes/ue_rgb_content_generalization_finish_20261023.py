"""Run two fixed split audits sequentially and collect a descriptive comparison."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=Path('/data/wangcheng/checkpoint/ue_rgb_content_generalization_audit_20261023')
TRAIN=Path('/data/wangcheng/checkpoint/ue_rgb_content_direction_audit_20261023')
WORKER=ROOT/'scripts/probes/ue_rgb_content_generalization_audit_20261023.py'
STUDY=Path('/data/wangcheng/checkpoint/ue_s2db_probe_20261022')

def read(p):
    return json.loads(p.read_text())

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):
            h.update(block)
    return h.hexdigest()

def save(p,d):
    t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(d,indent=2)+'\n')
    t.replace(p)

def main():
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'owner.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    hashes={str(p):sha(p) for p in (Path(__file__).resolve(),WORKER)}
    checkpoints={a:sha(STUDY/a/'last.pth') for a in ('A','D')}
    plan=dict(source_hashes=hashes,checkpoint_sha256=checkpoints,
        arms=['A','D'],splits=['val','test'],density_quantiles=[.125,.375,.625,.875],
        selection='Stable annotation-count order, four fixed quantile indices per split, no result-based selection',
        modes=['correct','constant','donor'],optimizer_steps=0,
        scope='Descriptive downstream-loss direction; sampled GT-positive training-style proposals, not deployed AP',
        donor='Cyclic next RGB within same split; original target geometry retained',
        test_role='Development test already used for diagnostics; no held-out confirmation claim',
        train_reference=str(TRAIN),resource='Sequential split workers, shared GPU after measured gate and readmission')
    if (OUT/'plan.json').exists():
        assert read(OUT/'plan.json')==plan,'Existing plan differs'
    else:
        save(OUT/'plan.json',plan)
    env=dict(os.environ,PYTHONUNBUFFERED='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    for split in plan['splits']:
        dest=OUT/split
        if not (dest/'complete.json').exists():
            save(OUT/'state.json',dict(stage='running',split=split,owner_pid=os.getpid()))
            with (OUT/(split+'.log')).open('a') as log:
                rc=subprocess.call([sys.executable,str(WORKER),'--split',split,'--out',str(dest)],
                    cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            if rc:
                save(OUT/'state.json',dict(stage='failed',split=split,returncode=rc,owner_pid=os.getpid()))
                raise RuntimeError((split,rc))
        assert read(dest/'complete.json')['complete']
    rows=[];artifacts={}
    for split,base in [('train',TRAIN),('val',OUT/'val'),('test',OUT/'test')]:
        for arm in plan['arms']:
            p=base/(arm+'.json'); d=read(p);artifacts[str(p)]=sha(p)
            assert d['complete'] and d['satellite_unchanged'] and d['adapter_unchanged']
            assert d['checkpoint_sha256']==checkpoints[arm] and len(d['records'])==4
            for mode in ('constant','donor'):
                cs=[r['contrasts'][mode] for r in d['records']]
                losses={}
                for name in ('loss_mask','loss_decoder_tail'):
                    xs=[c['losses'][name]['actual_loss_correct_minus_intervention'] for c in cs]
                    losses[name]=dict(mean_delta=sum(xs)/len(xs),favorable=sum(v<0 for v in xs),values=xs)
                rows.append(dict(split=split,arm=arm,intervention=mode,samples=4,losses=losses,
                    content_ratio=[c['content_difference_over_correct_total_rms'] for c in cs],
                    final_raw_ratio=[c['final_over_raw_logit_difference_rms'] for c in cs]))
    assert all(sha(Path(p))==h for p,h in hashes.items())
    save(OUT/'summary.json',rows)
    lines=['# 内容方向的跨划分诊断','',
        '每个划分固定4图，各A/D E10及三种RGB内容。无增强/dropout，FP32，无权重更新。',
        '不同划分的图不是配对样本；这里只作描述，不给跨划分显著性或总体泛化结论。',
        'loss来自固定GT正proposal诊断，不是部署AP。负差表示correct的实际加权loss更低。','',
        '|划分|臂|对照|mask平均差|mask有利图数|tail平均差|tail有利图数|',
        '|---|---|---|---:|---:|---:|---:|']
    for r in rows:
        m=r['losses']['loss_mask'];t=r['losses']['loss_decoder_tail']
        lines.append(f"|{r['split']}|{r['arm']}|{r['intervention']}|{m['mean_delta']:+.7f}|{m['favorable']}/4|{t['mean_delta']:+.7f}|{t['favorable']}/4|")
    lines.extend(['','原生UDPR选点随内容改变；raw/final差异不等同于固定点的UDPR因果效应。',
        'donor仅替换RGB，输入与目标几何不一致。小样本loss方向不能代替完整AP、匹配训练内容对照或多seed验证。'])
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    save(OUT/'complete.json',dict(complete=True,source_unchanged=True,optimizer_steps=0,
        plan_sha256=sha(OUT/'plan.json'),artifacts_sha256=artifacts))
    save(OUT/'state.json',dict(stage='complete',owner_pid=os.getpid()))

if __name__=='__main__':
    main()
