"""Predeclared L0--L6 content-paired study; three non-preemptive GPU slots.

Only GPUs 1, 2, 3 are eligible. Their first jobs are respectively L1, L2,
L3 correct training. No other GPU job may precede that GPU's first launch.
External owners are never interrupted; a lost exit status requires audit.
"""
import argparse
import fcntl
import math
import os
from pathlib import Path
import subprocess
import time

from ue_geometry_mechanism_schedule import read, sha, dump, proc_identity, gpu_snapshot

ROOT = Path(__file__).resolve().parents[2]
PYTHON = '/data/wangcheng/envs/cvt2/bin/python'
BASE_SHA = 'dd9106f4afe8ba72936316d7efa9621ca0c05d24836fa28c72ea322bed84ace1'
FIRST = {1: 'train_L1_correct', 2: 'train_L2_correct', 3: 'train_L3_correct'}


def arms():
    return [dict(name=f'L{i}_{c}', direction=f'L{i}', content=c)
            for i in range(7) for c in ('correct', 'constant')]


def sources():
    names = ['scripts/probes/ue_attention_ladder_probe.py',
             'scripts/probes/ue_attention_ladder_schedule.py',
             'scripts/probes/ue_attention_ladder_collect.py',
             'scripts/probes/ue_attention_ladder_cpu_gate.py',
             'scripts/probes/ue_geometry_mechanism_schedule.py',
             'scripts/probes/ue_v1_adapter.py', 'scripts/probes/ue_v1_adapter_analysis.py',
             'rsprompter/uav/attention_ladder.py', 'rsprompter/uav/s2db_geometry.py',
             'rsprompter/uav/local_adapter_probe.py', 'rsprompter/uav/local_adapter_variants.py',
             'data/ue_minimal_dataset.py', 'data/whu_instance_dataset.py',
             'utils/coco_eval_utils.py', 'inference/model_rebuild.py', 'inference/eval_records.py',
             'train/gt_samples.py', 'data/loader.py', 'rsprompter/ckpt_utils.py',
             'rsprompter/architecture_contract.py']
    for package in ('backbone', 'cspm', 'decoder', 'detection'):
        names += [str(p.relative_to(ROOT)) for p in sorted((ROOT / 'rsprompter' / package).glob('*.py'))]
    return list(dict.fromkeys(names))


def build_jobs(r, manifest):
    jobs, evaluations, comparisons = {}, [], []
    probe = [PYTHON, str(ROOT / 'scripts/probes/ue_attention_ladder_probe.py')]
    def add(key, cmd, artifact, deps=(), direction=None, priority=20):
        jobs[key] = dict(key=key, cmd=cmd, artifact=str(artifact), dependencies=list(deps),
                         gpu_required=direction is not None, direction=direction,
                         priority=priority, status='pending')
    def compare(key, base, candidate, deps, kind):
        out = r / 'analysis' / (key + '.json')
        cmd = [PYTHON, str(ROOT / 'scripts/probes/ue_v1_adapter_analysis.py'),
               '--baseline', str(base), '--candidate', str(candidate), '--output', str(out),
               '--split', 'test', '--bootstrap', '500']
        add(key, cmd, out, deps, priority=30)
        comparisons.append(dict(key=key, baseline=str(base), candidate=str(candidate), kind=kind, output=str(out)))
    init = ['--init', str(r / 'common_init.pth')]
    for direction in ('L0','L4','L5','L6'):
        key='gate_'+direction
        add(key, probe+['gate','--direction',direction,'--content','correct']+init+
            ['--out',str(r/'gates'/direction)],r/'gates'/direction/'gpu_gate.json',
            direction=direction,priority=2 if direction=='L0' else 4+int(direction[1:]))
        jobs[key]['resource_measurement_job']=True
    base = r / 'test/baseline/disabled/records.json'
    add('baseline_test', probe + ['eval', '--direction', 'L0', '--content', 'correct'] + init +
        ['--mode', 'disabled', '--eval-split', 'test', '--out', str(r / 'test/baseline')],
        r / 'test/baseline/disabled/evaluation_identity.json', direction='L0', priority=15)
    for arm in manifest['arms']:
        name, d, content = arm['name'], arm['direction'], arm['content']
        common = ['--direction', d, '--content', content] + init
        # First correct jobs are pinned in the admission loop, not merely prioritized.
        priority = 0 if d in ('L1','L2','L3') and content == 'correct' else (
            1 if d in ('L1','L2','L3') else 3 if d == 'L0' else 4 + int(d[1:]))
        train = 'train_' + name
        add(train, probe + ['train'] + common + ['--out', str(r / name)],
            r / name / 'last.pth', direction=d, priority=priority)
        primary = 'correct' if content == 'correct' else 'constant'
        for split, modes in [('val', [primary]), ('test', ['correct','constant','donor'] if content == 'correct' else ['constant'])]:
            for mode in modes:
                key = f'eval_{split}_{name}_{mode}'
                folder = r / split / name
                add(key, probe + ['eval'] + common + ['--adapter', str(r / name / 'last.pth'),
                    '--mode', mode, '--eval-split', split, '--out', str(folder)],
                    folder / mode / 'evaluation_identity.json', [train], d, 18 if split == 'test' else 22)
                evaluations.append(dict(key=key, arm=name, split=split, mode=mode, folder=str(folder / mode)))
        compare('baseline_' + name, base, r / 'test' / name / primary / 'records.json',
                ['baseline_test', f'eval_test_{name}_{primary}'], 'net_gain')
        if content == 'correct':
            for mode in ('constant','donor'):
                compare(f'dependency_{name}_{mode}', r / 'test' / name / mode / 'records.json',
                    r / 'test' / name / 'correct/records.json',
                    [f'eval_test_{name}_{mode}',f'eval_test_{name}_correct'], 'fixed_checkpoint_dependency')
    for i in range(7):
        d = f'L{i}'
        compare('content_' + d, r / 'test' / (d+'_constant') / 'constant/records.json',
            r / 'test' / (d+'_correct') / 'correct/records.json',
            [f'eval_test_{d}_constant_constant', f'eval_test_{d}_correct_correct'], 'matched_training_content')
        if i:
            compare('direction_' + d, r / 'test/L0_correct/correct/records.json',
                r / 'test' / (d+'_correct') / 'correct/records.json',
                ['eval_test_L0_correct_correct', f'eval_test_{d}_correct_correct'], 'direction_vs_common_baseline')
    prefinal = list(jobs)
    add('collect', [PYTHON, str(ROOT / 'scripts/probes/ue_attention_ladder_collect.py'), '--root', str(r)],
        r / 'complete_audit_summary.json', prefinal, priority=100)
    assert len(evaluations) == 42 and len(comparisons) == 41
    return jobs, evaluations, comparisons


def validated_gate(r, direction, manifest):
    """Resource registry may grow; each admitted job snapshots its exact gate."""
    path = r / 'resource_gate.json'
    if not path.exists(): return None
    registry = read(path)
    assert registry['init_sha256'] == manifest['init_sha256']
    item = registry.get('configurations', {}).get(direction)
    if not item or not item.get('all_passed'): return None
    assert item['peak_reserved_mib'] > 0 and item['fp32_resource_rois'] == 250
    actual = read(item['gate_path'])
    assert sha(item['gate_path']) == item['gate_sha256']
    contract=read(Path(item['gate_path']).parent/'gate_contract.json')
    assert contract['direction']==direction and contract['content']=='correct'
    assert contract['init_file_sha256']==manifest['init_sha256']
    assert all(manifest['source_hashes'][n] == h for n,h in actual['source_hashes'].items())
    return dict(registry_sha256=sha(path), configuration=item, actual=actual)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--register-only', action='store_true'); args = p.parse_args()
    r = args.root.resolve(); r.mkdir(parents=True, exist_ok=True)
    lock = (r/'owner.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    hashes = {n:sha(ROOT/n) for n in sources()}
    mp = r/'manifest.json'
    if mp.exists():
        manifest = read(mp)
        assert manifest['source_hashes'] == hashes, 'Registered source changed'
        assert manifest['init_sha256'] == sha(r/'common_init.pth')
    else:
        data = Path('/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021')
        manifest = dict(study='attention_ladder_L0_L6_content_pairs', arms=arms(), source_hashes=hashes,
            base_sha256=BASE_SHA, init_sha256=sha(r/'common_init.pth'), primary_split='test',
            selection='epoch10 last', epochs=10, updates=2670, no_ema=True, no_test_selection=True,
            initialization_seed=45, data_seed=44, epoch_seed_formula='1044000+epoch', bootstrap=500, bootstrap_seed=20261021,
            split_sha256={s:sha(data/'splits'/(s+'.json')) for s in ('train','val','test')},
            cpu_gate_sha256=sha(r/'cpu_gate.json'),
            cache_audit_identity_sha256=sha(r/'existing_cache_audit_identity.json'),
            allowed_gpus=[1,2,3], first_jobs={str(k):v for k,v in FIRST.items()},
            maximum_owned_gpu_jobs_per_gpu=1, maximum_cpu_jobs=2,
            registered_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()))
        assert read(r/'cpu_gate.json')['status']=='passed'
        _, evaluations, comparisons = build_jobs(r, manifest)
        manifest.update(evaluations=evaluations,comparisons=comparisons,test_evaluations=29,validation_evaluations=14)
        dump(mp,manifest)
        for n in hashes:
            target=r/'frozen_source'/n; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes((ROOT/n).read_bytes())
    expected,_,_=build_jobs(r,manifest)
    if args.register_only:
        dump(r/'job_plan.json',expected); print('Registered 14 training arms, 29 test and 14 val evaluations, 41 comparisons.'); return
    sp=r/'state.json'; state=read(sp) if sp.exists() else dict(stage='registered',jobs=expected)
    assert {k:v['cmd'] for k,v in state['jobs'].items()} == {k:v['cmd'] for k,v in expected.items()}
    state.update(owner_pid=os.getpid(),owner_identity=proc_identity(os.getpid()))
    children={}; streams={}
    for j in state['jobs'].values():
        if j['status']=='running' and proc_identity(j['pid'])!=j['process_identity']:
            j.update(status='failed',failure='Child exit status unavailable; manual audit required, no automatic retry')
    while True:
        for key,child in list(children.items()):
            rc=child.poll()
            if rc is None: continue
            streams.pop(key).close(); del children[key]; j=state['jobs'][key]
            j.update(returncode=rc,status='complete' if rc==0 and Path(j['artifact']).is_file() else 'failed')
            if j['status']=='complete' and j.get('resource_measurement_job'):
                try:
                    measured=read(j['artifact']);contract=read(Path(j['artifact']).parent/'gate_contract.json')
                    assert contract['direction']==j['direction'] and contract['content']=='correct'
                    assert contract['init_file_sha256']==manifest['init_sha256']
                    assert measured['resource_rois']==250 and measured['resource_chunk']==32
                    assert measured['satellite_parameters_buffers_exact'] and measured['satellite_no_grad']
                    assert all(manifest['source_hashes'][n]==h for n,h in measured['source_hashes'].items())
                    registry=read(r/'resource_gate.json')
                    registry['configurations'][j['direction']]=dict(all_passed=True,fp32_resource_rois=250,
                        peak_reserved_mib=max(measured['training_amp_peak']['reserved_MiB'],measured['fp32_250roi_peak']['reserved_MiB']),
                        gate_path=j['artifact'],gate_sha256=sha(j['artifact']))
                    dump(r/'resource_gate.json',registry)
                except Exception as error:
                    j.update(status='failed',failure='Gate artifact audit failed: '+repr(error))
            print('END',key,rc,j['status'],flush=True)
        for key,j in state['jobs'].items():
            if j['status']=='running' and key not in children and proc_identity(j['pid'])!=j['process_identity']:
                j.update(status='failed',failure='Adopted child exit status unavailable; manual audit required')
        failed=[k for k,j in state['jobs'].items() if j['status']=='failed']
        if failed:
            state.update(stage='failure_requires_audit',failed_jobs=failed);dump(sp,state)
            if not any(j['status']=='running' for j in state['jobs'].values()): return
            time.sleep(15); continue
        if all(j['status']=='complete' for j in state['jobs'].values()):
            state['stage']='complete';dump(sp,state);return
        assert all(sha(ROOT/n)==h for n,h in manifest['source_hashes'].items()),'Frozen source changed'
        running=[j for j in state['jobs'].values() if j['status']=='running']
        busy={j['gpu'] for j in running if j['gpu_required']};cpus=sum(not j['gpu_required'] for j in running)
        ready=[j for j in state['jobs'].values() if j['status']=='pending' and all(state['jobs'][d]['status']=='complete' for d in j['dependencies'])]
        ready.sort(key=lambda j:(j['priority'],list(state['jobs']).index(j['key'])))
        for j in ready:
            gpu=None
            if j['gpu_required']:
                if j.get('resource_measurement_job'):
                    gate=dict(resource_measurement=True,direction=j['direction'])
                    need=12288 if j['direction']=='L6' else 8192
                else:
                    gate=validated_gate(r,j['direction'],manifest)
                    if gate is None: continue
                    need=3072+math.ceil(1.25*gate['configuration']['peak_reserved_mib'])
                snapshot=gpu_snapshot();state['gpu_snapshot']=snapshot
                eligible=[]
                for device in snapshot['devices']:
                    g=device['gpu']
                    if g not in FIRST or g in busy: continue
                    first=state['jobs'][FIRST[g]]
                    if first['status']=='pending' and j['key']!=FIRST[g]: continue
                    if j['key'] in FIRST.values() and FIRST[g]!=j['key']: continue
                    if device['total_mib']-device['used_mib']>=need: eligible.append(device)
                if not eligible: continue
                gpu=max(eligible,key=lambda d:d['total_mib']-d['used_mib'])['gpu'];busy.add(gpu)
                j['admission']=dict(required_free_mib=need,snapshot=snapshot,gate=gate)
            elif cpus>=2: continue
            else: cpus+=1
            log=r/'logs'/(j['key']+'.log');log.parent.mkdir(parents=True,exist_ok=True);stream=log.open('ab')
            env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
            if gpu is not None: env['CUDA_VISIBLE_DEVICES']=str(gpu)
            child=subprocess.Popen(j['cmd'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            children[j['key']]=child;streams[j['key']]=stream
            identity=proc_identity(child.pid);assert identity is not None
            j.update(status='running',pid=child.pid,process_identity=identity,gpu=gpu,log=str(log),started_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()))
            print('START',j['key'],gpu,child.pid,flush=True);dump(sp,state)
        state['stage']='running';dump(sp,state);time.sleep(15)


if __name__=='__main__':main()
