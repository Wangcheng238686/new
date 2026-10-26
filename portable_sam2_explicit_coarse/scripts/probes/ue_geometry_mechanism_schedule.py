"""Fixed 2 x 2 x 2 geometry experiment; durable, non-preemptive co-tenant owner.

Registration freezes executable sources before training or any new test read.
Failures never trigger automatic retries or AP-dependent scheduling decisions.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
PYTHON = '/data/wangcheng/envs/cvt2/bin/python'
BASE_SHA = 'dd9106f4afe8ba72936316d7efa9621ca0c05d24836fa28c72ea322bed84ace1'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def arms():
    return [dict(name=f'{sampling}_{fusion}_{content}', sampling=sampling,
                 fusion=fusion, content=content)
            for sampling in ('bilinear', 'nearest')
            for fusion in ('concat', 'difference')
            for content in ('geometry', 'constant')]


def proc_identity(pid):
    """PID reuse safe identity, including kernel start ticks and command bytes."""
    try:
        folder = Path('/proc') / str(pid)
        stat = (folder / 'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        return dict(start_ticks=stat[19], cmdline_sha256=hashlib.sha256(
            (folder / 'cmdline').read_bytes()).hexdigest())
    except (FileNotFoundError, ProcessLookupError):
        return None


def gpu_snapshot():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.total,memory.used',
                                    '--format=csv,noheader,nounits'], text=True)
    devices = [dict(zip(('gpu', 'total_mib', 'used_mib'), map(int, row.split(','))))
               for row in rows.splitlines() if row.strip()]
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                        '--format=csv,noheader,nounits'], text=True)
    return dict(devices=devices, compute_processes=processes.splitlines())


def sources():
    names = ['scripts/probes/ue_geometry_mechanism_probe.py',
             'scripts/probes/ue_geometry_mechanism_schedule.py',
             'scripts/probes/ue_geometry_mechanism_collect.py',
             'scripts/probes/ue_v1_adapter.py', 'scripts/probes/ue_v1_adapter_analysis.py',
             'rsprompter/uav/native_geometry_mechanism.py',
             'rsprompter/uav/native_geometry_probe.py', 'rsprompter/uav/local_adapter_probe.py',
             'data/ue_minimal_dataset.py', 'data/whu_instance_dataset.py',
             'utils/coco_eval_utils.py', 'inference/model_rebuild.py', 'inference/eval_records.py',
             'train/gt_samples.py', 'data/loader.py', 'rsprompter/ckpt_utils.py',
             'rsprompter/architecture_contract.py']
    for package in ('backbone', 'cspm', 'decoder', 'detection'):
        names += [str(p.relative_to(ROOT)) for p in sorted((ROOT / 'rsprompter' / package).glob('*.py'))]
    return list(dict.fromkeys(names))


def build_jobs(r, manifest):
    jobs = {}
    probe = [PYTHON, str(ROOT / 'scripts/probes/ue_geometry_mechanism_probe.py')]
    analysis = [PYTHON, str(ROOT / 'scripts/probes/ue_v1_adapter_analysis.py')]
    def add(key, cmd, artifact, dependencies=(), gpu=False, configuration=None, priority=10):
        jobs[key] = dict(key=key, cmd=cmd, artifact=str(artifact), dependencies=list(dependencies),
                         gpu_required=gpu, configuration=configuration, priority=priority, status='pending')
    baseline = r / 'test/baseline/disabled/records.json'
    add('baseline_test', probe + ['eval', '--mode', 'disabled', '--eval-split', 'test',
         '--constants', str(r / 'common_constants.json'), '--init', str(r / 'common_init.pth'),
         '--out', str(r / 'test/baseline')], baseline, gpu=True, configuration='bilinear_concat', priority=3)
    evaluations = []
    comparisons = []
    def compare(key, first, second, deps, kind):
        out = r / 'analysis' / (key + '.json')
        add(key, analysis + ['--baseline', str(first), '--candidate', str(second),
            '--output', str(out), '--split', 'test', '--bootstrap', '500'], out, deps, priority=20)
        comparisons.append(dict(key=key, baseline=str(first), candidate=str(second), kind=kind, output=str(out)))
    for arm in manifest['arms']:
        name = arm['name']; config = arm['sampling'] + '_' + arm['fusion']; train = 'train_' + name
        common = ['--sampling', arm['sampling'], '--fusion', arm['fusion'], '--content', arm['content'],
                  '--constants', str(r / 'common_constants.json'), '--init', str(r / 'common_init.pth')]
        add(train, probe + ['train'] + common + ['--out', str(r / name)], r / name / 'last.pth',
            gpu=True, configuration=config, priority=5 if arm['content']=='geometry' else 6)
        primary = 'correct' if arm['content'] == 'geometry' else 'constant'
        for split, modes in [('val', [primary]), ('test', [primary] + (['constant', 'shuffle', 'zero'] if primary == 'correct' else []))]:
            for mode in modes:
                key = f'eval_{split}_{name}_{mode}'; folder = r / split / name
                add(key, probe + ['eval'] + common + ['--adapter', str(r / name / 'last.pth'),
                    '--mode', mode, '--eval-split', split, '--out', str(folder)],
                    folder / mode / 'evaluation_identity.json', [train], True, config, 12)
                evaluations.append(dict(key=key, arm=name, split=split, mode=mode, folder=str(folder / mode)))
        compare('baseline_' + name, baseline, r / 'test' / name / primary / 'records.json',
                ['baseline_test', f'eval_test_{name}_{primary}'], 'net_gain')
        if primary == 'correct':
            for mode in ('constant', 'shuffle'):
                compare('dependency_' + name + '_' + mode,
                        r / 'test' / name / mode / 'records.json', r / 'test' / name / 'correct/records.json',
                        [f'eval_test_{name}_{mode}', f'eval_test_{name}_correct'], 'fixed_checkpoint_dependency')
    for sampling in ('bilinear', 'nearest'):
        for fusion in ('concat', 'difference'):
            prefix = sampling + '_' + fusion
            compare('content_' + prefix, r / 'test' / (prefix + '_constant') / 'constant/records.json',
                    r / 'test' / (prefix + '_geometry') / 'correct/records.json',
                    ['eval_test_' + prefix + '_constant_constant', 'eval_test_' + prefix + '_geometry_correct'], 'matched_training_content')
    # These four contrasts were fixed before results, rather than selected from winners.
    for factor, levels, fixed_key, fixed_values in [
            ('sampling', ('bilinear', 'nearest'), 'fusion', ('concat', 'difference')),
            ('fusion', ('concat', 'difference'), 'sampling', ('bilinear', 'nearest'))]:
        for fixed in fixed_values:
            configs = [dict(sampling=fixed if fixed_key == 'sampling' else level,
                            fusion=fixed if fixed_key == 'fusion' else level) for level in levels]
            names = [c['sampling'] + '_' + c['fusion'] + '_geometry' for c in configs]
            compare(f'factor_{factor}_{fixed}', r / 'test' / names[0] / 'correct/records.json',
                    r / 'test' / names[1] / 'correct/records.json',
                    ['eval_test_' + n + '_correct' for n in names], 'factor_effect')
    prefinal = list(jobs)
    add('collect', [PYTHON, str(ROOT / 'scripts/probes/ue_geometry_mechanism_collect.py'), '--root', str(r)],
        r / 'complete_audit_summary.json', prefinal, priority=100)
    return jobs, evaluations, comparisons


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--register-only', action='store_true')
    args = p.parse_args(); r = args.root.resolve(); r.mkdir(parents=True, exist_ok=True)
    lock = (r / 'owner.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest_path = r / 'manifest.json'
    current_hashes = {name: sha(ROOT / name) for name in sources()}
    if manifest_path.exists():
        manifest = read(manifest_path)
        assert manifest['source_hashes'] == current_hashes, 'Registered executable source changed'
        assert manifest['init_sha256'] == sha(r / 'common_init.pth')
        assert manifest['constants_sha256'] == sha(r / 'common_constants.json')
    else:
        constants = read(r / 'common_constants.json')
        assert constants['split'] == 'train' and len(constants['mean_fields']) == 4
        manifest = dict(study='native_geometry_mechanism_2x2x2', arms=arms(), source_hashes=current_hashes,
                        base_sha256=BASE_SHA, init_sha256=sha(r / 'common_init.pth'),
                        constants_sha256=sha(r / 'common_constants.json'), primary_split='test',
                        selection='epoch10 last', epochs=10, updates=2670, seed=44,
                        epoch_seed_formula='1044000+epoch for E1..10', no_ema=True,
                        no_test_selection=True, bootstrap=500, bootstrap_seed=20261021,
                        registered_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()))
        jobs, evaluations, comparisons = build_jobs(r, manifest)
        manifest.update(evaluations=evaluations, comparisons=comparisons,
                        test_evaluations=21, validation_evaluations=8)
        assert len(evaluations) == 28 and len(comparisons) == 24
        dump(manifest_path, manifest)
        for name in current_hashes:
            target = r / 'frozen_source' / name; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
    expected_jobs, _, _ = build_jobs(r, manifest)
    if args.register_only:
        dump(r / 'job_plan.json', expected_jobs)
        print('Registered fixed eight-arm cohort; no jobs launched.', flush=True)
        return
    gate = read(r / 'resource_gate.json')
    assert gate['all_passed'] and gate['init_sha256'] == manifest['init_sha256']
    assert gate['constants_sha256'] == manifest['constants_sha256']
    required_configs = {s + '_' + f for s in ('bilinear', 'nearest') for f in ('concat', 'difference')}
    assert set(gate['configurations']) == required_configs
    for config in gate['configurations'].values():
        assert config['all_passed'] and config['fp32_resource_rois'] == 250
        assert config['peak_reserved_mib'] > 0
        actual_gate=read(config['gate_path'])
        assert sha(config['gate_path'])==config['gate_sha256']
        assert all(manifest['source_hashes'][n]==h for n,h in actual_gate['source_hashes'].items())
    state_path = r / 'state.json'
    state = read(state_path) if state_path.exists() else dict(stage='registered', jobs=expected_jobs)
    assert {k: v['cmd'] for k, v in state['jobs'].items()} == {k: v['cmd'] for k, v in expected_jobs.items()}
    state['owner_pid'] = os.getpid(); state['owner_identity'] = proc_identity(os.getpid())
    children = {}; streams = {}
    def save(): dump(state_path, state)
    for job in state['jobs'].values():
        if job['status'] == 'running' and proc_identity(job['pid']) != job['process_identity']:
            job['status'] = 'failed'
            job['failure'] = 'Prior owner lost child exit status; manual artifact audit required, no automatic retry'
    save()
    while True:
        for key, proc in list(children.items()):
            rc = proc.poll()
            if rc is None: continue
            streams.pop(key).close(); del children[key]
            job = state['jobs'][key]
            job.update(returncode=rc, status='complete' if rc == 0 and Path(job['artifact']).is_file() else 'failed')
            print('END', key, rc, job['status'], flush=True); save()
        for key, job in state['jobs'].items():
            if job['status'] == 'running' and key not in children and proc_identity(job['pid']) != job['process_identity']:
                job.update(status='failed', failure='Adopted process exited without observed return code; audit before repair')
                save()
        if all(j['status'] == 'complete' for j in state['jobs'].values()):
            state['stage'] = 'complete'; save(); return
        failed = [k for k, j in state['jobs'].items() if j['status'] == 'failed']
        if failed:
            state.update(stage='failure_requires_audit', failed_jobs=failed); save()
            if not any(j['status'] == 'running' for j in state['jobs'].values()): return
            time.sleep(15); continue
        assert all(sha(ROOT / name) == value for name, value in manifest['source_hashes'].items()), 'Frozen source changed'
        snapshot = gpu_snapshot(); state['gpu_snapshot'] = snapshot
        running = [j for j in state['jobs'].values() if j['status'] == 'running']
        busy = {j['gpu'] for j in running if j['gpu_required']}
        cpus = sum(not j['gpu_required'] for j in running)
        ready = [j for j in state['jobs'].values() if j['status'] == 'pending' and
                 all(state['jobs'][d]['status'] == 'complete' for d in j['dependencies'])]
        ready.sort(key=lambda j: (j['priority'], list(state['jobs']).index(j['key'])))
        for job in ready:
            gpu = None
            if job['gpu_required']:
                need = 3072 + math.ceil(1.25 * gate['configurations'][job['configuration']]['peak_reserved_mib'])
                # Refresh for every admission: external owners may start work at any time.
                snapshot = gpu_snapshot()
                eligible = [d for d in snapshot['devices'] if d['gpu'] not in busy and d['total_mib'] - d['used_mib'] >= need]
                if not eligible: continue
                chosen = max(eligible, key=lambda d: d['total_mib'] - d['used_mib'])
                gpu = chosen['gpu']; busy.add(gpu)
                job['admission'] = dict(required_free_mib=need, snapshot=snapshot)
            elif cpus >= 2: continue
            else: cpus += 1
            log = r / 'logs' / (job['key'] + '.log'); log.parent.mkdir(parents=True, exist_ok=True)
            stream = log.open('ab'); env = os.environ.copy()
            env.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
            if gpu is not None: env['CUDA_VISIBLE_DEVICES'] = str(gpu)
            child = subprocess.Popen(job['cmd'], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            children[job['key']] = child; streams[job['key']] = stream
            identity = proc_identity(child.pid)
            assert identity is not None, 'Child exited before identity could be captured'
            job.update(status='running', pid=child.pid, process_identity=identity, gpu=gpu,
                       log=str(log), started_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()))
            print('START', job['key'], gpu, child.pid, flush=True); save()
        state['stage'] = 'running'; save(); time.sleep(15)


if __name__ == '__main__':
    main()
