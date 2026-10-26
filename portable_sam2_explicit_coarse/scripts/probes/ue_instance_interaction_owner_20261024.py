"""Persistent measured-capacity owner for the preregistered 14-arm study.

Only this study's children are managed. Existing jobs are never terminated.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

WORK = Path(__file__).resolve().parents[2]
PYTHON = '/data/wangcheng/envs/cvt2/bin/python'
SCRIPT = 'scripts/probes/ue_instance_interaction_20261024.py'
OUT = Path('/data/wangcheng/checkpoint/ue_instance_interaction_20261024')
BASE = Path('/data/wangcheng/checkpoint/ue_geometry_conditional_online_20261024/geometry/test/prefix_records.json')


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2) + '\n')
    temp.replace(path)


def arm_name(seed, family, route, content, rgb):
    return f'seed{seed}_{family}_{route}_{content}_{rgb}'


def arms():
    values = []
    for family, seed in [('geom', 44), ('rgb', 44), ('geom', 45)]:
        for route in ('internal', 'head'):
            for content in ('geometry', 'constant'):
                for rgb in ('correct', 'constant') if family == 'rgb' else ('correct',):
                    if family == 'rgb' and content == 'constant' and rgb == 'constant':
                        continue
                    values.append(dict(seed=seed, family=family, route=route, content=content, rgb=rgb,
                        name=arm_name(seed, family, route, content, rgb)))
    assert len(values) == 14
    return values


def arguments(root, arm):
    return ['--root', str(root), '--out', str(root / arm['name']),
            '--seed', str(arm['seed']), '--family', arm['family'], '--route', arm['route'],
            '--content', arm['content'], '--rgb', arm['rgb']]


def records(root, arm, kind='combined', mode=None):
    mode = mode or arm['content']
    return root / arm['name'] / 'test' / mode / f'{kind}_records.json'


def build_jobs(root):
    jobs = []
    for arm in arms():
        args = arguments(root, arm)
        train = arm['name'] + '_train'
        jobs.append(dict(name=train, type='train', family=arm['family'], route=arm['route'],
            requires=[], marker=str(root / arm['name'] / 'training_complete.json'),
            command=[PYTHON, '-u', SCRIPT, 'train'] + args))
        evaluations = [('val', arm['content']), ('test', arm['content'])]
        if arm['content'] == 'geometry' and arm['rgb'] == 'correct':
            evaluations += [('test', 'constant'), ('test', 'donor')]
            if arm['family'] == 'rgb':
                evaluations += [('test', 'constant_rgb'), ('test', 'donor_rgb')]
        if arm == arms()[0]:
            evaluations += [('test', 'disabled')]
        for split, mode in evaluations:
            jobs.append(dict(name=f"{arm['name']}_{split}_{mode}", type='eval', family=arm['family'], route=arm['route'],
                requires=[train], marker=str(root / arm['name'] / split / mode / 'complete.json'),
                command=[PYTHON, '-u', SCRIPT, 'eval'] + args + ['--split', split, '--input', mode]))
    comparisons = []
    def add(name, b, c):
        comparisons.append((name, b, c))
    for seed in (44, 45):
        g = {route: dict(seed=seed, family='geom', route=route, content='geometry', rgb='correct',
             name=arm_name(seed, 'geom', route, 'geometry', 'correct')) for route in ('head', 'internal')}
        k = {route: {**g[route], 'content': 'constant', 'name': arm_name(seed, 'geom', route, 'constant', 'correct')}
             for route in ('head', 'internal')}
        add(f'geom_s{seed}_internal_vs_head', records(root, g['head']), records(root, g['internal']))
        add(f'geom_s{seed}_internal_vs_head_satellite', records(root, g['head'], 'satellite'), records(root, g['internal'], 'satellite'))
        add(f'geom_s{seed}_constant_internal_vs_head', records(root, k['head']), records(root, k['internal']))
        for route in ('head', 'internal'):
            a = g[route]
            add(f'geom_s{seed}_{route}_content', records(root, k[route]), records(root, a))
            add(f'geom_s{seed}_{route}_content_satellite', records(root, k[route], 'satellite'), records(root, a, 'satellite'))
            add(f'geom_s{seed}_{route}_net', BASE, records(root, a))
            add(f'geom_s{seed}_{route}_satellite_change', BASE, records(root, a, 'satellite'))
            add(f'geom_s{seed}_{route}_candidate_addition', records(root, a, 'satellite'), records(root, a))
            add(f'geom_s{seed}_{route}_constant_input', records(root, a, mode='constant'), records(root, a))
            add(f'geom_s{seed}_{route}_donor_input', records(root, a, mode='donor'), records(root, a))
    for route in ('head', 'internal'):
        a = dict(seed=44, family='rgb', route=route, content='geometry', rgb='correct',
                 name=arm_name(44, 'rgb', route, 'geometry', 'correct'))
        k = {**a, 'rgb': 'constant', 'name': arm_name(44, 'rgb', route, 'geometry', 'constant')}
        add(f'rgb_{route}_content', records(root, k), records(root, a))
        add(f'rgb_{route}_content_satellite', records(root, k, 'satellite'), records(root, a, 'satellite'))
        depth_constant = {**a, 'content': 'constant', 'name': arm_name(44, 'rgb', route, 'constant', 'correct')}
        add(f'rgb_{route}_depth_values', records(root, depth_constant), records(root, a))
        add(f'rgb_{route}_depth_values_satellite', records(root, depth_constant, 'satellite'), records(root, a, 'satellite'))
        add(f'rgb_{route}_net', BASE, records(root, a))
        add(f'rgb_{route}_satellite_change', BASE, records(root, a, 'satellite'))
        add(f'rgb_{route}_candidate_addition', records(root, a, 'satellite'), records(root, a))
        for mode in ('constant', 'donor', 'constant_rgb', 'donor_rgb'):
            add(f'rgb_{route}_{mode}_input', records(root, a, mode=mode), records(root, a))
    for rgb in ('correct', 'constant'):
        h = dict(seed=44, family='rgb', route='head', content='geometry', rgb=rgb,
                 name=arm_name(44, 'rgb', 'head', 'geometry', rgb))
        i = {**h, 'route': 'internal', 'name': arm_name(44, 'rgb', 'internal', 'geometry', rgb)}
        add(f'rgb_{rgb}_internal_vs_head', records(root, h), records(root, i))
        add(f'rgb_{rgb}_internal_vs_head_satellite', records(root, h, 'satellite'), records(root, i, 'satellite'))
    for name, b, c in comparisons:
        required = []
        for path in (b, c):
            if path == BASE:
                continue
            relative = path.relative_to(root)
            required.append('_'.join(relative.parts[:3]))
        jobs.append(dict(name=name, type='analysis', family=None, route=None, requires=required,
            marker=str(root / 'analysis' / f'{name}.json'), command=[PYTHON, '-u',
              'scripts/probes/ue_joint_encoding_fast_analysis.py', '--baseline', str(b), '--candidate', str(c),
              '--output', str(root / 'analysis' / f'{name}.json'), '--split', 'test', '--bootstrap', '500']))
    assert len({j['name'] for j in jobs}) == len(jobs)
    names = {j['name'] for j in jobs}
    assert all(set(j['requires']) <= names for j in jobs)
    return jobs


def valid_marker(job):
    path = Path(job['marker'])
    if not path.exists():
        return False
    value = json.loads(path.read_text())
    if job['type'] == 'analysis':
        return len(value.get('metrics', {})) == 4 and bool(value.get('accumulation_equivalence'))
    return value.get('complete') is True


def free_memory():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                   '--format=csv,noheader,nounits'], text=True)
    return {int(a): int(b) for a, b in (line.split(',') for line in raw.splitlines())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=OUT)
    p.add_argument('--plan-only', action='store_true')
    a = p.parse_args()
    root = a.root
    root.mkdir(parents=True, exist_ok=True)
    (root / 'analysis').mkdir(exist_ok=True)
    jobs = build_jobs(root)
    dump(root / 'job_plan.json', jobs)
    if a.plan_only:
        print('PLAN', len(arms()), 'arms', len(jobs), 'jobs')
        return
    assert json.loads((root / 'cpu_gate.json').read_text())['passed']
    for family in ('geom', 'rgb'):
        for route in ('head', 'internal'):
            gate = root / f'gate_{family}_{route}'
            assert json.loads((gate / 'gate.json').read_text())['passed']
            assert json.loads((gate / 'extended_gate.json').read_text())['passed']
    # A flock prevents a second owner from accidentally launching duplicate arms.
    import fcntl
    lock = (root / 'owner.lock').open('w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid()))
    lock.flush()
    running = {}
    done = {j['name'] for j in jobs if valid_marker(j)}
    failed = []
    while len(done) + len(failed) < len(jobs):
        baseline_marker = root / 'baseline/test/complete.json'
        baseline_ready = baseline_marker.exists()
        if baseline_ready:
            assert json.loads(baseline_marker.read_text())['historical_exact']
        else:
            launched = json.loads((root / 'baseline_launch.json').read_text())
            stat = Path('/proc') / str(launched['pid']) / 'stat'
            live = stat.exists() and stat.read_text().split()[2] != 'Z'
            if not live and not failed:
                failed.append(dict(name='baseline_replay', reason='ended without exact 459-image audit'))
        for name, row in list(running.items()):
            rc = row['process'].poll()
            if rc is None:
                continue
            row['log'].close()
            job = row['job']
            if rc == 0 and valid_marker(job):
                done.add(name)
            else:
                failed.append(dict(name=name, returncode=rc, marker_valid=valid_marker(job)))
            del running[name]
        if failed:
            # Keep all live children; do not cancel other arms. A failed job
            # is audited explicitly before any retry, never silently restarted.
            dump(root / 'state.json', dict(stage='failure_requires_audit', pid=os.getpid(),
                 failed=failed, done=sorted(done), running={k: v['process'].pid for k, v in running.items()}))
            if not running:
                raise RuntimeError(failed)
            time.sleep(10)
            continue
        free = free_memory()
        for job in jobs:
            name = job['name']
            if name in done or name in running or not set(job['requires']) <= done:
                continue
            env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
            gpu = None
            if job['type'] == 'analysis':
                if not baseline_ready:
                    continue
                if sum(v['job']['type'] == 'analysis' for v in running.values()) >= 4:
                    continue
            else:
                gate = root / f"gate_{job['family']}_{job['route']}"
                reports = [json.loads((gate / f).read_text()) for f in ('gate.json', 'extended_gate.json')]
                need = max(r['peak_reserved_MiB'] for r in reports) + 2048
                candidates = [g for g in free if free[g] >= need and
                    sum(v['gpu'] == g for v in running.values()) < 2]
                if not candidates:
                    continue
                gpu = max(candidates, key=lambda g: free[g])
                env['CUDA_VISIBLE_DEVICES'] = str(gpu)
                free[gpu] -= need
            logpath = root / 'owner_logs' / f'{name}.log'
            logpath.parent.mkdir(exist_ok=True)
            log = logpath.open('a')
            argv = list(job['command'])
            if job['type'] == 'train':
                out = Path(job['marker']).parent
                checkpoints = sorted(out.glob('epoch*.pth'), key=lambda x: int(x.stem[5:]))
                if checkpoints:
                    argv += ['--resume', str(checkpoints[-1])]
            proc = subprocess.Popen(argv, cwd=WORK, stdout=log, stderr=subprocess.STDOUT, env=env)
            running[name] = dict(process=proc, log=log, gpu=gpu, job=job)
            dump(root / 'launches' / f'{name}_{proc.pid}.json', dict(pid=proc.pid, gpu=gpu, command=argv,
                 owner_pid=os.getpid(), started=time.time()))
        dump(root / 'state.json', dict(stage='running', pid=os.getpid(), done=sorted(done), failed=failed,
             running={k: dict(pid=v['process'].pid, gpu=v['gpu'], type=v['job']['type']) for k, v in running.items()},
             pending=[j['name'] for j in jobs if j['name'] not in done and j['name'] not in running]))
        if len(done) < len(jobs):
            time.sleep(10)
    dump(root / 'owner_complete.json', dict(complete=True, done=sorted(done),
         warning='Execution complete only; final source/checkpoint/evidence audit is still required'))


if __name__ == '__main__':
    main()
