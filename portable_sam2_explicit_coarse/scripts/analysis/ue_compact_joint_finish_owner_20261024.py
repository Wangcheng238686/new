"""Wait for the live joint train/test owner, then run fixed read-only analyses.

This does not certify the scientific goal or replace the final checkpoint audit.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

WORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORK / 'scripts/analysis'))
import ue_compact_joint_paired_20261024 as paired
PYTHON = '/data/wangcheng/envs/cvt2/bin/python'
ROOT = paired.ROOT


def dump(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def ticks(pid):
    try:
        return Path('/proc', str(pid), 'stat').read_text().rsplit(') ', 1)[1].split()[19]
    except FileNotFoundError:
        return None


def jobs(root):
    result = [dict(name=name, kind='cpu', marker=str(root / 'paired_analysis' / (name + '.json')),
        command=[PYTHON, '-u', 'scripts/analysis/ue_compact_joint_paired_20261024.py',
                 '--root', str(root), '--comparison', name]) for name in sorted(paired.COMPARISONS)]
    for mode in ('constant_rgb', 'donor_rgb'):
        result.append(dict(name='fixed_' + mode, kind='gpu',
            marker=str(root / 'fixed_rgb_intervention' / mode / 'complete.json'),
            command=[PYTHON, '-u', 'scripts/analysis/ue_compact_joint_rgb_intervention_20261024.py',
                     '--root', str(root), '--mode', mode]))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--plan-only', action='store_true')
    args = p.parse_args()
    root = args.root
    plan = jobs(root)
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return
    folder = root / 'posttraining'
    folder.mkdir(exist_ok=True)
    lock = (folder / 'owner.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (folder / 'owner_launch.json').exists(), 'Explicit recovery review required'
    parent = json.loads((root / 'owner_launch.json').read_text())['pid']
    start = ticks(parent)
    assert start is not None, 'Training owner not confirmed live'
    files = [Path(__file__).resolve(), Path(paired.__file__).resolve(),
        WORK / 'scripts/analysis/ue_compact_joint_rgb_intervention_20261024.py',
        Path(paired.fast.__file__).resolve(), Path(paired.fast.original.__file__).resolve()]
    hashes = {str(p): paired.sha(p) for p in files}
    for path in files:
        dest = folder / 'source' / path.relative_to(WORK)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
    dump(folder / 'owner_launch.json', dict(pid=os.getpid(), started=time.time(),
        training_owner_pid=parent, training_owner_start_ticks=start, source_sha256=hashes))
    dump(folder / 'plan.json', plan)
    done, failed, running = set(), [], {}
    while not (root / 'execution_complete.json').exists():
        assert ticks(parent) == start, 'Training owner ended before completion; inspect actual tasks'
        dump(folder / 'state.json', dict(stage='waiting_for_live_train_test_owner',
            training_owner_pid=parent, observed=time.time(), tasks=10))
        time.sleep(30)
    finished = json.loads((root / 'execution_complete.json').read_text())
    assert finished['complete'] and finished['tasks'] == 8
    while len(done) < len(plan):
        assert all(paired.sha(p) == value for p, value in hashes.items())
        for name, child in list(running.items()):
            rc = child['process'].poll()
            if rc is None:
                continue
            child['log'].close()
            marker = Path(child['job']['marker'])
            if rc == 0 and marker.exists() and json.loads(marker.read_text()).get('complete'):
                done.add(name)
            else:
                failed.append(dict(name=name, returncode=rc, log=str(child['log_path'])))
            del running[name]
        if not failed:
            raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                '--format=csv,noheader,nounits'], text=True)
            available = {int(s.split(',')[0]): int(s.split(',')[1]) - 2048 for s in raw.splitlines()}
            raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                '--format=csv,noheader,nounits'], text=True)
            used = {int(s.split(',')[0]): int(s.split(',')[1]) for s in raw.splitlines()}
            for child in running.values():
                if child['job']['kind'] == 'gpu':
                    available[child['gpu']] -= max(0, 6160 - used.get(child['process'].pid, 0))
            mem = next(int(s.split()[1]) // 1024 for s in Path('/proc/meminfo').read_text().splitlines()
                       if s.startswith('MemAvailable:')) - 8192
            # Reserve a full 4GiB for each running CPU task; intentionally conservative.
            mem -= sum(4096 for c in running.values() if c['job']['kind'] == 'cpu')
            for job in plan:
                if job['name'] in done or job['name'] in running:
                    continue
                assert not Path(job['marker']).exists(), 'Unexpected prior output requires review'
                gpu = None
                if job['kind'] == 'gpu':
                    cards = [g for g, free in available.items() if free >= 6160]
                    if not cards:
                        continue
                    gpu = max(cards, key=available.get)
                    available[gpu] -= 6160
                elif mem < 4096:
                    continue
                else:
                    mem -= 4096
                env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
                env['CUDA_VISIBLE_DEVICES'] = '' if gpu is None else str(gpu)
                log_path = folder / (job['name'] + '.log')
                log = log_path.open('x')
                process = subprocess.Popen(job['command'], cwd=WORK, env=env,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                running[job['name']] = dict(process=process, log=log, log_path=log_path, job=job, gpu=gpu)
        dump(folder / 'state.json', dict(stage='failure_requires_review' if failed else 'postprocessing',
            done=sorted(done), failed=failed, observed=time.time(),
            running={n: dict(pid=c['process'].pid, gpu=c['gpu']) for n, c in running.items()}))
        if failed and not running:
            raise RuntimeError(failed)
        if len(done) < len(plan):
            time.sleep(15)
    dump(folder / 'complete.json', dict(complete=True, tasks=10, source_sha256=hashes,
        scope='eight matched paired reports and two fixed RGB interventions only',
        final_checkpoint_and_scientific_audit_still_required=True))


if __name__ == '__main__':
    main()
