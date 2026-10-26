"""Adopt the fixed study and admit work by measured memory, without slot caps.

Only the original control process is replaced. Its live children are adopted;
the original scientific runner, initialization and 119 jobs remain immutable.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import ue_instance_interaction_owner_20261024 as old

CONTEXT_ALLOWANCE_MIB = 512


def identity(pid):
    path = Path('/proc') / str(pid)
    try:
        stat = (path / 'stat').read_text().split()
        command = (path / 'cmdline').read_bytes().replace(b'\x00', b' ').decode()
    except FileNotFoundError:
        return None
    if stat[2] == 'Z':
        return None
    return dict(pid=pid, ticks=stat[21], command=command)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def memory():
    free = old.free_memory()
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                   '--format=csv,noheader,nounits'], text=True)
    used = {}
    for line in raw.splitlines():
        pid, amount = line.split(',')
        used[int(pid)] = int(amount)
    return free, used


def working_set(root, job):
    # Full completed epochs expose allocator growth not present in a short
    # resource gate. Add a CUDA context allowance; reserve device margin once.
    directory = root / f"gate_{job['family']}_{job['route']}"
    values = [json.loads((directory / name).read_text())['peak_reserved_MiB']
              for name in ('gate.json', 'extended_gate.json')]
    for arm in old.arms():
        if arm['family'] != job['family'] or arm['route'] != job['route']:
            continue
        for path in (root / arm['name']).glob('epoch*.json'):
            value = json.loads(path.read_text()).get('peak_reserved_MiB')
            if value is not None:
                values.append(value)
    return max(values) + CONTEXT_ALLOWANCE_MIB


def available_memory(free, used, running, budgets, margin):
    result = {gpu: value - margin for gpu, value in free.items()}
    for name, row in running.items():
        if row['gpu'] is not None:
            # A newly launched process may not yet have allocated its model.
            # Account for this outstanding allocation before admitting more.
            result[row['gpu']] -= max(0, budgets[name] - used.get(row['identity']['pid'], 0))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=old.OUT)
    parser.add_argument('--take-over', type=int, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = args.root
    jobs = old.build_jobs(root)
    jobmap = {job['name']: job for job in jobs}
    original = identity(args.take_over)
    assert original and 'ue_instance_interaction_owner_20261024.py' in original['command']
    state = json.loads((root / 'state.json').read_text())
    assert state['pid'] == args.take_over and state['stage'] == 'running' and not state['failed']
    running = {}
    for name, row in state['running'].items():
        actual = identity(row['pid'])
        assert actual, ('Child not alive during adoption', name)
        running[name] = dict(identity=actual, gpu=row['gpu'], job=jobmap[name], process=None, log=None)
    budgets = {name: working_set(root, row['job']) for name, row in running.items() if row['gpu'] is not None}
    free, used = memory()
    margin = 2048
    available = available_memory(free, used, running, budgets, margin)
    ready = []
    done = {job['name'] for job in jobs if old.valid_marker(job)}
    for job in jobs:
        if job['type'] != 'train' or job['name'] in running or job['name'] in done:
            continue
        if not set(job['requires']) <= done:
            continue
        need = working_set(root, job)
        cards = [gpu for gpu, amount in available.items() if amount >= need]
        if cards:
            gpu = max(cards, key=available.get)
            ready.append(dict(name=job['name'], gpu=gpu, working_set_MiB=need))
            available[gpu] -= need
    print('MEASURED ADMISSION', ready, flush=True)
    if args.dry_run:
        return
    assert ready, 'No additional work fits; do not replace a healthy owner needlessly'
    # Verify frozen core source before replacing only the controller.
    manifest = json.loads((root / 'study_manifest.json').read_text())
    assert all(sha(old.WORK / name) == value for name, value in manifest['sources'].items())
    os.kill(args.take_over, signal.SIGSTOP)
    snapshot = json.loads((root / 'state.json').read_text())
    assert snapshot['stage'] == 'running' and not snapshot['failed']
    # Capture the final stopped controller snapshot so no late launch is lost.
    running = {}
    for name, row in snapshot['running'].items():
        actual = identity(row['pid'])
        if actual:
            running[name] = dict(identity=actual, gpu=row['gpu'], job=jobmap[name], process=None, log=None)
        else:
            assert old.valid_marker(jobmap[name]), ('Ended without completion during handoff', name)
    transition = dict(original_owner=original, new_owner_pid=os.getpid(),
        adopted={name: row['identity'] for name, row in running.items()},
        initial_extra_admission=ready, device_margin_MiB=margin, CUDA_context_allowance_MiB=CONTEXT_ALLOWANCE_MIB,
        original_jobs=119, frozen_scientific_sources_unchanged=True,
        source_sha256=sha(__file__), created=time.time())
    old.dump(root / 'capacity_handoff.json', transition)
    os.kill(args.take_over, signal.SIGTERM)
    os.kill(args.take_over, signal.SIGCONT)
    lock = (root / 'owner.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    lock.seek(0)
    lock.truncate()
    lock.write(str(os.getpid()))
    lock.flush()
    failed = []
    done = {job['name'] for job in jobs if old.valid_marker(job)}
    while len(done) < len(jobs):
        for name, row in list(running.items()):
            process = row['process']
            alive = identity(row['identity']['pid'])
            if alive and alive['ticks'] == row['identity']['ticks']:
                continue
            rc = process.wait() if process is not None else None
            if row['log'] is not None:
                row['log'].close()
            if old.valid_marker(row['job']) and rc in (None, 0):
                done.add(name)
            else:
                failed.append(dict(name=name, returncode=rc, adopted=process is None))
            del running[name]
        if not failed:
            free, used = memory()
            budgets = {name: working_set(root, row['job']) for name, row in running.items() if row['gpu'] is not None}
            available = available_memory(free, used, running, budgets, margin)
            for job in jobs:
                name = job['name']
                if name in running or name in done or not set(job['requires']) <= done:
                    continue
                env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
                gpu = None
                if job['type'] == 'analysis':
                    if sum(row['gpu'] is None for row in running.values()) >= 4:
                        continue
                else:
                    need = working_set(root, job)
                    cards = [card for card, amount in available.items() if amount >= need]
                    if not cards:
                        continue
                    gpu = max(cards, key=available.get)
                    available[gpu] -= need
                    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
                command = list(job['command'])
                logpath = root / 'owner_logs' / f'{name}.log'
                log = logpath.open('a')
                process = subprocess.Popen(command, cwd=old.WORK, env=env, stdout=log, stderr=subprocess.STDOUT)
                actual = identity(process.pid)
                assert actual
                running[name] = dict(identity=actual, gpu=gpu, job=job, process=process, log=log)
                old.dump(root / 'launches' / f'{name}_{process.pid}.json', dict(pid=process.pid, gpu=gpu,
                    command=command, owner_pid=os.getpid(), started=time.time(), capacity_policy='measured-working-set'))
        old.dump(root / 'state.json', dict(stage='failure_requires_audit' if failed else 'running',
            pid=os.getpid(), done=sorted(done), failed=failed,
            running={name: dict(pid=row['identity']['pid'], gpu=row['gpu'], type=row['job']['type'])
                     for name, row in running.items()},
            pending=[job['name'] for job in jobs if job['name'] not in done and job['name'] not in running],
            capacity_policy='measured peak + context allowance, one 2048MiB device margin; no slot cap'))
        if failed and not running:
            raise RuntimeError(failed)
        if len(done) < len(jobs):
            time.sleep(10)
    old.dump(root / 'owner_complete.json', dict(complete=True, done=sorted(done),
        owner_pid=os.getpid(), scientific_plan_unchanged=True, foreground_audit_required=True))


if __name__ == '__main__':
    main()
