"""Adopt remaining immutable study jobs and use available CPU/RAM for analysis."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import ue_instance_interaction_capacity_owner_20261024 as cap
old = cap.old


def rss_mib(pid):
    try:
        values = (Path('/proc') / str(pid) / 'status').read_text().splitlines()
        return next(int(x.split()[1]) / 1024 for x in values if x.startswith('VmRSS:'))
    except (FileNotFoundError, StopIteration):
        return 0.


def available_ram():
    values = Path('/proc/meminfo').read_text().splitlines()
    return next(int(x.split()[1]) / 1024 for x in values if x.startswith('MemAvailable:'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--take-over', type=int, required=True)
    p.add_argument('--root', type=Path, default=old.OUT)
    p.add_argument('--cpu-jobs', type=int, default=12)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    assert 4 < args.cpu_jobs <= min(16, (os.cpu_count() or 1) // 2)
    root = args.root
    jobs = old.build_jobs(root)
    assert jobs == json.loads((root / 'job_plan.json').read_text()) and len(jobs) == 119
    jobmap = {j['name']: j for j in jobs}
    original = cap.identity(args.take_over)
    assert original and 'ue_instance_interaction_capacity_owner_20261024.py' in original['command']
    state = json.loads((root / 'state.json').read_text())
    assert state['pid'] == args.take_over and state['stage'] == 'running' and not state['failed']
    done = {j['name'] for j in jobs if old.valid_marker(j)}
    pending = [j for j in jobs if j['name'] not in done and j['name'] not in state['running']]
    assert pending and all(j['type'] == 'analysis' for j in pending), 'GPU scheduling still needed'
    ready = [j['name'] for j in pending if set(j['requires']) <= done]
    assert ready and available_ram() > 32768
    print('CPU ADMISSION', len(ready), 'ready, current', len(state['running']),
          'limit', args.cpu_jobs, 'MemAvailableMiB', available_ram(), flush=True)
    if args.dry_run:
        return
    manifest = json.loads((root / 'study_manifest.json').read_text())
    assert all(cap.sha(old.WORK / n) == sh for n, sh in manifest['sources'].items())
    finish = json.loads((root / 'finish_launch.json').read_text())
    finish_identity = cap.identity(finish['pid'])
    assert finish_identity and 'ue_instance_interaction_finish_20261024.py' in finish_identity['command']
    os.kill(finish['pid'], signal.SIGSTOP)
    os.kill(args.take_over, signal.SIGSTOP)
    try:
        state = json.loads((root / 'state.json').read_text())
        assert state['stage'] == 'running' and not state['failed']
        running = {}
        for n, row in state['running'].items():
            actual = cap.identity(row['pid'])
            if actual:
                running[n] = dict(identity=actual, gpu=row['gpu'], job=jobmap[n], process=None, log=None)
            else:
                assert old.valid_marker(jobmap[n]), n
        done = {j['name'] for j in jobs if old.valid_marker(j)}
        assert all(j['type'] == 'analysis' for j in jobs if j['name'] not in running and j['name'] not in done)
        history = root / 'control_history'
        history.mkdir(exist_ok=True)
        for name in ('owner_launch.json', 'finish_launch.json'):
            (history / f'pre_cpu12_{name}').write_bytes((root / name).read_bytes())
        sources = {n: cap.sha(old.WORK / n) for n in (
            'scripts/probes/ue_instance_interaction_analysis_owner_20261024.py',
            'scripts/probes/ue_instance_interaction_analysis_finish_20261024.py')}
        for n in sources:
            path = root / 'source' / n
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((old.WORK / n).read_bytes())
        handoff = dict(previous_owner=original, previous_finish=finish_identity,
            new_owner_pid=os.getpid(), adopted={n: x['identity'] for n, x in running.items()},
            cpu_jobs=args.cpu_jobs, analysis_budget_MiB=4096, available_ram_MiB=available_ram(),
            host_cpu_count=os.cpu_count(), memory_margin_MiB=8192, source_hashes=sources,
            original_jobs=119, original_plan_exact=True, no_children_interrupted=True, created=time.time())
        old.dump(root / 'analysis_capacity_handoff.json', handoff)
    except BaseException:
        os.kill(args.take_over, signal.SIGCONT)
        os.kill(finish['pid'], signal.SIGCONT)
        raise
    for pid in (finish['pid'], args.take_over):
        os.kill(pid, signal.SIGTERM)
        os.kill(pid, signal.SIGCONT)
    lock = (root / 'owner.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    lock.seek(0); lock.truncate(); lock.write(str(os.getpid())); lock.flush()
    old.dump(root / 'owner_launch.json', dict(pid=os.getpid(), started=time.time(), jobs=119,
                                             arms=14, cpu_capacity_revision=True))
    log = (root / 'analysis_finish_owner.log').open('a')
    bridge = subprocess.Popen([old.PYTHON, '-u', 'scripts/probes/ue_instance_interaction_analysis_finish_20261024.py',
        '--owner-pid', str(os.getpid())], cwd=old.WORK, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    old.dump(root / 'finish_launch.json', dict(pid=bridge.pid, owner_pid=os.getpid(),
        source_sha256=sources['scripts/probes/ue_instance_interaction_analysis_finish_20261024.py'], started=time.time(), cpu_capacity_revision=True))
    failed = []
    while len(done) < len(jobs):
        for n, row in list(running.items()):
            actual = cap.identity(row['identity']['pid'])
            if actual and actual['ticks'] == row['identity']['ticks']:
                continue
            rc = row['process'].wait() if row['process'] else None
            if row['log']:
                row['log'].close()
            if old.valid_marker(row['job']) and rc in (None, 0):
                done.add(n)
            else:
                failed.append(dict(name=n, returncode=rc))
            del running[n]
        available = available_ram() - 8192 - sum(max(0, 4096 - rss_mib(x['identity']['pid']))
            for x in running.values() if x['job']['type'] == 'analysis')
        if not failed:
            for job in jobs:
                n = job['name']
                if n in done or n in running or not set(job['requires']) <= done:
                    continue
                assert job['type'] == 'analysis'
                if sum(x['job']['type'] == 'analysis' for x in running.values()) >= args.cpu_jobs or available < 4096:
                    break
                log = (root / 'owner_logs' / f'{n}.log').open('a')
                process = subprocess.Popen(job['command'], cwd=old.WORK,
                    env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                actual = cap.identity(process.pid)
                assert actual
                running[n] = dict(identity=actual, gpu=None, job=job, process=process, log=log)
                available -= 4096
                old.dump(root / 'launches' / f'{n}_{process.pid}.json', dict(pid=process.pid, gpu=None,
                    command=job['command'], owner_pid=os.getpid(), started=time.time(), cpu_capacity_revision=True))
        old.dump(root / 'state.json', dict(stage='failure_requires_audit' if failed else 'running',
            pid=os.getpid(), done=sorted(done), failed=failed,
            running={n: dict(pid=x['identity']['pid'], gpu=x['gpu'], type=x['job']['type']) for n, x in running.items()},
            pending=[j['name'] for j in jobs if j['name'] not in done and j['name'] not in running],
            capacity_policy=f'CPU analyses <= {args.cpu_jobs}, 4096MiB working budget, 8192MiB host margin; original commands unchanged'))
        if failed and not running:
            raise RuntimeError(failed)
        if len(done) < len(jobs):
            time.sleep(10)
    old.dump(root / 'owner_complete.json', dict(complete=True, done=sorted(done), owner_pid=os.getpid(),
        scientific_plan_unchanged=True, foreground_audit_required=True))


if __name__ == '__main__':
    main()
