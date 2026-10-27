#!/usr/bin/env python3
"""Adopt live four-card jobs and admit a third group at the user's request.

Training sources, checkpoints, budgets and existing rank processes are unchanged.
Only the verified old owner is terminated. No training process receives a signal.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import ue_ubr_short_owner_20261027 as old


def live(identity):
    actual = old.proc_identity(identity['pid'])
    if not actual or actual['start_ticks'] != identity['start_ticks']:
        return False
    return Path('/proc/%d/stat' % identity['pid']).read_text().rsplit(') ', 1)[1].split()[0] != 'Z'


def run(root):
    state = json.loads((root / 'owner_state.json').read_text())
    manifest = json.loads((root / 'owner_manifest.json').read_text())
    if state['failures'] or state['stage'] != 'running':
        raise RuntimeError('Only a healthy running study may be adopted')
    identity = state['owner']
    actual = old.proc_identity(identity['pid'])
    if not live(identity) or str(Path(old.__file__).resolve()) not in actual['argv']:
        raise RuntimeError('Old owner identity mismatch')
    source = manifest['source_sha256']
    for name, expected in source.items():
        if old.sha(old.REPO / name) != expected:
            raise RuntimeError('Pinned source changed: ' + name)
    for job in state['jobs'].values():
        if job['status'] == 'running':
            ranks = old.rank_records(job['rank_launch_record'], job['launcher']['start_ticks'])
            if not live(job['launcher']) or len(ranks) != 4 or not all(r['currently_live'] for r in ranks):
                raise RuntimeError('Live four-rank identity mismatch')
    audit = root / 'adoption_three_groups_20261027'
    audit.mkdir(exist_ok=False)
    for name in ('owner_state.json', 'owner_manifest.json', 'owner_job_plan.json'):
        shutil.copyfile(root / name, audit / name)
    relative = str(Path(__file__).resolve().relative_to(old.REPO))
    commit = old.git('rev-parse', 'HEAD').decode().strip()
    if hashlib.sha256(old.git('show', commit + ':' + relative)).hexdigest() != old.sha(__file__):
        raise RuntimeError('Commit adoption controller before launch')
    # Independent torchrun sessions survive their old Python supervisor's exit.
    os.kill(identity['pid'], signal.SIGTERM)
    lock = (root / 'owner.lock').open('a')
    for attempt in range(20):
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            time.sleep(.25)
    else:
        raise RuntimeError('Old owner did not release lock; no job launched')
    # Remove the former additional 1GiB PER GROUP reservation. Keep measured
    # worst peak plus CUDA context per group, and 2GiB global margin per card.
    old.RESERVATION_MIB = old.PEAK_MIB + old.CONTEXT_MIB  # 7042 MiB
    policy = dict(user_request='start waiting four-GPU control now when memory permits',
        old_owner=identity, owner=old.proc_identity(os.getpid()), control_commit=commit,
        control_sha256=old.sha(__file__), numerical_commit=manifest['git_commit'],
        max_groups=3, per_card_reservation_mib=old.RESERVATION_MIB,
        all_card_margin_mib=old.MARGIN_MIB, existing_training_processes_untouched=True)
    old.atomic(audit / 'policy.json', policy)
    jobs, failures, children, streams = state['jobs'], [], {}, {}
    while True:
        for name, job in jobs.items():
            if job['status'] != 'running':
                continue
            child = children.get(name)
            rc = child.poll() if child is not None else None
            if (child is not None and rc is None) or (child is None and live(job['launcher'])):
                continue
            if name in streams:
                streams.pop(name).close()
            children.pop(name, None)
            valid, reason = old.validate_marker(job, source, manifest['git_commit']) if rc in (None, 0) else (False, 'nonzero child returncode')
            if valid and job['action'] == 'eval':
                result = json.loads(Path(job['marker']).read_text())
                trained = json.loads((root / job['arm'] / 'training_complete.json').read_text())
                valid = result['checkpoint_sha256'] == trained['checkpoint_sha256']
                reason = None if valid else 'fixed E10 checkpoint mismatch'
            job.update(status='complete' if valid else 'failed', returncode=rc,
                       completion_evidence='validated_marker' if valid else reason)
            if not valid:
                failures.append(name + ': ' + str(reason))
        if any(old.sha(old.REPO / n) != s for n, s in source.items()):
            failures.append('Pinned numerical/old control source changed')
        if any(old.sha(p) != s for p, s in manifest['source_order_sha256'].items()):
            failures.append('Sample order changed')
        cards = old.resources(jobs)
        running = sum(j['status'] == 'running' for j in jobs.values())
        if not failures:
            for name, job in jobs.items():
                if job['status'] != 'pending' or any(jobs[d]['status'] != 'complete' for d in job['dependencies']):
                    continue
                if running >= 3 or (job['action'] == 'eval' and running):
                    continue
                if any(cards[g]['admissible_free_mib'] < old.RESERVATION_MIB for g in old.GPUS):
                    continue
                log = root / 'logs' / (name + '.log')
                stream = log.open('x')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3', OPENBLAS_NUM_THREADS='1',
                           OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
                child = subprocess.Popen(job['argv'], cwd=old.WORK, env=env, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                children[name], streams[name] = child, stream
                job.update(status='running', launcher=old.proc_identity(child.pid),
                    launched=time.time(), log=str(log), ranks=[], GPU_indices=list(old.GPUS))
                old.atomic(root / 'owner_launches' / (name + '.json'), job)
                running += 1
                for card in cards.values():
                    card['admissible_free_mib'] -= old.RESERVATION_MIB
                print('START', name, child.pid, 'four GPUs', flush=True)
        complete = all(j['status'] == 'complete' for j in jobs.values())
        old.atomic(root / 'owner_state.json', dict(stage='complete' if complete else ('failure_requires_audit' if failures else 'running'),
            owner=policy['owner'], jobs=jobs, failures=failures, GPU_headroom=cards,
            running_groups=running, updated=time.time(), source_git_commit=manifest['git_commit'],
            control_policy=policy))
        if complete:
            old.atomic(root / 'owner_complete.json', dict(complete=True, jobs=len(jobs), policy=policy))
            return
        if failures and not running:
            raise RuntimeError(str(failures))
        time.sleep(15)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    run(p.parse_args().root.resolve())
