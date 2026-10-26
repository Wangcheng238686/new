"""Persistent four-arm joint train/test execution after foreground evidence review."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

import ue_compact_joint_run_20261024 as run

PYTHON = '/data/wangcheng/envs/cvt2/bin/python'


def jobs(root):
    values = []
    for arm in run.PROTOCOL['arms']:
        for phase in ('train', 'test'):
            values.append(dict(name=f'{arm}_{phase}', arm=arm, phase=phase,
                requires=[] if phase == 'train' else [f'{arm}_train'],
                marker=str(root / arm / ('training_complete.json' if phase == 'train' else 'test/complete.json')),
                command=[PYTHON, '-u', 'scripts/probes/ue_compact_joint_run_20261024.py', phase,
                         '--root', str(root), '--arm', arm]))
    return values


def valid(job):
    path = Path(job['marker'])
    if not path.exists():
        return False
    value = json.loads(path.read_text())
    if job['phase'] == 'train':
        return value.get('complete') and value.get('updates') == 26700 and value.get('epoch') == 100
    return value.get('complete') and value.get('split') == 'test' and value.get('images') == 459


def memory():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                   '--format=csv,noheader,nounits'], text=True)
    free = {int(row.split(',')[0]): int(row.split(',')[1]) for row in raw.splitlines()}
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                   '--format=csv,noheader,nounits'], text=True)
    used = {int(row.split(',')[0]): int(row.split(',')[1]) for row in raw.splitlines()}
    return free, used


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=run.ROOT)
    p.add_argument('--plan-only', action='store_true')
    args = p.parse_args()
    root = args.root
    plan = jobs(root)
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return
    lock = (root / 'execution.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    expected = run.sources()
    engineering = json.loads((root / 'engineering_complete.json').read_text())
    review = json.loads((root / 'scientific_launch_review.json').read_text())
    assert engineering['passed'] and engineering['source_hashes'] == expected
    assert review['decision'] == 'proceed_to_matched_joint_training' and review['source_hashes'] == expected
    # This is an agent evidence record, not a user permission request. It must
    # describe actual reviewed science; the owner never invents that decision.
    for path, sha in review['evidence_sha256'].items():
        assert run.core.C.file_sha(path) == sha
    assert not (root / 'execution_state.json').exists(), 'Existing execution requires explicit recovery review'
    run.core.C.dump(root / 'execution_plan.json', plan)
    source_file = Path(__file__).resolve()
    archived = root / 'source/scripts/probes' / source_file.name
    archived.parent.mkdir(parents=True, exist_ok=True)
    archived.write_bytes(source_file.read_bytes())
    owner_sha = run.core.C.file_sha(source_file)
    run.core.C.dump(root / 'owner_launch.json', dict(pid=os.getpid(), source_sha256=owner_sha,
        started=time.time(), tasks=8, arms=4, source_hashes=expected))
    working = 5648 + 512  # max-GT joint gate + CUDA context; device margin below
    done, failed, running = set(), [], {}
    while len(done) < len(plan):
        for name, row in list(running.items()):
            rc = row['process'].poll()
            if rc is None:
                continue
            row['log'].close()
            if rc == 0 and valid(row['job']):
                done.add(name)
            else:
                failed.append(dict(name=name, returncode=rc, log=str(row['log_path'])))
            del running[name]
        if not failed:
            assert run.sources() == expected and run.core.C.file_sha(source_file) == owner_sha
            free, used = memory()
            available = {gpu: value - 2048 for gpu, value in free.items()}
            for row in running.values():
                available[row['gpu']] -= max(0, working - used.get(row['process'].pid, 0))
            for job in plan:
                if job['name'] in done or job['name'] in running or not set(job['requires']) <= done:
                    continue
                if valid(job):
                    done.add(job['name'])
                    continue
                cards = [gpu for gpu, amount in available.items() if amount >= working]
                if not cards:
                    continue
                gpu = max(cards, key=available.get)
                available[gpu] -= working
                log_path = root / f"{job['name']}_owner.log"
                log = log_path.open('a')
                process = subprocess.Popen(job['command'], cwd=run.WORK,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OPENBLAS_NUM_THREADS='1',
                             OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                running[job['name']] = dict(process=process, log=log, log_path=log_path, job=job, gpu=gpu)
                run.core.C.dump(root / 'launches' / f"{job['name']}_{process.pid}.json",
                    dict(pid=process.pid, gpu=gpu, command=job['command'], started=time.time()))
        run.core.C.dump(root / 'execution_state.json', dict(stage='failure_requires_review' if failed else 'running',
            owner_pid=os.getpid(), done=sorted(done), failed=failed,
            running={name: dict(pid=row['process'].pid, gpu=row['gpu'], phase=row['job']['phase'])
                     for name, row in running.items()},
            pending=[j['name'] for j in plan if j['name'] not in done and j['name'] not in running],
            resource_policy='measured6160MiB working set; 2048MiB/device margin; no fixed slot cap'))
        if failed and not running:
            raise RuntimeError(failed)
        if len(done) < len(plan):
            time.sleep(10)
    assert all(valid(job) for job in plan)
    run.core.C.dump(root / 'execution_complete.json', dict(complete=True, tasks=8, owner_pid=os.getpid(),
        source_hashes=expected, foreground_final_audit_required=True))


if __name__ == '__main__':
    main()
