"""Co-rent two independent full100 four-GPU RGB-D seed groups."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from ue_rgbd_two_seed_ddp_20261027 import ROOT, WORK, PROTOCOL, hashes, base

PYTHON = '/data/wangcheng/envs/cvt2/bin/python'
SCRIPT = WORK / 'scripts/probes/ue_rgbd_two_seed_ddp_20261027.py'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    root = parser.parse_args().root
    lock = (root / 'owner.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (root / 'owner_launch.json').exists(), 'Recovery requires actual process/checkpoint audit'
    ready = json.loads((root / 'engineering_complete.json').read_text())
    assert ready['passed'] and ready['source_hashes'] == hashes()
    reserve = ready['per_GPU_reservation_MiB']
    jobs = {}
    for phase in ('train', 'eval'):
        for seed in PROTOCOL['seeds']:
            name = 'seed%d_%s' % (seed, phase)
            marker = root / ('seed%d' % seed) / ('training_complete.json' if phase == 'train' else 'test/complete.json')
            jobs[name] = dict(seed=seed, phase=phase, status='pending', marker=str(marker),
                requires=[] if phase == 'train' else ['seed%d_train' % seed],
                command=[PYTHON, '-u', '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                    '--nproc_per_node=4', str(SCRIPT), phase, '--seed', str(seed), '--root', str(root)])
    base.core.C.dump(root / 'job_plan.json', jobs)
    base.core.C.dump(root / 'owner_launch.json', dict(pid=os.getpid(), started=time.time(),
        git_commit=ready['git_commit'], source_sha256=base.core.C.file_sha(__file__),
        per_GPU_reservation_MiB=reserve, GPU_indices=[0, 1, 2, 3]))
    children, streams = {}, {}
    while True:
        for name, child in list(children.items()):
            code = child.poll()
            if code is None:
                continue
            streams.pop(name).close()
            del children[name]
            job = jobs[name]
            marker = Path(job['marker'])
            okay = code == 0 and marker.exists()
            if okay:
                receipt = json.loads(marker.read_text())
                okay = receipt.get('complete') and (receipt.get('updates') == 26700 if job['phase'] == 'train'
                                                    else receipt.get('images') == 459)
            job.update(status='complete' if okay else 'failed', returncode=code)
            print('END', name, code, job['status'], flush=True)
        failed = [name for name, job in jobs.items() if job['status'] == 'failed']
        if not failed:
            assert hashes() == ready['source_hashes']
            assert all(base.core.C.file_sha(WORK.parent / p) == h for p, h in ready['imported_sources'].items())
            raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                          '--format=csv,noheader,nounits'], text=True)
            free = {int(s.split(',')[0]): int(s.split(',')[1]) - PROTOCOL['device_margin_MiB']
                    for s in raw.splitlines()}
            raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                          '--format=csv,noheader,nounits'], text=True)
            used = {int(s.split(',')[0]): int(s.split(',')[1]) for s in raw.splitlines()
                    if s.split(',')[1].strip().isdigit()}
            for name in children:
                job = jobs[name]
                launch = root / ('seed%d' % job['seed']) / (job['phase'] + '_launch.json')
                ranks = json.loads(launch.read_text())['ranks'] if launch.exists() else []
                for gpu in range(4):
                    pid = next((r['pid'] for r in ranks if r['local_rank'] == gpu), None)
                    free[gpu] -= max(0, reserve - used.get(pid, 0))
            for name, job in jobs.items():
                if job['status'] != 'pending' or any(jobs[d]['status'] != 'complete' for d in job['requires']):
                    continue
                if any(free[gpu] < reserve for gpu in range(4)):
                    continue
                for gpu in range(4):
                    free[gpu] -= reserve
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3', OMP_NUM_THREADS='2',
                           MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
                path = root / 'logs' / (name + '.log')
                path.parent.mkdir(exist_ok=True)
                stream = path.open('w')
                child = subprocess.Popen(job['command'], cwd=WORK, env=env, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                children[name], streams[name] = child, stream
                job.update(status='running', launcher_pid=child.pid, log=str(path), GPU_indices=[0, 1, 2, 3])
                print('START', name, 'all4GPUs', child.pid, flush=True)
        complete = all(job['status'] == 'complete' for job in jobs.values())
        base.core.C.dump(root / 'state.json', dict(stage='complete' if complete else 'failure_requires_audit' if failed else 'running',
            owner_pid=os.getpid(), jobs=jobs, failed=failed))
        if complete:
            summaries = {}
            for seed in PROTOCOL['seeds']:
                folder = root / ('seed%d' % seed)
                training = json.loads((folder / 'training_complete.json').read_text())
                test = json.loads((folder / 'test/complete.json').read_text())
                assert test['checkpoint_sha256'] == training['best_checkpoint_sha256']
                summaries[str(seed)] = dict(best=training['best'], metrics=json.loads((folder / 'test/metrics.json').read_text()))
            base.core.C.dump(root / 'complete.json', dict(complete=True, seeds=summaries,
                git_commit=ready['git_commit'], scientific_review_required=True,
                scope='Two-seed RGBD full100 DDP performance; not a matched DDP satellite net-gain experiment'))
            return
        if failed and not children:
            raise RuntimeError(failed)
        time.sleep(15)


if __name__ == '__main__':
    main()
