"""Run the same read-only gradient diagnostic when E3/E10 anchors exist."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

WORK = Path(__file__).resolve().parents[2]
ROOT = Path('/data/wangcheng/checkpoint/ue_compact_joint_20261024')
PYTHON = '/data/wangcheng/envs/cvt2/bin/python'
DRIVER = WORK / 'scripts/analysis/ue_compact_joint_loss_direction_20261025.py'


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def ticks(pid):
    try:
        return Path('/proc', str(pid), 'stat').read_text().rsplit(') ', 1)[1].split()[19]
    except FileNotFoundError:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    args = p.parse_args()
    root = args.root
    folder = root / 'joint_loss_direction/owner'
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (folder / 'launch.json').exists(), 'Existing owner requires explicit recovery review'
    pid = json.loads((root / 'execution_state.json').read_text())['running']['rgbd_train']['pid']
    started = ticks(pid)
    assert started is not None
    initial = json.loads((root / 'joint_loss_direction/epoch1/complete.json').read_text())
    assert initial['complete'] and initial['rows'] == 8 and initial['no_optimizer_updates']
    driver_sha = sha(DRIVER)
    assert initial['protocol']['diagnostic_source_sha256'] == driver_sha
    owner_sha = sha(__file__)
    (folder / DRIVER.name).write_bytes(DRIVER.read_bytes())
    (folder / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    dump(folder / 'launch.json', dict(pid=os.getpid(), rgbd_training_pid=pid,
        rgbd_training_start_ticks=started, driver_sha256=driver_sha, owner_sha256=owner_sha,
        epochs=[3, 10], allocator_cap_MiB=14336, context_reserve_MiB=512, margin_MiB=2048))
    done, failed, running = set(), [], {}
    while len(done) < 2:
        assert sha(DRIVER) == driver_sha and sha(__file__) == owner_sha
        for epoch, child in list(running.items()):
            rc = child['process'].poll()
            if rc is None:
                continue
            child['log'].close()
            marker = root / 'joint_loss_direction' / ('epoch%d' % epoch) / 'complete.json'
            value = json.loads(marker.read_text()) if marker.exists() else {}
            if rc == 0 and value.get('complete') and value.get('rows') == 8 and value.get('no_optimizer_updates'):
                assert value['protocol']['diagnostic_source_sha256'] == driver_sha
                done.add(epoch)
            else:
                failed.append(dict(epoch=epoch, returncode=rc))
            del running[epoch]
        if not failed:
            raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                '--format=csv,noheader,nounits'], text=True)
            available = {int(s.split(',')[0]): int(s.split(',')[1]) - 2048 for s in raw.splitlines()}
            raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                '--format=csv,noheader,nounits'], text=True)
            used = {int(s.split(',')[0]): int(s.split(',')[1]) for s in raw.splitlines()}
            for child in running.values():
                available[child['gpu']] -= max(0, 14848 - used.get(child['process'].pid, 0))
            for epoch in (3, 10):
                if epoch in done or epoch in running:
                    continue
                anchor = root / 'rgbd' / ('epoch%d.pth' % epoch)
                epoch_log = root / 'rgbd' / ('epoch%d.json' % epoch)
                if not (anchor.exists() and epoch_log.exists()):
                    assert ticks(pid) == started, 'RGB-D process ended without required anchor; inspect before recovery'
                    continue
                cards = [g for g, free in available.items() if free >= 14848]
                if not cards:
                    continue
                gpu = max(cards, key=available.get)
                available[gpu] -= 14848
                log = (folder / ('epoch%d.log' % epoch)).open('x')
                command = [PYTHON, '-u', str(DRIVER), '--root', str(root), '--epoch', str(epoch)]
                process = subprocess.Popen(command, cwd=WORK,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OPENBLAS_NUM_THREADS='1',
                        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                running[epoch] = dict(process=process, gpu=gpu, log=log)
                dump(folder / ('epoch%d_launch.json' % epoch), dict(pid=process.pid, gpu=gpu, command=command))
        dump(folder / 'state.json', dict(stage='failure_requires_review' if failed else 'waiting_or_running',
            done=sorted(done), failed=failed, observed=time.time(),
            running={str(e): dict(pid=c['process'].pid, gpu=c['gpu']) for e, c in running.items()}))
        if failed and not running:
            raise RuntimeError(failed)
        if len(done) < 2:
            time.sleep(30)
    dump(folder / 'complete.json', dict(complete=True, epochs=[3, 10], driver_sha256=driver_sha,
        scope='same fixed8 train functions at two anchors, no training/selection changes; scientific review required'))


if __name__ == '__main__':
    main()
