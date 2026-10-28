"""Extend the waiting read-only finish plan; leave numerical training unchanged."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

WORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORK / 'scripts/analysis'))
import ue_compact_joint_finish_owner_20261024 as prior

ROOT = prior.ROOT
PYTHON = prior.PYTHON


def jobs(root):
    values = prior.jobs(root)
    for row in values:
        row['requires'] = []
    for mode in ('constant_rgb', 'donor_rgb'):
        for view in ('combined', 'original_path'):
            values.append(dict(name='fixed_%s_%s' % (mode, view), kind='cpu',
                requires=['fixed_' + mode], marker=str(root / 'fixed_rgb_intervention' / mode / ('paired_' + view + '.json')),
                command=[PYTHON, '-u', 'scripts/analysis/ue_compact_joint_fixed_rgb_paired_20261025.py',
                         '--root', str(root), '--mode', mode, '--view', view]))
    values.append(dict(name='final_checkpoint_audit', kind='cpu', requires=[], marker_flag='passed',
        marker=str(root / 'checkpoint_audit/final.json'), command=[PYTHON, '-u',
        'scripts/analysis/ue_compact_joint_checkpoint_audit_20261025.py', '--root', str(root), '--final']))
    values.append(dict(name='final_input_identity', kind='cpu', requires=[],
        marker=str(root / 'final_input_identity.json'), command=[PYTHON, '-u',
        'scripts/analysis/ue_compact_joint_final_inputs_20261025.py', '--root', str(root)]))
    before_cost = [r['name'] for r in values]
    for case in ('satellite', 'rgbd_combined', 'rgbd_original'):
        name = 'cost_' + case
        values.append(dict(name=name, kind='cost', requires=list(before_cost),
            marker=str(root / 'joint_cost' / case / 'complete.json'), command=[PYTHON, '-u',
            'scripts/analysis/ue_compact_joint_cost_20261025.py', '--root', str(root), '--case', case]))
        before_cost.append(name)
    assert len(values) == len({r['name'] for r in values}) == 19
    assert all(set(r['requires']) <= {v['name'] for v in values} for r in values)
    return values


def process(pid):
    try:
        raw = Path('/proc', str(pid), 'stat').read_text().rsplit(') ', 1)[1].split()
        return dict(state=raw[0], start_ticks=raw[19])
    except FileNotFoundError:
        return None


def live(pid, start):
    p = process(pid)
    return p is not None and p['state'] not in ('Z', 'X') and p['start_ticks'] == start


def gpu_state():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,uuid',
        '--format=csv,noheader,nounits'], text=True)
    cards = {int(s.split(',')[0]): dict(free=int(s.split(',')[1]), uuid=s.split(',')[2].strip()) for s in raw.splitlines()}
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory,gpu_uuid',
        '--format=csv,noheader,nounits'], text=True)
    apps = {int(s.split(',')[0]): dict(memory=int(s.split(',')[1]), uuid=s.split(',')[2].strip()) for s in raw.splitlines()}
    return cards, apps


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
    lock = (folder / 'owner.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (folder / 'owner_v2_launch.json').exists(), 'Explicit recovery review required'
    handoff = json.loads((folder / 'v2_handoff.json').read_text())
    assert not live(handoff['prior_pid'], handoff['prior_start_ticks'])
    parent = handoff['training_owner_pid']
    start = handoff['training_owner_start_ticks']
    assert live(parent, start), 'Actual train/test owner is not live'
    assert handoff['prior_waiting_only'] and handoff['prior_children'] == []
    files = [Path(__file__).resolve(), Path(prior.__file__).resolve(), Path(prior.paired.__file__).resolve(),
        Path(prior.paired.fast.__file__).resolve(), Path(prior.paired.fast.original.__file__).resolve()]
    files += [WORK / ('scripts/analysis/' + name) for name in [
        'ue_compact_joint_rgb_intervention_20261024.py', 'ue_compact_joint_fixed_rgb_paired_20261025.py',
        'ue_compact_joint_checkpoint_audit_20261025.py', 'ue_compact_joint_final_inputs_20261025.py',
        'ue_compact_joint_cost_20261025.py']]
    hashes = {str(path): prior.paired.sha(path) for path in files}
    expected = json.loads((root / 'initialization.json').read_text())['source_hashes']
    def immutable():
        assert all(prior.paired.sha(path) == value for path, value in hashes.items())
        assert all(prior.paired.sha(WORK / path) == value for path, value in expected.items())
    immutable()
    for path in files:
        dest = folder / 'source_v2' / path.relative_to(WORK)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
    prior.dump(folder / 'owner_v2_launch.json', dict(pid=os.getpid(), started=time.time(), tasks=19,
        training_owner_pid=parent, training_owner_start_ticks=start, source_sha256=hashes,
        training_source_hashes=expected, old_scientific_commands_preserved=10))
    prior.dump(folder / 'plan_v2.json', plan)
    while not (root / 'execution_complete.json').exists():
        assert live(parent, start), 'Train/test owner ended before completion; inspect actual tasks'
        immutable()
        prior.dump(folder / 'state_v2.json', dict(stage='waiting_for_live_train_test_owner',
            training_owner_pid=parent, observed=time.time(), tasks=19))
        time.sleep(30)
    complete = json.loads((root / 'execution_complete.json').read_text())
    assert complete['complete'] and complete['tasks'] == 8 and complete['source_hashes'] == expected
    done, failed, running, benchmark_gpu = set(), [], {}, None
    while len(done) < len(plan):
        immutable()
        for name, child in list(running.items()):
            rc = child['process'].poll()
            if rc is None:
                continue
            child['log'].close()
            marker = Path(child['job']['marker'])
            value = json.loads(marker.read_text()) if marker.exists() else {}
            if rc == 0 and value.get(child['job'].get('marker_flag', 'complete')):
                done.add(name)
            else:
                failed.append(dict(name=name, returncode=rc, log=str(child['log_path'])))
            del running[name]
        if not failed:
            cards, apps = gpu_state()
            available = {g: v['free']-2048 for g, v in cards.items()}
            for child in running.values():
                if child['gpu'] is not None:
                    available[child['gpu']] -= max(0, 6160-apps.get(child['process'].pid, {}).get('memory', 0))
            mem = next(int(s.split()[1])//1024 for s in Path('/proc/meminfo').read_text().splitlines()
                       if s.startswith('MemAvailable:')) - 8192
            mem -= sum(4096 for c in running.values() if c['job']['kind'] == 'cpu')
            for job in plan:
                name = job['name']
                if name in done or name in running or not set(job['requires']) <= done:
                    continue
                assert not Path(job['marker']).exists(), 'Unexpected prior output requires review'
                gpu = None
                if job['kind'] in ('gpu', 'cost'):
                    eligible = [g for g, free in available.items() if free >= 6160]
                    if job['kind'] == 'cost':
                        eligible = [g for g in eligible if not any(a['uuid'] == cards[g]['uuid'] for a in apps.values())]
                        if benchmark_gpu is not None:
                            eligible = [g for g in eligible if g == benchmark_gpu]
                    if not eligible:
                        continue
                    gpu = max(eligible, key=available.get)
                    if job['kind'] == 'cost':
                        benchmark_gpu = gpu
                    available[gpu] -= 6160
                elif mem < 4096:
                    continue
                else:
                    mem -= 4096
                command = job['command'] + (['--physical-gpu', str(gpu)] if job['kind'] == 'cost' else [])
                log_path = folder / ('v2_' + name + '.log')
                log = log_path.open('x')
                env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                           CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu))
                child = subprocess.Popen(command, cwd=WORK, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                running[name] = dict(process=child, gpu=gpu, job=job, log=log, log_path=log_path)
                prior.dump(folder / ('v2_' + name + '_launch.json'), dict(pid=child.pid, gpu=gpu, command=command))
        prior.dump(folder / 'state_v2.json', dict(stage='failure_requires_review' if failed else 'postprocessing',
            done=sorted(done), failed=failed, observed=time.time(), benchmark_gpu=benchmark_gpu,
            running={n: dict(pid=c['process'].pid, gpu=c['gpu']) for n, c in running.items()}))
        if failed and not running:
            raise RuntimeError(failed)
        if len(done) < len(plan):
            time.sleep(15)
    costs = {case: json.loads((root / 'joint_cost' / case / 'complete.json').read_text())
             for case in ('satellite', 'rgbd_combined', 'rgbd_original')}
    assert len({c['protocol']['gpu_uuid'] for c in costs.values()}) == 1
    assert costs['rgbd_combined']['protocol']['checkpoint_sha256'] == costs['rgbd_original']['protocol']['checkpoint_sha256']
    a = json.loads((root / 'joint_cost/rgbd_combined/rows.json').read_text())
    b = json.loads((root / 'joint_cost/rgbd_original/rows.json').read_text())
    expected_prefix = {(r['image_id'], r['repetition']): r['retained_prefix_sha256'] for r in a}
    assert len(expected_prefix) == len(b) == 9
    assert all(r['prediction_sha256'] == expected_prefix[(r['image_id'], r['repetition'])] for r in b)
    evidence = {job['marker']: prior.paired.sha(job['marker']) for job in plan}
    prior.dump(folder / 'complete_v2.json', dict(complete=True, tasks=19, source_sha256=hashes,
        artifact_sha256=evidence, cost_retained_prefix_exact=True, benchmark_gpu=benchmark_gpu,
        scope='engineering completion of fixed reports/audits and three selected-weight cost cases',
        final_scientific_review_still_required=True))


if __name__ == '__main__':
    main()
