#!/usr/bin/env python3
"""Four-card, source-pinned owner for the independent fast three-arm G/UBR study.

No single-card fallback. --detach creates a new owner session; normal mode is
suitable for a supervised terminal. Existing state always requires human audit,
never blind restart. No signals are sent to external or already running jobs.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

WORK = Path(__file__).resolve().parents[2]
REPO = WORK.parent
RUN = WORK / 'scripts/probes/ue_ubr_fast_run_20261027.py'
CONFIG = WORK / 'configs/ablations/ue_g_ubr_fast_protocol_20261027.json'
PYTHON = '/data/wangcheng/envs/cvt2/bin/python'
PRIORITY = ('G_UBR_geometry', 'G_no_tail', 'G_UBR_zero_geometry')
GPUS = (0, 1, 2, 3)
EFFICIENCY_RECEIPT = None
PEAK_MIB = 6530
SLACK_MIB = 0  # retain global2GiB margin, actual measured peak plus per-process context
CONTEXT_MIB = 512
RESERVATION_MIB = PEAK_MIB + SLACK_MIB + CONTEXT_MIB
MARGIN_MIB = 2048


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def start_ticks(pid):
    try:
        return Path('/proc/%d/stat' % pid).read_text().rsplit(') ', 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def proc_identity(pid):
    tick = start_ticks(pid)
    if tick is None:
        return None
    try:
        argv = Path('/proc/%d/cmdline' % pid).read_bytes().decode().rstrip('\0').split('\0')
    except (FileNotFoundError, ProcessLookupError):
        return None
    return dict(pid=pid, start_ticks=tick, argv=argv)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO)


def prepare_sources(receipt):
    # CPU import closure is not sufficient for lazy vendored SAM2 imports.
    # Union the BASE gate and the FAST gate's real built-model full-REPO
    # closure with new process-local runner/owner/config sources BEFORE launch.
    sys.path[:0] = [str(WORK), str(WORK / 'scripts/probes')]
    import ue_ubr_fast_run_20261027 as runner
    runner.configure(EFFICIENCY_RECEIPT)
    source = dict(runner.engine.source_inventory())
    efficiency = json.loads(EFFICIENCY_RECEIPT.read_text())
    for evidence in (receipt, efficiency):
        for name, expected in runner.normalize_sources(evidence).items():
            full = REPO / name
            if sha(full) != expected:
                raise RuntimeError('Gate source changed: ' + name)
            if name in source and source[name] != expected:
                raise RuntimeError('Conflicting source provenance: ' + name)
            source[name] = expected
    for name in runner.NEW_SOURCES:
        full = WORK / name
        source[str(full.relative_to(REPO))] = sha(full)
    source[str(Path(__file__).resolve().relative_to(REPO))] = sha(__file__)
    # This verifies passed gate and all numerical critical files, including the
    # two lazy SAM2 dependencies the original owner omitted.
    runner.efficiency_check(EFFICIENCY_RECEIPT, source)
    commit = git('rev-parse', 'HEAD').decode().strip()
    for name, expected in source.items():
        if hashlib.sha256(git('show', commit + ':' + name)).hexdigest() != expected:
            raise RuntimeError('Commit exact numerical/control source before launch: ' + name)
    return commit, source


def rank_records(path, launcher_started):
    if not Path(path).exists():
        return []
    record = json.loads(Path(path).read_text())
    ranks = record.get('ranks', [])
    if record.get('world_size') != 4 or len(ranks) != 4:
        raise RuntimeError('Invalid four-rank launch receipt: ' + str(path))
    if {int(row['local_rank']) for row in ranks} != set(GPUS):
        raise RuntimeError('Rank launch receipt must name local ranks 0..3')
    for row in ranks:
        actual = proc_identity(int(row['pid']))
        row['currently_live'] = bool(actual and str(actual['start_ticks']) == str(row['start_ticks']))
        if row['currently_live']:
            # Start ticks prevent PID reuse; command owns this exact study root.
            row['actual_argv'] = actual['argv']
            if str(RUN) not in actual['argv']:
                raise RuntimeError('Rank PID command is not this runner')
            if int(row['start_ticks']) < int(launcher_started):
                raise RuntimeError('Stale rank receipt predates this launcher')
    return ranks


def resources(jobs):
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
        '--format=csv,noheader,nounits'], text=True)
    cards = {}
    for line in raw.splitlines():
        index, uuid, free = [x.strip() for x in line.split(',')]
        if int(index) in GPUS:
            cards[int(index)] = dict(uuid=uuid, actual_free_mib=int(free),
                                      admissible_free_mib=int(free) - MARGIN_MIB)
    if set(cards) != set(GPUS):
        raise RuntimeError('Exactly physical GPU indices 0..3 must be available')
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
        '--format=csv,noheader,nounits'], text=True)
    usage = {}
    for line in raw.splitlines():
        fields = [x.strip() for x in line.split(',')]
        if len(fields) == 3 and fields[1].isdigit() and fields[2].isdigit():
            usage[(fields[0], int(fields[1]))] = int(fields[2])
    for job in jobs.values():
        if job['status'] != 'running':
            continue
        ranks = rank_records(job['rank_launch_record'], job['launcher']['start_ticks'])
        job['ranks'] = ranks
        for index, card in cards.items():
            pid = next((int(row['pid']) for row in ranks
                        if int(row['local_rank']) == index and row['currently_live']), None)
            observed = usage.get((card['uuid'], pid), 0)
            # nvidia-smi free already excludes actual residency. Reserve only the
            # unmaterialized part, including before real ranks have started.
            card['admissible_free_mib'] -= max(0, RESERVATION_MIB - observed)
    return cards


def command(args, action, arm):
    return [PYTHON, '-u', '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
        '--nproc_per_node=4', str(RUN), action, '--arm', arm, '--root', str(args.root),
        '--epochs', '10', '--engineering-receipt', str(args.engineering_receipt),
        '--allocator-cap-mib', str(args.allocator_cap_mib),
        '--efficiency-receipt', str(args.efficiency_receipt), '--split',
        'test' if action == 'eval' else 'val']


def validate_marker(job, source, commit):
    marker = Path(job['marker'])
    if not marker.exists():
        return False, 'missing completion marker'
    row = json.loads(marker.read_text())
    if row.get('complete') is not True or row.get('world_size') != 4:
        return False, 'incomplete or non-four-card result'
    if row.get('arm') != job['arm']:
        return False, 'arm mismatch'
    if job['action'] == 'train':
        if row.get('epoch') != 10 or row.get('updates') != 2670:
            return False, 'fixed budget mismatch'
        cp = marker.parent / 'epoch10.pth'
        if sha(cp) != row.get('checkpoint_sha256'):
            return False, 'checkpoint digest mismatch'
        contract = row['contract']
        if contract.get('git_commit') != commit:
            return False, 'checkpoint source commit mismatch'
        if any(source.get(name) != expected for name, expected in contract['source_hashes'].items()):
            return False, 'training source contract outside pinned owner inventory'
    else:
        if row.get('split') != 'test' or row.get('images') != 459 or row.get('precision') != 'FP32':
            return False, 'full459 FP32 test contract mismatch'
        # Evaluation checkpoint identity is compared by caller using the training marker.
    return True, None


def run(args):
    args.root.mkdir(parents=True, exist_ok=True)
    lock = (args.root / 'owner.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('A live owner holds owner.lock; refusing double scheduling')
    if (args.root / 'owner_state.json').exists():
        raise RuntimeError('Prior state needs explicit recovery audit, including live child process groups; no blind restart')
    receipt = json.loads(args.engineering_receipt.read_text())
    if receipt.get('passed') is not True or receipt.get('world_size') != 4 or len(receipt.get('ranks', [])) != 4:
        raise RuntimeError('Passed real four-rank engineering receipt required')
    if max(row.get('max_natural_active', 0) for row in receipt['ranks']) < 64:
        raise RuntimeError('Engineering receipt does not cover natural 64-active resource case')
    if max(row['inference250']['peak_reserved_mib'] for row in receipt['ranks']) > PEAK_MIB:
        raise RuntimeError('Original engineering peak exceeds fast-gate reservation; re-audit')
    commit, source = prepare_sources(receipt)
    # Original order files are part of the reproducibility contract and never altered.
    order_root = Path('/data/wangcheng/checkpoint/ue_compact_joint_20261024/rgbd')
    orders = {str(order_root / ('epoch%d.json' % epoch)): sha(order_root / ('epoch%d.json' % epoch))
              for epoch in range(1, 11)}
    protocol = json.loads(CONFIG.read_text())
    if protocol['epochs'] != 10 or protocol['world_size'] != 4 or set(protocol['arms']) != set(PRIORITY):
        raise RuntimeError('Owner is pinned to fixed10epoch three-arm four-card study')
    jobs = {}
    for action in ('train', 'eval'):
        for arm in PRIORITY:
            name = action + '_' + arm
            marker = (args.root / arm / 'training_complete.json' if action == 'train' else
                      args.root / 'evaluation' / 'test' / arm / 'complete.json')
            jobs[name] = dict(status='pending', action=action, arm=arm,
                argv=command(args, action, arm), marker=str(marker),
                dependencies=[] if action == 'train' else ['train_' + a for a in PRIORITY],
                rank_launch_record=str(args.root / 'rank_launches' /
                    ('%s_%s_%s.json' % (action, arm, 'val' if action == 'train' else 'test'))))
    owner = proc_identity(os.getpid())
    manifest = dict(owner=owner, created=time.time(), git_commit=commit,
        source_sha256=source, engineering_receipt=str(args.engineering_receipt),
        engineering_sha256=sha(args.engineering_receipt),
        efficiency_receipt=str(args.efficiency_receipt), efficiency_sha256=sha(args.efficiency_receipt),
        source_closure='base+fast built-model FULL REPO sources plus current runner/config/control; no WORK-only filter',
        source_order_sha256=orders,
        protocol=protocol, GPUs=list(GPUS), world_size_per_experiment=4,
        per_card_reservation_mib=RESERVATION_MIB,
        reservation_components=dict(measured_peak=PEAK_MIB, slack=SLACK_MIB, cuda_context=CONTEXT_MIB),
        allocator_cap_mib=args.allocator_cap_mib, all_card_margin_mib=MARGIN_MIB,
        max_concurrent_groups=args.max_groups, evaluation_concurrency=1,
        schedule='geometry then no_tail first, zero_geometry next; all training before serial fixedE10 fulltest',
        exposed_test='descriptive development-test only; no model/epoch/budget selection')
    atomic(args.root / 'owner_manifest.json', manifest)
    atomic(args.root / 'owner_job_plan.json', jobs)
    for name in source:
        dst = args.root / 'owner_frozen_source' / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, dst)
    children, streams = {}, {}
    failures = []
    while True:
        for name, child in list(children.items()):
            rc = child.poll()
            if rc is None:
                continue
            job = jobs[name]
            streams.pop(name).close()
            del children[name]
            try:
                accepted, reason = (validate_marker(job, source, commit) if rc == 0 else (False, 'nonzero child returncode'))
            except Exception as exc:
                accepted, reason = False, 'Completion audit failed: ' + repr(exc)
            if accepted and job['action'] == 'eval':
                marker = json.loads(Path(job['marker']).read_text())
                trained = json.loads((args.root / job['arm'] / 'training_complete.json').read_text())
                accepted = marker['checkpoint_sha256'] == trained['checkpoint_sha256']
                reason = None if accepted else 'eval checkpoint is not fixedE10'
            job.update(status='complete' if accepted else 'failed', returncode=rc,
                       completed=time.time(), completion_marker_sha256=sha(job['marker']) if Path(job['marker']).exists() else None,
                       failure_reason=reason)
            if not accepted:
                failures.append(name + ': ' + str(reason))
            print('END', name, rc, job['status'], flush=True)
        if not failures:
            try:
                if any(sha(REPO / name) != expected for name, expected in source.items()):
                    raise RuntimeError('Pinned source bytes changed')
                if any(sha(path) != expected for path, expected in orders.items()):
                    raise RuntimeError('Original sample-order bytes changed')
                if sha(args.engineering_receipt) != manifest['engineering_sha256']:
                    raise RuntimeError('Engineering receipt changed')
                if sha(args.efficiency_receipt) != manifest['efficiency_sha256']:
                    raise RuntimeError('Efficiency receipt changed')
                study = args.root / 'study_contract.json'
                actual_commit = json.loads(study.read_text())['git_commit'] if study.exists() else git('rev-parse', 'HEAD').decode().strip()
                if actual_commit != commit:
                    raise RuntimeError('Study source commit differs from pinned owner commit')
            except Exception as exc:
                failures.append(str(exc))
        try:
            cards = resources(jobs)
        except Exception as exc:
            # A telemetry/proc error halts admission, never kills healthy jobs.
            failures.append('Resource/rank audit failed: ' + repr(exc))
            cards = {}
        if not failures:
            for name, job in jobs.items():
                if job['status'] != 'pending' or any(jobs[d]['status'] != 'complete' for d in job['dependencies']):
                    continue
                if len(children) >= args.max_groups:
                    break
                if job['action'] == 'eval' and children:
                    continue  # fulltest runs serially after ALL three trainings
                if any(cards[g]['admissible_free_mib'] < RESERVATION_MIB for g in GPUS):
                    continue
                log = args.root / 'logs' / (name + '.log')
                log.parent.mkdir(exist_ok=True)
                stream = log.open('x')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3', OPENBLAS_NUM_THREADS='1',
                           OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1',
                           UE_UBR_FAST_SOURCE_COMMIT=commit)
                try:
                    child = subprocess.Popen(job['argv'], cwd=WORK, env=env, stdin=subprocess.DEVNULL,
                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                except Exception as exc:
                    stream.close()
                    job.update(status='failed', returncode=None, failure_reason='Popen failed: ' + repr(exc))
                    failures.append(name + ': Popen failed: ' + repr(exc))
                    break
                launcher = proc_identity(child.pid)
                if launcher is None:
                    # Preserve child for poll/returncode; do not relaunch it.
                    launcher = dict(pid=child.pid, start_ticks='0', argv=job['argv'])
                children[name], streams[name] = child, stream
                job.update(status='running', launcher=launcher, launched=time.time(),
                           log=str(log), GPU_indices=list(GPUS), returncode=None, ranks=[])
                atomic(args.root / 'owner_launches' / (name + '.json'), job)
                for card in cards.values():
                    card['admissible_free_mib'] -= RESERVATION_MIB
                print('START', name, 'four GPUs', child.pid, flush=True)
        complete = all(job['status'] == 'complete' for job in jobs.values())
        stage = 'complete' if complete else ('failure_requires_audit' if failures else 'running')
        atomic(args.root / 'owner_state.json', dict(stage=stage, owner=owner,
            updated=time.time(), jobs=jobs, failures=failures, GPU_headroom=cards,
            running_groups=len(children), source_git_commit=commit))
        if complete:
            atomic(args.root / 'owner_complete.json', dict(complete=True, jobs=len(jobs),
                git_commit=commit, completed=time.time(), scientific_review_required=True,
                no_automatic_claim_of_AP_gain=True))
            return
        if failures and not children:
            raise RuntimeError('Owner stopped NEW launches; kept existing children alive until exit: ' + repr(failures))
        time.sleep(15)


def main():
    global EFFICIENCY_RECEIPT, PEAK_MIB, RESERVATION_MIB
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--engineering-receipt', type=Path, required=True)
    p.add_argument('--efficiency-receipt', type=Path, required=True)
    p.add_argument('--allocator-cap-mib', type=int, default=0)
    p.add_argument('--max-groups', type=int, default=3, choices=[1, 2, 3],
                   help='three train arms maximum; actual all-GPU memory admission decides')
    p.add_argument('--detach', action='store_true')
    args = p.parse_args()
    args.root = args.root.resolve()
    args.engineering_receipt = args.engineering_receipt.resolve(strict=True)
    args.efficiency_receipt = args.efficiency_receipt.resolve(strict=True)
    EFFICIENCY_RECEIPT = args.efficiency_receipt
    efficiency = json.loads(EFFICIENCY_RECEIPT.read_text())
    if efficiency.get('passed') is not True or efficiency.get('world_size') != 4 or len(efficiency.get('ranks', [])) != 4:
        p.error('PASSED four-card real efficiency receipt required')
    measured = efficiency.get('max_peak_reserved_mib')
    if not isinstance(measured, (int, float)) or measured <= 0:
        p.error('efficiency receipt must publish max_peak_reserved_mib')
    import math
    PEAK_MIB = max(6530, math.ceil(measured))
    RESERVATION_MIB = PEAK_MIB + CONTEXT_MIB + SLACK_MIB
    if args.allocator_cap_mib == 0:
        args.allocator_cap_mib = max(9216, PEAK_MIB + 1024)
    if not PEAK_MIB <= args.allocator_cap_mib <= 24576:
        p.error('allocator cap must cover measured gate peak and fit the3090')
    if args.detach:
        args.root.mkdir(parents=True, exist_ok=True)
        log = args.root / 'owner.log'
        argv = [PYTHON, '-u', str(Path(__file__).resolve()), *[a for a in sys.argv[1:] if a != '--detach']]
        with log.open('a') as stream:
            child = subprocess.Popen(argv, cwd=WORK, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps(dict(detached_owner=proc_identity(child.pid), log=str(log), argv=argv)))
        return
    run(args)


if __name__ == '__main__':
    main()
