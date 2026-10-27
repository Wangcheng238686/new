"""Preserve and retire ONLY the two old UBR jobs for a fresh fast comparison.

Run after reviewed implementation readiness or full efficiency acceptance. No deletion,
checkpoint rewriting, unrelated process signals, or task-weight warm start.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

WORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORK / 'scripts/probes'))
import ue_ubr_short_owner_20261027 as owner


def require_identity(identity, tokens):
    actual = owner.proc_identity(identity['pid'])
    if not actual or str(actual['start_ticks']) != str(identity['start_ticks']):
        raise RuntimeError('Process identity changed: ' + str(identity))
    if not all(t in actual['argv'] for t in tokens):
        raise RuntimeError('Process command does not match this study')
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-root', type=Path, required=True)
    receipts = parser.add_mutually_exclusive_group(required=True)
    receipts.add_argument('--efficiency-receipt', type=Path)
    receipts.add_argument('--implementation-receipt', type=Path)
    parser.add_argument('--execute-stop', action='store_true')
    args = parser.parse_args()
    root = args.old_root.resolve()
    receipt = args.efficiency_receipt or args.implementation_receipt
    gate = json.loads(receipt.read_text())
    if args.efficiency_receipt:
        if not gate.get('passed') or gate.get('world_size') != 4:
            raise RuntimeError('A passed four-GPU efficiency gate is required first')
    else:
        if not gate.get('ready_to_retire_old_jobs') or not gate.get('independent_review_complete'):
            raise RuntimeError('Reviewed implementation readiness is required')
        for name, expected in gate['source_sha256'].items():
            if owner.sha(owner.REPO / name) != expected:
                raise RuntimeError('Implementation changed since readiness review')
    state = json.loads((root / 'owner_state.json').read_text())
    arms = ('G_UBR_geometry', 'G_UBR_zero_geometry')
    selected = []
    for arm in arms:
        job = state['jobs']['train_' + arm]
        require_identity(job['launcher'], [str(owner.RUN), str(root), arm, 'train'])
        if os.getpgid(job['launcher']['pid']) != job['launcher']['pid']:
            raise RuntimeError('Launcher is not an independent process-group leader')
        ranks = owner.rank_records(job['rank_launch_record'], job['launcher']['start_ticks'])
        if len(ranks) != 4 or not all(r['currently_live'] for r in ranks):
            raise RuntimeError('Expected four live ranks before requested retirement')
        selected.append(dict(arm=arm, launcher=job['launcher'], ranks=ranks))
    require_identity(state['owner'], [str(root)])
    if not any('ue_ubr_short_adopt_owner_' in x for x in state['owner']['argv']):
        raise RuntimeError('Unexpected old owner command')
    baseline = json.loads((root / 'G_no_tail/training_complete.json').read_text())
    checkpoint = root / 'G_no_tail/epoch10.pth'
    if baseline['updates'] != 2670 or baseline['checkpoint_sha256'] != owner.sha(checkpoint):
        raise RuntimeError('Completed reference checkpoint audit failed')
    # A source-inventory omission caused the old control failure. Accept only
    # after all actual training source bytes match BOTH current files and Git.
    for name, expected in baseline['contract']['source_hashes'].items():
        actual = owner.sha(owner.REPO / name)
        historical = hashlib.sha256(owner.git('show', baseline['contract']['git_commit'] + ':' + name)).hexdigest()
        if actual != expected or historical != expected:
            raise RuntimeError('Old reference numerical source drift: ' + name)
    report = dict(user_requested_fresh_three_arm_restart=True,
        numerical_sources_unchanged=True, completed_baseline_valid=True,
        old_baseline_checkpoint_sha256=owner.sha(checkpoint),
        readiness_receipt=str(receipt.resolve()),
        readiness_receipt_sha256=owner.sha(receipt),
        final_efficiency_gate_required_before_new_training=True,
        old_owner=state['owner'], selected_jobs=selected,
        controller_sha256=owner.sha(__file__), requested_stop=bool(args.execute_stop))
    if not args.execute_stop:
        print(json.dumps(report, indent=2))
        return
    audit = root / 'retirement_for_fast_restart_20261027'
    audit.mkdir(exist_ok=False)
    for name in ('owner_state.json', 'owner_manifest.json', 'owner_job_plan.json', 'study_contract.json'):
        shutil.copyfile(root / name, audit / name)
    for arm in arms + ('G_no_tail',):
        for name in ('progress.json', 'contract.json', 'training_complete.json'):
            path = root / arm / name
            if path.exists():
                target = audit / arm / name
                target.parent.mkdir(exist_ok=True)
                shutil.copyfile(path, target)
    owner.atomic(audit / 'audit_before_stop.json', report)
    # The supervisor must not launch any new job during retirement.
    os.kill(state['owner']['pid'], signal.SIGTERM)
    for job in selected:
        require_identity(job['launcher'], [str(owner.RUN), str(root), job['arm']])
        os.killpg(job['launcher']['pid'], signal.SIGTERM)
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        remaining = [r for j in selected for r in j['ranks']
                     if owner.start_ticks(r['pid']) == r['start_ticks']]
        if not remaining:
            break
        time.sleep(.5)
    else:
        owner.atomic(audit / 'stop_incomplete.json', dict(remaining=remaining))
        raise RuntimeError('Some verified old ranks still exist; audit before further signals')
    for job in selected:
        state['jobs']['train_' + job['arm']]['status'] = 'stopped_for_requested_efficiency_restart'
    state['jobs']['train_G_no_tail']['status'] = 'complete_independently_audited'
    state.update(stage='retired_for_requested_fast_restart', updated=time.time(),
                 retirement_audit=str(audit), historical_control_failures=state['failures'])
    owner.atomic(root / 'owner_state.json', state)
    owner.atomic(audit / 'complete.json', dict(stopped=True, old_ranks_absent=True,
        all_original_outputs_preserved=True, selected_jobs=selected))
    print(json.dumps(dict(retired=str(root), audit=str(audit), stopped=True)))


if __name__ == '__main__':
    main()
