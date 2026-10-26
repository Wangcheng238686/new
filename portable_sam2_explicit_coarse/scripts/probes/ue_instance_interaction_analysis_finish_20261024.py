"""Verify CPU handoff, then invoke the unchanged full scientific final audit."""
import argparse
import json
import os
import subprocess
import time

import ue_instance_interaction_capacity_owner_20261024 as cap
old = cap.old


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--owner-pid', type=int, required=True)
    args = p.parse_args()
    root = old.OUT
    initial = cap.identity(args.owner_pid)
    assert initial and 'ue_instance_interaction_analysis_owner_20261024.py' in initial['command']
    while not (root / 'owner_complete.json').exists():
        actual = cap.identity(args.owner_pid)
        assert actual and actual['ticks'] == initial['ticks']
        state = json.loads((root / 'state.json').read_text())
        # First state may briefly still belong to the stopped previous owner.
        assert state['stage'] != 'failure_requires_audit', state.get('failed')
        old.dump(root / 'finish_state.json', dict(stage='waiting_verified_analysis_owner',
            owner_pid=args.owner_pid, finish_pid=os.getpid(), done=len(state['done']), observed=time.time()))
        time.sleep(20)
    handoff = json.loads((root / 'analysis_capacity_handoff.json').read_text())
    assert handoff['new_owner_pid'] == args.owner_pid and handoff['original_jobs'] == 119
    assert old.build_jobs(root) == json.loads((root / 'job_plan.json').read_text())
    for name, sha in handoff['source_hashes'].items():
        assert cap.sha(old.WORK / name) == cap.sha(root / 'source' / name) == sha
    old.dump(root / 'analysis_capacity_final_audit.json', dict(passed=True, original119commands_exact=True,
        handoff_sha256=cap.sha(root / 'analysis_capacity_handoff.json'), source_hashes=handoff['source_hashes']))
    subprocess.run([old.PYTHON, '-u', 'scripts/probes/ue_instance_interaction_finish_20261024.py',
        '--owner-pid', str(args.owner_pid), '--audit-only'], cwd=old.WORK, check=True)
    done = json.loads((root / 'finish_complete.json').read_text())
    done['analysis_capacity_final_audit_sha256'] = cap.sha(root / 'analysis_capacity_final_audit.json')
    done['analysis_finish_source_sha256'] = cap.sha(__file__)
    old.dump(root / 'finish_complete.json', done)
    print('CPU CAPACITY BRIDGE FINAL AUDIT PASS', flush=True)


if __name__ == '__main__':
    main()
