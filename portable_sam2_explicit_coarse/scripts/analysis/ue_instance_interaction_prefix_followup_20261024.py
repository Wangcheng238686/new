"""Fixed-E10 original-path input dependency, using existing predictions only.

Triggered by an observed decomposition inconsistency; exploratory mechanism
analysis, not a newly selected training arm or test-driven model change.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

WORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORK / 'scripts/probes'))
import ue_instance_interaction_owner_20261024 as owner


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = owner.OUT
    arm = root / 'seed44_geom_internal_geometry_correct'
    out = root / 'prefix_input_followup'
    out.mkdir(exist_ok=True)
    cp_sha = sha(arm / 'last.pth')
    sources = {name: sha(WORK / name) for name in (
        'scripts/analysis/ue_instance_interaction_prefix_followup_20261024.py',
        'scripts/probes/ue_joint_encoding_fast_analysis.py',
        'scripts/probes/ue_joint_encoding_analysis.py', 'utils/coco_eval_utils.py')}
    paths = {}
    for mode in ('geometry', 'constant', 'donor'):
        directory = arm / 'test' / mode
        identity = json.loads((directory / 'complete.json').read_text())
        assert identity['complete'] and identity['images'] == 459
        assert identity['checkpoint_sha256'] == cp_sha
        paths[mode] = directory / 'satellite_records.json'
    provenance = dict(checkpoint_sha256=cp_sha, images=459, source_hashes=sources,
        input_record_sha256={key: sha(path) for key, path in paths.items()},
        purpose='Explore whether correct geometry hurts the conditioned original satellite path while helping added candidates',
        intervention='constant fields retain support; donor changes fields+support; both are fixed-weight input interventions',
        scope='post-observation mechanism follow-up; original 14 arms/119 jobs/model selection remain unchanged')
    owner.dump(out / 'protocol.json', provenance)
    for mode in ('constant', 'donor'):
        target = out / f'correct_minus_{mode}.json'
        if not target.exists():
            with (out / f'{mode}.log').open('a') as log:
                subprocess.run([owner.PYTHON, '-u', 'scripts/probes/ue_joint_encoding_fast_analysis.py',
                    '--baseline', str(paths[mode]), '--candidate', str(paths['geometry']),
                    '--output', str(target), '--split', 'test', '--bootstrap', '500'],
                    cwd=WORK, stdout=log, stderr=subprocess.STDOUT,
                    env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1'), check=True)
        result = json.loads(target.read_text())
        assert len(result['metrics']) == 4
        checks = result['accumulation_equivalence']['reference_checks']
        assert len(checks) == 24 and max(check['max_abs_error'] for check in checks) < 1e-12
        assert all(value['bootstrap_samples'] == 500 and value['seed'] == 20261021
                   for value in result['metrics'].values())
    assert all(sha(WORK / name) == expected for name, expected in sources.items())
    assert all(sha(path) == provenance['input_record_sha256'][mode] for mode, path in paths.items())
    assert sha(arm / 'last.pth') == cp_sha
    owner.dump(out / 'complete.json', dict(complete=True, original_weights_and_records_unchanged=True,
        comparisons=['correct_minus_constant', 'correct_minus_donor'], source_hashes=sources))
    print('PREFIX INPUT FOLLOWUP COMPLETE', flush=True)


if __name__ == '__main__':
    main()
