"""Foreground evidence review receipt for the first matched joint budget.

This records why an experiment is justified, not that its hypotheses succeeded.
"""
import json
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(WORK), str(WORK / 'scripts/probes')]
import ue_compact_joint_run_20261024 as run


def read(path):
    return json.loads(path.read_text())


def main():
    old, root = run.core.OUT, run.ROOT
    final = read(old / 'finish_complete.json')
    bindings = [('final_audit_sha256', 'final_audit.json'),
        ('additional_audit_sha256', 'additional_identity_audit.json'),
        ('summary_sha256', 'study_summary.json'), ('report_sha256', 'RESULTS.md'),
        ('analysis_capacity_final_audit_sha256', 'analysis_capacity_final_audit.json')]
    assert final['complete']
    for key, name in bindings:
        assert run.core.C.file_sha(old / name) == final[key]
    audit = read(old / 'final_audit.json')
    extra = read(old / 'additional_identity_audit.json')
    capacity = read(old / 'analysis_capacity_final_audit.json')
    assert audit['passed'] and audit['all_jobs'] == 119 and len(audit['checkpoints']) == 14
    assert len(audit['actual_evaluations']) == 45 and audit['source_unchanged']
    assert audit['common_shared_initialization_exact'] and audit['paired_training_order']
    assert audit['disabled_strong_satellite_exact']
    assert all(row['actual_optimizer_steps'] == [2670] for row in audit['checkpoints'])
    assert extra['passed'] and extra['input_cache_counts'] == dict(native=3060, anchors=3060)
    assert len(extra['checkpoint_bindings']) == 14 and len(extra['evaluation_bindings']) == 45
    assert capacity['passed'] and capacity['original119commands_exact']
    summary = read(old / 'study_summary.json')
    assert summary['complete'] and len(summary['comparisons']) == 60
    metrics = {}
    for name, values in summary['comparisons'].items():
        assert set(values) == {'MaskAP', 'BBoxAP', 'BoundaryAP_2pct', 'BoundaryAP_1pct'}
        raw = read(old / 'analysis' / f'{name}.json')
        checks = raw['accumulation_equivalence']['reference_checks']
        assert len(checks) == 24 and max(row['max_abs_error'] for row in checks) < 1e-12
        assert all(v['bootstrap_samples'] == 500 and v['seed'] == 20261021 for v in values.values())
    for name in ('rgb_internal_net', 'rgb_internal_content', 'rgb_internal_content_satellite',
                 'rgb_internal_constant_rgb_input', 'rgb_internal_donor_rgb_input',
                 'rgb_internal_depth_values', 'rgb_internal_depth_values_satellite'):
        metrics[name] = summary['comparisons'][name]['MaskAP']
    assert metrics['rgb_internal_net']['delta'] >= .002
    for name in ('rgb_internal_net', 'rgb_internal_content', 'rgb_internal_content_satellite',
                 'rgb_internal_constant_rgb_input'):
        assert metrics[name]['delta'] > 0 and metrics[name]['paired_image_bootstrap95'][0] > 0
    expected = run.sources()
    engineering = read(root / 'engineering_complete.json')
    assert engineering['passed'] and engineering['source_hashes'] == expected
    for directory in ('loss_direction_audit', 'compact_cost_audit'):
        value = read(old / directory / 'complete.json')
        assert value['complete']
        for name, sha in value['source_hashes'].items():
            assert run.core.C.file_sha(WORK / name) == sha
    evidence_paths = [old / name for _, name in bindings] + [old / 'finish_complete.json',
        old / 'loss_direction_audit/complete.json', old / 'compact_cost_audit/complete.json',
        root / 'engineering_complete.json', root / 'initialization.json',
        root / 'amp_retry_gate/complete.json', root / 'input_audit.json', Path(__file__).resolve()]
    receipt = dict(decision='proceed_to_matched_joint_training', source_hashes=expected,
        evidence_sha256={str(path): run.core.C.file_sha(path) for path in evidence_paths},
        reviewed_old_scope=dict(training_arms=14, evaluations=45, paired_analyses=60, fixed_jobs=119,
                                source_input_caches_rehashed=6120),
        key_MaskAP_evidence=metrics,
        rationale='RGB content and original-path effects plus net benefit justify testing common-foundation joint learning. Compact source removes unused deep stages. It does not guarantee joint gains.',
        caveats=[
            'Only one RGB training seed; nominal image bootstrap is not training variance or spatial generalization.',
            'Correct over donor RGB interval crosses zero; precise paired local appearance benefit not yet stable.',
            'Explicit four-field value contribution in RGB internal is unconfirmed; it is not a no-depth experiment.',
            'Reference-plane control must test all measured-depth effects, with correspondences/support rebuilt.',
            'Fixed K/relative rig prevents an informative cross-image calibration shuffle; K/T act as geometry parameters.',
            'Compact old inference still about 2x SAT evaluation wall with more decoded instances and input I/O; final joint cost must be remeasured.',
            'Old eval-mode loss gradients do not justify changing center weight before this matched budget.',
            'v1 test development exposure and same-city scope remain; test must not choose new epoch/structure.'
        ],
        next_budget=run.PROTOCOL, task_warm_start=False,
        success_still_unproven=['joint Mask AP net gain', 'RGB marginal value in joint training',
                               'measured-depth marginal value', 'training-seed stability'])
    run.core.C.dump(old / 'foreground_review.json', dict(passed=True, completed_scientific_scope=True,
        full_goal_complete=False, evidence_sha256=receipt['evidence_sha256'],
        scope='Old frozen mechanism study only; the final common-foundation joint goal remains active'))
    run.core.C.dump(root / 'scientific_launch_review.json', receipt)
    print('SCIENTIFIC REVIEW: PROCEED TO MATCHED JOINT TRAINING', flush=True)


if __name__ == '__main__':
    main()
