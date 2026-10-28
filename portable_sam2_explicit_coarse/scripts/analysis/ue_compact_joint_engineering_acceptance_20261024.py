"""Bind completed compact-model engineering gates to the current runner source."""
import argparse
import json
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(WORK), str(WORK / 'scripts/probes')]
import ue_compact_joint_run_20261024 as run


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=run.ROOT)
    args = p.parse_args()
    root = args.root
    expected = run.sources()
    evidence = {}
    for folder in ('runtime_gate', 'optimizer_resume_gate'):
        path = root / folder / 'complete.json'
        value = read(path)
        assert value['complete'] and value['source_hashes'] == expected
        evidence[folder] = run.core.C.file_sha(path)
    runtime = read(root / 'runtime_gate/complete.json')
    assert len(runtime['inference_cases']) == 4 and not runtime['reference_plane_measured_file_reads']
    for arm in run.PROTOCOL['arms']:
        metrics = read(root / 'runtime_gate' / arm / 'metrics.json')
        assert 'segm/AR@250' in metrics and 'bbox/AR@250' in metrics
        assert not any('AR@100' in name for name in metrics), 'Recall slot mislabeled'
    assert next(row for row in runtime['inference_cases'] if row['arm'] == 'rgbd')['forced_FP32_masks'] == 250
    assert runtime['restore_all_five_rng_exact'] and runtime['actual_reference_plane_accumulation'] == 8
    assert runtime['gate_source_sha256'] == run.core.C.file_sha(WORK / 'scripts/analysis/ue_compact_joint_runtime_gate_20261024.py')
    opt = read(root / 'optimizer_resume_gate/complete.json')
    assert opt['all_five_rng_exact'] and opt['model_optimizer_scaler_exact_before_resumed_update']
    assert opt['actual_resumed_update'] == 2 and opt['AMP_scale'] == 512
    for name, sha in opt['gate_source_hashes'].items():
        assert run.core.C.file_sha(name) == sha
    amp_path = root / 'amp_retry_gate/complete.json'
    amp = read(amp_path)
    assert amp['complete'] and amp['source_hashes'] == expected
    assert amp['retries'] == 1 and amp['optimizer_step_calls'] == amp['actual_updates'] == 1
    assert amp['same_augmented_batches'] and amp['same_parameters_before_retry']
    assert amp['final_AMP_scale'] == 256 and amp['loss_forward_max_abs_difference'] <= 1e-5
    assert amp['gate_source_sha256'] == run.core.C.file_sha(WORK / 'scripts/analysis/ue_compact_joint_amp_retry_gate_20261024.py')
    evidence['forced_AMP_retry'] = run.core.C.file_sha(amp_path)
    inputs = read(root / 'input_audit.json')
    assert inputs['passed'] and inputs['source_hashes'] == expected and inputs['reference_files'] == 3060
    assert inputs['actual_cache_and_camera_SHA_all_match']
    assert inputs['reference_manifest_sha256'] == run.core.C.file_sha(run.PLANE / 'manifest.json')
    assert inputs['build_complete_sha256'] == run.core.C.file_sha(run.PLANE / 'BUILD_COMPLETE.json')
    assert inputs['audit_source_sha256'] == run.core.C.file_sha(WORK / 'scripts/analysis/ue_compact_joint_input_audit_20261024.py')
    evidence['input_audit'] = run.core.C.file_sha(root / 'input_audit.json')
    foundation_path = run.core.OUT / 'compact_joint_foundation_gate_v2/complete.json'
    foundation = read(foundation_path)
    assert foundation['complete'] and not foundation['task_checkpoint_loaded']
    assert foundation['train_instances'] == 208 and len(foundation['steps']) == 3
    for name, sha in foundation['source_hashes'].items():
        assert run.core.C.file_sha(WORK / name) == sha
    init = read(root / 'initialization.json')
    assert not init['task_weights_loaded'] and init['source_hashes'] == expected
    assert init['satellite_initial_sha'] == foundation['common_satellite_initial_sha']
    assert init['common_init_sha256'] == run.core.C.file_sha(root / 'common_init.pth')
    evidence['foundation_backward'] = run.core.C.file_sha(foundation_path)
    p0_path = run.core.OUT / 'p0_full_val_gate/complete.json'
    p0 = read(p0_path)
    assert p0['complete'] and p0['images'] == 459 and p0['source_P0_exact'] and p0['fused_U128_exact']
    for name, sha in p0['source_hashes'].items():
        assert run.core.C.file_sha(WORK / name) == sha
    evidence['full_val_p0_equivalence'] = run.core.C.file_sha(p0_path)
    output = dict(passed=True, source_hashes=expected, protocol=run.PROTOCOL, evidence_sha256=evidence,
        acceptance_source_sha256=run.core.C.file_sha(__file__),
        scope='Foundation/gradients, 4-arm FP32 path/250 masks, actual no-depth isolation, all5 RNG and nonempty optimizer restore, full reference cache identity.',
        limits='Not full training, AP benefit, CUDA bit-exact replay, cross-rig generalization, or completion of the old119-job scientific study.',
        launch_policy='Engineering ready; foreground scientific evidence review and resource admission still required before automatic long training.')
    run.core.C.dump(root / 'engineering_complete.json', output)
    print('COMPACT JOINT ENGINEERING ACCEPTANCE PASS', flush=True)


if __name__ == '__main__':
    main()
