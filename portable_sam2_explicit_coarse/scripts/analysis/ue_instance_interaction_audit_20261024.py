"""Strict completion audit; fails while any required study artifact is missing."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'probes'))
import ue_instance_interaction_owner_20261024 as owner


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=owner.OUT)
    args = parser.parse_args()
    root = args.root
    manifest = json.loads((root / 'study_manifest.json').read_text())
    for name, expected in manifest['sources'].items():
        assert sha(ROOT / name) == sha(root / 'source' / name) == expected, name
    for name, expected in manifest['initializations'].items():
        assert sha(root / name) == expected
    init = {name: torch.load(root / name, map_location='cpu', weights_only=False)
            for name in manifest['initializations']}
    assert init['init_seed44.pth']['fingerprint'] != init['init_seed45.pth']['fingerprint']
    for key, value in init['init_seed44.pth']['adapter'].items():
        assert torch.equal(value, init['init_seed44_rgb.pth']['adapter'][key]), key
    jobs = owner.build_jobs(root)
    assert len(jobs) == manifest['planned_jobs']
    missing = [j['name'] for j in jobs if not owner.valid_marker(j)]
    assert not missing, ('Incomplete jobs', missing)
    expected_ids = json.loads(Path('/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021/splits/test.json').read_text())
    sample_orders = {}
    checkpoints = []
    for arm in owner.arms():
        out = root / arm['name']
        cp = torch.load(out / 'last.pth', map_location='cpu', weights_only=False)
        assert cp['epoch'] == 10 and cp['step'] == 2670
        contract = cp['contract']
        for key in ('seed', 'family', 'route', 'content'):
            assert contract[key] == arm[key]
        assert contract['rgb_content'] == arm['rgb']
        suffix = '_rgb' if arm['family'] == 'rgb' else ''
        initial = init[f"init_seed{arm['seed']}{suffix}.pth"]
        assert contract['initial_fingerprint'] == initial['fingerprint']
        assert all(v == manifest['sources'][k] for k, v in contract['source_hashes'].items())
        assert all(torch.isfinite(v).all() for v in cp['adapter'].values())
        steps = [int(value['step']) for value in cp['optimizer']['state'].values()]
        assert steps and set(steps) == {2670}, (arm['name'], set(steps))
        parameter_ids = cp['optimizer']['param_groups'][0]['params']
        # Original RGB backbone parameters may be present in the optimizer
        # group but must never acquire Adam state or change their tensors.
        if arm['family'] == 'rgb':
            for key, tensor in cp['adapter'].items():
                if key.startswith('rgb_encoder.') and 'lora_' not in key:
                    assert torch.equal(tensor, initial['adapter'][key]), key
        assert set(cp['rng_state']) == {'python', 'numpy', 'torch', 'cuda', 'loader_generator'}
        for epoch in range(1, 11):
            record = json.loads((out / f'epoch{epoch}.json').read_text())
            assert record['step'] == epoch * 267 and len(record['sample_order']) == 2136
            assert len(set(record['sample_order'])) == 2136
            key = (arm['seed'], epoch)
            if key in sample_orders:
                assert record['sample_order'] == sample_orders[key], ('Unpaired order', arm, epoch)
            else:
                sample_orders[key] = record['sample_order']
            assert record['frozen_SHA'] == contract['frozen_state_sha256']
        checkpoints.append(dict(arm=arm['name'], sha256=sha(out / 'last.pth'),
                                actual_optimizer_steps=sorted(set(steps))))
    assert sample_orders[44, 1] != sample_orders[45, 1]
    evaluations = []
    for job in jobs:
        if job['type'] != 'eval':
            continue
        directory = Path(job['marker']).parent
        identity = json.loads(Path(job['marker']).read_text())
        assert identity['images'] == 459 and identity['precision'] == 'FP32' and identity['max_dets'] == 250
        for name in ('satellite', 'combined'):
            rows = json.loads((directory / f'{name}_records.json').read_text())
            split = identity['split']
            ids = expected_ids if split == 'test' else json.loads(Path('/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021/splits/val.json').read_text())
            assert [r['image_id'] for r in rows] == ids
            assert all(len(r['dt']['scores']) <= 250 for r in rows)
            assert all(len(r['dt']['scores']) == len(r['dt']['rles']) for r in rows)
        evaluations.append(job['name'])
    disabled = root / owner.arms()[0]['name'] / 'test/disabled/satellite_records.json'
    historical = json.loads(owner.BASE.read_text())
    actual = json.loads(disabled.read_text())
    for a, b in zip(actual, historical):
        assert a['image_id'] == b['image_id'] and a['gt'] == b['gt'] and a['dt'] == b['dt'], 'Disabled baseline drift'
    for job in jobs:
        if job['type'] != 'analysis':
            continue
        analysis = json.loads(Path(job['marker']).read_text())
        for metric in analysis['metrics'].values():
            assert metric['bootstrap_samples'] == 500 and metric['seed'] == 20261021
        checks = analysis['accumulation_equivalence']['reference_checks']
        assert len(checks) == 24 and max(c['max_abs_error'] for c in checks) < 1e-12
    owner.dump(root / 'final_audit.json', dict(passed=True, checkpoints=checkpoints,
         paired_training_order=True, common_shared_initialization_exact=True,
         actual_evaluations=evaluations, all_jobs=len(jobs), source_unchanged=True,
         disabled_strong_satellite_exact=True,
         scope='Frozen E73 mechanism probes; no claim of full-budget common-initialization joint training or unseen-region generalization'))
    print('FINAL AUDIT PASS', len(checkpoints), 'checkpoints', len(evaluations), 'evaluations')


if __name__ == '__main__':
    main()
