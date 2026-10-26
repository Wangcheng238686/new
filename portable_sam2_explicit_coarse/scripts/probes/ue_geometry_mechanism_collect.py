"""Audit eight fixed geometry arms from actual checkpoints, histories and predictions.

Image bootstrap intervals are nominal, not evidence of training-seed stability.
Factor changes in matched content effects are reported as point estimates only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from ue_geometry_mechanism_schedule import read, sha, dump, arms

FROZEN = 'c6921c30416595fb994facd3d4d927408be9bdd636399fd669cfbe5968329536'
METRICS = ('MaskAP', 'BoundaryAP_2pct', 'BoundaryAP_1pct')
DATA = Path('/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021')


def tensor_digest(state):
    h = hashlib.sha256()
    for key, tensor in state.items():
        h.update(key.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def paired(base, candidate):
    assert len(base) == len(candidate) == 459
    for b, c in zip(base, candidate):
        assert b['image_id'] == c['image_id'] and b['gt'] == c['gt']
        assert all(b['dt'][k] == c['dt'][k] for k in ('bboxes', 'labels', 'scores'))


def audit_training(root, arm, manifest, init):
    folder = root / arm['name']; path = folder / 'last.pth'
    cp = torch.load(path, map_location='cpu', weights_only=False)
    assert cp['epoch'] == 10 and cp['step'] == 2670
    assert all(k in cp for k in ('adapter', 'optimizer', 'scaler', 'rng_state', 'contract'))
    assert 'loader_generator' in cp['rng_state']
    c = cp['contract']
    assert json.loads(json.dumps(c)) == read(folder / 'frozen_training_contract.json')
    for key in ('sampling', 'fusion', 'content'):
        assert c[key] == arm[key] and c['args'][key] == arm[key]
    assert c['base_sha256'] == manifest['base_sha256']
    assert c['satellite_state_sha256'] == FROZEN and c['satellite_frozen']
    assert c['init_file_sha256'] == manifest['init_sha256']
    assert c['initial_fingerprint'] == tensor_digest(init['adapter']) == init['fingerprint']
    assert c['constants_sha256'] == manifest['constants_sha256']
    assert np.allclose(c['constant_fields'], read(root / 'common_constants.json')['mean_fields'], rtol=0, atol=1e-7)
    assert c['epochs'] == 10 and c['updates'] == 2670 and c['batch'] == 1 and c['accum'] == 8
    assert not c['ema'] and c['selection'] == 'E10 last'
    assert c['epoch_reseed_from'] == 1 and c['epoch_seed_formula'] == '1044000+epoch'
    assert c['max_dets'] == [1, 10, 250] and c['eval_precision'] == 'fp32'
    for name, digest in c['source_hashes'].items():
        assert digest == manifest['source_hashes'][name] == sha(folder / 'source' / name)
    rows = [json.loads(line) for line in (folder / 'train.jsonl').read_text().splitlines()]
    assert len(rows) == 2670 and [r['update'] for r in rows] == list(range(1, 2671))
    orders = []
    training_ids = {str(i).zfill(6) for i in read(DATA / 'splits/train.json')}
    for epoch in range(1, 11):
        part = rows[(epoch-1)*267:epoch*267]
        assert all(r['epoch'] == epoch and r['epoch_seed'] == 1044000+epoch for r in part)
        assert [r['images'] for r in part] == list(range(8, 2137, 8))
        assert all(np.isfinite(r['loss']) and np.isfinite(r['grad_norm']) for r in part)
        e = read(folder / f'epoch{epoch}_summary.json')
        assert e['epoch'] == epoch and e['update'] == epoch*267 and e['microbatches'] == 2136
        assert e['satellite_state_sha256'] == FROZEN
        assert len(e['sample_order']) == len(set(e['sample_order'])) == 2136
        assert set(e['sample_order']) <= training_ids
        assert e['sample_order_sha256'] == hashlib.sha256(json.dumps(e['sample_order']).encode()).hexdigest()
        orders.append(e['sample_order_sha256'])
    for row in rows:
        expected_lr=1e-4*min(1.,row['update']/50)*(.1+.9*.5*(1+np.cos(np.pi*row['update']/2670)))
        assert abs(row['lr']-expected_lr)<1e-14
    assert abs(rows[-1]['lr']-1e-5) < 1e-14
    assert set(cp['adapter']) == set(init['adapter'])
    delta = {k: float((v.detach().cpu()-init['adapter'][k]).float().norm()) for k, v in cp['adapter'].items()}
    changes = read(folder / 'parameter_changes.json')
    assert changes['satellite_unchanged'] and changes['satellite_state_sha256'] == FROZEN
    assert all(abs(delta[k]-changes['adapter_changes'][k]) < 1e-6 for k in delta)
    assert any(v > 0 for v in delta.values())
    return dict(checkpoint_sha256=sha(path), updates=len(rows), sample_order_epoch_hashes=orders,
                initialization_fingerprint=init['fingerprint'], archive_hashes=c['archive_hashes'],
                adapter_changes=delta, satellite_epoch_audits_exact=True,
                satellite_evidence_scope='Runner hashes actual frozen model after every epoch; checkpoints contain adapter only')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    root = p.parse_args().root; manifest = read(root / 'manifest.json')
    assert manifest['arms'] == arms() and manifest['selection'] == 'epoch10 last'
    assert manifest['primary_split'] == 'test' and manifest['no_test_selection']
    assert len(manifest['comparisons']) == 24 and len(manifest['evaluations']) == 28
    assert sha(root / 'common_init.pth') == manifest['init_sha256']
    assert sha(root / 'common_constants.json') == manifest['constants_sha256']
    constants = read(root / 'common_constants.json'); assert constants['split'] == 'train'
    constant_rows=constants['rows']; assert len(constant_rows)==constants['images']==2142
    assert {int(x['point']) for x in constant_rows}==set(read(DATA/'splits/train.json'))
    constant_pixels=sum(x['pixels'] for x in constant_rows)
    assert constant_pixels==constants['supported_pixels']
    recomputed=np.array([x['sum_fields'] for x in constant_rows],dtype=np.float64).sum(0)/constant_pixels
    assert np.allclose(recomputed,constants['mean_fields'],rtol=0,atol=1e-12)
    assert constants['split_sha256']==sha(DATA/'splits/train.json')
    for name, digest in manifest['source_hashes'].items():
        assert sha(root / 'frozen_source' / name) == digest
    init = torch.load(root / 'common_init.pth', map_location='cpu', weights_only=False)
    initial_json = read(root / 'common_init.json')
    assert initial_json['sha256'] == manifest['init_sha256'] and initial_json['fingerprint'] == init['fingerprint']
    gate = read(root / 'resource_gate.json')
    assert gate['all_passed'] and gate['init_sha256'] == manifest['init_sha256']
    assert gate['constants_sha256'] == manifest['constants_sha256']
    for config, g in gate['configurations'].items():
        assert g['all_passed'] and g['fp32_resource_rois'] == 250
        path = Path(g['gate_path'])
        assert sha(path) == g['gate_sha256']
        actual = read(path)
        assert actual['resource_rois'] == 250 and actual['resource_chunk'] == 32
        assert actual['satellite_parameters_buffers_exact'] and actual['satellite_no_grad']
        assert actual['satellite_state_sha256'] == FROZEN
        for report in actual['identity'].values():
            assert report['zero_alpha_logits_exact'] and report['no_support_nonzero_alpha_logits_exact']
            if config.endswith('difference'): assert report['zero_fields_difference_identity']
        assert all(manifest['source_hashes'][n] == d for n, d in actual['source_hashes'].items())
        measured = max(actual['training_amp_peak']['reserved_MiB'], actual['fp32_250roi_peak']['reserved_MiB'])
        assert g['peak_reserved_mib'] >= measured
    audit = {a['name']: audit_training(root, a, manifest, init) for a in manifest['arms']}
    assert len({tuple(a['sample_order_epoch_hashes']) for a in audit.values()}) == 1
    assert len({json.dumps(a['archive_hashes'], sort_keys=True) for a in audit.values()}) == 1
    ids = {split: set(read(DATA / f'splits/{split}.json')) for split in ('train', 'val', 'test')}
    assert len(ids['train']) == 2142 and len(ids['val']) == len(ids['test']) == 459
    assert not ids['train'] & ids['val'] and not ids['train'] & ids['test'] and not ids['val'] & ids['test']
    base_path = root / 'test/baseline/disabled'; base = read(base_path / 'records.json')
    base_identity = read(base_path / 'evaluation_identity.json')
    assert base_identity['mode'] == 'disabled' and base_identity['adapter_sha256'] is None
    assert base_identity['base_sha256'] == manifest['base_sha256']
    assert base_identity['split'] == 'test' and base_identity['images'] == 459
    assert base_identity['inference_precision'] == 'fp32' and base_identity['score_key'] == 'scores'
    assert base_identity['max_dets'] == [1,10,250]
    assert len(base) == 459 and {r['image_id'] for r in base} == ids['test']
    baseline_metrics=read(base_path/'metrics.json')
    assert abs(baseline_metrics['segm/mAP']-.6455098296975914)<1e-12
    assert abs(baseline_metrics['bbox/mAP']-.6854287640342888)<1e-12
    evaluations = {}
    for ev in manifest['evaluations']:
        folder = Path(ev['folder']); identity = read(folder / 'evaluation_identity.json'); records = read(folder / 'records.json')
        assert identity['split'] == ev['split'] and identity['mode'] == ev['mode'] and identity['images'] == len(records) == 459
        assert identity['base_sha256'] == manifest['base_sha256']
        assert identity['adapter_sha256'] == audit[ev['arm']]['checkpoint_sha256']
        assert identity['init_sha256'] == manifest['init_sha256'] and identity['constants_sha256'] == manifest['constants_sha256']
        assert identity['score_key'] == 'scores' and identity['max_dets'] == [1,10,250] and identity['inference_precision'] == 'fp32'
        assert {r['image_id'] for r in records} == ids[ev['split']]
        assert all(manifest['source_hashes'][n] == d for n,d in identity['source_hashes'].items())
        if ev['split'] == 'test': paired(base, records)
        else:
            reference = root / 'val_reference.json'
            # Compare val records directly across all arms without trusting an existing reference artifact.
            if 'val_base' not in locals(): val_base = records
            paired(val_base, records)
        m = read(folder / 'metrics.json')
        if ev['split'] == 'test': assert abs(m['bbox/mAP'] - read(base_path / 'metrics.json')['bbox/mAP']) < 1e-12
        evaluations[ev['key']] = dict(identity=identity, records_sha256=sha(folder / 'records.json'), metrics=m)
    comparisons = {}; point_by_path = {}
    for comparison in manifest['comparisons']:
        result = read(comparison['output'])
        assert result['evaluation_split'] == 'test' and result['images'] == 459
        assert result['detector_and_gt_exact'] and result['selection'] == 'last'
        assert result['analysis_sha256'] == manifest['source_hashes']['scripts/probes/ue_v1_adapter_analysis.py']
        paired(read(comparison['baseline']), read(comparison['candidate']))
        for metric in METRICS:
            v = result['metrics'][metric]
            assert v['bootstrap_samples'] == 500 and v['seed'] == 20261021
            assert abs(v['delta'] - (v['candidate']-v['baseline'])) < 1e-12
            assert len(v['paired_image_bootstrap95']) == 2 and np.isfinite(v['paired_image_bootstrap95']).all()
            for side in ('baseline','candidate'):
                key=(comparison[side], metric)
                if key in point_by_path: assert abs(point_by_path[key]-v[side]) < 1e-12
                else: point_by_path[key]=v[side]
        for side in ('baseline','candidate'):
            mp = read(Path(comparison[side]).parent / 'metrics.json')
            assert abs(result['metrics']['MaskAP'][side]-mp['segm/mAP']) < 1e-12
        comparisons[comparison['key']] = dict(kind=comparison['kind'], result=result,
            baseline_records_sha256=sha(comparison['baseline']), candidate_records_sha256=sha(comparison['candidate']))
    interactions = {}
    for factor, fixed in [('sampling','concat'),('sampling','difference'),('fusion','bilinear'),('fusion','nearest')]:
        if factor == 'sampling': names = ['bilinear_'+fixed, 'nearest_'+fixed]
        else: names = [fixed+'_concat', fixed+'_difference']
        interactions[factor+'_'+fixed] = {m: comparisons['content_'+names[1]]['result']['metrics'][m]['delta'] -
            comparisons['content_'+names[0]]['result']['metrics'][m]['delta'] for m in METRICS}
    summary = dict(all_complete=True, manifest=manifest, training_audits=audit, evaluations=evaluations,
        comparisons=comparisons, content_effect_factor_differences_point_only=interactions,
        no_interaction_confidence_intervals=True,
        caveats=['one training seed; image bootstrap excludes optimization variance and spatial correlation',
                 'multiple exploratory comparisons; per-comparison intervals are nominal, not corrected',
                 'v1 test has already been used for mechanism exploration, not a fresh final benchmark',
                 'difference zero identity is a structural property, not evidence of content benefit',
                 'constant geometry retains UAV support; this is not a fully satellite-only capacity control'])
    dump(root / 'complete_audit_summary.json', summary)
    lines=['# Geometry mechanism cohort: complete fixed E10 audit', '',
        '|Sampling|Fusion|Training content|Mask AP|Delta vs base (pp)|Nominal 95% CI (pp)|',
        '|---|---|---|---:|---:|---|']
    for arm in manifest['arms']:
        v=comparisons['baseline_'+arm['name']]['result']['metrics']['MaskAP']
        lines.append('|'+ '|'.join([arm['sampling'],arm['fusion'],arm['content'],f"{v['candidate']:.6f}",
            f"{100*v['delta']:+.4f}",str([100*x for x in v['paired_image_bootstrap95']])])+'|')
    lines += ['', '## Matched training content effects', '']
    for key, value in comparisons.items():
        if value['kind'] != 'matched_training_content': continue
        v=value['result']['metrics']['MaskAP']
        lines.append(f"- {key}: {100*v['delta']:+.4f} pp; nominal 95% CI {[100*x for x in v['paired_image_bootstrap95']]}")
    lines += ['', 'Sampling/fusion changes in content effects are point estimates only (no interaction CI).',
              'Fixed-checkpoint perturbations establish dependency, not positive content contribution.',
              'All comparisons and Boundary AP metrics are in complete_audit_summary.json.', '', '## Limits', '']
    lines += ['- '+c for c in summary['caveats']]
    (root / 'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    print('Complete: 8 actual E10 checkpoints; 29 evaluations including baseline; 24 paired comparisons.')


if __name__ == '__main__': main()
