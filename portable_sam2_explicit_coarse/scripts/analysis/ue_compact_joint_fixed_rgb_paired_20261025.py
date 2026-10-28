"""Paired AP for fixed RGB-D weights: correct minus constant/donor RGB.

Consumes existing completed inference only. Does not run training/inference,
select epochs or change geometry. Donor RGB keeps target geometry inconsistent.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

WORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORK / 'scripts/analysis'))
import ue_compact_joint_paired_20261024 as paired


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=paired.ROOT)
    p.add_argument('--mode', choices=['constant_rgb', 'donor_rgb'], required=True)
    p.add_argument('--view', choices=['combined', 'original_path'], required=True)
    args = p.parse_args()
    root = args.root
    out = root / 'fixed_rgb_intervention' / args.mode / ('paired_' + args.view + '.json')
    assert not out.exists(), 'Preserve completed analysis'
    correct, _, contract, evidence = paired.load_arm(root, 'rgbd', args.view)
    folder = out.parent
    completed = paired.read(folder / 'complete.json')
    official = paired.read(root / 'rgbd/test/complete.json')
    assert completed['complete'] and completed['mode'] == args.mode and completed['arm'] == 'rgbd'
    assert completed['images'] == 459 and completed['split'] == 'test' and completed['precision'] == 'FP32'
    assert completed['max_dets'] == 250 and completed['model_state_unchanged'] is True
    assert completed['checkpoint_sha256'] == official['checkpoint_sha256']
    assert completed['training_source_hashes'] == contract['source_hashes']
    driver = WORK / 'scripts/analysis/ue_compact_joint_rgb_intervention_20261024.py'
    assert paired.sha(driver) == completed['source_sha256']
    assert paired.sha(folder / driver.name) == completed['source_sha256']
    assert paired.sha(folder / 'raw/records.json') == completed['raw_records_sha256']
    assert paired.sha(folder / 'raw/metrics.json') == completed['raw_metrics_sha256']
    raw_complete = paired.read(folder / 'raw/complete.json')
    assert raw_complete['source_hashes'] == contract['source_hashes']
    assert raw_complete['checkpoint_sha256'] == official['checkpoint_sha256']
    rows = paired.read(folder / 'raw/records.json')
    assert len(rows) == len({r['image_id'] for r in rows}) == 459
    intervened = [dict(image_id=r['image_id'], gt=r['gt'],
        dt=r['dt'] if args.view == 'combined' else r['original_path']) for r in rows]
    assert [r['image_id'] for r in correct] == [r['image_id'] for r in intervened]
    assert all(a['gt'] == b['gt'] for a, b in zip(correct, intervened))
    if args.mode == 'donor_rgb':
        ids = [r['image_id'] for r in correct]
        assert completed['donor_map'] == {str(ident): ids[(i + 459//2) % 459] for i, ident in enumerate(ids)}
        assert len(completed['rgb_source_sha256']) == 459 * 5
        assert all(paired.sha(path) == value for path, value in completed['rgb_source_sha256'].items())
    else:
        assert completed['donor_map'] is None and not completed['rgb_source_sha256']
    evidence.update({str(folder / name): paired.sha(folder / name) for name in
        ['complete.json', 'raw/complete.json', 'raw/records.json', 'raw/metrics.json', driver.name]})
    result = dict(complete=False, mode=args.mode, view=args.view, images=459, split='test',
        precision='FP32', max_dets=250, checkpoint_sha256=official['checkpoint_sha256'],
        selection='same official RGB-D validation-best full100epoch checkpoint in both conditions',
        difference='correct RGB minus intervened RGB; geometry/calibration/weights retained',
        evidence_sha256=evidence, source_sha256=paired.sha(__file__), metrics={},
        limitations='single training seed; development-exposed v1; nominal image bootstrap excludes training variance/spatial dependence/multiplicity; donor RGB retains target geometry and is physically inconsistent')
    paired.fast.audits.clear()
    samples = np.random.default_rng(20261021).integers(0, 459, (500, 459))
    for name, kind, ratio in [('MaskAP', 'segm', None), ('BBoxAP', 'bbox', None),
                              ('BoundaryAP_2pct', 'segm', .02), ('BoundaryAP_1pct', 'segm', .01)]:
        eb = paired.fast.original.evaluator(intervened, kind=kind, ratio=ratio)
        ec = paired.fast.original.evaluator(correct, kind=kind, ratio=ratio)
        pb, pc = paired.point_metrics(eb), paired.point_metrics(ec)
        assert abs(paired.fast.fast_sample_ap(eb, np.arange(459)) - pb['AP']) < 1e-12
        assert abs(paired.fast.fast_sample_ap(ec, np.arange(459)) - pc['AP']) < 1e-12
        deltas = [paired.fast.fast_sample_ap(ec, ix) - paired.fast.fast_sample_ap(eb, ix) for ix in samples]
        result['metrics'][name] = dict(intervened=pb, correct=pc, delta=pc['AP'] - pb['AP'],
            paired_image_bootstrap95=np.quantile(deltas, [.025, .975]).tolist(), bootstrap_samples=500, seed=20261021)
        print(args.mode, args.view, name, result['metrics'][name], flush=True)
    assert len(paired.fast.audits) == 24 and all(a['max_abs_error'] < 1e-12 for a in paired.fast.audits)
    result['accumulation_reference_checks'] = copy.deepcopy(paired.fast.audits)
    assert all(paired.sha(path) == value for path, value in evidence.items())
    result['complete'] = True
    (folder / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    temporary = out.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(out)


if __name__ == '__main__':
    main()
