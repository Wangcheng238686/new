"""Read-only, validation-best paired analysis for the locked compact joint study.

Does not run inference, select checkpoints, or import the numerical training runner.
Reuses the frozen COCO/boundary matcher and calibrated accumulation estimator.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

WORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORK / 'scripts/probes'))
import ue_joint_encoding_fast_analysis as fast

ROOT = Path('/data/wangcheng/checkpoint/ue_compact_joint_20261024')
COMPARISONS = {}
for view in ('combined', 'original_path'):
    COMPARISONS.update({
        'rgbd_net_' + view: ('satellite', 'rgbd', view),
        'rgb_content_' + view: ('constant_rgb', 'rgbd', view),
        'depth_content_' + view: ('reference_plane', 'rgbd', view),
    })
COMPARISONS.update(constant_rgb_net=('satellite', 'constant_rgb', 'combined'),
                   reference_plane_net=('satellite', 'reference_plane', 'combined'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def load_arm(root, arm, view):
    folder = root / arm
    initial = read(root / 'initialization.json')
    contract = read(folder / 'contract.json')
    done = read(folder / 'training_complete.json')
    complete = read(folder / 'test/complete.json')
    assert contract['arm'] == arm and done['contract'] == contract
    assert contract['initialization'] == initial and initial['task_weights_loaded'] is False
    assert contract['source_hashes'] == initial['source_hashes']
    assert contract['protocol']['epochs'] == 100 and contract['protocol']['EMA'] is False
    assert done['complete'] and done['epoch'] == 100 and done['updates'] == 26700
    assert complete['complete'] and complete['purpose'] == 'full_protocol'
    assert complete['split'] == 'test' and complete['images'] == 459
    assert complete['arm'] == arm and complete['precision'] == 'FP32' and complete['max_dets'] == 250
    assert complete['source_hashes'] == contract['source_hashes']
    assert complete['checkpoint_sha256'] == done['best_checkpoint_sha256']
    assert sha(folder / 'best_model.pth') == done['best_checkpoint_sha256']
    assert sha(folder / 'last.pth') == done['last_checkpoint_sha256']
    for name, expected in contract['source_hashes'].items():
        assert sha(WORK / name) == expected, name
        assert sha(root / 'source' / name) == expected, name
    epochs = [read(folder / ('epoch%d.json' % n)) for n in range(1, 101)]
    best_score, best_epoch = -1., None
    orders = []
    for n, row in enumerate(epochs, 1):
        assert row['epoch'] == n and row['step'] == n * 267
        order = row['sample_order']
        assert len(order) == len(set(order)) == 2136
        orders.append(order)
        score = row['validation']['segm/mAP']
        assert np.isfinite(score)
        if score > best_score:
            best_score, best_epoch = score, n
        assert row['best'] == dict(score=best_score, epoch=best_epoch)
    assert done['best'] == dict(score=best_score, epoch=best_epoch)
    best = read(folder / 'best_validation_metrics.json')
    assert best['epoch'] == best_epoch and best['metrics'] == epochs[best_epoch - 1]['validation']
    path = folder / 'test/records.json'
    rows = read(path)
    assert len(rows) == len({r['image_id'] for r in rows}) == 459
    selected = [dict(image_id=r['image_id'], gt=r['gt'],
                     dt=r['dt'] if view == 'combined' else r['original_path']) for r in rows]
    evidence = {str(p): sha(p) for p in [path, folder / 'contract.json',
        folder / 'training_complete.json', folder / 'test/complete.json',
        folder / 'test/metrics.json', folder / 'best_validation_metrics.json']}
    evidence.update({str(folder / ('epoch%d.json' % n)): sha(folder / ('epoch%d.json' % n))
                     for n in range(1, 101)})
    return selected, orders, contract, evidence


def point_metrics(ev):
    precision = ev.eval['precision']
    def average(a):
        a = a[a >= 0]
        return float(a.mean()) if a.size else None
    t75 = int(np.flatnonzero(np.isclose(ev.params.iouThrs, .75))[0])
    return dict(AP=average(precision[:, :, :, 0, -1]),
                AP75=average(precision[t75, :, :, 0, -1]),
                AP_small=average(precision[:, :, :, 1, -1]),
                AP_medium=average(precision[:, :, :, 2, -1]),
                AP_large=average(precision[:, :, :, 3, -1]),
                AR250=average(ev.eval['recall'][:, :, 0, -1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--comparison', choices=sorted(COMPARISONS))
    p.add_argument('--plan-only', action='store_true')
    args = p.parse_args()
    if args.plan_only:
        print(json.dumps(COMPARISONS, indent=2))
        return
    assert args.comparison, '--comparison is required'
    base_arm, cand_arm, view = COMPARISONS[args.comparison]
    out = args.root / 'paired_analysis' / (args.comparison + '.json')
    assert not out.exists(), 'Preserve prior completed analyses'
    base, bo, bc, be = load_arm(args.root, base_arm, view)
    cand, co, cc, ce = load_arm(args.root, cand_arm, view)
    assert bo == co, 'Training sample order differs'
    assert {k: v for k, v in bc.items() if k != 'arm'} == {k: v for k, v in cc.items() if k != 'arm'}
    assert [r['image_id'] for r in base] == [r['image_id'] for r in cand]
    assert all(b['gt'] == c['gt'] for b, c in zip(base, cand))
    result = dict(complete=False, comparison=args.comparison, view=view,
        selection='each-arm validation-best combined Mask AP250; ties earlier epoch; full100epochs',
        images=459, split='test', precision='FP32', max_dets=250,
        baseline_arm=base_arm, candidate_arm=cand_arm, metrics={}, evidence_sha256={**be, **ce},
        analysis_source_sha256={str(Path(__file__).resolve()): sha(__file__),
            str(Path(fast.__file__).resolve()): sha(fast.__file__),
            str(Path(fast.original.__file__).resolve()): sha(fast.original.__file__)},
        limitations='single training seed; development-exposed v1; nominal image bootstrap does not cover training variance, spatial dependence or multiple comparisons; reference-plane changes all depth-derived correspondence/support/fields jointly')
    fast.audits.clear()
    samples = np.random.default_rng(20261021).integers(0, 459, (500, 459))
    for name, kind, ratio in [('MaskAP', 'segm', None), ('BBoxAP', 'bbox', None),
                              ('BoundaryAP_2pct', 'segm', .02), ('BoundaryAP_1pct', 'segm', .01)]:
        eb = fast.original.evaluator(base, kind=kind, ratio=ratio)
        ec = fast.original.evaluator(cand, kind=kind, ratio=ratio)
        pb, pc = point_metrics(eb), point_metrics(ec)
        assert abs(fast.fast_sample_ap(eb, np.arange(459)) - pb['AP']) < 1e-12
        assert abs(fast.fast_sample_ap(ec, np.arange(459)) - pc['AP']) < 1e-12
        deltas = [fast.fast_sample_ap(ec, ix) - fast.fast_sample_ap(eb, ix) for ix in samples]
        result['metrics'][name] = dict(baseline=pb, candidate=pc, delta=pc['AP'] - pb['AP'],
            paired_image_bootstrap95=np.quantile(deltas, [.025, .975]).tolist(),
            bootstrap_samples=500, seed=20261021)
        print(name, result['metrics'][name], flush=True)
    assert len(fast.audits) == 24
    assert all(a['max_abs_error'] < 1e-12 for a in fast.audits)
    result['accumulation_reference_checks'] = fast.audits
    m = result['metrics']['MaskAP']
    result['signal'] = dict(positive_interval=m['paired_image_bootstrap95'][0] > 0,
        net_reference_threshold=m['delta'] >= .002 and m['paired_image_bootstrap95'][0] > 0,
        threshold_scope='net interpretation only for rgbd_net_combined; other comparisons are content/path diagnostics')
    # Reports are valid only for the same immutable prediction/contract artifacts.
    assert all(sha(p) == value for p, value in result['evidence_sha256'].items())
    result['complete'] = True
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(out)


if __name__ == '__main__':
    main()
