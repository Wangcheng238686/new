"""Exact vectorized COCO all-area accumulation for existing paired bootstraps.

Uses the frozen analysis for matching, boundary IoU, sampling, and reporting.
Only replaces COCO's Python precision-envelope loop with numpy.maximum.accumulate.
The first three resamplings of every evaluator are checked against the original.
"""
import json
from pathlib import Path
import sys

import numpy as np
import ue_joint_encoding_analysis as original

reference = original.sample_ap
audits = []


def fast_sample_ap(ev, indices):
    if not hasattr(ev, '_fast_all_area'):
        n = len(ev.params.imgIds)
        assert len(indices) == n
        assert ev.params.catIds == [ev.params.catIds[0]]
        ev._fast_all_area = ev.evalImgs[:n]
        ev._fast_verified = 0
    items = [ev._fast_all_area[int(i)] for i in indices]
    items = [e for e in items if e is not None]
    assert items
    ignored_gt = np.concatenate([e['gtIgnore'] for e in items])
    positive_gt = np.count_nonzero(ignored_gt == 0)
    assert positive_gt > 0
    scores = np.concatenate([e['dtScores'][:250] for e in items])
    order = np.argsort(-scores, kind='mergesort')
    matches = np.concatenate([e['dtMatches'][:, :250] for e in items], axis=1)[:, order]
    ignored_dt = np.concatenate([e['dtIgnore'][:, :250] for e in items], axis=1)[:, order]
    tp = np.cumsum(np.logical_and(matches, ~ignored_dt), axis=1).astype(np.float64)
    fp = np.cumsum(np.logical_and(~matches.astype(bool), ~ignored_dt), axis=1).astype(np.float64)
    rc = tp / positive_gt
    pr = tp / (fp + tp + np.spacing(1))
    envelope = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1]
    precisions = np.zeros((len(ev.params.iouThrs), len(ev.params.recThrs)), dtype=np.float64)
    for t in range(len(ev.params.iouThrs)):
        positions = np.searchsorted(rc[t], ev.params.recThrs, side='left')
        valid = positions < len(scores)
        precisions[t, valid] = envelope[t, positions[valid]]
    value = float(precisions.mean())
    if ev._fast_verified < 3:
        expected = reference(ev, indices)
        error = abs(value - expected)
        assert error < 1e-12, (value, expected, error)
        audits.append(dict(kind=ev.params.iouType, ratio=getattr(ev, 'ratio', None),
                           calibration=ev._fast_verified, max_abs_error=error))
        ev._fast_verified += 1
    return value


if __name__ == '__main__':
    original.sample_ap = fast_sample_ap
    original.main()
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    result = json.loads(output.read_text())
    result['accumulation_equivalence'] = dict(
        method='stable score sort; identical sampled-image order; vectorized precision envelope',
        reference_checks=audits, tolerance=1e-12,
        source_sha256=original.hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    temp = output.with_suffix('.tmp')
    temp.write_text(json.dumps(result, indent=2) + '\n')
    temp.replace(output)
