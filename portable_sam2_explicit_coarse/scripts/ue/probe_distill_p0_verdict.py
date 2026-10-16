#!/usr/bin/env python3
"""DISTILL-P0 verdict: combine static gates + G2t differentials into the
preregistered kill/route decision (docs/DISTILL-P0-GATES-20261016_preregistration.md).

Kill quantity per teacher: per-scene dR2 = (R2_epN - R2_ep0); differential
d = dR2_true - dR2_sham (same 40 val scenes, same g init); scene-paired
percentile bootstrap (10k draws, seed 20261014); teacher survives G2t iff
CI lower bound > 0.01.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

BASE = Path('/data2/wangcheng/checkpoint/surfbev_b0/probes/DISTILL-P0-20261016')
SEED = 20261014


def boot(x):
    r = np.random.default_rng(SEED)
    d = x[r.integers(0, len(x), (10000, len(x)))].mean(1)
    return {'mean': float(x.mean()), 'ci95': [float(v) for v in np.quantile(d, [.025, .975])]}


def main():
    static = json.loads((BASE / 'static_gates.json').read_text())
    out = {'g1': static['g1'], 'g2f': static['g2f'], 'g2t': {}, 'draws': 10000, 'seed': SEED}
    for teacher in ('desc', 'height'):
        arms = {}
        for mode in ('true', 'sham'):
            j = json.loads((BASE / f'g2t_{teacher}_{mode}.json').read_text())
            arms[mode] = j
        assert arms['true']['val_ids'] == arms['sham']['val_ids']
        d_true = np.array(arms['true']['r2_epN']) - np.array(arms['true']['r2_ep0'])
        d_sham = np.array(arms['sham']['r2_epN']) - np.array(arms['sham']['r2_ep0'])
        diff = d_true - d_sham
        b = boot(diff)
        out['g2t'][teacher] = {
            'dR2_true': boot(d_true), 'dR2_sham': boot(d_sham), 'diff': b,
            'r2_ep0_true': float(np.mean(arms['true']['r2_ep0'])),
            'r2_epN_true': float(np.mean(arms['true']['r2_epN'])),
            'r2_ep0_sham': float(np.mean(arms['sham']['r2_ep0'])),
            'r2_epN_sham': float(np.mean(arms['sham']['r2_epN'])),
            'survives': bool(b['ci95'][0] > 0.01)}
        print(f"G2t[{teacher}]", json.dumps(out['g2t'][teacher], indent=1), flush=True)
    route = 'height' if (out['g2t'].get('height', {}).get('survives') and not out['g2t'].get('desc', {}).get('survives')) else \
            ('desc' if out['g2t'].get('desc', {}).get('survives') else
             ('both' if out['g2t'].get('desc', {}).get('survives') and out['g2t'].get('height', {}).get('survives') else 'none'))
    out['route'] = route
    out['verdict'] = {
        'distill_usage_alive': route != 'none',
        'g1_note': 'S~1 for both teachers: no dataset-level spatial template exists in UE (no SCVT4-style centre-prior channel); G1 is non-discriminative between noise and scene-specific content in this regime - authority delegated to G2f/G2t per the design hierarchy',
        'recommended_teacher': route if route in ('desc', 'height', 'both') else None}
    (BASE / 'verdict.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out['verdict'], indent=1))


if __name__ == '__main__':
    main()
