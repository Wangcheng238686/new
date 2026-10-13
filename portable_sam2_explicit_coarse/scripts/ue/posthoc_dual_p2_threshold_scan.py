#!/usr/bin/env python3
"""POST-HOC diagnostic for DUAL-P2 (not a gate; interpretation only).

Scans a common threshold grid on val for both readers to test whether the
correct<SAT ordering is an artifact of the preregistered threshold rule.
Refits identical models (same fit/select split, same selection rule), then
reports per-arm val F1 at each common threshold and per-arm-best thresholds.
Writes posthoc_threshold_scan.json into the artifact dir.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from probe_dual_p2_readcap import (ARMS, SEED, CS, load_train, rows_for, f1,
                                   fit_mlp, predict, CONTRACT_SHA, EXPECTED)

CACHE = Path('/data1/wangcheng/checkpoint/surfbev_b0/probes/SURFBEV-P1-V2-R1a-20261012/cache')
OUT = Path('/data1/wangcheng/checkpoint/surfbev_b0/probes/DUAL-P2-READCAP-20261014')


def main():
    tr = load_train(CACHE)
    n_train = len(tr['ids'])
    r = np.random.default_rng(SEED)
    perm = r.permutation(n_train)
    fit_idx, sel_idx = perm[:555], perm[555:]
    t = 'boundary'
    fit_rows = rows_for(fit_idx, tr['lens'][t])
    sel_rows = rows_for(sel_idx, tr['lens'][t])
    y_full = tr[t]['y']
    grid = np.arange(0.1, 1.0, 0.1)
    report = {'posthoc': True, 'note': 'common-threshold sweep on val; interpretation only, not a gate',
              'grid': [float(g) for g in grid], 'per_arm_best': {}, 'common_threshold_curve': {}}
    for reader in ('mlp', 'lin'):
        models = {}
        for arm in ARMS:
            sc = StandardScaler().fit(tr[t]['x'][arm][fit_rows])
            xf = sc.transform(tr[t]['x'][arm][fit_rows])
            xs_ = sc.transform(tr[t]['x'][arm][sel_rows])
            if reader == 'mlp':
                m, _, _ = fit_mlp(xf.astype(np.float32), y_full[fit_rows],
                                  xs_.astype(np.float32), y_full[sel_rows],
                                  (64, 64), 1e-4, SEED + 21, True, [], f'posthoc/{arm}')
            else:
                m = LogisticRegression(C=10.0, class_weight='balanced', max_iter=2000,
                                       random_state=SEED).fit(xf, y_full[fit_rows])
            models[arm] = (m, sc)
        # per-scene F1 across the threshold grid on val
        curve = {arm: [] for arm in ARMS}
        files = sorted((CACHE / 'val').glob('*.npz'))
        scores = {arm: [] for arm in ARMS}
        ys, scene_ids = [], []
        for f in files:
            z = np.load(f)
            ys.append(z['yb'])
            scene_ids.append(str(z['scene_id']))
            for arm in ARMS:
                m, sc = models[arm]
                x = sc.transform(z[f'x_{arm}'].astype(np.float32))
                p = predict(m, x.astype(np.float32)) if reader == 'mlp' else m.predict_proba(x)[:, 1]
                scores[arm].append(p)
            z.close()
        for g in grid:
            for arm in ARMS:
                curve[arm].append(float(np.mean([f1(s >= g, y) for s, y in zip(scores[arm], ys)])))
        report['common_threshold_curve'][reader] = curve
        best = {arm: {'threshold': float(grid[int(np.argmax(curve[arm]))]),
                      'f1': float(max(curve[arm]))} for arm in ARMS}
        report['per_arm_best'][reader] = best
        print(reader, json.dumps(best))
    json.dump(report, open(OUT / 'posthoc_threshold_scan.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
