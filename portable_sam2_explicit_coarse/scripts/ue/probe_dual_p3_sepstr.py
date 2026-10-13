#!/usr/bin/env python3
"""DUAL-P3-SEPSTR: side-by-side stream readout on the frozen P1-V2 cache.

Arms: SAT (x_SAT, 64d), correct ([x_SAT | x_correct], 128d), sham90
([x_SAT | x_sham90], 128d). Same protocol/gates/lattice as DUAL-P2; see
docs/DUAL-P3-SEPSTR-20261014_preregistration.md.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from probe_dual_p2_readcap import (ARMS, SEED, TARGETS, HIDDENS, WDS, CS, THRS,
                                   N_FIT, MAX_EPOCH, CONTRACT_SHA, EXPECTED,
                                   sha256_file, f1, boot, fit_mlp, predict,
                                   load_train, rows_for, scene_split_check, pool_feats)


def concat_arm(x_by_arm, arm):
    if arm == 'SAT':
        return x_by_arm['SAT']
    return np.concatenate([x_by_arm['SAT'], x_by_arm[arm]], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--smoke-train', type=int, default=0)
    ap.add_argument('--smoke-val', type=int, default=0)
    a = ap.parse_args()
    smoke = bool(a.smoke_train or a.smoke_val)
    if smoke and not (a.smoke_train and a.smoke_val):
        raise SystemExit('smoke requires both limits')
    out = a.output_dir.resolve()
    if out == a.cache.resolve():
        raise RuntimeError('output directory must differ from the read-only cache')
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()) and not smoke:
        raise RuntimeError('output directory not empty')
    logf = open(out / 'run.log', 'a')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    got = sha256_file(a.cache / 'contract.json')
    if got != CONTRACT_SHA:
        raise RuntimeError(f'cache contract sha mismatch: {got}')
    counts = {s: len(list((a.cache / s).glob('*.npz'))) for s in ('train', 'val')}
    if not smoke and counts != EXPECTED:
        raise RuntimeError(f'cache counts {counts} != {EXPECTED}')
    log(f'cache verified contract_sha={CONTRACT_SHA[:16]} counts={counts} smoke={smoke}')

    t0 = time.time()
    tr = load_train(a.cache)
    n_train = len(tr['ids'])
    n_fit = a.smoke_train - max(1, a.smoke_train // 10) if smoke else N_FIT
    vfiles = sorted((a.cache / 'val').glob('*.npz'))
    val_ids = []
    for f in vfiles:
        z = np.load(f)
        val_ids.append(str(z['scene_id']))
        z.close()
    r = np.random.default_rng(SEED)
    perm = r.permutation(n_train)
    fit_idx, sel_idx = perm[:n_fit], perm[n_fit:]
    ids = tr['ids']
    scene_split_check(ids, fit_idx, sel_idx, val_ids)
    json.dump({'seed': SEED, 'n_train': n_train, 'fit': [ids[i] for i in fit_idx],
               'select': [ids[i] for i in sel_idx], 'val': val_ids},
              open(out / 'split.json', 'w'), indent=1)

    telemetry = []
    selection = {'targets': {}}
    models = {}
    for t in list(TARGETS) + ['iou']:
        cls = t != 'iou'
        fit_rows = rows_for(fit_idx, tr['lens'][t])
        sel_rows = rows_for(sel_idx, tr['lens'][t])
        y_full = tr[t]['y']
        sel_losses = {}
        mlp_models = {}
        for arm in ARMS:
            xa = concat_arm(tr[t]['x'], arm)
            sc = StandardScaler().fit(xa[fit_rows])
            xf = sc.transform(xa[fit_rows]).astype(np.float32)
            xs_ = sc.transform(xa[sel_rows]).astype(np.float32)
            yf, ys_ = y_full[fit_rows], y_full[sel_rows]
            for hi, hidden in enumerate(HIDDENS):
                for wi, wd in enumerate(WDS):
                    tag = f'{t}/{arm}/mlp/h{hi}/wd{wi}'
                    m, sl, ep = fit_mlp(xf, yf, xs_, ys_,
                                        hidden, wd, SEED + hi * 10 + wi, cls, telemetry, tag)
                    sel_losses[tag] = sl
                    mlp_models[arm, hi, wi] = (m, sc)

        def pair_score(hi, wi):
            return (sel_losses[f'{t}/SAT/mlp/h{hi}/wd{wi}'] + sel_losses[f'{t}/correct/mlp/h{hi}/wd{wi}']) / 2
        _, hi, wi = min(((pair_score(hi, wi), hi, wi)
                         for hi in range(len(HIDDENS)) for wi in range(len(WDS))))
        lin = {}
        lin_losses = {}
        for arm in ARMS:
            xa = concat_arm(tr[t]['x'], arm)
            sc = StandardScaler().fit(xa[fit_rows])
            xf = sc.transform(xa[fit_rows])
            xs_ = sc.transform(xa[sel_rows])
            yf, ys_ = y_full[fit_rows], y_full[sel_rows]
            for C in CS:
                tag = f'{t}/{arm}/lin/C{C}'
                if cls:
                    m = LogisticRegression(C=C, class_weight='balanced', max_iter=2000,
                                           random_state=SEED).fit(xf, yf)
                    p = m.predict_proba(xs_)[:, 1]
                    sl = float(log_loss(ys_, p, labels=[0, 1]))
                else:
                    m = Ridge(alpha=1 / C).fit(xf, yf)
                    p = m.predict(xs_)
                    sl = float(np.mean((p - ys_) ** 2))
                lin_losses[tag] = sl
                lin[arm, C] = (m, sc)

        def pair_lin(C):
            return (lin_losses[f'{t}/SAT/lin/C{C}'] + lin_losses[f'{t}/correct/lin/C{C}']) / 2
        best_lin = min((pair_lin(C), C) for C in CS)[1]
        selection['targets'][t] = {
            'mlp_hidden': list(HIDDENS[hi]), 'mlp_wd': WDS[wi],
            'mlp_select_loss_by_arm': {ar: sel_losses[f'{t}/{ar}/mlp/h{hi}/wd{wi}'] for ar in ARMS},
            'mlp_select_loss_full_grid': {k: v for k, v in sel_losses.items() if k.startswith(f'{t}/')},
            'lin_C': best_lin,
            'lin_select_loss_by_arm': {ar: lin_losses[f'{t}/{ar}/lin/C{best_lin}'] for ar in ARMS},
            'lin_select_loss_full_grid': {k: v for k, v in lin_losses.items() if k.startswith(f'{t}/')},
        }
        for arm in ARMS:
            models[t, arm, 'mlp'] = mlp_models[arm, hi, wi]
            models[t, arm, 'lin'] = lin[arm, best_lin]
        log(f'target {t}: mlp hidden={HIDDENS[hi]} wd={WDS[wi]} lin_C={best_lin} ({time.time()-t0:.0f}s)')

    thresholds = {}
    for reader in ('mlp', 'lin'):
        m, sc = models['boundary', 'correct', reader]
        sel_rows = rows_for(sel_idx, tr['lens']['boundary'])
        xa = concat_arm(tr['boundary']['x'], 'correct')
        xs_ = sc.transform(xa[sel_rows]).astype(np.float32)
        p = predict(m, xs_) if reader == 'mlp' else m.predict_proba(xs_)[:, 1]
        y = tr['boundary']['y'][sel_rows]
        pi = float(np.mean([tr['pi_yb'][i] for i in sel_idx]))
        best_t, best_f = None, -1.0
        for t_ in THRS:
            pr = p >= t_
            tp = ((pr) & (y > 0)).sum()
            fn = ((~pr) & (y > 0)).sum()
            fp = ((pr) & (y == 0)).sum()
            tn = ((~pr) & (y == 0)).sum()
            rec = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
            f1c = 2 * rec / (2 * rec + (1 - rec) + ((1 - pi) / max(pi, 1e-9)) * fpr)
            if f1c > best_f + 1e-12:
                best_t, best_f = float(t_), float(f1c)
        thresholds[reader] = {'threshold': best_t, 'pi': pi, 'select_f1_corrected': best_f}
    selection['thresholds'] = thresholds
    json.dump(selection, open(out / 'selection.json', 'w'), indent=1)
    with open(out / 'telemetry_fit.jsonl', 'w') as f:
        for row in telemetry:
            f.write(json.dumps(row) + '\n')

    if smoke:
        vfiles = vfiles[:a.smoke_val]
    rows = []
    for f in vfiles:
        z = np.load(f)
        row = {'scene_id': str(z['scene_id'])}
        x_by_arm = {arm: z[f'x_{arm}'].astype(np.float32) for arm in ARMS}
        for reader in ('mlp', 'lin'):
            t_ = thresholds[reader]['threshold']
            for arm in ARMS:
                x = concat_arm(x_by_arm, arm)
                mb, sb = models['boundary', arm, reader]
                me, se = models['error', arm, reader]
                xb = sb.transform(x).astype(np.float32)
                xe = se.transform(x).astype(np.float32)
                pb = predict(mb, xb) if reader == 'mlp' else mb.predict_proba(xb)[:, 1]
                pe = predict(me, xe) if reader == 'mlp' else me.predict_proba(xe)[:, 1]
                pooled = pool_feats(x, z['pred']) if len(z['pred']) else np.zeros((0, x.shape[1]), np.float32)
                mi, si = models['iou', arm, reader]
                if len(pooled):
                    xi = si.transform(pooled).astype(np.float32)
                    pi_ = predict(mi, xi) if reader == 'mlp' else mi.predict(xi)
                else:
                    pi_ = np.zeros(0)
                row[f'{reader}_{arm}'] = {
                    'f1': f1(pb >= t_, z['yb']),
                    'auc': float(roc_auc_score(z['ye'], pe)) if len(np.unique(z['ye'])) == 2 else 0.5,
                    'rho': float(spearmanr(pi_, z['qi']).statistic) if len(z['qi']) > 1 and np.std(z['qi']) > 0 else 0.0,
                }
        rows.append(row)
        z.close()

    if smoke:
        json.dump({'status': 'SMOKE_ONLY', 'n': {'train': n_train, 'fit': n_fit,
                   'select': len(sel_idx), 'val_evaluated': len(rows)},
                   'thresholds': thresholds, 'decision': None},
                  open(out / 'smoke_summary.json', 'w'), indent=1)
        log('SMOKE_ONLY complete')
        return

    with open(out / 'per_scene.jsonl', 'w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    met = {r: {arm: {q: np.array([row[f'{r}_{arm}'][q] for row in rows]) for q in ('f1', 'auc', 'rho')}
               for arm in ARMS} for r in ('mlp', 'lin')}
    diffs = {}
    for q in ('f1', 'auc', 'rho'):
        for b in ('SAT', 'sham90'):
            diffs[f'correct_minus_{b}_{q}'] = boot(met['mlp']['correct'][q] - met['mlp'][b][q])
    gate = {k: v['ci95'][0] > 0 for k, v in diffs.items()}
    f1_pass = gate['correct_minus_SAT_f1'] and gate['correct_minus_sham90_f1']
    auc_pass = gate['correct_minus_SAT_auc'] and gate['correct_minus_sham90_auc']
    rho_pass = gate['correct_minus_SAT_rho'] and gate['correct_minus_sham90_rho']
    sat_mlp_f1 = float(met['mlp']['SAT']['f1'].mean())
    sat_lin_f1 = float(met['lin']['SAT']['f1'].mean())
    degraded = sat_mlp_f1 < sat_lin_f1 - 0.005
    passed_gates = [g for g, ok in gate.items() if ok]
    if degraded:
        decision = 'READCAP_DEGRADED'
    elif f1_pass and auc_pass and rho_pass:
        decision = 'P3_PASS'
    elif not (f1_pass or auc_pass or rho_pass):
        decision = 'P3_NULL'
    else:
        pairs = [n for n, ok in (('f1', f1_pass), ('auc', auc_pass), ('rho', rho_pass)) if ok]
        decision = 'P3_PARTIAL(' + ','.join(pairs) + ')'
    summary = {
        'experiment': 'DUAL-P3-SEPSTR-20261014', 'status': 'complete', 'decision': decision,
        'passed_gates': passed_gates,
        'cache_contract_sha256': CONTRACT_SHA,
        'n': {'fit': n_fit, 'select': len(sel_idx), 'val': len(rows), 'test_read': 0},
        'thresholds': thresholds,
        'arms_mean': {r: {arm: {q: float(met[r][arm][q].mean()) for q in ('f1', 'auc', 'rho')}
                          for arm in ARMS} for r in ('mlp', 'lin')},
        'bootstrap_mlp': diffs, 'gates': gate,
        'parity_sentinel': {'sat_mlp_f1': sat_mlp_f1, 'sat_lin_f1': sat_lin_f1, 'degraded': bool(degraded)},
        'selection': selection['targets'],
    }
    json.dump(summary, open(out / 'summary.json', 'w'), indent=1)
    log(json.dumps({'decision': decision, 'gates': gate,
                    'sat_f1': {'mlp': sat_mlp_f1, 'lin': sat_lin_f1}}, indent=1))


if __name__ == '__main__':
    main()
