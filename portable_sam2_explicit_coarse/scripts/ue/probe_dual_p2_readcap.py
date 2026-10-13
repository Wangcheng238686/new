#!/usr/bin/env python3
"""DUAL-P2-READCAP: nonlinear readout-capacity probe on the frozen P1-V2 cache.

Reuses the R1a harvest verbatim (three arms, 64-dim projected features); swaps the
linear reader for a capacity-controlled MLP. Selection happens on a fit/select
split inside train; val 100 is scored once; test is never read.
See docs/DUAL-P2-READCAP-20261014_preregistration.md.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

SEED = 20261014
ARMS = ('SAT', 'correct', 'sham90')
TARGETS = ('boundary', 'error')
HIDDENS = ((64,), (128,), (64, 64))
WDS = (1e-4, 1e-3)
CS = (0.01, 0.1, 1.0, 10.0)
THRS = np.arange(0.1, 1.0, 0.1)
N_FIT, N_SELECT = 555, 139
MAX_EPOCH, PATIENCE, BATCH, LR = 50, 5, 4096, 1e-3
CONTRACT_SHA = 'af04fc445252b4b8902636eabe975559bfc18a86b21066841aeb4d3afb2e1732'
EXPECTED = {'train': 694, 'val': 100}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def f1(pred: np.ndarray, y: np.ndarray) -> float:
    t = int(((pred > 0) & (y > 0)).sum())
    return float(2 * t / max(2 * t + (pred > 0).sum() + (y > 0).sum() - 2 * t, 1))


def boot(x: np.ndarray) -> dict:
    r = np.random.default_rng(SEED)
    d = x[r.integers(0, len(x), (10000, len(x)))].mean(1)
    return {'mean': float(x.mean()), 'ci95': [float(v) for v in np.quantile(d, [.025, .975])]}


class MLP(nn.Module):
    def __init__(self, din: int, hidden, out: int = 1):
        super().__init__()
        layers, prev = [], din
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def fit_mlp(xtr, ytr, xse, yse, hidden, wd, seed, classification, telemetry, tag):
    """Train with early stop on select loss; restore best-select-loss weights."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = MLP(xtr.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=wd)
    xt = torch.from_numpy(xtr)
    yt = torch.from_numpy(ytr.astype(np.float32))
    lossf = nn.BCEWithLogitsLoss() if classification else nn.MSELoss()
    best = (math.inf, None, 0)
    n = len(xt)
    for epoch in range(MAX_EPOCH):
        model.train()
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            out = model(xt[idx]).squeeze(-1)
            loss = lossf(out, yt[idx])
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        model.eval()
        with torch.no_grad():
            ps = []
            for i in range(0, len(xse), 65536):
                ps.append(model(torch.from_numpy(xse[i:i + 65536])).squeeze(-1).numpy())
            pse = np.concatenate(ps)
        se_loss = float(log_loss(yse, 1 / (1 + np.exp(-pse)), labels=[0, 1])) if classification \
            else float(np.mean((pse - yse) ** 2))
        telemetry.append({'tag': tag, 'epoch': epoch, 'train_loss': tot / n, 'select_loss': se_loss})
        if se_loss < best[0] - 1e-6:
            best = (se_loss, {k: v.clone() for k, v in model.state_dict().items()}, epoch)
        elif epoch - best[2] >= PATIENCE:
            break
    model.load_state_dict(best[1])
    return model, best[0], best[2]


def predict(model, x, chunk=65536):
    model.eval()
    ps = []
    with torch.no_grad():
        for i in range(0, len(x), chunk):
            ps.append(model(torch.from_numpy(x[i:i + chunk])).squeeze(-1).numpy())
    return np.concatenate(ps)


def load_train(cache: Path):
    """Return per-target/arm stacked train arrays plus per-scene row lengths."""
    files = sorted((cache / 'train').glob('*.npz'))
    data = {t: {a: [] for a in ARMS} for t in TARGETS}
    ydata = {t: [] for t in TARGETS}
    lens = {t: [] for t in TARGETS}
    data['iou'] = {a: [] for a in ARMS}
    ydata['iou'] = []
    lens['iou'] = []
    yb_full = []
    ids = []
    for f in files:
        z = np.load(f)
        ids.append(str(z['scene_id']))
        for t in TARGETS:
            for a in ARMS:
                data[t][a].append(z[f'{t}_{a}'])
            y = z[f'y_{t}']
            ydata[t].append(y)
            lens[t].append(len(y))
        for a in ARMS:
            data['iou'][a].append(z[f'iou_{a}'])
        q = z['y_iou']
        ydata['iou'].append(q)
        lens['iou'].append(len(q))
        yb_full.append(float(z['yb'].mean()))
        z.close()
    out = {'ids': ids, 'pi_yb': yb_full, 'lens': lens}
    for t in list(TARGETS) + ['iou']:
        out[t] = {'x': {a: np.concatenate(data[t][a]) for a in ARMS}, 'y': np.concatenate(ydata[t])}
    return out


def rows_for(scene_idx, lens):
    """Row indices of the stacked arrays belonging to the given scenes."""
    off = np.concatenate([[0], np.cumsum(lens)])
    return np.concatenate([np.arange(off[i], off[i + 1]) for i in scene_idx])


def scene_split_check(ids, fit_idx, sel_idx, val_ids):
    fit_set = set(map(ids.__getitem__, fit_idx))
    sel_set = set(map(ids.__getitem__, sel_idx))
    val_set = set(val_ids)
    if fit_set & sel_set or fit_set & val_set or sel_set & val_set:
        raise RuntimeError('split scene_id intersection non-empty')
    if len(fit_set) != len(fit_idx) or len(sel_set) != len(sel_idx):
        raise RuntimeError('duplicate scene_id inside a split')


def pool_feats(x, pred_masks):
    """Mirror P1 pool(): maxpool 1024->128 per mask, mean of cell features."""
    o = []
    for m in pred_masks:
        t = torch.from_numpy(m.astype(np.float32))[None, None]
        q = torch.nn.functional.max_pool2d(t, 8, 8)[0, 0].numpy().astype(bool).reshape(-1)
        o.append(x[q].mean(0) if q.any() else np.zeros(x.shape[1]))
    return np.asarray(o, np.float32)


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
        # ---- MLP grid ----
        sel_losses = {}
        mlp_models = {}
        for a in ARMS:
            sc = StandardScaler().fit(tr[t]['x'][a][fit_rows])
            xf = sc.transform(tr[t]['x'][a][fit_rows]).astype(np.float32)
            xs_ = sc.transform(tr[t]['x'][a][sel_rows]).astype(np.float32)
            yf, ys_ = y_full[fit_rows], y_full[sel_rows]
            for hi, hidden in enumerate(HIDDENS):
                for wi, wd in enumerate(WDS):
                    tag = f'{t}/{a}/mlp/h{hi}/wd{wi}'
                    m, sl, ep = fit_mlp(xf, yf, xs_, ys_,
                                        hidden, wd, SEED + hi * 10 + wi, cls, telemetry, tag)
                    sel_losses[tag] = sl
                    mlp_models[a, hi, wi] = (m, sc)
        # common config across arms by mean(SAT, correct) select loss
        def pair_score(hi, wi):
            return (sel_losses[f'{t}/SAT/mlp/h{hi}/wd{wi}'] + sel_losses[f'{t}/correct/mlp/h{hi}/wd{wi}']) / 2
        best_cfg = min(((pair_score(hi, wi), hi, wi) for hi in range(len(HIDDENS)) for wi in range(len(WDS))))
        _, hi, wi = best_cfg
        # ---- linear parity anchor ----
        lin = {}
        lin_losses = {}
        for a in ARMS:
            sc = StandardScaler().fit(tr[t]['x'][a][fit_rows])
            xf = sc.transform(tr[t]['x'][a][fit_rows])
            xs_ = sc.transform(tr[t]['x'][a][sel_rows])
            yf, ys_ = y_full[fit_rows], y_full[sel_rows]
            for C in CS:
                tag = f'{t}/{a}/lin/C{C}'
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
                lin[a, C] = (m, sc)

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
        for a in ARMS:
            models[t, a, 'mlp'] = mlp_models[a, hi, wi]
            models[t, a, 'lin'] = lin[a, best_lin]
        log(f'target {t}: mlp hidden={HIDDENS[hi]} wd={WDS[wi]} lin_C={best_lin} ({time.time()-t0:.0f}s)')

    # ---- boundary threshold (prevalence-corrected F1 on select, correct arm) ----
    thresholds = {}
    for reader in ('mlp', 'lin'):
        m, sc = models['boundary', 'correct', reader]
        sel_rows = rows_for(sel_idx, tr['lens']['boundary'])
        xs_ = sc.transform(tr['boundary']['x']['correct'][sel_rows]).astype(np.float32)
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

    # ---- one-shot val evaluation ----
    if smoke:
        vfiles = vfiles[:a.smoke_val]
    rows = []
    for f in vfiles:
        z = np.load(f)
        row = {'scene_id': str(z['scene_id'])}
        for reader in ('mlp', 'lin'):
            t_ = thresholds[reader]['threshold']
            for arm in ARMS:
                x = z[f'x_{arm}'].astype(np.float32)
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
    if degraded:
        decision = 'READCAP_DEGRADED'
    elif f1_pass and auc_pass and rho_pass:
        decision = 'P2_PASS'
    elif f1_pass:
        decision = 'P2_PARTIAL_BOUNDARY'
    elif rho_pass:
        decision = 'P2_PARTIAL_QUALITY'
    else:
        decision = 'P2_NULL'
    summary = {
        'experiment': 'DUAL-P2-READCAP-20261014', 'status': 'complete', 'decision': decision,
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
