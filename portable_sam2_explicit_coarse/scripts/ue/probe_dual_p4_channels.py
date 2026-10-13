#!/usr/bin/env python3
"""DUAL-P4-CHANS: channel-split re-harvest + readout on the frozen B0 anchor.

Harvest stores pure per-channel features (sat128 single projection; 120-dim
correct/sham SurfaceBEV channel descriptors, never mixed with SAT). Readout
arms: SAT=sat128, correct=[sat|bev_c], sham90=[sat|bev_s]; protocol identical
to DUAL-P2/P3. train+val only; test is never read.
See docs/DUAL-P4-CHANS-20261014_preregistration.md.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import cv2, numpy as np, torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.ue_raw_pair_dataset import UERawPairDataset
from ue.features import observe_satellite, implementation_fingerprints
from probe_surfacebev_p1_h0 import cameras, make_uav_encoder, uav_descriptors
from rsprompter.surfacebev import rasterize_surfacebev
from probe_dual_p2_readcap import (ARMS, TARGETS, HIDDENS, WDS, CS, THRS, MAX_EPOCH,
                                   f1, boot, fit_mlp, predict, rows_for,
                                   scene_split_check, pool_feats)

SEED = 20261014
CHECKPOINT_SHA = 'fc3ead0e9490ffd9249cdbc4ce2ee9556bed5252efc3e475dfcc0f2aace65145'
H0_PATH = Path('/data1/wangcheng/checkpoint/surfbev_b0/probes/SURFBEV-P1-V2-H0-full-20261012/summary.json')
VERSION = 'DUAL-P4-CHANS-v1'


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def boundary(m):
    m = m.astype(np.uint8)
    return cv2.dilate(m, np.ones((5, 5), np.uint8)) - cv2.erode(m, np.ones((5, 5), np.uint8)) > 0


def p3mask(m):
    return F.max_pool2d(torch.from_numpy(m.astype(np.float32))[None, None], 8, 8)[0, 0].numpy().astype(np.uint8).reshape(-1)


def projections(dev):
    r = np.random.default_rng(SEED)
    return {'sat': (r.normal(size=(512, 128)) / np.sqrt(128)).astype('float32'),
            'desc': torch.tensor((r.normal(size=(256, 4)) / 16).astype('float32'), device=dev)}


def iiou(pred, gt):
    if not len(pred):
        return np.zeros(0, np.float32)
    if not len(gt):
        return np.zeros(len(pred), np.float32)
    a = pred.reshape(len(pred), -1).astype(np.int32)
    b = gt.reshape(len(gt), -1).astype(np.int32)
    i = a @ b.T
    d = a.sum(1)[:, None] + b.sum(1)[None] - i
    return (i / np.maximum(d, 1)).max(1).astype(np.float32)


def channels(p3, surf, p):
    """sat128 (16384,128) and bev120 (16384,120) exactly like P1's pre-comb halves."""
    sat = torch.from_numpy(p3).float().permute(1, 2, 0).reshape(-1, 512).numpy() @ p['sat']
    q = torch.einsum('bchw,cd->bdhw', surf[:, :256], p['desc']).permute(0, 2, 3, 1)
    q = torch.cat((q, surf[:, 256:].permute(0, 2, 3, 1)), 3).permute(1, 2, 0, 3).reshape(-1, 120)
    return sat.astype('float32'), q.cpu().numpy().astype('float32')


def pool(x, masks):
    o = []
    for m in masks:
        q = p3mask(m).astype(bool)
        o.append(x[q].mean(0) if q.any() else np.zeros(x.shape[1]))
    return np.asarray(o, np.float32)


def pack_scene(path, split, i, item, p3, pred, sc, ss, p):
    sat_c, bev_cc = channels(p3, sc, p)
    _, bev_cs = channels(p3, ss, p)
    gt = np.any(item['gt_masks'].astype(bool), 0) if len(item['gt_masks']) else np.zeros((1024, 1024), bool)
    yb = p3mask(boundary(gt))
    pe = np.any(pred, 0) if len(pred) else np.zeros_like(gt)
    ye = p3mask(boundary(pe) ^ boundary(gt))
    q = iiou(pred, item['gt_masks'])
    d = {'scene_id': np.array(item['img_metas']['scene_id']), 'yb': yb, 'ye': ye,
         'pred': pred, 'qi': q}
    if split == 'train':
        for tid, (name, y) in enumerate((('boundary', yb), ('error', ye))):
            pos = np.flatnonzero(y)
            neg = np.flatnonzero(~y.astype(bool))
            n = min(256, len(pos), len(neg))
            if not n:
                raise RuntimeError('class missing')
            r = np.random.default_rng(np.random.SeedSequence([SEED, i, tid]))
            sel = np.r_[r.choice(pos, n, False), r.choice(neg, n, False)]
            d['y_' + name] = y[sel]
            d['sat_' + name] = sat_c[sel]
            d['bevc_' + name] = bev_cc[sel]
            d['bevs_' + name] = bev_cs[sel]
        d['y_iou'] = q
        d['iou_sat'] = pool(sat_c, pred)
        d['iou_bevc'] = pool(bev_cc, pred)
        d['iou_bevs'] = pool(bev_cs, pred)
    else:
        d['x_sat'] = sat_c
        d['x_bevc'] = bev_cc
        d['x_bevs'] = bev_cs
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'wb') as f:
        np.savez_compressed(f, **d)
    os.replace(tmp, path)


def load_anchor(checkpoint_path, sam2_ckpt, device):
    """Pinned R1a-equivalent loader: the EXP-0006 anchor predates the
    ue_baseline sealing introduced after the P1-V2 run, so the current
    build_baseline cannot load it. Pin identity by checkpoint SHA +
    architecture contract + strict load + SAM2 SHA, exactly the checks the
    807d220-era harvest relied on (see prereg addendum)."""
    from inference.infer_from_checkpoint import (_load_checkpoint, _load_model_state,
                                                 _register_and_build, _resolve_model_config,
                                                 _resolve_sam2_repo, _select_state_dict, _snapshot)
    from ue.contract import assert_whu_model, architecture_contract, whu_model_config, file_sha256
    ckpt_sha = sha256_file(checkpoint_path)
    if ckpt_sha != CHECKPOINT_SHA:
        raise RuntimeError(f'anchor checkpoint sha mismatch: {ckpt_sha}')
    if not sam2_ckpt:
        raise RuntimeError('SAM2 checkpoint required')
    checkpoint = _load_checkpoint(checkpoint_path)
    snapshot = _snapshot(checkpoint)
    args = argparse.Namespace(config=None, sam2_ckpt=sam2_ckpt, sam2_repo=None)
    _resolve_sam2_repo(args, snapshot)
    cfg, source = _resolve_model_config(args, snapshot)
    assert_whu_model(cfg)
    expected_arch = architecture_contract(whu_model_config())
    if snapshot.get('architecture_contract') != expected_arch:
        raise RuntimeError('snapshot architecture contract mismatch')
    sam2_sha = snapshot.get('runtime_config', {}).get('sam2_checkpoint_sha256')
    if not sam2_sha or file_sha256(str(sam2_ckpt)) != sam2_sha:
        raise RuntimeError('SAM2 checkpoint differs from snapshot runtime config')
    model = _register_and_build(cfg).to(device).eval()
    report = _load_model_state(model, _select_state_dict(checkpoint, 'model'), False)
    if report['missing_keys'] or report['unexpected_keys']:
        raise RuntimeError('anchor must strict-load')
    return model, {'checkpoint_sha256': ckpt_sha, 'architecture_contract': expected_arch,
                   'sam2_checkpoint_sha256': sam2_sha, 'config_source': source,
                   'loading_mode': 'pinned-r1a-equivalent'}


def run_harvest(a, out, h):
    cache = out / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    model, identity = load_anchor(a.checkpoint, a.sam2_ckpt, dev)
    ed = torch.tensor(h['height_edges_cm'], device=dev)
    p = projections(dev)
    contract_json = {'version': VERSION, 'seed': SEED, 'checkpoint_sha256': identity['checkpoint_sha256'],
                     'anchor_identity': identity, 'h0_sha256': sha256_file(H0_PATH),
                     'split': 'surfbev_b0_split_v2.json', 'sat_dim': 128, 'bev_dim': 120,
                     'arms': list(ARMS), 'per_class_cap': 256,
                     'implementation_fingerprints': implementation_fingerprints()}
    (cache / 'contract.json').write_text(json.dumps(contract_json, indent=2))
    enc_arg = argparse.Namespace(device=a.device, sam2_ckpt=a.sam2_ckpt)
    encoder = make_uav_encoder(enc_arg, dev)
    ds = {s: UERawPairDataset(a.root, s, True, 'surfbev_b0_split_v2.json', load_targets=True)
          for s in ('train', 'val')}
    tele = cache / 'telemetry_train_val.jsonl'
    for s in ('train', 'val'):
        n = len(ds[s]) if s == 'train' or not a.max_val else min(a.max_val, len(ds[s]))
        if s == 'train' and a.max_train:
            n = min(a.max_train, n)
        d = cache / s
        d.mkdir(exist_ok=True)
        for i in range(n):
            fpath = d / f'{i:04d}.npz'
            if fpath.exists():
                print(s, i + 1, '/', n, 'cached', flush=True)
                continue
            it = ds[s][i]
            features, outputs, ph = observe_satellite(model, it, dev)
            p3 = features.pyramids[1][0].float().detach().cpu().numpy()
            x = getattr(outputs[0].pred_instances, 'masks')
            x = x.to_ndarray() if hasattr(x, 'to_ndarray') else (x.masks if hasattr(x, 'masks') else x)
            pred = (x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)).astype(bool)
            desc = uav_descriptors(encoder, it['drone_images'].to(dev))
            uk, uc, sk, sw = cameras(it, dev)
            sc, ct = rasterize_surfacebev(desc, it['devicez_rgb'].to(dev), uk, uc, sk, sw, ed, (128, 128))
            ss, st = rasterize_surfacebev(desc, it['devicez_rgb'].to(dev), uk, uc, sk, sw, ed, (128, 128), True)
            if ct['input_points'] != st['input_points'] or not (torch.isfinite(sc).all() and torch.isfinite(ss).all()):
                raise RuntimeError('surface contract')
            for z in (ct, st):
                if z['zbuffer_points'] <= 0 or z['covered_cells'] <= 0 or not np.isfinite(z['zbuffer_residual_mean']):
                    raise RuntimeError('SurfaceBEV telemetry contract')
            pack_scene(fpath, s, i, it, p3, pred, sc, ss, p)
            with open(tele, 'a') as q:
                q.write(json.dumps({'split': s, 'scene_id': it['img_metas']['scene_id'],
                                    'prediction_hash': ph, 'input_equal': True,
                                    'correct': ct, 'sham90': st}) + '\n')
            print(s, i + 1, '/', n, flush=True)
    if a.max_train or a.max_val:
        json.dump({'status': 'SMOKE_ONLY', 'train_used': a.max_train, 'val_used': a.max_val,
                   'test_read': False}, open(out / 'smoke_summary.json', 'w'), indent=1)
        return
    json.dump({'train': len(ds['train']), 'val': len(ds['val'])},
              open(cache / 'train_val_complete.json', 'w'), indent=1)


def load_train(cache: Path):
    """Per-target stacks: boundary/error/iou rows must never be interleaved."""
    files = sorted((cache / 'train').glob('*.npz'))
    ids, yb_full = [], []
    lens = {t: [] for t in list(TARGETS) + ['iou']}
    frag = {t: {k: [] for k in ('sat', 'bevc', 'bevs')} for t in list(TARGETS) + ['iou']}
    ydata = {t: [] for t in list(TARGETS) + ['iou']}
    for f in files:
        z = np.load(f)
        ids.append(str(z['scene_id']))
        for t in TARGETS:
            for k in ('sat', 'bevc', 'bevs'):
                frag[t][k].append(z[f'{k}_{t}'])
            y = z[f'y_{t}']
            ydata[t].append(y)
            lens[t].append(len(y))
        for k, name in (('sat', 'iou_sat'), ('bevc', 'iou_bevc'), ('bevs', 'iou_bevs')):
            frag['iou'][k].append(z[name])
        q = z['y_iou']
        ydata['iou'].append(q)
        lens['iou'].append(len(q))
        yb_full.append(float(z['yb'].mean()))
        z.close()
    out = {'ids': ids, 'pi_yb': yb_full, 'lens': lens}
    for t in list(TARGETS) + ['iou']:
        out[t] = {'x': {k: np.concatenate(v) for k, v in frag[t].items()},
                  'y': np.concatenate(ydata[t])}
    return out


def concat_arm(x, arm):
    if arm == 'SAT':
        return x['sat']
    if arm == 'correct':
        return np.concatenate([x['sat'], x['bevc']], axis=1)
    return np.concatenate([x['sat'], x['bevs']], axis=1)


def run_readout(a, out):
    cache = out / 'cache'
    counts = {s: len(list((cache / s).glob('*.npz'))) for s in ('train', 'val')}
    if counts != {'train': 694, 'val': 100}:
        raise RuntimeError(f'cache counts {counts} incomplete')
    tr = load_train(cache)
    n_train = len(tr['ids'])
    vfiles = sorted((cache / 'val').glob('*.npz'))
    val_ids = []
    for f in vfiles:
        z = np.load(f)
        val_ids.append(str(z['scene_id']))
        z.close()
    r = np.random.default_rng(SEED)
    perm = r.permutation(n_train)
    fit_idx, sel_idx = perm[:555], perm[555:]
    scene_split_check(tr['ids'], fit_idx, sel_idx, val_ids)
    json.dump({'seed': SEED, 'n_train': n_train, 'fit': [tr['ids'][i] for i in fit_idx],
               'select': [tr['ids'][i] for i in sel_idx], 'val': val_ids},
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
                    m, sl, ep = fit_mlp(xf, yf, xs_, ys_, hidden, wd,
                                        SEED + hi * 10 + wi, cls, telemetry, tag)
                    sel_losses[tag] = sl
                    mlp_models[arm, hi, wi] = (m, sc)

        def pair_score(hi, wi):
            return (sel_losses[f'{t}/SAT/mlp/h{hi}/wd{wi}'] + sel_losses[f'{t}/correct/mlp/h{hi}/wd{wi}']) / 2
        _, hi, wi = min(((pair_score(hi, wi), hi, wi)
                         for hi in range(len(HIDDENS)) for wi in range(len(WDS))))
        lin, lin_losses = {}, {}
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
                    pl = m.predict_proba(xs_)[:, 1]
                    sl = float(log_loss(ys_, pl, labels=[0, 1]))
                else:
                    m = Ridge(alpha=1 / C).fit(xf, yf)
                    pl = m.predict(xs_)
                    sl = float(np.mean((pl - ys_) ** 2))
                lin_losses[tag] = sl
                lin[arm, C] = (m, sc)
        best_lin = min(((lin_losses[f'{t}/SAT/lin/C{C}'] + lin_losses[f'{t}/correct/lin/C{C}']) / 2, C)
                       for C in CS)[1]
        selection['targets'][t] = {
            'mlp_hidden': list(HIDDENS[hi]), 'mlp_wd': WDS[wi],
            'mlp_select_loss_by_arm': {ar: sel_losses[f'{t}/{ar}/mlp/h{hi}/wd{wi}'] for ar in ARMS},
            'lin_C': best_lin,
            'lin_select_loss_by_arm': {ar: lin_losses[f'{t}/{ar}/lin/C{best_lin}'] for ar in ARMS}}
        for arm in ARMS:
            models[t, arm, 'mlp'] = mlp_models[arm, hi, wi]
            models[t, arm, 'lin'] = lin[arm, best_lin]
        print(f'target {t}: hidden={HIDDENS[hi]} wd={WDS[wi]} lin_C={best_lin}', flush=True)
    thresholds = {}
    for reader in ('mlp', 'lin'):
        m, sc = models['boundary', 'correct', reader]
        sel_rows = rows_for(sel_idx, tr['lens']['boundary'])
        xa = concat_arm(tr['boundary']['x'], 'correct')
        xs_ = sc.transform(xa[sel_rows]).astype(np.float32)
        pl = predict(m, xs_) if reader == 'mlp' else m.predict_proba(xs_)[:, 1]
        y = tr['boundary']['y'][sel_rows]
        pi = float(np.mean([tr['pi_yb'][i] for i in sel_idx]))
        best_t, best_f = None, -1.0
        for t_ in THRS:
            pr = pl >= t_
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
    rows = []
    for f in vfiles:
        z = np.load(f)
        row = {'scene_id': str(z['scene_id'])}
        xv = {'sat': z['x_sat'].astype(np.float32), 'bevc': z['x_bevc'].astype(np.float32),
              'bevs': z['x_bevs'].astype(np.float32)}
        for reader in ('mlp', 'lin'):
            t_ = thresholds[reader]['threshold']
            for arm in ARMS:
                x = concat_arm(xv, arm)
                mb, sb = models['boundary', arm, reader]
                me, se = models['error', arm, reader]
                xb = sb.transform(x).astype(np.float32)
                xe = se.transform(x).astype(np.float32)
                pb = predict(mb, xb) if reader == 'mlp' else mb.predict_proba(xb)[:, 1]
                pe = predict(me, xe) if reader == 'mlp' else me.predict_proba(xe)[:, 1]
                if len(z['pred']):
                    pooled = np.concatenate([pool(xv['sat'], z['pred']), pool(xv['bevc' if arm == 'correct' else 'bevs'], z['pred'])], axis=1) \
                        if arm != 'SAT' else pool(xv['sat'], z['pred'])
                else:
                    pooled = np.zeros((0, x.shape[1]), np.float32)
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
        decision = 'P4_PASS'
    elif not (f1_pass or auc_pass or rho_pass):
        decision = 'P4_NULL'
    else:
        pairs = [n for n, ok in (('f1', f1_pass), ('auc', auc_pass), ('rho', rho_pass)) if ok]
        decision = 'P4_PARTIAL(' + ','.join(pairs) + ')'
    boundary_closed = not (gate['correct_minus_SAT_f1'] or gate['correct_minus_sham90_f1'])
    summary = {'experiment': 'DUAL-P4-CHANS-20261014', 'status': 'complete', 'decision': decision,
               'passed_gates': passed_gates,
               'route_closure_clause': {'cell_level_surfacebev_descriptor_closed': bool(boundary_closed)},
               'n': {'fit': 555, 'select': len(sel_idx), 'val': len(rows), 'test_read': 0},
               'thresholds': thresholds,
               'arms_mean': {r: {arm: {q: float(met[r][arm][q].mean()) for q in ('f1', 'auc', 'rho')}
                              for arm in ARMS} for r in ('mlp', 'lin')},
               'bootstrap_mlp': diffs, 'gates': gate,
               'parity_sentinel': {'sat_mlp_f1': sat_mlp_f1, 'sat_lin_f1': sat_lin_f1,
                                   'degraded': bool(degraded)},
               'selection': selection['targets']}
    json.dump(summary, open(out / 'summary.json', 'w'), indent=1)
    print(json.dumps({'decision': decision, 'gates': gate,
                      'boundary_closed': bool(boundary_closed)}, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=('harvest', 'readout'), required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--checkpoint', type=Path,
                    default=Path('/data1/wangcheng/checkpoint/surfbev_b0/surfacebev_b0_v2_raw1024_noaug_ddp4_s44_EXP-0006/last_model_epoch100.pth'))
    ap.add_argument('--root', type=Path, default=ROOT / 'data/ue_raw_hero_uav_pairset_20261011')
    ap.add_argument('--sam2-ckpt', default=os.environ.get('SAM2_CKPT'))
    ap.add_argument('--max-train', type=int, default=0)
    ap.add_argument('--max-val', type=int, default=0)
    a = ap.parse_args()
    h = json.loads(H0_PATH.read_text())
    if h.get('verdict') != 'H0_PASS':
        raise RuntimeError('H0 artifact not passed')
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()) and not (out / 'cache/contract.json').is_file() and not (a.max_train or a.max_val):
        raise RuntimeError('output directory contains unidentified artifacts')
    if a.phase == 'harvest':
        run_harvest(a, out, h)
    else:
        run_readout(a, out)


if __name__ == '__main__':
    main()
