#!/usr/bin/env python3
"""DISTILL-P0 static gates: G1 (structure) + G2f (frozen differential).

G1  S = E_val|T_hat - mu_spatial| / E_val|T_hat - c_med|  per teacher
    (desc block and whitened zmean); kills a pure-noise teacher (S->1).
    Pure-template teachers pass G1 and are caught by the true/sham
    differential instead.
G2f ridge readout from the frozen EXP-0009 last P3 (512d/cell) to the
    whitened teacher; per-scene val R2, true minus sham, scene-paired
    bootstrap (10k draws, seed 20261014). CI lower bound > 0 passes.

Reads the teacher cache written by harvest_distill_teacher_cache.py.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'inference'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ue.dual  # noqa: F401
from ue.features import build_baseline, model_input
from rsprompter.detection.anchor_detector import RSPrompterAnchor
from data.ue_raw_pair_dataset import UERawPairDataset

SNAP = '/data2/wangcheng/dataset_frozen/ue_raw_hero_uav_pairset_20261014_frozen'
CKPT_SINGLE = '/data2/wangcheng/checkpoint/surfbev_b0/surfacebev_v2_512_sealed_ddp4_s44_EXP-0009/last_model_epoch100.pth'
BASE = Path('/data2/wangcheng/checkpoint/surfbev_b0/probes/DISTILL-P0-20261016')
CACHE = BASE / 'cache'
SEED = 20261014
N_FIT = 100   # train scenes for the ridge
N_VAL = 100


def load_cache(split, mode, sid):
    z = np.load(CACHE / f'{split}_{mode}' / f'{sid}.npz')
    return z['desc'].astype(np.float32), z['zmean'].astype(np.float32), z['mask']


def whitening():
    w = json.loads((CACHE / 'whitening.json').read_text())
    return (torch.tensor(w['desc_mu']), torch.tensor(w['desc_sigma']),
            w['zmean_mu'], w['zmean_sigma'])


def scene_ids(split):
    ds = UERawPairDataset(SNAP, split, False, 'surfbev_b0_split_v2.json', image_size=(512, 512))
    return [r['scene_dir'] for r in ds.records]


@torch.inference_mode()
def harvest_p3(model, split, ids, dev):
    ds = UERawPairDataset(SNAP, split, False, 'surfbev_b0_split_v2.json', image_size=(512, 512))
    out = {}
    for idx in range(len(ds)):
        it = ds[idx]
        sid = it['img_metas']['scene_id']
        if sid not in ids:
            continue
        data = model_input(model, it, dev)
        p3 = RSPrompterAnchor.extract_feat(model, data['inputs'])[0][1][0].float()
        out[sid] = p3.flatten(1).T.cpu()  # [4096, 512]
    return out


def ridge_cv_fit(X, Y, folds=5):
    n = X.shape[0]
    idx = torch.randperm(n)
    xm, ym = X.mean(0, keepdim=True), Y.mean(0, keepdim=True)
    Xc, Yc = X - xm, Y - ym
    best = (None, -1e18)
    for lam in torch.logspace(1, 8, 15):
        scores = []
        for f in range(folds):
            va = idx[f::folds]
            tr = idx[~torch.isin(idx, va)]
            A = Xc[tr].T @ Xc[tr] + lam * torch.eye(X.shape[1])
            W = torch.linalg.solve(A, Xc[tr].T @ Yc[tr])
            pred = Xc[va] @ W + ym
            ss_res = ((Y[va] - pred) ** 2).sum()
            ss_tot = ((Y[va] - ym) ** 2).sum().clamp_min(1e-9)
            scores.append(1 - float(ss_res / ss_tot))
        s = float(np.mean(scores))
        if s > best[1]:
            best = (lam, s)
    lam = best[0]
    A = Xc.T @ Xc + lam * torch.eye(X.shape[1])
    return torch.linalg.solve(A, Xc.T @ Yc), xm, ym


def r2_per_scene(P3s, teachers, W, xm, ym, mode):
    rows = []
    for sid in sorted(P3s):
        X = P3s[sid]
        desc, zmean, mask = load_cache('val', mode, sid)
        M = torch.from_numpy(mask)
        Y = torch.from_numpy(desc) if teachers == 'desc' else torch.from_numpy(zmean)[None]
        if teachers == 'desc':
            mu, sigma, _, _ = whitening()
            Y = (Y - mu[:, None, None]) / sigma[:, None, None]
            Y = Y.flatten(1).T
        else:
            _, _, zmu, zsig = whitening()
            Y = (Y - zmu) / zsig
            Y = Y.flatten(1).T
        Xc = (X - xm)[M.flatten()]
        Yc = (Y - ym)[M.flatten()]
        pred = Xc @ W
        ss_res = ((Yc - pred) ** 2).sum(0)
        ss_tot = ((Yc - Yc.mean(0, keepdim=True)) ** 2).sum(0).clamp_min(1e-9)
        rows.append(float((1 - ss_res / ss_tot).mean()))
    return np.array(rows)


def boot(x):
    r = np.random.default_rng(SEED)
    d = x[r.integers(0, len(x), (10000, len(x)))].mean(1)
    return {'mean': float(x.mean()), 'ci95': [float(v) for v in np.quantile(d, [.025, .975])]}


def g1_stats():
    res = {}
    tr_ids = scene_ids('train')
    va_ids = scene_ids('val')
    for teacher in ('desc', 'height'):
        acc_sum = acc_cnt = None
        vals = []
        for sid in tr_ids:
            desc, zmean, mask = load_cache('train', 'true', sid)
            T = torch.from_numpy(desc if teacher == 'desc' else zmean[None])  # [C,64,64]
            M = torch.from_numpy(mask)
            if acc_sum is None:
                acc_sum = torch.zeros_like(T)
                acc_cnt = torch.zeros(1, *T.shape[1:])
            acc_sum += T * M[None]
            acc_cnt += M[None].float()
            vals.append(T.flatten(1).T[M.flatten().bool()])
        mu_spatial = acc_sum / acc_cnt.clamp_min(1)
        c_med = torch.cat(vals).median(0).values
        num = den = 0.0
        for sid in va_ids:
            desc, zmean, mask = load_cache('val', 'true', sid)
            T = torch.from_numpy(desc if teacher == 'desc' else zmean[None])
            Mf = torch.from_numpy(mask).flatten().bool()
            Tv = T.flatten(1).T[Mf]
            mu_v = mu_spatial.flatten(1).T[Mf]
            num += float((Tv - mu_v).abs().sum())
            den += float((Tv - c_med[None]).abs().sum())
        res[teacher] = num / den
    return res


def main():
    torch.manual_seed(SEED)
    dev = torch.device(os.environ.get('P0_DEVICE', 'cuda:0'))
    BASE.mkdir(parents=True, exist_ok=True)

    s = g1_stats()
    tau = min(0.7, s['height'] + 0.1)
    g1 = {'S_desc': s['desc'], 'S_height': s['height'], 'tau1': tau,
          'desc_pass': bool(s['desc'] < tau)}
    print('G1', json.dumps(g1), flush=True)

    # G2f: frozen P3 -> teacher, true vs sham
    from ue.contract import checkpoint_identity
    manifest = BASE / 'identity_single.json'
    if not manifest.exists():
        ck = torch.load(CKPT_SINGLE, map_location='cpu', weights_only=False)
        manifest.write_text(json.dumps(checkpoint_identity(CKPT_SINGLE, ck), indent=2))
    model, _, identity, _ = build_baseline(CKPT_SINGLE, manifest, os.environ['SAM2_CKPT'], dev)
    tr_ids = scene_ids('train')[:N_FIT]
    va_ids = scene_ids('val')
    P3tr = harvest_p3(model, 'train', set(tr_ids), dev)
    P3va = harvest_p3(model, 'val', set(va_ids), dev)
    del model
    torch.cuda.empty_cache()

    g2f = {}
    for teacher in ('desc', 'height'):
        mu, sigma, zmu, zsig = whitening()
        per_mode = {}
        for mode in ('true', 'sham'):
            Xs, Ys = [], []
            for sid in tr_ids:
                desc, zmean, mask = load_cache('train', mode, sid)
                M = torch.from_numpy(mask).flatten()
                Y = torch.from_numpy(desc) if teacher == 'desc' else torch.from_numpy(zmean)[None]
                if teacher == 'desc':
                    Y = ((Y - mu[:, None, None]) / sigma[:, None, None]).flatten(1).T
                else:
                    Y = ((Y - zmu) / zsig).flatten(1).T
                Xs.append(P3tr[sid][M])
                Ys.append(Y[M])
            W, xm, ym = ridge_cv_fit(torch.cat(Xs), torch.cat(Ys))
            per_mode[mode] = r2_per_scene(P3va, teacher, W, xm, ym, mode)
        d = per_mode['true'] - per_mode['sham']
        g2f[teacher] = {'r2_true': boot(per_mode['true']), 'r2_sham': boot(per_mode['sham']),
                        'diff': boot(d), 'pass': bool(boot(d)['ci95'][0] > 0)}
        print(f'G2f[{teacher}]', json.dumps(g2f[teacher]), flush=True)

    out = {'identity_single': identity.get('baseline_id'), 'g1': g1, 'g2f': g2f,
           'seed': SEED, 'draws': 10000}
    (BASE / 'static_gates.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
