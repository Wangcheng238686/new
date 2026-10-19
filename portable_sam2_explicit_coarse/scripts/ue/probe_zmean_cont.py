#!/usr/bin/env python3
"""ZMEAN-CONT probe (2026-10-19 prereg): can frozen-geometry UAV height
evidence (zmean surface, pure DeviceZ backprojection, zero learned params)
classify "real inter-building interface" vs "within-instance component seam"?

Interfaces on the 512 sat grid (r=3):
  class 1 (real gap): pairs of DIFFERENT GT instances with overlapping
    dilations;
  class 0 (seam, S2): pairs of connected components of the SAME instance;
    S1 (erosion-split necks) reported separately.
Primary feature dz = |mean zmean over the two side bands| (cm); controls:
strip gapz / strip missing rate / appearance |delta gray|.

Gate: pooled AUC(dz) >= 0.65 AND scene-paired bootstrap CI95 lower > 0.5.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'inference'))
sys.path.insert(0, str(HERE))

import ue.dual  # noqa: F401
from rsprompter.surfacebev import rasterize_surfacebev
from data.ue_raw_pair_dataset import UERawPairDataset
from ue.dual import _k_and_w2c

SNAP = '/data2/wangcheng/dataset_frozen/ue_raw_hero_uav_pairset_20261014_frozen'
OUTDIR = Path('/data2/wangcheng/checkpoint/surfbev_b0/probes/ZMEAN-CONT-20261019')
BOOT_SEED = 20261014
R = 3          # interface dilation radius (px @512)
MIN_COVER = 0.6


def zmean_surface(it, dev):
    """Frozen-geometry zmean + missing mask at 512x512 (desc=zeros: stats
    channels never touch the descriptor block)."""
    depth = it['devicez_rgb'].to(dev)
    desc = depth.new_zeros((5, 256, 512, 512))
    cams = it['uav_cameras']
    uk = torch.stack([_k_and_w2c(p)[0].to(dev) for p in cams])
    uc = torch.stack([_k_and_w2c(p)[2].to(dev) for p in cams])
    sk, swc, _ = _k_and_w2c(it['satellite_camera'][0] if isinstance(it['satellite_camera'], list)
                            else it['satellite_camera'])
    bev, _ = rasterize_surfacebev(desc, depth, uk, uc, sk.to(dev), swc.to(dev),
                                  torch.tensor([-1e9, 1e9]), (512, 512),
                                  with_grad=False)
    return bev[0, 256].cpu().numpy(), bev[0, 260].cpu().numpy()  # zmean (cm), missing


def interfaces(masks):
    """[(sideA, sideB, strip, kind)] - bool 512x512 pixel sets."""
    out = []
    comps = []          # (instance_idx, component mask) for every instance
    for gi, m in enumerate(masks):
        m = np.asarray(m).astype(bool)
        lab, n = ndimage.label(m)
        for c in range(1, n + 1):
            comps.append((gi, lab == c))
        # S1: erosion-split necks inside one instance
        er = ndimage.binary_erosion(m, iterations=3)
        lab2, n2 = ndimage.label(er)
        if n2 >= 2:
            parts = [(lab2 == c) for c in range(1, n2 + 1)]
            for i in range(len(parts)):
                for j in range(i + 1, len(parts)):
                    A = ndimage.binary_dilation(parts[i], iterations=R)
                    B = ndimage.binary_dilation(parts[j], iterations=R)
                    if (A & B).any():
                        out.append(_pack(parts[i], parts[j], 's1'))
    # S2: adjacent components of the SAME instance
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            if comps[i][0] != comps[j][0]:
                continue
            A = ndimage.binary_dilation(comps[i][1], iterations=R)
            B = ndimage.binary_dilation(comps[j][1], iterations=R)
            if (A & B).any():
                out.append(_pack(comps[i][1], comps[j][1], 's2'))
    # class 1: adjacent DIFFERENT instances
    per_inst = []
    for gi, m in enumerate(masks):
        per_inst.append(np.asarray(m).astype(bool))
    for i in range(len(per_inst)):
        for j in range(i + 1, len(per_inst)):
            A = ndimage.binary_dilation(per_inst[i], iterations=R)
            B = ndimage.binary_dilation(per_inst[j], iterations=R)
            if (A & B).any():
                out.append(_pack(per_inst[i], per_inst[j], 'real'))
    return out


def _pack(A, B, kind):
    dA = ndimage.binary_dilation(A, iterations=R)
    dB = ndimage.binary_dilation(B, iterations=R)
    strip = dA & dB & ~(A | B)
    sd = ndimage.binary_dilation(strip, iterations=2)
    return dict(a=A & sd, b=B & sd, strip=strip, kind=kind)


def band_stats(band, zmean, missing, gray):
    n = int(band.sum())
    if n == 0:
        return None
    valid = band & (missing == 0)
    cov = valid.sum() / max(n, 1)
    z = float(zmean[valid].mean()) if valid.any() else float('nan')
    g = float(gray[valid].mean()) if valid.any() else float('nan')
    return dict(cover=cov, z=z, gray=g)


def auc(scores, labels):
    s, l = np.asarray(scores, float), np.asarray(labels, int)
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # tie-corrected ranks
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    pos, neg = l == 1, l == 0
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2)
                 / (pos.sum() * neg.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-val', type=int, default=100)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    dev = torch.device(a.device)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ds = UERawPairDataset(SNAP, 'val', True, 'surfbev_b0_split_v2.json',
                          image_size=(512, 512))
    rows = []
    for i in range(min(a.n_val, len(ds))):
        it = ds[i]
        zmean, missing = zmean_surface(it, dev)
        gray = it['img'].mean(0).cpu().numpy()          # appearance control
        for ifc in interfaces(it['gt_masks']):
            sa = band_stats(ifc['a'], zmean, missing, gray)
            sb = band_stats(ifc['b'], zmean, missing, gray)
            ss = band_stats(ifc['strip'], zmean, missing, gray)
            if sa is None or sb is None or sa['cover'] < MIN_COVER or sb['cover'] < MIN_COVER:
                continue
            rows.append({'scene': i, 'kind': ifc['kind'],
                         'label': 1 if ifc['kind'] == 'real' else 0,
                         'dz': abs(sa['z'] - sb['z']),
                         'gapz': ss['z'] if ss else float('nan'),
                         'gap_missing': 1.0 - (ss['cover'] if ss else 0.0),
                         'dgray': abs(sa['gray'] - sb['gray'])})
        if (i + 1) % 20 == 0:
            print(f'scene {i+1}: interfaces so far {len(rows)}', flush=True)

    (OUTDIR / 'rows.jsonl').write_text(
        ''.join(json.dumps(r) + '\n' for r in rows))
    by = lambda k, lbl: ([r[k] for r in rows if r['label'] == lbl])
    n1, n0 = sum(r['label'] == 1 for r in rows), sum(r['label'] == 0 for r in rows)
    print(f'interfaces: real={n1} seam={n0} '
          f'(s1={sum(r["kind"]=="s1" for r in rows)}, s2={sum(r["kind"]=="s2" for r in rows)})')

    def scene_auc(key, sel=lambda r: True):
        sub = [r for r in rows if sel(r)]
        per_scene = []
        for sc in sorted({r['scene'] for r in sub}):
            s = [r for r in sub if r['scene'] == sc]
            if min(sum(r['label'] == 1 for r in s), sum(r['label'] == 0 for r in s)) >= 1:
                per_scene.append(auc([r[key] for r in s], [r['label'] for r in s]))
        a_pool = auc([r[key] for r in sub], [r['label'] for r in sub])
        rng = np.random.default_rng(BOOT_SEED)
        d = np.array(per_scene)
        draws = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
        ci = [float(v) for v in np.quantile(draws, [.025, .975])]
        return a_pool, float(d.mean()), ci, len(per_scene)

    res = {}
    a_pool, a_scene, ci, nsc = scene_auc('dz')
    res['dz'] = {'pooled_auc': a_pool, 'scene_mean_auc': a_scene, 'ci95': ci,
                 'n_scenes': nsc}
    for key in ('gapz', 'gap_missing', 'dgray'):
        vals = [r[key] for r in rows]
        if any(v == v for v in vals):                     # not all-nan
            sub = [r for r in rows if r[key] == r[key]]
            res[key] = {'pooled_auc': auc([r[key] for r in sub],
                                          [r['label'] for r in sub])}
    for s1s2 in ('s1', 's2'):
        sub = [r for r in rows if r['kind'] in (s1s2, 'real')]
        if sum(r['label'] == 0 for r in sub) >= 10:
            res[f'dz_vs_{s1s2}'] = {'pooled_auc': auc([r['dz'] for r in sub],
                                                      [r['label'] for r in sub]),
                                    'n_seam': sum(r['label'] == 0 for r in sub)}
    gate = res['dz']
    verdict = 'PASS' if (gate['pooled_auc'] >= 0.65 and gate['ci95'][0] > 0.5) else 'FAIL'
    res['verdict'] = verdict
    res['gate'] = {'rule': 'pooled AUC(dz) >= 0.65 AND scene-paired bootstrap CI95 lower > 0.5'}
    res['dist'] = {'dz_real': by('dz', 1), 'dz_seam': by('dz', 0)}
    (OUTDIR / 'readout.json').write_text(json.dumps(
        {k: v for k, v in res.items() if k != 'dist'}, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k not in ('dist',)}, indent=1)[:1200])
    print('VERDICT:', verdict, flush=True)


if __name__ == '__main__':
    main()
