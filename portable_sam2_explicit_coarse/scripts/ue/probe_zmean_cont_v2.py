#!/usr/bin/env python3
"""ZMEAN-CONT v2 (2026-10-21): the preregistered post-cleaning re-check of the
continuity probe, on the finalized CitySample archive
(review99_v3_user_approved_20261021). Same protocol/gate as v1
(docs/DUAL-AXIS-SCREEN-20261018_preregistration.md, ZMEAN-CONT section):
classify real inter-building interfaces vs within-instance component seams
using frozen-geometry height evidence; primary gate = pooled AUC(dz) >= 0.65
AND scene-paired bootstrap CI95 lower > 0.5.

Amendments for v2 (logged in the prereg):
- archive swap: exact EXR camera-Z (cm float) replaces the old lossy packed
  devicez PNG; zmean surface computed by the same z-buffer + mean-world-Z
  math as rsprompter.surfacebev (stats path replicated locally because the
  rasterizer hard-codes the packed-depth decode);
- the cleaning removed the multi-connected population entirely
  (20.6% -> 0.0% on val): the seam class is now erosion-split necks (S1)
  only; S2 is reported for completeness (expected empty);
- world Z is real terrain (~-300 cm baseline), not z=0: dz features are
  differences, terrain slope inside the ~8px band adds minor noise.
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
from ue.dual import _k_and_w2c
from citysample_pair import CitySamplePairDataset

OUTDIR = Path('/data2/wangcheng/checkpoint/surfbev_b0/probes/ZMEAN-CONT-20261021')
BOOT_SEED = 20261014
R = 3
MIN_COVER = 0.6
STRIDE = 2          # depth sampling stride per view


def zmean_surface(it, dev):
    """z-buffered mean world-Z per satellite 512 cell, exact EXR depth."""
    depth = it['depth_cm'].to(dev)                                  # [5,512,512]
    V = depth.shape[0]
    sk, swc, _ = _k_and_w2c(it['satellite_camera'])
    sk, swc = sk.to(dev), swc.to(dev)
    ys, xs = torch.meshgrid(torch.arange(0, 512, STRIDE, device=dev),
                            torch.arange(0, 512, STRIDE, device=dev), indexing='ij')
    u = xs.reshape(-1).float()
    w = ys.reshape(-1).float()
    zmean = torch.zeros(512, 512, device=dev)
    count = torch.zeros(512, 512, device=dev)
    zmin = torch.full((512 * 512,), float('inf'), device=dev)
    # pass 1: z-buffer minimum per cell
    per_view = []
    for v in range(V):
        k, _, c2w = _k_and_w2c(it['uav_cameras'][v])
        k, c2w = k.to(dev), c2w.to(dev)
        zv = depth[v][ys, xs].reshape(-1)
        ok = torch.isfinite(zv) & (zv > 0) & (zv < 1e6)
        pc = torch.stack(((u[ok] - k[0, 2]) * zv[ok] / k[0, 0],
                          (w[ok] - k[1, 2]) * zv[ok] / k[1, 1], zv[ok]), -1)
        world = pc @ c2w[:3, :3].T + c2w[:3, 3]
        cam = world @ swc[:3, :3].T + swc[:3, 3]
        su = sk[0, 0] * cam[:, 0] / cam[:, 2].clamp_min(1e-6) + sk[0, 2]
        sv = sk[1, 1] * cam[:, 1] / cam[:, 2].clamp_min(1e-6) + sk[1, 2]
        ix = (su).long().clamp(0, 511)
        iy = (sv).long().clamp(0, 511)
        inb = (su >= 0) & (su < 512) & (sv >= 0) & (sv < 512) & (cam[:, 2] > 0)
        flat = iy.clamp(0, 511) * 512 + ix.clamp(0, 511)
        per_view.append((world, cam, flat, inb))
        zmin.scatter_reduce_(0, flat[inb], cam[inb][:, 2], reduce='amin', include_self=True)
    # pass 2: keep z-buffer winners, accumulate world-Z mean
    for world, cam, flat, inb in per_view:
        keep = inb & (cam[:, 2] <= zmin[flat] + 1e-3)
        zs = world[keep][:, 2]
        cnt = torch.zeros(512 * 512, device=dev)
        cnt.index_add_(0, flat[keep], torch.ones_like(zs))
        acc = torch.zeros(512 * 512, device=dev)
        acc.index_add_(0, flat[keep], zs)
        count += cnt.reshape(512, 512)
        zmean += acc.reshape(512, 512)
    zm = torch.where(count > 0, zmean / count.clamp_min(1), torch.full_like(zmean, float('nan')))
    return zm.cpu().numpy(), (count > 0).cpu().numpy()


def _pack(A, B, kind):
    dA = ndimage.binary_dilation(A, iterations=R)
    dB = ndimage.binary_dilation(B, iterations=R)
    strip = dA & dB & ~(A | B)
    sd = ndimage.binary_dilation(strip, iterations=2)
    return dict(a=A & sd, b=B & sd, strip=strip, kind=kind)


def interfaces(masks):
    out = []
    inst = [np.asarray(m).astype(bool) for m in masks]
    # S1: erosion-split necks inside an instance
    for m in inst:
        er = ndimage.binary_erosion(m, iterations=3)
        lab, n = ndimage.label(er)
        if n >= 2:
            parts = [(lab == c) for c in range(1, n + 1)]
            for i in range(len(parts)):
                for j in range(i + 1, len(parts)):
                    if (ndimage.binary_dilation(parts[i], iterations=R)
                            & ndimage.binary_dilation(parts[j], iterations=R)).any():
                        out.append(_pack(parts[i], parts[j], 's1'))
    # S2: components of the same instance (expected empty post-cleaning)
    comps = []
    for gi, m in enumerate(inst):
        lab, n = ndimage.label(m)
        for c in range(1, n + 1):
            comps.append((gi, lab == c))
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            if comps[i][0] != comps[j][0]:
                continue
            if (ndimage.binary_dilation(comps[i][1], iterations=R)
                    & ndimage.binary_dilation(comps[j][1], iterations=R)).any():
                out.append(_pack(comps[i][1], comps[j][1], 's2'))
    # class 1: adjacent different instances
    for i in range(len(inst)):
        for j in range(i + 1, len(inst)):
            if (ndimage.binary_dilation(inst[i], iterations=R)
                    & ndimage.binary_dilation(inst[j], iterations=R)).any():
                out.append(_pack(inst[i], inst[j], 'real'))
    return out


def band_stats(band, zm, valid, gray):
    n = int(band.sum())
    if n == 0:
        return None
    ok = band & valid
    cov = ok.sum() / max(n, 1)
    z = float(np.nanmean(zm[ok])) if ok.any() else float('nan')
    g = float(gray[ok].mean()) if ok.any() else float('nan')
    return dict(cover=float(cov), z=z, gray=g)


def auc(scores, labels):
    s, l = np.asarray(scores, float), np.asarray(labels, int)
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    pos, neg = l == 1, l == 0
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-val', type=int, default=459)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    dev = torch.device(a.device)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ds = CitySamplePairDataset(split='val', limit=a.n_val)
    rows = []
    for i in range(len(ds)):
        it = ds[i]
        zm, valid = zmean_surface(it, dev)
        gray = it['img'].mean(0).numpy()
        for ifc in interfaces(it['gt_masks']):
            sa = band_stats(ifc['a'], zm, valid, gray)
            sb = band_stats(ifc['b'], zm, valid, gray)
            ss = band_stats(ifc['strip'], zm, valid, gray)
            if sa is None or sb is None or sa['cover'] < MIN_COVER or sb['cover'] < MIN_COVER:
                continue
            if not (np.isfinite(sa['z']) and np.isfinite(sb['z'])):
                continue
            rows.append({'scene': i, 'kind': ifc['kind'],
                         'label': 1 if ifc['kind'] == 'real' else 0,
                         'dz': abs(sa['z'] - sb['z']),
                         'gapz': ss['z'] if ss and np.isfinite(ss['z']) else float('nan'),
                         'dgray': abs(sa['gray'] - sb['gray'])})
        if (i + 1) % 50 == 0:
            print(f'scene {i+1}/{len(ds)}: interfaces {len(rows)}', flush=True)
    (OUTDIR / 'rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    n1 = sum(r['label'] == 1 for r in rows)
    n0 = sum(r['label'] == 0 for r in rows)
    ns1 = sum(r['kind'] == 's1' for r in rows)
    ns2 = sum(r['kind'] == 's2' for r in rows)
    print(f'interfaces: real={n1} seam={n0} (s1={ns1} s2={ns2})')

    per_scene = []
    for sc in sorted({r['scene'] for r in rows}):
        s = [r for r in rows if r['scene'] == sc]
        if min(sum(r['label'] == 1 for r in s), sum(r['label'] == 0 for r in s)) >= 1:
            per_scene.append(auc([r['dz'] for r in s], [r['label'] for r in s]))
    a_pool = auc([r['dz'] for r in rows], [r['label'] for r in rows])
    rng = np.random.default_rng(BOOT_SEED)
    d = np.array(per_scene)
    draws = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    ci = [float(v) for v in np.quantile(draws, [.025, .975])]
    res = {'dz': {'pooled_auc': a_pool, 'scene_mean_auc': float(d.mean()),
                  'ci95': ci, 'n_scenes': len(per_scene)},
           'counts': {'real': n1, 'seam': n0, 's1': ns1, 's2': ns2}}
    for key in ('gapz', 'dgray'):
        sub = [r for r in rows if r[key] == r[key]]
        if len(sub) > 10:
            res[key] = {'pooled_auc': auc([r[key] for r in sub], [r['label'] for r in sub])}
    med = lambda lbl: float(np.median([r['dz'] for r in rows if r['label'] == lbl])) if any(r['label'] == lbl for r in rows) else None
    res['dz_median'] = {'real': med(1), 'seam': med(0)}
    verdict = 'PASS' if (a_pool >= 0.65 and ci[0] > 0.5) else 'FAIL'
    res['verdict'] = verdict
    res['gate'] = {'rule': 'pooled AUC(dz) >= 0.65 AND scene-paired bootstrap CI95 lower > 0.5'}
    (OUTDIR / 'readout.json').write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1)[:900])
    print('VERDICT:', verdict, flush=True)


if __name__ == '__main__':
    main()
