#!/usr/bin/env python3
"""PIXEL-CLS probe (2026-10-17): per-pixel building-view separability of the
raw UAV-view SAM2 features BEFORE backprojection.

Question (PI spec): is the building signal absent from the source features,
or washed out by the rasterizer's cross-view averaging? Labels come from
exact geometry: descriptor-cell centre pixel x DeviceZ true depth ->
3D world point (surfacebev.py lines 36-44 math) -> world (x,y) via the
dataset satellite contract -> GT instance-id lookup.

Gates (docs/PIXEL-CLS-PROBE-20261017_preregistration.md):
A: acc > 0.75 and acc - base_rate paired-bootstrap CI95 lower > 0
B: acc - base_rate < 0.05
sham (within-scene label shuffle) must fall back to base rate.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'inference'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ue.dual  # noqa: F401
from ue.features import build_baseline
from rsprompter.surfacebev import devicez_to_camera_z
from data.ue_raw_pair_dataset import UERawPairDataset
from ue.dual import _UAV_MEAN, _UAV_STD, _k_and_w2c

SNAP = '/data2/wangcheng/dataset_frozen/ue_raw_hero_uav_pairset_20261014_frozen'
CKPT_DUALNORM = '/data2/wangcheng/checkpoint/surfbev_b0/surfacebev_v2_512dualnorm_sealed_ddp4_s44_EXP-0014/last_model_epoch100.pth'
OUTDIR = Path('/data2/wangcheng/checkpoint/surfbev_b0/probes/PIXEL-CLS-20261017')
SEED = 44
BOOT_SEED = 20261014


@torch.inference_mode()
def harvest_scene(enc, it, dev):
    """Return per-cell (feature [N,C], label [N], view id [N]) for 5 views."""
    rgb = it['drone_images'].to(dev)            # [5,3,512,512]
    mean = rgb.new_tensor(_UAV_MEAN)[:, None, None]
    std = rgb.new_tensor(_UAV_STD)[:, None, None]
    _, fpn = enc((rgb - mean) / std)
    desc = max(tuple(fpn[-4:]), key=lambda f: f.shape[-2] * f.shape[-1]).float()
    V, C, fh, fw = desc.shape
    depth = it['devicez_rgb'].to(dev)           # [5,512,512,3] uint8 packed
    # GT union footprint lives on the satellite 512 grid; world (x,y) is
    # looked up via the dataset satellite contract below.
    yy, xx = torch.meshgrid(torch.arange(fh, device=dev), torch.arange(fw, device=dev), indexing='ij')
    px = ((xx.reshape(-1).float() + .5) * 512.0 / fw).long().clamp(max=511)
    py = ((yy.reshape(-1).float() + .5) * 512.0 / fh).long().clamp(max=511)
    z = devicez_to_camera_z(depth)[:, py, px]   # fn divides by 255 internally     # [V, fh*fw]
    u = ((xx.reshape(-1).float() + .5) * 512.0 / fw)[None].expand(V, -1)
    w = ((yy.reshape(-1).float() + .5) * 512.0 / fh)[None].expand(V, -1)

    feats, labels = [], []
    # GT union footprint at full 512 resolution (bool) -> tensor lookup
    gt_union = np.zeros((512, 512), dtype=bool)
    for m in it['gt_masks']:
        gt_union |= np.asarray(m).astype(bool)
    gt_t = torch.from_numpy(gt_union).to(dev)

    cams = it['uav_cameras']
    # satellite-camera projection for the footprint lookup: the affine
    # contract from the older ue_dual_stream dataset is METRE-scaled and
    # collapses on this cm-coordinate data (debug 2026-10-17); the camera
    # path is the geometry-audit-verified one (2.4 px median alignment).
    sc = it['satellite_camera']
    sk, s_w2c, _ = _k_and_w2c(sc[0] if isinstance(sc, list) else sc)
    sk, s_w2c = sk.to(dev), s_w2c.to(dev)
    for v in range(V):
        k, w2c, c2w = _k_and_w2c(cams[v])
        k, c2w = k.to(dev), c2w.to(dev)
        zv = z[v]                                            # [N]
        pc = torch.stack(((u[v] - k[0, 2]) * zv / k[0, 0],
                          (w[v] - k[1, 2]) * zv / k[1, 1], zv), -1)   # [N,3]
        world = pc @ c2w[:3, :3].T + c2w[:3, 3]              # [N,3]
        cam = world @ s_w2c[:3, :3].T + s_w2c[:3, 3]
        su = (sk[0, 0] * cam[:, 0] / cam[:, 2].clamp_min(1e-6) + sk[0, 2]).long().clamp(0, 511)
        sv = (sk[1, 1] * cam[:, 1] / cam[:, 2].clamp_min(1e-6) + sk[1, 2]).long().clamp(0, 511)
        valid = (zv > 0) & (cam[:, 2] > 0) & (world[:, 2] < 50000)   # sane depth/height
        lab = gt_t[sv, su] & valid                           # [N] bool
        feats.append(desc[v].flatten(1).T[valid])            # [n,C]
        labels.append(lab[valid])
    return torch.cat(feats), torch.cat(labels)


def train_mlp(Xtr, ytr, dim, steps=4000, lr=1e-3, batch=16384, seed=SEED, dev=None):
    torch.manual_seed(seed)
    Xtr, ytr = Xtr.to(dev), ytr.to(dev)
    net = torch.nn.Sequential(
        torch.nn.Linear(dim, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 1),
    ).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    n = Xtr.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, n, (min(batch, n),), device=dev)
        loss = F.binary_cross_entropy_with_logits(net(Xtr[idx]), ytr[idx][:, None].float())
        opt.zero_grad(); loss.backward(); opt.step()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-train', type=int, default=60)
    ap.add_argument('--n-val', type=int, default=40)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    dev = torch.device(a.device)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)

    from ue.contract import checkpoint_identity
    mp = OUTDIR / 'identity_dualnorm.json'
    if not mp.exists():
        ck = torch.load(CKPT_DUALNORM, map_location='cpu', weights_only=False)
        mp.write_text(json.dumps(checkpoint_identity(CKPT_DUALNORM, ck), indent=2))
    model_d, _, _, _ = build_baseline(CKPT_DUALNORM, mp, os.environ['SAM2_CKPT'], dev)
    enc = model_d.uav_encoder

    ds_tr = UERawPairDataset(SNAP, 'train', True, 'surfbev_b0_split_v2.json', image_size=(512, 512))
    ds_va = UERawPairDataset(SNAP, 'val', True, 'surfbev_b0_split_v2.json', image_size=(512, 512))

    def harvest(ds, n):
        Fb, Lb = [], []
        for i in range(n):
            f, l = harvest_scene(enc, ds[i], dev)
            Fb.append(f.cpu()); Lb.append(l.cpu())
            if (i + 1) % 10 == 0:
                print(f'harvest {i+1}/{n}', flush=True)
        return Fb, Lb

    Ftr, Ltr = harvest(ds_tr, a.n_train)
    Fva, Lva = harvest(ds_va, a.n_val)
    dim = Ftr[0].shape[1]
    Xtr, ytr = torch.cat(Ftr), torch.cat(Ltr)
    results = {'n_train': a.n_train, 'n_val': a.n_val, 'dim': dim,
               'train_cells': int(ytr.numel()),
               'train_base_rate': float(ytr.float().mean()),
               'meta': {'feature': 'frozen SAM2 base_plus fpn max-area level, per descriptor cell',
                        'label': 'DeviceZ backproject world (x,y) in GT footprint (satellite contract)',
                        'seed': SEED}}

    Xva, yva = torch.cat(Fva), torch.cat(Lva)
    base = float(yva.float().mean())
    # per-scene base rates for the paired bootstrap
    scene_pos = torch.stack([l.float().mean() for l in Lva])

    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    net = train_mlp((Xtr - mu) / sd, ytr, dim, dev=dev)
    with torch.no_grad():
        logit_va = net(((Xva - mu) / sd).to(dev))[:, 0].cpu()
    pred = logit_va > 0
    acc = float((pred == yva).float().mean())
    scores = torch.sigmoid(logit_va)
    order = np.argsort(-scores.numpy()); t = yva.numpy()[order]
    cum = np.cumsum(t); prec = cum / (np.arange(len(t)) + 1)
    ap = float(prec[np.where(t == 1)[0]].mean()) if (t == 1).any() else 0.0
    # per-scene accuracy for paired bootstrap vs that scene's base rate
    per_acc, per_base, start = [], [], 0
    for l in Lva:
        n = l.numel()
        per_acc.append(float((pred[start:start + n] == l).float().mean()))
        per_base.append(float(l.float().mean()))
        start += n
    d = np.array(per_acc) - np.array(per_base)
    rng = np.random.default_rng(BOOT_SEED)
    draws = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    ci = [float(v) for v in np.quantile(draws, [.025, .975])]
    results['readout'] = {'acc': acc, 'base_rate': base, 'ap': ap,
                          'acc_minus_base': float(d.mean()), 'ci95': ci,
                          'per_scene_acc': per_acc, 'per_scene_base': per_base}

    # sham: within-scene label permutation
    Lsh = []
    g = torch.Generator().manual_seed(SEED)
    for l in Lva:
        Lsh.append(l[torch.randperm(l.numel(), generator=g)])
    ysh = torch.cat(Lsh)
    acc_sh = float((pred == ysh).float().mean())
    results['sham_label_shuffle'] = {'acc': acc_sh,
                                     'note': 'same logits, permuted labels within scene'}

    d_mean, ci_lo = float(d.mean()), ci[0]
    if results['sham_label_shuffle']['acc'] - base > 0.05:
        verdict = 'SHAM-FAILED'
    elif acc > 0.75 and ci_lo > 0 and d_mean >= 0.05:
        verdict = 'A-PASS (attention aggregation has cargo)'
    elif d_mean < 0.05:
        verdict = 'B-FAIL (descriptors dead at source; stay with stats group)'
    else:
        verdict = 'PARTIAL (PI re-adjudication required)'
    results['verdict'] = verdict
    print(json.dumps({k: v for k, v in results.items()
                      if k not in ('readout',)}, ensure_ascii=False))
    r = results['readout']
    print(f"acc={r['acc']:.4f} base={r['base_rate']:.4f} d={r['acc_minus_base']:+.4f} "
          f"CI95=[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] AP={r['ap']:.4f} sham={acc_sh:.4f}")
    print('VERDICT:', verdict, flush=True)
    (OUTDIR / 'readout.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
