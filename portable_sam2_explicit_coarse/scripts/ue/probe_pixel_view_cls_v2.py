#!/usr/bin/env python3
"""PIXEL-CLS v2 (2026-10-21): per-pixel building-view separability of the
frozen SAM2 UAV features on the finalized CitySample archive - the new-era
supply check behind any descriptor-consuming dual arm (EVI-DECODER lineage).

Same gates as v1 (docs/PIXEL-CLS-PROBE-20261017_preregistration.md):
A-PASS = acc > 0.75 AND scene-paired (acc - base_rate) bootstrap CI95
lower > 0 AND d >= 0.05; B-FAIL = d < 0.05; sham (within-scene label
shuffle) must fall back to base rate. Labels via the verified camera path:
descriptor-cell centre -> EXR camera-Z backprojection -> world (x,y) ->
satellite projection -> GT COCO RLE union lookup.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'inference'))
sys.path.insert(0, str(HERE))

import ue.dual  # noqa: F401
from ue.dual import _UAV_MEAN, _UAV_STD, _k_and_w2c
from rsprompter.backbone.sam2_vision import RSSAM2VisionEncoder
from citysample_pair import CitySamplePairDataset

OUTDIR = Path('/data2/wangcheng/checkpoint/surfbev_b0/probes/PIXEL-CLS-20261021')
SEED = 44
BOOT_SEED = 20261014


@torch.no_grad()
def harvest_scene(enc, it, dev):
    """Per-cell (feature [N,C], label [N]) over the 5 views, desc grid."""
    rgb = it['drone_images'].to(dev)
    mean = rgb.new_tensor(_UAV_MEAN)[:, None, None]
    std = rgb.new_tensor(_UAV_STD)[:, None, None]
    _, fpn = enc((rgb - mean) / std)
    desc = max(tuple(fpn[-4:]), key=lambda f: f.shape[-2] * f.shape[-1]).float()
    V, C, fh, fw = desc.shape
    depth = it['depth_cm'].to(dev)                       # [5,512,512] cm
    gt_union = np.zeros((512, 512), dtype=bool)
    for m in it['gt_masks']:
        gt_union |= np.asarray(m).astype(bool)
    gt_t = torch.from_numpy(gt_union).to(dev)
    yy, xx = torch.meshgrid(torch.arange(fh, device=dev), torch.arange(fw, device=dev), indexing='ij')
    px = ((xx.reshape(-1).float() + .5) * 512.0 / fw).long().clamp(max=511)
    py = ((yy.reshape(-1).float() + .5) * 512.0 / fh).long().clamp(max=511)
    z = depth[:, py, px]                                  # [V, fh*fw]
    u = ((xx.reshape(-1).float() + .5) * 512.0 / fw)[None].expand(V, -1)
    w = ((yy.reshape(-1).float() + .5) * 512.0 / fh)[None].expand(V, -1)
    sk, s_w2c, _ = _k_and_w2c(it['satellite_camera'])
    sk, s_w2c = sk.to(dev), s_w2c.to(dev)
    feats, labels = [], []
    for v in range(V):
        k, w2c, c2w = _k_and_w2c(it['uav_cameras'][v])
        k, c2w = k.to(dev), c2w.to(dev)
        zv = z[v]
        pc = torch.stack(((u[v] - k[0, 2]) * zv / k[0, 0],
                          (w[v] - k[1, 2]) * zv / k[1, 1], zv), -1)
        world = pc @ c2w[:3, :3].T + c2w[:3, 3]
        cam = world @ s_w2c[:3, :3].T + s_w2c[:3, 3]
        su = (sk[0, 0] * cam[:, 0] / cam[:, 2].clamp_min(1e-6) + sk[0, 2]).long().clamp(0, 511)
        sv = (sk[1, 1] * cam[:, 1] / cam[:, 2].clamp_min(1e-6) + sk[1, 2]).long().clamp(0, 511)
        valid = (zv > 0) & (cam[:, 2] > 0) & (world[:, 2] < 50000) & (world[:, 2] > -50000)
        lab = gt_t[sv, su] & valid
        feats.append(desc[v].flatten(1).T[valid])
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
    ap.add_argument('--n-val', type=int, default=100)
    ap.add_argument('--device', default='cuda:1')
    a = ap.parse_args()
    dev = torch.device(a.device)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    enc = RSSAM2VisionEncoder(checkpoint_path=os.environ['SAM2_CKPT'],
                              model_size='base_plus', freeze_backbone=True,
                              use_gradient_checkpointing=False,
                              use_lora=False).to(dev).eval()

    ds_tr = CitySamplePairDataset(split='train', limit=a.n_train)
    ds_va = CitySamplePairDataset(split='val', limit=a.n_val)
    Ftr, Ltr, Fva, Lva = [], [], [], []
    for i in range(len(ds_tr)):
        f, l = harvest_scene(enc, ds_tr[i], dev)
        Ftr.append(f.cpu()); Ltr.append(l.cpu())
    for i in range(len(ds_va)):
        f, l = harvest_scene(enc, ds_va[i], dev)
        Fva.append(f.cpu()); Lva.append(l.cpu())
        if (i + 1) % 20 == 0:
            print(f'harvest val {i+1}/{len(ds_va)}', flush=True)
    dim = Ftr[0].shape[1]
    Xtr, ytr = torch.cat(Ftr), torch.cat(Ltr)
    Xva, yva = torch.cat(Fva), torch.cat(Lva)
    base = float(yva.float().mean())
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
    Lsh = []
    g = torch.Generator().manual_seed(SEED)
    for l in Lva:
        Lsh.append(l[torch.randperm(l.numel(), generator=g)])
    ysh = torch.cat(Lsh)
    acc_sh = float((pred == ysh).float().mean())
    d_mean, ci_lo = float(d.mean()), ci[0]
    if acc_sh - base > 0.05:
        verdict = 'SHAM-FAILED'
    elif acc > 0.75 and ci_lo > 0 and d_mean >= 0.05:
        verdict = 'A-PASS'
    elif d_mean < 0.05:
        verdict = 'B-FAIL'
    else:
        verdict = 'PARTIAL'
    res = {'archive': 'CitySample_SatUAV_TrainMinimal_v1_20261021', 'dim': dim,
           'n_train': len(ds_tr), 'n_val': len(ds_va),
           'train_base_rate': float(ytr.float().mean()),
           'readout': {'acc': acc, 'base_rate': base, 'ap': ap,
                       'acc_minus_base': d_mean, 'ci95': ci, 'sham': acc_sh},
           'verdict': verdict}
    print(json.dumps(res, indent=1))
    (OUTDIR / 'readout.json').write_text(json.dumps(res, indent=2))
    print('VERDICT:', verdict, flush=True)


if __name__ == '__main__':
    main()
