#!/usr/bin/env python3
"""EVI-DECODER probe (2026-10-19, arm E prereg): supervised bridge supply.

Trains a per-view conv decoder (frozen SAM2 base_plus descriptors, NO LoRA)
to output [building-evidence, footprint-boundary] logits on the 64x64
descriptor grid. Labels reuse the PIXEL-CLS geometry verbatim: per-cell
DeviceZ backprojection -> world (x,y) -> GT union footprint lookup; the
boundary channel is dilate3 ∧ ¬erode3 of the per-view evidence label.

Gate (docs/DUAL-AXIS-SCREEN-20261018_preregistration.md, arm E):
  evidence AP >= 0.90 and evidence Dice >= 0.55 on held-out val scenes;
  boundary Dice >= 0.25; per-scene paired bootstrap of Dice - base-map Dice
  CI95 lower > 0. Fail -> arm E does not launch (no decoded product).
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

import ue.dual  # noqa: F401  (registers the model space)
from rsprompter.backbone.sam2_vision import RSSAM2VisionEncoder
from rsprompter.surfacebev import devicez_to_camera_z
from data.ue_raw_pair_dataset import UERawPairDataset
from ue.dual import _UAV_MEAN, _UAV_STD, _k_and_w2c

SNAP = '/data2/wangcheng/dataset_frozen/ue_raw_hero_uav_pairset_20261014_frozen'
OUTDIR = Path('/data2/wangcheng/checkpoint/surfbev_b0/probes/EVI-DECODER-20261019')
SEED = 44
BOOT_SEED = 20261014


class EviDecoder(torch.nn.Module):
    """Preregistered architecture: 256->128->64->2, GN8 + ReLU, shared views."""

    def __init__(self, in_ch=256, hidden=(128, 64), out_ch=2):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, hidden[0], 3, padding=1, bias=False),
            torch.nn.GroupNorm(8, hidden[0]),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(hidden[0], hidden[1], 3, padding=1, bias=False),
            torch.nn.GroupNorm(8, hidden[1]),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(hidden[1], out_ch, 1),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def harvest_scene(enc, it, dev):
    """desc [V,C,h,w]; labels {evi,bnd,valid} [V,h,w] - PIXEL-CLS geometry."""
    rgb = it['drone_images'].to(dev)
    mean = rgb.new_tensor(_UAV_MEAN)[:, None, None]
    std = rgb.new_tensor(_UAV_STD)[:, None, None]
    _, fpn = enc((rgb - mean) / std)
    desc = max(tuple(fpn[-4:]), key=lambda f: f.shape[-2] * f.shape[-1]).float()
    V, C, fh, fw = desc.shape
    depth = it['devicez_rgb'].to(dev)
    gt_union = np.zeros((512, 512), dtype=bool)
    for m in it['gt_masks']:
        gt_union |= np.asarray(m).astype(bool)
    gt_t = torch.from_numpy(gt_union).to(dev)
    yy, xx = torch.meshgrid(torch.arange(fh, device=dev), torch.arange(fw, device=dev), indexing='ij')
    px = ((xx.reshape(-1).float() + .5) * 512.0 / fw).long().clamp(max=511)
    py = ((yy.reshape(-1).float() + .5) * 512.0 / fh).long().clamp(max=511)
    z = devicez_to_camera_z(depth)[:, py, px]                    # [V, fh*fw]
    u = ((xx.reshape(-1).float() + .5) * 512.0 / fw)[None].expand(V, -1)
    w = ((yy.reshape(-1).float() + .5) * 512.0 / fh)[None].expand(V, -1)

    sc = it['satellite_camera']
    sk, s_w2c, _ = _k_and_w2c(sc[0] if isinstance(sc, list) else sc)
    sk, s_w2c = sk.to(dev), s_w2c.to(dev)
    labs = torch.zeros(V, fh * fw, dtype=torch.bool, device=dev)
    valid = torch.zeros(V, fh * fw, dtype=torch.bool, device=dev)
    cams = it['uav_cameras']
    for v in range(V):
        k, w2c, c2w = _k_and_w2c(cams[v])
        k, c2w = k.to(dev), c2w.to(dev)
        zv = z[v]
        pc = torch.stack(((u[v] - k[0, 2]) * zv / k[0, 0],
                          (w[v] - k[1, 2]) * zv / k[1, 1], zv), -1)
        world = pc @ c2w[:3, :3].T + c2w[:3, 3]
        cam = world @ s_w2c[:3, :3].T + s_w2c[:3, 3]
        su = (sk[0, 0] * cam[:, 0] / cam[:, 2].clamp_min(1e-6) + sk[0, 2]).long().clamp(0, 511)
        sv = (sk[1, 1] * cam[:, 1] / cam[:, 2].clamp_min(1e-6) + sk[1, 2]).long().clamp(0, 511)
        ok = (zv > 0) & (cam[:, 2] > 0) & (world[:, 2] < 50000)
        labs[v] = gt_t[sv, su] & ok
        valid[v] = ok
    evi = labs.reshape(V, fh, fw)
    valid = valid.reshape(V, fh, fw)
    e = evi[:, None].float()
    dil = F.max_pool2d(e, 3, stride=1, padding=1)[:, 0] > 0.5
    ero = -F.max_pool2d(-e, 3, stride=1, padding=1)[:, 0] > -0.5
    bnd = dil & ~ero & valid
    return desc, evi, bnd, valid


def channel_loss(logits, y, mask, ch):
    lg = logits[:, ch][mask]
    yy = y[mask].float()
    bce = F.binary_cross_entropy_with_logits(lg, yy)
    p = torch.sigmoid(lg)
    inter = (p * yy).sum()
    dice = 1.0 - (2.0 * inter + 1.0) / (p.sum() + yy.sum() + 1.0)
    return bce + dice


@torch.inference_mode()
def val_pass(enc, dec, ds, n, dev):
    """Pooled AP (evidence) + per-scene Dice for both channels."""
    scores_all, labels_all = [], []
    d_evi, d_bnd, d_base = [], [], []
    for i in range(n):
        it = ds[i]
        desc, evi, bnd, valid = harvest_scene(enc, it, dev)
        logits = dec(desc)
        for v in range(desc.shape[0]):
            m = valid[v]
            scores_all.append(torch.sigmoid(logits[v, 0][m]).cpu())
            labels_all.append(evi[v][m].cpu())
        p = (torch.sigmoid(logits[:, 0]) > 0.5)
        pb = (torch.sigmoid(logits[:, 1]) > 0.5)
        for v in range(desc.shape[0]):
            m = valid[v]
            def dice(P, Y):
                inter = (P & Y).sum().float()
                return float((2 * inter + 1) / (P.sum() + Y.sum() + 1))
            if evi[v][m].any():
                d_evi.append(dice(p[v][m], evi[v][m]))
                allpos = torch.ones_like(evi[v][m])
                d_base.append(dice(allpos, evi[v][m]))
            if bnd[v][m].any():
                d_bnd.append(dice(pb[v][m], bnd[v][m]))
    s = torch.cat(scores_all).numpy()
    t = torch.cat(labels_all).numpy()
    order = np.argsort(-s)
    tt = t[order]
    cum = np.cumsum(tt)
    prec = cum / (np.arange(len(tt)) + 1)
    ap = float(prec[np.where(tt == 1)[0]].mean()) if (tt == 1).any() else 0.0
    return {'ap': ap, 'dice_evi': float(np.mean(d_evi)), 'dice_bnd': float(np.mean(d_bnd)),
            'dice_base': float(np.mean(d_base)),
            'per_scene_dice_evi': d_evi, 'per_scene_dice_bnd': d_bnd,
            'per_scene_dice_base': d_base}


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--n-train', type=int, default=300)
    ap_.add_argument('--n-val', type=int, default=100)
    ap_.add_argument('--epochs', type=int, default=40)
    ap_.add_argument('--lr', type=float, default=1e-3)
    ap_.add_argument('--device', default='cuda:0')
    a = ap_.parse_args()
    dev = torch.device(a.device)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    sam2 = os.environ['SAM2_CKPT']
    enc = RSSAM2VisionEncoder(checkpoint_path=sam2, model_size='base_plus',
                              freeze_backbone=True, use_gradient_checkpointing=False,
                              use_lora=False).to(dev).eval()
    for p in enc.parameters():
        p.requires_grad_(False)

    dec = EviDecoder().to(dev)
    opt = torch.optim.AdamW(dec.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    ds_tr = UERawPairDataset(SNAP, 'train', True, 'surfbev_b0_split_v2.json', image_size=(512, 512))
    ds_va = UERawPairDataset(SNAP, 'val', True, 'surfbev_b0_split_v2.json', image_size=(512, 512))
    print(f'scenes: train={min(a.n_train, len(ds_tr))} val={min(a.n_val, len(ds_va))}', flush=True)
    idx_tr = list(range(min(a.n_train, len(ds_tr))))

    best = {'dice_evi': -1.0, 'epoch': -1, 'state': None}
    patience, bad = 4, 0
    for ep in range(a.epochs):
        dec.train()
        tot = 0.0
        rng = np.random.default_rng(SEED * 1000 + ep)
        order = rng.permutation(idx_tr)
        for i in order:
            it = ds_tr[int(i)]
            with torch.no_grad():
                desc, evi, bnd, valid = harvest_scene(enc, it, dev)
            logits = dec(desc)                       # decoder grad path only
            loss = channel_loss(logits, evi, valid, 0) + channel_loss(logits, bnd, valid, 1)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        sched.step()
        if (ep + 1) % 4 == 0 or ep == a.epochs - 1:
            dec.eval()
            v = val_pass(enc, dec, ds_va, min(50, len(ds_va)), dev)
            print(f'ep{ep+1}: train_loss={tot/len(order):.4f} val(dice_evi={v["dice_evi"]:.4f} '
                  f'dice_bnd={v["dice_bnd"]:.4f} ap={v["ap"]:.4f})', flush=True)
            if v['dice_evi'] > best['dice_evi']:
                best = {'dice_evi': v['dice_evi'], 'epoch': ep + 1,
                        'state': {k: t.detach().cpu().clone() for k, t in dec.state_dict().items()}}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    print(f'early stop at ep{ep+1} (best ep{best["epoch"]})', flush=True)
                    break

    if best['state'] is not None:
        dec.load_state_dict(best['state'])
    dec.eval()
    full = val_pass(enc, dec, ds_va, min(a.n_val, len(ds_va)), dev)
    d = np.array(full['per_scene_dice_evi']) - np.array(full['per_scene_dice_base'])
    rng = np.random.default_rng(BOOT_SEED)
    draws = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    ci = [float(v) for v in np.quantile(draws, [.025, .975])]
    gate = {'ap': full['ap'], 'dice_evi': full['dice_evi'], 'dice_bnd': full['dice_bnd'],
            'dice_vs_base_mean': float(d.mean()), 'ci95': ci,
            'n_val': len(full['per_scene_dice_evi']), 'best_epoch': best['epoch']}
    ok = (gate['ap'] >= 0.90 and gate['dice_evi'] >= 0.55 and gate['dice_bnd'] >= 0.25
          and ci[0] > 0)
    gate['verdict'] = 'PASS' if ok else 'FAIL'
    print(json.dumps({k: v for k, v in gate.items()}, ensure_ascii=False), flush=True)
    torch.save({'state_dict': dec.state_dict(), 'arch': 'EviDecoder(256,128,64,2)',
                'meta': {'seed': SEED, 'n_train': a.n_train, 'epochs': a.epochs,
                         'best_epoch': best['epoch'], 'gate': gate,
                         'sam2_ckpt': sam2}}, OUTDIR / 'decoder.pt')
    (OUTDIR / 'readout.json').write_text(json.dumps(
        {'gate': gate,
         'per_scene': {k: full[k] for k in ('per_scene_dice_evi', 'per_scene_dice_bnd',
                                            'per_scene_dice_base')}}, indent=2))
    print('VERDICT:', gate['verdict'], flush=True)


if __name__ == '__main__':
    main()
