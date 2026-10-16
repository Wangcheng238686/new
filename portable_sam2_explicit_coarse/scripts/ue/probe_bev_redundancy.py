#!/usr/bin/env python3
"""DUAL-REDUNDANCY: how much of the (rescaled) BEV can satellite P3 rebuild?

Frozen-diagnosis probe for "why does using the UAV branch change nothing".
Level 1 (information): fit ridge + small-MLP readouts that predict each
normalized BEV channel from the single-stream (EXP-0009) P3 feature at the
same cell, on TRAIN scenes, and score R^2 on held-out VAL scenes. High R^2
means the fusion input is redundant by construction; the (1-R^2) residual is
the UAV-unique candidate signal. Group-wise R^2 separates the 256 descriptor
channels from the geometry/statistics channels.

Also reports the task-alignment of the residual (probe C): residual energy
density inside GT building footprints vs background cells on val scenes.

No training of the detection models; readouts are probe-local regressors.
"""
from __future__ import annotations
import argparse, json, os, sys
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
CKPT_DUALNORM = '/data2/wangcheng/checkpoint/surfbev_b0/surfacebev_v2_512dualnorm_sealed_ddp4_s44_EXP-0014/last_model_epoch100.pth'
OUTDIR = Path('/data2/wangcheng/checkpoint/surfbev_b0/probes/DUAL-REDUNDANCY-20261016')
GROUPS = {'descriptor_0_256': slice(0, 256), 'zmean_256': 256, 'zstd_257': 257,
          'vcount_258': 258, 'zres_259': 259, 'missing_260': 260, 'visible_261': 261}


@torch.inference_mode()
def harvest(model_s, model_d, ds, dev, n):
    P, B, M = [], [], []  # P3 cells, BEV' cells, GT mask cells
    for i in range(n):
        it = ds[i]
        data = model_input(model_s, it, dev)
        p3 = RSPrompterAnchor.extract_feat(model_s, data['inputs'])[0][1][0].float()  # [512,64,64]
        payload = {'drone_images': it['drone_images'][None].to(dev),
                   'devicez_rgb': it['devicez_rgb'][None].to(dev),
                   'uav_cameras': [it['uav_cameras']],
                   'satellite_camera': [it['satellite_camera']]}
        bev, _ = model_d._uav_bev(payload, tuple(p3.shape[-2:]))  # rescaled by bev_scale
        P.append(p3.flatten(1).T.cpu())   # [4096, 512]
        B.append(bev.flatten(1).T.cpu())  # [4096, 262]
        gt = np.zeros((512, 512), dtype=bool)
        for m in it['gt_masks']:
            gt |= np.asarray(m).astype(bool)
        # nearest-cell decimation of the 512 GT mask onto the 64x64 grid
        g = torch.from_numpy(gt.astype(np.float32))[None, None]
        g = torch.nn.functional.interpolate(g, size=p3.shape[-2:], mode='area')[0, 0]
        M.append((g > 0.5).flatten().cpu())
        if (i + 1) % 20 == 0:
            print(f'harvest {i+1}/{n}', flush=True)
    return torch.cat(P), torch.cat(B), torch.cat(M)


def ridge_fit(Xtr, Ytr, lam):
    d = Xtr.shape[1]
    A = Xtr.T @ Xtr + lam * torch.eye(d)
    return torch.linalg.solve(A, Xtr.T @ Ytr)


def r2(y, yhat):
    ss_res = ((y - yhat) ** 2).sum(0)
    ss_tot = ((y - y.mean(0, keepdim=True)) ** 2).sum(0)
    return 1 - ss_res / ss_tot.clamp_min(1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-train', type=int, default=60)
    ap.add_argument('--n-val', type=int, default=40)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    dev = torch.device(a.device)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(44)

    from ue.contract import checkpoint_identity
    for ck, name in ((CKPT_SINGLE, 'identity_single.json'), (CKPT_DUALNORM, 'identity_dualnorm.json')):
        mp = OUTDIR / name
        if not mp.exists():
            ck_loaded = torch.load(ck, map_location='cpu', weights_only=False)
            mp.write_text(json.dumps(checkpoint_identity(ck, ck_loaded), indent=2))
    ms, _, _, _ = build_baseline(CKPT_SINGLE, OUTDIR / 'identity_single.json', os.environ['SAM2_CKPT'], dev)
    md, _, _, _ = build_baseline(CKPT_DUALNORM, OUTDIR / 'identity_dualnorm.json', os.environ['SAM2_CKPT'], dev)

    ds_tr = UERawPairDataset(SNAP, 'train', True, 'surfbev_b0_split_v2.json', image_size=(512, 512))
    ds_va = UERawPairDataset(SNAP, 'val', True, 'surfbev_b0_split_v2.json', image_size=(512, 512))
    Ptr, Btr, _ = harvest(ms, md, ds_tr, dev, a.n_train)
    Pva, Bva, Mva = harvest(ms, md, ds_va, dev, a.n_val)

    def center(Ptr, Pva, Btr, Bva):
        pm, bm = Ptr.mean(0), Btr.mean(0)
        return Ptr - pm, Pva - pm, Btr - bm, Bva - bm, bm

    result = {'n_train_cells': int(Ptr.shape[0]), 'n_val_cells': int(Pva.shape[0]), 'ridge': {}, 'mlp': {}}
    Xt, Xv, Yt, Yv, bmean = center(Ptr, Pva, Btr, Bva)
    best = None
    for lam in (0.1, 1.0, 10.0, 100.0):
        W = ridge_fit(Xt, Yt, lam)
        sc = r2(Yv, Xv @ W)
        print(f'ridge lam={lam}: overall R2={sc.mean():.4f}', flush=True)
        if best is None or sc.mean() > best[1]:
            best = (lam, sc.mean(), sc)
    lam, overall, sc = best
    result['ridge'] = {'lambda': lam, 'overall_R2': float(overall),
                       'group_R2': {g: float(sc[s].mean()) if isinstance(s, slice) else float(sc[s])
                                    for g, s in GROUPS.items()},
                       'per_channel_R2_quartiles': [float(q) for q in torch.quantile(sc[:256], torch.tensor([.25, .5, .75]))]}
    resid = (Yv - Xv @ ridge_fit(Xt, Yt, lam))  # recompute W (deterministic)
    result['ridge']['resid_energy_frac'] = float((resid ** 2).sum() / (Yv ** 2).sum())

    # small MLP readout (2 hidden layers) for nonlinear capacity
    net = torch.nn.Sequential(torch.nn.Linear(512, 512), torch.nn.GELU(),
                              torch.nn.Linear(512, 512), torch.nn.GELU(),
                              torch.nn.Linear(512, 262)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xtd, Ytd = Xt.to(dev), Yt.to(dev)
    n = Xtd.shape[0]
    for ep in range(6):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, 8192):
            idx = perm[i:i + 8192]
            loss = ((net(Xtd[idx]) - Ytd[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        print(f'mlp epoch {ep}: loss={float(loss):.5f}', flush=True)
    with torch.no_grad():
        pred = net(Xv.to(dev)).cpu()
    scm = r2(Yv, pred)
    result['mlp'] = {'overall_R2': float(scm.mean()),
                     'group_R2': {g: float(scm[s].mean()) if isinstance(s, slice) else float(scm[s])
                                  for g, s in GROUPS.items()},
                     'per_channel_R2_quartiles': [float(q) for q in torch.quantile(scm[:256], torch.tensor([.25, .5, .75]))]}

    # probe C: residual energy at GT structures vs background (val, ridge residual)
    rn = resid.norm(dim=1)
    fg, bg = Mva.bool(), ~Mva.bool()
    result['residual_at'] = {'fg_mean': float(rn[fg].mean()), 'bg_mean': float(rn[bg].mean()),
                             'fg_over_bg': float(rn[fg].mean() / rn[bg].mean().clamp_min(1e-9)),
                             'fg_cell_frac': float(fg.float().mean())}
    # per-group residual contrast on the descriptor block
    rd = resid[:, :256].norm(dim=1)
    result['residual_at']['desc_fg_over_bg'] = float(rd[fg].mean() / rd[bg].mean().clamp_min(1e-9))

    json.dump(result, open(OUTDIR / 'redundancy.json', 'w'), indent=1)
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
