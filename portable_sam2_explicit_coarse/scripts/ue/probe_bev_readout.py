#!/usr/bin/env python3
"""BEV-READOUT-P0 (DUAL-WARMSTART-SCREEN-20261017): task-signal certification
of the frozen calibrated BEV cargo.

The complementary direction to DUAL-REDUNDANCY (which asked "can satellite P3
rebuild the BEV?" -> no, 96.5% unique energy): here we ask "can a supervised
readout extract building-footprint signal FROM the BEV?" Per channel group we
train a small MLP on TRAIN scenes (64x64 cells) to predict the GT union
footprint cell and score per-scene IoU/AP on held-out VAL scenes, with two
sham controls (permuted descriptor channels; scene-shifted pairing).

Verdict gates are preregistered in docs/DUAL-WARMSTART-SCREEN-20261017_preregistration.md:
R3-PASS / R3-CEIL / R3-NULL. No detection model is trained; readout quality is
necessary-not-sufficient evidence for fusion gains.
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
from data.ue_raw_pair_dataset import UERawPairDataset

SNAP = '/data2/wangcheng/dataset_frozen/ue_raw_hero_uav_pairset_20261014_frozen'
CKPT_DUALNORM = '/data2/wangcheng/checkpoint/surfbev_b0/surfacebev_v2_512dualnorm_sealed_ddp4_s44_EXP-0014/last_model_epoch100.pth'
OUTDIR = Path('/data2/wangcheng/checkpoint/surfbev_b0/probes/BEV-READOUT-20261017')
HW = (64, 64)          # P3-aligned grid, same cell contract as DUAL-REDUNDANCY
SEED = 44
BOOT_SEED = 20261014
GROUPS = {
    'desc_0_256': slice(0, 256),
    'zmean_256': slice(256, 257),
    'stats_256_262': slice(256, 262),
    'all_262': slice(0, 262),
}


@torch.inference_mode()
def harvest(model_d, ds, dev, n):
    B, M, S = [], [], []   # BEV cells per scene, GT cells per scene, scene ids
    for i in range(n):
        it = ds[i]
        payload = {'drone_images': it['drone_images'][None].to(dev),
                   'devicez_rgb': it['devicez_rgb'][None].to(dev),
                   'uav_cameras': [it['uav_cameras']],
                   'satellite_camera': [it['satellite_camera']]}
        bev, _ = model_d._uav_bev(payload, HW)   # calibrated by bev_scale (0014 arm)
        B.append(bev.flatten(1).T.cpu())         # [4096, 262]
        print(f'  scene {i+1}/{n} done', flush=True)
        gt = np.zeros((512, 512), dtype=bool)
        for m in it['gt_masks']:
            gt |= np.asarray(m).astype(bool)
        g = torch.from_numpy(gt.astype(np.float32))[None, None]
        g = F.interpolate(g, size=HW, mode='area')[0, 0]
        M.append((g > 0.5).flatten().cpu())
        S.append(it['img_metas']['scene_id'])
        if (i + 1) % 20 == 0:
            print(f'harvest {i+1}/{n}', flush=True)
    return B, M, S


def train_mlp(Xtr, ytr, dim, steps=4000, lr=1e-3, batch=16384, seed=SEED, dev=None):
    torch.manual_seed(seed)
    if dev is not None:
        Xtr, ytr = Xtr.to(dev), ytr.to(dev)
    net = torch.nn.Sequential(
        torch.nn.Linear(dim, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 1),
    ).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    n = Xtr.shape[0]
    for step in range(steps):
        idx = torch.randint(0, n, (min(batch, n),), device=Xtr.device)
        loss = F.binary_cross_entropy_with_logits(net(Xtr[idx]), ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return net


def per_scene_iou(logits, targets):
    """logits/targets: [S, cells] -> per-scene IoU at 0.5"""
    pred = (logits > 0).float()
    inter = (pred * targets).sum(1)
    union = ((pred + targets) > 0).float().sum(1)
    return (inter / union.clamp_min(1)).numpy()


def pooled_ap(scores, targets):
    """Average precision over all cells, positives = building cells."""
    scores_np = scores.numpy() if isinstance(scores, torch.Tensor) else np.asarray(scores)
    t_np = targets.numpy() if isinstance(targets, torch.Tensor) else np.asarray(targets)
    order = np.argsort(-scores_np)
    t = t_np[order]
    cum_tp = np.cumsum(t)
    precision = cum_tp / (np.arange(len(t)) + 1)
    idx = np.where(t == 1)[0]
    return float(precision[idx].mean()) if len(idx) else 0.0


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--n-train', type=int, default=60)
    ap_.add_argument('--n-val', type=int, default=40)
    ap_.add_argument('--device', default='cuda:0')
    a = ap_.parse_args()
    dev = torch.device(a.device)
    global DEV
    DEV = dev
    torch.set_num_threads(8)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)

    from ue.contract import checkpoint_identity
    mp = OUTDIR / 'identity_dualnorm.json'
    if not mp.exists():
        ck = torch.load(CKPT_DUALNORM, map_location='cpu', weights_only=False)
        mp.write_text(json.dumps(checkpoint_identity(CKPT_DUALNORM, ck), indent=2))
    model_d, _, _, _ = build_baseline(CKPT_DUALNORM, mp, os.environ['SAM2_CKPT'], dev)

    ds_tr = UERawPairDataset(SNAP, 'train', True, 'surfbev_b0_split_v2.json', image_size=(512, 512))
    ds_va = UERawPairDataset(SNAP, 'val', True, 'surfbev_b0_split_v2.json', image_size=(512, 512))
    Btr, Mtr, Str = harvest(model_d, ds_tr, dev, a.n_train)
    Bva, Mva, Sva = harvest(model_d, ds_va, dev, a.n_val)

    # z-score standardisation from train statistics (applied per group)
    def group_tensor(B, sl):
        X = torch.cat([b[:, sl] for b in B])          # [S*cells, g]
        return X

    results = {'n_train': a.n_train, 'n_val': a.n_val, 'groups': {}, 'meta': {
        'bev_source': 'EXP-0014 calibrated (bev_scale)', 'grid': list(HW),
        'seed': SEED, 'readout': 'MLP g->256->256->1, BCE, AdamW 1e-3, 4000 steps'}}

    val_targets = torch.cat(Mva)                       # [S*cells]
    val_target_scenes = torch.stack(Mva)               # [S, cells]
    coverage = val_target_scenes.float().mean(1)
    const_iou = coverage.numpy()                       # all-ones prediction IoU == coverage

    # fixed channel permutation for the descriptor sham (same for every scene)
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(256, generator=g)

    def run_group(name, train_feats_fn, val_feats_fn, dim):
        Xtr = train_feats_fn(); Xva = val_feats_fn()
        mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
        net = train_mlp((Xtr - mu) / sd, torch.cat(Mtr)[:, None].float(), dim, dev=dev)
        with torch.no_grad():
            logits_va = net(((Xva - mu) / sd).to(dev))[:, 0].cpu()
        logits_scenes = logits_va.view(len(Bva), -1)
        iou = per_scene_iou(logits_scenes, val_target_scenes)
        apv = pooled_ap(torch.sigmoid(logits_va), val_targets)
        return {'iou_mean': float(iou.mean()),
                'iou_per_scene': [float(v) for v in iou],
                'ap': apv}

    for gname, sl in GROUPS.items():
        print(f'training readout {gname} ...', flush=True)
        results['groups'][gname] = run_group(
            gname,
            lambda sl=sl: group_tensor(Btr, sl),
            lambda sl=sl: group_tensor(Bva, sl),
            sl.stop - sl.start)
        print(gname, results['groups'][gname]['iou_mean'],
              'AP', results['groups'][gname]['ap'], flush=True)

    # sham 1: permuted descriptor channels
    def perm_tr():
        return torch.cat([b[:, :256][:, perm] for b in Btr])
    def perm_va():
        return torch.cat([b[:, :256][:, perm] for b in Bva])
    results['groups']['sham_perm_desc'] = run_group('sham_perm_desc', perm_tr, perm_va, 256)

    # sham 2: scene-shifted pairing (logits from BEV of scene i scored
    # against the GT of scene i+1 - the pairing is wrong, must collapse)
    Call = torch.cat(Btr).to(DEV); Cva = torch.cat(Bva).to(DEV)
    mu, sd = Call.mean(0), Call.std(0).clamp_min(1e-6)
    net = train_mlp((Call - mu) / sd, torch.cat(Mtr)[:, None].float(), 262, dev=DEV)
    with torch.no_grad():
        lg = net((Cva - mu) / sd)[:, 0].cpu().view(len(Bva), -1)
    targets_shift = torch.stack(Mva[1:] + Mva[:1])
    iou_shift = per_scene_iou(lg, targets_shift)
    results['groups']['sham_scene_shift'] = {
        'iou_mean': float(iou_shift.mean()),
        'iou_per_scene': [float(v) for v in iou_shift], 'ap': None}
    results['const_prior'] = {'iou_mean': float(const_iou.mean()),
                              'iou_per_scene': [float(v) for v in const_iou]}

    # preregistered contrast: desc vs zmean, per-scene paired bootstrap
    d = (np.array(results['groups']['desc_0_256']['iou_per_scene'])
         - np.array(results['groups']['zmean_256']['iou_per_scene']))
    rng = np.random.default_rng(BOOT_SEED)
    draws = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    results['contrast_desc_minus_zmean'] = {
        'mean': float(d.mean()),
        'ci95': [float(v) for v in np.quantile(draws, [.025, .975])]}
    zc = (np.array(results['groups']['zmean_256']['iou_per_scene']) - const_iou)
    draws_z = zc[rng.integers(0, len(zc), (10000, len(zc)))].mean(1)
    results['contrast_zmean_minus_const'] = {
        'mean': float(zc.mean()),
        'ci95': [float(v) for v in np.quantile(draws_z, [.025, .975])]}

    # verdict per preregistered gates
    c1 = results['contrast_desc_minus_zmean']
    c2 = results['contrast_zmean_minus_const']
    sham_ok = (results['groups']['sham_scene_shift']['iou_mean']
               < results['groups']['all_262']['iou_mean'] - 0.10)
    if not sham_ok:
        verdict = 'SHAM_FAILED (probe invalid)'
    elif c1['mean'] >= 0.05 and c1['ci95'][0] > 0:
        verdict = 'R3-PASS'
    elif c2['mean'] < 0.03:
        verdict = 'R3-CEIL'
    else:
        verdict = 'R3-NULL'
    results['verdict'] = verdict
    print(json.dumps({k: v for k, v in results.items() if k != 'groups'}, ensure_ascii=False))
    print('VERDICT:', verdict, flush=True)
    (OUTDIR / 'readout.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
