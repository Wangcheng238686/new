#!/usr/bin/env python3
"""DISTILL-P0 G2t: trainability differential probe (the authoritative kill gate).

From the EXP-0009 last checkpoint, with the task loss OFF and only the masked
L1 distillation term (lambda=1), fine-tune toward a cached teacher (desc block
or whitened zmean; true geometry or sham90) for a few epochs. Measure
per-scene val R2(P3 -> that teacher) at epoch 0 and after training with a
refit ridge readout (E3 protocol). The kill quantity -(dR2_true - dR2_sham)
per teacher, scene-paired bootstrap CI lower bound > 0.01 - is computed by the
verdict step from the four arms' JSONs.

Probe-grade: fp32, single GPU shared with running arms (kill-the-probe-first
policy), voided artifacts, no wrapper, consumes no EXP-ID.
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
from probe_distill_p0_static import ridge_cv_fit, whitening

SNAP = '/data2/wangcheng/dataset_frozen/ue_raw_hero_uav_pairset_20261014_frozen'
CKPT_SINGLE = '/data2/wangcheng/checkpoint/surfbev_b0/surfacebev_v2_512_sealed_ddp4_s44_EXP-0009/last_model_epoch100.pth'
BASE = Path('/data2/wangcheng/checkpoint/surfbev_b0/probes/DISTILL-P0-20261016')
ACCUM = 4
LR = 5e-4
WARMUP = 100
N_FIT_SCENES = 60
N_VAL_SCENES = 40


def load_cache(split, mode, sid):
    z = np.load(BASE / 'cache' / f'{split}_{mode}' / f'{sid}.npz')
    return z['desc'].astype(np.float32), z['zmean'].astype(np.float32), z['mask']


def teacher_of(sid, split, mode, teacher):
    wmu, wsig, zmu, zsig = whitening()
    desc, zmean, mask = load_cache(split, mode, sid)
    if teacher == 'desc':
        T = (torch.from_numpy(desc) - wmu[:, None, None]) / wsig[:, None, None]
    else:
        T = ((torch.from_numpy(zmean) - zmu) / zsig)[None]
    return T, torch.from_numpy(mask)


@torch.inference_mode()
def scene_rows(model, dev, ids, split, mode, teacher):
    ds = UERawPairDataset(SNAP, split, False, 'surfbev_b0_split_v2.json', image_size=(512, 512), load_targets=False)
    rows = []
    for idx in range(len(ds)):
        it = ds[idx]
        sid = it['img_metas']['scene_id']
        if sid not in ids:
            continue
        data = model_input(model, it, dev)
        p3 = RSPrompterAnchor.extract_feat(model, data['inputs'])[0][1][0].float()
        T, M = teacher_of(sid, split, mode, teacher)
        rows.append((sid, p3.flatten(1).T.cpu(), T.flatten(1).T, M.flatten()))
    return rows


@torch.inference_mode()
def eval_r2(model, dev, teacher, mode, fit_ids, val_ids):
    fit = scene_rows(model, dev, set(fit_ids), 'train', mode, teacher)
    X = torch.cat([r[1] for r in fit]); Y = torch.cat([r[2] for r in fit])
    M = torch.cat([r[3] for r in fit]).bool()
    W, xm, ym = ridge_cv_fit(X[M], Y[M])
    per_scene = []
    for sid, Xs, Ys, Ms in scene_rows(model, dev, set(val_ids), 'val', mode, teacher):
        sel = Ms.bool()
        Xc = (Xs - xm)[sel]; Yc = (Ys - ym)[sel]
        pred = Xc @ W
        ssr = ((Yc - pred) ** 2).sum(0)
        sst = ((Yc - Yc.mean(0, keepdim=True)) ** 2).sum(0).clamp_min(1e-9)
        per_scene.append(float((1 - ssr / sst).mean()))
    return np.array(per_scene)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher', choices=['desc', 'height'], required=True)
    ap.add_argument('--mode', choices=['true', 'sham'], required=True)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    dev = torch.device(a.device)

    ds = UERawPairDataset(SNAP, 'train', False, 'surfbev_b0_split_v2.json', image_size=(512, 512), load_targets=False)
    fit_ids = {r['scene_dir'] for r in ds.records[:N_FIT_SCENES]}
    ds_val = UERawPairDataset(SNAP, 'val', False, 'surfbev_b0_split_v2.json', image_size=(512, 512))
    val_ids = [r['scene_dir'] for r in ds_val.records[:N_VAL_SCENES]]
    val_set = set(val_ids)

    from ue.contract import checkpoint_identity
    manifest = BASE / 'identity_single.json'
    if not manifest.exists():
        ck = torch.load(CKPT_SINGLE, map_location='cpu', weights_only=False)
        manifest.write_text(json.dumps(checkpoint_identity(CKPT_SINGLE, ck), indent=2))
    model, _, identity, _ = build_baseline(CKPT_SINGLE, manifest, os.environ['SAM2_CKPT'], dev)

    torch.manual_seed(44)  # identical g init across modes of the same teacher
    g = torch.nn.Conv2d(512, 256 if a.teacher == 'desc' else 1, 1).to(dev)
    torch.nn.init.kaiming_normal_(g.weight)
    opt = torch.optim.AdamW([
        {'params': [p for p in model.parameters() if p.requires_grad], 'lr': LR, 'weight_decay': 0.05},
        {'params': list(g.parameters()), 'lr': LR, 'weight_decay': 0.0}])

    model.eval()
    r2_ep0 = eval_r2(model, dev, a.teacher, a.mode, fit_ids, val_ids)
    print(f'[{a.teacher}/{a.mode}] ep0 mean R2 = {r2_ep0.mean():.4f}', flush=True)

    model.train()
    step, loss_log = 0, []
    for ep in range(a.epochs):
        tot, seen, last_log = 0.0, 0, 0.0
        opt.zero_grad(set_to_none=True)
        for idx in range(len(ds)):
            it = ds[idx]
            sid = it['img_metas']['scene_id']
            data = model_input(model, it, dev)
            p3 = RSPrompterAnchor.extract_feat(model, data['inputs'])[0][1][0]
            T, M = teacher_of(sid, 'train', a.mode, a.teacher)
            pred = g(p3.unsqueeze(0))[0]
            diff = (pred - T.to(dev)).abs().mean(0)
            loss = (diff * M.to(dev)).sum() / M.sum().clamp_min(1)
            (loss / ACCUM).backward()
            tot += float(loss); seen += 1
            if seen % ACCUM == 0:
                step += 1
                lr = LR * min(1.0, step / WARMUP)
                for grp in opt.param_groups:
                    grp['lr'] = lr
                if torch.isfinite(torch.stack([p.grad.abs().max() if p.grad is not None else torch.zeros((), device=dev) for p in g.parameters()]).all()):
                    opt.step()
                opt.zero_grad(set_to_none=True)
            if seen % 400 == 0:
                print(f'[{a.teacher}/{a.mode}] ep{ep} scene{seen}/{len(ds)} avg_loss={tot/seen:.4f}', flush=True)
        loss_log.append({'epoch': ep, 'mean_loss': tot / max(seen, 1)})

    model.eval()
    r2_epN = eval_r2(model, dev, a.teacher, a.mode, fit_ids, val_ids)
    print(f'[{a.teacher}/{a.mode}] epN mean R2 = {r2_epN.mean():.4f}', flush=True)
    out = {'teacher': a.teacher, 'mode': a.mode, 'identity': identity.get('baseline_id'),
           'epochs': a.epochs, 'r2_ep0': r2_ep0.tolist(), 'r2_epN': r2_epN.tolist(),
           'val_ids': val_ids, 'loss_log': loss_log}
    (BASE / f'g2t_{a.teacher}_{a.mode}.json').write_text(json.dumps(out))


if __name__ == '__main__':
    main()
