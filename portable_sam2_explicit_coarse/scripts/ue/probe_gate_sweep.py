#!/usr/bin/env python3
"""DUAL-GATE-SWEEP: does the trained function respond to the UAV pathway?

Level-2 (function) probe for "why does using the UAV branch change nothing".
On a trained dual checkpoint, scale every fusion gate by k in {0,.25,.5,1,2,4,8}
and measure per-scene AP on the first N val scenes; add a channel-permuted
gate at k=1 as a sham control. Flat curve = pathway functionally inert;
monotone fall = the network wired itself around a harmful signal;
rise anywhere = miscalibrated gate (would argue for retraining).

No training; pure inference on last-epoch checkpoints.
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

import probe_dual_ab_ladder as L
import ue.dual  # noqa: F401
from data.ue_raw_pair_dataset import UERawPairDataset

OUT = Path('/data2/wangcheng/checkpoint/surfbev_b0/probes/DUAL-GATE-SWEEP-20261016')
SWEEP = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def gates_of(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if n.startswith('gate')}


def set_gates(model, saved, factor, permute=False, seed=44):
    g = torch.Generator().manual_seed(seed)
    for n, p in model.named_parameters():
        if n in saved:
            v = saved[n] * factor
            if permute:
                idx = torch.randperm(v.numel(), generator=g)
                v = v.reshape(-1)[idx].reshape(v.shape)
            p.data.copy_(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True)
    ap.add_argument('--scenes', type=int, default=40)
    ap.add_argument('--device', default='cuda:1')
    a = ap.parse_args()
    dev = torch.device(a.device)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.arm == 'dual_norm':  # EXP-0014 registered locally; ladder ARMS stop at EXP-0013
        L.ARMS['dual_norm'] = dict(dual=True,
            ckpt=L.CKPT_DIR / 'surfacebev_v2_512dualnorm_sealed_ddp4_s44_EXP-0014/last_model_epoch100.pth',
            log=L.LOG_DIR / 'EXP-0014-ddp4.log')
    model, identity = L.load_arm(a.arm, dev, uav_off=False)
    saved = gates_of(model)
    print('gates:', {n: tuple(v.shape) for n, v in saved.items()}, flush=True)
    ds = UERawPairDataset(str(L.SNAP), 'val', True, 'surfbev_b0_split_v2.json', image_size=(512, 512))
    ds.records = ds.records[:a.scenes]
    rows = []
    for k in SWEEP:
        set_gates(model, saved, k)
        r = L.per_scene_ap(model, ds, dev, True)
        rows.append({'k': k, 'segm': float(np.mean([x['segm'] for x in r])),
                     'bbox': float(np.mean([x['bbox'] for x in r])),
                     'segm_scene_mean_deltas_null': None})
        print(f"k={k}: segm={rows[-1]['segm']:.4f} bbox={rows[-1]['bbox']:.4f}", flush=True)
    set_gates(model, saved, 1.0, permute=True)
    r = L.per_scene_ap(model, ds, dev, True)
    rows.append({'k': 'permute', 'segm': float(np.mean([x['segm'] for x in r])),
                 'bbox': float(np.mean([x['bbox'] for x in r]))})
    print(f"k=permute: segm={rows[-1]['segm']:.4f} bbox={rows[-1]['bbox']:.4f}", flush=True)
    json.dump({'arm': a.arm, 'identity': identity.get('baseline_id'), 'n_scenes': a.scenes,
               'sweep': rows}, open(OUT / f'{a.arm}.json', 'w'), indent=1)
    # restore before exit (courtesy; process-local model)
    set_gates(model, saved, 1.0)


if __name__ == '__main__':
    main()
