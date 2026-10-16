#!/usr/bin/env python3
"""Parity check for WARM-SCREEN arms (DUAL-WARMSTART-SCREEN-20261017).

Governance invariant: with the satellite stack frozen and initialised from
EXP-0009, the UAV-off forward of a warm-screen checkpoint must be bit-identical
to the EXP-0009 baseline forward (same weights, eval mode, fusion skipped).
This script builds both models, runs the first N val scenes through
predict() and asserts bitwise-equal detections.

Usage: check_warm_parity.py --warm-ckpt PATH [--n 3]
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'inference'))
sys.path.insert(0, str(HERE))

import probe_dual_ab_ladder as L

CKPT_SINGLE = '/data2/wangcheng/checkpoint/surfbev_b0/surfacebev_v2_512_sealed_ddp4_s44_EXP-0009/last_model_epoch100.pth'
OUTDIR = Path('/data2/wangcheng/checkpoint/surfbev_b0/probes/WARM-SCREEN-20261017')


def summarize(o):
    pi = o[0].pred_instances
    return (pi.bboxes.detach().cpu().numpy().tobytes(),
            pi.scores.detach().cpu().numpy().tobytes(),
            np.asarray(pi.masks.masks if hasattr(pi.masks, 'masks')
                       else pi.masks.detach().cpu().numpy()).tobytes())


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--warm-ckpt', required=True)
    ap.add_argument('--n', type=int, default=3)
    a = ap.parse_args()
    dev = torch.device('cuda:0')
    OUTDIR.mkdir(parents=True, exist_ok=True)

    ms, _, _, _ = L.build_baseline(CKPT_SINGLE, L.manifest_for('single'),
                                   os.environ['SAM2_CKPT'], dev)
    mident = OUTDIR / f'identity_{a.warm_ckpt.split("/")[-2]}.json'
    if not mident.exists():
        ck = torch.load(a.warm_ckpt, map_location='cpu', weights_only=False)
        mident.write_text(json.dumps(L.checkpoint_identity(a.warm_ckpt, ck), indent=2))
    mw, _, _, _ = L.build_baseline(a.warm_ckpt, mident, os.environ['SAM2_CKPT'], dev)
    mw.uav_eval_off = True

    ds = L.UERawPairDataset(str(L.SNAP), 'val', True, 'surfbev_b0_split_v2.json',
                            image_size=(512, 512))
    ok_all = True
    for i in range(a.n):
        it = ds[i]
        d1 = L.model_input(ms, it, dev)
        s1 = summarize(ms.predict(d1['inputs'], d1['data_samples'], rescale=False))
        d2 = L.model_input(mw, it, dev)
        mw._uav_pending = L.uav_payload(it, dev)   # payload present; eval_off must skip it
        s2 = summarize(mw.predict(d2['inputs'], d2['data_samples'], rescale=False))
        ok = all(x == y for x, y in zip(s1, s2))
        ok_all &= ok
        print(f'scene {i}: parity={"BITWISE-OK" if ok else "MISMATCH"}', flush=True)
    verdict = 'PARITY-PASS' if ok_all else 'PARITY-FAIL'
    print('VERDICT:', verdict)
    (OUTDIR / f'parity_{a.warm_ckpt.split("/")[-2]}.json').write_text(
        json.dumps({'warm_ckpt': a.warm_ckpt, 'n_scenes': a.n, 'verdict': verdict}, indent=2))
    sys.exit(0 if ok_all else 1)


if __name__ == '__main__':
    main()
