#!/usr/bin/env python3
"""Per-scene on/off evaluation for a WARM-SCREEN checkpoint (20261017).

Reuses the DUAL-AB-LADDER per-scene COCO AP protocol verbatim (same val 100,
same seed, same model rebuild path) for an arbitrary checkpoint path, and
contrasts against the existing EXP-0009 single rows
(probes/DUAL-AB-LADDER-20261016/rows_single.jsonl) with scene-paired
percentile bootstrap (10k draws, seed 20261014) - preregistered contrasts of
docs/DUAL-WARMSTART-SCREEN-20261017_preregistration.md.

Usage: eval_warm_screen.py --ckpt PATH --tag warm_r1 [--off] [--max-val N]
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
from ue.features import build_baseline  # same-file protocol reuse

SNAP = L.SNAP
OUTDIR = Path('/data2/wangcheng/checkpoint/surfbev_b0/probes/WARM-SCREEN-20261017')
SINGLE_ROWS = Path('/data2/wangcheng/checkpoint/surfbev_b0/probes/DUAL-AB-LADDER-20261016/rows_single.jsonl')


def boot_pairs(x_on, x_single):
    d = x_on - x_single
    r = np.random.default_rng(L.SEED)
    draws = d[r.integers(0, len(d), (10000, len(d)))].mean(1)
    return {'on_mean': float(x_on.mean()), 'single_mean': float(x_single.mean()),
            'delta_mean': float(d.mean()),
            'ci95': [float(v) for v in np.quantile(draws, [.025, .975])]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--off', action='store_true', help='evaluate UAV-off pass')
    ap.add_argument('--max-val', type=int, default=100)
    a = ap.parse_args()
    dev = torch.device('cuda:0')
    OUTDIR.mkdir(parents=True, exist_ok=True)
    suffix = '_off' if a.off else '_on'
    rows_path = OUTDIR / f'rows_{a.tag}{suffix}.jsonl'
    if rows_path.exists():
        print(f'{rows_path} exists; skip', flush=True)
        return

    manifest = OUTDIR / f'identity_{a.tag}.json'
    if not manifest.exists():
        ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
        manifest.write_text(json.dumps(L.checkpoint_identity(a.ckpt, ck), indent=2))
    model, _, _, _ = build_baseline(a.ckpt, manifest, os.environ['SAM2_CKPT'], dev)
    if hasattr(model, 'uav_eval_off'):
        model.uav_eval_off = bool(a.off)

    ds = L.UERawPairDataset(str(SNAP), 'val', True, 'surfbev_b0_split_v2.json',
                            image_size=(512, 512))
    if a.max_val < len(ds):
        ds.records = ds.records[:a.max_val]
    os.environ['ARM_TAG'] = a.tag
    os.environ['ARM_OFF'] = '1' if a.off else ''
    rows = L.per_scene_ap(model, ds, dev, dual=True)
    tmp = rows_path.with_suffix('.tmp')
    tmp.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    os.replace(tmp, rows_path)
    print(f'wrote {rows_path} ({len(rows)} scenes) '
          f'segm={np.mean([r["segm"] for r in rows]):.4f} '
          f'bbox={np.mean([r["bbox"] for r in rows]):.4f}', flush=True)

    # contrasts vs single (only on the ON pass, both passes evaluated)
    single = [json.loads(l) for l in SINGLE_ROWS.read_text().splitlines() if l.strip()]
    single_by_id = {r['scene_id']: r for r in single}
    this_by_id = {r['scene_id']: r for r in rows}
    common = [sid for sid in this_by_id if sid in single_by_id]
    if len(common) >= 10 and not a.off:
        s_seg = np.array([single_by_id[s]['segm'] for s in common])
        s_bbox = np.array([single_by_id[s]['bbox'] for s in common])
        o_seg = np.array([this_by_id[s]['segm'] for s in common])
        o_bbox = np.array([this_by_id[s]['bbox'] for s in common])
        summary = {'tag': a.tag, 'n_scenes': len(common),
                   'segm': boot_pairs(o_seg, s_seg), 'bbox': boot_pairs(o_bbox, s_bbox)}
        off_rows = OUTDIR / f'rows_{a.tag}_off.jsonl'
        if off_rows.exists():
            off_by_id = {json.loads(l)['scene_id']: json.loads(l)
                         for l in off_rows.read_text().splitlines() if l.strip()}
            oc = [s for s in common if s in off_by_id]
            if oc:
                summary['on_minus_off'] = {
                    'segm': boot_pairs(np.array([this_by_id[s]['segm'] for s in oc]),
                                       np.array([off_by_id[s]['segm'] for s in oc])),
                    'bbox': boot_pairs(np.array([this_by_id[s]['bbox'] for s in oc]),
                                       np.array([off_by_id[s]['bbox'] for s in oc]))}
        spath = OUTDIR / f'summary_{a.tag}.json'
        spath.write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
