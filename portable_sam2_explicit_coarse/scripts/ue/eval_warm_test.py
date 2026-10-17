#!/usr/bin/env python3
"""Test-split opening for the WARM full runs (EXP-0018/0019 prereg clause):
standard per-scene COCO inference on the sealed test 151, for the single
anchor (EXP-0009) and each seed's last checkpoint. Rows land in
probes/WARM-SCREEN-20261017/rows_*_test.jsonl; a summarizer pairs them once
all three exist. Results are for the final report only (no gate decisions).

Usage: eval_warm_test.py --ckpt PATH --tag t [--single]
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
from ue.features import build_baseline

SNAP = L.SNAP
OUTDIR = Path('/data2/wangcheng/checkpoint/surfbev_b0/probes/WARM-SCREEN-20261017')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--single', action='store_true')
    a = ap.parse_args()
    dev = torch.device('cuda:0')
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows_path = OUTDIR / f'rows_{a.tag}_test.jsonl'
    if rows_path.exists():
        print(f'{rows_path} exists; skip', flush=True)
        return
    manifest = OUTDIR / f'identity_{a.tag}_test.json'
    if not manifest.exists():
        ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
        manifest.write_text(json.dumps(L.checkpoint_identity(a.ckpt, ck), indent=2))
    model, _, _, _ = build_baseline(a.ckpt, manifest, os.environ['SAM2_CKPT'], dev)
    ds = L.UERawPairDataset(str(SNAP), 'test', True, 'surfbev_b0_split_v2.json',
                            image_size=(512, 512))
    os.environ['ARM_TAG'] = a.tag + '_test'
    os.environ['ARM_OFF'] = ''
    rows = L.per_scene_ap(model, ds, dev, dual=not a.single)
    tmp = rows_path.with_suffix('.tmp')
    tmp.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    os.replace(tmp, rows_path)
    print(f'wrote {rows_path} ({len(rows)} scenes) '
          f'segm={np.mean([r["segm"] for r in rows]):.4f} '
          f'bbox={np.mean([r["bbox"] for r in rows]):.4f}', flush=True)

    # paired summary once single + both seeds exist
    single_p = OUTDIR / 'rows_single_test.jsonl'
    seeds = ('warm_r1g_full_s44', 'warm_r1g_full_s45')
    if all((OUTDIR / f'rows_{t}_test.jsonl').exists() for t in seeds) and single_p.exists():
        single = {r['scene_id']: r for r in map(json.loads, single_p.read_text().splitlines())}
        rng = np.random.default_rng(L.SEED)
        summary = {}
        for t in seeds:
            rows_d = {r['scene_id']: r for r in map(json.loads, (OUTDIR / f'rows_{t}_test.jsonl').read_text().splitlines())}
            common = [s for s in rows_d if s in single]
            for k in ('segm', 'bbox'):
                d = np.array([rows_d[s][k] - single[s][k] for s in common])
                draws = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
                summary[f'{t}_{k}'] = {'mean': float(d.mean()),
                                       'ci95': [float(v) for v in np.quantile(draws, [.025, .975])]}
        sp = OUTDIR / 'summary_test.json'
        sp.write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
