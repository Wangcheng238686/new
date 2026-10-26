"""TRAIN-ONOFF-20261023: train-split on/off paired eval for a warm dual ckpt.

Question: did the UAV pathway learn anything ON THE TRAIN distribution
(H-DIST: distribution overfit) or nothing anywhere (H-OPT)?
Protocol preregistered in docs/TRAIN-ONOFF-20261023_conclusion.md
(deterministic first-200 train records, on/off modes, paired bootstrap 10k,
seed 20261014 — ladder-verbatim tooling).
"""
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
OUTDIR = Path('/data2/wangcheng/checkpoint/surfbev_b0/probes/TRAIN-ONOFF-20261023')


def boot_pairs(x_on, x_off):
    d = x_on - x_off
    r = np.random.default_rng(L.SEED)
    draws = d[r.integers(0, len(d), (10000, len(d)))].mean(1)
    return {'on_mean': float(x_on.mean()), 'off_mean': float(x_off.mean()),
            'delta_mean': float(d.mean()),
            'ci95': [float(v) for v in np.quantile(draws, [.025, .975])]}


def run_mode(a, model, tag, off):
    rows_path = OUTDIR / f'rows_{tag}_{"off" if off else "on"}.jsonl'
    if rows_path.exists():
        print(f'{rows_path} exists; skip', flush=True)
        return [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]
    if hasattr(model, 'uav_eval_off'):
        model.uav_eval_off = bool(off)
    os.environ['ARM_TAG'] = tag
    os.environ['ARM_OFF'] = '1' if off else ''
    ds = L.UERawPairDataset(str(SNAP), 'train', True, 'surfbev_b0_split_v2.json',
                            image_size=(512, 512))
    ds.records = ds.records[:a.n_train]
    rows = L.per_scene_ap(model, ds, torch.device('cuda:0'), dual=True)
    tmp = rows_path.with_suffix('.tmp')
    tmp.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    os.replace(tmp, rows_path)
    print(f'wrote {rows_path} ({len(rows)} scenes) '
          f'segm={np.mean([r["segm"] for r in rows]):.4f} '
          f'bbox={np.mean([r["bbox"] for r in rows]):.4f}', flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--n-train', type=int, default=200)
    a = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    manifest = OUTDIR / f'identity_{a.tag}.json'
    if not manifest.exists():
        ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
        manifest.write_text(json.dumps(L.checkpoint_identity(a.ckpt, ck), indent=2))
    model, _, _, _ = build_baseline(a.ckpt, manifest, os.environ['SAM2_CKPT'],
                                    torch.device('cuda:0'))
    on = run_mode(a, model, a.tag, off=False)
    off = run_mode(a, model, a.tag, off=True)
    on_by = {r['scene_id']: r for r in on}
    off_by = {r['scene_id']: r for r in off}
    common = [s for s in on_by if s in off_by]
    summary = {'tag': a.tag, 'n_scenes': len(common), 'split': 'train[:%d]' % a.n_train}
    for m in ('segm', 'bbox', 'segm_s', 'segm_m', 'segm_l', 'bbox_s', 'bbox_m', 'bbox_l'):
        x_on = np.array([on_by[s][m] for s in common])
        x_off = np.array([off_by[s][m] for s in common])
        summary[m] = boot_pairs(x_on, x_off)
    out = OUTDIR / f'summary_{a.tag}.json'
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
