#!/usr/bin/env python3
"""EXP-0014: harvest per-channel RMS statistics of the single-bin BEV.

Deterministic offline pass over the FULL train split (694 scenes) of the
frozen snapshot. Mirrors the frozen branch of RSPrompterAnchorDual._uav_bev
line for line (same encoder construction, same _k_and_w2c camera contract,
same rasterize_surfacebev single-bin call at the P3 grid) so the constants
this produces are exactly the statistics of the tensor the fusion arm sees.

Output JSON: /data2/.../bev_stats/exp0014_bev_scale_p3.json with
schema/rms/scale/n_scenes/hw/floor; the trainer-side model multiplies the
BEV by `scale` before the 1x1 projection (frozen buffer, never trained).
No val/test scene is opened.
"""
from __future__ import annotations
import hashlib, json, os, sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mmengine.config import Config
from mmdet.registry import MODELS

from ue.dual import _k_and_w2c  # camera contract shared with the training path
from rsprompter.surfacebev import rasterize_surfacebev
from data.ue_raw_pair_dataset import UERawPairDataset

SNAP = '/data2/wangcheng/dataset_frozen/ue_raw_hero_uav_pairset_20261014_frozen'
OUT = Path('/data2/wangcheng/checkpoint/surfbev_b0/bev_stats/exp0014_bev_scale_p3.json')
FLOOR = 0.05
HW = (64, 64)  # P3 grid of the 512-native protocol


@torch.inference_mode()
def main():
    dev = torch.device(os.environ.get('HARVEST_DEVICE', 'cuda:0'))
    cfg = Config.fromfile(str(ROOT / 'configs/ue_whu_a3sgm_512_dual.py'))
    enc = dict(cfg.model.uav_encoder)
    name = enc.pop('type')
    encoder = MODELS.build({'type': name, **enc}).to(dev).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=dev)[:, None, None]
    edges = torch.tensor([-1e9, 1e9])

    ds = UERawPairDataset(SNAP, 'train', True, 'surfbev_b0_split_v2.json', image_size=(512, 512))
    sq = torch.zeros(0)
    n_cells = 0
    for i in range(len(ds)):
        it = ds[i]
        rgb = it['drone_images'].to(dev)
        _, fpn = encoder((rgb - mean) / std)
        desc = max(tuple(fpn[-4:]), key=lambda f: f.shape[-2] * f.shape[-1]).float()
        cams = it['uav_cameras']
        uk, _, uc = zip(*(_k_and_w2c(p) for p in cams))
        sk, swc, _ = _k_and_w2c(it['satellite_camera'])
        uk = tuple(k.to(dev) for k in uk); uc = tuple(c.to(dev) for c in uc)
        bev, stats = rasterize_surfacebev(
            desc, it['devicez_rgb'].to(dev), torch.stack(uk), torch.stack(uc),
            sk.to(dev), swc.to(dev), edges.to(dev), HW)
        bev = bev[0]  # single bin -> [262, 64, 64]
        if sq.numel() == 0:
            sq = torch.zeros(bev.shape[0], device=dev)
        sq += bev.pow(2).sum(dim=(1, 2))
        n_cells += bev.shape[1] * bev.shape[2]
        if (i + 1) % 50 == 0:
            print(f'{i+1}/{len(ds)} scenes', flush=True)
    rms = (sq / n_cells).sqrt().cpu()
    scale = 1.0 / rms.clamp_min(FLOOR)
    payload = {
        'schema': 'exp0014-bev-scale-v1', 'n_scenes': len(ds),
        'hw': list(HW), 'floor': FLOOR, 'n_cells_per_scene': HW[0] * HW[1],
        'rms': [float(v) for v in rms], 'scale': [float(v) for v in scale],
        'source': {'split': 'train', 'manifest': 'surfbev_b0_split_v2.json',
                   'data_root': SNAP,
                   'encoder_config': 'configs/ue_whu_a3sgm_512_dual.py#uav_encoder',
                   'height_edges': '[-1e9, 1e9] (single bin, EXP-0010 path)'}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1))
    print('sha256', hashlib.sha256(OUT.read_bytes()).hexdigest())
    print('rms describe: desc[0:256] median=%.4g max=%.4g | zmean=%.4g zstd=%.4g vcount=%.4g zres=%.4g'
          % (rms[:256].median(), rms[:256].max(), rms[256], rms[257], rms[258], rms[259]))
    print('scale: desc median=%.4g | zmean=%.4g zres=%.4g' % (scale[:256].median(), scale[256], scale[259]))


if __name__ == '__main__':
    main()
