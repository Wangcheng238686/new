#!/usr/bin/env python3
"""DISTILL-P0: cache the frozen distillation teachers (desc + height, true + sham90).

One pass over train(694)+val(100) of the frozen snapshot. For every scene we
rasterize the frozen SAM2 stride-4 descriptors to the 64x64 P3 grid exactly as
the dual arms do (same encoder construction, same _k_and_w2c contract, single
bin), once with true geometry and once with sham90 (world rotation about the
satellite camera centre - the lineage's negative control). The teacher is
frozen, so the cache is exact; all downstream gates (G1/G2f/G2t) read from it.

Outputs per scene: desc[256,64,64] fp16, zmean[64,64] fp32, mask[64,64] bool
(mask = covered = missing-flag 0, taken from the UNSCALED bev).
Whitening stats (covered-cell per-channel mu/sigma) are written for both
teachers from the train split with a sha256 for the prereg appendix.
"""
from __future__ import annotations
import hashlib, json, os, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mmengine.config import Config
from mmdet.registry import MODELS

from ue.dual import _k_and_w2c
from rsprompter.surfacebev import rasterize_surfacebev
from data.ue_raw_pair_dataset import UERawPairDataset

SNAP = '/data2/wangcheng/dataset_frozen/ue_raw_hero_uav_pairset_20261014_frozen'
OUT = Path('/data2/wangcheng/checkpoint/surfbev_b0/probes/DISTILL-P0-20261016/cache')
HW = (64, 64)


@torch.inference_mode()
def scene_bev(encoder, mean, std, edges, it, dev, sham):
    rgb = it['drone_images'].to(dev)
    _, fpn = encoder((rgb - mean) / std)
    desc = max(tuple(fpn[-4:]), key=lambda f: f.shape[-2] * f.shape[-1]).float()
    uk, _, uc = zip(*(_k_and_w2c(p) for p in it['uav_cameras']))
    sk, swc, _ = _k_and_w2c(it['satellite_camera'])
    uk = tuple(k.to(dev) for k in uk)
    uc = tuple(c.to(dev) for c in uc)
    bev, _ = rasterize_surfacebev(
        desc, it['devicez_rgb'].to(dev), torch.stack(uk), torch.stack(uc),
        sk.to(dev), swc.to(dev), edges.to(dev), HW, sham90=sham)
    bev = bev[0]  # [262, 64, 64]
    return bev[:256], bev[256], (bev[260] == 0)


def main():
    dev = torch.device(os.environ.get('HARVEST_DEVICE', 'cuda:0'))
    cfg = Config.fromfile(str(ROOT / 'configs/ue_whu_a3sgm_512_dual.py'))
    enc = dict(cfg.model.uav_encoder)
    name = enc.pop('type')
    encoder = MODELS.build({'type': name, **enc}).to(dev).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=dev)[:, None, None]
    edges = torch.tensor([-1e9, 1e9])

    manifest = []
    for split, n_expected in (('train', 694), ('val', 100)):
        ds = UERawPairDataset(SNAP, split, True, 'surfbev_b0_split_v2.json', image_size=(512, 512))
        assert len(ds) == n_expected, (split, len(ds))
        for sham in (False, True):
            mode = 'sham' if sham else 'true'
            d = OUT / f'{split}_{mode}'
            d.mkdir(parents=True, exist_ok=True)
            sums = torch.zeros(256)
            sqs = torch.zeros(256)
            z_sum, z_sq, z_n = 0.0, 0.0, 0
            for i in range(len(ds)):
                it = ds[i]
                desc, zmean, mask = scene_bev(encoder, mean, std, edges, it, dev, sham)
                sid = it['img_metas']['scene_id']
                np.savez_compressed(d / f'{sid}.npz',
                                    desc=desc.half().cpu().numpy(),
                                    zmean=zmean.float().cpu().numpy(),
                                    mask=mask.cpu().numpy())
                if split == 'train' and not sham:
                    m = mask.to(dev)
                    n = int(m.sum())
                    if n:
                        sums += (desc * m[None]).sum(dim=(1, 2)).cpu()
                        sqs += (desc.pow(2) * m[None]).sum(dim=(1, 2)).cpu()
                        z_sum += float((zmean * m).sum())
                        z_sq += float((zmean.pow(2) * m).sum())
                        z_n += n
                manifest.append(f'{split}/{mode}/{sid}')
            print(f'{split}/{mode}: {len(ds)} scenes', flush=True)
        if split == 'train':
            cnt = None
            # recount covered cells over the train cache (true mode) for exact denominators
            cnt = sum(int(np.load(OUT / f'train_true/{p.name}')['mask'].sum())
                      for p in sorted((OUT / 'train_true').glob('*.npz')))
            mu = (sums / cnt).tolist()
            sigma = ((sqs / cnt - (sums / cnt) ** 2).clamp_min(1e-12).sqrt()).tolist()
            zmu, zsigma = z_sum / cnt, max(z_sq / cnt - (z_sum / cnt) ** 2, 1e-12) ** 0.5
            stats = {'schema': 'distill-teacher-whitening-v1', 'n_train_scenes': n_expected,
                     'covered_cells': int(cnt), 'hw': list(HW),
                     'desc_mu': mu, 'desc_sigma': sigma,
                     'zmean_mu': zmu, 'zmean_sigma': zsigma}
            sp = OUT / 'whitening.json'
            sp.write_text(json.dumps(stats))
            print('whitening sha256', hashlib.sha256(sp.read_bytes()).hexdigest())
    (OUT / 'MANIFEST.txt').write_text('\n'.join(sorted(manifest)))
    print('cache sha256', hashlib.sha256((OUT / 'MANIFEST.txt').read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
