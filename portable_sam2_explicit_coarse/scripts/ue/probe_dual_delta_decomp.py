#!/usr/bin/env python3
"""DUAL-DELTA-DECOMP: why is the injected delta large but AP-flat?

Mechanism probe (frozen diagnosis, no training). On the EXP-0010 last
checkpoint and a handful of frozen-snapshot val scenes this measures, per
scene and aggregated:

1. BEV input magnitude by channel group (descriptor 0:256 vs the cm-scale
   world-Z statistics channels zmean/zstd/zres, vcount, flags 256:262) and
   each group's exact contribution to ||proj(bev)|| (leave-one-group-out
   reconstruction of the linear 1x1 conv);
2. the preregistered drift operator delta_L = mean|gate| / mean|proj(bev)|
   vs the effective gate sum|gate*proj(bev)| / sum|proj(bev)| (i.e. what
   fraction of the projected BEV actually survives gating);
3. the spatial structure of delta: energy fraction of the per-channel
   spatial-mean (bias-like) component and of an 8x8 coarse component, plus
   the structure-only residual;
4. cross-scene stability: pairwise cosine of flattened delta across scenes,
   raw and spatially-demeaned (structure-only) - a near-constant delta map
   across scenes means the injection acts as a scene-independent bias;
5. alignment with the satellite base feature (cos(base, delta)) and the
   delta/base norm ratio (cross-checked against training telemetry).

Read-only with respect to the sealed packages; reuses the A/B loader
harness (ue.features.build_baseline, UERawPairDataset).
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

from rsprompter.detection.anchor_detector import RSPrompterAnchor
import ue.dual  # noqa: F401 - registers RSPrompterAnchorDual / UEDualPreprocessor
from ue.features import build_baseline, model_input
from data.ue_raw_pair_dataset import UERawPairDataset

SNAP = Path('/data2/wangcheng/dataset_frozen/ue_raw_hero_uav_pairset_20261014_frozen')
CKPT_DUAL = Path('/data2/wangcheng/checkpoint/surfbev_b0/surfacebev_v2_512dual_sealed_ddp4_s44_EXP-0010/last_model_epoch100.pth')
OUT = Path('/data2/wangcheng/checkpoint/surfbev_b0/probes/DUAL-DELTA-DECOMP-20261016')

GROUPS = {'descriptor_0_256': slice(0, 256), 'zmean_256': slice(256, 257),
          'zstd_257': slice(257, 258), 'vcount_258': slice(258, 259),
          'zres_259': slice(259, 260), 'missing_260': slice(260, 261),
          'visible_261': slice(261, 262)}


def uav_payload(item, dev):
    return {'drone_images': item['drone_images'][None].to(dev),
            'devicez_rgb': item['devicez_rgb'][None].to(dev),
            'uav_cameras': [item['uav_cameras']],
            'satellite_camera': [item['satellite_camera']]}


@torch.inference_mode()
def analyse_scene(model, item, dev):
    data = model_input(model, item, dev)
    inputs = data['inputs']
    # base pyramid without the UAV tap: call the sealed parent forward directly
    out = RSPrompterAnchor.extract_feat(model, inputs)
    base = out[0][1].float()          # P3, target_level of EXP-0010
    bev, coverage = model._uav_bev(uav_payload(item, dev), tuple(base.shape[-2:]))
    bev = bev.float()
    proj = model.proj
    gate = model.gate.detach().float()
    full = proj(bev.unsqueeze(0))[0]                      # [512, h, w]
    delta = gate.view(-1, 1, 1) * full                     # [512, h, w]

    norms = lambda t: float(t.norm())
    per = {'coverage': coverage,
           'base_norm': norms(base), 'bev_norm': norms(bev),
           'proj_norm': norms(full), 'delta_norm': norms(delta),
           'delta_base_ratio': norms(delta) / max(norms(base), 1e-9),
           'mean_abs_proj': float(full.abs().mean()),
           'mean_abs_gate': float(gate.abs().mean()),
           'prereg_delta_L': float(gate.abs().mean()) / max(float(full.abs().mean()), 1e-12),
           'effective_gate': float(delta.abs().sum()) / max(float(full.abs().sum()), 1e-12),
           'cos_base_delta': float(torch.cosine_similarity(
               base.reshape(1, -1), delta.reshape(1, -1)))}

    # channel-group magnitudes inside the BEV tensor and their exact
    # contribution to the projected output (linear conv: removable per group)
    for name, sl in GROUPS.items():
        part = bev[sl]
        per[f'bev_abs_{name}'] = float(part.abs().mean())
        removed = proj((bev.clone().index_fill(0, torch.arange(sl.start, sl.stop, device=bev.device), 0.0)).unsqueeze(0))[0]
        per[f'proj_norm_drop_{name}'] = norms(full) - norms(removed)   # >0: group raises norm
        bev_mut = bev.clone(); bev_mut[sl] = 0.0
        per[f'proj_recon_err_{name}'] = norms(full - proj(bev_mut.unsqueeze(0))[0])

    # spatial structure of delta: per-channel spatial mean (bias) component,
    # 8x8 coarse component, structure residual
    bias = full.mean(dim=(1, 2), keepdim=True).expand_as(full) * gate.view(-1, 1, 1)
    resid = delta - bias
    resid_u = resid.unsqueeze(0)
    coarse = torch.nn.functional.interpolate(resid_u, size=(8, 8), mode='area')
    coarse = torch.nn.functional.interpolate(coarse, size=delta.shape[-2:], mode='bilinear', align_corners=False)[0]
    fine = resid - coarse
    dn = max(norms(delta), 1e-9)
    per['bias_energy_frac'] = norms(bias) ** 2 / (norms(delta) ** 2 + 1e-12)
    per['coarse_energy_frac'] = norms(coarse) ** 2 / (norms(delta) ** 2 + 1e-12)
    per['fine_energy_frac'] = norms(fine) ** 2 / (norms(delta) ** 2 + 1e-12)
    per['structure_cos'] = float(torch.cosine_similarity(
        resid.reshape(1, -1), (delta - bias).reshape(1, -1)))
    per['_delta_flat'] = delta.reshape(-1)
    per['_delta_struct'] = resid.reshape(-1)
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-val', type=int, default=6)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--ckpt', default=str(CKPT_DUAL))
    ap.add_argument('--out', default=str(OUT))
    a = ap.parse_args()
    ckpt_path = Path(a.ckpt)
    out_dir = Path(a.out)
    dev = torch.device(a.device)
    out_dir.mkdir(parents=True, exist_ok=True)

    from ue.contract import checkpoint_identity
    manifest = out_dir / 'identity_dual.json'
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if not manifest.exists():
        manifest.write_text(json.dumps(checkpoint_identity(ckpt_path, ckpt), indent=2))

    model, _, identity, _ = build_baseline(ckpt_path, manifest, os.environ['SAM2_CKPT'], dev)
    gate = model.gate.detach().float()
    gate_stats = {'mean': float(gate.mean()), 'mean_abs': float(gate.abs().mean()),
                  'absmax': float(gate.abs().max()), 'std': float(gate.std()),
                  'n_channels_gt_1e-2': int((gate.abs() > 1e-2).sum()),
                  'n_channels_gt_1e-3': int((gate.abs() > 1e-3).sum()),
                  'top5_abs': sorted((float(abs(v)) for v in gate), reverse=True)[:5]}

    ds = UERawPairDataset(str(SNAP), 'val', True, 'surfbev_b0_split_v2.json', image_size=(512, 512))
    ds.records = ds.records[:a.max_val]
    rows = []
    for i in range(len(ds)):
        per = analyse_scene(model, ds[i], dev)
        d_flat = per.pop('_delta_flat'); d_struct = per.pop('_delta_struct')
        per['scene_id'] = ds[i]['img_metas']['scene_id']
        rows.append({**per, '_flat': d_flat, '_struct': d_struct})
        print(f'[scene {i+1}/{len(ds)}] ratio={per["delta_base_ratio"]:.3f} '
              f'bias_frac={per["bias_energy_frac"]:.3f} eff_gate={per["effective_gate"]:.3e}', flush=True)

    flats = torch.stack([r.pop('_flat') for r in rows])
    structs = torch.stack([r.pop('_struct') for r in rows])
    cos = torch.nn.functional.cosine_similarity(flats[:, None], flats[None, :], dim=2)
    cos_s = torch.nn.functional.cosine_similarity(structs[:, None], structs[None, :], dim=2)
    iu = torch.triu_indices(len(rows), len(rows), 1)
    cross = {'pairwise_cos_delta_raw_mean': float(cos[iu[0], iu[1]].mean()),
             'pairwise_cos_delta_raw_min': float(cos[iu[0], iu[1]].min()),
             'pairwise_cos_structure_mean': float(cos_s[iu[0], iu[1]].mean()),
             'pairwise_cos_structure_min': float(cos_s[iu[0], iu[1]].min())}

    agg = {}
    for k in rows[0]:
        if k == 'scene_id':
            continue
        vals = [r[k] for r in rows]
        agg[k] = {'mean': float(np.mean(vals)), 'min': float(np.min(vals)), 'max': float(np.max(vals))}
    result = {'identity': identity.get('baseline_id'), 'checkpoint': str(ckpt_path),
              'n_scenes': len(rows), 'gate_checkpoint': gate_stats,
              'per_scene': rows, 'aggregate': agg, 'cross_scene': cross}
    json.dump(result, open(out_dir / 'decomp.json', 'w'), indent=1)
    print(json.dumps({'gate_checkpoint': gate_stats, 'aggregate': agg, 'cross_scene': cross}, indent=1))


if __name__ == '__main__':
    main()
