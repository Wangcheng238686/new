#!/usr/bin/env python3
"""DUAL-P4-CHANS anchor harvest, executable inside the 807d220 worktree.

The EXP-0006 anchor (pre-sealing, c5v2-lineage architecture) can only be
loaded by the 807d220-era inference stack that produced the P1-V2 R1a cache.
This script mirrors 807d220's probe_surfacebev_p1.py build/pred_p3/rasterize
path verbatim; only the packing changes: pure channels (sat128 single
projection; 120-dim correct/sham BEV descriptors), seed 20261014, no fused
second projection, train+val only, test never read.
The readout runs in the current worktree (probe_dual_p4_channels.py --phase
readout) on the cache produced here.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
import cv2, numpy as np, torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'inference'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.ue_raw_pair_dataset import UERawPairDataset
from inference.infer_from_checkpoint import (_load_checkpoint, _load_model_state,
                                             _register_and_build, _resolve_model_config,
                                             _resolve_sam2_repo, _select_state_dict, _snapshot)
from rsprompter.surfacebev import rasterize_surfacebev
from probe_surfacebev_p1_h0 import (EXPECTED_B0_PATH, EXPECTED_B0_SHA256, EXPECTED_RUN_TAG,
                                    cameras, make_uav_encoder, uav_descriptors,
                                    sha_file, model_input, digest_predictions)

SEED = 20261014
H0 = Path('/data1/wangcheng/checkpoint/surfbev_b0/probes/SURFBEV-P1-V2-H0-full-20261012/summary.json')
VERSION = 'DUAL-P4-CHANS-anchor-harvest-v1'


def args():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--checkpoint', type=Path, default=EXPECTED_B0_PATH)
    p.add_argument('--root', type=Path, default=ROOT / 'data/ue_raw_hero_uav_pairset_20261011')
    p.add_argument('--sam2-ckpt', default='/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt')
    p.add_argument('--max-train', type=int, default=0)
    p.add_argument('--max-val', type=int, default=0)
    return p.parse_args()


def boundary(m):
    m = m.astype(np.uint8)
    return cv2.dilate(m, np.ones((5, 5), np.uint8)) - cv2.erode(m, np.ones((5, 5), np.uint8)) > 0


def p3mask(m):
    return F.max_pool2d(torch.from_numpy(m.astype(np.float32))[None, None], 8, 8)[0, 0].numpy().astype(np.uint8).reshape(-1)


def atomic_json(p, d):
    q = p.with_suffix(p.suffix + '.tmp')
    q.write_text(json.dumps(d, indent=2))
    os.replace(q, p)


def atomic_npz(p, **d):
    q = p.with_suffix(p.suffix + '.tmp')
    with open(q, 'wb') as f:
        np.savez_compressed(f, **d)
    os.replace(q, p)


@torch.inference_mode()
def pred_p3(model, item, dev):
    a = model_input(model, item, dev)
    o = model.predict(a['inputs'], a['data_samples'], rescale=False)
    digest = digest_predictions(o)
    b = model_input(model, item, dev)
    p3 = model.extract_feat(b['inputs'], b['data_samples'])[0][1][0].float()
    c = model_input(model, item, dev)
    z = model.predict(c['inputs'], c['data_samples'], rescale=False)
    if digest != digest_predictions(z) or p3.shape != (512, 128, 128):
        raise RuntimeError('B0 passive parity/P3 contract')
    x = getattr(o[0].pred_instances, 'masks')
    x = x.to_ndarray() if hasattr(x, 'to_ndarray') else (x.masks if hasattr(x, 'masks') else x)
    return p3.detach().cpu().numpy(), (x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)).astype(bool), digest


def build(a, dev):
    ck = _load_checkpoint(a.checkpoint)
    snap = _snapshot(ck)
    ta = snap['training_args']
    if ck['epoch'] != 99 or ta.get('run_tag') != EXPECTED_RUN_TAG or ta.get('seed') != 44:
        raise RuntimeError('B0 lineage')
    if sha_file(a.checkpoint) != EXPECTED_B0_SHA256:
        raise RuntimeError('anchor sha mismatch')
    la = argparse.Namespace(config=None, sam2_ckpt=a.sam2_ckpt, sam2_repo=None)
    _resolve_sam2_repo(la, snap)
    cfg, source = _resolve_model_config(la, snap)
    m = _register_and_build(cfg).to(dev).eval()
    r = _load_model_state(m, _select_state_dict(ck, 'model'), False)
    if r['missing_keys'] or r['unexpected_keys']:
        raise RuntimeError('strict B0 load')
    return m, make_uav_encoder(la, dev), cfg, source


def projections(dev):
    r = np.random.default_rng(SEED)
    return {'sat': (r.normal(size=(512, 128)) / np.sqrt(128)).astype('float32'),
            'desc': torch.tensor((r.normal(size=(256, 4)) / 16).astype('float32'), device=dev)}


def channels(p3, surf, p):
    sat = torch.from_numpy(p3).float().permute(1, 2, 0).reshape(-1, 512).numpy() @ p['sat']
    q = torch.einsum('bchw,cd->bdhw', surf[:, :256], p['desc']).permute(0, 2, 3, 1)
    q = torch.cat((q, surf[:, 256:].permute(0, 2, 3, 1)), 3).permute(1, 2, 0, 3).reshape(-1, 120)
    return sat.astype('float32'), q.cpu().numpy().astype('float32')


def iiou(pred, gt):
    if not len(pred):
        return np.zeros(0, np.float32)
    if not len(gt):
        return np.zeros(len(pred), np.float32)
    a = pred.reshape(len(pred), -1).astype(np.int32)
    b = gt.reshape(len(gt), -1).astype(np.int32)
    i = a @ b.T
    d = a.sum(1)[:, None] + b.sum(1)[None] - i
    return (i / np.maximum(d, 1)).max(1).astype(np.float32)


def pool(x, masks):
    o = []
    for m in masks:
        q = p3mask(m).astype(bool)
        o.append(x[q].mean(0) if q.any() else np.zeros(x.shape[1]))
    return np.asarray(o, np.float32)


def pack_scene(path, split, i, item, p3, pred, sc, ss, p):
    sat_c, bev_cc = channels(p3, sc, p)
    _, bev_cs = channels(p3, ss, p)
    gt = np.any(item['gt_masks'].astype(bool), 0) if len(item['gt_masks']) else np.zeros((1024, 1024), bool)
    yb = p3mask(boundary(gt))
    pe = np.any(pred, 0) if len(pred) else np.zeros_like(gt)
    ye = p3mask(boundary(pe) ^ boundary(gt))
    q = iiou(pred, item['gt_masks'])
    d = {'scene_id': np.array(item['img_metas']['scene_id']), 'yb': yb, 'ye': ye,
         'pred': pred, 'qi': q}
    if split == 'train':
        for tid, (name, y) in enumerate((('boundary', yb), ('error', ye))):
            pos = np.flatnonzero(y)
            neg = np.flatnonzero(~y.astype(bool))
            n = min(256, len(pos), len(neg))
            if not n:
                raise RuntimeError('class missing')
            r = np.random.default_rng(np.random.SeedSequence([SEED, i, tid]))
            sel = np.r_[r.choice(pos, n, False), r.choice(neg, n, False)]
            d['y_' + name] = y[sel]
            d['sat_' + name] = sat_c[sel]
            d['bevc_' + name] = bev_cc[sel]
            d['bevs_' + name] = bev_cs[sel]
        d['y_iou'] = q
        d['iou_sat'] = pool(sat_c, pred)
        d['iou_bevc'] = pool(bev_cc, pred)
        d['iou_bevs'] = pool(bev_cs, pred)
    else:
        d['x_sat'] = sat_c
        d['x_bevc'] = bev_cc
        d['x_bevs'] = bev_cs
    atomic_npz(path, **d)


def fingerprints():
    files = (Path(__file__).resolve(), ROOT / 'rsprompter/surfacebev.py',
             ROOT / 'data/ue_raw_pair_dataset.py', ROOT / 'scripts/ue/probe_surfacebev_p1_h0.py',
             ROOT / 'inference/infer_from_checkpoint.py')
    return {str(p_.relative_to(ROOT)): hashlib.sha256(p_.read_bytes()).hexdigest() for p_ in files}


def main():
    a = args()
    h = json.loads(H0.read_text())
    if h.get('verdict') != 'H0_PASS':
        raise RuntimeError('H0 not passed')
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache = out / 'cache'
    cache.mkdir(exist_ok=True)
    mark = torch.arange(128 * 128).reshape(128, 128)
    pack = mark[None, :, :, None].repeat(12, 1, 1, 10).permute(1, 2, 0, 3).reshape(-1, 120)
    if not torch.equal(pack[:, 0], mark.reshape(-1)) or not torch.equal(pack[:, 110], mark.reshape(-1)):
        raise RuntimeError('P3/SurfaceBEV packing failed')
    dev = torch.device(a.device)
    m, e, cfg, source = build(a, dev)
    ed = torch.tensor(h['height_edges_cm'], device=dev)
    p = projections(dev)
    contract = {'version': VERSION, 'seed': SEED, 'checkpoint_sha256': EXPECTED_B0_SHA256,
                'h0_sha256': sha_file(H0), 'split': 'surfbev_b0_split_v2.json',
                'height_edges_cm': h['height_edges_cm'],
                'sat_dim': 128, 'bev_dim': 120, 'per_class_cap': 256,
                'anchor_load': {'worktree_commit': '807d220', 'config_source': str(source),
                                'architecture': snap_arch(cfg)},
                'implementation_fingerprints': fingerprints()}
    atomic_json(cache / 'contract.json', contract)
    ds = {s: UERawPairDataset(a.root, s, True, 'surfbev_b0_split_v2.json', load_targets=True)
          for s in ('train', 'val')}
    tele = cache / 'telemetry_train_val.jsonl'
    for s, lim in (('train', a.max_train), ('val', a.max_val)):
        n = len(ds[s]) if not lim else min(lim, len(ds[s]))
        d = cache / s
        d.mkdir(exist_ok=True)
        for i in range(n):
            f = d / f'{i:04d}.npz'
            if f.exists():
                print(s, i + 1, '/', n, 'cached', flush=True)
                continue
            it = ds[s][i]
            p3, pred, ph = pred_p3(m, it, dev)
            desc = uav_descriptors(e, it['drone_images'].to(dev))
            uk, uc, sk, sw = cameras(it, dev)
            sc, ct = rasterize_surfacebev(desc, it['devicez_rgb'].to(dev), uk, uc, sk, sw, ed, (128, 128))
            ss, st = rasterize_surfacebev(desc, it['devicez_rgb'].to(dev), uk, uc, sk, sw, ed, (128, 128), True)
            if ct['input_points'] != st['input_points'] or not (torch.isfinite(sc).all() and torch.isfinite(ss).all()):
                raise RuntimeError('surface contract')
            for z in (ct, st):
                if z['zbuffer_points'] <= 0 or z['covered_cells'] <= 0 or not np.isfinite(z['zbuffer_residual_mean']):
                    raise RuntimeError('SurfaceBEV telemetry contract')
            pack_scene(f, s, i, it, p3, pred, sc, ss, p)
            with open(tele, 'a') as q:
                q.write(json.dumps({'split': s, 'scene_id': it['img_metas']['scene_id'],
                                    'prediction_hash': ph, 'descriptor_shape': list(desc.shape),
                                    'input_equal': True, 'correct': ct, 'sham90': st}) + '\n')
            print(s, i + 1, '/', n, flush=True)
    if a.max_train or a.max_val:
        atomic_json(out / 'smoke_summary.json', {'status': 'SMOKE_ONLY', 'train_used': a.max_train,
                                                 'val_used': a.max_val, 'test_read': False})
        return
    atomic_json(cache / 'train_val_complete.json', {'train': len(ds['train']), 'val': len(ds['val'])})


def snap_arch(cfg):
    try:
        m = cfg.model.to_dict() if hasattr(cfg.model, 'to_dict') else dict(cfg.model)
        return {'type': m.get('type'), 'keys': len(m)}
    except Exception:
        return {'type': None, 'keys': 0}


if __name__ == '__main__':
    main()
