#!/usr/bin/env python3
"""R1C-ATTR attribution ladder (2026-10-17 prereg): eval-time controls that
separate "attention routes UAV scene content" from "module params / static
prior + frozen base" contributions.

C1 mismatch: scene i is fed the UAV payload of scene N-1-i (the v2 split is
spatially ordered, so reversed pairing = maximally distant scenes). The
satellite grid/queries stay correct for scene i; only the content is wrong.
C2 constant: the pathway's descriptor block is replaced by the train-split
mean output map (30 scenes); the 6 stats channels stay per-scene live.
C4 constant-full: the ENTIRE bev (descriptor + stats) replaced by the
train-split mean - separates live-stats content from pure template.

Usage: probe_attr_ladder.py --ckpt PATH --tag t --mode {c1,c2} [--attn|--raster]
  --attn   : R1C arm - replace/permute the attention descriptor block
  --raster : R1G arm - replace/permute the rasterized descriptor block [0:256]
"""
from __future__ import annotations
import argparse, json, os, sys, types
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

OUTDIR = Path('/data2/wangcheng/checkpoint/surfbev_b0/probes/WARM-SCREEN-20261017')


@torch.inference_mode()
def harvest_mean_block(model, ds, n, mode, dev):
    """Mean descriptor-block output over the first n TRAIN scenes."""
    acc = None
    for i in range(n):
        it = ds[i]
        model._uav_pending = L.uav_payload(it, dev)
        data = L.model_input(model, it, dev)
        model.predict(data['inputs'], data['data_samples'], rescale=False)
        blk = model._attr_block_cache
        model._attr_block_cache = None
        acc = blk.float().cpu() if acc is None else acc + blk.float().cpu()
    return (acc / n)


def install_hooks(model, mode):
    """Cache the descriptor block per forward and allow replacement."""
    core = model
    use_attn = getattr(core, 'bev_desc_attention', False)
    orig_uav_bev = core._uav_bev

    def patched(uav, hw, sat_feat=None):
        bev, cov = orig_uav_bev(uav, hw, sat_feat=sat_feat)
        if mode in ('full', 'c4'):    # C4: the whole BEV tensor
            core._attr_block_cache = bev
        elif use_attn:              # R1C: descriptor block = channels [0:attn_dim]
            d = int(core.bev_attn.v_proj.out_features)
            core._attr_block_cache = bev[:d]
        else:                       # R1G: rasterized descriptor block = [0:256]
            core._attr_block_cache = bev[:256]
        return bev, cov
    core._uav_bev = patched


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--mode', choices=('c1', 'c2', 'c4'), required=True)
    ap.add_argument('--train-scenes', type=int, default=30)
    ap.add_argument('--max-val', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    dev = torch.device(a.device)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows_path = OUTDIR / f'rows_{a.tag}_{a.mode}.jsonl'
    if rows_path.exists():
        print(f'{rows_path} exists; skip', flush=True)
        return

    manifest = OUTDIR / f'identity_{a.tag}.json'
    if not manifest.exists():
        ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
        manifest.write_text(json.dumps(L.checkpoint_identity(a.ckpt, ck), indent=2))
    model, _, _, _ = build_baseline(a.ckpt, manifest, os.environ['SAM2_CKPT'], dev)
    use_attn = bool(getattr(model, 'bev_desc_attention', False))
    dim = int(model.bev_attn.v_proj.out_features) if use_attn else 256
    install_hooks(model, a.mode)

    if a.mode == 'c4':
        # full-BEV constant: descriptor block AND stats block both replaced
        # by the train-split mean - separates live-stats content from pure
        # template (forced by the C2~=C0 outcome; amendment logged in prereg)
        ds_tr = L.UERawPairDataset(str(L.SNAP), 'train', True,
                                   'surfbev_b0_split_v2.json', image_size=(512, 512))
        mean_full = harvest_mean_block(model, ds_tr, a.train_scenes, 'full', dev)
        orig_uav_bev4 = model._uav_bev

        def const_full(uav, hw, sat_feat=None, _orig=orig_uav_bev4, _m=mean_full):
            bev, cov = _orig(uav, hw, sat_feat=sat_feat)
            if tuple(bev.shape[-2:]) != tuple(_m.shape[-2:]):
                _mb = torch.nn.functional.interpolate(_m[None], size=bev.shape[-2:],
                                                      mode='bilinear', align_corners=False)[0]
            else:
                _mb = _m
            return _mb.to(bev.device, bev.dtype), cov
        model._uav_bev = const_full

    if a.mode == 'c2':
        ds_tr = L.UERawPairDataset(str(L.SNAP), 'train', True,
                                   'surfbev_b0_split_v2.json', image_size=(512, 512))
        mean_blk = harvest_mean_block(model, ds_tr, a.train_scenes, a.mode, dev)
        # freeze the block: patch _uav_bev to splice the mean back in
        orig_uav_bev = model._uav_bev

        def const_block(uav, hw, sat_feat=None, _orig=orig_uav_bev, _m=mean_blk, _d=dim):
            bev, cov = _orig(uav, hw, sat_feat=sat_feat)  # [C,H,W]
            if tuple(bev.shape[-2:]) != tuple(_m.shape[-2:]):
                _mb = torch.nn.functional.interpolate(_m[None], size=bev.shape[-2:],
                                                      mode='bilinear', align_corners=False)[0]
            else:
                _mb = _m
            return torch.cat((_mb.to(bev.device, bev.dtype), bev[_d:]), dim=0), cov
        model._uav_bev = const_block

    ds = L.UERawPairDataset(str(L.SNAP), 'val', True, 'surfbev_b0_split_v2.json',
                            image_size=(512, 512))
    if a.max_val:
        ds.records = ds.records[:a.max_val]
    N = len(ds)
    rows = []
    for i in range(N):
        it = ds[i]
        if a.mode == 'c1':
            # reversed pairing: content of scene N-1-i, satellite of scene i
            pit = ds[N - 1 - i]
            model._uav_pending = L.uav_payload(pit, dev)
        data = L.model_input(model, it, dev)
        if a.mode != 'c1':
            model._uav_pending = L.uav_payload(it, dev)
        o = model.predict(data['inputs'], data['data_samples'], rescale=False)
        pi = o[0].pred_instances
        # per-scene COCO AP identical to the ladder protocol
        row = {'scene_id': it['img_metas']['scene_id']}
        from pycocotools import mask as maskUtils
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        masks = pi.masks
        masks = masks.masks if hasattr(masks, 'masks') else (masks.detach().cpu().numpy()
                                                             if isinstance(masks, torch.Tensor) else np.asarray(masks))
        masks = np.asarray(masks).astype(bool)
        boxes = pi.bboxes.detach().cpu().numpy().astype(np.float64)
        scores = pi.scores.detach().cpu().numpy().astype(float)
        gt = np.asarray(it['gt_masks']).astype(bool)
        gt_anns, pred = [], []
        for g in (L.rle(m) for m in gt):
            x, y, w, h = maskUtils.toBbox(g)
            gt_anns.append({'id': len(gt_anns) + 1, 'image_id': 1, 'category_id': 1,
                            'segmentation': g, 'bbox': [float(x), float(y), float(w), float(h)],
                            'area': float(maskUtils.area(g)), 'iscrowd': 0})
        for b, s, m in zip(boxes, scores, masks):
            x1, y1, x2, y2 = b
            pred.append({'image_id': 1, 'category_id': 1, 'score': float(s),
                         'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                         'segmentation': L.rle(m)})
        row['n_gt'], row['n_pred'] = len(gt_anns), len(pred)
        for iou_type in ('bbox', 'segm'):
            coco = COCO()
            coco.dataset = {'images': [{'id': 1, 'height': 512, 'width': 512}],
                            'annotations': gt_anns, 'categories': [{'id': 1, 'name': 'b'}]}
            coco.createIndex()
            if pred:
                dt = coco.loadRes([dict(p) for p in pred])
                e = COCOeval(coco, dt, iou_type)
                e.params.imgIds = [1]
                e.evaluate(); e.accumulate(); e.summarize()
                row[iou_type] = float(e.stats[0])
            else:
                row[iou_type] = 0.0
        rows.append(row)
        if (i + 1) % 25 == 0:
            print(f'{a.mode} {i+1}/{N}', flush=True)
    tmp = rows_path.with_suffix('.tmp')
    tmp.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    os.replace(tmp, rows_path)
    print(f'wrote {rows_path} segm={np.mean([r["segm"] for r in rows]):.4f} '
          f'bbox={np.mean([r["bbox"] for r in rows]):.4f}', flush=True)


if __name__ == '__main__':
    main()
