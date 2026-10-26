#!/usr/bin/env python3
"""WARM-BA-H precondition gate (prereg docs/WARM-BA-H-20261022_preregistration.md):
error decomposition of the v1 single-stream base (EXP-0025 best ep71) on val 459.

Measures whether split-type INTERFACE errors (adjacent predicted pairs matching
the SAME GT instance) carry enough mass for a merge-adjudicator to matter:
  gate = split-pair members >= 3% of per-scene predictions
         AND >= 10% of total error mass (misses + false alarms + split members
         + merge-type preds). Also caches per-scene predictions (RLE+boxes+
scores) for the adjudication harvest. Protocol: cap250 + maxDets250 +
channel-order-fixed, identical to the ledger anchor reads.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import cv2
from scipy import ndimage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'inference'))
sys.path.insert(0, str(HERE))

import ue.dual  # noqa: F401
from mmengine.config import Config
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmdet.models.data_preprocessors import DetDataPreprocessor
from pycocotools import mask as maskUtils
from citysample_pair import CitySamplePairDataset

try:
    MMENGINE_MODELS.register_module(module=DetDataPreprocessor)
except Exception:
    pass

D = Path('/data1/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021')
OUTDIR = Path('/data2/wangcheng/checkpoint/surfbev_b0/probes/WARM-BA-H-20261022')
CKPT = '/data2/wangcheng/checkpoint/citysample_v1/citysample_v1_512single_defaug_ddp4_s44_EXP-0025/best_model.pth'
R = 3
IOU_MATCH = 0.5


def rle(m):
    return maskUtils.encode(np.asfortranarray(m.astype(np.uint8)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    dev = torch.device(a.device)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cfg = Config.fromfile(str(ROOT / 'configs/ue_whu_a3sgm_512.py'))
    model = MODELS.build(cfg.model).to(dev).eval()
    model.test_cfg['rcnn']['max_per_img'] = 250
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(ck['model'], strict=False)
    assert not [k for k in missing if 'lora' not in k] and not unexpected
    print(f'base loaded epoch={ck.get("epoch")}', flush=True)

    ds = CitySamplePairDataset(split='val', limit=459)
    cache = open(OUTDIR / 'pred_cache_val.jsonl', 'w')
    agg = dict(scenes=0, preds=0, gts=0, miss=0, false=0, split_pairs=0,
               split_members=0, merge_preds=0, soft_split_pairs=0)
    per_scene = []
    for i in range(len(ds)):
        it = ds[i]
        img = cv2.imread(str(D / f'points/{it["scene_id"]}/satellite/rgb.png'))
        sample = DetDataSample()
        sample.set_metainfo(dict(ori_shape=(512, 512), img_shape=(512, 512),
                                 scale_factor=(1.0, 1.0)))
        data = model.data_preprocessor(
            {'inputs': [torch.from_numpy(img.copy()).permute(2, 0, 1)],
             'data_samples': [sample]}, training=False)
        with torch.no_grad():
            o = model.predict(data['inputs'], data['data_samples'], rescale=False)
        pi = o[0].pred_instances
        masks = pi.masks
        masks = masks.masks if hasattr(masks, 'masks') else (
            masks.detach().cpu().numpy() if isinstance(masks, torch.Tensor)
            else np.asarray(masks))
        P = np.asarray(masks).astype(bool)
        S = pi.scores.detach().cpu().numpy().astype(float)
        gt = it['gt_masks']
        n_pred, n_gt = len(P), len(gt)
        pr = [rle(m) for m in P]

        # matching matrix pred x gt
        if n_pred and n_gt:
            gr = [rle(m) for m in gt]
            ious = maskUtils.iou(pr, gr, [0] * len(gr))   # [n_pred, n_gt]
            best_gt = ious.argmax(1)
            best_iou = ious.max(1)
        else:
            ious = np.zeros((n_pred, max(n_gt, 1)))
            best_gt, best_iou = np.zeros(0, int), np.zeros(0)
        matched_gt = set(best_gt[best_iou >= IOU_MATCH].tolist())
        miss = n_gt - len(matched_gt)
        false = int((best_iou < IOU_MATCH).sum())
        # merge-type: pred matching >=2 GTs at >=0.5
        merge_preds = int(sum(1 for p in range(n_pred)
                              if (ious[p] >= IOU_MATCH).sum() >= 2))
        # split-type: adjacent pred pairs with same matched GT (both >=0.5)
        split_members = set()
        soft_members = set()
        n_split_pairs = n_soft = 0
        for x in range(n_pred):
            for y in range(x + 1, n_pred):
                Ax = ndimage.binary_dilation(P[x], iterations=R)
                By = ndimage.binary_dilation(P[y], iterations=R)
                if not (Ax & By).any():
                    continue
                if best_iou[x] >= IOU_MATCH and best_iou[y] >= IOU_MATCH \
                        and best_gt[x] == best_gt[y]:
                    n_split_pairs += 1
                    split_members.update((x, y))
                elif best_iou[x] >= 0.3 and best_iou[y] >= 0.3 \
                        and best_gt[x] == best_gt[y]:
                    n_soft += 1
                    soft_members.update((x, y))
        agg['scenes'] += 1
        agg['preds'] += n_pred
        agg['gts'] += n_gt
        agg['miss'] += miss
        agg['false'] += false
        agg['split_pairs'] += n_split_pairs
        agg['split_members'] += len(split_members)
        agg['merge_preds'] += merge_preds
        agg['soft_split_pairs'] += n_soft
        per_scene.append(dict(scene=it['scene_id'], n_pred=n_pred, n_gt=n_gt,
                              miss=miss, false=false, split=len(split_members),
                              merge=merge_preds))
        cache.write(json.dumps(dict(
            scene_id=it['scene_id'], scores=S.tolist(),
            rles=[{'size': r['size'],
                   'counts': r['counts'].decode() if isinstance(r['counts'], bytes) else r['counts']}
                  for r in pr])) + '\n')
        if (i + 1) % 50 == 0:
            print(f'{i+1}/{len(ds)} scenes', flush=True)
    cache.close()

    total_err = agg['miss'] + agg['false'] + agg['split_members'] + agg['merge_preds']
    out = dict(agg, total_error_mass=total_err,
               split_frac_of_preds=agg['split_members'] / max(agg['preds'], 1),
               split_frac_of_errors=agg['split_members'] / max(total_err, 1))
    out['gate'] = {'rule': 'split members >= 3% of preds AND >= 10% of error mass',
                   'pass': bool(out['split_frac_of_preds'] >= 0.03
                                and out['split_frac_of_errors'] >= 0.10)}
    (OUTDIR / 'decompose_val.json').write_text(json.dumps(out, indent=2))
    (OUTDIR / 'decompose_per_scene.json').write_text(json.dumps(per_scene))
    print(json.dumps(out, indent=2))
    print('PRECONDITION:', 'PASS' if out['gate']['pass'] else 'FAIL', flush=True)


if __name__ == '__main__':
    main()
