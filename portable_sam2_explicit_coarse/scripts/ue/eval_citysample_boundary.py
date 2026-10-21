#!/usr/bin/env python3
"""Boundary AP on the CitySample test split (PI spec 2026-10-21).

Protocol (standard boundary-IoU AP, e.g. Boundary IoU / BMask lineage):
boundary band = mask XOR erode(mask, 3x3 kernel x d iterations) with 1px
zero-padding so image-truncated boundaries count; d = round(ratio * diag),
ratio 0.02 -> 14 px (primary), 0.01 -> 7 px (strict) on 512x512.
GT and detections are converted to boundary RLEs, then evaluated with the
COCO machinery at fixed maxDets=250 (model cap 250). GT area stays the
ORIGINAL mask area so AP_s/m/l refer to real object sizes. Tiny masks whose
erosion is empty keep the full mask as their boundary (reference behaviour).
Standard segm AP (cap250) reported alongside for anchoring.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch
import cv2
from PIL import Image

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
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from citysample_pair import decode_rle

try:
    MMENGINE_MODELS.register_module(module=DetDataPreprocessor)
except Exception:
    pass

D = Path('/data1/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021')
OUTDIR = Path('/data2/wangcheng/checkpoint/surfbev_b0/probes/CITYSAMPLE-BOUNDARY-AP-20261021')
KERNEL = np.ones((3, 3), np.uint8)


def rle(m):
    return maskUtils.encode(np.asfortranarray(m.astype(np.uint8)))


def mask_to_boundary(mask, d):
    """Reference boundary band: pad 1px, erode 3x3 x d, band = mask & ~eroded."""
    m = mask.astype(np.uint8)
    ys, xs = np.where(m)
    y1, y2 = max(0, ys.min() - d - 2), min(512, ys.max() + d + 3)
    x1, x2 = max(0, xs.min() - d - 2), min(512, xs.max() + d + 3)
    crop = m[y1:y2, x1:x2]
    padded = cv2.copyMakeBorder(crop, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    eroded = cv2.erode(padded, KERNEL, iterations=d)[1:-1, 1:-1]
    band = ((crop > 0) & (eroded == 0)).astype(np.uint8)
    out = np.zeros((512, 512), np.uint8)
    out[y1:y2, x1:x2] = band
    return out


def coco_stats(ev):
    p = ev.params
    prec = ev.eval['precision']
    ar = {name: i for i, name in enumerate(p.areaRngLbl)}

    def ap_at(iou_idx=None, area_idx=0):
        sl = prec[:, :, 0, area_idx, 2]
        sl = sl if iou_idx is None else sl[iou_idx:iou_idx + 1]
        v = sl[sl > -1]
        return float(v.mean()) if v.size else float('nan')
    return {'AP': ap_at(None, ar['all']), 'AP50': ap_at(0, ar['all']),
            'AP75': ap_at(5, ar['all']), 'AP_s': ap_at(None, ar['small']),
            'AP_m': ap_at(None, ar['medium']), 'AP_l': ap_at(None, ar['large'])}


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--ckpt', required=True)
    ap_.add_argument('--tag', default='latest')
    ap_.add_argument('--split', default='test')
    ap_.add_argument('--max-per-img', type=int, default=250)
    ap_.add_argument('--ratios', default='0.02,0.01')
    ap_.add_argument('--device', default='cuda:0')
    a = ap_.parse_args()
    dev = torch.device(a.device)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ratios = [float(r) for r in a.ratios.split(',')]
    diags = {r: max(1, int(round(r * np.sqrt(512 ** 2 + 512 ** 2)))) for r in ratios}
    print('boundary dilation px:', diags)

    cfg = Config.fromfile(str(ROOT / 'configs/ue_whu_a3sgm_512.py'))
    model = MODELS.build(cfg.model).to(dev).eval()
    model.test_cfg['rcnn']['max_per_img'] = int(a.max_per_img)
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(ck['model'], strict=False)
    assert not [k for k in missing if 'lora' not in k] and not unexpected
    print(f'loaded epoch={ck.get("epoch")}')

    coco_raw = json.load(open(D / f'annotations/{a.split}.json'))
    gt_bnd = {r: {} for r in ratios}     # image_id -> list of boundary anns
    for ann in coco_raw['annotations']:
        m = decode_rle(ann['segmentation'])
        for r in ratios:
            gt_bnd[r].setdefault(ann['image_id'], []).append(
                {'id': ann['id'], 'image_id': ann['image_id'], 'category_id': 1,
                 'area': ann['area'], 'iscrowd': 0, 'segmentation': rle(mask_to_boundary(m, diags[r]))})
    imgs = [{'id': e['id'], 'width': 512, 'height': 512, 'file_name': e['file_name']}
            for e in coco_raw['images']]
    cats = [{'id': 1, 'name': 'building'}]

    det_std, det_bnd = [], {r: [] for r in ratios}
    for n, e in enumerate(coco_raw['images']):
        img = np.asarray(Image.open(D / e['file_name']).convert('RGB'))
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
        boxes = pi.bboxes.detach().cpu().numpy().astype(np.float64)
        scores = pi.scores.detach().cpu().numpy().astype(float)
        for b, s, m in zip(boxes, scores, np.asarray(masks).astype(bool)):
            x1, y1, x2, y2 = b
            det_std.append({'image_id': e['id'], 'category_id': 1, 'score': float(s),
                            'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                            'segmentation': rle(m)})
            for r in ratios:
                det_bnd[r].append({'image_id': e['id'], 'category_id': 1, 'score': float(s),
                                   'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                                   'segmentation': rle(mask_to_boundary(m, diags[r]))})
        if (n + 1) % 50 == 0:
            print(f'{n+1}/{len(imgs)} scenes', flush=True)

    out = {'tag': a.tag, 'ckpt': a.ckpt, 'epoch': ck.get('epoch'), 'split': a.split,
           'max_per_img': int(a.max_per_img), 'dilation_px': {str(r): diags[r] for r in ratios},
           'n_pred': len(det_std)}

    def run_eval(gt_anns_by_img, dets, name):
        coco = COCO()
        anns = [a_ for lst in gt_anns_by_img.values() for a_ in lst]
        coco.dataset = {'images': imgs, 'annotations': anns, 'categories': cats}
        coco.createIndex()
        dt = coco.loadRes([dict(d) for d in dets])
        ev = COCOeval(coco, dt, 'segm')
        ev.params.maxDets = [1, 10, int(a.max_per_img)]
        ev.evaluate(); ev.accumulate()
        out[name] = coco_stats(ev)
        print(name, json.dumps({k: round(v, 4) for k, v in out[name].items()}))

    # standard segm anchor (plain COCO on original masks)
    coco = COCO(str(D / f'annotations/{a.split}.json'))
    dt = coco.loadRes([dict(d) for d in det_std])
    ev = COCOeval(coco, dt, 'segm')
    ev.params.maxDets = [1, 10, int(a.max_per_img)]
    ev.evaluate(); ev.accumulate()
    out['standard_segm'] = coco_stats(ev)
    print('standard_segm', json.dumps({k: round(v, 4) for k, v in out['standard_segm'].items()}))
    for r in ratios:
        run_eval(gt_bnd[r], det_bnd[r], f'boundary_{str(r)}')
    (OUTDIR / f'{a.tag}_{a.split}.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
