#!/usr/bin/env python3
"""Ad-hoc PI-authorized test peek (2026-10-21): COCO segm/bbox AP of the
running CitySample single-stream baselines' BEST checkpoints on the sealed
test split (459 scenes). This is a mid-training preview, NOT the
preregistered EXP-0024/25 test收口 (which stays last-epoch-100, both arms,
scene-paired bootstrap).

Usage: eval_citysample_test.py --ckpt PATH --tag NAME [--split test]
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch
import cv2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'inference'))
sys.path.insert(0, str(HERE))

import ue.dual  # noqa: F401  (registers RSPrompterAnchor space)
from mmengine.config import Config
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
# nested BaseModel.__init__ resolves data_preprocessor through the mmengine
# root registry - mirror ue.dual's dual-scope registration for the plain type
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmdet.models.data_preprocessors import DetDataPreprocessor
try:
    MMENGINE_MODELS.register_module(module=DetDataPreprocessor)
except Exception:
    pass
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

OUTDIR = Path('/data2/wangcheng/checkpoint/surfbev_b0/probes/CITYSAMPLE-TEST-PEEK-20261021')


def rle(m):
    return maskUtils.encode(np.asfortranarray(m.astype(np.uint8)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data-root', default='/data1/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--max-per-img', type=int, default=0,
                    help='override test_cfg.rcnn.max_per_img (0 = config default 100); '
                         'test has 41% scenes >100 GT (max 238) - the default cap binds')
    a = ap.parse_args()
    D = Path(a.data_root)
    dev = torch.device(a.device)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(str(ROOT / 'configs/ue_whu_a3sgm_512.py'))
    model = MODELS.build(cfg.model).to(dev).eval()
    if a.max_per_img:
        model.test_cfg['rcnn']['max_per_img'] = int(a.max_per_img)
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    sd = ck['model'] if 'model' in ck else ck['state_dict']
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not [k for k in missing if 'lora' not in k], f'missing {missing[:5]}'
    assert not unexpected, f'unexpected {unexpected[:5]}'
    print(f'[{a.tag}] loaded ckpt epoch={ck.get("epoch")} split={a.split}')

    coco = COCO(str(D / f'annotations/{a.split}.json'))
    img_ids = coco.getImgIds()
    ann_count = {iid: len(coco.getAnnIds(imgIds=[iid])) for iid in img_ids}
    results = []
    per_scene_pred = {}
    for n, iid in enumerate(img_ids):
        e = coco.loadImgs([iid])[0]
        # channel-order fix (2026-10-22): read with OpenCV => BGR, exactly
        # like the training/val loader; the preprocessor's bgr_to_rgb then
        # yields RGB for the model. The previous PIL-RGB read was flipped
        # twice and fed the model BGR (all prior test reads under-report).
        img = cv2.imread(str(D / e['file_name']))
        if img is None:
            raise RuntimeError(f'unreadable image {e["file_name"]}')
        sample = DetDataSample()
        sample.set_metainfo(dict(ori_shape=(512, 512), img_shape=(512, 512),
                                 scale_factor=(1.0, 1.0), scene_id=str(iid)))
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
            results.append({'image_id': iid, 'category_id': 1, 'score': float(s),
                            'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                            'segmentation': rle(m)})
        per_scene_pred[int(iid)] = int(len(boxes))
        if (n + 1) % 50 == 0:
            print(f'{n+1}/{len(img_ids)} scenes, preds={len(results)}', flush=True)

    out = {'tag': a.tag, 'ckpt': a.ckpt, 'epoch': ck.get('epoch'), 'split': a.split,
           'n_scenes': len(img_ids), 'n_pred': len(results),
           'max_per_img': int(a.max_per_img) if a.max_per_img else 100,
           'scenes_at_cap': int(sum(1 for v in per_scene_pred.values() if v >= (a.max_per_img or 100))),
           'gt_gt_cap': int(sum(1 for v in ann_count.values() if v > (a.max_per_img or 100)))}
    for t in ('segm', 'bbox'):
        dt = coco.loadRes([dict(r) for r in results])
        ev = COCOeval(coco, dt, t)
        # COCOeval's own maxDets=100 default silently discards outputs beyond
        # rank 100 - align it with the model cap or the override is unmeasured
        md = int(a.max_per_img) if a.max_per_img else 100
        ev.params.maxDets = [1, 10, md]
        ev.evaluate(); ev.accumulate()
        # non-default maxDets makes summarize() stats[0] return -1 (pycocotools
        # quirk) - compute the COCO stats directly from the precision array
        p = ev.params
        prec = ev.eval['precision']                      # [T,R,K,A,M]
        stats = {}
        def ap_at(iou_idx=None, area_idx=0):
            sl = prec[:, :, 0, area_idx, 2]
            sl = sl if iou_idx is None else sl[iou_idx:iou_idx + 1]
            v = sl[sl > -1]
            return float(v.mean()) if v.size else float('nan')
        ar = {name: i for i, name in enumerate(p.areaRngLbl)}
        stats['AP'] = ap_at(None, ar['all'])
        stats['AP50'] = ap_at(0, ar['all'])
        stats['AP75'] = ap_at(5, ar['all'])
        for s_, n_ in (('AP_s', 'small'), ('AP_m', 'medium'), ('AP_l', 'large')):
            stats[s_] = ap_at(None, ar[n_])
        out[t] = stats
        print(t, json.dumps({k: round(v, 4) for k, v in stats.items()}))
    (OUTDIR / f'{a.tag}_{a.split}.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
