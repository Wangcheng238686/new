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
from PIL import Image

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

D = Path('/data1/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021')
OUTDIR = Path('/data2/wangcheng/checkpoint/surfbev_b0/probes/CITYSAMPLE-TEST-PEEK-20261021')


def rle(m):
    return maskUtils.encode(np.asfortranarray(m.astype(np.uint8)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    dev = torch.device(a.device)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(str(ROOT / 'configs/ue_whu_a3sgm_512.py'))
    model = MODELS.build(cfg.model).to(dev).eval()
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    sd = ck['model'] if 'model' in ck else ck['state_dict']
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not [k for k in missing if 'lora' not in k], f'missing {missing[:5]}'
    assert not unexpected, f'unexpected {unexpected[:5]}'
    print(f'[{a.tag}] loaded ckpt epoch={ck.get("epoch")} split={a.split}')

    coco = COCO(str(D / f'annotations/{a.split}.json'))
    img_ids = coco.getImgIds()
    results = []
    for n, iid in enumerate(img_ids):
        e = coco.loadImgs([iid])[0]
        img = np.asarray(Image.open(D / e['file_name']).convert('RGB'))
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
        if (n + 1) % 50 == 0:
            print(f'{n+1}/{len(img_ids)} scenes, preds={len(results)}', flush=True)

    out = {'tag': a.tag, 'ckpt': a.ckpt, 'epoch': ck.get('epoch'), 'split': a.split,
           'n_scenes': len(img_ids), 'n_pred': len(results)}
    for t in ('segm', 'bbox'):
        dt = coco.loadRes([dict(r) for r in results])
        ev = COCOeval(coco, dt, t)
        ev.evaluate(); ev.accumulate(); ev.summarize()
        out[t] = {lab: float(ev.stats[i]) for i, lab in enumerate(
            ('AP', 'AP50', 'AP75', 'AP_s', 'AP_m', 'AP_l'))}
    (OUTDIR / f'{a.tag}_{a.split}.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
