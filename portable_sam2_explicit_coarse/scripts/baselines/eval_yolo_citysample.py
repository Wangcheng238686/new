#!/usr/bin/env python
"""CitySample-Sat test 推理与评估 (YOLO11-seg, ultralytics)

用 best.pt 在 test 459 上预测 -> 输出 COCO json -> pycocotools 评估
  bbox 与 segm 的 mAP / AP50 / AP75 / APs / APm / APl, maxDets=[1,10,250]
  (协议 P1/P2: test 单图实例最大 238, 250 上限; imgsz=512 原生不上采样)

用法:
  python eval_yolo_citysample.py CKPT [--out-dir DIR] [--vis] [--vis-limit 20]
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils

# 16 visually distinct colors (BGR) —— 与本项目 visualize_instances.py 保持一致
PALETTE = [
    (106, 0, 255), (182, 89, 155), (183, 193, 0), (1, 158, 137),
    (27, 106, 255), (213, 172, 0), (191, 128, 64), (30, 128, 31),
    (151, 31, 160), (23, 43, 201), (222, 157, 82), (164, 156, 9),
    (56, 2, 122), (18, 183, 241), (17, 95, 122), (23, 208, 53),
]
MASK_ALPHA = 0.35

VIZ_MONO_BGR = os.environ.get('VIZ_MONO_BGR', '')
VIZ_ALPHA = float(os.environ.get('VIZ_ALPHA', str(MASK_ALPHA)))
def _viz_color(i):
    if VIZ_MONO_BGR:
        return tuple(int(c) for c in VIZ_MONO_BGR.split(','))
    return PALETTE[i % len(PALETTE)]


DS_ROOT = Path('/data1/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v3_20261021')
GT_JSON = str(DS_ROOT / 'annotations/test.json')
IMG_DIR = DS_ROOT / 'yolo_seg/images/test'

MAX_DET = 250      # P1/P2 协议
IMGSZ = 512        # P4: 原生分辨率, 不上采样


def stem2id_map(coco_gt):
    # file_name: points/000163/satellite/rgb.png -> stem 000163
    return {Path(im['file_name']).parts[1]: im['id']
            for im in coco_gt.dataset['images']}


def predict_to_coco(model, coco_gt, max_images=0):
    s2i = stem2id_map(coco_gt)
    cat_id = coco_gt.dataset['categories'][0]['id']
    preds = []
    results = model.predict(
        source=str(IMG_DIR), stream=True, batch=8, imgsz=IMGSZ,
        conf=0.001, iou=0.7, max_det=MAX_DET, device=0, verbose=False)
    n_img = 0
    for r in results:
        stem = Path(r.path).stem
        image_id = s2i[stem]
        h, w = r.orig_shape
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        scores = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)
        xyn = r.masks.xyn if r.masks is not None else []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            seg = None
            if i < len(xyn) and len(xyn[i]) >= 3:
                poly = (np.asarray(xyn[i]) * [w, h]).flatten().tolist()
                seg = [poly]
            if seg is None:
                continue
            preds.append(dict(
                image_id=image_id, category_id=cat_id,
                bbox=[float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                score=float(scores[i]), segmentation=seg))
        n_img += 1
        if max_images and n_img >= max_images:
            break
    print(f'predicted {n_img} images, {len(preds)} instances')
    return preds


def evaluate(preds, gt_json, out_dir):
    coco_gt = COCO(gt_json)
    metrics = {}
    for task in ('bbox', 'segm'):
        coco_dt = coco_gt.loadRes(json.loads(json.dumps(preds)))
        if task == 'segm':
            for p in coco_dt.anns.values():
                img = coco_gt.imgs[p['image_id']]
                h, w = img['height'], img['width']
                rles = mask_utils.frPyObjects(p['segmentation'], h, w)
                p['segmentation'] = rles[0] if len(rles) == 1 else mask_utils.merge(rles)
        E = COCOeval(coco_gt, coco_dt, task)
        # P1 协议: maxDets=[1,10,250] (test 单图实例最大 238)
        E.params.maxDets = [1, 10, MAX_DET]
        E.evaluate()
        E.accumulate()
        E.summarize()
        s = E.stats
        # pycocotools 陷阱: summarize() 的 s[0](AP@[.50:.95]) 硬编码 maxDets=100,
        # 我们的 maxDets 无 100 条目会得 -1; 从 eval 数组显式按 maxDets=250 计算
        prec = E.eval['precision'][:, :, :, 0, -1]
        prec = prec[prec > -1]
        ap_all = float(prec.mean()) if prec.size else float('nan')
        metrics[f'{task}_mAP'] = round(ap_all, 4)
        metrics[f'{task}_mAP50'] = round(float(s[1]), 4)
        metrics[f'{task}_mAP75'] = round(float(s[2]), 4)
        metrics[f'{task}_mAPs'] = round(float(s[3]), 4)
        metrics[f'{task}_mAPm'] = round(float(s[4]), 4)
        metrics[f'{task}_mAPl'] = round(float(s[5]), 4)
    os.makedirs(out_dir, exist_ok=True)
    out = Path(out_dir) / 'test_metrics.json'
    out.write_text(json.dumps(metrics, indent=2))
    print('saved', out)
    return metrics


def visualize(model, vis_dir, vis_limit=20, score_thr=0.5):
    os.makedirs(vis_dir, exist_ok=True)
    results = model.predict(
        source=str(IMG_DIR), stream=True, batch=8, imgsz=IMGSZ,
        conf=score_thr, iou=0.7, max_det=MAX_DET, device=0, verbose=False)
    for n, r in enumerate(results):
        if n >= vis_limit:
            break
        img = r.orig_img
        overlay = img.copy()
        masks = r.masks.data.cpu().numpy() if r.masks is not None else np.zeros((0, *img.shape[:2]))
        for i, m in enumerate(masks):
            if m.max() == 0:
                continue
            m_resized = cv2.resize(m, (img.shape[1], img.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
            overlay[m_resized > 0] = _viz_color(i)
        out = cv2.addWeighted(overlay, VIZ_ALPHA, img, 1.0 - VIZ_ALPHA, 0)
        cv2.imwrite(str(Path(vis_dir) / (Path(r.path).stem + '.png')), out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--vis', action='store_true')
    ap.add_argument('--vis-limit', type=int, default=20)
    ap.add_argument('--score-thr', type=float, default=0.5)
    ap.add_argument('--max-images', type=int, default=0)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.checkpoint)

    out_dir = args.out_dir or str(Path(args.checkpoint).parent / 'test_out')
    os.makedirs(out_dir, exist_ok=True)

    coco_gt = COCO(GT_JSON)
    preds = predict_to_coco(model, coco_gt, args.max_images)
    (Path(out_dir)).mkdir(parents=True, exist_ok=True)
    suffix = '.subset' if args.max_images else ''
    (Path(out_dir) / f'pred_coco{suffix}.json').write_text(json.dumps(preds))
    evaluate(preds, GT_JSON, out_dir)
    if args.vis:
        visualize(model, str(Path(out_dir) / 'vis'), args.vis_limit,
                  args.score_thr)


if __name__ == '__main__':
    main()
