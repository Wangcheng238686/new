#!/usr/bin/env python
"""CitySample-Sat test 推理与评估 (MaskDINO, detectron2 v0.6 栈)

与 eval_maskdino_whu.py 的差异:
  - 数据集: citysample_v1_test (CitySample v1 卫星侧 459 图)
  - 评估: d2 CocoEvaluator(maxDets 硬编码 100) 弃用, 改为预测落盘后
    pycocotools maxDets=[1,10,250] + BoundaryAP(1%/2%)(boundary_ap.py)

用法:
  python citysample_sat_test_adet.py YAML CKPT --out-dir DIR [--dataset citysample_v1_test]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import torch

DS_ROOT = '/data1/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021'
GT_JSON = f'{DS_ROOT}/annotations/test.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('checkpoint')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--dataset', default='citysample_v1_test')
    ap.add_argument('--gt-json', default=GT_JSON)
    ap.add_argument('--max-images', type=int, default=0, help='0=全部(调试用)')
    ap.add_argument('--dump-prefix', default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    from detectron2.config import get_cfg
    from detectron2.data import DatasetCatalog
    from detectron2.data.datasets import register_coco_instances
    from maskdino import add_maskdino_config

    CS_ROOT = '/data1/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021'
    try:
        register_coco_instances(
            args.dataset, {},
            f'{CS_ROOT}/annotations/{args.dataset.split("_")[-1]}.json', CS_ROOT)
    except AssertionError:
        pass  # 已注册
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    from pycocotools import mask as _mm

    cfg = get_cfg()
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config)
    cfg.MODEL.WEIGHTS = args.checkpoint
    model = build_model(cfg)
    DetectionCheckpointer(model).load(args.checkpoint)
    model.eval()

    dataset_dicts = DatasetCatalog.get(args.dataset)
    if args.max_images:
        dataset_dicts = dataset_dicts[:args.max_images]

    preds = []
    n_img = 0
    with torch.no_grad():
        for d in dataset_dicts:
            h, w = d['height'], d['width']
            img = cv2.imread(d['file_name'], cv2.IMREAD_COLOR)
            outputs = model([{'image': torch.from_numpy(
                img.astype('float32').transpose(2, 0, 1)), 'height': h, 'width': w}])
            inst = outputs[0]['instances'].to('cpu')
            if inst.has('scores'):
                boxes = (inst.pred_boxes.tensor.numpy()
                         if inst.has('pred_boxes') else None)
                for k in range(len(inst)):
                    rec = {'image_id': d['image_id'], 'category_id': 1,
                           'score': float(inst.scores[k])}
                    if boxes is not None:
                        x1, y1, x2, y2 = boxes[k][:4]
                        rec['bbox'] = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                    if inst.has('pred_masks'):
                        m = np.asfortranarray(inst.pred_masks[k].numpy().astype(np.uint8))
                        rec['segmentation'] = _mm.encode(m)
                    if 'segmentation' in rec:
                        preds.append(rec)
            n_img += 1

    if args.dump_prefix:
        def _enc(o):
            if isinstance(o, (bytes, bytearray)):
                return o.decode('utf-8', 'replace')
            if isinstance(o, (np.ndarray, np.generic)):
                return o.item() if o.ndim == 0 else o.tolist()
            return o
        out_json = args.dump_prefix + '.segm.json'
        with open(out_json, 'w') as f:
            json.dump(preds, f, cls=lambda *a, **k: json.JSONEncoder(
                *a, default=_enc, **k))
        print(f'dumped {len(preds)} preds -> {out_json}')

    from boundary_ap import evaluate_boundary, evaluate_maxdets
    print(f'\n===== CitySample-Sat test metrics (maxDets=250) =====')
    metrics = evaluate_maxdets(preds, args.gt_json)
    metrics.update(evaluate_boundary(preds, args.gt_json))
    for k in sorted(metrics):
        print(f'{k}: {metrics[k]:.4f}' if isinstance(metrics[k], float) else f'{k}: {metrics[k]}')
    out_json = os.path.join(args.out_dir, 'test_metrics.json')
    with open(out_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'saved -> {out_json} | images={n_img}')


if __name__ == '__main__':
    main()
