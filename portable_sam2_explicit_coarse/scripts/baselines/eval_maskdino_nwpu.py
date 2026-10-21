#!/usr/bin/env python
"""NWPU VHR-10 val 集推理与可视化 (MaskDINO, detectron2 v0.6 官方栈)

val 兼任 test 终评 (130 图/743 实例, 10 类)。其余同 eval_maskdino_whu.py:
官方 test 预处理 DatasetMapper(is_train=False) —— RGB + MIN_SIZE_TEST=800
(MAX 1333) + pixel mean/std 归一化; 输出由 detector_postprocess 还原到原图尺度
输出: bbox 与 segm 的 mAP / AP50 / AP75 / APs / APm / APl (COCOEvaluator)

用法:
  python eval_maskdino_nwpu.py CKPT --config CONFIG [--out-dir DIR] [--vis]
"""
import argparse
import json
import os

import cv2
import torch

PALETTE = [
    (106, 0, 255), (182, 89, 155), (183, 193, 0), (1, 158, 137),
    (27, 106, 255), (213, 172, 0), (191, 128, 64), (30, 128, 31),
    (151, 31, 160), (23, 43, 201), (222, 157, 82), (164, 156, 9),
    (56, 2, 122), (18, 183, 241), (17, 95, 122), (23, 208, 53),
]
MASK_ALPHA = 0.35

# 单色模式(对比图用): VIZ_MONO_BGR="B,G,R" 所有实例同色; VIZ_ALPHA 覆盖透明度
VIZ_MONO_BGR = os.environ.get('VIZ_MONO_BGR', '')
VIZ_ALPHA = float(os.environ.get('VIZ_ALPHA', str(MASK_ALPHA)))
def _viz_color(i):
    if VIZ_MONO_BGR:
        return tuple(int(c) for c in VIZ_MONO_BGR.split(','))
    return PALETTE[i % len(PALETTE)]


def visualize(d, instances, vis_dir, score_thr):
    img = cv2.imread(d['file_name'], cv2.IMREAD_COLOR)
    keep = instances.scores > score_thr
    masks = instances.pred_masks[keep].cpu().numpy()
    overlay = img.copy()
    for i, m in enumerate(masks):
        if m.max() == 0:
            continue
        overlay[m > 0] = _viz_color(i)
    out = cv2.addWeighted(overlay, VIZ_ALPHA, img, 1.0 - VIZ_ALPHA, 0)
    stem = os.path.splitext(os.path.basename(d['file_name']))[0]
    cv2.imwrite(os.path.join(vis_dir, stem + '.png'), out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    ap.add_argument('--config', default='configs/nwpu/maskdino_R50_nwpu.yaml')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--vis', action='store_true')
    ap.add_argument('--vis-limit', type=int, default=20)
    ap.add_argument('--score-thr', type=float, default=0.5)
    ap.add_argument('--max-images', type=int, default=0)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.checkpoint), 'test_out')
    os.makedirs(out_dir, exist_ok=True)
    vis_dir = os.path.join(out_dir, 'vis') if args.vis else None
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    from detectron2.config import get_cfg
    from detectron2.data import build_detection_test_loader
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.evaluation.coco_evaluation import COCOEvaluator

    from maskdino import add_maskdino_config
    from train_nwpu import register_nwpu

    register_nwpu()
    cfg = get_cfg()
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config)
    cfg.MODEL.WEIGHTS = args.checkpoint
    cfg.freeze()

    model = build_model(cfg)
    DetectionCheckpointer(model).load(args.checkpoint)
    model.eval()

    # val 兼任 test 终评
    evaluator = COCOEvaluator('nwpu_val', tasks=None, distributed=False,
                              output_dir=out_dir)
    evaluator.reset()
    loader = build_detection_test_loader(cfg, 'nwpu_val')

    n, n_vis = 0, 0
    with torch.no_grad():
        for inputs in loader:
            outputs = model(inputs)
            inst = outputs[0]['instances'].to('cpu')
            d = {'file_name': inputs[0]['file_name'],
                 'height': inputs[0]['height'], 'width': inputs[0]['width'],
                 'image_id': inputs[0]['image_id']}
            evaluator.process([d], [{'instances': inst}])
            if vis_dir and n_vis < args.vis_limit:
                visualize(d, inst, vis_dir, args.score_thr)
                n_vis += 1
            n += 1
            if args.max_images and n >= args.max_images:
                break

    results = evaluator.evaluate()
    print('\n===== NWPU VHR-10 val(=test) metrics =====')
    flat = {}
    for task, metrics in results.items():
        print(f'--- {task} ---')
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                print(f'{k}: {v:.4f}')
                flat[f'{task}/{k}'] = float(v)
    out_json = os.path.join(out_dir, 'test_metrics.json')
    with open(out_json, 'w') as f:
        json.dump(flat, f, indent=2)
    print(f'saved -> {out_json} | images={n} vis={n_vis}')


if __name__ == '__main__':
    main()
