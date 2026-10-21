#!/usr/bin/env python
"""WHU-1024 测试集推理与可视化 (detectron2/AdelaiDet 系: condinst / solov2)

输出: 指定任务(bbox/segm)的 AP / AP50 / AP75 / APs / APm / APl (CocoEvaluator)
可视化(可选): 仅将预测 mask 叠加到原图, 不画框不写置信度, 每实例独立颜色
  (调色板与 alpha=0.35 对齐本项目 scripts/visualize_instances.py 风格)

用法:
  python whu_test_adet.py YAML CKPT --out-dir DIR --tasks bbox,segm
      [--vis] [--vis-limit N] [--score-thr 0.5] [--max-images N]
"""
import argparse
import json
import os

import cv2
import numpy as np
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



def visualize(image_path, instances, vis_dir, score_thr):
    if len(instances) == 0 or not instances.has('pred_masks'):
        return  # 空预测(旧版 d2 会剥掉全部字段)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    keep = instances.scores > score_thr
    masks = instances.pred_masks[keep].cpu().numpy()
    overlay = img.copy()
    for i, m in enumerate(masks):
        if m.max() == 0:
            continue
        color = _viz_color(i)
        overlay[m > 0] = color
    out = cv2.addWeighted(overlay, VIZ_ALPHA, img, 1.0 - VIZ_ALPHA, 0)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(os.path.join(vis_dir, stem + '.png'), out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('checkpoint')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--tasks', default='segm', help='bbox,segm 或 segm(SOLOv2)')
    ap.add_argument('--dataset', default='whu1024_test')
    ap.add_argument('--vis', action='store_true')
    ap.add_argument('--vis-dir', default=None)
    ap.add_argument('--vis-limit', type=int, default=0, help='0=全部')
    ap.add_argument('--score-thr', type=float, default=0.5)
    ap.add_argument('--max-images', type=int, default=0, help='0=全部(调试用)')
    ap.add_argument('--dump-prefix', default=None,
                    help='COCO 格式落盘全部预测(bbox+segmentation)供选图/拼图')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    vis_dir = None
    if args.vis:
        vis_dir = args.vis_dir or os.path.join(args.out_dir, 'vis')
        os.makedirs(vis_dir, exist_ok=True)

    from adet.config import get_cfg
    import adet.data  # noqa: F401 注册 WHU 数据集
    from detectron2.data import DatasetCatalog
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.evaluation.coco_evaluation import COCOEvaluator as CocoEvaluator

    cfg = get_cfg()
    cfg.merge_from_file(args.config)
    cfg.MODEL.WEIGHTS = args.checkpoint
    model = build_model(cfg)
    DetectionCheckpointer(model).load(args.checkpoint)
    model.eval()

    tasks = tuple(t.strip() for t in args.tasks.split(',') if t.strip())
    evaluator = CocoEvaluator(args.dataset, cfg, False, args.out_dir)
    evaluator.reset()

    dataset_dicts = DatasetCatalog.get(args.dataset)
    if args.max_images:
        dataset_dicts = dataset_dicts[:args.max_images]

    n_vis = 0
    dump = [] if args.dump_prefix else None
    with torch.no_grad():
        for d in dataset_dicts:
            h, w = d['height'], d['width']
            img = cv2.imread(d['file_name'], cv2.IMREAD_COLOR)
            outputs = model([{'image': torch.from_numpy(
                img.astype('float32').transpose(2, 0, 1)),
                'height': h, 'width': w}])
            inst = outputs[0]['instances'].to('cpu')
            evaluator.process([d], [{'instances': inst}])
            if dump is not None and inst.has('scores'):
                import numpy as _np
                from pycocotools import mask as _mm
                boxes = inst.pred_boxes.tensor.numpy() if inst.has('pred_boxes') else []
                for k in range(len(inst)):
                    cat = int(inst.pred_classes[k]) + 1 if inst.has('pred_classes') else 1
                    rec = {'image_id': d['image_id'], 'category_id': cat,
                           'score': float(inst.scores[k]),
                           'bbox': [float(v) for v in boxes[k][:4]] if len(boxes) else [0,0,0,0]}
                    # xyxy -> xywh
                    x1, y1, x2, y2 = rec['bbox']
                    rec['bbox'] = [x1, y1, x2 - x1, y2 - y1]
                    if inst.has('pred_masks'):
                        m = _np.asfortranarray(inst.pred_masks[k].numpy().astype(_np.uint8))
                        rec['segmentation'] = _mm.encode(m)
                    dump.append(rec)
            if vis_dir is not None and (args.vis_limit <= 0
                                        or n_vis < args.vis_limit):
                visualize(d['file_name'], inst, vis_dir, args.score_thr)
                n_vis += 1

    if dump is not None:
        class _NumpyEncoder(json.JSONEncoder):
            def default(self, o):
                import numpy as _np
                if isinstance(o, (_np.ndarray, _np.generic)):
                    return o.item() if o.ndim == 0 else o.tolist()
                if isinstance(o, bytes):
                    return o.decode('utf-8', 'replace')
                return super().default(o)
        out_json = args.dump_prefix + '.segm.json'
        with open(out_json, 'w') as f:
            json.dump(dump, f, cls=_NumpyEncoder)
        print(f'dumped {len(dump)} preds -> {out_json}')

    results = evaluator.evaluate()
    print('\n===== WHU test metrics =====')
    flat = {}
    for task, metrics in results.items():
        print(f'--- {task} ---')
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                print(f'{k}: {v:.4f}')
                flat[f'{task}/{k}'] = float(v)
    out_json = os.path.join(args.out_dir, 'test_metrics.json')
    with open(out_json, 'w') as f:
        json.dump(flat, f, indent=2)
    print(f'saved -> {out_json} | images={len(dataset_dicts)} vis={n_vis}')


if __name__ == '__main__':
    main()
