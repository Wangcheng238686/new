#!/usr/bin/env python
"""CitySample-Sat test 推理与评估 (mmdet/mmengine 系通用: rsprompter/maskrcnn/msrcnn/htc/mask2former/rtmdet/rs4d/catnet)

与 whu_test_mmdet.py 的差异:
  - GT 指向 CitySample 卫星侧 annotations/test.json (459 图)
  - 评估不用 mmengine CocoMetric(其 maxDets 硬编码 [1,10,100]), 改为
    预测手转 COCO json 后直接 pycocotools, maxDets=[1,10,250] (P1 协议:
    test 单图实例最大 238)
  - 可视化沿用同风格(仅彩色 mask 叠加, 每实例独立颜色)

用法:
  python citysample_sat_test.py CONFIG CKPT --out-dir DIR [--gt-json PATH]
      [--vis] [--vis-dir DIR] [--vis-limit N] [--score-thr 0.5]
      [--max-images N] [--dump-prefix PATH]
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmengine.utils import import_modules_from_strings
from torch.utils.data import DataLoader

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

DS_ROOT = '/data1/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021'
MAX_DET_EVAL = 250

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


def visualize_sample(sample, vis_dir, score_thr):
    img = cv2.imread(sample.img_path, cv2.IMREAD_COLOR)
    pred = sample.pred_instances
    keep = np.asarray(pred.scores.detach().cpu()) > score_thr
    m = pred.masks
    t = m.masks if hasattr(m, 'masks') else m
    if hasattr(t, 'detach'):
        t = t.detach().cpu().numpy()
    masks = np.asarray(t)[keep]
    overlay = img.copy()
    for i, m in enumerate(masks):
        if m.max() == 0:
            continue
        color = _viz_color(i)
        overlay[m > 0] = color
    out = cv2.addWeighted(overlay, VIZ_ALPHA, img, 1.0 - VIZ_ALPHA, 0)
    stem = os.path.splitext(os.path.basename(sample.img_path))[0]
    cv2.imwrite(os.path.join(vis_dir, stem + '.png'), out)


class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.ndarray, np.generic)):
            return o.item() if o.ndim == 0 else o.tolist()
        if isinstance(o, bytes):
            return o.decode('utf-8', 'replace')
        return super().default(o)


def evaluate_maxdets(preds, gt_json):
    """pycocotools 双任务评估, maxDets=[1,10,250]."""
    coco_gt = COCO(gt_json)
    cat_id = coco_gt.dataset['categories'][0]['id']
    for p in preds:
        p['category_id'] = cat_id
    metrics = {}
    # mmdet encode_mask_results 的 RLE counts 为 bytes, dumps 需用 _Enc
    # (否则默认编码器抛 bytes not serializable)
    coco_dt = coco_gt.loadRes(json.loads(json.dumps(preds, cls=_Enc)))
    for task in ('bbox', 'segm'):
        E = COCOeval(coco_gt, coco_dt, task)
        E.params.maxDets = [1, 10, MAX_DET_EVAL]
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
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('checkpoint')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--gt-json', default=f'{DS_ROOT}/annotations/test.json')
    ap.add_argument('--vis', action='store_true')
    ap.add_argument('--vis-dir', default=None)
    ap.add_argument('--vis-limit', type=int, default=0, help='0=全部')
    ap.add_argument('--score-thr', type=float, default=0.5)
    ap.add_argument('--max-images', type=int, default=0, help='0=全部(调试用)')
    ap.add_argument('--dump-prefix', default=None,
                    help='设置时落盘 COCO 预测 json, 如 /xxx/pred -> pred.segm.json')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.vis:
        vis_dir = args.vis_dir or os.path.join(args.out_dir, 'vis')
        os.makedirs(vis_dir, exist_ok=True)

    cfg = Config.fromfile(args.config)
    # torch>=2.6 默认 weights_only=True 会拒绝含 mmengine meta 的自训 ckpt
    import torch.serialization  # noqa: F401
    _orig_torch_load = torch.load
    torch.load = lambda *a, **k: _orig_torch_load(
        *a, **{**{'weights_only': False}, **k})
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)
    init_default_scope(cfg.get('default_scope', 'mmdet'))

    from mmdet.registry import DATASETS, MODELS
    from mmdet.utils import register_all_modules
    register_all_modules()

    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = model.cuda().eval()

    ds_cfg = cfg.test_dataloader
    dataset = DATASETS.build(ds_cfg.dataset)
    loader = DataLoader(
        dataset,
        batch_size=1,  # 逐张推理(与 GPU 上其他训练共存)
        num_workers=ds_cfg.num_workers,
        collate_fn=pseudo_collate,
        shuffle=False)

    n_img, n_vis = 0, 0
    preds = []
    with torch.no_grad():
        from mmdet.structures.mask import encode_mask_results
        for data in loader:
            data_samples = model.test_step(data)
            for smp in data_samples:
                pi = smp.pred_instances.cpu()
                masks = encode_mask_results(pi.masks)
                for k in range(len(pi.scores)):
                    b = pi.bboxes[k].cpu().tolist()
                    preds.append({
                        'image_id': int(smp.img_id),
                        'score': float(pi.scores[k]),
                        'bbox': [b[0], b[1], b[2] - b[0], b[3] - b[1]],
                        'segmentation': masks[k]})
            if args.vis:
                for s in data_samples:
                    if args.vis_limit <= 0 or n_vis < args.vis_limit:
                        visualize_sample(s, vis_dir, args.score_thr)
                        n_vis += 1
            n_img += len(data_samples)
            if args.max_images and n_img >= args.max_images:
                print(f'[debug] max-images reached: {n_img}')
                break

    if args.dump_prefix:
        out_json = args.dump_prefix + '.segm.json'
        with open(out_json, 'w') as f:
            json.dump(preds, f, cls=_Enc)
        print(f'dumped {len(preds)} preds -> {out_json}')

    print(f'\n===== CitySample-Sat test metrics (maxDets={MAX_DET_EVAL}) =====')
    metrics = evaluate_maxdets(preds, args.gt_json)
    for k in sorted(metrics):
        print(f'{k}: {metrics[k]:.4f}')
    out_json = os.path.join(args.out_dir, 'test_metrics.json')
    with open(out_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'saved -> {out_json} | images={n_img} vis={n_vis}')


if __name__ == '__main__':
    main()
