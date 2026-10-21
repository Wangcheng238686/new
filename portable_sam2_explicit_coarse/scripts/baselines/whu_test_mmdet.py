#!/usr/bin/env python
"""WHU-1024 测试集推理与可视化 (mmdet/mmengine 系: maskrcnn/msrcnn/htc/scnet/mask2former/rs4d/catnet)

输出: bbox 与 segm 的 mAP / AP50 / AP75 / APs / APm / APl (CocoMetric)
可视化(可选): 仅将预测 mask 叠加到原图, 不画框不写置信度, 每实例独立颜色
  (调色板与 alpha=0.35 对齐本项目 scripts/visualize_instances.py 风格)

用法:
  python whu_test_mmdet.py CONFIG CKPT [--out-dir DIR] [--vis] [--vis-dir DIR]
      [--vis-limit N] [--score-thr 0.5] [--max-images N]
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

# 16 visually distinct colors (BGR) —— 与本项目 visualize_instances.py 保持一致
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



def visualize_sample(sample, vis_dir, score_thr):
    """仅叠加彩色 mask: 每实例一个颜色, 无框/无文字/无置信度."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('checkpoint')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--vis', action='store_true')
    ap.add_argument('--vis-dir', default=None)
    ap.add_argument('--vis-limit', type=int, default=0, help='0=全部')
    ap.add_argument('--score-thr', type=float, default=0.5)
    ap.add_argument('--max-images', type=int, default=0, help='0=全部(调试用)')
    ap.add_argument('--dump-prefix', default=None,
                    help='设置时额外以 COCO 格式落盘全部预测(供选图脚本用), '
                         '如 /xxx/pred -> pred.segm.json')
    ap.add_argument('--use-format-only', action='store_true',
                    help='用 mmengine CocoMetric format_only 落盘(默认改为手转, '
                         'format_only 对 Mask2Former 等会产生损坏 RLE)')
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

    from mmdet.registry import DATASETS, METRICS, MODELS
    from mmdet.utils import register_all_modules
    register_all_modules()

    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = model.cuda().eval()

    ds_cfg = cfg.test_dataloader
    dataset = DATASETS.build(ds_cfg.dataset)
    loader = DataLoader(
    dataset,
    batch_size=1,  # 可视化/低显存场景逐张推理(与 GPU 上其他训练共存)
    num_workers=ds_cfg.num_workers,
    collate_fn=pseudo_collate,
    shuffle=False)

    # 经 mmengine Evaluator 做 BaseDataSample->dict 转换后再喂 metric
    # (直接调 metric.process 会因 DetDataSample 不可下标而报错)
    from mmengine.evaluator import Evaluator
    metrics = [cfg.test_evaluator]
    if args.dump_prefix and args.use_format_only:
        fmt = dict(type='CocoMetric', metric=['bbox', 'segm'], format_only=True,
                   outfile_prefix=args.dump_prefix,
                   ann_file=cfg.test_evaluator.get('ann_file'))
        metrics.append(fmt)
    evaluator = Evaluator(metrics=metrics)
    evaluator.dataset_meta = dataset.metainfo

    n_img, n_vis = 0, 0
    manual_dump = [] if args.dump_prefix and not args.use_format_only else None
    with torch.no_grad():
        for data in loader:
            data_samples = model.test_step(data)
            evaluator.process(data_samples=data_samples)
            if manual_dump is not None:
                from mmdet.structures.mask import encode_mask_results
                for smp in data_samples:
                    pi = smp.pred_instances.cpu()
                    masks = encode_mask_results(pi.masks)
                    for k in range(len(pi.scores)):
                        b = pi.bboxes[k].cpu().tolist()
                        rec = {'image_id': int(smp.img_id),
                               'category_id': 1,
                               'score': float(pi.scores[k]),
                               'bbox': [b[0], b[1], b[2]-b[0], b[3]-b[1]],
                               'segmentation': masks[k]}
                        manual_dump.append(rec)
            if args.vis:
                for s in data_samples:
                    if args.vis_limit <= 0 or n_vis < args.vis_limit:
                        visualize_sample(s, vis_dir, args.score_thr)
                        n_vis += 1
            n_img += len(data_samples)
            if args.max_images and n_img >= args.max_images:
                print(f'[debug] max-images reached: {n_img}')
                break

    results = evaluator.evaluate(len(dataset))
    if manual_dump is not None:
        import numpy as _np

        class _Enc(json.JSONEncoder):
            def default(self, o):
                if isinstance(o, (_np.ndarray, _np.generic)):
                    return o.item() if o.ndim == 0 else o.tolist()
                if isinstance(o, bytes):
                    return o.decode('utf-8', 'replace')
                return super().default(o)
        out_json = args.dump_prefix + '.segm.json'
        with open(out_json, 'w') as f:
            json.dump(manual_dump, f, cls=_Enc)
        print(f'manual dumped {len(manual_dump)} preds -> {out_json}')
    print('\n===== WHU test metrics =====')
    for k in sorted(results):
        print(f'{k}: {results[k]:.4f}')
    out_json = os.path.join(args.out_dir, 'test_metrics.json')
    with open(out_json, 'w') as f:
        json.dump({k: float(v) for k, v in results.items()}, f, indent=2)
    print(f'saved -> {out_json} | images={n_img} vis={n_vis}')


if __name__ == '__main__':
    main()
