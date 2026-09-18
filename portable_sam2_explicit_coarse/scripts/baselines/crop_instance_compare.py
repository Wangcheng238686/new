#!/usr/bin/env python
"""实例级跨模型对比图: 每个 GT 实例一行 [GT | ours | 模型2 | ...] 同框 crop 拼接.

实例挑选: Ours 与对手(rival)的逐实例匹配 IoU 差值降序(ours 好/对手差 优先),
对手漏检(IoU=0)排在最前; 同一图最多 --per-image 个实例防止单图霸榜.
输出:
  <out>/grid.png                 所有入选实例的拼接大图(每行一个实例)
  <out>/crops/<img>_<idx>/       每实例单独 crop(各模型), 供自行重拼

用法:
  python crop_instance_compare.py --gt test.json \
    --pred ours=.../predictions.json --pred yolo=.../pred_coco.json \
    --rival yolo [--limit 12] [--per-image 2] [--margin 0.25] [--min-size 72]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from pick_viz_images import load_gt, norm_records

MODEL_COLORS = {  # BGR 与 viz_mono_compare.sh 一致
    'gt': (0, 0, 170), 'ours': (170, 0, 0), 'yolo': (0, 80, 170),
    'catnet': (0, 130, 0), 'maskrcnn': (120, 0, 120), 'htc': (40, 70, 160),
    'msrcnn': (150, 70, 0), 'condinst': (0, 110, 150), 'solov2': (100, 100, 0),
}
ALPHA = 0.55
GAP = 4


def match_instances(dt_list, gt_rles, iou_thr=0.5):
    """返回 gt_idx -> (pred_rle, iou); 贪心按 score 降序."""
    matched = {}
    used = set()
    for _, drle in dt_list or []:
        best, bj = 0.0, -1
        for j, grle in enumerate(gt_rles):
            if j in used:
                continue
            iou = mask_utils.iou([drle], [grle], [0])[0][0]
            if iou > best:
                best, bj = iou, j
        if best >= iou_thr and bj >= 0:
            matched[bj] = (drle, best)
            used.add(bj)
        if len(used) == len(gt_rles):
            break
    return matched


def crop_with_mask(img, rle, color, bbox, H, W, alpha=ALPHA):
    x1, y1, x2, y2 = bbox
    tile = img[y1:y2, x1:x2].copy()
    if rle is not None:
        m = mask_utils.decode(rle)[y1:y2, x1:x2]
        ov = tile.copy()
        ov[m > 0] = color
        tile = cv2.addWeighted(ov, alpha, tile, 1 - alpha, 0)
    return tile


def expand_box(bbox, H, W, margin, min_size):
    x, y, w, h = bbox
    cx, cy = x + w / 2, y + h / 2
    s = max(w, h, min_size) * (1 + 2 * margin)
    x1 = int(max(0, min(W - 1, cx - s / 2)))
    x2 = int(max(x1 + 1, min(W, cx + s / 2)))
    y1 = int(max(0, min(H - 1, cy - s / 2)))
    y2 = int(max(y1 + 1, min(H, cy + s / 2)))
    return x1, y1, x2, y2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', required=True)
    ap.add_argument('--pred', action='append', required=True, help='name=path')
    ap.add_argument('--rival', default=None)
    ap.add_argument('--images-dir', required=True)
    ap.add_argument('--limit', type=int, default=12)
    ap.add_argument('--per-image', type=int, default=2)
    ap.add_argument('--score-thr', type=float, default=0.5)
    ap.add_argument('--iou-thr', type=float, default=0.5)
    ap.add_argument('--margin', type=float, default=0.25)
    ap.add_argument('--min-size', type=int, default=72)
    ap.add_argument('--out', default='viz_instance_compare')
    args = ap.parse_args()

    imgs, gt_rles_all = load_gt(args.gt)
    stem2id = {Path(im['file_name']).stem: iid for iid, im in imgs.items()}
    models = {}
    for spec in args.pred:
        name, path = spec.split('=', 1)
        recs = norm_records(path, imgs, stem2id)
        models[name] = {iid: [r for r in lst if r[0] >= args.score_thr]
                        for iid, lst in recs.items()}

    names = list(models)
    rival = args.rival or (names[1] if len(names) > 1 else None)
    ours = names[0]

    # 逐实例: (advantage, iid, gt_idx)
    cands = []
    per_img_cnt = {}
    for iid, gt_rles in gt_rles_all.items():
        if not gt_rles:
            continue
        m_ours = match_instances(models.get(ours, {}).get(iid), gt_rles, args.iou_thr)
        m_riv = match_instances(models.get(rival, {}).get(iid), gt_rles, args.iou_thr) \
            if rival else {}
        for j in range(len(gt_rles)):
            o_iou = m_ours[j][1] if j in m_ours else 0.0
            r_iou = m_riv[j][1] if j in m_riv else 0.0
            adv = (o_iou - r_iou, o_iou)  # 对手漏检/差 + ours 好 优先
            if adv[0] > 0.05 and o_iou > 0.6:
                cands.append((adv, iid, j))
    cands.sort(key=lambda x: (-x[0][0], -x[0][1]))
    picked = []
    for adv, iid, j in cands:
        if per_img_cnt.get(iid, 0) >= args.per_image:
            continue
        per_img_cnt[iid] = per_img_cnt.get(iid, 0) + 1
        picked.append((adv, iid, j))
        if len(picked) >= args.limit:
            break
    print(f'候选 {len(cands)} 实例, 选中 {len(picked)}')

    out = Path(args.out)
    (out / 'crops').mkdir(parents=True, exist_ok=True)
    img_cache = {}
    rows = []
    for k, (adv, iid, j) in enumerate(picked):
        im = imgs[iid]
        H, W = im['height'], im['width']
        stem = Path(im['file_name']).stem
        if stem not in img_cache:
            img_cache[stem] = cv2.imread(str(Path(args.images_dir) / im['file_name']))
        img = img_cache[stem]
        if img is None:
            continue
        gt_rles = gt_rles_all[iid]
        # GT bbox(该实例)
        gm = mask_utils.decode(gt_rles[j])
        ys, xs = np.where(gm > 0)
        bbox = expand_box((xs.min(), ys.min(), xs.max() - xs.min() + 1,
                           ys.max() - ys.min() + 1), H, W, args.margin, args.min_size)
        tiles = [('gt', crop_with_mask(img, gt_rles[j], MODEL_COLORS['gt'], bbox, H, W))]
        for name in names:
            m = match_instances(models.get(name, {}).get(iid), gt_rles, args.iou_thr)
            rle = m[j][0] if j in m else None
            tiles.append((name, crop_with_mask(img, rle, MODEL_COLORS.get(name, (80, 80, 80)),
                                               bbox, H, W)))
        th = max(t.shape[0] for _, t in tiles)
        tw = max(t.shape[1] for _, t in tiles)
        row = []
        for name, t in tiles:
            canvas = np.zeros((th, tw, 3), np.uint8)
            canvas[:t.shape[0], :t.shape[1]] = t
            row.append(canvas)
        rows.append((stem, j, row))
        d = out / 'crops' / f'{stem}_inst{j}'
        d.mkdir(exist_ok=True)
        for (name, _), canvas in zip(tiles, row):
            cv2.imwrite(str(d / f'{name}.png'), canvas)

    if rows:
        ch = max(r[2][0].shape[0] for r in rows)
        cw = max(r[2][0].shape[1] for r in rows)
        ncol = len(rows[0][2])
        grid = np.full((len(rows) * (ch + GAP) + GAP, ncol * (cw + GAP) + GAP, 3),
                       255, np.uint8)
        for ri, (stem, j, row) in enumerate(rows):
            for ci, canvas in enumerate(row):
                y = GAP + ri * (ch + GAP) + (ch - canvas.shape[0]) // 2
                x = GAP + ci * (cw + GAP) + (cw - canvas.shape[1]) // 2
                grid[y:y + canvas.shape[0], x:x + canvas.shape[1]] = canvas
        cv2.imwrite(str(out / 'grid.png'), grid)
        print(f'grid -> {out}/grid.png  ({len(rows)} 实例 x {ncol} 列, cell {cw}x{ch})')
        for (stem, j, row), (adv, iid, jj) in zip(rows, picked):
            print(f'  {stem}_inst{j}: ours-rival IoU差={adv[0]:.3f} ours_IoU={adv[1]:.3f}')


if __name__ == '__main__':
    main()
