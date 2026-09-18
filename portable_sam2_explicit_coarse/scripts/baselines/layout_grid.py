#!/usr/bin/env python
"""论文用网格拼接: 3 行实例 x (原图|GT|基线a..f|Ours) 列, 正方形 crop.

选取: Ours 逐实例匹配 IoU 最高的 top-N(最接近 GT), 每图最多 1 个实例.
每列模型单色(同 viz_mono_compare 色表), 漏检格为无叠加原图.
输出: <out>/grid.png + <out>/crops/<img>_inst<j>/{raw,gt,<model>}.png

用法:
  python layout_grid.py --gt test.json --images-dir .../test \
    --ours ours=path --baseline catnet=path --baseline yolo=path ... \
    [--rows 3] [--cell 256] [--margin 0.3] [--out grid_out]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from pick_viz_images import load_gt, norm_records
from crop_instance_compare import match_instances, MODEL_COLORS, ALPHA

GAP = 4


def square_box(mask, H, W, margin, min_size, fixed_side=None):
    ys, xs = np.where(mask > 0)
    x1, y1 = xs.min(), ys.min()
    x2, y2 = xs.max() + 1, ys.max() + 1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = fixed_side or max(x2 - x1, y2 - y1, min_size) * (1 + 2 * margin)
    side = max(side, max(x2 - x1, y2 - y1) * (1 + 2 * margin))
    half = side / 2
    sx1 = int(max(0, min(W - side, cx - half)))
    sy1 = int(max(0, min(H - side, cy - half)))
    side = int(min(side, W - sx1, H - sy1))
    return sx1, sy1, sx1 + int(side), sy1 + int(side)


def crop_tile(img, rles, color, box, boxes=None, cell=256, faint_rles=None, binary=False):
    """视野内实例 mask 叠加 + 框; faint_rles 以半透明度画(未匹配/分割不达标);
    binary=True 时输出黑底白掩码(仅匹配实例, 不画框/faint)."""
    x1, y1, x2, y2 = box
    if binary:
        tile = np.zeros((y2 - y1, x2 - x1, 3), np.uint8)
        for rle in (rles or []):
            m = mask_utils.decode(rle)[y1:y2, x1:x2]
            tile[m > 0] = 255
        return cv2.resize(tile, (cell, cell), interpolation=cv2.INTER_NEAREST)
    tile = img[y1:y2, x1:x2].copy()
    if faint_rles:
        ov = tile.copy()
        for rle in faint_rles:
            m = mask_utils.decode(rle)[y1:y2, x1:x2]
            ov[m > 0] = color
        tile = cv2.addWeighted(ov, ALPHA / 2, tile, 1 - ALPHA / 2, 0)
    if rles:
        ov = tile.copy()
        for rle in rles:
            m = mask_utils.decode(rle)[y1:y2, x1:x2]
            ov[m > 0] = color
        tile = cv2.addWeighted(ov, ALPHA, tile, 1 - ALPHA, 0)
    sx = cell / tile.shape[1]
    sy = cell / tile.shape[0]
    tile = cv2.resize(tile, (cell, cell),
                      interpolation=cv2.INTER_AREA if sx < 1 else cv2.INTER_LINEAR)
    if boxes:
        for bx, by, bw, bh in boxes:
            px1, py1 = int((bx - x1) * sx), int((by - y1) * sy)
            px2, py2 = int((bx + bw - x1) * sx), int((by + bh - y1) * sy)
            cv2.rectangle(tile, (px1, py1), (px2, py2), color, 1)
    return tile


def boxes_in_view(rles, box, cover=0.3):
    """主要落在 crop 视野内的实例: [(gt_idx, (x,y,w,h))]."""
    x1, y1, x2, y2 = box
    out = []
    for j, rle in enumerate(rles):
        bx, by, bw, bh = mask_utils.toBbox(rle)
        inter = max(0, min(bx + bw, x2) - max(bx, x1)) * \
            max(0, min(by + bh, y2) - max(by, y1))
        if inter > cover * bw * bh:
            out.append((j, (bx, by, bw, bh)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', required=True)
    ap.add_argument('--images-dir', required=True)
    ap.add_argument('--ours', required=True, help='name=path')
    ap.add_argument('--baseline', action='append', default=[],
                    help='name=path (可多次, 顺序即列序)')
    ap.add_argument('--rows', type=int, default=3)
    ap.add_argument('--cell', type=int, default=256)
    ap.add_argument('--min-instance', type=int, default=48,
                    help='实例最小边(px), 太小的 crop 不美观')
    ap.add_argument('--score-thr', type=float, default=0.5)
    ap.add_argument('--iou-thr', type=float, default=0.5)
    ap.add_argument('--rival', default=None,
                    help='最强基线名: 选取条件改为 ours_IoU>=--ours-min 且按与该基线的'
                         ' IoU 差距降序(保证 Ours 最优的同时对手明显失败)')
    ap.add_argument('--ours-min', type=float, default=0.9,
                    help='rival 模式下 Ours 的 IoU 下限')
    ap.add_argument('--stem-dist', type=int, default=50,
                    help='图号数值距离小于该值视为同区域航拍 tile, 跳过(多样性)')
    ap.add_argument('--density-window', type=int, default=220,
                    help='密集度统计的固定窗口边长(px, 原图尺度)')
    ap.add_argument('--binary', action='store_true',
                    help='GT/各模型列输出黑白二值掩码(实例白/背景黑, 仅匹配成功实例, 无框)')
    ap.add_argument('--pick-stem', default=None,
                    help='手动指定选样图(stem), 行内在该图视野最优实例')
    ap.add_argument('--hard-bucket', action='store_true',
                    help='只在大目标(side>=128)且异形(fill<0.68 或 solidity<0.82)实例中选, '
                         '条件: Ours IoU>=ours-min 且最强基线落后>=hard-gap')
    ap.add_argument('--hard-gap', type=float, default=0.25)
    ap.add_argument('--view-ours-min', type=float, default=0.9,
                    help='rival 模式下视野内 Ours 的 GT 匹配率下限')
    ap.add_argument('--min-neighbors', type=int, default=4,
                    help='crop 视野内 GT 实例数下限(密集/复杂场景筛选)')
    ap.add_argument('--margin', type=float, default=0.3)
    ap.add_argument('--out', default='viz_grid')
    args = ap.parse_args()

    imgs, gt_rles_all = load_gt(args.gt)
    stem2id = {Path(im['file_name']).stem: iid for iid, im in imgs.items()}

    def load(spec):
        name, path = spec.split('=', 1)
        recs = norm_records(path, imgs, stem2id)
        return name, {iid: [r for r in lst if r[0] >= args.score_thr]
                      for iid, lst in recs.items()}

    ours_name, ours_preds = load(args.ours)
    baselines = [load(b) for b in args.baseline]

    # 候选: Ours IoU 高 + 实例够大 + 每图 1 个; rival 模式下按差距排序
    rival_preds = None
    if args.rival:
        rival_preds = dict(baselines)[args.rival]
    im_h_w = {iid: (im['height'], im['width']) for iid, im in imgs.items()}
    _ms_riv_cache = {}
    if args.hard_bucket:
        _all = {ours_name: ours_preds}
        _all.update(dict(baselines))
        for iid, rles in gt_rles_all.items():
            if rles:
                _ms_riv_cache[iid] = {n: match_instances(m.get(iid), rles, args.iou_thr)
                                      for n, m in _all.items() if n != ours_name}
    cands = []
    for iid, gt_rles in gt_rles_all.items():
        if not gt_rles:
            continue
        m = match_instances(ours_preds.get(iid), gt_rles, args.iou_thr)
        m_riv = match_instances(rival_preds.get(iid), gt_rles, args.iou_thr) \
            if rival_preds is not None else {}
        for j, (rle, iou) in m.items():
            if rival_preds is not None and iou < args.ours_min:
                continue
            gm = mask_utils.decode(gt_rles[j])
            ys, xs = np.where(gm > 0)
            if max(xs.max() - xs.min(), ys.max() - ys.min()) + 1 < args.min_instance:
                continue
            # 密集度 + 视野级表现: 以实例为中心的固定窗口(渲染视野即此窗口)
            H0, W0 = im_h_w[iid]
            half = args.density_window / 2
            cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
            wx1 = int(max(0, min(W0 - args.density_window, cx - half)))
            wy1 = int(max(0, min(H0 - args.density_window, cy - half)))
            wside = int(min(args.density_window, W0 - wx1, H0 - wy1))
            view = boxes_in_view(gt_rles, (wx1, wy1, wx1 + wside, wy1 + wside))
            density = len(view)
            if density < args.min_neighbors:
                continue
            view_idx0 = [jx for jx, _ in view]
            if args.hard_bucket:
                bx0, by0, bw0, bh0 = mask_utils.toBbox(rle)
                side0 = max(bw0, bh0)
                fillr = int(gm.sum()) / max(1, bw0 * bh0)
                cn, _ = cv2.findContours(gm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                sol = 1.0
                if cn:
                    c = max(cn, key=cv2.contourArea)
                    ha = cv2.contourArea(cv2.convexHull(c))
                    sol = cv2.contourArea(c) / ha if ha > 0 else 1.0
                if side0 < 128 or (fillr >= 0.68 and sol >= 0.82):
                    continue
                ms_r = _ms_riv_cache.get(iid, {})
                others = [ms_r[n][j][1] if j in ms_r[n] else 0.0 for n in ms_r]
                if not others or max(others) > iou - args.hard_gap:
                    continue
                cands.append((iou - max(others), iou, iid, j, density))
                continue
            if rival_preds is not None:
                # 视野内 Ours 匹配率须达标(整片好), key=对手视野漏检数
                ours_hit = sum(1 for jx in view_idx0 if jx in m)
                if ours_hit / density < args.view_ours_min:
                    continue
                riv_hit = sum(1 for jx in view_idx0 if jx in m_riv)
                key = density - riv_hit
            else:
                key = iou
            cands.append((key, iou, iid, j, density))
    cands.sort(reverse=True)
    picked, used_img, used_nums = [], set(), []
    if args.pick_stem:
        stem, _, inst = args.pick_stem.partition(':')
        target = stem2id.get(stem)
        if inst:
            want = int(inst)
            cands = [c for c in cands if c[2] == target and c[3] == want] or \
                [(iou, iou, target, want, 1)]
        else:
            cands = [c for c in cands if c[2] == target] or \
                [(iou, iou, target, 0, 1)]
    for key, iou, iid, j, density in cands:
        if iid in used_img:
            continue
        # 同区域航拍 tile 去重: 图号数值太近跳过
        try:
            num = int(Path(imgs[iid]['file_name']).stem)
        except ValueError:
            num = iid
        if any(abs(num - u) < args.stem_dist for u in used_nums):
            continue
        used_img.add(iid)
        used_nums.append(num)
        picked.append((key, iou, iid, j, density))
        if len(picked) >= args.rows:
            break
    print(f'候选 {len(cands)}, 选中 {len(picked)}: '
          + ', '.join(f'{Path(imgs[iid]["file_name"]).stem}_inst{j}'
                      f'(IoU={iou:.3f},gap={key:.3f},n={d})'
                      for key, iou, iid, j, d in picked))

    out = Path(args.out)
    (out / 'crops').mkdir(parents=True, exist_ok=True)
    cols = ['raw', 'gt'] + [n for n, _ in baselines] + [ours_name]
    rows_tiles = []
    for key, iou, iid, j, density in picked:
        im = imgs[iid]
        H, W = im['height'], im['width']
        stem = Path(im['file_name']).stem
        img = cv2.imread(str(Path(args.images_dir) / im['file_name']))
        if img is None:
            continue
        gt_rles = gt_rles_all[iid]
        gm = mask_utils.decode(gt_rles[j])
        box = square_box(gm, H, W, args.margin, args.min_instance,
                         fixed_side=args.density_window)

        in_view = boxes_in_view(gt_rles, box)
        view_idx = [jx for jx, _ in in_view]

        def model_tile(name, preds):
            m = match_instances(preds.get(iid), gt_rles, args.iou_thr)
            rles = [m[jx][0] for jx in view_idx if jx in m]
            used = {id(m[jx][0]) for jx in view_idx if jx in m}
            faint = [] if args.binary else \
                [r for s2, r in preds.get(iid, []) if id(r) not in used]
            color = MODEL_COLORS.get(name, (80, 80, 80))
            pboxes = [] if args.binary else \
                [tuple(float(v) for v in mask_utils.toBbox(r))
                 for _, r in preds.get(iid, [])]
            return crop_tile(img, rles, color, box, boxes=pboxes, cell=args.cell,
                             faint_rles=faint, binary=args.binary)

        raw = img[box[1]:box[3], box[0]:box[2]].copy()
        tiles = [cv2.resize(raw, (args.cell, args.cell))]
        tiles.append(crop_tile(img, [gt_rles[jx] for jx in view_idx],
                               MODEL_COLORS['gt'], box,
                               boxes=[] if args.binary else [b for _, b in in_view],
                               cell=args.cell, binary=args.binary))
        for name, preds in baselines:
            tiles.append(model_tile(name, preds))
        tiles.append(model_tile(ours_name, ours_preds))
        rows_tiles.append((stem, j, iou, tiles))
        d = out / 'crops' / f'{stem}_inst{j}'
        d.mkdir(exist_ok=True)
        for cname, t in zip(cols, tiles):
            cv2.imwrite(str(d / f'{cname}.png'), t)

    # 列标签: image | GT | (a)..(f) 基线序 | ours
    labels = ['image', 'GT'] + [f'({chr(ord("a") + k)})' for k in range(len(baselines))] + ['ours']
    label_h = int(args.cell * 0.22)
    n = args.cell + GAP
    grid = np.full((GAP + len(rows_tiles) * n + label_h, GAP + len(cols) * n, 3),
                   255, np.uint8)
    for ri, (_, _, _, tiles) in enumerate(rows_tiles):
        for ci, t in enumerate(tiles):
            y = GAP + ri * n
            x = GAP + ci * n
            grid[y:y + args.cell, x:x + args.cell] = t
    for ci, text in enumerate(labels):
        cx = GAP + ci * n + args.cell // 2
        cy = GAP + len(rows_tiles) * n + label_h // 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = args.cell / 300.0
        thick = 2 if args.cell < 320 else 3
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        cv2.putText(grid, text, (cx - tw // 2, cy + th // 2), font, scale,
                    (0, 0, 0), thick, cv2.LINE_AA)
    cv2.imwrite(str(out / 'grid.png'), grid)
    print(f'grid -> {out}/grid.png  ({len(rows_tiles)} 行 x {len(cols)} 列, cell {args.cell})')
    print('列序: ' + ' | '.join(cols))


if __name__ == '__main__':
    main()
