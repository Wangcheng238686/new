#!/usr/bin/env python
"""图1 (motivation) 组件化渲染: 全部小组件统一正方形, 分别交付 + 组合两行版.

上行(现有范式): 影像 -> 隐式实例表征 -> 隐式解码掩码(CATNet 真实预测)
下行(Ours):     影像 -> 表征 -> 候选区域局部粗形状(真实 CSPM) ->
                三种提示(点/框/稠密掩码, 真实挖掘) -> 最终掩码(Ours, 映射回原图)

交付:
  --components-dir 下 8 个正方形组件 png
  --out 组合两行大图
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mm

sys.path.insert(0, str(Path(__file__).parent))
from pick_viz_images import load_gt, norm_records
from crop_instance_compare import match_instances

GT_JSON = '/data1/wangcheng/dataset/WHU/2.4 annotation/annotation/test.json'
IMG_DIR = '/data1/wangcheng/dataset/WHU/2.2 test/test'
OURS = '/home/wangcheng/project/new/viz_data/ours_test_predictions.json'
IMPLICIT = ('/data1/wangcheng/checkpoint/whu1024_baselines/catnet/'
            'cat_mask_rcnn_r50_3x_whu1024_bs4ai2/test_out/pred_full.segm.json')
COARSE = '/home/wangcheng/project/new/viz_data/coarse_teaser/coarse_dump.json'
SAM2_REPO = '/home/wangcheng/project/new/sam2'
SAM2_CKPT = '/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt'

BLUE = (214, 155, 91)     # 黑底掩码蓝 (BGR)
INST_BLUE = (255, 80, 0)  # 实例 A 叠加 (BGR 蓝)
INST_RED = (0, 0, 230)    # 实例 B 叠加 (BGR 红)
FG_PT = (60, 200, 60)
DARK = (60, 60, 60)
BORDER = (190, 190, 190)


def square_crop_region(ba, bb, margin, size=1024):
    """以两实例 union 中心取正方形裁块."""
    xa, ya, wa, ha = ba; xb, yb, wb, hb = bb
    cx = (min(xa, xb) + max(xa + wa, xb + wb)) / 2
    cy = (min(ya, yb) + max(ya + ha, yb + hb)) / 2
    side = int(max(max(wa, wb), max(ha, hb)) * (1 + 2 * margin))
    half = side / 2
    x1 = int(max(0, min(size - side, max(0, cx - half))))
    y1 = int(max(0, min(size - side, max(0, cy - half))))
    return x1, y1, x1 + side, y1 + side


def binary_tile(mask):
    """掩码 -> 黑底蓝掩码正方形块."""
    s = max(mask.shape)
    sq = np.zeros((s, s), np.uint8)
    sq[:mask.shape[0], :mask.shape[1]] = mask
    tile = np.zeros((s, s, 3), np.uint8)
    tile[sq > 0] = BLUE
    return tile


def image_tile(img):
    return img


def repr_card(embed, size, smooth=1.5):
    """SAM2 图像嵌入 -> PCA 主成分 -> Turbo 伪彩单层正方形表征卡片."""
    C, HH, WW = embed.shape
    X = embed.reshape(C, -1).T.astype(np.float32)
    mean, eig = cv2.PCACompute(X, mean=None, maxComponents=1)
    ch = ((X - mean) @ eig.T).reshape(HH, WW)
    lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)
    ch = np.clip((ch - lo) / max(1e-6, hi - lo), 0, 1)
    big = cv2.resize((ch * 255).astype(np.uint8), (size, size),
                     interpolation=cv2.INTER_CUBIC)
    big = cv2.GaussianBlur(big, (0, 0), smooth)
    card = cv2.applyColorMap(big, cv2.COLORMAP_TURBO)
    return cv2.copyMakeBorder(card, 2, 2, 2, 2, cv2.BORDER_CONSTANT,
                              value=(250, 250, 250))


def overlay_tile(base, matches, rles, x1, y1, ia, ib):
    """实例掩码半透明映射回底图(红/蓝)."""
    p = base.copy()
    for gi, col in ((ia, INST_BLUE), (ib, INST_RED)):
        if gi in matches and matches[gi][0] is not None:
            m = mm.decode(matches[gi][0])[y1:y1 + p.shape[0],
                                          x1:x1 + p.shape[1]] > 0
            ov = p.copy(); ov[m] = col
            p = cv2.addWeighted(ov, 0.45, p, 0.55, 0)
    return p


def arrow(h=14):
    a = np.full((h, 30, 3), 255, np.uint8)
    cv2.arrowedLine(a, (2, h // 2), (27, h // 2), DARK, 2, cv2.LINE_AA,
                    tipLength=0.4)
    return a


def bordered(t):
    return cv2.copyMakeBorder(t, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=BORDER)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stem', default='300838')
    ap.add_argument('--ia', type=int, default=5)
    ap.add_argument('--ib', type=int, default=7)
    ap.add_argument('--margin', type=float, default=0.5)
    ap.add_argument('--export-size', type=int, default=512,
                    help='组件导出分辨率(正方形)')
    ap.add_argument('--tile-h', type=int, default=150,
                    help='组合大图中瓦片的布局高度')
    ap.add_argument('--components-dir',
                    default='/home/wangcheng/project/new/fig1_components')
    ap.add_argument('--out', default='/home/wangcheng/project/new/fig1_rows.png')
    ap.add_argument('--device', default='cuda:3')
    args = ap.parse_args()

    imgs, gt_all = load_gt(GT_JSON)
    iid = [i for i, im in imgs.items()
           if Path(im['file_name']).stem == args.stem][0]
    rles = gt_all[iid]
    ba = [float(v) for v in mm.toBbox(rles[args.ia])]
    bb = [float(v) for v in mm.toBbox(rles[args.ib])]
    x1, y1, x2, y2 = square_crop_region(ba, bb, args.margin)
    img = cv2.imread(str(Path(IMG_DIR) / f'{args.stem}.TIF'))
    crop = img[y1:y2, x1:x2]
    Hc, Wc = crop.shape[:2]

    stem = args.stem
    stem2id = {stem: iid}
    recs_ours = norm_records(OURS, imgs, stem2id)
    recs_md = norm_records(IMPLICIT, imgs, stem2id)
    preds_ours = {i: [r for r in l if r[0] >= 0.5] for i, l in recs_ours.items()}
    preds_md = {i: [r for r in l if r[0] >= 0.5] for i, l in recs_md.items()}
    mo_ours = match_instances(preds_ours.get(iid), rles, 0.5)
    mo_md = match_instances(preds_md.get(iid), rles, 0.5)
    dump = json.load(open(COARSE))[stem]

    coarse_full = np.zeros((1024, 1024), np.uint8)
    for k in (1, 3):   # inst5/inst7 对应的 proposal coarse
        if dump[k]['coarse_rle'] is not None:
            coarse_full |= (mm.decode(dump[k]['coarse_rle']) > 0).astype(np.uint8)
    coarse = coarse_full[y1:y2, x1:x2]
    pts = [q for k in (1, 3) for q in dump[k].get('points', [])
           if x1 <= q['x'] < x2 and y1 <= q['y'] < y2]

    # --- SAM2 图像嵌入(真实) ---
    sys.path.insert(0, SAM2_REPO)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model = build_sam2('configs/sam2/sam2_hiera_b+.yaml', SAM2_CKPT,
                       device=args.device)
    predictor = SAM2ImagePredictor(model)
    predictor.set_image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    embed = predictor._features['image_embed'][0].cpu().numpy()

    EX = args.export_size
    comp = {}
    comp['01_image'] = crop
    comp['02_repr'] = repr_card(embed, EX)[2:-2, 2:-2]
    comp['03_mask_implicit'] = overlay_tile(crop, mo_md, rles, x1, y1,
                                            args.ia, args.ib)
    # 粗形状: 逐候选区域局部坐标监督 —— 每条 proposal 一个局部 patch
    # (框内影像 + 该区域学到的粗形状叠加 + 橙色框边 = 局部坐标系)
    patches = []
    pad = 6
    for k in (1, 3):
        bx = dump[k]['box']
        px1 = int(max(0, bx[0] - pad)); py1 = int(max(0, bx[1] - pad))
        px2 = int(min(1024, bx[2] + pad)); py2 = int(min(1024, bx[3] + pad))
        patch = img[py1:py2, px1:px2].copy()
        cm = mm.decode(dump[k]['coarse_rle'])[py1:py2, px1:px2] > 0
        ov = patch.copy(); ov[cm] = BLUE
        patch = cv2.addWeighted(ov, 0.5, patch, 0.5, 0)
        cv2.rectangle(patch, (0, 0), (patch.shape[1] - 1, patch.shape[0] - 1),
                      (0, 165, 255), 4)
        patches.append(patch)
    half_w = EX // 2 - 9
    inner_h = EX - 18
    tile = np.full((EX, EX, 3), 255, np.uint8)
    for i, patch in enumerate(patches):
        sc = min(half_w / patch.shape[1], inner_h / patch.shape[0])
        pw_, ph_ = int(patch.shape[1] * sc), int(patch.shape[0] * sc)
        rs = cv2.resize(patch, (pw_, ph_), interpolation=cv2.INTER_LANCZOS4)
        xo = 9 + i * (half_w + 6) + (half_w - pw_) // 2
        yo = 9 + (inner_h - ph_) // 2
        tile[yo:yo + ph_, xo:xo + pw_] = rs
    comp['04_coarse_shape'] = tile
    # --- 三种提示(紧凑正方形裁块 + 加粗标记) ---
    xa, ya, wa, ha = ba; xb, yb, wb, hb = bb
    cx = (min(xa, xb) + max(xa + wa, xb + wb)) / 2
    cy = (min(ya, yb) + max(ya + ha, yb + hb)) / 2
    tside = int(max(max(wa, wb), max(ha, hb)) * 1.25)
    tx1 = int(max(0, min(1024 - tside, max(0, cx - tside / 2))))
    ty1 = int(max(0, min(1024 - tside, max(0, cy - tside / 2))))
    tight = img[ty1:ty1 + tside, tx1:tx1 + tside]
    pv = tight.copy()
    for q in pts:
        px, py = int(round(q['x'] - tx1)), int(round(q['y'] - ty1))
        if q['label'] == 1:
            cv2.circle(pv, (px, py), 10, FG_PT, -1, cv2.LINE_AA)
            cv2.circle(pv, (px, py), 10, (255, 255, 255), 3, cv2.LINE_AA)
        else:
            cv2.circle(pv, (px, py), 10, (0, 0, 230), -1, cv2.LINE_AA)
            cv2.circle(pv, (px, py), 10, (255, 255, 255), 3, cv2.LINE_AA)
    comp['05_prompt_points'] = pv
    bv = tight.copy()
    for gi in (args.ia, args.ib):
        if gi in mo_ours and mo_ours[gi][0] is not None:
            bx = mm.toBbox(mo_ours[gi][0])
            cv2.rectangle(bv, (int(bx[0]) - tx1, int(bx[1]) - ty1),
                          (int(bx[0] + bx[2]) - tx1, int(bx[1] + bx[3]) - ty1),
                          (0, 165, 255), 5)
    comp['06_prompt_box'] = bv
    c = coarse_full[ty1:ty1 + tside, tx1:tx1 + tside].astype(np.float32)
    c = cv2.GaussianBlur(c, (0, 0), 5)
    c = np.clip(c / max(1e-6, c.max()), 0, 1)
    comp['07_prompt_dense'] = cv2.applyColorMap(
        (c * 255).astype(np.uint8), cv2.COLORMAP_JET)
    comp['08_mask_ours'] = overlay_tile(crop, mo_ours, rles, x1, y1,
                                        args.ia, args.ib)

    # --- 统一正方形导出 ---
    out_dir = Path(args.components_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles = {}
    for name, t in comp.items():
        if t.shape[0] != t.shape[1]:
            s = max(t.shape)
            sq = np.full((s, s, 3), 255, np.uint8)
            sq[(s - t.shape[0]) // 2:(s - t.shape[0]) // 2 + t.shape[0],
               (s - t.shape[1]) // 2:(s - t.shape[1]) // 2 + t.shape[1]] = t
            t = sq
        t = cv2.resize(t, (EX, EX), interpolation=cv2.INTER_LANCZOS4)
        tiles[name] = bordered(t)
        cv2.imwrite(str(out_dir / f'{name}.png'), tiles[name])

    # --- 组合两行大图(瓦片全部正方形, 两行等宽) ---
    h = args.tile_h
    layout = {k: cv2.resize(t, (h + 2, h + 2), interpolation=cv2.INTER_AREA)
              for k, t in tiles.items()}
    bot = [layout['01_image'], layout['02_repr'], layout['04_coarse_shape'],
           layout['05_prompt_points'], layout['06_prompt_box'],
           layout['07_prompt_dense'], layout['08_mask_ours']]
    ARW_W = 34
    w_bot = len(bot) * (h + 2) + ARW_W * (len(bot) - 1)
    W = w_bot + 24
    gap_top = (W - 24 - 3 * (h + 2)) // 2
    top = [layout['01_image'], layout['02_repr'], layout['03_mask_implicit']]
    top_h = h + 2
    H = 14 + top_h + 14 + 30 + h + 2 + 14
    canvas = np.full((H, W, 3), 255, np.uint8)

    def paste_row(row_tiles, y0, bg, gap):
        total = sum(t.shape[1] for t in row_tiles) + gap * (len(row_tiles) - 1)
        x0 = (W - total) // 2
        row_h = max(t.shape[0] for t in row_tiles)
        cv2.rectangle(canvas, (x0 - 10, y0 - 6),
                      (x0 + total + 10, y0 + row_h + 6), bg, -1)
        for i, t in enumerate(row_tiles):
            ya_ = y0 + (row_h - t.shape[0]) // 2
            canvas[ya_:ya_ + t.shape[0], x0:x0 + t.shape[1]] = t
            x0 += t.shape[1]
            if i < len(row_tiles) - 1:
                a = arrow()
                ya2 = y0 + row_h // 2 - a.shape[0] // 2
                x0 += (gap - a.shape[1]) // 2
                canvas[ya2:ya2 + a.shape[0], x0:x0 + a.shape[1]] = a
                x0 += gap - (gap - a.shape[1]) // 2

    paste_row(top, 14, (243, 243, 243), gap_top)
    paste_row(bot, 14 + top_h + 14 + 30, (255, 246, 238), ARW_W)
    cv2.imwrite(args.out, canvas)
    print(f'saved {args.out} {canvas.shape[1]}x{canvas.shape[0]} | '
          f'components[{EX}px] -> {out_dir} ({len(tiles)} files) | pts={len(pts)}')


if __name__ == '__main__':
    main()
