#!/usr/bin/env python
"""论文首页 teaser (v3): 全部要素来自 Ours 管线真实中间产物, 契合论文叙事.

panel (a): proposal 框(橙, 我们的两级检测) + proposal 条件下的 coarse mask
  (品红半透明, shape-prior 低分辨率先验) + 挖掘出的前景/背景点(绿点/黑点)
  + GT 蓝红细轮廓; 局部放大框指向两实例接触带的模糊边界.
panel (b): SAM2 精细化后的最终实例掩码(蓝/红半透明+轮廓).
数据: viz_data/coarse_teaser/coarse_dump.json (dump_coarse_teaser.py 产出).
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mm
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from pick_viz_images import load_gt, norm_records
from crop_instance_compare import match_instances

GT_JSON = '/data1/wangcheng/dataset/WHU/2.4 annotation/annotation/test.json'
IMG_DIR = '/data1/wangcheng/dataset/WHU/2.2 test/test'
OURS = '/home/wangcheng/project/new/viz_data/ours_test_predictions.json'
DUMP = '/home/wangcheng/project/new/viz_data/coarse_teaser/coarse_dump.json'
FONT = '/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf'
FONT_B = '/usr/share/fonts/opentype/urw-base35/NimbusRoman-Bold.otf'

ORANGE = (0, 165, 255)
BLUE = (255, 80, 0)
RED = (0, 0, 230)
GLUE = (170, 0, 170)
FG_PT = (60, 200, 60)     # 前景点(绿)
BG_PT = (30, 30, 30)      # 背景点(黑)
WHITE = (255, 255, 255)
DARK = (35, 35, 35)


def contour_pop(m, color, canvas, thick=2):
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, cs, -1, WHITE, thick + 2)
    cv2.drawContours(canvas, cs, -1, color, thick)


def draw_point(canvas, x, y, fg):
    """前景=实心绿点; 背景=白描边×. 在最终画布尺度绘制, 避免放大图二次膨胀."""
    x, y = int(round(x)), int(round(y))
    if fg:
        cv2.circle(canvas, (x, y), 4, FG_PT, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 4, WHITE, 1, cv2.LINE_AA)
    else:
        cv2.drawMarker(canvas, (x, y), WHITE, cv2.MARKER_TILTED_CROSS, 12, 4,
                       cv2.LINE_AA)
        cv2.drawMarker(canvas, (x, y), BG_PT, cv2.MARKER_TILTED_CROSS, 12, 2,
                       cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stem', required=True)
    ap.add_argument('--ia', type=int, required=True)
    ap.add_argument('--ib', type=int, required=True)
    ap.add_argument('--out', default='teaser_ours.png')
    ap.add_argument('--zoom-at', default=None, help='x,y 全图坐标; 默认接触带质心')
    ap.add_argument('--margin', type=float, default=0.8)
    ap.add_argument('--panel-h', type=int, default=400)
    ap.add_argument('--coarse-alpha', type=float, default=0.38)
    ap.add_argument('--no-text', action='store_true')
    args = ap.parse_args()

    imgs, gt_all = load_gt(GT_JSON)
    stem2id = {Path(im['file_name']).stem: i for i, im in imgs.items()}
    iid = stem2id[args.stem]
    rles = gt_all[iid]
    ra, rb = rles[args.ia], rles[args.ib]
    ma, mb = mm.decode(ra), mm.decode(rb)
    img = cv2.imread(str(Path(IMG_DIR) / f'{args.stem}.TIF'))

    # --- Ours 管线产物 ---
    dump = json.load(open(DUMP))[args.stem]
    recs = norm_records(OURS, imgs, stem2id)
    preds = {i: [r for r in l if r[0] >= 0.5] for i, l in recs.items()}
    mo = match_instances(preds.get(iid), rles, 0.5)

    def best_prop(g):
        """覆盖 GT 实例 g 最大的 proposal 索引."""
        bk, bc = None, 0.0
        for k, p in enumerate(dump):
            if p['coarse_rle'] is None:
                continue
            cov = (mm.decode(p['coarse_rle']) & g).sum() / g.sum()
            if cov > bc:
                bc, bk = cov, k
        return bk

    ka, kb = best_prop(ma), best_prop(mb)
    pa_, pb_ = dump[ka], dump[kb]
    ca_ = mm.decode(pa_['coarse_rle'])
    cb_ = mm.decode(pb_['coarse_rle'])

    # --- 裁块 ---
    xa, ya, wa, ha_ = mm.toBbox(ra); xb, yb, wb, hb_ = mm.toBbox(rb)
    mx, my = args.margin * max(wa, wb), args.margin * max(ha_, hb_)
    x1 = int(max(0, min(xa, xb) - mx)); x2 = int(min(1024, max(xa+wa, xb+wb) + mx))
    y1 = int(max(0, min(ya, yb) - my)); y2 = int(min(1024, max(ya+ha_, yb+hb_) + my))
    crop = img[y1:y2, x1:x2].copy()
    Hc, Wc = crop.shape[:2]

    def local(b): return [int(b[0])-x1, int(b[1])-y1, int(b[2])-x1, int(b[3])-y1]

    # ---------- panel (a) ----------
    left = crop.copy()
    pts_main = []   # (x_local, y_local, is_fg) 缩放后统一绘制
    for cm in (ca_, cb_):
        c = cm[y1:y2, x1:x2] > 0
        ov = left.copy(); ov[c] = GLUE
        left = cv2.addWeighted(ov, args.coarse_alpha, left, 1 - args.coarse_alpha, 0)
        cs, _ = cv2.findContours(c.astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(left, cs, -1, (120, 30, 120), 1, cv2.LINE_AA)
    contour_pop(ma[y1:y2, x1:x2], BLUE, left, 1)
    contour_pop(mb[y1:y2, x1:x2], RED, left, 1)
    for p in (pa_, pb_):
        bx = local(p['box'])
        cv2.rectangle(left, (bx[0], bx[1]), (bx[2], bx[3]), ORANGE, 2)
        for q in p.get('points', []):
            if x1 <= q['x'] < x2 and y1 <= q['y'] < y2:
                pts_main.append((q['x'] - x1, q['y'] - y1, q['label'] == 1))

    # 放大中心: 接触带(两 GT 膨胀相交)质心
    if args.zoom_at:
        zx, zy = [int(v) for v in args.zoom_at.split(',')]
        zx, zy = zx - x1, zy - y1
    else:
        k = np.ones((15, 15), np.uint8)
        near = (cv2.dilate(ma, k) > 0) & (cv2.dilate(mb, k) > 0)
        if near.any():
            ys_, xs_ = np.where(near)
            zx, zy = int(xs_.mean()) - x1, int(ys_.mean()) - y1
        else:
            zx, zy = Wc // 2, Hc // 2
    Z = 64
    zx = max(Z, min(Wc - Z, zx)); zy = max(Z, min(Hc - Z, zy))
    zx1, zy1, zx2, zy2 = zx - Z, zy - Z, zx + Z, zy + Z
    pts_zoom = []
    src = crop[zy1:zy2, zx1:zx2].copy()
    for cm in (ca_, cb_):
        c = cm[zy1+y1:zy2+y1, zx1+x1:zx2+x1] > 0
        ov = src.copy(); ov[c] = GLUE
        src = cv2.addWeighted(ov, args.coarse_alpha, src, 1 - args.coarse_alpha, 0)
        cs, _ = cv2.findContours(c.astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(src, cs, -1, (120, 30, 120), 1, cv2.LINE_AA)
    contour_pop(ma[zy1+y1:zy2+y1, zx1+x1:zx2+x1], BLUE, src, 2)
    contour_pop(mb[zy1+y1:zy2+y1, zx1+x1:zx2+x1], RED, src, 2)
    for p in (pa_, pb_):
        for q in p.get('points', []):
            qx, qy = q['x'] - x1, q['y'] - y1
            if zx1 <= qx < zx2 and zy1 <= qy < zy2:
                pts_zoom.append((qx - zx1, qy - zy1, q['label'] == 1))
    scale = min(2.6, (Hc - 12) / (zy2 - zy1))
    inset = cv2.resize(src, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    cv2.rectangle(left, (zx1, zy1), (zx2, zy2), WHITE, 3)
    cv2.rectangle(left, (zx1, zy1), (zx2, zy2), DARK, 1)

    # ---------- panel (b) ----------
    right = crop.copy()
    fa = mo[args.ia][0] if args.ia in mo else None
    fb = mo[args.ib][0] if args.ib in mo else None
    for rle, color in [(fa, BLUE), (fb, RED)]:
        if rle is None:
            continue
        m = mm.decode(rle)[y1:y2, x1:x2]
        ov = right.copy(); ov[m > 0] = color
        right = cv2.addWeighted(ov, 0.45, right, 0.55, 0)
        contour_pop(m, color, right)
    for p in (pa_, pb_):
        bx = local(p['box'])
        cv2.rectangle(right, (bx[0], bx[1]), (bx[2], bx[3]), ORANGE, 2)

    # ---------- 合成: [inset] -- [a] -> [b] ----------
    GAP = 34
    LINK = 12
    sc = max(1.0, args.panel_h / Hc)
    L = cv2.resize(left, None, fx=sc, fy=sc, interpolation=cv2.INTER_LANCZOS4)
    R = cv2.resize(right, None, fx=sc, fy=sc, interpolation=cv2.INTER_LANCZOS4)
    ins = cv2.resize(inset, None, fx=sc, fy=sc, interpolation=cv2.INTER_LANCZOS4)
    ph, pw = L.shape[:2]
    ih, iw = ins.shape[:2]
    ix0 = 2
    iy = int(max(4, min(ph - ih - 4, zy * sc - ih / 2)))
    ax0 = ix0 + iw + LINK
    CAP = 0 if args.no_text else 40
    # 点标记: 在最终分辨率上绘制(面板 a 随 sc 缩放, 放大图随 scale*sc 缩放),
    # 必须在贴入 canvas 之前画在 L/ins 上
    for x, y, is_fg in pts_main:
        draw_point(L, x * sc, y * sc, is_fg)
    for x, y, is_fg in pts_zoom:
        draw_point(ins, x * scale * sc, y * scale * sc, is_fg)
    canvas = np.full((ph + CAP, ax0 + pw + GAP + pw, 3), 255, np.uint8)
    canvas[iy:iy+ih, ix0:ix0+iw] = ins
    cv2.rectangle(canvas, (ix0-1, iy-1), (ix0+iw, iy+ih), WHITE, 2)
    cv2.rectangle(canvas, (ix0-2, iy-2), (ix0+iw+1, iy+ih+1), DARK, 1)
    canvas[0:ph, ax0:ax0+pw] = L
    canvas[0:ph, ax0+pw+GAP:ax0+pw*2+GAP] = R
    for (sx, sy), (tx, ty) in [((ax0 + zx1 * sc, zy1 * sc), (ix0 + iw, iy)),
                               ((ax0 + zx1 * sc, zy2 * sc), (ix0 + iw, iy + ih))]:
        cv2.line(canvas, (int(sx), int(sy)), (int(tx), int(ty)), DARK, 1, cv2.LINE_AA)
    cv2.arrowedLine(canvas, (ax0 + pw - 2, ph // 2), (ax0 + pw + GAP + 2, ph // 2),
                    DARK, 2, cv2.LINE_AA, tipLength=0.4)
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    if not args.no_text:
        d = ImageDraw.Draw(pil)
        f = ImageFont.truetype(FONT, 21)
        fb_ = ImageFont.truetype(FONT_B, 21)
        d.text((6, ph + 8), '(a) proposal-conditioned coarse masks + mined prompts',
               font=f, fill=(60, 60, 60))
        d.text((ax0 + pw + GAP + 6, ph + 8), '(b) SAM2-refined instances (Ours)',
               font=fb_, fill=(20, 20, 20))
        d.text((ix0 + 8, iy + ih - 32), 'ambiguous boundary', font=f,
               fill=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))
    cv2.imwrite(args.out, cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))
    print(f'saved {args.out} | crop=({x1},{y1})-({x2},{y2}) {Wc}x{Hc} '
          f'propA=k{ka} propB=k{kb} zoom=({zx+x1},{zy+y1}) '
          f'oursA={"hit" if fa is not None else "miss"} oursB={"hit" if fb is not None else "miss"}')


if __name__ == '__main__':
    main()
