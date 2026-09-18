#!/usr/bin/env python
"""论文首页 challenging-example teaser 图 (v2).

左 panel (a): 真实遥感裁块 + 橙色 proposal box(两相邻纹理相似建筑) + 品红半透明
  粘连初始 mask(基线真实骑跨预测, 形态学清洗) + GT 蓝红轮廓(白色衬底) +
  放大镜式局部放大框指向 ambiguous boundary.
右 panel (b): 克制的 instance-aware separation(Ours: 两实例蓝/红半透明+轮廓).
v2 改进: 粘连mask去噪、轮廓白衬底、放大镜连线、面板加高、(a)/(b)标题.
字体: NimbusRoman (Times 形近).
"""
import argparse
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
FONT = '/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf'
FONT_B = '/usr/share/fonts/opentype/urw-base35/NimbusRoman-Bold.otf'

ORANGE = (0, 165, 255)      # proposal box
BLUE = (255, 80, 0)         # instance A
RED = (0, 0, 230)           # instance B
GLUE = (170, 0, 170)        # 粘连初始 mask(品红)
WHITE = (255, 255, 255)
DARK = (35, 35, 35)


def contour_pop(m, color, canvas, thick=2):
    """带白色衬底的轮廓, 任何底色上都清晰."""
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, cs, -1, WHITE, thick + 2)
    cv2.drawContours(canvas, cs, -1, color, thick)


def clean_glue(pm, ma, mb):
    """骑跨 mask 去碎斑: 开运算+最大连通域+闭运算; 若清洗后不再骑跨则返回原 mask."""
    m = pm.copy()
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n > 2:  # 背景+最大1个
        keep = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        m = (lab == keep).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    if (m & ma).sum() > 0.28 * ma.sum() and (m & mb).sum() > 0.28 * mb.sum():
        return m
    return pm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stem', required=True)
    ap.add_argument('--ia', type=int, required=True)
    ap.add_argument('--ib', type=int, required=True)
    ap.add_argument('--out', default='teaser.png')
    ap.add_argument('--zoom-at', default=None, help='x,y 手动指定放大中心(全图坐标)')
    ap.add_argument('--glue-source', default='catnet',
                    help='粘连初始mask来源: catnet/maskdino/ours')
    ap.add_argument('--margin', type=float, default=0.8, help='裁块外扩比例')
    ap.add_argument('--panel-h', type=int, default=400)
    ap.add_argument('--glue-alpha', type=float, default=0.32)
    ap.add_argument('--glue-clip', type=int, default=0,
                    help='粘连mask裁剪到橙框外扩 pad 像素内(0=不裁剪)')
    ap.add_argument('--glue-npz', default=None,
                    help='SAM2 box-only 探针 npz(sam2_boxonly_probe.py): 用其 union 掩码作粘连mask')
    ap.add_argument('--no-text', action='store_true', help='纯净版: 图内无任何文字与底部标题带')
    ap.add_argument('--no-gt', action='store_true',
                    help='panel(a)不画GT蓝红轮廓(避免暗示逐实例边界已知)')
    ap.add_argument('--right-contour', action='store_true',
                    help='panel(b)掩码加彩色描边(默认无描边)')
    args = ap.parse_args()

    imgs, gt_all = load_gt(GT_JSON)
    stem2id = {Path(im['file_name']).stem: i for i, im in imgs.items()}
    iid = stem2id[args.stem]
    rles = gt_all[iid]
    ra, rb = rles[args.ia], rles[args.ib]
    ma, mb = mm.decode(ra), mm.decode(rb)
    img = cv2.imread(str(Path(IMG_DIR) / f'{args.stem}.TIF'))

    # Ours 预测
    recs = norm_records(OURS, imgs, stem2id)
    preds = {i: [r for r in l if r[0] >= 0.5] for i, l in recs.items()}
    mo = match_instances(preds.get(iid), rles, 0.5)
    # 骑跨粘连预测候选池(当 --glue-npz 未提供时使用)
    glue_pool = preds
    if args.glue_source != 'ours':
        gp = {'catnet': '/data1/wangcheng/checkpoint/whu1024_baselines/catnet/cat_mask_rcnn_r50_3x_whu1024_bs4ai2/test_out/pred_full.segm.json',
              'maskdino': '/data1/wangcheng/checkpoint/whu1024_baselines/maskdino/pred_manual.segm.json'}[args.glue_source]
        gr = norm_records(gp, imgs, stem2id)
        glue_pool = {i: [r for r in l if r[0] >= 0.5] for i, l in gr.items()}

    # 裁块: 两实例 union 外扩 margin
    xa, ya, wa, ha_ = mm.toBbox(ra); xb, yb, wb, hb_ = mm.toBbox(rb)
    mx, my = args.margin * max(wa, wb), args.margin * max(ha_, hb_)
    x1 = int(max(0, min(xa, xb) - mx)); x2 = int(min(1024, max(xa+wa, xb+wb) + mx))
    y1 = int(max(0, min(ya, yb) - my)); y2 = int(min(1024, max(ya+ha_, yb+hb_) + my))

    # 粘连 mask 来源 A: SAM2 box-only 探针(叙事: 候选框无形状线索 -> 解码器不分离)
    npz_mask = None
    if args.glue_npz:
        z = np.load(args.glue_npz)
        npz_mask = z['union_masks'][0].astype(np.uint8)
        gx1, gy1 = int(z['crop_box'][0]), int(z['crop_box'][1])
        bridge = np.zeros((1024, 1024), np.uint8)
        bridge[gy1:gy1 + npz_mask.shape[0], gx1:gx1 + npz_mask.shape[1]] = npz_mask
        glue = bridge
    else:
        # 来源 B: 骑跨预测(同时盖 35% 两个实例)
        bridge = None
        for s_, r in glue_pool.get(iid, []):
            pm = mm.decode(r)
            if (pm & ma).sum() > 0.35 * ma.sum() and (pm & mb).sum() > 0.35 * mb.sum():
                bridge = pm
                break
        if bridge is None:
            print('WARN: 无骑跨预测, 退化为 GT union 示意')
            glue = ma | mb
        else:
            glue = clean_glue(bridge, ma, mb)
        if args.glue_clip > 0 and bridge is not None:
            pad = int(args.glue_clip)
            bx1 = int(min(xa, xb)) - pad; by1 = int(min(ya, yb)) - pad
            bx2 = int(max(xa + wa, xb + wb)) + pad; by2 = int(max(ya + ha_, yb + hb_)) + pad
            keep = np.zeros(glue.shape, np.uint8)
            keep[max(0, by1):min(1024, by2), max(0, bx1):min(1024, bx2)] = 255
            keep = cv2.GaussianBlur(keep, (31, 31), 0)
            glue = glue & (keep > 127)
    crop = img[y1:y2, x1:x2].copy()
    Hc, Wc = crop.shape[:2]
    ga = glue[y1:y2, x1:x2] > 0

    # ---------- 左 panel (a) ----------
    left = crop.copy()
    ov = left.copy(); ov[ga] = GLUE
    left = cv2.addWeighted(ov, args.glue_alpha, left, 1 - args.glue_alpha, 0)
    if not args.no_gt:
        contour_pop(ma[y1:y2, x1:x2], BLUE, left)
        contour_pop(mb[y1:y2, x1:x2], RED, left)
    # 橙色 proposal box = 两实例 union bbox
    px1, py1 = int(min(xa, xb)) - x1, int(min(ya, yb)) - y1
    px2, py2 = int(max(xa+wa, xb+wb)) - x1, int(max(ya+ha_, yb+hb_)) - y1
    cv2.rectangle(left, (px1, py1), (px2, py2), ORANGE, 2)

    # 放大中心: 两实例膨胀相交区质心(默认) 或手动(全图坐标)
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
            zx, zy = (px1 + px2) // 2, (py1 + py2) // 2
    Z = 64  # zoom 半窗(裁块坐标)
    zx = max(Z, min(Wc - Z, zx)); zy = max(Z, min(Hc - Z, zy))
    zx1, zy1, zx2, zy2 = zx - Z, zy - Z, zx + Z, zy + Z
    src = crop[zy1:zy2, zx1:zx2].copy()
    ovz = src.copy(); ovz[glue[zy1+y1:zy2+y1, zx1+x1:zx2+x1] > 0] = GLUE
    src = cv2.addWeighted(ovz, args.glue_alpha, src, 1 - args.glue_alpha, 0)
    if not args.no_gt:
        contour_pop(ma[zy1+y1:zy2+y1, zx1+x1:zx2+x1], BLUE, src, 2)
        contour_pop(mb[zy1+y1:zy2+y1, zx1+x1:zx2+x1], RED, src, 2)
    # inset 不贴进 panel (a) 内部, 而是作为独立放大图在合成阶段向左延伸出去,
    # 保证 (a)/(b) 底图尺寸一致, 避免误读为两块不同大小的影像.
    scale = min(2.6, (Hc - 12) / (zy2 - zy1))
    inset = cv2.resize(src, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    # 源框(白衬底)
    cv2.rectangle(left, (zx1, zy1), (zx2, zy2), WHITE, 3)
    cv2.rectangle(left, (zx1, zy1), (zx2, zy2), DARK, 1)

    # ---------- 右 panel (b) ----------
    right = crop.copy()
    pa = mo[args.ia][0] if args.ia in mo else None
    pb = mo[args.ib][0] if args.ib in mo else None
    for rle, color in [(pa, BLUE), (pb, RED)]:
        if rle is None:
            continue
        m = mm.decode(rle)[y1:y2, x1:x2]
        ov = right.copy(); ov[m > 0] = color
        right = cv2.addWeighted(ov, 0.45, right, 0.55, 0)
        if args.right_contour:
            contour_pop(m, color, right)
        # 逐实例检测框(我们两级检测的输出), 与左图单候选框形成对照
        bx = mm.toBbox(rle)
        cv2.rectangle(right, (int(bx[0]) - x1, int(bx[1]) - y1),
                      (int(bx[0] + bx[2]) - x1, int(bx[1] + bx[3]) - y1), ORANGE, 2)

    # ---------- 合成 + 标注(PIL) ----------
    GAP = 34
    LINK = 12   # inset 与 panel (a) 的间距
    sc = max(1.0, args.panel_h / Hc)
    L = cv2.resize(left, None, fx=sc, fy=sc, interpolation=cv2.INTER_LANCZOS4)
    R = cv2.resize(right, None, fx=sc, fy=sc, interpolation=cv2.INTER_LANCZOS4)
    ins = cv2.resize(inset, None, fx=sc, fy=sc, interpolation=cv2.INTER_LANCZOS4)
    ph, pw = L.shape[:2]
    ih, iw = ins.shape[:2]
    ix0 = 2
    iy = int(max(4, min(ph - ih - 4, zy * sc - ih / 2)))
    ax0 = ix0 + iw + LINK  # panel (a) 左缘
    CAP = 0 if args.no_text else 40
    canvas = np.full((ph + CAP, ax0 + pw + GAP + pw, 3), 255, np.uint8)
    canvas[iy:iy+ih, ix0:ix0+iw] = ins
    cv2.rectangle(canvas, (ix0-1, iy-1), (ix0+iw, iy+ih), WHITE, 2)
    cv2.rectangle(canvas, (ix0-2, iy-2), (ix0+iw+1, iy+ih+1), DARK, 1)
    canvas[0:ph, ax0:ax0+pw] = L
    canvas[0:ph, ax0+pw+GAP:ax0+pw*2+GAP] = R
    # 连线: 源框左侧两角 -> inset 右侧两角, 视觉向左延伸
    for (sx, sy), (tx, ty) in [((ax0 + zx1 * sc, zy1 * sc), (ix0 + iw, iy)),
                               ((ax0 + zx1 * sc, zy2 * sc), (ix0 + iw, iy + ih))]:
        cv2.line(canvas, (int(sx), int(sy)), (int(tx), int(ty)), DARK, 1, cv2.LINE_AA)
    cv2.arrowedLine(canvas, (ax0 + pw - 2, ph // 2), (ax0 + pw + GAP + 2, ph // 2),
                    DARK, 2, cv2.LINE_AA, tipLength=0.4)
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    if not args.no_text:
        d = ImageDraw.Draw(pil)
        f = ImageFont.truetype(FONT, 21)
        fb = ImageFont.truetype(FONT_B, 21)
        d.text((6, ph + 8), '(a) coarse mask w/ ambiguous boundary', font=f, fill=(60, 60, 60))
        d.text((ax0 + pw + GAP + 6, ph + 8), '(b) instance-aware separation (Ours)', font=fb, fill=(20, 20, 20))
        # inset 内小标签(画布坐标)
        d.text((ix0 + 8, iy + ih - 32), 'ambiguous', font=f, fill=(255, 255, 255),
               stroke_width=1, stroke_fill=(0, 0, 0))
    out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.out, out)
    print(f'saved {args.out} | crop=({x1},{y1})-({x2},{y2}) {Wc}x{Hc} zoom=({zx+x1},{zy+y1}) '
          f'scale={sc:.2f} bridge={"real" if bridge is not None else "union-fallback"} '
          f'oursA={"hit" if pa is not None else "miss"} oursB={"hit" if pb is not None else "miss"} '
          f'iouA={mo[args.ia][1]:.2f} iouB={mo[args.ib][1]:.2f}' if args.ia in mo and args.ib in mo else 'miss')


if __name__ == '__main__':
    main()
