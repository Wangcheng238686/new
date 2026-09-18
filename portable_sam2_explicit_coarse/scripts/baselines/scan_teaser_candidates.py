#!/usr/bin/env python
"""扫描适合 teaser 的困难样本对: 两相邻纹理相似建筑 + 基线骑跨粘连 + Ours 成功分离.

筛选: 建筑面积够大(>=min_area)、bbox gap<=gap_max、裁块外扩 margin 后不出图、
      catnet/maskdino 至少一家骑跨(>=35% x2)、Ours 双命中(IoU>=iou_min).
排序: 双基线骑跨 > 面积更均等 > Ours min IoU.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from pycocotools import mask as mm

sys.path.insert(0, str(Path(__file__).parent))
from pick_viz_images import load_gt, norm_records
from crop_instance_compare import match_instances

GT_JSON = '/data1/wangcheng/dataset/WHU/2.4 annotation/annotation/test.json'
OURS = '/home/wangcheng/project/new/viz_data/ours_test_predictions.json'
BASE = '/data1/wangcheng/checkpoint/whu1024_baselines'
GLUE_SRC = {
    'catnet': f'{BASE}/catnet/cat_mask_rcnn_r50_3x_whu1024_bs4ai2/test_out/pred_full.segm.json',
    'maskdino': f'{BASE}/maskdino/pred_manual.segm.json',
}


def bbox_gap(a, b):
    """两 bbox 空隙(负=重叠)."""
    gx = max(a[0], b[0]) - min(a[0]+a[2], b[0]+b[2])
    gy = max(a[1], b[1]) - min(a[1]+a[3], b[1]+b[3])
    return max(gx, gy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-area', type=int, default=3500)
    ap.add_argument('--gap-max', type=int, default=8)
    ap.add_argument('--iou-min', type=float, default=0.6)
    ap.add_argument('--margin', type=float, default=0.8)
    ap.add_argument('--top', type=int, default=12)
    args = ap.parse_args()

    imgs, gt_all = load_gt(GT_JSON)
    stem2id = {Path(im['file_name']).stem: i for i, im in imgs.items()}
    recs_ours = {i: [r for r in l if r[0] >= 0.5]
                 for i, l in norm_records(OURS, imgs, stem2id).items()}
    recs_glue = {}
    for name, p in GLUE_SRC.items():
        recs_glue[name] = {i: [r for r in l if r[0] >= 0.5]
                           for i, l in norm_records(p, imgs, stem2id).items()}

    out = []
    for iid, rles in gt_all.items():
        if len(rles) < 2:
            continue
        boxes = [mm.toBbox(r) for r in rles]
        areas = [mm.area(r) for r in rles]
        # 廉价过滤先行: 面积/gap/出图(全 RLE 级, 无 decode)
        cand = []
        for a in range(len(rles)):
            for b in range(a + 1, len(rles)):
                if not (args.min_area <= areas[a] and args.min_area <= areas[b]):
                    continue
                if bbox_gap(boxes[a], boxes[b]) > args.gap_max:
                    continue
                x1 = min(boxes[a][0], boxes[b][0]) - args.margin * max(boxes[a][2], boxes[b][2])
                x2 = max(boxes[a][0]+boxes[a][2], boxes[b][0]+boxes[b][2]) + args.margin * max(boxes[a][2], boxes[b][2])
                y1 = min(boxes[a][1], boxes[b][1]) - args.margin * max(boxes[a][3], boxes[b][3])
                y2 = max(boxes[a][1]+boxes[a][3], boxes[b][1]+boxes[b][3]) + args.margin * max(boxes[a][3], boxes[b][3])
                if x1 < 4 or y1 < 4 or x2 > 1020 or y2 > 1020:
                    continue
                cand.append((a, b, x1, x2, y1, y2))
        if not cand:
            continue
        # 骑跨覆盖: RLE 级 iou 矩阵(C 实现), I = iou*(A+B)/(1+iou)
        cov = {name: [] for name in GLUE_SRC}  # name -> list[dict(gt_idx -> coverage)]
        for name, pool in recs_glue.items():
            pl = pool.get(iid, [])
            if not pl:
                continue
            ioum = mm.iou([r for s_, r in pl], list(rles), [0] * len(rles))
            pa = np.array([mm.area(r) for s_, r in pl])
            ga = np.array(areas)
            inter = ioum * (pa[:, None] + ga[None, :]) / (1.0 + ioum + 1e-9)
            covrg = inter / ga[None, :]
            for k in range(len(pl)):
                hits = {j for j in range(len(rles)) if covrg[k, j] > 0.35}
                cov[name].append(hits)
        mo = match_instances(recs_ours.get(iid), rles, 0.5)
        for a, b, x1, x2, y1, y2 in cand:
            # Ours 双命中
            if a not in mo or b not in mo:
                continue
            iou_o = min(mo[a][1], mo[b][1])
            if iou_o < args.iou_min:
                continue
            # 骑跨来源统计(查预计算的覆盖集合)
            glues = [name for name, lst in cov.items()
                     if any(a in hs and b in hs for hs in lst)]
            if not glues:
                continue
            eq = min(areas[a], areas[b]) / max(areas[a], areas[b])
            cw, ch = x2 - x1, y2 - y1
            out.append((len(glues), eq, iou_o, iid, a, b,
                        int(areas[a]), int(areas[b]), int(bbox_gap(boxes[a], boxes[b])),
                        glues, int(cw), int(ch), stem_of(imgs, iid)))

    out.sort(key=lambda t: (-t[0], -t[1], -t[2]))
    print(f'{"stem":10s} ia ib areaA areaB gap glues        oursMinIoU cropW cropH')
    for row in out[:args.top]:
        n_g, eq, iou_o, iid, a, b, A, B, gap, glues, cw, ch, stem = row
        print(f'{stem:10s} {a:2d} {b:2d} {A:5d} {B:5d} {gap:4d} {",".join(glues):12s} '
              f'{iou_o:.3f}      {cw:4d} {ch:4d}')


def stem_of(imgs, iid):
    return Path(imgs[iid]['file_name']).stem


if __name__ == '__main__':
    main()
