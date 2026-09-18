#!/usr/bin/env python
"""按逐图指标自动挑选适合论文展示的可视化对比图.

对每张 test 图、每个模型: score 过滤 -> mask 与 GT 贪心匹配(IoU>=0.5)
-> 单图 P / R / F1@0.5 / meanIoU / FP / FN。
输出:
  1) <out>/per_image_scores.csv       全部图 x 模型指标
  2) <out>/recommend_ours_top.txt     Ours 综合分最高的 N 张(挑"我们效果好"的图)
  3) <out>/recommend_advantage.txt    Ours - 指定对手 差距最大的 N 张(对比拉差距)
用法:
  python pick_viz_images.py --gt test.json \
      --pred ours=path/to/predictions.json --pred yolo=path/to/pred_coco.json \
      [--ours ours] [--rival yolo] [--top 20] [--score-thr 0.5] [--iou-thr 0.5]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils


def load_gt(gt_json):
    gt = json.load(open(gt_json))
    imgs = {im['id']: im for im in gt['images']}
    anns = {}
    for a in gt['annotations']:
        anns.setdefault(a['image_id'], []).append(a)
    # 每图 GT RLE 列表
    gt_rles = {}
    for iid, alist in anns.items():
        im = imgs[iid]
        h, w = im['height'], im['width']
        rles = []
        for a in alist:
            seg = a['segmentation']
            if isinstance(seg, dict):
                rles.append(mask_utils.frPyObjects(seg, h, w)
                            if isinstance(seg.get('counts'), list) else seg)
            else:
                rs = mask_utils.frPyObjects(seg, h, w)
                rles.append(rs[0] if len(rs) == 1 else mask_utils.merge(rs))
        gt_rles[iid] = rles
    return imgs, gt_rles


def norm_records(pred_json, imgs, stem2id):
    """统一为 {image_id: [ (score, rle) ]}; 支持 COCO 与 ours 契约两种格式."""
    preds = json.load(open(pred_json))
    out = {}
    for p in preds:
        seg = p.get('segmentation')
        score = float(p.get('score', 0))
        if 'image_id' in p and p['image_id'] in imgs:      # COCO 格式
            im = imgs[p['image_id']]
            iid = p['image_id']
        else:                                              # ours 契约格式(file_name)
            stem = Path(p.get('file_name', '')).stem
            if stem not in stem2id:
                continue
            iid = stem2id[stem]
            im = imgs[iid]
        h, w = im['height'], im['width']
        if isinstance(seg, dict):
            rle = seg if not isinstance(seg.get('counts'), list) \
                else mask_utils.frPyObjects(seg, h, w)
            mh, mw = int(rle['size'][0]), int(rle['size'][1])
            if (mh, mw) != (h, w):                          # model-frame -> 原图
                import cv2 as _cv2
                m = mask_utils.decode(rle)
                m = _cv2.resize(m, (w, h), interpolation=_cv2.INTER_NEAREST)
                rle = mask_utils.encode(np.asfortranarray(m))
        elif isinstance(seg, list) and len(seg) and len(seg[0]) >= 6:
            rs = mask_utils.frPyObjects(seg, h, w)
            rle = rs[0] if len(rs) == 1 else mask_utils.merge(rs)
        else:
            continue
        out.setdefault(iid, []).append((score, rle))
    for iid in out:
        out[iid].sort(key=lambda x: -x[0])
    return out


def match_image(dt_list, gt_rles, iou_thr):
    """贪心匹配(按分数降序的 DT 对 GT 最大 IoU)."""
    if not gt_rles:
        return None  # 空 GT 图不参与(没有意义)
    matched_gt, tps_ious = set(), []
    n_fp = 0
    for _, drle in dt_list or []:
        best_iou, best_j = 0.0, -1
        for j, grle in enumerate(gt_rles):
            if j in matched_gt:
                continue
            iou = mask_utils.iou([drle], [grle], [0])[0][0]
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thr and best_j >= 0:
            matched_gt.add(best_j)
            tps_ious.append(best_iou)
        else:
            n_fp += 1
    tp = len(tps_ious)
    fn = len(gt_rles) - tp
    prec = tp / (tp + n_fp) if tp + n_fp else 0.0
    rec = tp / len(gt_rles)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    miou = float(np.mean(tps_ious)) if tps_ious else 0.0
    return dict(tp=tp, fp=n_fp, fn=fn, precision=prec, recall=rec,
                f1=f1, mean_iou=miou,
                composite=f1 * (0.5 + 0.5 * miou))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', required=True)
    ap.add_argument('--pred', action='append', required=True,
                    help='name=path (name 用于列名, path 为预测 json)')
    ap.add_argument('--ours', default=None, help='--pred 中的 Ours 名字')
    ap.add_argument('--rival', default=None, help='对比优势计算的对手名字')
    ap.add_argument('--top', type=int, default=20)
    ap.add_argument('--score-thr', type=float, default=0.5)
    ap.add_argument('--iou-thr', type=float, default=0.5)
    ap.add_argument('--out', default='viz_pick')
    args = ap.parse_args()

    imgs, gt_rles = load_gt(args.gt)
    stem2id = {Path(im['file_name']).stem: iid for iid, im in imgs.items()}

    models = {}
    for spec in args.pred:
        name, path = spec.split('=', 1)
        recs = norm_records(path, imgs, stem2id)
        models[name] = {iid: [r for r in lst if r[0] >= args.score_thr]
                        for iid, lst in recs.items()}
        print(f'[{name}] loaded {sum(len(v) for v in models[name].values())} preds '
              f'over {len(models[name])} images')

    all_ids = sorted(gt_rles.keys())
    per_img = {}
    for iid in all_ids:
        row = {}
        for name, recs in models.items():
            row[name] = match_image(recs.get(iid), gt_rles[iid], args.iou_thr)
        if any(v is not None for v in row.values()):
            per_img[iid] = row

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = list(models)
    with open(out / 'per_image_scores.csv', 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['image_id', 'file_name'] +
                    [f'{n}_{k}' for n in names
                     for k in ('f1', 'mean_iou', 'precision', 'recall', 'fp', 'fn')])
        for iid, row in per_img.items():
            flat = []
            for n in names:
                r = row.get(n)
                flat += ([f"{r['f1']:.3f}", f"{r['mean_iou']:.3f}",
                          f"{r['precision']:.3f}", f"{r['recall']:.3f}",
                          r['fp'], r['fn']] if r else ['nan'] * 6)
            wr.writerow([iid, imgs[iid]['file_name']] + flat)

    ours = args.ours or names[0]
    # 1) Ours 综合分 top
    scored = []
    for iid, row in per_img.items():
        r = row.get(ours)
        if r and r['tp'] >= 3:  # 实例太少的图对比不直观
            scored.append((r['composite'], iid, r))
    scored.sort(reverse=True)
    with open(out / 'recommend_ours_top.txt', 'w') as f:
        f.write(f'# rank\tcomposite\tf1\tmeanIoU\tP\tR\tfile\n')
        for k, (s, iid, r) in enumerate(scored[:args.top]):
            f.write(f"{k+1}\t{s:.3f}\t{r['f1']:.3f}\t{r['mean_iou']:.3f}\t"
                    f"{r['precision']:.3f}\t{r['recall']:.3f}\t{imgs[iid]['file_name']}\n")

    # 2) 对手差距 top
    if args.rival and args.rival in models:
        adv = []
        for iid, row in per_img.items():
            o, v = row.get(ours), row.get(args.rival)
            if o and v and o['tp'] >= 3:
                adv.append((o['composite'] - v['composite'], iid, o, v))
        adv.sort(reverse=True)
        with open(out / 'recommend_advantage.txt', 'w') as f:
            f.write(f'# rank\tadvantage\tours_f1\tours_mIoU\trival_f1\trival_mIoU\tfile\n')
            for k, (d, iid, o, v) in enumerate(adv[:args.top]):
                f.write(f"{k+1}\t{d:.3f}\t{o['f1']:.3f}\t{o['mean_iou']:.3f}\t"
                        f"{v['f1']:.3f}\t{v['mean_iou']:.3f}\t{imgs[iid]['file_name']}\n")

    print(f'saved -> {out}/per_image_scores.csv, recommend_ours_top.txt'
          + (', recommend_advantage.txt' if args.rival else ''))
    print('top-5 ours:')
    for s, iid, r in scored[:5]:
        print(f'  {imgs[iid]["file_name"]}  composite={s:.3f} f1={r["f1"]:.3f} mIoU={r["mean_iou"]:.3f}')


if __name__ == '__main__':
    main()
