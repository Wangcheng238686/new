#!/usr/bin/env python
"""BoundaryAP —— CitySample 对比算法边界指标（用户口径 2026-10-22）

定义（内部边界带, 非轮廓两侧扩张）:
  B(M) = M - Erode(M, d)          3x3 全 1 核, 连续腐蚀 d 次, 图像外按背景(0)处理
  d   = max(1, round(r * sqrt(H^2 + W^2)))   按整图对角线
  512x512: r=2% -> d=14, r=1% -> d=7

匹配质量:
  q(P,G) = min(MaskIoU, BoundaryIoU), BoundaryIoU = |B(P)∩B(G)| / |B(P)∪B(G)|

AP: COCO 式 —— 阈值 0.50:0.05:0.95(10 档取均), 按原检测 score 全局降序贪心一对一
匹配(每档阈值独立匹配集), 每图预测实例上限 250, 主结果覆盖全部尺寸(无 area 过滤)。

用法(CLI, 对已落盘预测):
  python boundary_ap.py --pred-json PRED.json --gt-json GT.json --out-json OUT.json
pred 的 segmentation 兼容: RLE dict(counts bytes/未压缩 list) 与 polygon list。

作为模块: evaluate_boundary(preds, gt_json, max_dets=250) -> dict
"""
import argparse
import json
from collections import defaultdict

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

KERNEL3 = np.ones((3, 3), np.uint8)
REC_THRS = np.linspace(0.0, 1.00, 101)
IOU_THRS = np.arange(0.5, 1.0, 0.05)


def band_width(H, W, r):
    return max(1, int(round(r * np.hypot(H, W))))


def decode_seg(seg, h, w):
    """segmentation(RLE dict / polygon list) -> HxW bool."""
    if isinstance(seg, dict):
        counts = seg.get('counts')
        if isinstance(counts, (list, np.ndarray)):
            rle = mask_utils.frPyObjects(seg, h, w)
        else:
            rle = seg
        return mask_utils.decode(rle).astype(bool)
    rles = mask_utils.frPyObjects(seg, h, w)
    if len(rles) == 1:
        return mask_utils.decode(rles[0]).astype(bool)
    return mask_utils.merge(rles).astype(bool) > 0


def band_indices(mask_bool, d):
    """B(M) = M & ~Erode(M,d); 图像外按背景 -> 先零填充 d 再腐蚀后裁回."""
    m = mask_bool.astype(np.uint8)
    mp = np.pad(m, d, constant_values=0)
    er = cv2.erode(mp, KERNEL3, iterations=d)[d:-d, d:-d]
    band = m.astype(bool) & ~(er.astype(bool))
    return np.flatnonzero(band.ravel())


def _ap_from_tp_fp(scores, tp, n_gt):
    """COCO 101 点插值 AP(precision envelope + recThrs 0:.01:1)."""
    if n_gt == 0 or len(scores) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind='mergesort')
    tp = np.asarray(tp, dtype=np.float64)[order]
    fp = 1.0 - tp
    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    # searchsorted(left) 给出首个 recall >= recThr 的位置; 越界(未达)记 0
    idx = np.searchsorted(recall, REC_THRS, side='left')
    out = np.zeros_like(REC_THRS)
    valid = idx < len(precision)
    out[valid] = precision[idx[valid]]
    return float(np.mean(out))


def boundary_ap(preds_by_img, gts_by_img, H, W, r, max_dets=250):
    """preds_by_img/gts_by_img: {img_id: [(score, mask_bool), ...]} -> (AP, d)"""
    d = band_width(H, W, r)
    all_scores, all_tp = [], []
    n_gt_total = 0
    for img_id, gt_list in gts_by_img.items():
        n_gt_total += len(gt_list)
        pred_list = sorted(preds_by_img.get(img_id, []),
                           key=lambda x: -x[0])[:max_dets]
        if not pred_list:
            continue
        matched = np.zeros((len(pred_list), len(IOU_THRS)))
        if gt_masks := [m for _, m in gt_list]:
            gt_idx = [np.flatnonzero(m.ravel()) for m in gt_masks]
            gt_band = [band_indices(m, d) for m in gt_masks]
            q_mat = np.zeros((len(pred_list), len(gt_masks)))
            for i, (s, pm) in enumerate(pred_list):
                p_idx = np.flatnonzero(pm.ravel())
                if p_idx.size == 0:
                    continue
                p_band = band_indices(pm, d)
                for j in range(len(gt_masks)):
                    inter = np.intersect1d(p_idx, gt_idx[j], assume_unique=True).size
                    if inter == 0:
                        continue  # 掩码不相交 -> 边界带(掩码子集)亦不相交
                    mask_iou = inter / (p_idx.size + gt_idx[j].size - inter)
                    ib = np.intersect1d(p_band, gt_band[j], assume_unique=True).size
                    ub = p_band.size + gt_band[j].size - ib
                    q_mat[i, j] = min(mask_iou, ib / ub if ub > 0 else 0.0)
            # 每档阈值独立匹配集; 预测按 score 降序贪心一对一(COCO 语义 q > thr)
            for k, thr in enumerate(IOU_THRS):
                used = np.zeros(len(gt_masks), dtype=bool)
                for i in range(len(pred_list)):
                    row = q_mat[i]
                    cand = np.where((row > thr) & ~used)[0]
                    if cand.size:
                        j = cand[np.argmax(row[cand])]
                        used[j] = True
                        matched[i, k] = 1.0
        for i, (s, _) in enumerate(pred_list):
            all_scores.append(s)
            all_tp.append(matched[i].tolist())
    scores_arr = np.asarray(all_scores)
    tp_arr = np.asarray(all_tp).reshape(len(all_scores), len(IOU_THRS))
    aps = [_ap_from_tp_fp(scores_arr, tp_arr[:, k], n_gt_total)
           for k in range(len(IOU_THRS))]
    return float(np.mean(aps)), d


def evaluate_boundary(preds, gt_json, max_dets=250):
    """preds: list[{image_id, score, segmentation}]; 返回两尺度 BoundaryAP."""
    coco = COCO(gt_json)
    dims = {im['id']: (im['height'], im['width']) for im in coco.dataset['images']}
    gts_by_img = defaultdict(list)
    for ann in coco.dataset['annotations']:
        h, w = dims[ann['image_id']]
        gts_by_img[ann['image_id']].append(
            (1.0, decode_seg(ann['segmentation'], h, w)))
    preds_by_img = defaultdict(list)
    for p in preds:
        h, w = dims[p['image_id']]
        preds_by_img[p['image_id']].append(
            (float(p['score']), decode_seg(p['segmentation'], h, w)))
    # 当前数据集全图 512x512, 取首图尺寸; 多尺寸需按图分组(从简不展开)
    H, W = dims[next(iter(gts_by_img))]
    out = {}
    for name, r in (('BoundaryAP_2pct', 0.02), ('BoundaryAP_1pct', 0.01)):
        ap, d = boundary_ap(preds_by_img, gts_by_img, H, W, r, max_dets)
        out[name] = round(ap, 4)
        out[f'{name}_d'] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred-json', required=True)
    ap.add_argument('--gt-json', required=True)
    ap.add_argument('--out-json', default=None)
    ap.add_argument('--max-dets', type=int, default=250)
    args = ap.parse_args()

    def _dec(o):
        if isinstance(o, (bytes, bytearray)):
            return o.decode('utf-8', 'replace')
        if isinstance(o, (np.ndarray, np.generic)):
            return o.item() if o.ndim == 0 else o.tolist()
        return o
    preds = json.loads(json.dumps(json.load(open(args.pred_json)), default=_dec))
    out = evaluate_boundary(preds, args.gt_json, max_dets=args.max_dets)
    print(json.dumps(out, indent=2))
    if args.out_json:
        with open(args.out_json, 'w') as f:
            json.dump(out, f, indent=2)
        print('saved', args.out_json)


if __name__ == '__main__':
    main()
