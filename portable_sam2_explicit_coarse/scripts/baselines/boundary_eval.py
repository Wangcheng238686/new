#!/usr/bin/env python
"""WHU-1024 test 预测的边界掩码指标(Boundary AP)评估器

输入: GT COCO json + 预测 COCO json(segmentation 为 polygon 或 RLE)
方法: 每实例 mask -> 边界带(距背景 <= 带宽的掩码像素) -> 以边界带作为
  'segm' 走标准 COCOeval(maxDets=[1,10,100], 统计口径=100, 与主表一致)
带宽口径(可双跑):
  coco : w = max(1, round(2*sqrt(A)/20))  # COCO2019 挑战 Boundary AP 口径 r=2
  fix2 : w = 2 px                            # 固定 2px 严苛口径
s/m/l 分组沿用原 mask 面积(与主表一致)。

用法: python boundary_eval.py PRED_JSON [--gt GT] [--mode coco|fix2]
        [--out OUT_JSON] [--tag NAME] [--workers 8]
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.ndimage import distance_transform_edt


def to_rles(seg, h, w):
    """segmentation(polygon/RLE) -> [rle, ...]"""
    if isinstance(seg, dict):
        c = seg.get('counts')
        if isinstance(c, (bytes, str)):
            return [seg]
        return [mask_utils.frPyObjects(seg, h, w)]
    out = []
    for el in seg:
        if isinstance(el, dict):
            c = el.get('counts')
            out.append(el if isinstance(c, (bytes, str))
                       else mask_utils.frPyObjects(el, h, w)[0])
        elif isinstance(el, (int, float)):
            # seg 本身是 flat polygon [x1,y1,...]
            r = mask_utils.frPyObjects([[float(v) for v in seg]], h, w)
            return list(r)
        else:
            out.extend(mask_utils.frPyObjects([[float(v) for v in el]], h, w))
    return out


def _band_one(args):
    """单实例: (rles, h, w, bw_mode) -> 边界带 RLE(多部件 merge)
    仅在 bbox+带宽窗口内做 EDT, 大幅加速(小目标 1024^2 -> ~50^2)"""
    rles, h, w, bw_mode = args
    if len(rles) > 1:
        rle = mask_utils.merge(rles)
    else:
        rle = rles[0]
    area = float(mask_utils.area(rle))
    if area <= 0:
        return None
    if bw_mode == 'coco':
        bw = max(1, int(round(2 * np.sqrt(area) / 20)))  # COCO2019 挑战: w=r*sqrt(A)/20, r=2
    elif bw_mode == 'fix2':
        bw = 2
    else:
        bw = 4
    bx, by, bwid, bhei = mask_utils.toBbox(rle).tolist()
    x0, y0 = max(0, int(bx) - bw), max(0, int(by) - bw)
    x1 = min(w, int(bx + bwid) + bw + 2)
    y1 = min(h, int(by + bhei) + bw + 2)
    m = mask_utils.decode(rle)[y0:y1, x0:x1]
    if area <= bw * bw * 4:  # 极小目标整体即边界
        band = m
    else:
        dist = distance_transform_edt(m)
        band = ((dist <= bw) & (m > 0)).astype(np.uint8)
    full = np.zeros((h, w), np.uint8, order='F')
    full[y0:y1, x0:x1] = band
    if full.sum() == 0:
        return None
    return mask_utils.encode(full)


def build_boundary_anns(anns, img_wh, bw_mode, workers=8):
    """anns: [{'image_id','segmentation','score?','area'?}] -> 边界带 ann 列表"""
    jobs = []
    meta = []
    for a in anns:
        h, w = img_wh[a['image_id']]
        rles = to_rles(a['segmentation'], h, w)
        jobs.append((rles, h, w, bw_mode))
        meta.append(a)
    with Pool(workers) as p:
        bands = p.map(_band_one, jobs, chunksize=64)
    out = []
    for a, b, j in zip(meta, bands, jobs):
        if b is None:
            continue
        rle = b[0] if isinstance(b, list) else b
        rle = dict(size=rle['size'], counts=rle['counts'])
        item = dict(image_id=a['image_id'], category_id=a.get('category_id', 1),
                    segmentation=rle)
        if 'score' in a:
            item['score'] = a['score']
            # bbox 取边界带的外接框(匹配需要, 对边界指标无直接意义)
            bb = mask_utils.toBbox(rle).tolist()
            item['bbox'] = bb
            item['area'] = float(mask_utils.area(rle))
        else:
            item['bbox'] = a.get('bbox', mask_utils.toBbox(rle).tolist())
            # GT 保留原 mask 面积用于 s/m/l 分组
            item['area'] = float(a.get('area') or sum(
                mask_utils.area(r) for r in j[0]))
        out.append(item)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pred_json')
    ap.add_argument('--gt', default='/data1/wangcheng/dataset/WHU/2.4 annotation/annotation/test.json')
    ap.add_argument('--mode', default='coco', choices=['coco', 'fix2', 'fix4'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    gt_raw = json.load(open(args.gt))
    img_wh = {im['id']: (im['height'], im['width']) for im in gt_raw['images']}
    cat_id = gt_raw['categories'][0]['id']

    print(f'[{args.tag or args.pred_json}] building GT boundary bands ({args.mode}) ...')
    gt_anns = [dict(image_id=a['image_id'], category_id=a['category_id'],
                    segmentation=a['segmentation'], area=a['area'], bbox=a['bbox'])
               for a in gt_raw['annotations']]
    gt_bands = build_boundary_anns(gt_anns, img_wh, args.mode, args.workers)
    print(f'  GT: {len(gt_anns)} -> {len(gt_bands)} bands')

    preds_raw = json.load(open(args.pred_json))
    # category_id 归一为 GT 真实 id: RS4D fork 的 CocoMetric 写 0 起始, 其余写 1
    dt_anns = [dict(image_id=p['image_id'], category_id=cat_id,
                    segmentation=p['segmentation'], score=p['score'])
               for p in preds_raw if p.get('segmentation')]
    print(f'  Pred: building {len(dt_anns)} boundary bands ...')
    dt_bands = build_boundary_anns(dt_anns, img_wh, args.mode, args.workers)

    # 组装成可被 COCO/loadRes 消化的形式
    gt_for_eval = {im['id']: im for im in gt_raw['images']}
    gt_dict = dict(images=gt_raw['images'], categories=gt_raw['categories'],
                   annotations=[
                       dict(id=i + 1, image_id=b['image_id'], category_id=cat_id,
                            segmentation=b['segmentation'], area=b['area'],
                            iscrowd=0, bbox=b['bbox'])
                       for i, b in enumerate(gt_bands)])
    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(dt_bands)

    E = COCOeval(coco_gt, coco_dt, 'segm')
    E.evaluate()
    E.accumulate()
    E.summarize()
    names = ['mAP', 'mAP50', 'mAP75', 'mAPs', 'mAPm', 'mAPl']
    metrics = {f'boundary_{n}': round(float(E.stats[i]), 4) for i, n in enumerate(names)}
    metrics['mode'] = args.mode
    metrics['tag'] = args.tag or os.path.basename(args.pred_json)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.pred_json)),
                                   f'boundary_{args.mode}.json')
    json.dump(metrics, open(out, 'w'), indent=2)
    print('saved', out)


if __name__ == '__main__':
    main()
