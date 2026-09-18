#!/usr/bin/env python
"""NWPU VHR-10 COCO 实例分割标注 -> YOLO seg 格式 (ultralytics)

对每个 split 生成:
  /data1/wangcheng/dataset/NWPU VHR-10 dataset/yolo_seg/images/<split>/<fname> (符号链接到原图)
  /data1/wangcheng/dataset/NWPU VHR-10 dataset/yolo_seg/labels/<split>/<stem>.txt
    每实例一行: `<cls-1> x1 y1 x2 y2 ...` (归一化多边形, 10 类, cls = COCO category_id - 1)
RLE 标注自动解码为最大外轮廓 polygon 兜底 (NWPU 官方为 polygon, 一般不触发)。

用法: python coco2yolo_seg_nwpu.py [--splits train val]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
from ultralytics.data.converter import merge_multi_segment

NWPU_ROOT = Path('/data1/wangcheng/dataset/NWPU VHR-10 dataset')
OUT_ROOT = NWPU_ROOT / 'yolo_seg'
IMG_DIR = NWPU_ROOT / 'positive image set'

SPLITS = {
    'train': NWPU_ROOT / 'coco_split/NWPU_instances_train.json',
    'val': NWPU_ROOT / 'coco_split/NWPU_instances_val.json',
}

# 10 类, COCO category_id 1-10 -> YOLO cls 0-9
CLASSES = ['airplane', 'ship', 'storage_tank', 'baseball_diamond', 'tennis_court',
           'basketball_court', 'ground_track_field', 'harbor', 'bridge', 'vehicle']


def rle_to_polygons(rle):
    m = mask_utils.decode(rle)
    import cv2
    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    polys = [c.reshape(-1, 2).astype(float) for c in contours if len(c) >= 3]
    return polys


def convert_split(split):
    ann_path = SPLITS[split]
    out_img = OUT_ROOT / 'images' / split
    out_lbl = OUT_ROOT / 'labels' / split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    with open(ann_path) as f:
        data = json.load(f)

    anns_by_img = {}
    for ann in data['annotations']:
        if ann.get('iscrowd', 0):
            continue
        anns_by_img.setdefault(ann['image_id'], []).append(ann)

    n_img, n_ins, n_empty = len(data['images']), 0, 0
    for img in data['images']:
        w, h = img['width'], img['height']
        stem = Path(img['file_name']).stem
        lines = []
        for ann in anns_by_img.get(img['id'], []):
            segs = ann['segmentation']
            if isinstance(segs, dict):
                segs = rle_to_polygons(segs)
            polys = []
            for seg in segs:
                if isinstance(seg, list) and len(seg) >= 6:
                    flat = seg if isinstance(seg[0], (int, float)) else sum(seg, [])
                    if len(flat) >= 6:
                        polys.append(flat)
            if not polys:
                continue
            if len(polys) > 1:
                # ultralytics 约定: 多部件实例桥接合并为一行(一个实例)
                pairs = [np.asarray(flat, dtype=float).reshape(-1, 2).tolist()
                         for flat in polys]
                try:
                    pairs = merge_multi_segment(pairs)
                except Exception:
                    pass
                flat = [v for pt in pairs for v in pt]
            else:
                flat = polys[0]
            pts = np.asarray(flat, dtype=float).reshape(-1, 2).clip(0)
            # 归一化并裁剪到 [0,1]
            nx = (pts[:, 0] / w).clip(0, 1)
            ny = (pts[:, 1] / h).clip(0, 1)
            # YOLO 格式必须 x1,y1,x2,y2 交错(曾错写成 [全部x,全部y] 拼接)
            xy = np.empty(2 * len(nx), dtype=float)
            xy[0::2] = nx
            xy[1::2] = ny
            cls = ann['category_id'] - 1
            lines.append(f'{cls} ' + ' '.join(f'{v:.6f}' for v in xy.tolist()))
        (out_lbl / f'{stem}.txt').write_text('\n'.join(lines) + ('\n' if lines else ''))
        dst = out_img / Path(img['file_name']).name
        if not dst.exists():
            os.symlink(IMG_DIR / img['file_name'], dst)
        n_ins += len(lines)
        if not lines:
            n_empty += 1
    print(f'[{split}] images={n_img} instances(lines)={n_ins} empty={n_empty}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', nargs='+', default=['train', 'val'])
    args = ap.parse_args()
    for s in args.splits:
        convert_split(s)
    yaml = OUT_ROOT / 'data.yaml'
    yaml.write_text(
        '# NWPU VHR-10 instance segmentation (YOLO seg format)\n'
        f'path: {OUT_ROOT}\n'
        'train: images/train\n'
        'val: images/val\n'
        'names:\n' + ''.join(f'  {i}: {n}\n' for i, n in enumerate(CLASSES)))
    print('wrote', yaml)


if __name__ == '__main__':
    main()
