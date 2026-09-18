#!/usr/bin/env python
"""WHU-1024 COCO 实例分割标注 -> YOLO seg 格式 (ultralytics)

对每个 split 生成:
  /data1/wangcheng/dataset/WHU/yolo_seg/images/<split>/<fname>  (符号链接到原图)
  /data1/wangcheng/dataset/WHU/yolo_seg/labels/<split>/<stem>.txt
    每实例一行: `0 x1 y1 x2 y2 ...` (归一化多边形, 单类 building)
    无标注背景图 -> 空 txt (YOLO 负样本, 保持 train 2943 全量协议)
RLE 标注自动解码为最大外轮廓 polygon 兜底 (WHU 官方为 polygon, 一般不触发)。

用法: python coco2yolo_seg.py [--splits train val test]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
from ultralytics.data.converter import merge_multi_segment

WHU_ROOT = Path('/data1/wangcheng/dataset/WHU')
OUT_ROOT = WHU_ROOT / 'yolo_seg'

SPLITS = {
    'train': (WHU_ROOT / '2.4 annotation/annotation/train.json', '2.1 train/train'),
    'val': (WHU_ROOT / '2.4 annotation/annotation/validation.json', '2.3 valid/validation'),
    'test': (WHU_ROOT / '2.4 annotation/annotation/test.json', '2.2 test/test'),
}


def rle_to_polygons(rle):
    m = mask_utils.decode(rle)
    import cv2
    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    polys = [c.reshape(-1, 2).astype(float) for c in contours if len(c) >= 3]
    return polys


def convert_split(split):
    ann_path, img_sub = SPLITS[split]
    img_dir = WHU_ROOT / img_sub
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
            pts = np.asarray(flat, dtype=float).clip(0)
            # 归一化并裁剪到 [0,1]
            nx = (pts[0::2] / w).clip(0, 1)
            ny = (pts[1::2] / h).clip(0, 1)
            # YOLO 格式必须 x1,y1,x2,y2 交错(曾错写成 [全部x,全部y] 拼接)
            xy = np.empty(2 * len(nx), dtype=float)
            xy[0::2] = nx
            xy[1::2] = ny
            lines.append('0 ' + ' '.join(f'{v:.6f}' for v in xy.tolist()))
        (out_lbl / f'{stem}.txt').write_text('\n'.join(lines) + ('\n' if lines else ''))
        dst = out_img / Path(img['file_name']).name
        if not dst.exists():
            os.symlink(img_dir / img['file_name'], dst)
        n_ins += len(lines)
        if not lines:
            n_empty += 1
    print(f'[{split}] images={n_img} instances(lines)={n_ins} empty={n_empty}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    args = ap.parse_args()
    for s in args.splits:
        convert_split(s)
    yaml = OUT_ROOT / 'data.yaml'
    yaml.write_text(
        '# WHU-1024 building instance segmentation (YOLO seg format)\n'
        f'path: {OUT_ROOT}\n'
        'train: images/train\n'
        'val: images/val\n'
        'test: images/test\n'
        'names:\n'
        '  0: building\n')
    print('wrote', yaml)


if __name__ == '__main__':
    main()
