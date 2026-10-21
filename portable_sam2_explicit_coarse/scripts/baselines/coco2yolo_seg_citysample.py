#!/usr/bin/env python
"""CitySample-Sat 卫星侧 COCO 实例分割标注 -> YOLO seg 格式 (ultralytics)

对每个 split 生成:
  <root>/yolo_seg/images/<split>/<point_id>.png  (符号链接到卫星原图)
  <root>/yolo_seg/labels/<split>/<point_id>.txt
    每实例一行: `0 x1 y1 x2 y2 ...` (归一化多边形, 单类 building)

与 WHU 版 (coco2yolo_seg.py) 的差异:
  - 唯一 stem 用采样点号 (file_name 形如 points/000163/satellite/rgb.png,
    所有原图同名 rgb.png, 不能用 Path.stem)
  - segmentation 为未压缩列优先 RLE ({size, counts:list}), 必须先
    frPyObjects 压缩再 decode (数据集 README 明确要求 annToRLE 先行)
  - 无空图 (3060 点均有标注), 不产生负样本空 txt, 但保留兜底逻辑

用法: python coco2yolo_seg_citysample.py [--splits train val test]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from ultralytics.data.converter import merge_multi_segment

DS_ROOT = Path('/data1/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v3_20261021')
OUT_ROOT = DS_ROOT / 'yolo_seg'

SPLITS = {
    'train': DS_ROOT / 'annotations/train.json',
    'val': DS_ROOT / 'annotations/val.json',
    'test': DS_ROOT / 'annotations/test.json',
}


def stem_of(file_name):
    # points/000163/satellite/rgb.png -> 000163 (采样点号, 全数据集唯一)
    return Path(file_name).parts[1]


def rle_to_polygons(coco_api, ann):
    rle = coco_api.annToRLE(ann)  # 未压缩 counts -> 压缩 RLE (README 要求)
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

    coco = COCO(str(ann_path))
    imgs = coco.dataset['images']

    n_img, n_ins, n_empty = 0, 0, 0
    for img in imgs:
        w, h = img['width'], img['height']
        stem = stem_of(img['file_name'])
        lines = []
        for ann in coco.imgToAnns[img['id']]:
            if ann.get('iscrowd', 0):
                continue
            segs = ann['segmentation']
            polys = []
            if isinstance(segs, dict):
                # 未压缩 RLE: 解码为最大外轮廓 polygon (孔洞由外轮廓近似)
                polys = rle_to_polygons(coco, ann)
            else:
                for seg in segs:
                    if isinstance(seg, list) and len(seg) >= 6:
                        flat = seg if isinstance(seg[0], (int, float)) else sum(seg, [])
                        if len(flat) >= 6:
                            polys.append(flat)
            polys = [p for p in polys if len(p) >= 3]
            if not polys:
                continue
            if len(polys) > 1:
                # ultralytics 约定: 多部件实例桥接合并为一行(一个实例)
                pairs = [np.asarray(p, dtype=float).reshape(-1, 2).tolist()
                         for p in polys]
                try:
                    pairs = merge_multi_segment(pairs)
                except Exception:
                    pass
                flat = [v for pt in pairs for v in pt]
            else:
                flat = list(polys[0].flatten()) if isinstance(polys[0], np.ndarray) else list(polys[0])
            pts = np.asarray(flat, dtype=float).clip(0)
            # 归一化并裁剪到 [0,1]; YOLO 格式必须 x1,y1,x2,y2 交错
            nx = (pts[0::2] / w).clip(0, 1)
            ny = (pts[1::2] / h).clip(0, 1)
            xy = np.empty(2 * len(nx), dtype=float)
            xy[0::2] = nx
            xy[1::2] = ny
            lines.append('0 ' + ' '.join(f'{v:.6f}' for v in xy.tolist()))
        (out_lbl / f'{stem}.txt').write_text('\n'.join(lines) + ('\n' if lines else ''))
        dst = out_img / f'{stem}.png'
        if not dst.exists():
            os.symlink(DS_ROOT / img['file_name'], dst)
        n_img += 1
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
        '# CitySample-Sat satellite-only building instance seg (YOLO seg format)\n'
        f'path: {OUT_ROOT}\n'
        'train: images/train\n'
        'val: images/val\n'
        'test: images/test\n'
        'names:\n'
        '  0: building\n')
    print('wrote', yaml)


if __name__ == '__main__':
    main()
