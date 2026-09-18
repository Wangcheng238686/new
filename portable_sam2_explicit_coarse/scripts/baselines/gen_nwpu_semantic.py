#!/usr/bin/env python
"""从 NWPU train json 实例掩码合成 SCNet 语义分支所需的二值语义图。

(对应 WHU 版 gen_whu_semantic.py, 用于 SCNet 基线)

输出: "/data1/wangcheng/dataset/NWPU VHR-10 dataset/derived_sem/train/<stem>.png"
      前景(任意类别实例)像素=1, 背景=255(loss ignore_index=255)
"""
import os
import sys

import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from PIL import Image

ANN = '/data1/wangcheng/dataset/NWPU VHR-10 dataset/coco_split/NWPU_instances_train.json'
OUT_DIR = '/data1/wangcheng/dataset/NWPU VHR-10 dataset/derived_sem/train'


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    coco = COCO(ANN)
    n_done = 0
    for img_id, img_info in coco.imgs.items():
        stem = os.path.splitext(img_info['file_name'])[0]
        out_path = os.path.join(OUT_DIR, stem + '.png')
        if os.path.exists(out_path):
            n_done += 1
            continue
        sem = np.full((img_info['height'], img_info['width']), 255, dtype=np.uint8)
        for ann in coco.imgToAnns.get(img_id, []):
            rles = coco.annToRLE(ann)
            m = maskUtils.decode(rles).astype(bool)
            sem[m] = 1
        Image.fromarray(sem).save(out_path)
        n_done += 1
        if n_done % 500 == 0:
            print(f'{n_done}/{len(coco.imgs)}', flush=True)
    print(f'done: {n_done} maps -> {OUT_DIR}')


if __name__ == '__main__':
    sys.exit(main())
