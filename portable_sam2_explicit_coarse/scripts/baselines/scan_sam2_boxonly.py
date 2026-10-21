#!/usr/bin/env python
"""扫描: 在紧邻建筑对上, SAM2 仅用候选框解码是否真的发生粘连/越界.

对每对 (stem, ia, ib) 测三种提示:
  union : 两栋的 union 单框  -> 期望失败模式: 单连通域同时盖两栋 / 大量盖住接触带
  sep   : 两栋各一框         -> 期望失败模式: 掩码越过自身 GT, 侵入邻居
输出每对的真实指标, 供挑选叙事成立且视觉明显的样本.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mm
from pycocotools.coco import COCO

GT_JSON = '/data1/wangcheng/dataset/WHU/2.4 annotation/annotation/test.json'
IMG_DIR = '/data1/wangcheng/dataset/WHU/2.2 test/test'
SAM2_REPO = '/home/wangcheng/project/new/sam2'
SAM2_CKPT = '/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt'

PAIRS = [
    ('300838', 7, 8), ('310401', 18, 19), ('3001075', 38, 41), ('30078', 34, 36),
    ('31028', 9, 13), ('3001387', 42, 43), ('300865', 14, 17), ('300169', 3, 4),
    ('300538', 22, 25), ('300220', 22, 28), ('300646', 5, 10), ('300838', 5, 7),
    ('300838', 5, 8), ('3001075', 0, 38), ('30078', 34, 35),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:3')
    ap.add_argument('--margin', type=float, default=0.8)
    args = ap.parse_args()

    coco = COCO(GT_JSON)
    stem2id = {Path(im['file_name']).stem: i for i, im in coco.imgs.items()}

    sys.path.insert(0, SAM2_REPO)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model = build_sam2('configs/sam2/sam2_hiera_b+.yaml', SAM2_CKPT,
                       device=args.device)
    predictor = SAM2ImagePredictor(model)

    cache_img = None
    print(f'{"stem":9s}{"pair":8s}{"union合并域":10s}{"union接触带":10s}{"union盖A/B":12s}'
          f'{"sep越界A->B":11s}{"sep越界B->A":11s}')
    for stem, ia, ib in PAIRS:
        iid = stem2id[stem]
        anns = sorted(coco.loadAnns(coco.getAnnIds(imgIds=[iid])), key=lambda a: a['id'])
        bx_a = [float(v) for v in anns[ia]['bbox']]
        bx_b = [float(v) for v in anns[ib]['bbox']]
        ra = mm.decode(coco.annToRLE(anns[ia]))
        rb = mm.decode(coco.annToRLE(anns[ib]))
        xa, ya, wa, ha = bx_a; xb, yb, wb, hb = bx_b
        mx = args.margin * max(wa, wb); my = args.margin * max(ha, hb)
        x1 = int(max(0, min(xa, xb) - mx)); x2 = int(min(1024, max(xa+wa, xb+wb) + mx))
        y1 = int(max(0, min(ya, yb) - my)); y2 = int(min(1024, max(ya+ha, yb+hb) + my))
        if cache_img is None or cache_img[0] != stem:
            cache_img = (stem, cv2.cvtColor(
                cv2.imread(str(Path(IMG_DIR) / f'{stem}.TIF')), cv2.COLOR_BGR2RGB))
        crop = cache_img[1][y1:y2, x1:x2]
        Hc, Wc = crop.shape[:2]
        ra_c = ra[y1:y2, x1:x2] > 0
        rb_c = rb[y1:y2, x1:x2] > 0
        k = np.ones((15, 15), np.uint8)
        strip = (cv2.dilate(ra_c.astype(np.uint8), k) > 0) & \
                (cv2.dilate(rb_c.astype(np.uint8), k) > 0) & ~ra_c & ~rb_c

        loc = lambda bx: np.array([[bx[0]-x1, bx[1]-y1, bx[0]+bx[2]-x1, bx[1]+bx[3]-y1]],
                                  dtype=np.float32)
        predictor.set_image(crop)
        # union
        masks, scores, _ = predictor.predict(box=loc([min(xa,xb), min(ya,yb),
             max(xa+wa,xb+wb)-min(xa,xb), max(ya+ha,yb+hb)-min(ya,yb)]),
             multimask_output=False)
        m = masks[0]
        if m.shape != (Hc, Wc):
            m = cv2.resize(m.astype(np.uint8), (Wc, Hc), interpolation=cv2.INTER_NEAREST)
        m = m > 0
        n, lab = cv2.connectedComponents(m.astype(np.uint8))
        comp_a = np.bincount(lab.ravel(), weights=ra_c.ravel().astype(float))
        comp_b = np.bincount(lab.ravel(), weights=rb_c.ravel().astype(float))
        merged = 0
        if n > 1:
            main_a = int(np.argmax(comp_a[1:])) + 1
            cov_a = comp_a[main_a] / max(1, ra_c.sum())
            cov_b = comp_b[main_a] / max(1, rb_c.sum())
            merged = (n - 1) if (cov_a < 0.8 or cov_b < 0.8) else 1
        strip_cov = (m & strip).sum() / max(1, strip.sum())
        # sep
        masks2, _, _ = predictor.predict(box=np.concatenate([loc(bx_a), loc(bx_b)]),
                                         multimask_output=False)
        masks2 = np.asarray(masks2)
        if masks2.ndim == 4:
            masks2 = masks2.squeeze(1)
        m2 = []
        for mk in masks2:
            if mk.shape != (Hc, Wc):
                mk = cv2.resize(mk.astype(np.uint8), (Wc, Hc),
                                interpolation=cv2.INTER_NEAREST)
            m2.append(mk > 0)
        bleed_a = (m2[0] & rb_c).sum() / max(1, m2[0].sum())
        bleed_b = (m2[1] & ra_c).sum() / max(1, m2[1].sum())
        print(f'{stem:9s}{str(ia)+"-"+str(ib):8s}'
              f'{str(merged)+"域":10s}{strip_cov:9.0%}  '
              f'{comp_a[1:].max()/max(1,ra_c.sum()):.0%}/{comp_b[1:].max()/max(1,rb_c.sum()):.0%}   '
              f'{bleed_a:10.1%} {bleed_b:10.1%}')


if __name__ == '__main__':
    main()
