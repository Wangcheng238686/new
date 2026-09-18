#!/usr/bin/env python
"""叙事验证/素材导出: 只给 SAM2 一个候选框(box prompt), 看掩码解码能否分离相邻建筑.

变体:
  union -- 两栋建筑的 union 框(单个候选框)
  sep   -- 两栋各一个框(两个候选框)
输出: npz 保存每种变体的掩码, 供 teaser 渲染.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mm

GT_JSON = '/data1/wangcheng/dataset/WHU/2.4 annotation/annotation/test.json'
IMG_DIR = '/data1/wangcheng/dataset/WHU/2.2 test/test'
SAM2_REPO = '/home/wangcheng/project/new/sam2'
SAM2_CKPT = '/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stem', default='300838')
    ap.add_argument('--ia', type=int, default=7)
    ap.add_argument('--ib', type=int, default=8)
    ap.add_argument('--margin', type=float, default=0.8)
    ap.add_argument('--out', default='/home/wangcheng/project/new/viz_data/sam2_boxonly.npz')
    ap.add_argument('--device', default='cuda:3')
    args = ap.parse_args()

    from pycocotools.coco import COCO
    coco = COCO(GT_JSON)
    iid = [i for i, im in coco.imgs.items()
           if Path(im['file_name']).stem == args.stem][0]
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[iid]))
    anns = sorted(anns, key=lambda a: a['id'])
    ra = mm.decode(coco.annToRLE(anns[args.ia]))
    rb = mm.decode(coco.annToRLE(anns[args.ib]))
    xa, ya, wa, ha = anns[args.ia]['bbox']
    xb, yb, wb, hb = anns[args.ib]['bbox']
    xa, ya, wa, ha = [float(v) for v in (xa, ya, wa, ha)]
    xb, yb, wb, hb = [float(v) for v in (xb, yb, wb, hb)]

    mx, my = args.margin * max(wa, wb), args.margin * max(ha, hb)
    x1 = int(max(0, min(xa, xb) - mx)); x2 = int(min(1024, max(xa+wa, xb+wb) + mx))
    y1 = int(max(0, min(ya, yb) - my)); y2 = int(min(1024, max(ya+ha, yb+hb) + my))
    img = cv2.imread(str(Path(IMG_DIR) / f'{args.stem}.TIF'))
    crop = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)

    def to_crop(bx, by, bw, bh):
        return [bx - x1, by - y1, bx + bw - x1, by + bh - y1]

    union_box = to_crop(min(xa, xb), min(ya, yb),
                        max(xa+wa, xb+wb) - min(xa, xb),
                        max(ya+ha, yb+hb) - min(ya, yb))
    box_a = to_crop(xa, ya, wa, ha)
    box_b = to_crop(xb, yb, wb, hb)

    sys.path.insert(0, SAM2_REPO)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2('configs/sam2/sam2_hiera_b+.yaml', SAM2_CKPT,
                       device=args.device)
    predictor = SAM2ImagePredictor(model)
    predictor.set_image(crop)

    Hc, Wc = crop.shape[:2]
    ra_c = ra[y1:y2, x1:x2]
    rb_c = rb[y1:y2, x1:x2]
    out = {'crop_box': np.array([x1, y1, x2, y2])}
    variants = {
        'union': np.array([union_box], dtype=np.float32),
        'sep': np.array([box_a, box_b], dtype=np.float32),
    }
    for name, boxes in variants.items():
        masks, scores, _ = predictor.predict(
            box=boxes, multimask_output=False)
        masks = np.asarray(masks).squeeze(1) if masks.ndim == 4 else np.asarray(masks)
        out[f'{name}_scores'] = np.asarray(scores).reshape(-1)
        print(f'{name}: n={len(masks)} scores={np.round(np.asarray(scores).reshape(-1), 3)}')
        fixed = []
        for i, m in enumerate(masks):
            if m.shape != (Hc, Wc):
                m = cv2.resize(m.astype(np.uint8), (Wc, Hc),
                               interpolation=cv2.INTER_NEAREST)
            m = (m > 0).astype(np.uint8)
            fixed.append(m)
            inter_a = (m & ra_c).sum() / ra_c.sum()
            inter_b = (m & rb_c).sum() / rb_c.sum()
            print(f'  mask{i}: 盖A={inter_a:.2f} 盖B={inter_b:.2f} '
                  f'A&B内={(m & ra_c & rb_c).sum()}')
        out[f'{name}_masks'] = np.stack(fixed)
    np.savez_compressed(args.out, **out)
    print(f'saved -> {args.out}')


if __name__ == '__main__':
    main()
