#!/usr/bin/env python
"""Original-scale evaluation for the VHR-10 run.

The training/validation pipeline resizes each variable-size image to a fixed
1024x1024 canvas (anisotropic stretch) and evaluates there, against GT masks
that were NEAREST-resampled into that space.  The RSPrompter baseline instead
evaluates at the original image scale against the pristine polygon GT.

This script maps the saved predictions.json (1024-space RLE/boxes) back to
each image's original size and runs the identical COCO evaluation against the
untouched val json, so the two protocols can be compared directly.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from pycocotools import mask as mu
from pycocotools.coco import COCO

from utils.coco_eval_utils import VHR10_CATEGORIES, run_coco_eval


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="predictions.json (1024 space)")
    ap.add_argument("--gt", required=True, help="original val json (pristine)")
    ap.add_argument("--model-size", type=int, default=1024)
    args = ap.parse_args()

    gt = json.load(open(args.gt))
    sizes = {im["id"]: (im["height"], im["width"]) for im in gt["images"]}
    preds = json.load(open(args.pred))
    print(f"predictions: {len(preds)}  gt images: {len(sizes)}")

    mapped = []
    for p in preds:
        h, w = sizes[p["image_id"]]
        rle = p["segmentation"]
        m = mu.decode(rle)
        if rle["size"][0] != h or rle["size"][1] != w:
            m = cv2.resize(
                m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
            )
            m = (m > 0.5).astype(np.uint8)
        new_rle = mu.encode(np.asfortranarray(m))
        new_rle["counts"] = new_rle["counts"].decode("utf-8")
        x, y, bw, bh = p["bbox"]
        sx, sy = w / args.model_size, h / args.model_size
        mapped.append(
            {
                "image_id": p["image_id"],
                "category_id": int(p["category_id"]),
                "bbox": [x * sx, y * sy, bw * sx, bh * sy],
                "score": float(p["score"]),
                "segmentation": new_rle,
            }
        )

    coco_gt = COCO()
    coco_gt.dataset = {
        "images": gt["images"],
        "annotations": gt["annotations"],
        "categories": VHR10_CATEGORIES,
    }
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(mapped)

    segm = run_coco_eval(coco_gt, coco_dt, iou_type="segm")
    bbox = run_coco_eval(coco_gt, coco_dt, iou_type="bbox")
    print("\n===== original-scale evaluation =====")
    for name, m in (("segm", segm), ("bbox", bbox)):
        for k in [f"{name}/mAP", f"{name}/mAP_50", f"{name}/mAP_75",
                  f"{name}/mAP_s", f"{name}/mAP_m", f"{name}/mAP_l"]:
            if k in m:
                print(f"  {k:<12} {m[k]:.4f}")


if __name__ == "__main__":
    main()
