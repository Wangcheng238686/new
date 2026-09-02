#!/usr/bin/env python
"""Boundary AP evaluation on saved COCO predictions (pure CPU post-process).

Implements the Boundary-IoU-style metric (Cheng et al., CVPR'21): both GT
and predicted masks are trimmed to a boundary band of width d px (mask minus
its d-px interior erosion), then evaluated with the standard COCO protocol.
Trim distance is FIXED PIXELS (not %-of-diagonal) because this dataset has
many objects smaller than 2% of the image diagonal.
"""
import argparse
import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from pycocotools import mask as mu
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.mask import frPyObjects

_G = {}


def _init(gt_json, pred_json, d):
    _G["d"] = d
    _G["gt"] = json.load(open(gt_json))
    _G["preds"] = json.load(open(pred_json))


def _single_rle(seg, h, w):
    if isinstance(seg, dict):
        r = dict(seg)
        if isinstance(r.get("counts"), str):
            r["counts"] = r["counts"].encode()
        return mu.merge([r])
    return mu.merge(list(frPyObjects(seg, h, w)))


def _band(mask: np.ndarray, d: int) -> np.ndarray:
    """mask minus its d-px interior erosion (distance-transform based)."""
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    return ((dist <= d) & (mask > 0)).astype(np.uint8)


def _proc_image(iid):
    d = _G["d"]
    gt_imgs = _G["gt"]["images"]
    h = next(im["height"] for im in gt_imgs if im["id"] == iid)
    w = next(im["width"] for im in gt_imgs if im["id"] == iid)
    out = {"gt": [], "dt": []}
    for a in _G["gt"]["annotations"]:
        if a["image_id"] != iid:
            continue
        m = mu.decode(_single_rle(a["segmentation"], h, w))
        rle = mu.encode(np.asfortranarray(_band(m, d)))
        rle["counts"] = rle["counts"].decode()
        out["gt"].append({**a, "segmentation": rle})
    for p in _G["preds"]:
        if p["image_id"] != iid:
            continue
        m = mu.decode(_single_rle(p["segmentation"], 1024, 1024))
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.float32), (w, h),
                           interpolation=cv2.INTER_LINEAR)
        rle = mu.encode(np.asfortranarray(_band(m, d)))
        rle["counts"] = rle["counts"].decode()
        out["dt"].append({**p, "segmentation": rle})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--d", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-images", type=int, default=0,
                    help="0=all; smaller for quick/oracle checks")
    ap.add_argument("--oracle", action="store_true",
                    help="evaluate GT against itself (sanity: bAP=1.0)")
    args = ap.parse_args()

    for d in args.d:
        with Pool(args.workers, initializer=_init,
                  initargs=(args.gt, args.pred, d)) as pool:
            iids = sorted({im["id"] for im in json.load(open(args.gt))["images"]})
            if args.max_images:
                iids = iids[: args.max_images]
            results = pool.map(_proc_image, iids, chunksize=8)

        gt_base = json.load(open(args.gt))
        gt_imgs = [im for im in gt_base["images"] if im["id"] in set(iids)]
        gt_anns = []
        for r in results:
            for a in r["gt"]:
                gt_anns.append({
                    "id": a["id"], "image_id": a["image_id"],
                    "category_id": a["category_id"], "iscrowd": 0,
                    "area": int(mu.area(a["segmentation"])),
                    "bbox": list(mu.toBbox(a["segmentation"])),
                    "segmentation": a["segmentation"],
                })
        pred_anns = []
        src = (a for r in results for a in r["gt"]) if args.oracle else (
            a for r in results for a in r["dt"]
        )
        for a in src:
            pred_anns.append({
                "image_id": a["image_id"], "category_id": a["category_id"],
                "score": float(a.get("score", 0.99)),
                "segmentation": a["segmentation"],
            })
        coco_gt = COCO()
        coco_gt.dataset = {
            "images": gt_imgs,
            "annotations": gt_anns,
            "categories": gt_base["categories"],
        }
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(pred_anns)
        ev = COCOeval(coco_gt, coco_dt, "segm")
        ev.params.imgIds = iids
        ev.evaluate(); ev.accumulate(); ev.summarize()
        print(
            f"[d={d}px] boundaryAP={ev.stats[0]:.4f} "
            f"bAP50={ev.stats[1]:.4f} bAP75={ev.stats[2]:.4f} "
            f"bAR@100={ev.stats[8]:.4f}"
        )


if __name__ == "__main__":
    main()
