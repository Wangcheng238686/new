from collections import OrderedDict
from typing import List, Tuple

import numpy as np


def _mask_to_rle(binary_mask: np.ndarray) -> dict:
    """Convert binary mask (H, W) to COCO RLE format."""
    from pycocotools import mask as mask_util

    mask_fortran = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_util.encode(mask_fortran)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def _xyxy_to_xywh(bbox: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to [x, y, w, h]."""
    out = bbox.copy()
    out[..., 2] = bbox[..., 2] - bbox[..., 0]
    out[..., 3] = bbox[..., 3] - bbox[..., 1]
    return out


def build_coco_gt_and_dt(
    all_gt: List[dict],
    all_dt: List[dict],
    img_metas_list: List[dict],
    category_name: str = "building",
) -> Tuple:
    """Build pycocotools COCO objects for GT and detections."""
    from pycocotools.coco import COCO

    images = []
    annotations = []
    predictions = []
    ann_id = 1

    for img_id, (gt, dt, meta) in enumerate(zip(all_gt, all_dt, img_metas_list), start=1):
        h, w = meta["img_shape"][:2]
        images.append({"id": img_id, "height": int(h), "width": int(w)})

        gt_bboxes = gt["bboxes"]
        gt_labels = gt["labels"]
        gt_masks = gt["masks"]

        if hasattr(gt_masks, "masks"):
            gt_masks = gt_masks.masks

        num_gt = len(gt_labels)
        for i in range(num_gt):
            bbox_xywh = _xyxy_to_xywh(gt_bboxes[i]).tolist()
            area = float(bbox_xywh[2] * bbox_xywh[3])
            seg_rle = _mask_to_rle(gt_masks[i])
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": bbox_xywh,
                    "area": area,
                    "segmentation": seg_rle,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        dt_bboxes = dt["bboxes"]
        dt_scores = dt["scores"]
        dt_masks = dt["masks"]

        if hasattr(dt_masks, "masks"):
            dt_masks = dt_masks.masks

        num_dt = len(dt_scores)
        for i in range(num_dt):
            bbox_xywh = _xyxy_to_xywh(dt_bboxes[i]).tolist()
            seg_rle = _mask_to_rle(dt_masks[i])
            predictions.append(
                {
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": bbox_xywh,
                    "score": float(dt_scores[i]),
                    "segmentation": seg_rle,
                }
            )

    gt_dataset = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": category_name}],
    }
    coco_gt = COCO()
    coco_gt.dataset = gt_dataset
    coco_gt.createIndex()

    if predictions:
        coco_dt = coco_gt.loadRes(predictions)
    else:
        coco_dt = COCO()
        coco_dt.dataset = {
            "images": images,
            "annotations": [],
            "categories": gt_dataset["categories"],
        }
        coco_dt.createIndex()

    return coco_gt, coco_dt


def run_coco_eval(coco_gt, coco_dt, iou_type: str = "bbox", max_dets: list = None) -> OrderedDict:
    """Run COCOeval and return metrics dict."""
    from pycocotools.cocoeval import COCOeval

    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    if max_dets is not None:
        coco_eval.params.maxDets = max_dets
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    n = len(coco_eval.params.maxDets)
    metric_names = [
        "mAP",
        "mAP_50",
        "mAP_75",
        "mAP_s",
        "mAP_m",
        "mAP_l",
        f"AR@{coco_eval.params.maxDets[0]}",
        f"AR@{coco_eval.params.maxDets[1]}",
        f"AR@{coco_eval.params.maxDets[min(2, n - 1)]}",
        "AR_s",
        "AR_m",
        "AR_l",
    ]
    results = OrderedDict()
    prefix = "bbox" if iou_type == "bbox" else "segm"
    for name, val in zip(metric_names, coco_eval.stats):
        results[f"{prefix}/{name}"] = float(val)
    return results
