"""Canonical COCO GT/DT construction and COCOeval runner.

Since the 2026-09 consolidation there is ONE implementation of each: the
training-side variants (custom-maxDets primary-AP recompute for the unified
maxDet=100/150 best-model protocol, and pre-computed "rles" fast paths that
release dense 1024x1024 masks during validation). Checkpoint inference and
probes consume the same functions via the default maxDets path.
"""

import logging
from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# NWPU VHR-10 10 categories (COCO instance-mask conversion); train labels
# 0..9 map to ids 1..10 (label+1). Order matches the release annotations.json.
VHR10_CATEGORIES = [
    {"id": 1, "name": "airplane"},
    {"id": 2, "name": "ship"},
    {"id": 3, "name": "storage_tank"},
    {"id": 4, "name": "baseball_diamond"},
    {"id": 5, "name": "tennis_court"},
    {"id": 6, "name": "basketball_court"},
    {"id": 7, "name": "ground_track_field"},
    {"id": 8, "name": "harbor"},
    {"id": 9, "name": "bridge"},
    {"id": 10, "name": "vehicle"},
]


def _mask_to_rle(binary_mask: np.ndarray) -> dict:
    """Convert binary mask (H, W) to COCO RLE format."""
    from pycocotools import mask as mask_util

    mask_fortran = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_util.encode(mask_fortran)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def _masks_to_rles(binary_masks: np.ndarray) -> List[dict]:
    """Batch-encode N binary masks as JSON-safe COCO RLE records."""
    from pycocotools import mask as mask_util

    masks = np.asarray(binary_masks, dtype=np.uint8)
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim != 3:
        raise ValueError(f"Expected masks with shape [N,H,W], got {masks.shape}")
    if len(masks) == 0:
        return []

    # pycocotools batch encode expects [H,W,N] in Fortran order. Encoding here
    # lets validation release dense 1024x1024 masks immediately per image.
    encoded = mask_util.encode(
        np.asfortranarray(np.moveaxis(masks, 0, -1))
    )
    if isinstance(encoded, dict):
        encoded = [encoded]
    rles = []
    for rle in encoded:
        rle = dict(rle)
        counts = rle.get("counts")
        if isinstance(counts, bytes):
            rle["counts"] = counts.decode("utf-8")
        rles.append(rle)
    return rles





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
    score_key: str = "scores",
    categories: Optional[List[dict]] = None,
) -> Tuple:
    """Build pycocotools COCO objects for GT and detections.

    category_id is derived as ``label + 1``.  For single-class runs (WHU)
    labels are all 0 so this keeps the historical ``category_id=1`` behavior;
    for multi-class runs (e.g. VHR-10 10 classes) labels 0..9 map to the
    official category ids 1..10 via ``categories``.
    """
    # Lazy import: the sys.path bootstrap for the project root runs inside
    # main(), so module-level imports of project packages would fail under
    # torchrun.
    from utils.coco_eval_utils import _xyxy_to_xywh
    from pycocotools.coco import COCO

    if categories is None:
        categories = [{"id": 1, "name": "building"}]

    images = []
    annotations = []
    predictions = []
    ann_id = 1

    for img_id, (gt, dt, meta) in enumerate(
        zip(all_gt, all_dt, img_metas_list), start=1
    ):
        h, w = meta["img_shape"][:2]
        images.append({"id": img_id, "height": int(h), "width": int(w)})

        # GT annotations
        gt_bboxes = gt["bboxes"]
        gt_labels = gt["labels"]
        gt_rles = gt.get("rles")
        gt_masks = gt.get("masks")
        if gt_rles is None:
            if hasattr(gt_masks, "masks"):
                gt_masks = gt_masks.masks
            gt_rles = _masks_to_rles(gt_masks)

        num_gt = len(gt_labels)
        for i in range(num_gt):
            bbox_xywh = _xyxy_to_xywh(gt_bboxes[i]).tolist()
            area = float(bbox_xywh[2] * bbox_xywh[3])
            seg_rle = gt_rles[i]
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(gt_labels[i]) + 1,
                    "bbox": bbox_xywh,
                    "area": area,
                    "segmentation": seg_rle,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        # Predictions
        dt_bboxes = dt["bboxes"]
        if score_key not in dt:
            raise KeyError(
                f"Configured COCO score key {score_key!r} is missing from "
                "predictions; refusing a silent score fallback"
            )
        dt_scores = dt[score_key]
        dt_rles = dt.get("rles")
        dt_masks = dt.get("masks")
        if dt_rles is None:
            if hasattr(dt_masks, "masks"):
                dt_masks = dt_masks.masks
            dt_rles = _masks_to_rles(dt_masks)

        num_dt = len(dt_scores)
        dt_labels = dt.get("labels")
        if dt_labels is None:
            dt_labels = np.zeros(len(dt_scores), dtype=np.int64)
        for i in range(num_dt):
            bbox_xywh = _xyxy_to_xywh(dt_bboxes[i]).tolist()
            seg_rle = dt_rles[i]
            predictions.append(
                {
                    "image_id": img_id,
                    "category_id": int(dt_labels[i]) + 1,
                    "bbox": bbox_xywh,
                    "score": float(dt_scores[i]),
                    "segmentation": seg_rle,
                }
            )

    # Build COCO GT
    gt_dataset = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    coco_gt = COCO()
    coco_gt.dataset = gt_dataset
    coco_gt.createIndex()

    # Build COCO DT
    if predictions:
        coco_dt = coco_gt.loadRes(predictions)
    else:
        coco_dt = COCO()
        coco_dt.dataset = {"images": images, "annotations": [], "categories": gt_dataset["categories"]}
        coco_dt.createIndex()

    return coco_gt, coco_dt




def run_coco_eval(coco_gt, coco_dt, iou_type: str = "bbox", max_dets=None) -> OrderedDict:
    """Run COCOeval and return metrics dict."""
    from pycocotools.cocoeval import COCOeval

    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    if max_dets is not None:
        coco_eval.params.maxDets = max_dets
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    if max_dets is not None:
        # summarize() hardcodes maxDets=100 for the primary AP stat and
        # yields -1 when 100 is absent from params.maxDets. Recompute the
        # IoU-averaged AP from the accumulated precision array at the
        # custom cap (no reliance on private summarize helpers).
        try:
            _mind = list(coco_eval.params.maxDets).index(list(max_dets)[-1])
            _prec = coco_eval.eval["precision"]
            _aps = []
            for _k in range(_prec.shape[2]):
                _p = _prec[:, :, _k, 0, _mind]
                _v = _p[_p > -1]
                if _v.size:
                    _aps.append(float(_v.mean()))
            if _aps:
                coco_eval.stats[0] = sum(_aps) / len(_aps)
        except Exception:
            # A silent -1 here would poison best-model selection for the whole
            # run (constant metric -> best never updates); surface it loudly.
            logger.warning(
                "run_coco_eval custom-maxDets recompute failed for "
                "iou_type=%s max_dets=%s; mAP may read -1",
                iou_type,
                list(max_dets),
                exc_info=True,
            )

    metric_names = [
        "mAP", "mAP_50", "mAP_75", "mAP_s", "mAP_m", "mAP_l",
        "AR@1", "AR@10", "AR@100", "AR_s", "AR_m", "AR_l",
    ]
    results = OrderedDict()
    prefix = "bbox" if iou_type == "bbox" else "segm"
    for name, val in zip(metric_names, coco_eval.stats):
        results[f"{prefix}/{name}"] = float(val)
    return results


