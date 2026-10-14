"""Standard AP100 and explicitly named dense-scene diagnostics."""

from .coco_eval_utils import run_coco_eval


def dense_diagnostics(coco_gt, coco_dt, iou_type):
    metrics = run_coco_eval(coco_gt, coco_dt, iou_type=iou_type, max_dets=[1, 10, 300])
    return {
        f"{key.replace('AR@100', 'AR@300')}_max300": value
        for key, value in metrics.items()
    }
