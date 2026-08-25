#!/usr/bin/env python

import argparse
import gc
import logging
import math
import os
import sys
from collections import OrderedDict
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler

from mmdet.structures import DetDataSample
from mmdet.structures.mask import BitmapMasks
from mmengine.structures import InstanceData


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("portable_sam_fusion")


# ---------------------------------------------------------------------------
# COCO-style evaluation helpers (from inference_rsprompter_fusion.py)
# ---------------------------------------------------------------------------
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
) -> Tuple:
    """Build pycocotools COCO objects for GT and detections."""
    from pycocotools.coco import COCO

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
                    "category_id": 1,
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
        for i in range(num_dt):
            bbox_xywh = _xyxy_to_xywh(dt_bboxes[i]).tolist()
            seg_rle = dt_rles[i]
            predictions.append(
                {
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": bbox_xywh,
                    "score": float(dt_scores[i]),
                    "segmentation": seg_rle,
                }
            )

    # Build COCO GT
    gt_dataset = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "building"}],
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


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _metric_value(metrics: Dict[str, float], metric_name: str) -> Optional[float]:
    if not metric_name:
        return None
    value = metrics.get(metric_name)
    if value is None:
        return None
    return float(value)


def _weighted_metric_value(metrics: Dict[str, float], spec: str) -> Optional[float]:
    """Parse '0.5*bbox/mAP_75+0.5*segm/mAP_75' style metric expressions."""
    total = 0.0
    found = False
    for raw_part in spec.split("+"):
        part = raw_part.strip()
        if not part:
            continue
        if "*" in part:
            weight_text, metric_name = part.split("*", 1)
            weight = float(weight_text.strip())
            metric_name = metric_name.strip()
        else:
            weight = 1.0
            metric_name = part
        value = _metric_value(metrics, metric_name)
        if value is None:
            return None
        total += weight * value
        found = True
    return total if found else None


def _metric_expression_value(metrics: Dict[str, float], spec: str) -> Optional[float]:
    if "+" in spec or "*" in spec:
        return _weighted_metric_value(metrics, spec)
    return _metric_value(metrics, spec)


def _save_verified_best_checkpoint(
    checkpoint_dir: Path,
    prefix: str,
    score_label: str,
    score_value: float,
    epoch: int,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    best_metrics: Dict,
    history: Dict,
    config_snapshot: Optional[Dict],
    scaler: Optional[Any],
    ema_state: Optional[Dict] = None,
) -> bool:
    new_best = checkpoint_dir / f"{prefix}_epoch{epoch + 1}.pth"
    _save_checkpoint(
        new_best,
        model,
        optimizer,
        scheduler,
        epoch,
        best_metrics,
        history,
        config_snapshot=config_snapshot,
        scaler=scaler,
        ema_state=ema_state,
    )
    try:
        _verify = torch.load(str(new_best), map_location="cpu", weights_only=False)
        assert "model" in _verify, "Missing 'model' key"
        assert "optimizer" in _verify, "Missing 'optimizer' key"
        assert "scheduler" in _verify, "Missing 'scheduler' key"
        assert "config_snapshot" in _verify, "Missing 'config_snapshot' key"
        logger.info("Saved %s with %s: %.4f", prefix, score_label, score_value)
    except Exception as e:
        logger.error("Checkpoint verification FAILED for %s, keeping old best: %s", prefix, e)
        new_best.unlink(missing_ok=True)
        return False

    best_link = checkpoint_dir / f"{prefix}.pth"
    for old in checkpoint_dir.glob(f"{prefix}_epoch*.pth"):
        if old != new_best:
            old.unlink(missing_ok=True)
    best_link.unlink(missing_ok=True)
    best_link.symlink_to(new_best.name)
    return True


def _extract_instances_numpy(
    instances, img_shape: Tuple[int, int], encode_masks: bool = False
) -> dict:
    """Extract InstanceData, optionally replacing dense masks with COCO RLE."""
    bboxes = instances.bboxes
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.detach().cpu().numpy()
    labels = instances.labels
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    scores = None
    if hasattr(instances, "scores"):
        scores = instances.scores
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()

    masks = None
    if hasattr(instances, "masks"):
        masks = instances.masks
        if hasattr(masks, "masks"):
            masks = masks.masks
        elif hasattr(masks, "to_ndarray"):
            masks = masks.to_ndarray()
        elif isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()

    # Ensure masks shape matches img_shape
    h, w = img_shape[:2]
    if masks is not None and len(masks) > 0 and masks.shape[1:] != (h, w):
        resized = []
        for m in masks:
            resized.append(
                cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            )
        masks = np.stack(resized, axis=0)

    if masks is None or len(masks) == 0:
        masks = np.zeros((0, h, w), dtype=np.uint8)

    out = {"bboxes": bboxes, "labels": labels}
    if encode_masks:
        mask_count = len(masks)
        out["rles"] = _masks_to_rles(masks)
        out["mask_count"] = mask_count
        out["mask_fill_sum"] = (
            float(np.count_nonzero(masks)) / float(max(1, h * w))
            if mask_count > 0
            else 0.0
        )
    else:
        out["masks"] = masks
    if scores is not None:
        out["scores"] = scores
    if hasattr(instances, "mask_scores"):
        mask_scores = instances.mask_scores
        if isinstance(mask_scores, torch.Tensor):
            mask_scores = mask_scores.detach().cpu().numpy()
        out["mask_scores"] = mask_scores
    return out


def _summarize_mask_density(all_dt: List[dict]) -> Tuple[float, float]:
    """Return average binary mask fill ratio and masks per validation image."""
    weighted_fill = 0.0
    total_masks = 0
    for dt in all_dt:
        if "mask_count" in dt and "mask_fill_sum" in dt:
            total_masks += int(dt["mask_count"])
            weighted_fill += float(dt["mask_fill_sum"])
            continue
        masks = dt.get("masks")
        if masks is None or len(masks) == 0:
            continue
        pixels_per_mask = int(np.prod(masks.shape[1:]))
        weighted_fill += float(np.count_nonzero(masks)) / max(1, pixels_per_mask)
        total_masks += len(masks)
    return (
        weighted_fill / max(total_masks, 1),
        total_masks / max(len(all_dt), 1),
    )


def _resolve_ddp_timeout_seconds() -> int:
    raw_value = os.environ.get(
        "TORCH_DDP_TIMEOUT_SECONDS", os.environ.get("NCCL_TIMEOUT", "1800")
    )
    try:
        timeout_seconds = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "TORCH_DDP_TIMEOUT_SECONDS must be a positive integer, "
            f"got {raw_value!r}"
        ) from exc
    if timeout_seconds < 1:
        raise ValueError(
            "TORCH_DDP_TIMEOUT_SECONDS must be >= 1, "
            f"got {timeout_seconds}"
        )
    return timeout_seconds


def _resolve_ddp_control_timeout_seconds() -> int:
    raw_value = os.environ.get("TORCH_DDP_CONTROL_TIMEOUT_SECONDS", "86400")
    try:
        timeout_seconds = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "TORCH_DDP_CONTROL_TIMEOUT_SECONDS must be a positive integer, "
            f"got {raw_value!r}"
        ) from exc
    if timeout_seconds < 1:
        raise ValueError(
            "TORCH_DDP_CONTROL_TIMEOUT_SECONDS must be >= 1, "
            f"got {timeout_seconds}"
        )
    return timeout_seconds


def _init_distributed() -> Dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    launched_by_torchrun = os.environ.get("LOCAL_RANK", None) is not None
    should_init = (world_size > 1) or launched_by_torchrun
    if should_init and not dist.is_initialized():
        # Bind each worker before NCCL creates communicators. Initializing NCCL
        # while every process still points at cuda:0 can leave rank 0 with an
        # avoidable memory imbalance.
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=_resolve_ddp_timeout_seconds()),
        )
    distributed = dist.is_initialized()
    control_group = None
    if distributed and world_size > 1:
        # Validation is intentionally rank-0-only and can take longer than the
        # training NCCL timeout. Keep epoch control on CPU/Gloo so idle ranks do
        # not hold an NCCL collective open throughout validation/COCO eval.
        control_group = dist.new_group(
            ranks=list(range(world_size)),
            backend="gloo",
            timeout=timedelta(seconds=_resolve_ddp_control_timeout_seconds()),
        )
    return {
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "distributed": int(distributed),
        "control_group": control_group,
    }


def _broadcast_stop_training(
    stop_training: bool, control_group: Optional[Any]
) -> bool:
    """Synchronize rank-0's epoch decision without occupying NCCL."""
    if not dist.is_initialized():
        return bool(stop_training)
    if control_group is None:
        raise RuntimeError(
            "distributed epoch control requires the CPU/Gloo control group"
        )
    stop_tensor = torch.tensor(
        [1 if stop_training else 0], device="cpu", dtype=torch.int32
    )
    dist.broadcast(stop_tensor, src=0, group=control_group)
    return bool(stop_tensor.item())


def _is_main_process(rank: int) -> bool:
    return rank == 0


def _build_optimizer(
    model: torch.nn.Module,
    lr: float,
    sat_backbone_lr_mult: float,
    sat_other_lr_mult: float,
    bbox_head_lr_mult: float,
    drone_lr_mult: float,
    mask_decoder_lr_mult: float = 0.05,
    no_mask_lr_mult: float = 0.1,
    prompt_encoder_lr_mult: float = 0.1,
    shape_prior_lr_mult: float = 1.0,
    p2_boundary_refiner_lr_mult: float = 1.0,
    quality_head_lr_mult: float = 1.0,
    scene_align_lr_mult: float = 2.0,
    weight_decay: float = 0.05,
) -> optim.Optimizer:
    params_sat_backbone = []
    names_sat_backbone = []
    params_mask_decoder = []
    names_mask_decoder = []
    params_no_mask = []
    names_no_mask = []
    params_prompt_encoder = []
    names_prompt_encoder = []
    params_shape_prior = []
    names_shape_prior = []
    params_p2_boundary_refiner = []
    names_p2_boundary_refiner = []
    params_quality_head = []
    names_quality_head = []
    params_bbox_head = []
    names_bbox_head = []
    params_sat_other = []
    names_sat_other = []
    params_drone = []
    names_drone = []
    params_scene_align = []
    names_scene_align = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        match_name = name.replace("_fsdp_wrapped_module.", "").replace("module.", "")
        if "scene_alignment" in match_name:
            params_scene_align.append(param)
            names_scene_align.append(name)
        elif (
            match_name.startswith("drone_encoder.")
            or match_name.startswith("guidance.")
            or match_name.startswith("bottom_up.")
            or "roi_head.mask_head.uav_roi_" in match_name
            or "roi_head.mask_head.uav_dense_prompt_scale" in match_name
            or "roi_head.mask_head.uav_prompt_adapter_" in match_name
        ):
            params_drone.append(param)
            names_drone.append(name)
        elif match_name.startswith("backbone.") or match_name.startswith("shared_image_embedding."):
            params_sat_backbone.append(param)
            names_sat_backbone.append(name)
        elif (
            "roi_head.mask_head.mask_decoder.decoder.iou_prediction_head" in match_name
            or "roi_head.mask_head.quality_head" in match_name
        ):
            params_quality_head.append(param)
            names_quality_head.append(name)
        elif "roi_head.mask_head.mask_decoder" in match_name:
            params_mask_decoder.append(param)
            names_mask_decoder.append(name)
        elif "roi_head.mask_head.no_mask_embed" in match_name:
            params_no_mask.append(param)
            names_no_mask.append(name)
        elif "roi_head.mask_head.prompt_encoder" in match_name:
            params_prompt_encoder.append(param)
            names_prompt_encoder.append(name)
        elif "roi_head.mask_head.p2_boundary_refiner" in match_name:
            params_p2_boundary_refiner.append(param)
            names_p2_boundary_refiner.append(name)
        elif (
            "roi_head.mask_head.shape_injector" in match_name
            or "roi_head.mask_head.shape_prompt_scale" in match_name
            or "roi_head.mask_head.roi_dense" in match_name
        ):
            params_shape_prior.append(param)
            names_shape_prior.append(name)
        elif (
            bbox_head_lr_mult > 0
            and match_name.startswith("roi_head.bbox_head.")
        ):
            params_bbox_head.append(param)
            names_bbox_head.append(name)
        else:
            params_sat_other.append(param)
            names_sat_other.append(name)

    param_groups = []
    audit_groups = []
    if params_sat_backbone:
        param_groups.append(
            {
                "params": params_sat_backbone,
                "lr": lr * sat_backbone_lr_mult,
                "name": "sat_backbone",
            }
        )
        audit_groups.append(("sat_backbone", params_sat_backbone, names_sat_backbone, lr * sat_backbone_lr_mult, weight_decay))
    if params_mask_decoder:
        param_groups.append(
            {
                "params": params_mask_decoder,
                "lr": lr * mask_decoder_lr_mult,
                "weight_decay": weight_decay,
                "name": "mask_decoder",
            }
        )
        audit_groups.append(("mask_decoder", params_mask_decoder, names_mask_decoder, lr * mask_decoder_lr_mult, weight_decay))
    if params_no_mask:
        param_groups.append(
            {
                "params": params_no_mask,
                "lr": lr * no_mask_lr_mult,
                "weight_decay": weight_decay,
                "name": "no_mask_embed",
            }
        )
        audit_groups.append(("no_mask_embed", params_no_mask, names_no_mask, lr * no_mask_lr_mult, weight_decay))
    if params_prompt_encoder:
        param_groups.append(
            {
                "params": params_prompt_encoder,
                "lr": lr * prompt_encoder_lr_mult,
                "weight_decay": 0.0,
                "name": "prompt_encoder",
            }
        )
        audit_groups.append(("prompt_encoder", params_prompt_encoder, names_prompt_encoder, lr * prompt_encoder_lr_mult, 0.0))
    if params_shape_prior:
        param_groups.append(
            {
                "params": params_shape_prior,
                "lr": lr * shape_prior_lr_mult,
                "name": "shape_prior",
            }
        )
        audit_groups.append(("shape_prior", params_shape_prior, names_shape_prior, lr * shape_prior_lr_mult, weight_decay))
    if params_p2_boundary_refiner:
        param_groups.append(
            {
                "params": params_p2_boundary_refiner,
                "lr": lr * p2_boundary_refiner_lr_mult,
                "weight_decay": weight_decay,
                "name": "p2_boundary_refiner",
            }
        )
        audit_groups.append(
            (
                "p2_boundary_refiner",
                params_p2_boundary_refiner,
                names_p2_boundary_refiner,
                lr * p2_boundary_refiner_lr_mult,
                weight_decay,
            )
        )
    if params_quality_head:
        param_groups.append(
            {
                "params": params_quality_head,
                "lr": lr * quality_head_lr_mult,
                "name": "quality_head",
            }
        )
        audit_groups.append(("quality_head", params_quality_head, names_quality_head, lr * quality_head_lr_mult, weight_decay))
    if params_bbox_head:
        param_groups.append(
            {
                "params": params_bbox_head,
                "lr": lr * bbox_head_lr_mult,
                "name": "bbox_head",
            }
        )
        audit_groups.append(("bbox_head", params_bbox_head, names_bbox_head, lr * bbox_head_lr_mult, weight_decay))
    if params_sat_other:
        param_groups.append(
            {
                "params": params_sat_other,
                "lr": lr * sat_other_lr_mult,
                "name": "sat_other",
            }
        )
        audit_groups.append(("sat_other", params_sat_other, names_sat_other, lr * sat_other_lr_mult, weight_decay))
    if params_drone:
        param_groups.append(
            {"params": params_drone, "lr": lr * drone_lr_mult, "name": "drone_guidance"}
        )
        audit_groups.append(("drone_guidance", params_drone, names_drone, lr * drone_lr_mult, weight_decay))
    if params_scene_align:
        param_groups.append(
            {"params": params_scene_align, "lr": lr * drone_lr_mult * scene_align_lr_mult, "name": "scene_alignment"}
        )
        audit_groups.append(("scene_alignment", params_scene_align, names_scene_align, lr * drone_lr_mult * scene_align_lr_mult, weight_decay))

    if not param_groups:
        raise RuntimeError(
            "No trainable parameters found. Check freezing / requires_grad."
        )

    if int(os.environ.get("RANK", "0")) == 0:
        for group_name, params, names, group_lr, group_wd in audit_groups:
            count = sum(p.numel() for p in params)
            first = names[0] if names else "<none>"
            last = names[-1] if names else "<none>"
            logger.info(
                "Optimizer group %-15s params=%d tensors=%d lr=%.3e wd=%.3e first=%s last=%s",
                group_name,
                count,
                len(params),
                group_lr,
                group_wd,
                first,
                last,
            )

    return optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay, eps=1e-6)


def _set_norm_eval(model: torch.nn.Module):
    for m in model.modules():
        if isinstance(
            m, (torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm, torch.nn.GroupNorm)
        ):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False


def _get_model_core(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap DDP/FSDP to reach the underlying model."""
    return model.module if hasattr(model, "module") else model


def _assert_no_mask_embedding_contract(
    model: torch.nn.Module, context: str
) -> None:
    model_core = _get_model_core(model)
    mask_head = getattr(getattr(model_core, "roi_head", None), "mask_head", None)
    validator = getattr(mask_head, "assert_no_mask_embedding_contract", None)
    if validator is not None:
        validator(context)


class ExponentialMovingAverage:
    """EMA shadow weights used only for eval/export, never for backprop."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        if not (0.0 < decay < 1.0):
            raise ValueError(f"EMA decay must be in (0,1), got {decay}")
        self.decay = float(decay)
        state = _get_model_core(model).state_dict()
        self.ema_state = {k: v.detach().clone() for k, v in state.items()}
        self.backup_state = None

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        model_state = _get_model_core(model).state_dict()
        for key, value in model_state.items():
            value_detached = value.detach()
            if key not in self.ema_state:
                self.ema_state[key] = value_detached.clone()
                continue
            ema_value = self.ema_state[key]
            if torch.is_floating_point(value_detached):
                ema_value.mul_(self.decay).add_(value_detached, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(value_detached)

        stale_keys = [k for k in self.ema_state.keys() if k not in model_state]
        for stale_key in stale_keys:
            self.ema_state.pop(stale_key, None)

    def apply_to(self, model: torch.nn.Module):
        model_core = _get_model_core(model)
        current_state = model_core.state_dict()
        self.backup_state = {k: v.detach().clone() for k, v in current_state.items()}
        model_core.load_state_dict(self.ema_state, strict=False)
        _assert_no_mask_embedding_contract(model_core, "EMA state load")

    def restore(self, model: torch.nn.Module):
        if self.backup_state is None:
            return
        model_core = _get_model_core(model)
        model_core.load_state_dict(self.backup_state, strict=False)
        _assert_no_mask_embedding_contract(model_core, "EMA state restore")
        self.backup_state = None

    def state_dict(self) -> Dict:
        return {
            "decay": self.decay,
            "ema_state": {k: v.detach().clone() for k, v in self.ema_state.items()},
        }

    def load_state_dict(self, state: Dict):
        self.decay = float(state.get("decay", self.decay))
        loaded = state.get("ema_state", {})
        default_device = None
        for tensor in self.ema_state.values():
            default_device = tensor.device
            break

        for key in list(self.ema_state.keys()):
            if key in loaded:
                target = self.ema_state[key]
                source = loaded[key].detach().to(device=target.device, dtype=target.dtype)
                target.copy_(source)
        for key, value in loaded.items():
            if key not in self.ema_state:
                source = value.detach()
                if default_device is not None:
                    source = source.to(device=default_device)
                self.ema_state[key] = source.clone()


def _get_quality_head(model: torch.nn.Module) -> torch.nn.Module:
    model_core = model.module if hasattr(model, "module") else model
    mask_head = getattr(getattr(model_core, "roi_head", None), "mask_head", None)
    quality_head = getattr(mask_head, "quality_head", None)
    if quality_head is None:
        mask_decoder = getattr(mask_head, "mask_decoder", None)
        decoder = getattr(mask_decoder, "decoder", None)
        quality_head = getattr(decoder, "iou_prediction_head", None)
    if quality_head is None or not getattr(mask_head, "quality_head_enabled", False):
        raise RuntimeError("--train-quality-head-only requires QUALITY_HEAD_ENABLED=1")
    return quality_head


def _set_quality_head_only_train_mode(model: torch.nn.Module) -> None:
    """Keep C2 deterministic while training the native SAM2 quality head."""
    model_core = model.module if hasattr(model, "module") else model
    model_core.eval()
    _get_quality_head(model_core).train()


def _log_trainable_state(model: torch.nn.Module):
    model_core = model.module if hasattr(model, "module") else model
    roi_head = getattr(model_core, "roi_head", None)
    mask_head = getattr(roi_head, "mask_head", None)
    checks = {
        "backbone": getattr(model_core, "backbone", None),
        "shared_image_embedding": getattr(model_core, "shared_image_embedding", None),
        "mask_decoder": getattr(mask_head, "mask_decoder", None) if mask_head is not None else None,
        "no_mask_embed": getattr(mask_head, "no_mask_embed", None) if mask_head is not None else None,
        "prompt_encoder": getattr(mask_head, "prompt_encoder", None) if mask_head is not None else None,
        "shape_injector": getattr(mask_head, "shape_injector", None) if mask_head is not None else None,
        "p2_boundary_refiner": getattr(
            mask_head, "p2_boundary_refiner", None
        ) if mask_head is not None else None,
        "quality_head": _get_quality_head(model_core)
        if mask_head is not None and getattr(mask_head, "quality_head_enabled", False)
        else None,
        "uav_roi_retriever": getattr(mask_head, "uav_roi_retriever", None) if mask_head is not None else None,
        "uav_roi_gate": getattr(mask_head, "uav_roi_gate", None) if mask_head is not None else None,
        "uav_roi_dense_prompt": getattr(mask_head, "uav_roi_dense_prompt", None) if mask_head is not None else None,
        "uav_roi_prompt_adapter": getattr(mask_head, "uav_roi_prompt_adapter", None) if mask_head is not None else None,
        "uav_prompt_adapter_sparse_scale": getattr(mask_head, "uav_prompt_adapter_sparse_scale", None) if mask_head is not None else None,
        "uav_prompt_adapter_dense_scale": getattr(mask_head, "uav_prompt_adapter_dense_scale", None) if mask_head is not None else None,
    }
    for name, module in checks.items():
        if module is None:
            logger.info("Trainable state %-22s absent", name)
            continue
        if isinstance(module, torch.nn.Parameter):
            total = module.numel()
            trainable = module.numel() if module.requires_grad else 0
        else:
            params = list(module.parameters())
            total = sum(p.numel() for p in params)
            trainable = sum(p.numel() for p in params if p.requires_grad)
        logger.info("Trainable state %-22s %d / %d", name, trainable, total)


def _log_uav_debug_stats(
    model: torch.nn.Module,
    epoch: int,
    step: int,
    enabled_prefixes: Optional[set] = None,
) -> None:
    model_core = model.module if hasattr(model, "module") else model
    getter = getattr(model_core, "get_prompt_debug_stats", None)
    if getter is None:
        getter = getattr(model_core, "get_uav_debug_stats", None)
    if getter is None:
        return
    stats = getter()
    if not stats:
        return
    groups: Dict[str, Dict[str, float]] = {}
    for key, value in sorted(stats.items()):
        if torch.is_tensor(value):
            if value.numel() != 1:
                continue
            value = float(value.detach().float().item())
        if not isinstance(value, (int, float)):
            continue
        prefix, _, name = key.partition("/")
        if not name:
            prefix, name = "UAV", prefix
        if enabled_prefixes is not None and prefix not in enabled_prefixes:
            continue
        groups.setdefault(prefix, {})[name] = float(value)
    for prefix, values in groups.items():
        payload = " ".join(f"{k}={v:.4g}" for k, v in values.items())
        logger.info("[DEBUG-%s] epoch=%d step=%d %s", prefix, epoch + 1, step, payload)


_DENSE_RESIDUAL_MONITOR_KEYS = (
    "DENSE/residual_alpha",
    "DENSE/source_delta_norm",
    "DENSE/applied_delta_norm",
    "DENSE/source_delta_ratio",
    "DENSE/applied_delta_ratio",
    "DENSE/canvas_mean",
    "DENSE/canvas_std",
    "DENSE/canvas_min",
    "DENSE/canvas_max",
    "GAUSSIAN/valid_ratio",
    "GAUSSIAN/foreground_area_ratio",
    "GAUSSIAN/edt_max",
    "GAUSSIAN/center_x_normalized",
    "GAUSSIAN/center_y_normalized",
    "GAUSSIAN/canvas_area",
)


def _accumulate_dense_residual_monitor(
    model: torch.nn.Module,
    sums: Dict[str, float],
    counts: Dict[str, int],
) -> None:
    """Accumulate detached dense-residual diagnostics from the latest forward."""
    model_core = model.module if hasattr(model, "module") else model
    getter = getattr(model_core, "get_prompt_debug_stats", None)
    if getter is None:
        return
    stats = getter() or {}
    for key in _DENSE_RESIDUAL_MONITOR_KEYS:
        value = stats.get(key)
        if torch.is_tensor(value):
            if value.numel() != 1:
                continue
            value = float(value.detach().float().item())
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        sums[key] += float(value)
        counts[key] += 1


def _reduce_dense_residual_monitor(
    sums: Dict[str, float],
    counts: Dict[str, int],
    device: torch.device,
    distributed: bool,
) -> Dict[str, float]:
    """Return rank-aggregated epoch means for the fixed dense monitor fields."""
    packed = torch.tensor(
        [
            item
            for key in _DENSE_RESIDUAL_MONITOR_KEYS
            for item in (sums[key], float(counts[key]))
        ],
        dtype=torch.float64,
        device=device,
    )
    if distributed and dist.is_initialized():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    result = {}
    for index, key in enumerate(_DENSE_RESIDUAL_MONITOR_KEYS):
        total = float(packed[2 * index].item())
        count = float(packed[2 * index + 1].item())
        if count > 0:
            result[key] = total / count
    return result


_P2BR_EPOCH_FIELDS = (
    "roi_count",
    "valid_support_count",
    "rejected_support_count",
    "search_coverage_sum",
    "delta_abs_sum",
    "delta_nonzero_sum",
    "delta_saturated_sum",
    "projected_feature_norm_sum",
    "highpass_feature_norm_sum",
    "raw_dice_sum",
    "raw_iou_sum",
    "refined_dice_sum",
    "refined_iou_sum",
    "raw_boundary_tp",
    "raw_boundary_fp",
    "raw_boundary_fn",
    "refined_boundary_tp",
    "refined_boundary_fp",
    "refined_boundary_fn",
    "boundary_loss_local_sum",
    "boundary_loss_valid_roi_count",
    "boundary_loss_support_pixels",
    "boundary_loss_foreground_pixels",
)


def _stat_scalar(stats: Dict[str, Any], key: str) -> Optional[float]:
    value = stats.get(key)
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = float(value.detach().float().item())
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return float(value)


def _accumulate_p2br_epoch_monitor(
    model: torch.nn.Module, sums: Dict[str, float]
) -> None:
    model_core = model.module if hasattr(model, "module") else model
    getter = getattr(model_core, "get_prompt_debug_stats", None)
    if getter is None:
        return
    stats = getter() or {}
    roi_count = _stat_scalar(stats, "P2BR/roi_count")
    if roi_count is None:
        return
    direct = {
        "roi_count": "P2BR/roi_count",
        "valid_support_count": "P2BR/valid_support_count",
        "rejected_support_count": "P2BR/rejected_support_count",
        "search_coverage_sum": "P2BR/search_coverage_sum",
        "delta_abs_sum": "P2BR/delta_abs_sum",
        "delta_nonzero_sum": "P2BR/delta_nonzero_sum",
        "delta_saturated_sum": "P2BR/delta_saturated_sum",
        "projected_feature_norm_sum": "P2BR/projected_feature_norm_sum",
        "highpass_feature_norm_sum": "P2BR/highpass_feature_norm_sum",
        "raw_boundary_tp": "COARSE/raw_boundary_tp",
        "raw_boundary_fp": "COARSE/raw_boundary_fp",
        "raw_boundary_fn": "COARSE/raw_boundary_fn",
        "refined_boundary_tp": "COARSE/refined_boundary_tp",
        "refined_boundary_fp": "COARSE/refined_boundary_fp",
        "refined_boundary_fn": "COARSE/refined_boundary_fn",
        "boundary_loss_local_sum": "COARSE/boundary_loss_local_sum",
        "boundary_loss_valid_roi_count": "COARSE/loss_valid_roi_count",
        "boundary_loss_support_pixels": "COARSE/loss_support_pixel_sum",
        "boundary_loss_foreground_pixels": "COARSE/loss_support_foreground_sum",
    }
    for out_key, stat_key in direct.items():
        value = _stat_scalar(stats, stat_key)
        if value is not None:
            sums[out_key] += value
    for out_key, stat_key in (
        ("raw_dice_sum", "COARSE/raw_dice_score"),
        ("raw_iou_sum", "COARSE/raw_iou"),
        ("refined_dice_sum", "COARSE/refined_dice_score"),
        ("refined_iou_sum", "COARSE/refined_iou"),
    ):
        value = _stat_scalar(stats, stat_key)
        if value is not None:
            sums[out_key] += value * roi_count


def _reduce_p2br_epoch_monitor(
    sums: Dict[str, float], device: torch.device, distributed: bool
) -> Dict[str, float]:
    packed = torch.tensor(
        [sums[key] for key in _P2BR_EPOCH_FIELDS],
        dtype=torch.float64,
        device=device,
    )
    if distributed and dist.is_initialized():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    values = {
        key: float(packed[index].item())
        for index, key in enumerate(_P2BR_EPOCH_FIELDS)
    }
    roi_count = values["roi_count"]
    if roi_count <= 0:
        return {}
    valid_count = max(values["valid_support_count"], 1.0)
    loss_count = max(values["boundary_loss_valid_roi_count"], 1.0)

    def _boundary_f1(prefix: str) -> float:
        tp = values[f"{prefix}_boundary_tp"]
        fp = values[f"{prefix}_boundary_fp"]
        fn = values[f"{prefix}_boundary_fn"]
        return (2.0 * tp) / max(2.0 * tp + fp + fn, 1e-12)

    return {
        "roi_count": roi_count,
        "valid_support_ratio": values["valid_support_count"] / roi_count,
        "rejected_support_ratio": values["rejected_support_count"] / roi_count,
        "search_coverage": values["search_coverage_sum"] / roi_count,
        "delta_abs": values["delta_abs_sum"] / roi_count,
        "delta_nonzero_ratio": values["delta_nonzero_sum"] / roi_count,
        "delta_saturated_ratio": values["delta_saturated_sum"] / roi_count,
        "projected_feature_norm": values["projected_feature_norm_sum"] / roi_count,
        "highpass_feature_norm": values["highpass_feature_norm_sum"] / roi_count,
        "raw_dice": values["raw_dice_sum"] / roi_count,
        "raw_iou": values["raw_iou_sum"] / roi_count,
        "refined_dice": values["refined_dice_sum"] / roi_count,
        "refined_iou": values["refined_iou_sum"] / roi_count,
        "raw_boundary_f1": _boundary_f1("raw"),
        "refined_boundary_f1": _boundary_f1("refined"),
        "boundary_loss_raw": values["boundary_loss_local_sum"] / loss_count,
        "boundary_support_fg_ratio": values["boundary_loss_foreground_pixels"]
        / max(values["boundary_loss_support_pixels"], 1.0),
        "valid_count_for_norm": valid_count,
    }


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_metrics: Dict,
    history: Dict,
    config_snapshot: Optional[Dict] = None,
    scaler: Optional[Any] = None,
    ema_state: Optional[Dict] = None,
):
    """Save checkpoint with model, optimizer, scheduler, config, and training state.

    Saves to a temporary file first, then atomically renames to avoid corruption.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    ckpt = {
        "checkpoint_schema_version": int(
            (config_snapshot or {}).get("checkpoint_schema_version", 0)
        ),
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_metrics": best_metrics,
        "history": history,
        "config_snapshot": config_snapshot or {},
        "architecture_contract": (config_snapshot or {}).get(
            "architecture_contract", {}
        ),
    }
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    if ema_state is not None:
        ckpt["ema_state"] = ema_state
    # Atomic save: write to .tmp then rename
    tmp_path = path.with_suffix(".pth.tmp")
    torch.save(ckpt, str(tmp_path))
    tmp_path.replace(path)


def _load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[Any] = None,
    expected_architecture: Optional[Dict] = None,
) -> Dict:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if expected_architecture is not None:
        from rsprompter.architecture_contract import assert_checkpoint_architecture

        assert_checkpoint_architecture(
            ckpt, expected_architecture, operation="RESUME", allow_cross_arch=False
        )
    model_to_load = model.module if hasattr(model, "module") else model
    # 审查 3.5：RESUME 续训时模型结构应与当前运行完全一致，默认用 strict=True。
    # 若 ckpt 是旧结构（新增/删除了模块），strict=True 会失败——此时设置
    # env RESUME_STRICT=0 回退到宽松模式（仅打印 missing/unexpected 明细）。
    _resume_strict = os.environ.get("RESUME_STRICT", "1") != "0"
    if _resume_strict:
        # strict=True 时 load_state_dict 不返回 msg，missing/unexpected 会直接抛错
        model_to_load.load_state_dict(ckpt["model"], strict=True)
        if int(os.environ.get("RANK", "0")) == 0:
            logger.info("RESUME model load: strict=True (all keys matched)")
    else:
        # 问题 13：宽松模式补打 missing/unexpected keys 明细
        _resume_msg = model_to_load.load_state_dict(ckpt["model"], strict=False)
        if int(os.environ.get("RANK", "0")) == 0:
            logger.info(
                "RESUME model load: strict=False, missing=%d, unexpected=%d",
                len(_resume_msg.missing_keys), len(_resume_msg.unexpected_keys),
            )
            if _resume_msg.missing_keys:
                logger.info("  missing (first 20): %s", _resume_msg.missing_keys[:20])
            if _resume_msg.unexpected_keys:
                logger.info("  unexpected (first 20): %s", _resume_msg.unexpected_keys[:20])
    _assert_no_mask_embedding_contract(model_to_load, "RESUME checkpoint load")
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt





def _load_init_checkpoint(
    path: Path,
    model: torch.nn.Module,
    exclude_prefixes: Tuple[str, ...] = (),
    expected_architecture: Optional[Dict] = None,
    allow_cross_arch: bool = False,
) -> Dict[str, object]:
    """Load matching tensors for initialization, with explicit legacy migration."""
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if expected_architecture is not None:
        from rsprompter.architecture_contract import assert_checkpoint_architecture

        assert_checkpoint_architecture(
            ckpt,
            expected_architecture,
            operation="INIT_FROM",
            allow_cross_arch=allow_cross_arch,
        )
    source_state = ckpt.get("model", ckpt)
    model_to_load = model.module if hasattr(model, "module") else model
    target_state = model_to_load.state_dict()
    from rsprompter.ckpt_migration import migrate_legacy_shape_gate_state_dict

    source_state, migrations = migrate_legacy_shape_gate_state_dict(
        source_state, target_state
    )

    compatible = {}
    excluded = 0
    shape_mismatch = 0
    unexpected = 0
    for name, value in source_state.items():
        if any(name.startswith(prefix) for prefix in exclude_prefixes):
            excluded += 1
            continue
        if name not in target_state:
            unexpected += 1
            continue
        if target_state[name].shape != value.shape:
            shape_mismatch += 1
            continue
        compatible[name] = value

    load_result = model_to_load.load_state_dict(compatible, strict=False)
    _assert_no_mask_embedding_contract(model_to_load, "INIT_FROM checkpoint load")
    return {
        "loaded": len(compatible),
        "excluded": excluded,
        "shape_mismatch": shape_mismatch,
        "unexpected": unexpected,
        "missing": len(load_result.missing_keys),
        "migrated": len(migrations),
        "migration_pairs": migrations,
    }


_CANONICAL_DETECTION_LOSS_KEYS = {
    "loss_rpn_cls",
    "loss_rpn_bbox",
    "loss_cls",
    "loss_bbox",
    "loss_bbox_cls",
    "loss_bbox_reg",
}


def _set_model_current_epoch(model, epoch_number: int) -> None:
    """Set 1-based epoch on modules with epoch-dependent schedules."""
    core = model.module if hasattr(model, "module") else model
    seen = set()
    for module in core.modules():
        if id(module) in seen:
            continue
        seen.add(id(module))
        setter = getattr(module, "set_current_epoch", None)
        if callable(setter):
            setter(int(epoch_number))


def _detection_loss_weight(args, epoch_number: int) -> float:
    if epoch_number <= int(args.det_loss_stage1_end):
        return float(args.det_loss_weight_stage1)
    if epoch_number <= int(args.det_loss_stage2_end):
        return float(args.det_loss_weight_stage2)
    return float(args.det_loss_weight_stage3)


def _scale_loss_value(value, scale: float):
    if isinstance(value, torch.Tensor):
        return value * float(scale)
    if isinstance(value, list):
        return [_scale_loss_value(item, scale) for item in value]
    if isinstance(value, tuple):
        return tuple(_scale_loss_value(item, scale) for item in value)
    return value


def _apply_detection_loss_weight(loss_dict: dict, scale: float, device) -> dict:
    """Scale detector losses as one group without changing their ratios."""
    for key in list(loss_dict.keys()):
        if key in _CANONICAL_DETECTION_LOSS_KEYS or key.startswith("loss_rpn_"):
            loss_dict[key] = _scale_loss_value(loss_dict[key], scale)
    loss_dict["schedule/det_weight"] = torch.tensor(
        float(scale), device=device
    )
    return loss_dict


def _materialize_p2_boundary_refiner_loss(
    loss_dict: dict,
    model: torch.nn.Module,
    distributed: bool,
) -> dict:
    """Convert local ROI sums into a four-rank global per-ROI mean.

    DDP averages gradients across ranks. Multiplying the local sum by
    ``world_size / global_count`` therefore yields the exact mean over all
    valid ROIs in the current micro-batch. Gradient accumulation keeps the
    historical equal-micro-batch contract.
    """
    local_sum = loss_dict.pop("_p2br_local_sum", None)
    local_count = loss_dict.pop("_p2br_valid_count", None)
    if local_sum is None and local_count is None:
        return loss_dict
    if not torch.is_tensor(local_sum) or not torch.is_tensor(local_count):
        raise TypeError("P2 boundary refiner loss metadata must be tensors")
    global_count = local_count.detach().float().clone()
    world_size = 1
    if distributed and dist.is_initialized():
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()

    model_core = model.module if hasattr(model, "module") else model
    mask_head = getattr(getattr(model_core, "roi_head", None), "mask_head", None)
    refiner = getattr(mask_head, "p2_boundary_refiner", None)
    if refiner is None:
        raise RuntimeError("received P2 refiner loss metadata without the module")
    if float(global_count.item()) > 0:
        normalized = local_sum * (
            float(world_size) / float(global_count.item())
        )
    else:
        normalized = local_sum * 0.0
    loss_dict["loss_p2_boundary_refiner"] = (
        float(refiner.boundary_loss_weight) * normalized
    )
    return loss_dict


def main():
    parser = argparse.ArgumentParser(
        description="portable_sam_fusion: RSPrompter(SAM) + Drone semantic guidance"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--data-root", type=str, default="/data/wangcheng/dataset/university-test/S"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--checkpoint-dir", type=str, default="/tmp/portable_sam_fusion_ckpts"
    )
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--init-from", type=str, default=None)
    parser.add_argument(
        "--test-only",
        action="store_true",
        default=False,
        help="Load weights via --resume-from and run COCO evaluation once on the "
        "WHU test split (2.2 test/test + test.json), then exit. No training. "
        "Requires --resume-from pointing at a checkpoint with a matching "
        "architecture contract.",
    )
    parser.add_argument(
        "--allow-cross-arch-init",
        action="store_true",
        help="Explicitly allow INIT_FROM from a different or legacy architecture.",
    )
    parser.add_argument(
        "--init-exclude-prefixes",
        type=str,
        default="",
        help="Comma-separated model state prefixes excluded from --init-from",
    )
    parser.add_argument("--gpu-id", type=int, default=0)

    parser.add_argument("--use-drone", action="store_true", default=False)
    parser.add_argument(
        "--use-whu-coco",
        action="store_true",
        default=False,
        help="使用 WHU 公开数据集（COCO 格式，自带 train/validation 划分）",
    )
    parser.add_argument(
        "--subset-ratio",
        type=float,
        default=None,
        help="Deprecated compatibility alias: set both train and val subset ratios.",
    )
    parser.add_argument(
        "--train-subset-ratio",
        type=float,
        default=1.0,
        help="Deterministic fraction of WHU training images to load; default 1.0.",
    )
    parser.add_argument(
        "--val-subset-ratio",
        type=float,
        default=1.0,
        help="Deterministic fraction of WHU validation images to load; default 1.0.",
    )
    parser.add_argument(
        "--drone-data-root",
        type=str,
        default="/data/wangcheng/dataset/university-test/D",
    )
    parser.add_argument("--num-views", type=int, default=15)
    parser.add_argument("--image-size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--drone-image-size", type=int, nargs=2, default=[512, 512])
    parser.add_argument(
        "--random-sample", action="store_true", help="Randomly sample drone images"
    )
    parser.add_argument(
        "--normalize-drone",
        dest="normalize_drone",
        action="store_true",
        help="Apply dataset-side normalization to drone images (WARNING: causes double "
             "normalization because the UAV encoder also normalizes internally).",
    )
    parser.add_argument(
        "--no-normalize-drone",
        dest="normalize_drone",
        action="store_false",
        help="Disable dataset-side normalization (default). The UAV encoder's internal "
             "self.norm handles normalization, so dataset-side normalization is redundant.",
    )
    parser.set_defaults(normalize_drone=False)

    parser.add_argument("--use-fsdp", action="store_true")
    parser.add_argument("--fsdp-sharding", type=str, default="FULL_SHARD")
    parser.add_argument("--fsdp-min-num-params", type=int, default=1000000)

    parser.add_argument("--freeze-bn", action="store_true")
    parser.add_argument(
        "--train-quality-head-only",
        action="store_true",
        help="Freeze C2 and optimize only the native SAM2 IoU quality head.",
    )

    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=1,
        help="Validation batch size (legacy WHU1024 baseline default: 1)",
    )
    parser.add_argument("--val-every-n-epochs", type=int, default=1)
    parser.add_argument(
        "--compute-val-loss",
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            "Whether validation also runs a second loss forward on positive-GT "
            "batches. COCO prediction/evaluation is unaffected."
        ),
    )
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-start-epoch", type=int, default=20)
    parser.add_argument("--early-stopping-min-delta", type=float, default=5e-4)
    parser.add_argument(
        "--early-stopping-smooth-window",
        type=int,
        default=5,
        help="Moving-average window for the resolved early-stopping metric.",
    )
    parser.add_argument("--det-loss-stage1-end", type=int, default=5)
    parser.add_argument("--det-loss-stage2-end", type=int, default=10)
    parser.add_argument("--det-loss-weight-stage1", type=float, default=1.0)
    parser.add_argument("--det-loss-weight-stage2", type=float, default=1.0)
    parser.add_argument("--det-loss-weight-stage3", type=float, default=1.0)
    parser.add_argument(
        "--max-nonfinite-batches",
        type=int,
        default=0,
        help="Abort after this many synchronized non-finite batches; 0 disables abort.",
    )

    parser.add_argument("--sat-backbone-lr-mult", type=float, default=1.0)
    parser.add_argument("--sat-other-lr-mult", type=float, default=1.0)
    parser.add_argument(
        "--bbox-head-lr-mult",
        type=float,
        default=0.0,
        help="Use >0 to place bbox_head in a separate optimizer group",
    )
    parser.add_argument("--drone-lr-mult", type=float, default=0.5)
    parser.add_argument("--mask-decoder-lr-mult", type=float, default=1.0)
    parser.add_argument("--no-mask-lr-mult", type=float, default=1.0)
    parser.add_argument("--prompt-encoder-lr-mult", type=float, default=0.1)
    parser.add_argument("--shape-prior-lr-mult", type=float, default=1.0)
    parser.add_argument("--p2-boundary-refiner-lr-mult", type=float, default=1.0)
    parser.add_argument("--quality-head-lr-mult", type=float, default=1.0)
    parser.add_argument("--scene-align-lr-mult", type=float, default=2.0, help="Learning rate multiplier for scene alignment parameters")
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--max-scenes", type=int, default=1000, help="Maximum number of scenes for scene-specific alignment")
    parser.add_argument("--warmup-epochs", type=int, default=0, help="Number of warmup epochs for learning rate scheduling")
    parser.add_argument("--warmup-iters", type=int, default=100, help="Number of warmup optimizer steps (overrides warmup-epochs if > 0)")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay for optimizer")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate for regularization")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # --- AMP / EMA (训练基础设施) ---
    parser.add_argument("--amp", type=int, default=0, help="Enable AMP mixed precision when set to 1")
    parser.add_argument("--ema-enabled", type=int, default=1, help="Enable EMA shadow weights")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay in (0,1)")
    parser.add_argument("--ema-update-every", type=int, default=1, help="Update EMA every N optimizer steps")
    parser.add_argument("--ema-eval", type=int, default=1, help="Use EMA shadow weights for validation")
    parser.add_argument("--ema-save-best", type=int, default=1, help="Save best_model using EMA shadow weights")
    parser.add_argument("--ema-eval-start-epoch", type=int, default=5, help="Start using EMA for eval after this epoch")
    parser.add_argument("--uav-debug-stats", type=int, choices=[0, 1], default=0,
                        help="Print per-scheme UAV diagnostic stats during training")
    parser.add_argument("--uav-debug-stats-interval", type=int, default=50,
                        help="Training iteration interval for UAV diagnostic stats")
    parser.add_argument("--prompt-debug-stats", type=int, choices=[0, 1], default=0,
                        help="Print shape-point/coarse-mask prompt diagnostics")
    parser.add_argument("--prompt-debug-stats-interval", type=int, default=50,
                        help="Training iteration interval for prompt diagnostics")
    parser.add_argument("--depth-debug-stats", type=int, choices=[0, 1], default=0,
                        help="Print depth module diagnostic stats during training")
    parser.add_argument("--depth-debug-stats-interval", type=int, default=50,
                        help="Training iteration interval for depth diagnostic stats")
    # UAV 双流 A/B/C/D 消融参数。默认 None 表示尊重 config/env 中的值；
    # shell 脚本传入后会在这里覆盖 cfg, 方便同一份 config 切实验。
    parser.add_argument("--uav-fpn-enabled", type=int, choices=[0, 1], default=None,
                        help="方案 A: 是否启用 UAV BEV -> satellite FPN deformable fusion")
    parser.add_argument("--uav-fpn-use-bottom-up", type=int, choices=[0, 1], default=None,
                        help="方案 A: deformable fusion 后是否启用 bottom-up PAFPN 聚合")
    parser.add_argument("--uav-fpn-num-heads", type=int, default=None,
                        help="方案 A: FPN deformable fusion 的 attention head 数")
    parser.add_argument("--uav-fpn-num-points", type=int, default=None,
                        help="方案 A: 每个 head 在 UAV BEV 上采样的点数")
    parser.add_argument("--uav-fpn-max-offset", type=float, default=None,
                        help="方案 A: deformable sampling 相对偏移上限")
    parser.add_argument("--uav-align-loss-weight", type=float, default=None,
                        help="方案 A: UAV/卫星几何对齐辅助损失权重")
    parser.add_argument("--uav-concentration-weight", type=float, default=None,
                        help="方案 A: 采样点集中度辅助损失权重")
    parser.add_argument("--uav-roi-retrieval-enabled", type=int, choices=[0, 1], default=None,
                        help="B/C/D 公共前置: ROI 级 UAV BEV deformable retrieval")
    parser.add_argument("--uav-roi-gate-enabled", type=int, choices=[0, 1], default=None,
                        help="方案 B: 用 UAV ROI context 对 mask_head ROI feature 做 FiLM gate")
    parser.add_argument("--uav-roi-dense-prompt-enabled", type=int, choices=[0, 1], default=None,
                        help="方案 C: 用 UAV ROI dense prompt 替代 shape prior dense prompt")
    parser.add_argument(
        "--uav-roi-dense-combine",
        type=str,
        default=None,
        choices=["exclusive", "replace"],
        help="方案 C: 当前仅支持 replace/exclusive, 都表示替换 shape prior 而非融合",
    )
    parser.add_argument("--uav-roi-bev-dim", type=int, default=None,
                        help="B/C/D retrieval 输入 BEV 通道数, 默认 128 BEV + 1 height/depth = 129")
    parser.add_argument("--uav-roi-d-inner", type=int, default=None,
                        help="B/C/D retrieval 内部 context 维度")
    parser.add_argument("--uav-roi-num-heads", type=int, default=None,
                        help="B/C/D retrieval 的 attention head 数")
    parser.add_argument("--uav-roi-num-points", type=int, default=None,
                        help="B/C/D retrieval 每个 head 采样点数")
    parser.add_argument("--uav-roi-max-offset", type=float, default=None,
                        help="B/C/D retrieval ROI 中心附近的最大归一化采样偏移")
    parser.add_argument("--uav-roi-multi-scale", type=int, choices=[0, 1], default=None,
                        help="B retrieval: 构造多层 BEV 金字塔并按 ROI 尺度选层")
    parser.add_argument("--uav-roi-level-match", type=int, choices=[0, 1], default=None,
                        help="B retrieval: 是否用 ROI 尺度匹配 BEV 层级")
    parser.add_argument("--uav-roi-num-bev-levels", type=int, default=None,
                        help="B retrieval: 多层 BEV 金字塔层数")
    parser.add_argument("--uav-roi-finest-scale", type=float, default=None,
                        help="B retrieval: ROI 分配到 P4 的尺度基准")
    parser.add_argument("--uav-roi-offset-loss-weight", type=float, default=None,
                        help="B retrieval: offset 范数辅助约束权重")
    parser.add_argument("--uav-roi-attn-uniform-loss-weight", type=float, default=None,
                        help="B retrieval: attention 防塌缩辅助约束权重")
    parser.add_argument("--uav-roi-dense-scale-init", type=float, default=None,
                        help="方案 C dense prompt scale 初值, 0 表示严格热启动")
    parser.add_argument("--uav-roi-dense-mid-channels", type=int, default=None,
                        help="方案 C dense prompt 小头的中间通道数")
    parser.add_argument("--uav-roi-prompt-adapter-enabled", type=int, choices=[0, 1], default=None,
                        help="方案 D: BEV-conditioned prompt adapter before SAM2 mask decoder")
    parser.add_argument("--uav-roi-prompt-adapter-hidden-dim", type=int, default=None,
                        help="方案 D prompt adapter MLP hidden dim")
    parser.add_argument("--uav-roi-prompt-adapter-sparse-scale-init", type=float, default=None,
                        help="方案 D sparse prompt delta scale 初值, 0 表示严格热启动")
    parser.add_argument("--uav-roi-prompt-adapter-dense-scale-init", type=float, default=None,
                        help="方案 D dense prompt delta scale 初值, 0 表示严格热启动")
    parser.add_argument("--uav-roi-prompt-adapter-use-confidence", type=int, choices=[0, 1], default=None,
                        help="方案 D: 是否用 confidence gate 缩放 BEV adapter delta")

    args = parser.parse_args()

    if int(args.det_loss_stage1_end) < 1:
        raise ValueError("det_loss_stage1_end must be >= 1")
    if int(args.det_loss_stage2_end) <= int(args.det_loss_stage1_end):
        raise ValueError(
            "det_loss_stage2_end must be greater than det_loss_stage1_end"
        )
    for _name in (
        "det_loss_weight_stage1",
        "det_loss_weight_stage2",
        "det_loss_weight_stage3",
    ):
        if float(getattr(args, _name)) < 0:
            raise ValueError(f"{_name} must be non-negative")
    if int(args.early_stopping_start_epoch) < 1:
        raise ValueError("early_stopping_start_epoch must be >= 1")
    if int(args.early_stopping_patience) < 0:
        raise ValueError("early_stopping_patience must be >= 0")
    if float(args.early_stopping_min_delta) < 0:
        raise ValueError("early_stopping_min_delta must be >= 0")
    if int(args.early_stopping_smooth_window) < 1:
        raise ValueError("early_stopping_smooth_window must be >= 1")
    if int(args.val_batch_size) < 1:
        raise ValueError("val_batch_size must be >= 1")
    if int(args.val_every_n_epochs) < 1:
        raise ValueError("val_every_n_epochs must be >= 1")

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    import mmdet.models
    from mmengine.config import Config
    from mmengine.registry import MODELS as MMENGINE_MODELS
    from mmdet.registry import MODELS
    from mmdet.models.data_preprocessors import DetDataPreprocessor

    if "DetDataPreprocessor" not in MMENGINE_MODELS:
        MMENGINE_MODELS.register_module(
            name="DetDataPreprocessor", module=DetDataPreprocessor
        )

    import rsprompter

    # === 固定随机种子 (可复现) ===
    import random as _random
    _random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # CUDNN_BENCHMARK=1 trades bit-level reproducibility for conv autotuning
    # on fixed-size inputs (public runners keep the deterministic default).
    cudnn_benchmark = os.environ.get("CUDNN_BENCHMARK", "0") == "1"
    torch.backends.cudnn.deterministic = not cudnn_benchmark
    torch.backends.cudnn.benchmark = cudnn_benchmark
    if int(os.environ.get("RANK", "0")) == 0:
        print(
            f"[seed] Random seed fixed to {args.seed} "
            f"| cudnn.deterministic={not cudnn_benchmark} "
            f"| cudnn.benchmark={cudnn_benchmark}"
        )

    dist_info = _init_distributed()
    distributed = bool(dist_info.get("distributed", 0))
    rank = int(dist_info["rank"])
    control_group = dist_info.get("control_group")

    if torch.cuda.is_available():
        device = (
            f"cuda:{dist_info['local_rank']}" if distributed else f"cuda:{args.gpu_id}"
        )
    else:
        device = "cpu"

    is_main = _is_main_process(rank)
    checkpoint_dir = Path(args.checkpoint_dir)
    if is_main:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "DDP process-group timeout: %d seconds",
            _resolve_ddp_timeout_seconds(),
        )
        if control_group is not None:
            logger.info(
                "DDP validation control: backend=gloo timeout=%d seconds",
                _resolve_ddp_control_timeout_seconds(),
            )

    cfg = Config.fromfile(args.config)
    from rsprompter.architecture_contract import architecture_contract

    resolved_architecture = architecture_contract(cfg.model.to_dict())
    sam2_checkpoint_path = os.environ.get("SAM2_CKPT", "")
    sam2_identity = [sam2_checkpoint_path, ""]
    if is_main and sam2_checkpoint_path:
        from rsprompter.architecture_contract import file_sha256

        sam2_identity[1] = file_sha256(sam2_checkpoint_path)
    if distributed and dist.is_initialized():
        dist.broadcast_object_list(
            sam2_identity, src=0, group=control_group
        )
    if is_main:
        logger.info(
            "Architecture contract: id=%s schema=%d fingerprint=%s",
            resolved_architecture["architecture_id"],
            resolved_architecture["schema_version"],
            resolved_architecture["model_fingerprint"],
        )

    # B/C/D 都依赖 ROI retrieval；C 还依赖 SAM2 PromptEncoder 的 masks= 通道。
    # 这里再做一次联动兜底, 避免只改顶层镜像或只改 shell 参数导致半开配置。
    if hasattr(cfg.model, 'max_scenes'):
        cfg.model.max_scenes = args.max_scenes
    model = MODELS.build(cfg.model)
    model.to(device)

    exclusive_scopes = sum(
        int(flag)
        for flag in (
            args.train_quality_head_only,
        )
    )
    if exclusive_scopes > 1:
        raise RuntimeError(
            "Experimental module-only training scopes are mutually exclusive"
        )
    if args.train_quality_head_only:
        model.requires_grad_(False)
        quality_head = _get_quality_head(model)
        quality_head.requires_grad_(True)
        _set_quality_head_only_train_mode(model)
        if is_main:
            model_core = model.module if hasattr(model, "module") else model
            quality_mask_head = getattr(
                getattr(model_core, "roi_head", None), "mask_head", None
            )
            quality_type = getattr(
                quality_mask_head, "quality_head_type", "native_iou"
            )
            logger.info(
                "Training scope: quality head only (%s); loaded C2 baseline is frozen in eval mode",
                quality_type,
            )

    model_for_preproc = model.module if hasattr(model, "module") else model
    data_preprocessor = model_for_preproc.data_preprocessor
    evaluation_mask_head = model_for_preproc.roi_head.mask_head
    segm_score_mode = getattr(evaluation_mask_head, "segm_score_mode", "detector")
    segm_score_key = getattr(evaluation_mask_head, "segm_score_key", "scores")
    if segm_score_mode not in {"detector", "mask_quality"}:
        raise RuntimeError(f"Invalid segmentation score mode: {segm_score_mode!r}")
    if segm_score_key not in {"scores", "mask_scores"}:
        raise RuntimeError(f"Invalid segmentation score key: {segm_score_key!r}")
    if is_main:
        logger.info(
            "Evaluation score contract: bbox=scores segm=%s mode=%s",
            segm_score_key,
            segm_score_mode,
        )

    enable_drone_branch = getattr(model_for_preproc, "enable_drone_branch", False)
    if args.use_drone and not enable_drone_branch:
        logger.warning(
            "--use-drone is set but model has enable_drone_branch=False. "
            "Disabling drone data loading."
        )
        args.use_drone = False
    if enable_drone_branch and not args.use_drone:
        logger.info(
            "Model has enable_drone_branch=True but --use-drone not set. "
            "Will use drone branch if drone data is available."
        )

    if args.freeze_bn:
        _set_norm_eval(model)

    if distributed and not args.use_fsdp:
        from torch.nn.parallel import DistributedDataParallel as DDP

        # find_unused_parameters=True：PE freeze_all=True 后有参数不收梯度，
        # 且 mask_head 的 ROI 数量随 batch 动态变化，必须开启避免 DDP hang。
        model = DDP(
            model,
            device_ids=[int(device.split(":")[1])]
            if device.startswith("cuda")
            else None,
            find_unused_parameters=True,
        )

    if distributed and args.use_fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import ShardingStrategy

        sharding = getattr(
            ShardingStrategy, args.fsdp_sharding, ShardingStrategy.FULL_SHARD
        )
        ignored_modules = None
        model_base = model.module if hasattr(model, "module") else model
        if (
            getattr(model_base, "freeze_drone", False)
            and hasattr(model_base, "drone_encoder")
            and model_base.drone_encoder is not None
        ):
            model_base.drone_encoder.requires_grad_(False)
            model_base.drone_encoder.eval()
            ignored_modules = [model_base.drone_encoder]

        model = FSDP(
            model,
            sharding_strategy=sharding,
            device_id=int(device.split(":")[1]) if device.startswith("cuda") else None,
            ignored_modules=ignored_modules,
            use_orig_params=True,
            min_num_params=int(args.fsdp_min_num_params),
        )

    from data import create_train_loader

    train_subset_ratio = float(args.train_subset_ratio)
    val_subset_ratio = float(args.val_subset_ratio)
    if args.subset_ratio is not None:
        if args.train_subset_ratio != 1.0 or args.val_subset_ratio != 1.0:
            raise ValueError(
                "--subset-ratio cannot be combined with --train-subset-ratio "
                "or --val-subset-ratio"
            )
        train_subset_ratio = val_subset_ratio = float(args.subset_ratio)
    for name, value in (
        ("train_subset_ratio", train_subset_ratio),
        ("val_subset_ratio", val_subset_ratio),
    ):
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be in (0, 1], got {value}")

    train_loader, val_loader, train_dataset = (
        create_train_loader(
            data_root=args.data_root,
            batch_size=args.batch_size,
            # Horizontal flip is safe for dual-stream after fix:
            # only intrinsics cx is adjusted (cx -> W-cx); extrinsics are unchanged.
            flip_prob=0.5,
            vflip_prob=0.0,
            gaussian_noise_prob=0.0,
            gaussian_noise_std=0.02,
            random_erasing_prob=0.0,
            random_erasing_scale=(0.02, 0.15),
            random_erasing_ratio=(0.3, 3.3),
            image_size=tuple(args.image_size),
            use_drone=args.use_drone,
            drone_data_root=args.drone_data_root,
            num_views=args.num_views,
            drone_image_size=tuple(args.drone_image_size),
            distributed=distributed,
            rank=rank,
            world_size=int(dist_info.get("world_size", 1)),
            val_ratio=args.val_ratio,
            val_batch_size=args.val_batch_size,
            random_sample=args.random_sample,
            normalize_drone=args.normalize_drone,
            dataset_format="whu_coco" if args.use_whu_coco else "labelme",
            whu_train_ann_file=os.environ.get(
                "WHU_TRAIN_ANN_FILE", "2.4 annotation/annotation/train.json"
            ),
            whu_val_ann_file=os.environ.get(
                "WHU_VAL_ANN_FILE", "2.4 annotation/annotation/validation.json"
            ),
            whu_train_img_subdir=os.environ.get(
                "WHU_TRAIN_IMG_SUBDIR", "2.1 train/train"
            ),
            whu_val_img_subdir=os.environ.get(
                "WHU_VAL_IMG_SUBDIR", "2.3 valid/validation"
            ),
            train_subset_ratio=train_subset_ratio,
            val_subset_ratio=val_subset_ratio,
            seed=int(args.seed),
        )
    )
    # Preserve the proven WHU1024 baseline validation contract: rank 0 runs
    # the complete validation loader while the other DDP ranks wait at the
    # existing end-of-epoch synchronization point.  Iterating the unsharded
    # loader on every rank duplicates full-resolution mask accumulation and
    # can exhaust host memory.
    if distributed and not is_main:
        val_loader = None
    if is_main:
        logger.info(
            "Validation policy: rank0_only=%d batch_size=%d every_n_epochs=%d compute_loss=%d",
            int(distributed),
            int(args.val_batch_size),
            int(args.val_every_n_epochs),
            int(args.compute_val_loss),
        )

    num_batches_per_epoch = len(train_loader)
    if args.max_train_batches > 0:
        num_batches_per_epoch = min(num_batches_per_epoch, args.max_train_batches)

    # 问题 10：scheduler 必须按 optimizer step 计数，而不是 mini-batch 数。
    # 之前 cosine_t_max / warmup_iters 都按 mini-batch 算，但 scheduler.step() 只在
    # optimizer.step() 后调用一次（grad_accum 组才一次），导致 grad_accum>1 时
    # cosine 永远走不到 eta_min、warmup 需要 accum 倍 epoch 才结束。
    grad_accum_for_sched = max(1, int(args.grad_accum_steps))
    optimizer_steps_per_epoch = math.ceil(num_batches_per_epoch / grad_accum_for_sched)
    total_optimizer_steps = max(1, args.epochs * optimizer_steps_per_epoch)

    if is_main:
        logger.info(
            "Dataset size: %d | mini-batch/epoch=%d | grad_accum=%d | "
            "optimizer_steps/epoch=%d | total_optimizer_steps=%d",
            len(train_dataset), num_batches_per_epoch, grad_accum_for_sched,
            optimizer_steps_per_epoch, total_optimizer_steps,
        )

    optimizer = _build_optimizer(
        model.module if hasattr(model, "module") else model,
        lr=args.lr,
        sat_backbone_lr_mult=args.sat_backbone_lr_mult,
        sat_other_lr_mult=args.sat_other_lr_mult,
        bbox_head_lr_mult=args.bbox_head_lr_mult,
        drone_lr_mult=args.drone_lr_mult,
        mask_decoder_lr_mult=args.mask_decoder_lr_mult,
        no_mask_lr_mult=args.no_mask_lr_mult,
        prompt_encoder_lr_mult=args.prompt_encoder_lr_mult,
        shape_prior_lr_mult=args.shape_prior_lr_mult,
        p2_boundary_refiner_lr_mult=args.p2_boundary_refiner_lr_mult,
        quality_head_lr_mult=args.quality_head_lr_mult,
        scene_align_lr_mult=args.scene_align_lr_mult,
        weight_decay=args.weight_decay,
    )
    if is_main:
        _log_trainable_state(model)

    # 问题 10：warmup / cosine 全部按 optimizer step 计数（与 scheduler.step() 调用频率一致）。
    # --warmup-iters 现在表示 optimizer step 数（而非 mini-batch），语义更准确；
    # grad_accum==1 时 optimizer step == mini-batch，行为与旧版完全一致。
    if args.warmup_iters > 0:
        warmup_optimizer_steps = args.warmup_iters
        if is_main:
            logger.info(f"Using user-specified warmup: {warmup_optimizer_steps} optimizer steps")
    elif args.warmup_epochs > 0:
        warmup_optimizer_steps = args.warmup_epochs * optimizer_steps_per_epoch
        if is_main:
            logger.info(f"Using epoch-based warmup: {args.warmup_epochs} epochs "
                        f"({warmup_optimizer_steps} optimizer steps)")
    else:
        warmup_optimizer_steps = min(50, optimizer_steps_per_epoch)
        if is_main:
            logger.info(f"Using default warmup: {warmup_optimizer_steps} optimizer steps")

    cosine_t_max = max(1, total_optimizer_steps - warmup_optimizer_steps)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_t_max, eta_min=args.lr * 0.001
    )
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.001, total_iters=warmup_optimizer_steps
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_optimizer_steps],
    )
    # 保留旧变量名供 config_snapshot 引用（向后兼容）
    warmup_iters = warmup_optimizer_steps

    # AMP scaler (--amp 0 关闭混合精度，用 fp32 训练)
    amp_enabled = bool(args.amp) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    # === 构建训练 config 快照 (保存到 checkpoint, 推理时可直接加载) ===
    eval_bbox = _env_flag("EVAL_BBOX", "1")
    # Historical WHU1024 baseline reports/selects bbox/mAP as its primary bbox
    # metric.  Keep mAP_75 in the detailed metrics, but do not silently use it
    # as the headline or best-bbox checkpoint criterion.
    save_bbox_best_metric = os.environ.get("SAVE_BBOX_BEST_METRIC", "bbox/mAP")
    save_composite_best = _env_flag("SAVE_COMPOSITE_BEST", "0")
    composite_metric_spec = os.environ.get(
        "SAVE_COMPOSITE_WEIGHTS", "0.5*bbox/mAP_75+0.5*segm/mAP_75"
    )
    early_stopping_metric_spec = os.environ.get(
        "EARLY_STOPPING_METRIC", "segm/mAP"
    )
    resolved_mask_head_cfg = cfg.model.roi_head.mask_head
    resolved_point_warmup = dict(
        resolved_mask_head_cfg.get("point_warmup_cfg", {})
    )
    resolved_shape_point_miner = dict(
        resolved_mask_head_cfg.get("shape_point_miner_cfg", {})
    )
    resolved_coarse_loss = dict(
        resolved_mask_head_cfg.get("coarse_mask_loss_cfg", {})
    )
    resolved_shape_prior = dict(
        resolved_mask_head_cfg.get("shape_prior_cfg", {})
    )
    config_snapshot = {
        "checkpoint_schema_version": 2,
        "architecture_contract": dict(resolved_architecture),
        "config_path": str(args.config),
        "training_args": dict(vars(args)),
        "data_config": {
            "dataset_format": "whu_coco" if args.use_whu_coco else "labelme",
            "data_root": str(args.data_root),
            "image_size": list(args.image_size),
            "single_class": True,
            "validation": {
                "ann_file": "2.4 annotation/annotation/validation.json",
                "image_subdir": "2.3 valid/validation",
            },
            "test": {
                "ann_file": "2.4 annotation/annotation/test.json",
                "image_subdir": "2.2 test/test",
            },
        },
        "runtime_config": {
            "sam2_repo": os.environ.get("SAM2_REPO", ""),
            "sam2_checkpoint": os.environ.get("SAM2_CKPT", ""),
            "sam2_checkpoint_sha256": sam2_identity[1],
            "sam2_model_size": os.environ.get("SAM2_MODEL_SIZE", ""),
            "ddp_timeout_seconds": _resolve_ddp_timeout_seconds(),
            "ddp_control_backend": "gloo" if control_group is not None else "none",
            "ddp_control_timeout_seconds": _resolve_ddp_control_timeout_seconds(),
        },
        "train_subset_ratio": float(train_subset_ratio),
        "val_subset_ratio": float(val_subset_ratio),
        "prompt_generator_mode": os.environ.get(
            "PROMPT_GENERATOR_MODE", "explicit_mask"
        ),
        "explicit_prompt_mode": os.environ.get("EXPLICIT_PROMPT_MODE", ""),
        "prompt_sparse_mode": os.environ.get("PROMPT_SPARSE_MODE", ""),
        "prompt_encoder_enabled": os.environ.get("PROMPT_ENCODER_ENABLED", ""),
        "shape_prior_enabled": os.environ.get("SHAPE_PRIOR_ENABLED", ""),
        "explicit_use_box_prompt": os.environ.get("EXPLICIT_USE_BOX_PROMPT", ""),
        "use_shape_dense": os.environ.get("USE_SHAPE_DENSE", ""),
        "prompt_mode": os.environ.get("PROMPT_MODE", ""),
        "neck_type": os.environ.get("NECK_TYPE", ""),
        "prompt_dense_mode": os.environ.get("PROMPT_DENSE_MODE", ""),
        "eval_bbox": int(eval_bbox),
        "save_bbox_best_metric": save_bbox_best_metric,
        "save_composite_best": int(save_composite_best),
        "composite_metric_spec": composite_metric_spec,
        "early_stopping_metric_spec": early_stopping_metric_spec,
        "early_stopping_patience": int(args.early_stopping_patience),
        "early_stopping_start_epoch": int(args.early_stopping_start_epoch),
        "early_stopping_min_delta": float(args.early_stopping_min_delta),
        "early_stopping_smooth_window": int(
            args.early_stopping_smooth_window
        ),
        "det_loss_schedule": {
            "stage1_end": int(args.det_loss_stage1_end),
            "stage2_end": int(args.det_loss_stage2_end),
            "stage1_weight": float(args.det_loss_weight_stage1),
            "stage2_weight": float(args.det_loss_weight_stage2),
            "stage3_weight": float(args.det_loss_weight_stage3),
        },
        "point_warmup": {
            "enabled": bool(resolved_point_warmup.get("enabled", False)),
            "no_point_epochs": int(
                resolved_point_warmup.get("no_point_epochs", 0)
            ),
            "one_pair_epochs": int(
                resolved_point_warmup.get("one_pair_epochs", 0)
            ),
            "full_2p2n_start_epoch": int(
                resolved_point_warmup.get("full_2p2n_start_epoch", 1)
            ),
        },
        "shape_point_mining": {
            "adaptive_validity": bool(
                resolved_shape_point_miner.get("adaptive_validity", False)
            ),
        },
        "shape_loss_schedule": {
            "mode": resolved_coarse_loss.get("schedule_mode", "fixed"),
            "fixed_weight": float(resolved_coarse_loss.get("weight", 0.0)),
            "ranges": [
                dict(item)
                for item in resolved_coarse_loss.get("weight_schedule", [])
            ],
        },
        "shape_dense_gate": {
            "mode": resolved_shape_prior.get("shape_scale_mode", ""),
            "init": resolved_shape_prior.get("prompt_scale_init", ""),
        },
        "freeze_decoder": os.environ.get("FREEZE_DECODER", ""),
        "freeze_no_mask": os.environ.get("FREEZE_NO_MASK", ""),
        "shape_prior_loss_weight": float(
            resolved_mask_head_cfg.get("shape_prior_loss_weight", 0.0)
        ),
        "p2_boundary_refiner": dict(
            resolved_mask_head_cfg.get("p2_boundary_refiner_cfg", {})
        ),
        "train_quality_head_only": int(args.train_quality_head_only),
        "quality_head_enabled": os.environ.get("QUALITY_HEAD_ENABLED", "0"),
        "quality_head_type": os.environ.get("QUALITY_HEAD_TYPE", "native_iou"),
        "quality_head_loss_weight": os.environ.get("QUALITY_HEAD_LOSS_WEIGHT", "1.0"),
        "quality_head_loss_beta": os.environ.get("QUALITY_HEAD_LOSS_BETA", "0.1"),
        "quality_head_score_alpha": os.environ.get("QUALITY_HEAD_SCORE_ALPHA", "0.5"),
        "quality_head_hidden_dim": os.environ.get("QUALITY_HEAD_HIDDEN_DIM", "256"),
        "quality_head_detach_inputs": os.environ.get("QUALITY_HEAD_DETACH_INPUTS", "1"),
        "bbox_arch": os.environ.get("AP75_BBOX_ARCH", ""),
        "bbox_iou_loss_weight": os.environ.get("AP75_BBOX_IOU_LOSS_WEIGHT", ""),
        "init_from": args.init_from or "",
        "init_exclude_prefixes": args.init_exclude_prefixes,
        "uav_fpn_enabled": args.uav_fpn_enabled,
        "uav_fpn_use_bottom_up": args.uav_fpn_use_bottom_up,
        "uav_fpn_num_heads": args.uav_fpn_num_heads,
        "uav_fpn_num_points": args.uav_fpn_num_points,
        "uav_fpn_max_offset": args.uav_fpn_max_offset,
        "uav_align_loss_weight": args.uav_align_loss_weight,
        "uav_concentration_weight": args.uav_concentration_weight,
        "uav_roi_retrieval_enabled": args.uav_roi_retrieval_enabled,
        "uav_roi_gate_enabled": args.uav_roi_gate_enabled,
        "uav_roi_dense_prompt_enabled": args.uav_roi_dense_prompt_enabled,
        "uav_roi_dense_combine": args.uav_roi_dense_combine,
        "uav_roi_prompt_adapter_enabled": args.uav_roi_prompt_adapter_enabled,
        "uav_roi_prompt_adapter_hidden_dim": args.uav_roi_prompt_adapter_hidden_dim,
        "uav_roi_prompt_adapter_sparse_scale_init": args.uav_roi_prompt_adapter_sparse_scale_init,
        "uav_roi_prompt_adapter_dense_scale_init": args.uav_roi_prompt_adapter_dense_scale_init,
        "uav_roi_prompt_adapter_use_confidence": args.uav_roi_prompt_adapter_use_confidence,
        "uav_roi_bev_dim": args.uav_roi_bev_dim,
        "uav_roi_d_inner": args.uav_roi_d_inner,
        "uav_roi_num_heads": args.uav_roi_num_heads,
        "uav_roi_num_points": args.uav_roi_num_points,
        "uav_roi_max_offset": args.uav_roi_max_offset,
        "uav_roi_multi_scale": args.uav_roi_multi_scale,
        "uav_roi_level_match": args.uav_roi_level_match,
        "uav_roi_num_bev_levels": args.uav_roi_num_bev_levels,
        "uav_roi_finest_scale": args.uav_roi_finest_scale,
        "uav_roi_offset_loss_weight": args.uav_roi_offset_loss_weight,
        "uav_roi_attn_uniform_loss_weight": args.uav_roi_attn_uniform_loss_weight,
        "uav_roi_dense_scale_init": args.uav_roi_dense_scale_init,
        "uav_roi_dense_mid_channels": args.uav_roi_dense_mid_channels,
        "uav_debug_stats": args.uav_debug_stats,
        "uav_debug_stats_interval": args.uav_debug_stats_interval,
        "depth_debug_stats": args.depth_debug_stats,
        "depth_debug_stats_interval": args.depth_debug_stats_interval,
        "depth_enabled": os.environ.get("DEPTH_ENABLED", ""),
        "depth_encoder": os.environ.get("DEPTH_ENCODER", ""),
        "depth_pretrained_path": os.environ.get("DEPTH_PRETRAINED_PATH", ""),
        "depth_require_pretrained": os.environ.get("DEPTH_REQUIRE_PRETRAINED", ""),
        "use_drone": args.use_drone,
        "seed": args.seed,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "epochs": args.epochs,
        "warmup_iters": warmup_iters,
        # 问题 10：scheduler 真实进度依据（optimizer step 维度）
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "total_optimizer_steps": total_optimizer_steps,
        "effective_batch_size": args.batch_size * args.grad_accum_steps * max(1, int(os.environ.get("WORLD_SIZE", "1"))),
        "weight_decay": args.weight_decay,
        "lr_mults": {
            "sat_backbone": args.sat_backbone_lr_mult,
            "sat_other": args.sat_other_lr_mult,
            "bbox_head": args.bbox_head_lr_mult,
            "mask_decoder": args.mask_decoder_lr_mult,
            "no_mask": args.no_mask_lr_mult,
            "prompt_encoder": args.prompt_encoder_lr_mult,
            "shape_prior": args.shape_prior_lr_mult,
            "p2_boundary_refiner": args.p2_boundary_refiner_lr_mult,
            "drone": args.drone_lr_mult,
            "scene_align": args.scene_align_lr_mult,
        },
        "image_size": list(args.image_size),
        "mask_head_config": dict(cfg.model.roi_head.mask_head)
            if hasattr(cfg.model.roi_head, "mask_head") else {},
        "neck_config": dict(cfg.model.get("neck", {})),
        # 完整解析后的 cfg.model (含 type/backbone/roi_head/...): 推理时可据此直接重建模型,
        # 无需再传 .py config 或重设环境变量。新字段, 旧 ckpt 没有时推理脚本自动 fallback。
        "model_config": cfg.model.to_dict(),
    }

    start_epoch = 0
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    best_segm_map = 0.0  # Track best segm mAP
    best_bbox_score = 0.0
    best_composite_score = 0.0
    best_early_stopping_score = float("-inf")
    early_counter = 0
    early_stopping_metric_history = []
    history = {"epochs": [], "train_losses": [], "val_losses": [], "val_metrics": [], "learning_rates": []}

    # EMA shadow weights (eval/save-best only, never for backprop)
    ema = None
    ema_update_steps = 0
    ema_enabled = bool(args.ema_enabled) and (0.0 < float(args.ema_decay) < 1.0)
    if ema_enabled:
        ema = ExponentialMovingAverage(model, decay=float(args.ema_decay))
        if is_main:
            logger.info(
                "EMA enabled: decay=%.4f update_every=%d eval=%d save_best=%d eval_start_epoch=%d",
                float(args.ema_decay), int(args.ema_update_every),
                int(args.ema_eval), int(args.ema_save_best), int(args.ema_eval_start_epoch),
            )

    if args.resume_from and args.init_from:
        raise ValueError("--resume-from and --init-from are mutually exclusive")

    if args.init_from:
        exclude_prefixes = tuple(
            prefix.strip()
            for prefix in args.init_exclude_prefixes.split(",")
            if prefix.strip()
        )
        init_stats = _load_init_checkpoint(
            Path(args.init_from),
            model,
            exclude_prefixes=exclude_prefixes,
            expected_architecture=resolved_architecture,
            allow_cross_arch=args.allow_cross_arch_init,
        )
        if is_main:
            logger.info(
                "Initialized model from %s loaded=%d excluded=%d shape_mismatch=%d unexpected=%d missing=%d migrated=%d prefixes=%s",
                args.init_from,
                init_stats["loaded"],
                init_stats["excluded"],
                init_stats["shape_mismatch"],
                init_stats["unexpected"],
                init_stats["missing"],
                init_stats["migrated"],
                exclude_prefixes or "<none>",
            )
            if init_stats.get("migration_pairs"):
                logger.info(
                    "Checkpoint key migration (INIT_FROM): %s",
                    init_stats["migration_pairs"],
                )

    if args.resume_from:
        ckpt = _load_checkpoint(
            Path(args.resume_from),
            model,
            optimizer,
            scheduler,
            scaler,
            expected_architecture=resolved_architecture,
        )
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best = ckpt.get("best_metrics", {}) or {}
        best_val_loss = float(best.get("best_val_loss", best_val_loss))
        best_train_loss = float(best.get("best_train_loss", best_train_loss))
        best_segm_map = float(best.get("best_segm_map", best_segm_map))
        best_bbox_score = float(best.get("best_bbox_score", best_bbox_score))
        best_composite_score = float(best.get("best_composite_score", best_composite_score))
        best_early_stopping_score = float(
            best.get("best_early_stopping_score", best_early_stopping_score)
        )
        early_counter = int(best.get("early_counter", early_counter))
        early_stopping_metric_history = list(
            best.get(
                "early_stopping_metric_history",
                early_stopping_metric_history,
            )
        )
        history = ckpt.get("history", history) or history
        if ema is not None and "ema_state" in ckpt:
            ema.load_state_dict(ckpt["ema_state"])
            if is_main:
                logger.info("Restored EMA shadow weights from checkpoint")
        # 恢复 config 快照 (从 checkpoint 加载, 保证续训一致)
        if "config_snapshot" in ckpt:
            saved_snapshot = ckpt["config_snapshot"]
            if isinstance(saved_snapshot, dict):
                # Preserve the resumed run's recorded values while retaining
                # new schema fields for checkpoints produced by newer code.
                config_snapshot = {**config_snapshot, **saved_snapshot}
                config_snapshot["checkpoint_schema_version"] = 2
                config_snapshot["architecture_contract"] = dict(
                    resolved_architecture
                )
        if is_main:
            logger.info(
                "Resumed from %s (start_epoch=%d, best_segm_map=%.4f, best_bbox_score=%.4f, best_composite_score=%.4f)",
                args.resume_from,
                start_epoch,
                best_segm_map,
                best_bbox_score,
                best_composite_score,
            )

    # ===== --test-only: evaluate on the held-out WHU test split, then exit =====
    if args.test_only:
        if not args.resume_from:
            raise SystemExit("--test-only requires --resume-from to load weights.")
        from data import create_test_loader

        test_loader = create_test_loader(
            data_root=args.data_root,
            ann_file="2.4 annotation/annotation/test.json",
            image_subdir="2.2 test/test",
            image_size=tuple(args.image_size),
            batch_size=int(args.val_batch_size),
            num_workers=4,
            seed=int(args.seed),
        )
        # Non-main ranks skip iteration to match the rank0_only validation
        # contract (no NCCL collective held open during COCO eval).
        if distributed and not is_main:
            test_loader = None

        model.eval()
        if args.freeze_bn:
            _set_norm_eval(model)

        all_gt: List[dict] = []
        all_dt: List[dict] = []
        all_img_metas: List[dict] = []

        if test_loader is not None:
            if is_main:
                logger.info(
                    "Test-only evaluation: checkpoint=%s, test split=%d images",
                    args.resume_from,
                    len(test_loader.dataset),
                )
            model_for_eval = model.module if hasattr(model, "module") else model
            # Runtime-only max_per_img override for dense scenes (>100
            # instances per tile). Mutates the live test_cfg objects, so the
            # config file and the architecture fingerprint stay untouched.
            _test_max_per_img = int(os.environ.get("TEST_MAX_PER_IMG", "0") or 0)
            if _test_max_per_img > 0:
                _model_test_cfg = getattr(model_for_eval, "test_cfg", None)
                if _model_test_cfg is not None and "rcnn" in _model_test_cfg:
                    _model_test_cfg["rcnn"]["max_per_img"] = _test_max_per_img
                _roi_test_cfg = getattr(
                    getattr(model_for_eval, "roi_head", None), "test_cfg", None
                )
                if _roi_test_cfg is not None and "max_per_img" in _roi_test_cfg:
                    _roi_test_cfg["max_per_img"] = _test_max_per_img
                if is_main:
                    logger.info(
                        "TEST_MAX_PER_IMG=%d override applied (model-side cap)",
                        _test_max_per_img,
                    )
            _test_eval_max_dets = (
                [1, 10, _test_max_per_img] if _test_max_per_img > 0 else None
            )
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_loader):
                    if args.max_val_batches > 0 and batch_idx >= args.max_val_batches:
                        break
                    imgs = batch["imgs"].to(device)
                    img_metas = batch["img_metas"]
                    gt_bboxes = [b.to(device) for b in batch["gt_bboxes"]]
                    gt_labels = [l.to(device) for l in batch["gt_labels"]]
                    gt_masks = batch["gt_masks"]

                    data_samples = []
                    for i in range(len(imgs)):
                        ds = DetDataSample()
                        ds.set_metainfo(img_metas[i])
                        gt_instances = InstanceData()
                        num_inst = len(gt_labels[i])
                        if num_inst > 0:
                            valid_mask = gt_labels[i] >= 0
                            num_valid = int(valid_mask.sum().item())
                            if num_valid > 0:
                                valid_indices = (
                                    valid_mask.nonzero().squeeze(-1)[:num_valid].cpu()
                                )
                                gt_instances.bboxes = gt_bboxes[i][valid_indices]
                                gt_instances.labels = gt_labels[i][valid_indices]
                                masks = gt_masks[i][valid_indices.cpu().numpy()]
                                gt_instances.masks = BitmapMasks(
                                    masks, *img_metas[i]["img_shape"]
                                )
                            else:
                                gt_instances.bboxes = torch.zeros(
                                    (0, 4), dtype=torch.float32, device=device
                                )
                                gt_instances.labels = torch.zeros(
                                    (0,), dtype=torch.int64, device=device
                                )
                                gt_instances.masks = BitmapMasks(
                                    np.zeros(
                                        (0, *img_metas[i]["img_shape"]), dtype=np.uint8
                                    ),
                                    *img_metas[i]["img_shape"],
                                )
                        else:
                            gt_instances.bboxes = torch.zeros(
                                (0, 4), dtype=torch.float32, device=device
                            )
                            gt_instances.labels = torch.zeros(
                                (0,), dtype=torch.int64, device=device
                            )
                            gt_instances.masks = BitmapMasks(
                                np.zeros(
                                    (0, *img_metas[i]["img_shape"]), dtype=np.uint8
                                ),
                                *img_metas[i]["img_shape"],
                            )
                        ds.gt_instances = gt_instances
                        data_samples.append(ds.to(device))

                    # Mirror the validation loop: run the model's
                    # data_preprocessor (padding/normalization) before predict.
                    processed = model_for_eval.data_preprocessor(
                        {"inputs": imgs, "data_samples": data_samples},
                        training=False,
                    )
                    model_inputs = processed["inputs"]
                    data_samples = processed["data_samples"]

                    outputs = model_for_eval.predict(
                        model_inputs, data_samples, rescale=False
                    )
                    for j, output in enumerate(outputs):
                        meta = img_metas[j]
                        img_shape = meta["img_shape"]
                        gt_inst = data_samples[j].gt_instances
                        gt_np = _extract_instances_numpy(
                            gt_inst, img_shape, encode_masks=True
                        )
                        all_gt.append(gt_np)
                        pred_inst = (
                            output.pred_instances
                            if hasattr(output, "pred_instances")
                            else output
                        )
                        pred_np = _extract_instances_numpy(
                            pred_inst, img_shape, encode_masks=True
                        )
                        all_dt.append(pred_np)
                        all_img_metas.append(meta)

        if is_main:
            total_gt = sum(len(g["labels"]) for g in all_gt)
            total_dt = sum(len(d["scores"]) for d in all_dt if "scores" in d)
            avg_mask_fill, avg_masks_per_image = _summarize_mask_density(all_dt)
            logger.info(
                "Test samples: GT=%d, DT=%d, avg_masks_per_image=%.2f, avg_mask_fill=%.4f",
                total_gt,
                total_dt,
                avg_masks_per_image,
                avg_mask_fill,
            )
            if total_gt > 0:
                coco_gt, coco_bbox_dt = build_coco_gt_and_dt(
                    all_gt, all_dt, all_img_metas, score_key="scores"
                )
                if eval_bbox:
                    bbox_metrics = run_coco_eval(
                        coco_gt, coco_bbox_dt, iou_type="bbox",
                        max_dets=_test_eval_max_dets,
                    )
                    logger.info(
                        "Test bbox/mAP: %.4f bbox/mAP_75: %.4f",
                        bbox_metrics.get("bbox/mAP", 0.0),
                        bbox_metrics.get("bbox/mAP_75", 0.0),
                    )
                if segm_score_key == "scores":
                    coco_segm_dt = coco_bbox_dt
                else:
                    _, coco_segm_dt = build_coco_gt_and_dt(
                        all_gt,
                        all_dt,
                        all_img_metas,
                        score_key=segm_score_key,
                    )
                segm_metrics = run_coco_eval(
                    coco_gt, coco_segm_dt, iou_type="segm",
                    max_dets=_test_eval_max_dets,
                )
                logger.info(
                    "Test segm/mAP: %.4f segm/mAP_50: %.4f segm/mAP_75: %.4f segm/mAP_s: %.4f segm/mAP_m: %.4f",
                    segm_metrics.get("segm/mAP", 0.0),
                    segm_metrics.get("segm/mAP_50", 0.0),
                    segm_metrics.get("segm/mAP_75", 0.0),
                    segm_metrics.get("segm/mAP_s", 0.0),
                    segm_metrics.get("segm/mAP_m", 0.0),
                )

                # Full COCO stat dump: every metric key of both evals, plus a
                # test_metrics.json next to the checkpoint dir for scripting.
                _full_test_metrics = {}
                if eval_bbox:
                    _full_test_metrics.update(bbox_metrics)
                _full_test_metrics.update(segm_metrics)
                for _mk, _mv in _full_test_metrics.items():
                    logger.info("Test metric %s: %.6f", _mk, float(_mv))
                try:
                    import json as _json

                    _metrics_path = checkpoint_dir / "test_metrics.json"
                    with _metrics_path.open("w", encoding="utf-8") as _mh:
                        _json.dump(
                            {
                                "checkpoint": str(args.resume_from),
                                "test_eval_max_dets": _test_eval_max_dets,
                                "metrics": _full_test_metrics,
                            },
                            _mh,
                            indent=2,
                        )
                    logger.info("Full test metrics written: %s", _metrics_path)
                except Exception:
                    logger.warning("Failed to write test_metrics.json", exc_info=True)

                # Optional prediction visualization (TEST_VIS_OUT=<dir>).
                # Records reuse the infer_from_checkpoint predictions.json
                # contract (model-frame RLE + absolute file_name), which
                # scripts/visualize_instances.py pred mode rescales onto the
                # stored images automatically.
                _vis_out = os.environ.get("TEST_VIS_OUT", "").strip()
                if _vis_out:
                    _vis_thr = float(os.environ.get("TEST_VIS_SCORE_THR", "0.3"))
                    _vis_limit = int(os.environ.get("TEST_VIS_LIMIT", "50"))
                    _vis_records = []
                    for _dt, _meta in zip(all_dt, all_img_metas):
                        _fn = str(
                            _meta.get("img_path") or _meta.get("filename") or ""
                        )
                        _rles = _dt.get("rles")
                        if _rles is None and _dt.get("masks") is not None:
                            _masks = _dt["masks"]
                            if hasattr(_masks, "masks"):
                                _masks = _masks.masks
                            _rles = _masks_to_rles(_masks)
                        for _bb, _sc, _rl in zip(
                            _dt["bboxes"], _dt["scores"], _rles or []
                        ):
                            if float(_sc) < _vis_thr:
                                continue
                            _counts = _rl["counts"]
                            if isinstance(_counts, bytes):
                                _counts = _counts.decode("utf-8")
                            _vis_records.append(
                                {
                                    "file_name": _fn,
                                    "category_id": 1,
                                    "bbox": [
                                        float(_bb[0]),
                                        float(_bb[1]),
                                        float(_bb[2] - _bb[0]),
                                        float(_bb[3] - _bb[1]),
                                    ],
                                    "score": float(_sc),
                                    "segmentation": {
                                        "size": _rl["size"],
                                        "counts": _counts,
                                    },
                                }
                            )
                    _vis_dir = Path(_vis_out).expanduser().resolve()
                    _vis_dir.mkdir(parents=True, exist_ok=True)
                    _pred_json = _vis_dir / "predictions.json"
                    import json as _vis_json

                    with _pred_json.open("w", encoding="utf-8") as _vh:
                        _vis_json.dump(_vis_records, _vh)
                    logger.info(
                        "Test visualization records: %d (score>=%.2f) -> %s",
                        len(_vis_records),
                        _vis_thr,
                        _pred_json,
                    )
                    import subprocess
                    import sys as _sys

                    _project_root = Path(__file__).resolve().parents[1]
                    _vis_cmd = [
                        _sys.executable,
                        "scripts/visualize_instances.py",
                        "pred",
                        "--pred-json",
                        str(_pred_json),
                        "--out-dir",
                        str(_vis_dir / "images"),
                        "--limit",
                        str(_vis_limit),
                    ]
                    logger.info("Visualization: %s", " ".join(_vis_cmd))
                    subprocess.run(
                        _vis_cmd, check=False, cwd=str(_project_root)
                    )
            else:
                logger.warning("No valid GT in test split; skipping COCO eval.")
            logger.info("Test-only evaluation complete. Exiting (no training).")
        # Ensure all ranks reach the exit together.
        if dist_info.get("world_size", 1) > 1:
            dist.barrier()
        return

    nonfinite_batches = 0
    # Validation order is deterministic within a run. Cache compact GT RLE
    # records after the first validation instead of rebuilding them every epoch.
    cached_val_gt_records: Optional[List[dict]] = None
    for epoch in range(start_epoch, args.epochs):
        epoch_number = epoch + 1
        _set_model_current_epoch(model, epoch_number)
        det_loss_weight = _detection_loss_weight(args, epoch_number)
        if distributed:
            sampler = getattr(train_loader, "sampler", None)
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)

        model.train()
        if args.train_quality_head_only:
            _set_quality_head_only_train_mode(model)
        elif args.freeze_bn:
            _set_norm_eval(model)

        total_loss = 0.0
        loss_meter = {}
        dense_monitor_sums = {key: 0.0 for key in _DENSE_RESIDUAL_MONITOR_KEYS}
        dense_monitor_counts = {key: 0 for key in _DENSE_RESIDUAL_MONITOR_KEYS}
        p2br_monitor_sums = {key: 0.0 for key in _P2BR_EPOCH_FIELDS}
        num_batches = 0
        grad_accum = args.grad_accum_steps
        train_batches_limit = len(train_loader)
        if args.max_train_batches > 0:
            train_batches_limit = min(train_batches_limit, args.max_train_batches)
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(train_loader):
            if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
                break

            imgs = batch["imgs"].to(device)
            img_metas = batch["img_metas"]
            gt_bboxes = [b.to(device) for b in batch["gt_bboxes"]]
            gt_labels = [l.to(device) for l in batch["gt_labels"]]
            gt_masks = batch["gt_masks"]

            # Prepare data_samples and move to device
            data_samples: List[DetDataSample] = []
            for i in range(len(imgs)):
                ds = DetDataSample()
                ds.set_metainfo(img_metas[i])
                gt_instances = InstanceData()
                num_inst = len(gt_labels[i])
                if num_inst > 0:
                    valid_mask = gt_labels[i] >= 0
                    num_valid = int(valid_mask.sum().item())
                    if num_valid > 0:
                        valid_indices = (
                            valid_mask.nonzero().squeeze(-1)[:num_valid].cpu()
                        )
                        gt_instances.bboxes = gt_bboxes[i][valid_indices]
                        gt_instances.labels = gt_labels[i][valid_indices]
                        masks = gt_masks[i][valid_indices.cpu().numpy()]
                        gt_instances.masks = BitmapMasks(
                            masks, *img_metas[i]["img_shape"]
                        )
                    else:
                        gt_instances.bboxes = torch.zeros(
                            (0, 4), dtype=torch.float32, device=device
                        )
                        gt_instances.labels = torch.zeros(
                            (0,), dtype=torch.int64, device=device
                        )
                        gt_instances.masks = BitmapMasks(
                            np.zeros((0, *img_metas[i]["img_shape"]), dtype=np.uint8),
                            *img_metas[i]["img_shape"],
                        )
                else:
                    gt_instances.bboxes = torch.zeros(
                        (0, 4), dtype=torch.float32, device=device
                    )
                    gt_instances.labels = torch.zeros(
                        (0,), dtype=torch.int64, device=device
                    )
                    gt_instances.masks = BitmapMasks(
                        np.zeros((0, *img_metas[i]["img_shape"]), dtype=np.uint8),
                        *img_metas[i]["img_shape"],
                    )
                ds.gt_instances = gt_instances
                data_samples.append(ds.to(device))

            # Match the historical WHU1024 baseline: empty-GT training batches
            # still enter the detector loss. A rank-local early continue here
            # would make DDP ranks execute different collective sequences when
            # their DistributedSampler shards contain different empty images.

            # Synchronize data preprocessing for both images and ground truth
            processed = data_preprocessor(
                {"inputs": imgs, "data_samples": data_samples}, training=True
            )
            imgs = processed["inputs"]
            data_samples = processed["data_samples"]

            model_inputs = imgs

            model_for_loss = model.module if hasattr(model, "module") else model

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                loss_dict = model_for_loss.loss(model_inputs, data_samples)
                loss_dict = _materialize_p2_boundary_refiner_loss(
                    loss_dict, model, distributed
                )
                loss_dict = _apply_detection_loss_weight(
                    loss_dict, det_loss_weight, device
                )
                loss_parts = []
                for k, v in loss_dict.items():
                    if "loss" not in k:
                        continue
                    if isinstance(v, torch.Tensor):
                        if not args.train_quality_head_only or v.requires_grad:
                            loss_parts.append(v)
                    elif isinstance(v, (list, tuple)):
                        loss_parts.extend(
                            x
                            for x in v
                            if isinstance(x, torch.Tensor)
                            and (not args.train_quality_head_only or x.requires_grad)
                        )
                loss = sum(loss_parts) if loss_parts else torch.tensor(0.0, device=device)
                # 审查 3.6 / 问题 10：loss 归一化按当前 batch 所属 accumulation group 的
                # 真实长度，而非固定 grad_accum。之前只在最后一个 batch 修正分母，导致
                # 末组前面的 batch（如 batch_idx=4,5，末组真实只有 3 个）仍除以 grad_accum，
                # 该组总梯度被低估。
                group_start = (batch_idx // grad_accum) * grad_accum
                group_end = min(group_start + grad_accum, train_batches_limit)
                current_group_size = max(1, group_end - group_start)
                loss_for_backward = loss / float(current_group_size)

            # Record detailed losses
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    val = v.detach().item()
                    loss_meter[k] = loss_meter.get(k, 0.0) + val
                elif isinstance(v, list):
                    # Handle case where loss might be a list of tensors (though rare in final output dict)
                    val = sum(
                        x.detach().item() for x in v if isinstance(x, torch.Tensor)
                    )
                    loss_meter[k] = loss_meter.get(k, 0.0) + val
            _accumulate_dense_residual_monitor(
                model, dense_monitor_sums, dense_monitor_counts
            )
            _accumulate_p2br_epoch_monitor(model, p2br_monitor_sums)

            finite_flag = torch.isfinite(loss).to(dtype=torch.int32)
            if distributed and dist.is_initialized():
                dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
            if finite_flag.item() == 0:
                nonfinite_batches += 1
                if is_main:
                    nonfinite_parts = [
                        name
                        for name, value in loss_dict.items()
                        if isinstance(value, torch.Tensor)
                        and (not args.train_quality_head_only or value.requires_grad)
                        and not torch.isfinite(value).all()
                    ]
                    logger.error(
                        "Non-finite loss detected (epoch=%d count=%d parts=%s).",
                        epoch + 1,
                        nonfinite_batches,
                        nonfinite_parts or "remote-rank",
                    )
                optimizer.zero_grad(set_to_none=True)
                if (
                    args.max_nonfinite_batches > 0
                    and nonfinite_batches >= args.max_nonfinite_batches
                ):
                    raise FloatingPointError(
                        "Non-finite batch limit reached: "
                        f"{nonfinite_batches}/{args.max_nonfinite_batches}"
                    )
                continue

            global_step = epoch * max(1, train_batches_limit) + batch_idx + 1
            if (
                is_main
                and (
                    (
                        args.uav_debug_stats
                        and args.uav_debug_stats_interval > 0
                        and global_step % args.uav_debug_stats_interval == 0
                    )
                    or (
                        args.prompt_debug_stats
                        and args.prompt_debug_stats_interval > 0
                        and global_step % args.prompt_debug_stats_interval == 0
                    )
                    or (
                        args.depth_debug_stats
                        and args.depth_debug_stats_interval > 0
                        and global_step % args.depth_debug_stats_interval == 0
                    )
                )
            ):
                enabled_prefixes = set()
                if (
                    args.uav_debug_stats
                    and args.uav_debug_stats_interval > 0
                    and global_step % args.uav_debug_stats_interval == 0
                ):
                    enabled_prefixes.update({"A", "ROI", "B", "C", "D"})
                if (
                    args.prompt_debug_stats
                    and args.prompt_debug_stats_interval > 0
                    and global_step % args.prompt_debug_stats_interval == 0
                ):
                    enabled_prefixes.update({"SP", "COARSE", "JITTER", "GATE", "DENSE", "CONTEXT", "BOX"})
                if (
                    args.depth_debug_stats
                    and args.depth_debug_stats_interval > 0
                    and global_step % args.depth_debug_stats_interval == 0
                ):
                    enabled_prefixes.add("DEPTH")
                _log_uav_debug_stats(model, epoch, global_step, enabled_prefixes)
            scaler.scale(loss_for_backward).backward()

            # Step optimizer every grad_accum batches or at the last batch
            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == train_batches_limit:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
                ema_update_steps += 1
                if ema is not None and (ema_update_steps % max(1, int(args.ema_update_every)) == 0):
                    ema.update(model)
                optimizer.zero_grad(set_to_none=True)
                # Step scheduler after optimizer step (iteration-level)
                scheduler.step()
                # 问题 10：每 50 个 optimizer step 打印进度，便于确认 cosine/warmup 走对
                if is_main and ema_update_steps % 50 == 0:
                    _cur_lr = optimizer.param_groups[0]["lr"]
                    logger.info(
                        "epoch %d | mini-batch %d/%d | optimizer_step %d/%d | lr=%.3e",
                        epoch + 1, batch_idx + 1, train_batches_limit,
                        ema_update_steps, total_optimizer_steps, _cur_lr,
                    )

            total_loss += float(loss.detach().item())
            num_batches += 1
        avg_loss = total_loss / max(1, num_batches)
        # Collect LRs for all groups
        lr_groups = {}
        for group in optimizer.param_groups:
            name = group.get("name", "unknown")
            lr_groups[name] = group["lr"]

        dense_monitor_epoch = _reduce_dense_residual_monitor(
            dense_monitor_sums,
            dense_monitor_counts,
            device=device,
            distributed=distributed,
        )
        p2br_monitor_epoch = _reduce_p2br_epoch_monitor(
            p2br_monitor_sums, device=device, distributed=distributed
        )

        if is_main:
            loss_str = []
            for k, v in loss_meter.items():
                avg_k = v / max(1, num_batches)
                loss_str.append(f"{k}={avg_k:.4f}")
            logger.info(f"Epoch {epoch + 1} detailed losses: {', '.join(loss_str)}")
            if dense_monitor_epoch:
                logger.info(
                    "Epoch %d dense residual monitor: alpha=%.6f "
                    "source_delta_norm=%.4f applied_delta_norm=%.4f "
                    "source_delta_ratio=%.6f applied_delta_ratio=%.6f",
                    epoch_number,
                    dense_monitor_epoch.get("DENSE/residual_alpha", float("nan")),
                    dense_monitor_epoch.get("DENSE/source_delta_norm", float("nan")),
                    dense_monitor_epoch.get("DENSE/applied_delta_norm", float("nan")),
                    dense_monitor_epoch.get("DENSE/source_delta_ratio", float("nan")),
                    dense_monitor_epoch.get("DENSE/applied_delta_ratio", float("nan")),
                )
                if "GAUSSIAN/valid_ratio" in dense_monitor_epoch:
                    logger.info(
                        "Epoch %d Gaussian dense prompt: valid=%.2f%% "
                        "fg_area=%.4f edt_max=%.4f center=(%.4f,%.4f) "
                        "canvas_area=%.4f canvas[min/mean/max]=%.4f/%.4f/%.4f",
                        epoch_number,
                        100.0 * dense_monitor_epoch["GAUSSIAN/valid_ratio"],
                        dense_monitor_epoch.get(
                            "GAUSSIAN/foreground_area_ratio", float("nan")
                        ),
                        dense_monitor_epoch.get("GAUSSIAN/edt_max", float("nan")),
                        dense_monitor_epoch.get(
                            "GAUSSIAN/center_x_normalized", float("nan")
                        ),
                        dense_monitor_epoch.get(
                            "GAUSSIAN/center_y_normalized", float("nan")
                        ),
                        dense_monitor_epoch.get(
                            "GAUSSIAN/canvas_area", float("nan")
                        ),
                        dense_monitor_epoch.get("DENSE/canvas_min", float("nan")),
                        dense_monitor_epoch.get("DENSE/canvas_mean", float("nan")),
                        dense_monitor_epoch.get("DENSE/canvas_max", float("nan")),
                    )
            if p2br_monitor_epoch:
                logger.info(
                    "Epoch %d P2 boundary refiner: raw_dice=%.4f refined_dice=%.4f "
                    "raw_iou=%.4f refined_iou=%.4f raw_boundary_f1=%.4f "
                    "refined_boundary_f1=%.4f",
                    epoch_number,
                    p2br_monitor_epoch["raw_dice"],
                    p2br_monitor_epoch["refined_dice"],
                    p2br_monitor_epoch["raw_iou"],
                    p2br_monitor_epoch["refined_iou"],
                    p2br_monitor_epoch["raw_boundary_f1"],
                    p2br_monitor_epoch["refined_boundary_f1"],
                )
                logger.info(
                    "Epoch %d P2 boundary support: valid=%.2f%% rejected=%.2f%% "
                    "coverage=%.4f delta_abs=%.6f nonzero=%.2f%% saturated=%.2f%% "
                    "p2_norm=%.4f highpass_norm=%.4f support_fg=%.2f%%",
                    epoch_number,
                    100.0 * p2br_monitor_epoch["valid_support_ratio"],
                    100.0 * p2br_monitor_epoch["rejected_support_ratio"],
                    p2br_monitor_epoch["search_coverage"],
                    p2br_monitor_epoch["delta_abs"],
                    100.0 * p2br_monitor_epoch["delta_nonzero_ratio"],
                    100.0 * p2br_monitor_epoch["delta_saturated_ratio"],
                    p2br_monitor_epoch["projected_feature_norm"],
                    p2br_monitor_epoch["highpass_feature_norm"],
                    100.0 * p2br_monitor_epoch["boundary_support_fg_ratio"],
                )
            averaged_components = {
                key: value / max(1, num_batches) for key, value in loss_meter.items()
            }
            det_component = sum(
                value for key, value in averaged_components.items()
                if key in _CANONICAL_DETECTION_LOSS_KEYS or key.startswith("loss_rpn_")
            )
            final_mask_component = float(averaged_components.get("loss_mask", 0.0))
            shape_component = float(averaged_components.get("loss_shape_prior", 0.0))
            refiner_component = float(
                averaged_components.get("loss_p2_boundary_refiner", 0.0)
            )
            component_total = (
                det_component + final_mask_component + shape_component
                + refiner_component
            )
            if component_total > 0:
                logger.info(
                    "Epoch %d primary loss shares: det=%.2f%% final_mask=%.2f%% "
                    "shape=%.2f%% p2_boundary=%.2f%%",
                    epoch_number,
                    100.0 * det_component / component_total,
                    100.0 * final_mask_component / component_total,
                    100.0 * shape_component / component_total,
                    100.0 * refiner_component / component_total,
                )
            logger.info(
                "Epoch %d schedule: det_loss_weight=%.3f early_stop_start=%d min_delta=%.4g",
                epoch_number, det_loss_weight,
                int(args.early_stopping_start_epoch),
                float(args.early_stopping_min_delta),
            )
            # Debug: log current learning rates
            lr_str = ", ".join([f"{k}={v:.2e}" for k, v in lr_groups.items()])
            logger.info(f"Epoch {epoch + 1} learning rates: {lr_str}")

        val_loss = None
        val_metrics = None
        ema_applied_for_eval = False
        use_ema_this_epoch = (
            ema is not None
            and bool(args.ema_eval)
            and (epoch + 1) >= max(1, int(args.ema_eval_start_epoch))
        )
        if (
            val_loader is not None
            and (epoch + 1) % int(args.val_every_n_epochs) == 0
        ):
            if use_ema_this_epoch:
                ema.apply_to(model)
                ema_applied_for_eval = True
            model.eval()
            if args.freeze_bn:
                _set_norm_eval(model)

            collect_gt_records = cached_val_gt_records is None
            all_gt: List[dict] = (
                [] if collect_gt_records else cached_val_gt_records
            )
            all_dt: List[dict] = []
            all_img_metas: List[dict] = []
            total_val_loss = 0.0
            val_batches = 0
            skipped_no_gt_val_loss_batches = 0
            seen_val_batches = 0

            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if args.max_val_batches > 0 and batch_idx >= args.max_val_batches:
                        break
                    seen_val_batches += 1

                    imgs = batch["imgs"].to(device)
                    img_metas = batch["img_metas"]
                    gt_bboxes = [b.to(device) for b in batch["gt_bboxes"]]
                    gt_labels = [l.to(device) for l in batch["gt_labels"]]
                    gt_masks = batch["gt_masks"]

                    # Prepare data_samples and move to device
                    data_samples = []
                    for i in range(len(imgs)):
                        ds = DetDataSample()
                        ds.set_metainfo(img_metas[i])
                        gt_instances = InstanceData()
                        num_inst = len(gt_labels[i])
                        if num_inst > 0:
                            valid_mask = gt_labels[i] >= 0
                            num_valid = int(valid_mask.sum().item())
                            if num_valid > 0:
                                valid_indices = (
                                    valid_mask.nonzero().squeeze(-1)[:num_valid].cpu()
                                )
                                gt_instances.bboxes = gt_bboxes[i][valid_indices]
                                gt_instances.labels = gt_labels[i][valid_indices]
                                masks = gt_masks[i][valid_indices.cpu().numpy()]
                                gt_instances.masks = BitmapMasks(
                                    masks, *img_metas[i]["img_shape"]
                                )
                            else:
                                gt_instances.bboxes = torch.zeros(
                                    (0, 4), dtype=torch.float32, device=device
                                )
                                gt_instances.labels = torch.zeros(
                                    (0,), dtype=torch.int64, device=device
                                )
                                gt_instances.masks = BitmapMasks(
                                    np.zeros(
                                        (0, *img_metas[i]["img_shape"]), dtype=np.uint8
                                    ),
                                    *img_metas[i]["img_shape"],
                                )
                        else:
                            gt_instances.bboxes = torch.zeros(
                                (0, 4), dtype=torch.float32, device=device
                            )
                            gt_instances.labels = torch.zeros(
                                (0,), dtype=torch.int64, device=device
                            )
                            gt_instances.masks = BitmapMasks(
                                np.zeros(
                                    (0, *img_metas[i]["img_shape"]), dtype=np.uint8
                                ),
                                *img_metas[i]["img_shape"],
                            )
                        ds.gt_instances = gt_instances
                        data_samples.append(ds.to(device))

                    # Optional validation loss is a second model forward. Public
                    # ablation runners disable it for fast point screening; all
                    # images still participate in prediction and COCO evaluation.
                    val_batch_has_valid_gt = False
                    if args.compute_val_loss:
                        for ds in data_samples:
                            labels = getattr(
                                ds.gt_instances, "labels", torch.tensor([])
                            )
                            if labels.numel() > 0 and (labels >= 0).any():
                                val_batch_has_valid_gt = True
                                break
                        if not val_batch_has_valid_gt:
                            skipped_no_gt_val_loss_batches += 1

                    # Synchronize data preprocessing for both images and ground truth
                    processed = data_preprocessor(
                        {"inputs": imgs, "data_samples": data_samples}, training=False
                    )
                    imgs_processed = processed["inputs"]
                    data_samples = processed["data_samples"]

                    model_inputs = imgs_processed

                    model_for_eval = model.module if hasattr(model, "module") else model

                    # ---- Compute loss ----
                    if args.compute_val_loss and val_batch_has_valid_gt:
                        loss_dict = model_for_eval.loss(model_inputs, data_samples)
                        loss_dict = _apply_detection_loss_weight(
                            loss_dict, det_loss_weight, device
                        )
                        val_loss_parts = []
                        for k, v in loss_dict.items():
                            if "loss" not in k.lower():
                                continue
                            if isinstance(v, torch.Tensor):
                                val_loss_parts.append(v)
                            elif isinstance(v, (list, tuple)):
                                val_loss_parts.extend(
                                    x for x in v if isinstance(x, torch.Tensor)
                                )
                        loss = (
                            sum(val_loss_parts)
                            if val_loss_parts
                            else torch.tensor(0.0, device=device)
                        )
                        total_val_loss += float(loss.detach().item())
                        val_batches += 1

                    # ---- Run prediction for COCO evaluation ----
                    outputs = model_for_eval.predict(model_inputs, data_samples, rescale=False)

                    # Collect GT and predictions
                    for j, output in enumerate(outputs):
                        meta = img_metas[j]
                        img_shape = meta["img_shape"]

                        # GT - from data_samples
                        if collect_gt_records:
                            gt_inst = data_samples[j].gt_instances
                            gt_np = _extract_instances_numpy(
                                gt_inst, img_shape, encode_masks=True
                            )
                            all_gt.append(gt_np)

                        # Predictions - from output.pred_instances (InstanceData)
                        pred_inst = output.pred_instances if hasattr(output, "pred_instances") else output
                        pred_np = _extract_instances_numpy(
                            pred_inst, img_shape, encode_masks=True
                        )
                        all_dt.append(pred_np)

                        all_img_metas.append(meta)

            if args.compute_val_loss and skipped_no_gt_val_loss_batches > 0:
                logger.warning(
                    (
                        "Epoch %d skipped val loss on batches with no valid GT: "
                        "%d/%d (%.2f%%); these batches were still included in "
                        "COCO evaluation"
                    ),
                    epoch + 1,
                    skipped_no_gt_val_loss_batches,
                    seen_val_batches,
                    100.0
                    * skipped_no_gt_val_loss_batches
                    / max(1, seen_val_batches),
                )

            val_loss = (
                total_val_loss / max(1, val_batches)
                if args.compute_val_loss
                else None
            )

            if collect_gt_records:
                cached_val_gt_records = all_gt
                logger.info(
                    "Cached validation GT as RLE records: images=%d",
                    len(cached_val_gt_records),
                )
            elif len(all_gt) != len(all_dt):
                raise RuntimeError(
                    "Cached validation GT count no longer matches predictions: "
                    f"GT={len(all_gt)} DT={len(all_dt)}"
                )

            # ---- COCO-style evaluation ----
            val_metrics = {}
            total_gt = sum(len(g["labels"]) for g in all_gt)
            total_dt = sum(len(d["scores"]) for d in all_dt if "scores" in d)
            avg_mask_fill, avg_masks_per_image = _summarize_mask_density(all_dt)

            if is_main:
                logger.info(
                    (
                        "Validation samples: GT=%d, DT=%d, "
                        "avg_masks_per_image=%.2f, avg_mask_fill=%.4f"
                    ),
                    total_gt,
                    total_dt,
                    avg_masks_per_image,
                    avg_mask_fill,
                )

            if total_gt > 0:
                # Honor TEST_MAX_PER_IMG during per-epoch validation so that
                # best-model selection runs at the same maxDets contract as
                # the final --test-only report (previously this env only
                # affected the test-only path, silently leaving best selection
                # at the COCO default maxDets=100). The model-side cap mirrors
                # the test-only override: dense tiles would otherwise batch
                # hundreds of instances through the SAM2 decoder (~10 GB
                # spike) only for the eval to keep the top 150 anyway. Ranking
                # is score-based, so metrics are identical either way.
                _val_max_dets = None
                _val_md_env = int(os.environ.get("TEST_MAX_PER_IMG", "0") or 0)
                if _val_md_env > 0:
                    _val_max_dets = [1, 10, _val_md_env]
                    _val_model_for_eval = (
                        model.module if hasattr(model, "module") else model
                    )
                    _val_test_cfg = getattr(_val_model_for_eval, "test_cfg", None)
                    if _val_test_cfg is not None and "rcnn" in _val_test_cfg:
                        _val_test_cfg["rcnn"]["max_per_img"] = _val_md_env
                    _val_roi_cfg = getattr(
                        getattr(_val_model_for_eval, "roi_head", None),
                        "test_cfg",
                        None,
                    )
                    if _val_roi_cfg is not None and "max_per_img" in _val_roi_cfg:
                        _val_roi_cfg["max_per_img"] = _val_md_env
                # The segmentation score contract is stored in cfg.model and
                # consumed identically by validation and checkpoint inference.
                # Bbox candidate selection/ranking always remains detector-score based.
                coco_gt, coco_bbox_dt = build_coco_gt_and_dt(
                    all_gt, all_dt, all_img_metas, score_key="scores"
                )
                if eval_bbox:
                    bbox_metrics = run_coco_eval(
                        coco_gt, coco_bbox_dt, iou_type="bbox",
                        max_dets=_val_max_dets,
                    )
                    val_metrics.update(bbox_metrics)
                    logger.info(
                        "Validation bbox/mAP: %.4f bbox/mAP_75: %.4f",
                        bbox_metrics.get("bbox/mAP", 0.0),
                        bbox_metrics.get("bbox/mAP_75", 0.0),
                    )
                if segm_score_key == "scores":
                    coco_segm_dt = coco_bbox_dt
                else:
                    _, coco_segm_dt = build_coco_gt_and_dt(
                        all_gt,
                        all_dt,
                        all_img_metas,
                        score_key=segm_score_key,
                    )
                segm_metrics = run_coco_eval(
                    coco_gt, coco_segm_dt, iou_type="segm",
                    max_dets=_val_max_dets,
                )
                val_metrics.update(segm_metrics)
                logger.info(
                    "Validation segm/mAP: %.4f",
                    segm_metrics.get("segm/mAP", 0.0),
                )

                # Optional compatibility metric at a second maxDets setting.
                # Used by accelerated runs whose primary eval runs at a
                # non-default maxDets (TEST_MAX_PER_IMG) but that still need a
                # curve directly comparable with historical maxDets=100 logs.
                # Stored under suffixed keys so best-model/early-stop lookups
                # on the plain "segm/mAP" key are unaffected.
                compat_max_dets = int(os.environ.get("VAL_COMPAT_MAX_DETS", "0") or 0)
                if compat_max_dets > 0:
                    for _iou_type, _dt in (
                        ("bbox", coco_bbox_dt),
                        ("segm", coco_segm_dt),
                    ):
                        _compat = run_coco_eval(
                            coco_gt,
                            _dt,
                            iou_type=_iou_type,
                            max_dets=[1, 10, compat_max_dets],
                        )
                        val_metrics.update(
                            {
                                f"{k}@md{compat_max_dets}": v
                                for k, v in _compat.items()
                            }
                        )
                        logger.info(
                            "Validation %s/mAP@maxDet%d: %.4f",
                            _iou_type,
                            compat_max_dets,
                            _compat.get(f"{_iou_type}/mAP", 0.0),
                        )

        # Match the baseline's post-validation cleanup and reduce allocator
        # fragmentation across long runs.  Non-main ranks reach this point
        # early and then wait at the existing epoch-end DDP synchronization.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        stop_training = False
        if is_main:
            history["epochs"].append(epoch)
            history["train_losses"].append(avg_loss)
            history["val_losses"].append(val_loss)
            history["val_metrics"].append(val_metrics)
            history["learning_rates"].append(lr_groups)

            # Update best metrics tracking
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss

            if avg_loss < best_train_loss:
                best_train_loss = avg_loss

            # Check if this is the best model based on segm/mAP
            is_best = False
            is_bbox_best = False
            is_composite_best = False
            current_segm_map = 0.0
            current_bbox_score = None
            current_composite_score = None
            if val_metrics is not None and "segm/mAP" in val_metrics:
                current_segm_map = val_metrics["segm/mAP"]
                if current_segm_map > best_segm_map:
                    best_segm_map = current_segm_map
                    is_best = True
                    if is_main:
                        logger.info(f"New best segm/mAP: {best_segm_map:.4f}")

            if val_metrics is not None and save_bbox_best_metric:
                current_bbox_score = _metric_value(val_metrics, save_bbox_best_metric)
                if current_bbox_score is not None and current_bbox_score > best_bbox_score:
                    best_bbox_score = current_bbox_score
                    is_bbox_best = True
                    logger.info("New best %s: %.4f", save_bbox_best_metric, best_bbox_score)

            if val_metrics is not None and save_composite_best:
                current_composite_score = _weighted_metric_value(val_metrics, composite_metric_spec)
                if (
                    current_composite_score is not None
                    and current_composite_score > best_composite_score
                ):
                    best_composite_score = current_composite_score
                    is_composite_best = True
                    logger.info(
                        "New best composite (%s): %.4f",
                        composite_metric_spec,
                        best_composite_score,
                    )

            current_early_stopping_score = None
            if val_metrics is not None:
                current_early_stopping_score = _metric_expression_value(
                    val_metrics, early_stopping_metric_spec
                )
                if current_early_stopping_score is None:
                    raise RuntimeError(
                        "Early-stopping metric could not be resolved: "
                        f"{early_stopping_metric_spec!r}. Available metrics: "
                        f"{sorted(val_metrics.keys())}"
                    )
                if current_early_stopping_score is not None:
                    early_stopping_metric_history.append(
                        float(current_early_stopping_score)
                    )
                    smooth_window = max(
                        1, int(args.early_stopping_smooth_window)
                    )
                    smoothed_early_stopping_score = float(
                        np.mean(
                            early_stopping_metric_history[-smooth_window:]
                        )
                    )
                    improved = (
                        smoothed_early_stopping_score
                        > best_early_stopping_score
                        + float(args.early_stopping_min_delta)
                    )
                    if improved:
                        best_early_stopping_score = (
                            smoothed_early_stopping_score
                        )
                        early_counter = 0
                        logger.info(
                            "New best early-stop metric (%s): raw=%.4f "
                            "smooth(%d)=%.4f",
                            early_stopping_metric_spec,
                            current_early_stopping_score,
                            smooth_window,
                            best_early_stopping_score,
                        )
                    elif epoch_number >= int(args.early_stopping_start_epoch):
                        early_counter += 1
                        logger.info(
                            "Early-stop metric (%s): raw=%.4f smooth(%d)=%.4f "
                            "best=%.4f counter=%d",
                            early_stopping_metric_spec,
                            current_early_stopping_score,
                            smooth_window,
                            smoothed_early_stopping_score,
                            best_early_stopping_score,
                            early_counter,
                        )
                    else:
                        logger.info(
                            "Early-stop metric (%s): raw=%.4f smooth(%d)=%.4f; "
                            "counter disabled before epoch %d",
                            early_stopping_metric_spec,
                            current_early_stopping_score,
                            smooth_window,
                            smoothed_early_stopping_score,
                            int(args.early_stopping_start_epoch),
                        )

            best_metrics = {
                "best_train_loss": best_train_loss,
                "best_val_loss": best_val_loss,
                "best_segm_map": best_segm_map,
                "best_bbox_score": best_bbox_score,
                "best_bbox_metric": save_bbox_best_metric,
                "best_composite_score": best_composite_score,
                "best_composite_metric": composite_metric_spec,
                "best_early_stopping_score": best_early_stopping_score,
                "early_stopping_metric": early_stopping_metric_spec,
                "early_counter": int(early_counter),
                "early_stopping_start_epoch": int(args.early_stopping_start_epoch),
                "early_stopping_min_delta": float(args.early_stopping_min_delta),
                "early_stopping_smooth_window": int(
                    args.early_stopping_smooth_window
                ),
                "early_stopping_metric_history": list(
                    early_stopping_metric_history
                ),
            }

            # Save best model: write to new file first, verify, then delete old.
            # Only rank 0 does checkpoint I/O; all ranks synchronize after this
            # block below. Placing a barrier inside `if is_main` deadlocks DDP.
            ema_state_to_save = ema.state_dict() if ema is not None else None
            if is_best:
                _save_verified_best_checkpoint(
                    checkpoint_dir,
                    "best_model",
                    "segm/mAP",
                    best_segm_map,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    best_metrics,
                    history,
                    config_snapshot,
                    scaler,
                    ema_state=ema_state_to_save,
                )


            if is_bbox_best and current_bbox_score is not None:
                _save_verified_best_checkpoint(
                    checkpoint_dir,
                    "best_bbox_model",
                    save_bbox_best_metric,
                    current_bbox_score,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    best_metrics,
                    history,
                    config_snapshot,
                    scaler,
                    ema_state=ema_state_to_save,
                )

            if is_composite_best and current_composite_score is not None:
                _save_verified_best_checkpoint(
                    checkpoint_dir,
                    "best_composite_model",
                    composite_metric_spec,
                    current_composite_score,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    best_metrics,
                    history,
                    config_snapshot,
                    scaler,
                    ema_state=ema_state_to_save,
                )

            if epoch == args.epochs - 1 and _env_flag("SAVE_LAST_MODEL", "0"):
                last_path = checkpoint_dir / f"last_model_epoch{epoch + 1}.pth"
                _save_checkpoint(
                    last_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_metrics,
                    history,
                    config_snapshot,
                    scaler,
                )
                logger.info("Saved final checkpoint: %s", last_path)

            # Periodic last_checkpoint.pth: overwrite every epoch so a crash can
            # be resumed from the most recent epoch without losing progress.
            # Controlled by SAVE_LAST_CHECKPOINT (default on). Unlike best_model,
            # this is not gated on metric improvement.
            if _env_flag("SAVE_LAST_CHECKPOINT", "1") and epoch < args.epochs - 1:
                _last_ckpt = checkpoint_dir / "last_checkpoint.pth"
                _save_checkpoint(
                    _last_ckpt,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_metrics,
                    history,
                    config_snapshot,
                    scaler,
                    ema_state=ema_state_to_save,
                )

            # Restore training weights after EMA-based eval/save
            if ema_applied_for_eval:
                ema.restore(model)
                ema_applied_for_eval = False

            # Logging: metric evaluation and early stopping do not depend on
            # whether the optional validation-loss forward is enabled.
            lr_str = " ".join([f"{k}={v:.2e}" for k, v in lr_groups.items()])
            if val_metrics is not None:
                segm_info = f" segm/mAP={current_segm_map:.4f}" if current_segm_map > 0 else ""
                bbox_info = (
                    f" {save_bbox_best_metric}={current_bbox_score:.4f}"
                    if current_bbox_score is not None
                    else ""
                )
                composite_info = (
                    f" composite={current_composite_score:.4f}"
                    if current_composite_score is not None
                    else ""
                )
                val_info = (
                    f"{val_loss:.6f}" if val_loss is not None else "disabled"
                )
                logger.info(
                    "Epoch %d/%d | train=%.6f val=%s%s%s%s | best_segm=%.4f best_bbox=%.4f best_comp=%.4f | %s",
                    epoch + 1,
                    args.epochs,
                    avg_loss,
                    val_info,
                    segm_info,
                    bbox_info,
                    composite_info,
                    best_segm_map,
                    best_bbox_score,
                    best_composite_score,
                    lr_str,
                )
            else:
                logger.info(
                    "Epoch %d/%d | train=%.6f | %s",
                    epoch + 1,
                    args.epochs,
                    avg_loss,
                    lr_str,
                )

            if (
                current_early_stopping_score is not None
                and epoch_number >= int(args.early_stopping_start_epoch)
                and args.early_stopping_patience > 0
                and early_counter >= int(args.early_stopping_patience)
            ):
                logger.info(
                    "Early stopping triggered (patience=%d)",
                    int(args.early_stopping_patience),
                )
                stop_training = True

        # Rank 0 decides whether to stop after its rank-0-only validation. The
        # CPU/Gloo broadcast doubles as the epoch boundary: non-main ranks wait
        # here without keeping an NCCL collective alive during long COCO eval.
        if distributed and dist.is_initialized() and control_group is not None:
            stop_training = _broadcast_stop_training(
                stop_training, control_group
            )
        if stop_training:
            break

    if distributed and dist.is_initialized():
        if control_group is not None:
            dist.destroy_process_group(control_group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
