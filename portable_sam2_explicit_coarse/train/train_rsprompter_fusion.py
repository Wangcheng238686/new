#!/usr/bin/env python

import argparse
import gc
import logging
import math
import os
import sys
from collections import OrderedDict
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

        # Predictions
        dt_bboxes = dt["bboxes"]
        dt_scores = dt.get(score_key, dt["scores"])
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


def run_coco_eval(coco_gt, coco_dt, iou_type: str = "bbox") -> OrderedDict:
    """Run COCOeval and return metrics dict."""
    from pycocotools.cocoeval import COCOeval

    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

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


def _extract_instances_numpy(instances, img_shape: Tuple[int, int]) -> dict:
    """Extract bboxes, labels, scores, masks from InstanceData to numpy."""
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

    out = {"bboxes": bboxes, "labels": labels, "masks": masks}
    if scores is not None:
        out["scores"] = scores
    if hasattr(instances, "mask_scores"):
        mask_scores = instances.mask_scores
        if isinstance(mask_scores, torch.Tensor):
            mask_scores = mask_scores.detach().cpu().numpy()
        out["mask_scores"] = mask_scores
    return out


def _init_distributed() -> Dict[str, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    launched_by_torchrun = os.environ.get("LOCAL_RANK", None) is not None
    should_init = (world_size > 1) or launched_by_torchrun
    if should_init and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    distributed = dist.is_initialized()
    return {
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "distributed": int(distributed),
    }


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
    densebr_lr_mult: float = 1.0,
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
    params_densebr = []
    names_densebr = []
    params_densebr_scalar = []
    names_densebr_scalar = []
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
        elif "roi_head.mask_head.densebr" in match_name:
            if match_name.endswith(".beta_logit"):
                params_densebr_scalar.append(param)
                names_densebr_scalar.append(name)
            else:
                params_densebr.append(param)
                names_densebr.append(name)
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
                "weight_decay": weight_decay * 0.1,
                "name": "mask_decoder",
            }
        )
        audit_groups.append(("mask_decoder", params_mask_decoder, names_mask_decoder, lr * mask_decoder_lr_mult, weight_decay * 0.1))
    if params_no_mask:
        param_groups.append(
            {
                "params": params_no_mask,
                "lr": lr * no_mask_lr_mult,
                "weight_decay": 0.0,
                "name": "no_mask_embed",
            }
        )
        audit_groups.append(("no_mask_embed", params_no_mask, names_no_mask, lr * no_mask_lr_mult, 0.0))
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
    if params_densebr:
        param_groups.append(
            {
                "params": params_densebr,
                "lr": lr * densebr_lr_mult,
                "name": "densebr",
            }
        )
        audit_groups.append(("densebr", params_densebr, names_densebr, lr * densebr_lr_mult, weight_decay))
    if params_densebr_scalar:
        param_groups.append(
            {
                "params": params_densebr_scalar,
                "lr": lr * densebr_lr_mult,
                "weight_decay": 0.0,
                "name": "densebr_scalar",
            }
        )
        audit_groups.append(("densebr_scalar", params_densebr_scalar, names_densebr_scalar, lr * densebr_lr_mult, 0.0))
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

    def restore(self, model: torch.nn.Module):
        if self.backup_state is None:
            return
        model_core = _get_model_core(model)
        model_core.load_state_dict(self.backup_state, strict=False)
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


def _set_densebr_only_train_mode(model: torch.nn.Module) -> None:
    """Keep the loaded C2 baseline deterministic while training DenseBR."""
    model_core = model.module if hasattr(model, "module") else model
    model_core.eval()
    mask_head = getattr(getattr(model_core, "roi_head", None), "mask_head", None)
    densebr = getattr(mask_head, "densebr", None)
    if densebr is None:
        raise RuntimeError("--train-densebr-only requires an enabled DenseBR module")
    densebr.train()


def _set_densebr_runtime_active(model: torch.nn.Module, active: bool) -> bool:
    """Enable or bypass DenseBR while preserving one stable model layout."""
    model_core = model.module if hasattr(model, "module") else model
    mask_head = getattr(getattr(model_core, "roi_head", None), "mask_head", None)
    densebr = getattr(mask_head, "densebr", None)
    if densebr is None:
        return False
    mask_head.densebr_runtime_active = bool(active)
    return True


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
        "densebr": getattr(mask_head, "densebr", None) if mask_head is not None else None,
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
) -> Dict:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
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
) -> Dict[str, object]:
    """Load matching tensors for initialization, with explicit legacy migration."""
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
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


def main():
    parser = argparse.ArgumentParser(
        description="portable_sam_fusion: RSPrompter(SAM) + Drone semantic guidance"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--data-root", type=str, default="/data/wangcheng/dataset/university-test/S"
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--checkpoint-dir", type=str, default="/tmp/portable_sam_fusion_ckpts"
    )
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--init-from", type=str, default=None)
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
        "--train-densebr-only",
        action="store_true",
        help="Freeze the loaded baseline and optimize only roi_head.mask_head.densebr.",
    )
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
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--early-stopping-start-epoch", type=int, default=1)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--det-loss-stage1-end", type=int, default=5)
    parser.add_argument("--det-loss-stage2-end", type=int, default=10)
    parser.add_argument("--det-loss-weight-stage1", type=float, default=1.0)
    parser.add_argument("--det-loss-weight-stage2", type=float, default=0.75)
    parser.add_argument("--det-loss-weight-stage3", type=float, default=0.50)
    parser.add_argument(
        "--max-nonfinite-batches",
        type=int,
        default=0,
        help="Abort after this many synchronized non-finite batches; 0 disables abort.",
    )

    parser.add_argument("--sat-backbone-lr-mult", type=float, default=0.5)
    parser.add_argument("--sat-other-lr-mult", type=float, default=1.0)
    parser.add_argument(
        "--bbox-head-lr-mult",
        type=float,
        default=0.0,
        help="Use >0 to place bbox_head in a separate optimizer group",
    )
    parser.add_argument("--drone-lr-mult", type=float, default=0.5)
    parser.add_argument("--mask-decoder-lr-mult", type=float, default=0.05)
    parser.add_argument("--no-mask-lr-mult", type=float, default=0.1)
    parser.add_argument("--prompt-encoder-lr-mult", type=float, default=0.1)
    parser.add_argument("--shape-prior-lr-mult", type=float, default=1.0)
    parser.add_argument("--densebr-lr-mult", type=float, default=1.0)
    parser.add_argument(
        "--densebr-enable-epoch",
        type=int,
        default=1,
        help="1-based epoch that enables DenseBR; earlier epochs use the exact baseline path.",
    )
    parser.add_argument("--quality-head-lr-mult", type=float, default=1.0)
    parser.add_argument("--scene-align-lr-mult", type=float, default=2.0, help="Learning rate multiplier for scene alignment parameters")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-scenes", type=int, default=1000, help="Maximum number of scenes for scene-specific alignment")
    parser.add_argument("--warmup-epochs", type=int, default=0, help="Number of warmup epochs for learning rate scheduling")
    parser.add_argument("--warmup-iters", type=int, default=0, help="Number of warmup iterations (overrides warmup-epochs if > 0)")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay for optimizer")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate for regularization")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # --- AMP / EMA (训练基础设施) ---
    parser.add_argument("--amp", type=int, default=1, help="Enable AMP mixed precision when set to 1")
    parser.add_argument("--ema-enabled", type=int, default=0, help="Enable EMA shadow weights")
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
    parser.add_argument("--densebr-debug-stats", type=int, choices=[0, 1], default=0,
                        help="Print DenseBR diagnostic stats during training")
    parser.add_argument("--densebr-debug-stats-interval", type=int, default=50,
                        help="Training iteration interval for DenseBR diagnostic stats")
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[seed] Random seed fixed to {args.seed} | cudnn.deterministic=True")

    dist_info = _init_distributed()
    distributed = bool(dist_info.get("distributed", 0))
    rank = int(dist_info["rank"])

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

    cfg = Config.fromfile(args.config)

    # B/C/D 都依赖 ROI retrieval；C 还依赖 SAM2 PromptEncoder 的 masks= 通道。
    # 这里再做一次联动兜底, 避免只改顶层镜像或只改 shell 参数导致半开配置。
    if hasattr(cfg.model, 'max_scenes'):
        cfg.model.max_scenes = args.max_scenes
    model = MODELS.build(cfg.model)
    model.to(device)

    exclusive_scopes = sum(
        int(flag)
        for flag in (
            args.train_densebr_only,
            args.train_quality_head_only,
        )
    )
    if exclusive_scopes > 1:
        raise RuntimeError(
            "Experimental module-only training scopes are mutually exclusive"
        )
    if args.train_densebr_only:
        model.requires_grad_(False)
        mask_head = getattr(getattr(model, "roi_head", None), "mask_head", None)
        densebr = getattr(mask_head, "densebr", None)
        if densebr is None:
            raise RuntimeError("--train-densebr-only requires DENSEBR_ENABLED=1")
        densebr.requires_grad_(True)
        _set_densebr_only_train_mode(model)
        if is_main:
            logger.info("Training scope: DenseBR only; loaded C2 baseline is frozen in eval mode")
    elif args.train_quality_head_only:
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
            whu_train_ann_file="2.4 annotation/annotation/train.json",
            whu_val_ann_file="2.4 annotation/annotation/validation.json",
            whu_train_img_subdir="2.1 train/train",
            whu_val_img_subdir="2.3 valid/validation",
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
            "Validation policy: rank0_only=%d batch_size=%d every_n_epochs=%d",
            int(distributed),
            int(args.val_batch_size),
            int(args.val_every_n_epochs),
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
        densebr_lr_mult=args.densebr_lr_mult,
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
    save_bbox_best_metric = os.environ.get("SAVE_BBOX_BEST_METRIC", "bbox/mAP_75")
    save_composite_best = _env_flag("SAVE_COMPOSITE_BEST", "0")
    composite_metric_spec = os.environ.get(
        "SAVE_COMPOSITE_WEIGHTS", "0.5*bbox/mAP_75+0.5*segm/mAP_75"
    )
    early_stopping_metric_spec = os.environ.get(
        "EARLY_STOPPING_METRIC", "segm/mAP"
    )
    config_snapshot = {
        "checkpoint_schema_version": 1,
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
            "sam2_model_size": os.environ.get("SAM2_MODEL_SIZE", ""),
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
        "det_loss_schedule": {
            "stage1_end": int(args.det_loss_stage1_end),
            "stage2_end": int(args.det_loss_stage2_end),
            "stage1_weight": float(args.det_loss_weight_stage1),
            "stage2_weight": float(args.det_loss_weight_stage2),
            "stage3_weight": float(args.det_loss_weight_stage3),
        },
        "point_warmup": {
            "no_point_epochs": os.environ.get("POINT_WARMUP_NO_POINT_EPOCHS", ""),
            "one_pair_epochs": os.environ.get("POINT_WARMUP_ONE_PAIR_EPOCHS", ""),
            "full_2p2n_start_epoch": os.environ.get("POINT_WARMUP_FULL_START_EPOCH", ""),
        },
        "shape_loss_schedule": {
            "stage1_end": os.environ.get("SHAPE_LOSS_STAGE1_END", ""),
            "stage1_weight": os.environ.get("SHAPE_LOSS_WEIGHT_STAGE1", ""),
            "stage2_weight": os.environ.get("SHAPE_LOSS_WEIGHT_STAGE2", ""),
        },
        "shape_dense_gate": {
            "mode": os.environ.get("SHAPE_SCALE_MODE", ""),
            "init": os.environ.get("SHAPE_SCALE_INIT", ""),
        },
        "freeze_decoder": os.environ.get("FREEZE_DECODER", ""),
        "freeze_no_mask": os.environ.get("FREEZE_NO_MASK", ""),
        "shape_prior_loss_weight": os.environ.get("SHAPE_PRIOR_LOSS_WEIGHT", ""),
        "densebr_enabled": os.environ.get("DENSEBR_ENABLED", "0"),
        "densebr_enable_epoch": int(args.densebr_enable_epoch),
        "densebr_type": "canonical_prompt_refiner",
        "densebr_beta_init": os.environ.get("DENSEBR_BETA_INIT", "0.05"),
        "densebr_beta_max": os.environ.get("DENSEBR_BETA_MAX", "0.20"),
        "densebr_delta_logit_max": os.environ.get(
            "DENSEBR_DELTA_LOGIT_MAX", "2.0"
        ),
        "densebr_quality_gate_init": os.environ.get(
            "DENSEBR_QUALITY_GATE_INIT", "0.50"
        ),
        "densebr_roi_channels": os.environ.get("DENSEBR_ROI_CHANNELS", "64"),
        "densebr_cue_channels": os.environ.get("DENSEBR_CUE_CHANNELS", "32"),
        "densebr_mid_channels": os.environ.get("DENSEBR_MID_CHANNELS", "64"),
        "densebr_detach_prompt_cues": os.environ.get(
            "DENSEBR_DETACH_PROMPT_CUES", "1"
        ),
        "train_densebr_only": int(args.train_densebr_only),
        "train_quality_head_only": int(args.train_quality_head_only),
        "densebr_debug_stats": args.densebr_debug_stats,
        "densebr_debug_stats_interval": args.densebr_debug_stats_interval,
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
            "densebr": args.densebr_lr_mult,
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
    best_densebr_active_segm_map = 0.0
    early_counter = 0
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
            Path(args.init_from), model, exclude_prefixes=exclude_prefixes
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
        ckpt = _load_checkpoint(Path(args.resume_from), model, optimizer, scheduler, scaler)
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
        best_densebr_active_segm_map = float(
            best.get("best_densebr_active_segm_map", best_densebr_active_segm_map)
        )
        early_counter = int(best.get("early_counter", early_counter))
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
                config_snapshot["checkpoint_schema_version"] = 1
        if is_main:
            logger.info(
                "Resumed from %s (start_epoch=%d, best_segm_map=%.4f, best_bbox_score=%.4f, best_composite_score=%.4f)",
                args.resume_from,
                start_epoch,
                best_segm_map,
                best_bbox_score,
                best_composite_score,
            )

    nonfinite_batches = 0
    for epoch in range(start_epoch, args.epochs):
        epoch_number = epoch + 1
        _set_model_current_epoch(model, epoch_number)
        det_loss_weight = _detection_loss_weight(args, epoch_number)
        densebr_active = epoch_number >= max(1, int(args.densebr_enable_epoch))
        has_densebr = _set_densebr_runtime_active(model, densebr_active)
        densebr_delayed = has_densebr and int(args.densebr_enable_epoch) > 1
        if densebr_delayed and epoch_number == int(args.densebr_enable_epoch):
            early_counter = 0
            best_early_stopping_score = float("-inf")
            if is_main:
                logger.info(
                    "DenseBR activated at epoch %d; reset early-stopping state for refinement phase",
                    epoch_number,
                )
        if distributed:
            sampler = getattr(train_loader, "sampler", None)
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)

        model.train()
        if args.train_densebr_only:
            _set_densebr_only_train_mode(model)
        elif args.train_quality_head_only:
            _set_quality_head_only_train_mode(model)
        elif args.freeze_bn:
            _set_norm_eval(model)

        total_loss = 0.0
        loss_meter = {}
        num_batches = 0
        grad_accum = args.grad_accum_steps
        train_batches_limit = len(train_loader)
        if args.max_train_batches > 0:
            train_batches_limit = min(train_batches_limit, args.max_train_batches)
        optimizer.zero_grad(set_to_none=True)
        skipped_empty = 0
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

            batch_has_valid_gt = False
            for ds in data_samples:
                labels = getattr(ds.gt_instances, "labels", torch.tensor([]))
                if labels.numel() > 0 and (labels >= 0).any():
                    batch_has_valid_gt = True
                    break
            if not batch_has_valid_gt:
                skipped_empty += 1
                continue
            
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
                loss_dict = _apply_detection_loss_weight(
                    loss_dict, det_loss_weight, device
                )
                loss_parts = []
                for k, v in loss_dict.items():
                    if "loss" not in k:
                        continue
                    if isinstance(v, torch.Tensor):
                        if not (args.train_densebr_only or args.train_quality_head_only) or v.requires_grad:
                            loss_parts.append(v)
                    elif isinstance(v, (list, tuple)):
                        loss_parts.extend(
                            x
                            for x in v
                            if isinstance(x, torch.Tensor)
                            and (not (args.train_densebr_only or args.train_quality_head_only) or x.requires_grad)
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
                        and (not (args.train_densebr_only or args.train_quality_head_only) or value.requires_grad)
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
                    or (
                        args.densebr_debug_stats
                        and args.densebr_debug_stats_interval > 0
                        and global_step % args.densebr_debug_stats_interval == 0
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
                if (
                    args.densebr_debug_stats
                    and args.densebr_debug_stats_interval > 0
                    and global_step % args.densebr_debug_stats_interval == 0
                ):
                    enabled_prefixes.update(
                        {"DENSEBR", "DENSEBR_PROMPT", "DENSEBR_FEEDBACK"}
                    )
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
        if skipped_empty > 0 and is_main:
            logger.info(
                "Epoch %d: skipped %d batches with no valid GT (empty / background images)",
                epoch + 1, skipped_empty,
            )

        # Collect LRs for all groups
        lr_groups = {}
        for group in optimizer.param_groups:
            name = group.get("name", "unknown")
            lr_groups[name] = group["lr"]

        if is_main:
            loss_str = []
            for k, v in loss_meter.items():
                avg_k = v / max(1, num_batches)
                loss_str.append(f"{k}={avg_k:.4f}")
            logger.info(f"Epoch {epoch + 1} detailed losses: {', '.join(loss_str)}")
            averaged_components = {
                key: value / max(1, num_batches) for key, value in loss_meter.items()
            }
            det_component = sum(
                value for key, value in averaged_components.items()
                if key in _CANONICAL_DETECTION_LOSS_KEYS or key.startswith("loss_rpn_")
            )
            final_mask_component = float(averaged_components.get("loss_mask", 0.0))
            shape_component = float(averaged_components.get("loss_shape_prior", 0.0))
            component_total = det_component + final_mask_component + shape_component
            if component_total > 0:
                logger.info(
                    "Epoch %d primary loss shares: det=%.2f%% final_mask=%.2f%% shape=%.2f%%",
                    epoch_number,
                    100.0 * det_component / component_total,
                    100.0 * final_mask_component / component_total,
                    100.0 * shape_component / component_total,
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

            all_gt: List[dict] = []
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

                    # Match the historical baseline: loss requires positive
                    # GT, but GT-empty images still participate in COCO
                    # evaluation so their false positives are counted.
                    val_batch_has_valid_gt = False
                    for ds in data_samples:
                        labels = getattr(ds.gt_instances, "labels", torch.tensor([]))
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
                    if val_batch_has_valid_gt:
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
                        gt_inst = data_samples[j].gt_instances
                        gt_np = _extract_instances_numpy(gt_inst, img_shape)
                        all_gt.append(gt_np)

                        # Predictions - from output.pred_instances (InstanceData)
                        pred_inst = output.pred_instances if hasattr(output, "pred_instances") else output
                        pred_np = _extract_instances_numpy(pred_inst, img_shape)
                        all_dt.append(pred_np)

                        all_img_metas.append(meta)

            if skipped_no_gt_val_loss_batches > 0:
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

            val_loss = total_val_loss / max(1, val_batches)

            # ---- COCO-style evaluation ----
            val_metrics = {}
            total_gt = sum(len(g["labels"]) for g in all_gt)
            total_dt = sum(len(d["scores"]) for d in all_dt if "scores" in d)

            if is_main:
                logger.info(f"Validation samples: GT={total_gt}, DT={total_dt}")

            if total_gt > 0:
                # Historical WHU1024 evaluation uses detector scores for both
                # bbox and segmentation ranking and one shared COCO DT object.
                coco_gt, coco_dt = build_coco_gt_and_dt(
                    all_gt, all_dt, all_img_metas, score_key="scores"
                )
                if eval_bbox:
                    bbox_metrics = run_coco_eval(coco_gt, coco_dt, iou_type="bbox")
                    val_metrics.update(bbox_metrics)
                    logger.info(
                        "Validation bbox/mAP: %.4f bbox/mAP_75: %.4f",
                        bbox_metrics.get("bbox/mAP", 0.0),
                        bbox_metrics.get("bbox/mAP_75", 0.0),
                    )

                segm_metrics = run_coco_eval(coco_gt, coco_dt, iou_type="segm")
                val_metrics.update(segm_metrics)
                logger.info(
                    "Validation segm/mAP: %.4f",
                    segm_metrics.get("segm/mAP", 0.0),
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
            is_densebr_active_best = False
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
                if densebr_delayed and densebr_active and current_segm_map > best_densebr_active_segm_map:
                    best_densebr_active_segm_map = current_segm_map
                    is_densebr_active_best = True
                    logger.info(
                        "New best DenseBR-active segm/mAP: %.4f",
                        best_densebr_active_segm_map,
                    )

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
            if val_metrics is not None and (not densebr_delayed or densebr_active):
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
                    improved = (
                        current_early_stopping_score
                        > best_early_stopping_score
                        + float(args.early_stopping_min_delta)
                    )
                    if improved:
                        best_early_stopping_score = current_early_stopping_score
                        early_counter = 0
                        logger.info(
                            "New best early-stop metric (%s): %.4f",
                            early_stopping_metric_spec,
                            best_early_stopping_score,
                        )
                    elif epoch_number >= int(args.early_stopping_start_epoch):
                        early_counter += 1
                    else:
                        logger.info(
                            "Early-stop counter disabled before epoch %d",
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
                "best_densebr_active_segm_map": best_densebr_active_segm_map,
                "early_stopping_metric": early_stopping_metric_spec,
                "early_counter": int(early_counter),
                "early_stopping_start_epoch": int(args.early_stopping_start_epoch),
                "early_stopping_min_delta": float(args.early_stopping_min_delta),
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

            if is_densebr_active_best:
                _save_verified_best_checkpoint(
                    checkpoint_dir,
                    "best_densebr_active_model",
                    "segm/mAP",
                    best_densebr_active_segm_map,
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

            # Logging
            if val_loss is not None:
                lr_str = " ".join([f"{k}={v:.2e}" for k, v in lr_groups.items()])
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
                logger.info(
                    "Epoch %d/%d | train=%.6f val=%.6f%s%s%s | best_segm=%.4f best_bbox=%.4f best_comp=%.4f | %s",
                    epoch + 1,
                    args.epochs,
                    avg_loss,
                    val_loss,
                    segm_info,
                    bbox_info,
                    composite_info,
                    best_segm_map,
                    best_bbox_score,
                    best_composite_score,
                    lr_str,
                )
                if (
                    (not densebr_delayed or densebr_active)
                    and epoch_number >= int(args.early_stopping_start_epoch)
                    and args.early_stopping_patience > 0
                    and early_counter >= int(
                    args.early_stopping_patience
                    )
                ):
                    logger.info(
                        "Early stopping triggered (patience=%d)",
                        int(args.early_stopping_patience),
                    )
                    stop_training = True
            else:
                lr_str = " ".join([f"{k}={v:.2e}" for k, v in lr_groups.items()])
                logger.info(
                    "Epoch %d/%d | train=%.6f | %s",
                    epoch + 1,
                    args.epochs,
                    avg_loss,
                    lr_str,
                )

        # Keep all DDP ranks aligned after rank-0-only checkpoint/logging work.
        # The stop flag is decided on rank 0 and broadcast so non-main ranks do
        # not continue into the next epoch after rank 0 exits.
        if distributed and dist.is_initialized():
            stop_tensor = torch.tensor(
                [1 if stop_training else 0], device=device, dtype=torch.int32
            )
            dist.broadcast(stop_tensor, src=0)
            dist.barrier()
            stop_training = bool(stop_tensor.item())
        if stop_training:
            break

    if distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
