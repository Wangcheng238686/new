#!/usr/bin/env python

import argparse
from datetime import timedelta
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler

from mmdet.structures import DetDataSample
from mmdet.structures.mask import BitmapMasks
from mmengine.structures import InstanceData

# Ensure project root is importable when launched as a file path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.coco_eval_utils import build_coco_gt_and_dt, run_coco_eval


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("portable_sam_fusion")


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

    mask_prob_mean = None
    if hasattr(instances, "mask_prob_mean"):
        mask_prob_mean = instances.mask_prob_mean
        if isinstance(mask_prob_mean, torch.Tensor):
            mask_prob_mean = mask_prob_mean.detach().cpu().numpy()
    mask_prob_ge_03 = None
    if hasattr(instances, "mask_prob_ge_03"):
        mask_prob_ge_03 = instances.mask_prob_ge_03
        if isinstance(mask_prob_ge_03, torch.Tensor):
            mask_prob_ge_03 = mask_prob_ge_03.detach().cpu().numpy()
    mask_prob_ge_05 = None
    if hasattr(instances, "mask_prob_ge_05"):
        mask_prob_ge_05 = instances.mask_prob_ge_05
        if isinstance(mask_prob_ge_05, torch.Tensor):
            mask_prob_ge_05 = mask_prob_ge_05.detach().cpu().numpy()
    mask_prob_ge_07 = None
    if hasattr(instances, "mask_prob_ge_07"):
        mask_prob_ge_07 = instances.mask_prob_ge_07
        if isinstance(mask_prob_ge_07, torch.Tensor):
            mask_prob_ge_07 = mask_prob_ge_07.detach().cpu().numpy()

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
    if mask_prob_mean is not None:
        out["mask_prob_mean"] = mask_prob_mean
    if mask_prob_ge_03 is not None:
        out["mask_prob_ge_03"] = mask_prob_ge_03
    if mask_prob_ge_05 is not None:
        out["mask_prob_ge_05"] = mask_prob_ge_05
    if mask_prob_ge_07 is not None:
        out["mask_prob_ge_07"] = mask_prob_ge_07
    return out


def _summarize_mask_density(all_dt: List[dict]) -> Tuple[float, float]:
    """Return average predicted mask fill ratio and average masks per image."""
    total_ratio = 0.0
    total_masks = 0
    total_images = max(len(all_dt), 1)
    for dt in all_dt:
        masks = dt.get("masks")
        if masks is None or len(masks) == 0:
            continue
        masks = masks.astype(np.float32)
        total_ratio += float(masks.mean()) * len(masks)
        total_masks += len(masks)
    avg_fill_ratio = total_ratio / max(total_masks, 1)
    avg_masks_per_image = total_masks / total_images
    return avg_fill_ratio, avg_masks_per_image


def _summarize_mask_confidence(all_dt: List[dict]) -> Dict[str, float]:
    """Summarize raw predicted mask probability statistics."""
    stats = {
        "mask_mean": 0.0,
        "mask_ge_03": 0.0,
        "mask_ge_05": 0.0,
        "mask_ge_07": 0.0,
    }
    total_masks = 0
    for dt in all_dt:
        mask_prob_mean = dt.get("mask_prob_mean")
        mask_prob_ge_03 = dt.get("mask_prob_ge_03")
        mask_prob_ge_05 = dt.get("mask_prob_ge_05")
        mask_prob_ge_07 = dt.get("mask_prob_ge_07")
        if mask_prob_mean is None or len(mask_prob_mean) == 0:
            continue
        count = len(mask_prob_mean)
        total_masks += count
        stats["mask_mean"] += float(np.mean(mask_prob_mean)) * count
        stats["mask_ge_03"] += float(np.mean(mask_prob_ge_03)) * count
        stats["mask_ge_05"] += float(np.mean(mask_prob_ge_05)) * count
        stats["mask_ge_07"] += float(np.mean(mask_prob_ge_07)) * count
    if total_masks == 0:
        return stats
    for key in list(stats.keys()):
        stats[key] /= total_masks
    return stats


def _init_distributed() -> Dict[str, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    launched_by_torchrun = os.environ.get("LOCAL_RANK", None) is not None
    should_init = (world_size > 1) or launched_by_torchrun
    if should_init and not dist.is_initialized():
        timeout_seconds = int(
            os.environ.get(
                "TORCH_DDP_TIMEOUT_SECONDS",
                os.environ.get("NCCL_TIMEOUT", "600"),
            )
        )
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=max(1, timeout_seconds)),
        )
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


def _set_random_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.use_deterministic_algorithms(False)


def _build_optimizer(
    model: torch.nn.Module,
    lr: float,
    sat_backbone_lr_mult: float,
    sat_other_lr_mult: float,
    drone_lr_mult: float,
    scene_align_lr_mult: float = 2.0,
    weight_decay: float = 0.05,
) -> optim.Optimizer:
    params_sat_backbone = []
    params_sat_other = []
    params_drone = []
    params_scene_align = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "scene_alignment" in name:
            params_scene_align.append(param)
        elif name.startswith("drone_encoder.") or name.startswith("guidance."):
            params_drone.append(param)
        elif name.startswith("contrastive_loss.") or name.startswith("consistency_loss."):
            params_drone.append(param)
        elif name.startswith("backbone.") or name.startswith("shared_image_embedding."):
            params_sat_backbone.append(param)
        else:
            params_sat_other.append(param)

    param_groups = []
    if params_sat_backbone:
        param_groups.append(
            {
                "params": params_sat_backbone,
                "lr": lr * sat_backbone_lr_mult,
                "name": "sat_backbone",
            }
        )
    if params_sat_other:
        param_groups.append(
            {
                "params": params_sat_other,
                "lr": lr * sat_other_lr_mult,
                "name": "sat_other",
            }
        )
    if params_drone:
        param_groups.append(
            {"params": params_drone, "lr": lr * drone_lr_mult, "name": "drone_guidance"}
        )
    if params_scene_align:
        param_groups.append(
            {"params": params_scene_align, "lr": lr * drone_lr_mult * scene_align_lr_mult, "name": "scene_alignment"}
        )

    if not param_groups:
        raise RuntimeError(
            "No trainable parameters found. Check freezing / requires_grad."
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


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_metrics: Dict,
    history: Dict,
    train_args=None,
    ema_state: Optional[Dict] = None,
    early_stop_metric_history: Optional[List[float]] = None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    ckpt = {
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_metrics": best_metrics,
        "history": history,
    }
    if train_args is not None:
        ckpt["train_args"] = vars(train_args)
    if ema_state is not None:
        ckpt["ema_state"] = ema_state
    if early_stop_metric_history is not None:
        ckpt["early_stop_metric_history"] = early_stop_metric_history
    torch.save(ckpt, str(path))


def _load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
) -> Dict:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    model_to_load = model.module if hasattr(model, "module") else model
    model_to_load.load_state_dict(ckpt["model"], strict=False)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt


def _apply_model_overrides(cfg, args):
    if hasattr(cfg.model, "max_scenes") or args.use_drone:
        cfg.model.max_scenes = args.max_scenes

    if args.iimr_enabled == 0:
        if hasattr(cfg.model, "roi_head"):
            if "iimr" in cfg.model.roi_head:
                cfg.model.roi_head.iimr.enabled = False
            else:
                cfg.model.roi_head.iimr = dict(enabled=False)
    elif hasattr(cfg.model, "roi_head"):
        if "iimr" not in cfg.model.roi_head:
            cfg.model.roi_head.iimr = dict(enabled=True)
        if args.iimr_use_topology_memory in (0, 1):
            cfg.model.roi_head.iimr.use_topology_memory = bool(
                args.iimr_use_topology_memory
            )
        if args.iimr_topology_to_decoder in (0, 1):
            cfg.model.roi_head.iimr.topology_to_decoder = bool(
                args.iimr_topology_to_decoder
            )
        if args.iimr_use_gt_topology_tokens in (0, 1):
            cfg.model.roi_head.iimr.use_gt_topology_tokens = bool(
                args.iimr_use_gt_topology_tokens
            )
        if args.iimr_gt_topology_only_first_iter in (0, 1):
            cfg.model.roi_head.iimr.gt_topology_only_first_iter = bool(
                args.iimr_gt_topology_only_first_iter
            )
        if args.iimr_use_dynamic_prompting in (0, 1):
            cfg.model.roi_head.iimr.use_dynamic_prompting = bool(
                args.iimr_use_dynamic_prompting
            )
        if args.iimr_num_uncertainty_points > 0:
            cfg.model.roi_head.iimr.num_uncertainty_points = int(
                args.iimr_num_uncertainty_points
            )
        if 0.0 <= args.iimr_dynamic_balance_ratio <= 1.0:
            cfg.model.roi_head.iimr.dynamic_balance_ratio = float(
                args.iimr_dynamic_balance_ratio
            )
        if args.iimr_dynamic_min_point_distance >= 0:
            cfg.model.roi_head.iimr.dynamic_min_point_distance = int(
                args.iimr_dynamic_min_point_distance
            )
        if args.iimr_dynamic_start_iteration >= 1:
            cfg.model.roi_head.iimr.dynamic_start_iteration = int(
                args.iimr_dynamic_start_iteration
            )
        if args.iimr_dynamic_point_mix >= 0:
            cfg.model.roi_head.iimr.dynamic_point_mix = float(
                args.iimr_dynamic_point_mix
            )
        if args.iimr_dynamic_fusion_mode:
            cfg.model.roi_head.iimr.dynamic_fusion_mode = str(
                args.iimr_dynamic_fusion_mode
            )
        if args.iimr_detach_mask_path in (0, 1):
            cfg.model.roi_head.iimr.detach_mask_path = bool(args.iimr_detach_mask_path)
        if args.iimr_num_iterations > 0:
            cfg.model.roi_head.iimr.num_iterations = int(args.iimr_num_iterations)
        if args.iimr_intermediate_loss_weight >= 0:
            cfg.model.roi_head.iimr.intermediate_loss_weight = float(
                args.iimr_intermediate_loss_weight
            )
        if args.iimr_use_learnable_residual in (0, 1):
            cfg.model.roi_head.iimr.use_learnable_residual = bool(
                args.iimr_use_learnable_residual
            )
        if args.iimr_memory_residual_weight >= 0:
            cfg.model.roi_head.iimr.memory_residual_weight = float(
                args.iimr_memory_residual_weight
            )
        if args.iimr_use_roi_pos_encoding in (0, 1):
            cfg.model.roi_head.iimr.use_roi_pos_encoding = bool(
                args.iimr_use_roi_pos_encoding
            )
        if args.iimr_use_geometry_aware_kv in (0, 1):
            cfg.model.roi_head.iimr.use_geometry_aware_kv = bool(
                args.iimr_use_geometry_aware_kv
            )
        if args.iimr_use_boundary_uncertainty in (0, 1):
            cfg.model.roi_head.iimr.use_boundary_uncertainty = bool(
                args.iimr_use_boundary_uncertainty
            )
        if args.iimr_boundary_kernel_size > 0:
            cfg.model.roi_head.iimr.boundary_kernel_size = int(
                args.iimr_boundary_kernel_size
            )
        if args.iimr_boundary_mix_weight >= 0:
            cfg.model.roi_head.iimr.boundary_mix_weight = float(
                args.iimr_boundary_mix_weight
            )
        if args.iimr_adaptive_balance in (0, 1):
            cfg.model.roi_head.iimr.adaptive_balance = bool(
                args.iimr_adaptive_balance
            )

    if args.rcnn_max_per_img > 0:
        cfg.model.test_cfg.rcnn.max_per_img = int(args.rcnn_max_per_img)
    if args.mask_target_size is not None:
        cfg.model.train_cfg.rcnn.mask_size = tuple(int(v) for v in args.mask_target_size)
    if args.rpn_train_max_per_img > 0:
        cfg.model.train_cfg.rpn_proposal.max_per_img = int(args.rpn_train_max_per_img)
    if args.rpn_test_max_per_img > 0:
        cfg.model.test_cfg.rpn.max_per_img = int(args.rpn_test_max_per_img)
    rpn_assigner = cfg.model.train_cfg.rpn.assigner
    if args.rpn_pos_iou_thr >= 0:
        rpn_assigner.pos_iou_thr = float(args.rpn_pos_iou_thr)
    if args.rpn_neg_iou_thr >= 0:
        rpn_assigner.neg_iou_thr = float(args.rpn_neg_iou_thr)
    if args.rpn_min_pos_iou >= 0:
        rpn_assigner.min_pos_iou = float(args.rpn_min_pos_iou)
    if 0.0 <= args.rcnn_pos_fraction <= 1.0:
        cfg.model.train_cfg.rcnn.sampler.pos_fraction = float(args.rcnn_pos_fraction)
    if args.dense_prompt_mode:
        cfg.model.roi_head.mask_head.dense_prompt_mode = str(args.dense_prompt_mode)
    if args.fpn_type:
        cfg.model.neck.fpn_type = str(args.fpn_type)
    if args.topdown_downstream_head:
        cfg.model.roi_head.topdown_downstream_head = str(args.topdown_downstream_head)


def _resolve_iimr_runtime_handle(model: torch.nn.Module):
    """Return roi_head that owns runtime IIMR switches, or None when unavailable."""
    model_core = model.module if hasattr(model, "module") else model
    roi_head = getattr(model_core, "roi_head", None)
    if roi_head is None:
        return None
    if not hasattr(roi_head, "iimr_enabled"):
        return None
    return roi_head


def _compute_iimr_runtime_state(
    epoch_idx: int,
    base_enabled: bool,
    base_residual_weight: float,
    enable_after_epochs: int,
    residual_warmup_epochs: int,
    residual_start_weight: float,
) -> Tuple[bool, float]:
    """Compute runtime IIMR on/off and residual weight for current epoch."""
    if not base_enabled:
        return False, 0.0

    current_epoch = epoch_idx + 1
    if enable_after_epochs > 0 and current_epoch <= enable_after_epochs:
        return False, residual_start_weight

    enabled = True
    if residual_warmup_epochs <= 0:
        return enabled, base_residual_weight

    active_epoch = current_epoch - enable_after_epochs
    progress = min(1.0, max(0.0, active_epoch / float(residual_warmup_epochs)))
    residual_weight = residual_start_weight + (
        base_residual_weight - residual_start_weight
    ) * progress
    return enabled, residual_weight


def _apply_iimr_runtime_state(
    model: torch.nn.Module,
    iimr_enabled: bool,
    memory_residual_weight: float,
) -> bool:
    """Apply epoch-level runtime IIMR state to roi_head in-place."""
    roi_head = _resolve_iimr_runtime_handle(model)
    if roi_head is None:
        return False

    roi_head.iimr_enabled = bool(iimr_enabled)
    if hasattr(roi_head, "memory_residual_weight"):
        roi_head.memory_residual_weight = float(memory_residual_weight)
    return True


def _build_train_data_loaders(args, distributed: bool, rank: int, world_size: int):
    from data import create_train_loader

    if len(args.multi_scale_img_scale) % 2 != 0:
        raise ValueError("multi_scale_img_scale must have an even number of ints")

    return create_train_loader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        flip_prob=0.5,
        vflip_prob=0.0,
        gaussian_noise_prob=0.0,
        gaussian_noise_std=0.02,
        random_erasing_prob=0.0,
        random_erasing_scale=(0.02, 0.15),
        random_erasing_ratio=(0.3, 3.3),
        image_size=tuple(args.image_size),
        multi_scale_resize_prob=args.multi_scale_resize_prob,
        multi_scale_mode=args.multi_scale_mode,
        multi_scale_img_scale=[
            (args.multi_scale_img_scale[i], args.multi_scale_img_scale[i + 1])
            for i in range(0, len(args.multi_scale_img_scale), 2)
        ],
        use_drone=args.use_drone,
        drone_data_root=args.drone_data_root,
        num_views=args.num_views,
        drone_image_size=tuple(args.drone_image_size),
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        val_ratio=args.val_ratio,
        val_batch_size=args.val_batch_size,
        random_sample=args.random_sample,
        normalize_drone=args.normalize_drone,
        seed=int(args.seed),
        dataset_format=args.dataset_format,
        whu_train_ann_file=args.whu_train_ann_file,
        whu_val_ann_file=args.whu_val_ann_file,
        whu_train_img_subdir=args.whu_train_img_subdir,
        whu_val_img_subdir=args.whu_val_img_subdir,
        whu_single_class=bool(args.whu_single_class),
        whu_enable_category_mapping=bool(args.whu_enable_category_mapping),
        num_workers=getattr(args, 'num_workers', 4),
    )


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
    parser.add_argument("--gpu-id", type=int, default=0)

    parser.add_argument("--use-drone", action="store_true", default=False)
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
        action="store_true",
        default=True,
        help="Normalize drone images",
    )

    parser.add_argument("--use-fsdp", action="store_true")
    parser.add_argument("--fsdp-sharding", type=str, default="FULL_SHARD")
    parser.add_argument("--fsdp-min-num-params", type=int, default=1000000)

    parser.add_argument("--freeze-bn", action="store_true")

    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument(
        "--dataset-format",
        type=str,
        default="labelme",
        choices=["labelme", "whu_coco"],
        help="Dataset parser format. labelme keeps the current behavior; whu_coco loads WHU COCO-style split jsons.",
    )
    parser.add_argument("--whu-train-ann-file", type=str, default="annotations/train.json")
    parser.add_argument("--whu-val-ann-file", type=str, default="annotations/val.json")
    parser.add_argument("--whu-train-img-subdir", type=str, default="train/image")
    parser.add_argument("--whu-val-img-subdir", type=str, default="val/image")
    parser.add_argument("--whu-single-class", type=int, default=1, help="Set 1 to map all WHU instances to label 0")
    parser.add_argument(
        "--whu-enable-category-mapping",
        type=int,
        default=0,
        help="Reserved for future multi-class extension on WHU COCO annotations",
    )
    parser.add_argument(
        "--multi-scale-resize-prob",
        type=float,
        default=0.0,
        help="Probability of mmdet-style multi-scale resize augmentation "
        "applied before the fixed resize-to-image_size. 0 disables.",
    )
    parser.add_argument(
        "--multi-scale-mode",
        type=str,
        default="range",
        choices=["range", "value"],
        help="Multi-scale sampling: 'range' samples long/short edges from "
        "min/max of img_scale tuples (mmdet semantics); 'value' picks "
        "one tuple uniformly.",
    )
    parser.add_argument(
        "--multi-scale-img-scale",
        type=int,
        nargs="+",
        default=[1344, 819, 1344, 1344],
        help="Flattened list of (W,H) pairs, e.g. '1344 819 1344 1344' "
        "encodes [(1344,819),(1344,1344)].",
    )
    parser.add_argument(
        "--eval-category-name",
        type=str,
        default="building",
        help="Single-class category name shown in COCO-style eval output",
    )
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--val-every-n-epochs", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument(
        "--early-stopping-smooth-window",
        type=int,
        default=5,
        help="Moving-average window for early stopping metric",
    )
    parser.add_argument(
        "--early-stopping-min-epochs",
        type=int,
        default=20,
        help="Do not trigger early stopping before this epoch count",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=5e-4,
        help="Minimum smoothed-metric improvement to reset patience",
    )
    parser.add_argument(
        "--early-stopping-metric",
        type=str,
        default="segm_map",
        choices=["segm_map", "val_loss"],
        help="Metric used by smoothed early stopping",
    )
    parser.add_argument("--ema-enabled", type=int, default=1, help="Enable EMA shadow weights")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay in (0,1)")
    parser.add_argument("--ema-update-every", type=int, default=1, help="Update EMA every N optimizer steps")
    parser.add_argument("--ema-eval", type=int, default=1, help="Use EMA shadow weights for validation")
    parser.add_argument("--ema-save-best", type=int, default=1, help="Save best_model using EMA shadow weights")
    parser.add_argument(
        "--ema-eval-start-epoch",
        type=int,
        default=5,
        help="Start using EMA for validation/save from this epoch (1-based)",
    )

    parser.add_argument("--sat-backbone-lr-mult", type=float, default=0.5)
    parser.add_argument("--sat-other-lr-mult", type=float, default=1.0)
    parser.add_argument("--drone-lr-mult", type=float, default=0.5)
    parser.add_argument("--scene-align-lr-mult", type=float, default=2.0, help="Learning rate multiplier for scene alignment parameters")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-scenes", type=int, default=1000, help="Maximum number of scenes for scene-specific alignment")
    parser.add_argument("--warmup-epochs", type=int, default=0, help="Number of warmup epochs for learning rate scheduling")
    parser.add_argument("--warmup-iters", type=int, default=0, help="Number of warmup iterations (overrides warmup-epochs if > 0)")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay for optimizer")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument(
        "--debug-small-object-stats",
        type=int,
        default=0,
        help="Enable structured small/medium/large metric logging and training-time small-object diagnostics",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed for reproducible training/data loading")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers per process")
    parser.add_argument("--deterministic", type=int, default=1, help="Use deterministic CUDA/CUDNN settings when set to 1")
    parser.add_argument("--amp", type=int, default=1, help="Enable AMP mixed precision when set to 1")
    parser.add_argument("--iimr-enabled", type=int, default=1, help="Enable IIMR (1=enabled, 0=disabled)")
    parser.add_argument("--iimr-use-topology-memory", type=int, default=-1, help="Override IIMR topology memory (-1 keeps config, 0 disables, 1 enables)")
    parser.add_argument("--iimr-topology-to-decoder", type=int, default=-1, help="Override topology_to_decoder (-1 keeps config, 0 disables, 1 enables)")
    parser.add_argument("--iimr-use-gt-topology-tokens", type=int, default=-1, help="Override training-time GT topology token injection (-1 keeps config, 0 disables, 1 enables)")
    parser.add_argument("--iimr-gt-topology-only-first-iter", type=int, default=-1, help="Override whether GT topology tokens are used only at the first iteration (-1 keeps config, 0 disables, 1 enables)")
    parser.add_argument("--iimr-use-dynamic-prompting", type=int, default=-1, help="Override IIMR dynamic prompting (-1 keeps config, 0 disables, 1 enables)")
    parser.add_argument("--iimr-num-uncertainty-points", type=int, default=-1, help="Override dynamic prompting uncertainty point count when > 0")
    parser.add_argument("--iimr-dynamic-balance-ratio", type=float, default=-1.0, help="Override the positive-point ratio in uncertainty sampling when in [0,1]")
    parser.add_argument("--iimr-dynamic-min-point-distance", type=int, default=-1, help="Override the minimum pixel distance between sampled uncertainty points when >= 0")
    parser.add_argument("--iimr-dynamic-start-iteration", type=int, default=-1, help="Override the first iteration index that consumes dynamic prompts when >= 1")
    parser.add_argument("--iimr-dynamic-point-mix", type=float, default=-1.0, help="Override the additive strength of point embeddings in dynamic prompting when >= 0")
    parser.add_argument("--iimr-dynamic-fusion-mode", type=str, default="", help="Override dynamic prompt fusion mode (e.g. add, gated)")
    parser.add_argument("--iimr-detach-mask-path", type=int, default=-1, help="Override IIMR detach_mask_path (-1 keeps config, 0 disables, 1 enables)")
    parser.add_argument("--iimr-num-iterations", type=int, default=-1, help="Override IIMR num_iterations when > 0")
    parser.add_argument("--iimr-intermediate-loss-weight", type=float, default=-1.0, help="Override IIMR intermediate loss weight when >= 0")
    parser.add_argument("--iimr-use-learnable-residual", type=int, default=-1, help="Override learnable residual weight (-1 keeps config, 0 disables, 1 enables)")
    parser.add_argument("--iimr-memory-residual-weight", type=float, default=-1.0, help="Override IIMR memory_residual_weight when >= 0")
    parser.add_argument(
        "--iimr-enable-after-epochs",
        type=int,
        default=0,
        help="Delay IIMR activation for the first N epochs (0 disables delay)",
    )
    parser.add_argument(
        "--iimr-residual-warmup-epochs",
        type=int,
        default=0,
        help="Linearly warm up residual weight for N active epochs (0 disables warmup)",
    )
    parser.add_argument(
        "--iimr-residual-start-weight",
        type=float,
        default=0.0,
        help="Residual weight used before IIMR activation and warmup start",
    )
    parser.add_argument("--iimr-use-roi-pos-encoding", type=int, default=-1, help="Override ROI spatial position encoding (-1 keeps config, 0 disables, 1 enables)")
    parser.add_argument("--iimr-use-geometry-aware-kv", type=int, default=-1, help="Override geometry-aware KV injection (-1 keeps config, 0 disables, 1 enables)")
    parser.add_argument("--iimr-use-boundary-uncertainty", type=int, default=-1, help="Override boundary-aware uncertainty sampling (-1 keeps config, 0 disables, 1 enables)")
    parser.add_argument("--iimr-boundary-kernel-size", type=int, default=-1, help="Override boundary kernel size when > 0")
    parser.add_argument("--iimr-boundary-mix-weight", type=float, default=-1.0, help="Override boundary mix weight when >= 0")
    parser.add_argument("--iimr-adaptive-balance", type=int, default=-1, help="Override adaptive balance (-1 keeps config, 0 disables, 1 enables)")
    parser.add_argument("--rcnn-score-thr", type=float, default=-1.0, help="Override test_cfg.rcnn.score_thr when >= 0")
    parser.add_argument("--rcnn-max-per-img", type=int, default=-1, help="Override test_cfg.rcnn.max_per_img when > 0")
    parser.add_argument("--mask-thr-binary", type=float, default=-1.0, help="Override test_cfg.rcnn.mask_thr_binary when >= 0")
    parser.add_argument("--mask-loss-type", type=str, default="", help="Override roi_head.mask_head.loss_mask.type when non-empty")
    parser.add_argument("--mask-loss-pos-weight", type=float, default=-1.0, help="Override roi_head.mask_head.loss_mask.pos_weight when >= 0")
    parser.add_argument("--mask-logit-scale", type=float, default=-1.0, help="Override roi_head.mask_head.mask_logit_scale when > 0")
    parser.add_argument("--mask-logit-bias", type=float, default=0.0, help="Override roi_head.mask_head.mask_logit_bias")
    parser.add_argument(
        "--dense-prompt-mode",
        type=str,
        default="",
        help="Override roi_head.mask_head.dense_prompt_mode (e.g. none, residual)",
    )
    parser.add_argument(
        "--mask-target-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Override train_cfg.rcnn.mask_size",
    )
    parser.add_argument("--rpn-train-max-per-img", type=int, default=-1, help="Override train_cfg.rpn_proposal.max_per_img when > 0")
    parser.add_argument("--rpn-test-max-per-img", type=int, default=-1, help="Override test_cfg.rpn.max_per_img when > 0")
    parser.add_argument("--rpn-pos-iou-thr", type=float, default=-1.0, help="Override train_cfg.rpn.assigner.pos_iou_thr when >= 0")
    parser.add_argument("--rpn-neg-iou-thr", type=float, default=-1.0, help="Override train_cfg.rpn.assigner.neg_iou_thr when >= 0")
    parser.add_argument("--rpn-min-pos-iou", type=float, default=-1.0, help="Override train_cfg.rpn.assigner.min_pos_iou when >= 0")
    parser.add_argument("--rcnn-pos-fraction", type=float, default=-1.0, help="Override train_cfg.rcnn.sampler.pos_fraction when in [0,1]")
    parser.add_argument(
        "--fpn-type",
        type=str,
        default="",
        help="Override neck.fpn_type (e.g. fused, topdown, lateral_enhanced, direct, direct_fpn)",
    )
    parser.add_argument(
        "--topdown-downstream-head",
        type=str,
        default="",
        help="Override roi_head.topdown_downstream_head (e.g. none, dual_level_residual)",
    )

    args = parser.parse_args()

    os.environ["DEBUG_SMALL_OBJECT_STATS"] = "1" if args.debug_small_object_stats == 1 else "0"

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
    import models

    dist_info = _init_distributed()
    distributed = bool(dist_info.get("distributed", 0))
    rank = int(dist_info["rank"])
    global_seed = int(args.seed) + rank
    _set_random_seed(global_seed, deterministic=bool(args.deterministic))

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
            "Reproducibility: seed=%d deterministic=%s amp=%s",
            int(args.seed),
            bool(args.deterministic),
            bool(args.amp) and torch.cuda.is_available(),
        )

    cfg = Config.fromfile(args.config)
    _apply_model_overrides(cfg, args)
    
    model = MODELS.build(cfg.model)
    model.to(device)

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

        model = DDP(
            model,
            device_ids=[int(device.split(":")[1])]
            if device.startswith("cuda")
            else None,
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

    train_loader, val_loader, train_dataset = _build_train_data_loaders(
        args=args,
        distributed=distributed,
        rank=rank,
        world_size=int(dist_info.get("world_size", 1)),
    )

    num_batches_per_epoch = len(train_loader)
    if args.max_train_batches > 0:
        num_batches_per_epoch = min(num_batches_per_epoch, args.max_train_batches)

    if is_main:
        logger.info("Dataset size: %d", len(train_dataset))

    optimizer = _build_optimizer(
        model.module if hasattr(model, "module") else model,
        lr=args.lr,
        sat_backbone_lr_mult=args.sat_backbone_lr_mult,
        sat_other_lr_mult=args.sat_other_lr_mult,
        drone_lr_mult=args.drone_lr_mult,
        scene_align_lr_mult=args.scene_align_lr_mult,
        weight_decay=args.weight_decay,
    )

    # Warmup scheduling: support both epoch-based and iteration-based warmup
    if args.warmup_iters > 0:
        # User-specified iteration-based warmup
        warmup_iters = args.warmup_iters
        if is_main:
            logger.info(f"Using user-specified warmup: {warmup_iters} iterations")
    elif args.warmup_epochs > 0:
        # Epoch-based warmup for better stability with fusion modules
        warmup_iters = args.warmup_epochs * num_batches_per_epoch
        if is_main:
            logger.info(f"Using epoch-based warmup: {args.warmup_epochs} epochs ({warmup_iters} iters)")
    else:
        # Fallback to iteration-based warmup (original behavior)
        warmup_iters = min(50, num_batches_per_epoch)
        if is_main:
            logger.info(f"Using default iteration-based warmup: {warmup_iters} iters")

    cosine_t_max = max(1, args.epochs * num_batches_per_epoch - warmup_iters)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_t_max, eta_min=args.lr * 0.001
    )
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.001, total_iters=warmup_iters
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_iters],
    )

    # AMP scaler
    amp_enabled = bool(args.amp) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 0
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    best_segm_map = 0.0  # Track best segm mAP
    early_counter = 0
    best_early_stop_score = float("-inf")
    early_stop_metric_history: List[float] = []
    history = {"epochs": [], "train_losses": [], "val_losses": [], "val_metrics": [], "learning_rates": []}

    ema = None
    ema_update_steps = 0
    ema_enabled = bool(args.ema_enabled) and (0.0 < float(args.ema_decay) < 1.0)
    if ema_enabled:
        ema = ExponentialMovingAverage(model, decay=float(args.ema_decay))
        if is_main:
            logger.info(
                "EMA enabled: decay=%.6f update_every=%d eval=%s save_best=%s eval_start_epoch=%d",
                float(args.ema_decay),
                max(1, int(args.ema_update_every)),
                bool(args.ema_eval),
                bool(args.ema_save_best),
                max(1, int(args.ema_eval_start_epoch)),
            )

    if args.resume_from:
        ckpt = _load_checkpoint(Path(args.resume_from), model, optimizer, scheduler)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best = ckpt.get("best_metrics", {}) or {}
        best_val_loss = float(best.get("best_val_loss", best_val_loss))
        best_train_loss = float(best.get("best_train_loss", best_train_loss))
        best_segm_map = float(best.get("best_segm_map", best_segm_map))
        best_early_stop_score = float(best.get("best_early_stop_score", best_early_stop_score))
        history = ckpt.get("history", history) or history
        early_stop_metric_history = list(ckpt.get("early_stop_metric_history", early_stop_metric_history))
        if ema is not None and "ema_state" in ckpt:
            ema.load_state_dict(ckpt["ema_state"])
            if is_main:
                logger.info("Loaded EMA state from checkpoint.")
        if is_main:
            logger.info(
                "Resumed from %s (start_epoch=%d, best_segm_map=%.4f)",
                args.resume_from,
                start_epoch,
                best_segm_map,
            )

    roi_head_runtime = _resolve_iimr_runtime_handle(model)
    base_iimr_enabled = bool(getattr(roi_head_runtime, "iimr_enabled", False))
    base_iimr_residual_weight = float(
        getattr(roi_head_runtime, "memory_residual_weight", 0.0)
    )
    if args.iimr_memory_residual_weight >= 0:
        base_iimr_residual_weight = float(args.iimr_memory_residual_weight)
    residual_start_weight = float(args.iimr_residual_start_weight)

    if is_main:
        logger.info(
            "P0 IIMR schedule: base_enabled=%s base_residual_weight=%.4f "
            "enable_after_epochs=%d residual_warmup_epochs=%d residual_start_weight=%.4f",
            base_iimr_enabled,
            base_iimr_residual_weight,
            int(args.iimr_enable_after_epochs),
            int(args.iimr_residual_warmup_epochs),
            residual_start_weight,
        )

    for epoch in range(start_epoch, args.epochs):
        runtime_iimr_enabled, runtime_residual_weight = _compute_iimr_runtime_state(
            epoch_idx=epoch,
            base_enabled=base_iimr_enabled,
            base_residual_weight=base_iimr_residual_weight,
            enable_after_epochs=max(0, int(args.iimr_enable_after_epochs)),
            residual_warmup_epochs=max(0, int(args.iimr_residual_warmup_epochs)),
            residual_start_weight=residual_start_weight,
        )
        applied_runtime_state = _apply_iimr_runtime_state(
            model=model,
            iimr_enabled=runtime_iimr_enabled,
            memory_residual_weight=runtime_residual_weight,
        )
        if is_main and applied_runtime_state:
            logger.info(
                "Epoch %d IIMR runtime: enabled=%s memory_residual_weight=%.4f",
                epoch + 1,
                runtime_iimr_enabled,
                runtime_residual_weight,
            )

        if distributed:
            sampler = getattr(train_loader, "sampler", None)
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)

        model.train()
        if args.freeze_bn:
            _set_norm_eval(model)

        total_loss = 0.0
        loss_meter = {}
        num_batches = 0
        roi_scale_grad_abs_sum = 0.0
        roi_scale_grad_steps = 0
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

            # Synchronize data preprocessing for both images and ground truth
            processed = data_preprocessor(
                {"inputs": imgs, "data_samples": data_samples}, training=True
            )
            imgs = processed["inputs"]
            data_samples = processed["data_samples"]

            if args.use_drone and "drone_images" in batch:
                model_inputs = {
                    "sat": imgs,
                    "drone_images": batch["drone_images"].to(device),
                    "intrinsics": batch["intrinsics"].to(device),
                    "extrinsics": batch["extrinsics"].to(device),
                }
                scene_indices = batch.get("scene_indices")
                if scene_indices is not None:
                    scene_indices = scene_indices.to(device)
            else:
                model_inputs = imgs
                scene_indices = None

            model_for_loss = model.module if hasattr(model, "module") else model

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                loss_dict = model_for_loss.loss(model_inputs, data_samples, scene_indices=scene_indices)
                if is_main and epoch == start_epoch and batch_idx == 0:
                    logger.info(
                        "First train batch loss keys: %s",
                        ", ".join(sorted(loss_dict.keys())),
                    )
                loss_parts = []
                for k, v in loss_dict.items():
                    if k.startswith("debug_"):
                        continue
                    if "loss" not in k:
                        continue
                    if isinstance(v, torch.Tensor):
                        loss_parts.append(v)
                    elif isinstance(v, (list, tuple)):
                        loss_parts.extend(x for x in v if isinstance(x, torch.Tensor))
                loss = sum(loss_parts) if loss_parts else torch.tensor(0.0, device=device)
                loss_for_backward = loss / grad_accum

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

            if not torch.isfinite(loss):
                if is_main:
                    logger.error("Non-finite loss detected (epoch=%d).", epoch + 1)
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss_for_backward).backward()

            # Step optimizer every grad_accum batches or at the last batch
            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == train_batches_limit:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

                # Track ROI-PE scale gradient before optimizer step clears grads.
                model_for_stats = model.module if hasattr(model, "module") else model
                named_params = dict(model_for_stats.named_parameters())
                roi_scale_param = named_params.get("roi_head.iterative_reasoner.roi_pos_scale")
                if roi_scale_param is not None and roi_scale_param.grad is not None:
                    roi_scale_grad_abs_sum += float(roi_scale_param.grad.detach().abs().mean().item())
                    roi_scale_grad_steps += 1

                scaler.step(optimizer)
                scaler.update()
                ema_update_steps += 1
                if ema is not None and (ema_update_steps % max(1, int(args.ema_update_every)) == 0):
                    ema.update(model)
                optimizer.zero_grad(set_to_none=True)
                # Step scheduler after optimizer step (iteration-level)
                scheduler.step()

            total_loss += float(loss.detach().item())
            num_batches += 1

            # Release intermediate tensors to prevent memory fragmentation
            del loss_dict, loss, loss_for_backward, imgs, data_samples
            if not args.use_drone:
                del model_inputs
        avg_loss = total_loss / max(1, num_batches)

        # Collect LRs for all groups
        lr_groups = {}
        for group in optimizer.param_groups:
            name = group.get("name", "unknown")
            lr_groups[name] = group["lr"]

        if is_main:
            loss_str = []
            debug_str = []
            for k, v in loss_meter.items():
                avg_k = v / max(1, num_batches)
                if k.startswith("debug_"):
                    debug_str.append(f"{k}={avg_k:.4f}")
                else:
                    loss_str.append(f"{k}={avg_k:.4f}")
            logger.info(f"Epoch {epoch + 1} detailed losses: {', '.join(loss_str)}")
            if debug_str:
                logger.info(f"Epoch {epoch + 1} mask debug: {', '.join(debug_str)}")
            # Debug: log current learning rates
            lr_str = ", ".join([f"{k}={v:.2e}" for k, v in lr_groups.items()])
            logger.info(f"Epoch {epoch + 1} learning rates: {lr_str}")

            model_for_stats = model.module if hasattr(model, "module") else model
            named_params = dict(model_for_stats.named_parameters())
            roi_scale_param = named_params.get("roi_head.iterative_reasoner.roi_pos_scale")
            roi_l4_w = named_params.get("roi_head.iterative_reasoner.roi_pos_encoder.4.weight")
            roi_l4_b = named_params.get("roi_head.iterative_reasoner.roi_pos_encoder.4.bias")
            if roi_scale_param is not None:
                roi_scale_val = float(roi_scale_param.detach().item())
                roi_scale_grad_mean = roi_scale_grad_abs_sum / max(1, roi_scale_grad_steps)
                roi_l4_w_abs = float(roi_l4_w.detach().abs().mean().item()) if roi_l4_w is not None else float("nan")
                roi_l4_b_abs = float(roi_l4_b.detach().abs().mean().item()) if roi_l4_b is not None else float("nan")
                logger.info(
                    "Epoch %d roi-pe stats: scale=%.6e scale_grad_abs_mean=%.6e l4_w_abs_mean=%.6e l4_b_abs_mean=%.6e",
                    epoch + 1,
                    roi_scale_val,
                    roi_scale_grad_mean,
                    roi_l4_w_abs,
                    roi_l4_b_abs,
                )

        val_loss = None
        val_metrics = None
        ema_applied_for_eval = False
        use_ema_this_epoch = (
            ema is not None
            and bool(args.ema_eval)
            and (epoch + 1) >= max(1, int(args.ema_eval_start_epoch))
        )
        if val_loader is not None and (epoch + 1) % args.val_every_n_epochs == 0:
            if use_ema_this_epoch:
                ema.apply_to(model)
                ema_applied_for_eval = True

            # Only rank 0 runs validation; other ranks wait at barrier.
            # This avoids OOM when all GPUs run validation concurrently.
            val_metrics = {}
            val_loss = None

            if is_main:
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

                        # Loss needs positive GT, but COCO evaluation must still
                        # include GT-empty images so false positives are counted.
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

                        if args.use_drone and "drone_images" in batch:
                            model_inputs = {
                                "sat": imgs_processed,
                                "drone_images": batch["drone_images"].to(device),
                                "intrinsics": batch["intrinsics"].to(device),
                                "extrinsics": batch["extrinsics"].to(device),
                            }
                            scene_indices = batch.get("scene_indices")
                            if scene_indices is not None:
                                scene_indices = scene_indices.to(device)
                        else:
                            model_inputs = imgs_processed
                            scene_indices = None

                        model_for_eval = model.module if hasattr(model, "module") else model

                        # ---- Compute loss ----
                        if val_batch_has_valid_gt:
                            loss_dict = model_for_eval.loss(model_inputs, data_samples, scene_indices=scene_indices)
                            val_loss_parts = []
                            for k, v in loss_dict.items():
                                if "loss" not in k.lower():
                                    continue
                                if isinstance(v, torch.Tensor):
                                    val_loss_parts.append(v)
                                elif isinstance(v, (list, tuple)):
                                    val_loss_parts.extend(x for x in v if isinstance(x, torch.Tensor))
                            loss = sum(val_loss_parts) if val_loss_parts else torch.tensor(0.0, device=device)
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

                            # Predictions - from output (InstanceData)
                            pred_np = _extract_instances_numpy(output, img_shape)
                            all_dt.append(pred_np)

                            all_img_metas.append(meta)

                if skipped_no_gt_val_loss_batches > 0:
                    logger.warning(
                        (
                            "Epoch %d skipped val loss on batches with no valid GT: "
                            "%d/%d (%.2f%%); these batches were still included in COCO evaluation"
                        ),
                        epoch + 1,
                        skipped_no_gt_val_loss_batches,
                        seen_val_batches,
                        100.0 * skipped_no_gt_val_loss_batches / max(1, seen_val_batches),
                    )

                val_loss = total_val_loss / max(1, val_batches)

                # ---- COCO-style evaluation ----
                total_gt = sum(len(g["labels"]) for g in all_gt)
                total_dt = sum(len(d["scores"]) for d in all_dt if "scores" in d)
                avg_mask_fill_ratio, avg_masks_per_image = _summarize_mask_density(all_dt)
                mask_conf_stats = _summarize_mask_confidence(all_dt)

                logger.info(
                    (
                        "Validation samples: GT=%d, DT=%d, avg_masks_per_image=%.2f, "
                        "avg_mask_fill=%.4f, mask_mean=%.4f, ge03=%.4f, ge05=%.4f, ge07=%.4f"
                    ),
                    total_gt,
                    total_dt,
                    avg_masks_per_image,
                    avg_mask_fill_ratio,
                    mask_conf_stats["mask_mean"],
                    mask_conf_stats["mask_ge_03"],
                    mask_conf_stats["mask_ge_05"],
                    mask_conf_stats["mask_ge_07"],
                )

                if total_gt > 0:
                    coco_gt, coco_dt = build_coco_gt_and_dt(
                        all_gt,
                        all_dt,
                        all_img_metas,
                        category_name=args.eval_category_name,
                    )

                    bbox_metrics = run_coco_eval(coco_gt, coco_dt, iou_type="bbox")
                    val_metrics.update(bbox_metrics)
                    logger.info(f"Validation bbox/mAP: {bbox_metrics.get('bbox/mAP', 0.0):.4f}")

                    segm_metrics = run_coco_eval(coco_gt, coco_dt, iou_type="segm")
                    val_metrics.update(segm_metrics)
                    logger.info(f"Validation segm/mAP: {segm_metrics.get('segm/mAP', 0.0):.4f}")

                    # Always log scale-specific metrics for small object tracking
                    segm_ap_s = segm_metrics.get("segm/mAP_s", 0.0)
                    segm_ap_m = segm_metrics.get("segm/mAP_m", 0.0)
                    segm_ap_l = segm_metrics.get("segm/mAP_l", 0.0)
                    segm_ar_s = segm_metrics.get("segm/AR_s", 0.0)
                    logger.info(
                        "segm AP_small=%.4f AP_medium=%.4f AP_large=%.4f AR_small=%.4f",
                        segm_ap_s, segm_ap_m, segm_ap_l, segm_ar_s,
                    )

                    # F-0: maxDets diagnostic — check if size metrics are truncated
                    segm_metrics_300 = run_coco_eval(coco_gt, coco_dt, iou_type="segm", max_dets=[1, 10, 100, 300])
                    ar_small_100 = segm_metrics.get("segm/AR_s", 0.0)
                    ar_small_300 = segm_metrics_300.get("segm/AR_s", 0.0)
                    logger.info(
                        "[maxDets diagnostic] segm AR_small: maxDets=100 → %.4f, maxDets=300 → %.4f, Δ=%.4f",
                        ar_small_100, ar_small_300, ar_small_300 - ar_small_100,
                    )
                    if args.debug_small_object_stats == 1:
                        logger.info(
                            (
                                "Size metrics | "
                                "bbox: small=%.4f medium=%.4f large=%.4f | "
                                "segm: small=%.4f medium=%.4f large=%.4f | "
                                "recall: bbox_small=%.4f bbox_medium=%.4f bbox_large=%.4f | "
                                "segm_small=%.4f segm_medium=%.4f segm_large=%.4f"
                            ),
                            bbox_metrics.get("bbox/mAP_s", 0.0),
                            bbox_metrics.get("bbox/mAP_m", 0.0),
                            bbox_metrics.get("bbox/mAP_l", 0.0),
                            segm_metrics.get("segm/mAP_s", 0.0),
                            segm_metrics.get("segm/mAP_m", 0.0),
                            segm_metrics.get("segm/mAP_l", 0.0),
                            bbox_metrics.get("bbox/AR_s", 0.0),
                            bbox_metrics.get("bbox/AR_m", 0.0),
                            bbox_metrics.get("bbox/AR_l", 0.0),
                            segm_metrics.get("segm/AR_s", 0.0),
                            segm_metrics.get("segm/AR_m", 0.0),
                            segm_metrics.get("segm/AR_l", 0.0),
                        )

            if ema_applied_for_eval:
                ema.restore(model)
                ema_applied_for_eval = False

            # Synchronize all ranks after validation
            if distributed:
                dist.barrier()

        # Reduce allocator fragmentation across long runs, especially after
        # validation forward/loss/predict passes.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

        # Keep early-stop decision consistent across ranks to avoid
        # collective order mismatch in the next epoch.
        should_stop = False

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
            current_bbox_map = 0.0
            current_segm_map = 0.0
            if val_metrics is not None and "bbox/mAP" in val_metrics:
                current_bbox_map = val_metrics["bbox/mAP"]
            if val_metrics is not None and "segm/mAP" in val_metrics:
                current_segm_map = val_metrics["segm/mAP"]
                if current_segm_map > best_segm_map:
                    best_segm_map = current_segm_map
                    is_best = True
                    if is_main:
                        logger.info(f"New best segm/mAP: {best_segm_map:.4f}")

            if val_loss is not None:
                if args.early_stopping_metric == "val_loss":
                    raw_early_metric = -float(val_loss)
                else:
                    raw_early_metric = float(current_segm_map)

                early_stop_metric_history.append(raw_early_metric)
                smooth_window = max(1, int(args.early_stopping_smooth_window))
                smooth_metric = float(np.mean(early_stop_metric_history[-smooth_window:]))
                min_delta = float(args.early_stopping_min_delta)

                if smooth_metric > (best_early_stop_score + min_delta):
                    best_early_stop_score = smooth_metric
                    early_counter = 0
                elif (epoch + 1) >= int(args.early_stopping_min_epochs):
                    early_counter += 1

                if is_main:
                    logger.info(
                        "Early-stop metric (%s): raw=%.4f smooth(%d)=%.4f best_smooth=%.4f counter=%d",
                        args.early_stopping_metric,
                        raw_early_metric,
                        smooth_window,
                        smooth_metric,
                        best_early_stop_score,
                        early_counter,
                    )

            best_metrics = {
                "best_train_loss": best_train_loss,
                "best_val_loss": best_val_loss,
                "best_segm_map": best_segm_map,
                "best_early_stop_score": best_early_stop_score,
            }

            # 注意：已禁用每个 epoch 的 checkpoint 保存，只保存 best_model.pth
            # if (epoch + 1) % args.val_every_n_epochs == 0:
            #     _save_checkpoint(
            #         checkpoint_dir / f"epoch_{epoch + 1}.pth",
            #         model,
            #         optimizer,
            #         scheduler,
            #         epoch,
            #         best_metrics,
            #         history,
            #     )
            #     if is_main:
            #         logger.info(f"Saved checkpoint: epoch_{epoch + 1}.pth")

            # Save best model if segm/mAP improved
            if is_best:
                use_ema_for_save = (
                    ema is not None
                    and bool(args.ema_save_best)
                    and (epoch + 1) >= max(1, int(args.ema_eval_start_epoch))
                )
                ema_state_to_save = ema.state_dict() if ema is not None else None
                ema_applied_for_save = False
                if use_ema_for_save:
                    ema.apply_to(model)
                    ema_applied_for_save = True
                _save_checkpoint(
                    checkpoint_dir / "best_model.pth",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_metrics,
                    history,
                    train_args=args,
                    ema_state=ema_state_to_save,
                    early_stop_metric_history=early_stop_metric_history,
                )
                if ema_applied_for_save:
                    ema.restore(model)
                if is_main:
                    saved_mode = "ema" if use_ema_for_save else "raw"
                    logger.info(
                        "Saved best model with segm/mAP: %.4f (weights=%s)",
                        best_segm_map,
                        saved_mode,
                    )

            # Logging
            if val_loss is not None:
                lr_str = " ".join([f"{k}={v:.2e}" for k, v in lr_groups.items()])
                bbox_info = f" bbox/mAP={current_bbox_map:.4f}" if current_bbox_map > 0 else ""
                segm_info = f" segm/mAP={current_segm_map:.4f}" if current_segm_map > 0 else ""
                logger.info(
                    "Epoch %d/%d | train=%.6f val=%.6f%s%s | best_segm=%.4f | %s",
                    epoch + 1,
                    args.epochs,
                    avg_loss,
                    val_loss,
                    bbox_info,
                    segm_info,
                    best_segm_map,
                    lr_str,
                )
                if args.early_stopping_patience > 0 and early_counter >= int(
                    args.early_stopping_patience
                ):
                    logger.info(
                        "Early stopping triggered (patience=%d, metric=%s, min_epochs=%d)",
                        int(args.early_stopping_patience),
                        args.early_stopping_metric,
                        int(args.early_stopping_min_epochs),
                    )
                    should_stop = True
            else:
                lr_str = " ".join([f"{k}={v:.2e}" for k, v in lr_groups.items()])
                logger.info(
                    "Epoch %d/%d | train=%.6f | %s",
                    epoch + 1,
                    args.epochs,
                    avg_loss,
                    lr_str,
                )

        if distributed and dist.is_initialized():
            stop_tensor = torch.tensor(
                [1 if should_stop else 0],
                device=device,
                dtype=torch.int32,
            )
            dist.broadcast(stop_tensor, src=0)
            should_stop = bool(stop_tensor.item())

        if should_stop:
            break

    if distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
