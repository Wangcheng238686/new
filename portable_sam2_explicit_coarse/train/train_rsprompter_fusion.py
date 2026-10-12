#!/usr/bin/env python

import argparse
import gc
import logging
import math
import time
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch import Tensor
from torch.utils.data.distributed import DistributedSampler

from mmdet.structures import DetDataSample
from mmdet.structures.mask import BitmapMasks
from mmengine.structures import InstanceData


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("portable_sam_fusion")


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
    # Lazy import: the project-root sys.path bootstrap runs inside main(); this
    # module-level helper must not bind project packages at import time.
    from utils.coco_eval_utils import _masks_to_rles

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


def _capture_git_state() -> Dict[str, str]:
    """Fingerprint the executed code: HEAD, dirty flag, tracked-diff sha.

    Complements the runner's resolved_hyperparameters git_* lines so both the
    training log and every checkpoint pin the exact working-tree state, not
    just the commit (a dirty tree silently diverging from HEAD is what made
    the resumed paper run hard to trace).
    """
    import subprocess

    def _git(*argv: str) -> str:
        try:
            return subprocess.run(
                ["git", *argv],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        except Exception:
            return ""

    head = _git("rev-parse", "HEAD") or "unknown"
    status = _git("status", "--porcelain")
    dirty = "1" if status else "0"
    code_files = ",".join(
        line.split(maxsplit=1)[1].strip()
        for line in status.splitlines()
        if line.split(maxsplit=1)[-1].strip().endswith((".py", ".sh", ".md"))
    )[:400]
    try:
        diff_sha = subprocess.run(
            ["git", "diff", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        import hashlib

        diff_sha16 = (
            hashlib.sha256(diff_sha.stdout.encode("utf-8")).hexdigest()[:16]
            if diff_sha.stdout
            else "none"
        )
    except Exception:
        diff_sha16 = "unknown"
    return {
        "git_commit": head,
        "git_dirty": dirty,
        "git_dirty_code_files": code_files or "none",
        "git_diff_sha16": diff_sha16,
    }


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
    mask_decoder_lr_mult: float = 0.05,
    no_mask_lr_mult: float = 0.1,
    prompt_encoder_lr_mult: float = 0.1,
    shape_prior_lr_mult: float = 1.0,
    decoder_tail_lr_mult: float = 1.0,
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
    params_iou_head = []
    names_iou_head = []
    params_decoder_tail = []
    names_decoder_tail = []
    params_bbox_head = []
    names_bbox_head = []
    params_sat_other = []
    names_sat_other = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        match_name = name.replace("_fsdp_wrapped_module.", "").replace("module.", "")
        if "roi_head.mask_head.decoder_tail_refiner" in match_name:
            params_decoder_tail.append(param)
            names_decoder_tail.append(name)
        elif match_name.startswith("backbone.") or match_name.startswith("shared_image_embedding."):
            params_sat_backbone.append(param)
            names_sat_backbone.append(name)
        elif "roi_head.mask_head.mask_decoder.decoder.iou_prediction_head" in match_name:
            # The IoU head keeps its own group (historical C2-era contract)
            # even though the quality-rerank module is gone.
            params_iou_head.append(param)
            names_iou_head.append(name)
        elif "roi_head.mask_head.mask_decoder" in match_name:
            params_mask_decoder.append(param)
            names_mask_decoder.append(name)
        elif "roi_head.mask_head.no_mask_embed" in match_name:
            params_no_mask.append(param)
            names_no_mask.append(name)
        elif "roi_head.mask_head.prompt_encoder" in match_name:
            params_prompt_encoder.append(param)
            names_prompt_encoder.append(name)
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
    if params_iou_head:
        param_groups.append({"params": params_iou_head, "lr": lr,
                             "name": "iou_head"})
        audit_groups.append(("iou_head", params_iou_head, names_iou_head, lr, weight_decay))
    if params_decoder_tail:
        param_groups.append({"params": params_decoder_tail, "lr": lr * decoder_tail_lr_mult,
                             "weight_decay": weight_decay, "name": "decoder_tail"})
        audit_groups.append(("decoder_tail", params_decoder_tail, names_decoder_tail,
                             lr * decoder_tail_lr_mult, weight_decay))
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


# Zero-init residual auxiliary heads whose gradients were side-loaded into
# the single global grad-norm clip.  Audit 2026-09-14: with the legacy clip
# these heads usurp the base's budget (tail EMA clipped-grad L2 0.158 of the
# 0.5 norm at E220 while a0_sg saturates it base-only at 0.458), imposing a
# one-directional, time-varying downscale of base gradients from step 1 --
# the mechanism behind the persistent bbox deficits of every auxiliary-head
# arm (a3sg/a3sgt/a3sgtm/densecapres256) while the coupling-free tail-only
# heat-start passes both gates.  P2/renderer lineages keep the legacy clip.
_AUX_CLIP_MODULE_PREFIXES = (
    "roi_head.mask_head.decoder_tail_refiner.",
)


def _partition_aux_clip_params(model: torch.nn.Module):
    """Split parameters into (base, aux_groups) for decoupled grad clipping.

    aux_groups has one entry per aux module actually present.  Params whose
    grad is None (e.g. the frozen base in tail-only mode) are skipped by
    clip_grad_norm_ itself, matching legacy semantics of clipping
    ``model.parameters()`` over whichever params carry grads.
    """
    named = list(_get_model_core(model).named_parameters())
    groups = [
        [param for name, param in named if name.startswith(prefix)]
        for prefix in _AUX_CLIP_MODULE_PREFIXES
    ]
    groups = [group for group in groups if group]
    aux_ids = {id(param) for group in groups for param in group}
    base = [param for _, param in named if id(param) not in aux_ids]
    return base, groups


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
        model_core = _get_model_core(model)
        state = model_core.state_dict()
        self.ema_state = {k: v.detach().clone() for k, v in state.items()}
        self.backup_state = None
        # Frozen tensors (params with requires_grad=False, plus buffers) must
        # be copied verbatim, never averaged: averaging a constant across tens
        # of thousands of steps accumulates fp32 rounding drift (~1e-6 per
        # the fast-150 run), which trips strict pretrained-value contract
        # checks (no_mask_embed) when the EMA state is loaded for inference.
        _frozen = {n for n, p in model_core.named_parameters() if not p.requires_grad}
        _frozen.update(n for n, _ in model_core.named_buffers())
        self._frozen_keys = frozenset(k for k in state if k in _frozen)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        model_state = _get_model_core(model).state_dict()
        for key, value in model_state.items():
            value_detached = value.detach()
            if key not in self.ema_state:
                self.ema_state[key] = value_detached.clone()
                continue
            ema_value = self.ema_state[key]
            if (
                torch.is_floating_point(value_detached)
                and key not in self._frozen_keys
            ):
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
        "decoder_tail_refiner": getattr(
            mask_head, "decoder_tail_refiner", None
        ) if mask_head is not None else None,
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


# Telemetry prefixes the prompt-debug channel prints (SP points, COARSE
# aux loss, GATE/DENSE dense channel, TAIL margin tail, BOX/JITTER prompt
# box bus, CONTEXT coarse head).
_PROMPT_TELEMETRY_PREFIXES = {"SP", "COARSE", "JITTER", "GATE", "DENSE",
                              "TAIL", "CONTEXT", "BOX"}


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
    "DENSECAP/roi_count",
    "DENSECAP/delta_abs_mean",
    "DENSECAP/delta_nonzero_ratio",
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




_P2BR_R1_GRAD_FIELDS = (
    "probe_count",
    "corrective_grad_sq_sum",
    "keep_grad_sq_sum",
    "gradient_dot_sum",
)


_DENSECAP_GRAD_BRANCHES = ("trunk", "output")
_DENSECAP_GRAD_TERMS = ("final_mask", "aux")


def _verify_ddp_parameter_sync_after_first_update(
    model: torch.nn.Module, device: torch.device, distributed: bool
) -> Optional[float]:
    """Return the largest rank-to-mean parameter deviation after one update.

    This is deliberately run once after the first nonzero-learning-rate optimizer step.  It is a
    fail-fast audit for the DDP wrapper path, not an every-step reduction.
    """
    if not (distributed and dist.is_initialized()):
        return None
    world_size = float(dist.get_world_size())
    local_max = torch.zeros((), dtype=torch.float32, device=device)
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        rank_mean = parameter.detach().float().clone()
        dist.all_reduce(rank_mean, op=dist.ReduceOp.SUM)
        rank_mean.div_(world_size)
        local_max = torch.maximum(
            local_max, (parameter.detach().float() - rank_mean).abs().max()
        )
    dist.all_reduce(local_max, op=dist.ReduceOp.MAX)
    return float(local_max.item())


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
    allowed_missing_prefixes: Tuple[str, ...] = (),
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
    # Historical legacy-key migration (shape_prompt_scale era) was removed
    # with its arms; a3sgm checkpoints load with zero migrations.
    migrations = 0
    migration_pairs: list = []

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
    if allowed_missing_prefixes:
        illegal_missing = [
            key for key in load_result.missing_keys
            if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
        ]
        if illegal_missing or shape_mismatch or unexpected:
            raise RuntimeError(
                "cross-architecture init whitelist violated: "
                f"missing={illegal_missing}, shape_mismatch={shape_mismatch}, unexpected={unexpected}"
            )
    _assert_no_mask_embedding_contract(model_to_load, "INIT_FROM checkpoint load")
    return {
        "loaded": len(compatible),
        "excluded": excluded,
        "shape_mismatch": shape_mismatch,
        "unexpected": unexpected,
        "missing": len(load_result.missing_keys),
        "missing_keys": tuple(load_result.missing_keys),
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


def _shard_val_loader(val_loader, rank: int, world_size: int) -> Tuple[Any, int]:
    """Rebuild the validation loader so each rank owns batches rank, rank+W, ...

    The val dataset pipeline is deterministic (no flip/noise/erasing/multi-scale)
    and the loader never shuffles, so batching the same Subset indices with the
    same batch_size/collate reproduces the exact per-batch composition of the
    unsharded loader: per-image tensors and padding are unchanged and only the
    ownership of batches differs.  Returns (shard_loader, total_images).
    """
    dataset = val_loader.dataset
    batch_size = val_loader.batch_size or 1
    total = len(dataset)
    num_batches = math.ceil(total / batch_size)
    owned_indices: List[int] = []
    for b in range(rank, num_batches, world_size):
        owned_indices.extend(range(b * batch_size, min((b + 1) * batch_size, total)))
    from torch.utils.data import DataLoader, Subset

    shard = DataLoader(
        Subset(dataset, owned_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=val_loader.num_workers,
        collate_fn=val_loader.collate_fn,
        pin_memory=True,
        drop_last=False,
        persistent_workers=val_loader.num_workers > 0 and len(owned_indices) > 0,
    )
    return shard, total


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

    parser.add_argument(
        "--use-whu-coco",
        action="store_true",
        default=False,
        help="使用 WHU 公开数据集（COCO 格式，自带 train/validation 划分）",
    )
    parser.add_argument(
        "--use-vhr10-coco",
        action="store_true",
        default=False,
        help="使用 NWPU VHR-10 数据集（COCO 实例掩码版，10 类）",
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


    parser.add_argument("--freeze-bn", action="store_true")

    parser.add_argument("--image-size", type=int, nargs=2, default=[512, 512])
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
    parser.add_argument("--mask-decoder-lr-mult", type=float, default=1.0)
    parser.add_argument("--no-mask-lr-mult", type=float, default=1.0)
    parser.add_argument("--prompt-encoder-lr-mult", type=float, default=0.1)
    parser.add_argument("--shape-prior-lr-mult", type=float, default=1.0)
    parser.add_argument("--decoder-tail-lr-mult", type=float, default=1.0)
    parser.add_argument("--decoupled-aux-clip", action="store_true",
                        help="clip base params and each aux head separately at "
                             "max_norm 0.5 instead of one global clip; keeps the "
                             "base's clip semantics bit-identical to aux-free arms")
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--warmup-epochs", type=int, default=0, help="Number of warmup epochs for learning rate scheduling")
    parser.add_argument("--warmup-iters", type=int, default=100, help="Number of warmup optimizer steps (overrides warmup-epochs if > 0)")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay for optimizer")
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
    parser.add_argument("--prompt-debug-stats", type=int, choices=[0, 1], default=0,
                        help="Print shape-point/coarse-mask prompt diagnostics")
    parser.add_argument("--prompt-debug-stats-interval", type=int, default=50,
                        help="Training iteration interval for prompt diagnostics")
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

    # Canonical multi-class category tables live in utils.coco_eval_utils so
    # the trainer and checkpoint inference share one definition.
    # Canonical COCO-eval implementation lives in utils.coco_eval_utils
    # (single source for trainer, checkpoint inference and probes).
    from utils.coco_eval_utils import (
        VHR10_CATEGORIES,
        _masks_to_rles,
        build_coco_gt_and_dt,
        run_coco_eval,
    )

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

    model = MODELS.build(cfg.model)
    model.to(device)

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

    if args.freeze_bn:
        _set_norm_eval(model)

    if distributed:
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

    # --- train-time augmentation knobs (env-driven; defaults preserve the
    # historical hflip-only protocol, individual wrappers opt in) ---
    train_flip_prob = float(os.environ.get("TRAIN_FLIP_PROB", "0.5"))
    train_vflip_prob = float(os.environ.get("TRAIN_VFLIP_PROB", "0.0"))
    train_ms_resize_prob = float(
        os.environ.get("TRAIN_MULTI_SCALE_RESIZE_PROB", "0.0")
    )
    train_ms_mode = os.environ.get("TRAIN_MULTI_SCALE_MODE", "value")
    train_ms_img_scale_spec = os.environ.get(
        "TRAIN_MULTI_SCALE_IMG_SCALE",
        "896:896,960:960,1024:1024,1088:1088,1152:1152",
    )
    train_ms_img_scale = None
    if train_ms_img_scale_spec.strip():
        train_ms_img_scale = []
        for part in train_ms_img_scale_spec.split(","):
            w_text, h_text = part.strip().split(":", 1)
            train_ms_img_scale.append((int(w_text), int(h_text)))
    for name, value in (
        ("TRAIN_FLIP_PROB", train_flip_prob),
        ("TRAIN_VFLIP_PROB", train_vflip_prob),
        ("TRAIN_MULTI_SCALE_RESIZE_PROB", train_ms_resize_prob),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0,1], got {value}")
    if train_ms_mode not in ("value", "range"):
        raise ValueError(
            f"TRAIN_MULTI_SCALE_MODE must be value or range, got {train_ms_mode}"
        )

    train_loader, val_loader, train_dataset = (
        create_train_loader(
            data_root=args.data_root,
            batch_size=args.batch_size,
            # Horizontal flip is safe for dual-stream after fix:
            # only intrinsics cx is adjusted (cx -> W-cx); extrinsics are unchanged.
            # Vertical flip treats nadir imagery as orientation-free; img,
            # bbox and mask are flipped in sync inside the dataset.
            flip_prob=train_flip_prob,
            vflip_prob=train_vflip_prob,
            gaussian_noise_prob=0.0,
            gaussian_noise_std=0.02,
            random_erasing_prob=0.0,
            random_erasing_scale=(0.02, 0.15),
            random_erasing_ratio=(0.3, 3.3),
            # Multi-scale jitter resizes BEFORE the fixed resize back to
            # image_size: the network (and frozen SAM2) always receives its
            # native 1024x1024 input; only object scale varies. Kept off by
            # default — random_erasing stays disabled because it erases image
            # pixels without updating GT masks (label corruption for
            # instance segmentation).
            multi_scale_resize_prob=train_ms_resize_prob,
            multi_scale_mode=train_ms_mode,
            multi_scale_img_scale=train_ms_img_scale,
            image_size=tuple(args.image_size),
            distributed=distributed,
            rank=rank,
            world_size=int(dist_info.get("world_size", 1)),
            val_batch_size=args.val_batch_size,
            dataset_format=("vhr10_coco" if args.use_vhr10_coco else "whu_coco"),
            vhr10_train_ann_file=os.environ.get(
                "VHR10_TRAIN_ANN_FILE", "coco_split/instances_train.json"
            ),
            vhr10_val_ann_file=os.environ.get(
                "VHR10_VAL_ANN_FILE", "coco_split/instances_val.json"
            ),
            vhr10_train_img_subdir=os.environ.get(
                "VHR10_TRAIN_IMG_SUBDIR", "positive image set"
            ),
            vhr10_val_img_subdir=os.environ.get(
                "VHR10_VAL_IMG_SUBDIR", "positive image set"
            ),
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
    # next training all-reduce.  Iterating the unsharded loader on every rank
    # duplicates full-resolution mask accumulation and can exhaust host
    # memory.  VAL_SHARD_ACROSS_RANKS=1 opts into sharded validation instead:
    # every rank owns loader batches rank, rank+W, ... (identical batch
    # composition because the val pipeline is deterministic), predictions are
    # gathered to rank 0 over the gloo control group and reassembled in
    # global batch order, so COCO inputs match the rank0-only path exactly.
    val_shard_world = int(dist_info.get("world_size", 1)) if distributed else 1
    val_shard_enabled = (
        os.environ.get("VAL_SHARD_ACROSS_RANKS", "0") == "1"
        and distributed
        and val_shard_world > 1
    )
    val_shard_total = -1
    if val_shard_enabled:
        if args.compute_val_loss:
            raise SystemExit(
                "VAL_SHARD_ACROSS_RANKS=1 requires --compute-val-loss 0: a "
                "shard-mean validation loss would silently change the metric."
            )
        if args.max_val_batches > 0:
            # The gather count check would otherwise fire only on rank 0 and
            # leave the other ranks blocked on the gloo control group until
            # its 86400s timeout.
            raise SystemExit(
                "VAL_SHARD_ACROSS_RANKS=1 is incompatible with "
                "--max-val-batches (truncated shards cannot be reassembled)."
            )
        val_loader, val_shard_total = _shard_val_loader(
            val_loader, rank, val_shard_world
        )
    elif distributed and not is_main:
        val_loader = None
    if is_main:
        logger.info(
            "Validation policy: rank0_only=%d sharded=%d batch_size=%d every_n_epochs=%d compute_loss=%d",
            int(distributed and not val_shard_enabled),
            int(val_shard_enabled),
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
        mask_decoder_lr_mult=args.mask_decoder_lr_mult,
        no_mask_lr_mult=args.no_mask_lr_mult,
        prompt_encoder_lr_mult=args.prompt_encoder_lr_mult,
        shape_prior_lr_mult=args.shape_prior_lr_mult,
        decoder_tail_lr_mult=args.decoder_tail_lr_mult,
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
        "train_augmentation": {
            "flip_prob": train_flip_prob,
            "vflip_prob": train_vflip_prob,
            "multi_scale_resize_prob": train_ms_resize_prob,
            "multi_scale_mode": train_ms_mode,
            "multi_scale_img_scale": train_ms_img_scale,
        },
        # HEAD alone cannot identify the executed code when the working tree
        # is dirty; carry the dirty flag + diff fingerprint so a checkpoint
        # uniquely pins its code state (same fields as the runner's
        # resolved_hyperparameters git_* lines).
        "git_state": _capture_git_state(),
        "data_config": {
            "dataset_format": ("vhr10_coco" if args.use_vhr10_coco else "whu_coco"),
            "data_root": str(args.data_root),
            "image_size": list(args.image_size),
            "single_class": not args.use_vhr10_coco,
            "num_classes": 10 if args.use_vhr10_coco else 1,
            "validation": (
                {
                    "ann_file": os.environ.get(
                        "VHR10_VAL_ANN_FILE", "coco_split/instances_val.json"
                    ),
                    "image_subdir": os.environ.get(
                        "VHR10_VAL_IMG_SUBDIR", "positive image set"
                    ),
                }
                if args.use_vhr10_coco
                else {
                    "ann_file": "2.4 annotation/annotation/validation.json",
                    "image_subdir": "2.3 valid/validation",
                }
            ),
            "test": (
                {
                    "ann_file": os.environ.get(
                        "VHR10_VAL_ANN_FILE", "coco_split/instances_val.json"
                    ),
                    "image_subdir": os.environ.get(
                        "VHR10_VAL_IMG_SUBDIR", "positive image set"
                    ),
                }
                if args.use_vhr10_coco
                else {
                    "ann_file": "2.4 annotation/annotation/test.json",
                    "image_subdir": "2.2 test/test",
                }
            ),
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
        # Display-only mirror of the prompt form (the whole env-mirror family
        # is gone: nothing re-reads these keys at checkpoint load time).
        "explicit_prompt_mode": os.environ.get("EXPLICIT_PROMPT_MODE", ""),
        "neck_type": os.environ.get("NECK_TYPE", ""),
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
        "shape_prior_loss_weight": float(
            resolved_mask_head_cfg.get("shape_prior_loss_weight", 0.0)
        ),
        "p2_boundary_refiner": dict(
            resolved_mask_head_cfg.get("p2_boundary_refiner_cfg", {})
        ),
        "init_from": args.init_from or "",
        "init_exclude_prefixes": args.init_exclude_prefixes,
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
    ddp_sync_audited = False
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
        allowed_missing_prefixes = ()
        init_stats = _load_init_checkpoint(
            Path(args.init_from),
            model,
            exclude_prefixes=exclude_prefixes,
            expected_architecture=resolved_architecture,
            allow_cross_arch=args.allow_cross_arch_init,
            allowed_missing_prefixes=allowed_missing_prefixes,
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

    nonfinite_batches = 0
    # Validation order is deterministic within a run. Cache compact GT RLE
    # records after the first validation instead of rebuilding them every epoch.
    cached_val_gt_records: Optional[List[dict]] = None
    decoupled_clip_partition = None
    if args.decoupled_aux_clip:
        decoupled_clip_partition = _partition_aux_clip_params(model)
        if is_main:
            logger.info(
                "Clip decomposition ON: base %d tensors at max_norm 0.5 + aux groups %s",
                len(decoupled_clip_partition[0]),
                [len(group) for group in decoupled_clip_partition[1]],
            )

    for epoch in range(start_epoch, args.epochs):
        epoch_number = epoch + 1
        _set_model_current_epoch(model, epoch_number)
        det_loss_weight = _detection_loss_weight(args, epoch_number)
        if distributed:
            sampler = getattr(train_loader, "sampler", None)
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)

        model.train()
        if args.freeze_bn:
            _set_norm_eval(model)

        total_loss = 0.0
        loss_meter = {}
        dense_monitor_sums = {key: 0.0 for key in _DENSE_RESIDUAL_MONITOR_KEYS}
        dense_monitor_counts = {key: 0 for key in _DENSE_RESIDUAL_MONITOR_KEYS}
        num_batches = 0
        grad_accum = args.grad_accum_steps
        train_batches_limit = len(train_loader)
        if args.max_train_batches > 0:
            train_batches_limit = min(train_batches_limit, args.max_train_batches)
        # SEGMENT_TIMING=1: per-micro-batch phase timers (prep/fwd/item/bwd),
        # averaged and logged every 50 steps.  Zero overhead when disabled.
        _seg_timing = os.environ.get("SEGMENT_TIMING", "0") == "1"
        _seg_sum = {"prep": 0.0, "fwd": 0.0, "item": 0.0, "bwd": 0.0, "total": 0.0}
        _seg_n = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(train_loader):
            if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
                break

            _seg_t0 = time.perf_counter() if _seg_timing else 0.0
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

            _seg_t1 = time.perf_counter() if _seg_timing else 0.0
            model_inputs = imgs

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                # Invoke the DDP wrapper itself. Calling ``model.module.loss``
                # bypasses DDP's reducer and makes each rank update independently.
                loss_dict = model(model_inputs, data_samples, mode="loss")
                loss_dict = _apply_detection_loss_weight(
                    loss_dict, det_loss_weight, device
                )
                # Key-name filter matters: telemetry tensors (TAIL/*, SP/*)
                # live in the same dict and must never join the loss sum.
                loss_parts = []
                for k, v in loss_dict.items():
                    if "loss" not in k:
                        continue
                    if isinstance(v, torch.Tensor):
                        loss_parts.append(v)
                    elif isinstance(v, (list, tuple)):
                        loss_parts.extend(
                            x for x in v if isinstance(x, torch.Tensor)
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

            _seg_t2 = time.perf_counter() if _seg_timing else 0.0
            if _seg_timing:
                _seg_sum["prep"] += _seg_t1 - _seg_t0
                _seg_sum["fwd"] += _seg_t2 - _seg_t1
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
            prompt_stats_due = (
                args.prompt_debug_stats
                and args.prompt_debug_stats_interval > 0
                and global_step % args.prompt_debug_stats_interval == 0
            )
            if is_main and prompt_stats_due:
                _log_uav_debug_stats(model, epoch, global_step, _PROMPT_TELEMETRY_PREFIXES)
            _seg_t3 = time.perf_counter() if _seg_timing else 0.0
            if _seg_timing:
                _seg_sum["item"] += _seg_t3 - _seg_t2
            scaler.scale(loss_for_backward).backward()
            if _seg_timing:
                _seg_t3b = time.perf_counter()
                _seg_sum["bwd"] += _seg_t3b - _seg_t3
                _seg_sum["total"] += _seg_t3b - _seg_t0
                _seg_n += 1
                if is_main and _seg_n % 100 == 0:
                    logger.info(
                        "TRAIN/TIME (avg %d micro-steps) prep=%.3fs fwd=%.3fs "
                        "item=%.3fs bwd=%.3fs total=%.3fs",
                        _seg_n,
                        _seg_sum["prep"] / _seg_n,
                        _seg_sum["fwd"] / _seg_n,
                        _seg_sum["item"] / _seg_n,
                        _seg_sum["bwd"] / _seg_n,
                        _seg_sum["total"] / _seg_n,
                    )
                    _seg_sum = {k: 0.0 for k in _seg_sum}
                    _seg_n = 0

            # Step optimizer every grad_accum batches or at the last batch
            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == train_batches_limit:
                scaler.unscale_(optimizer)
                if decoupled_clip_partition is not None:
                    base_clip_params, aux_clip_groups = decoupled_clip_partition
                    torch.nn.utils.clip_grad_norm_(base_clip_params, 0.5)
                    for aux_clip_group in aux_clip_groups:
                        torch.nn.utils.clip_grad_norm_(aux_clip_group, 0.5)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
                ema_update_steps += 1
                if (
                    not ddp_sync_audited
                    and any(float(group["lr"]) > 0.0 for group in optimizer.param_groups)
                ):
                    ddp_max_parameter_deviation = _verify_ddp_parameter_sync_after_first_update(
                        model, device, distributed
                    )
                    if ddp_max_parameter_deviation is not None:
                        ddp_sync_audited = True
                    if is_main and ddp_max_parameter_deviation is not None:
                        logger.info(
                            "DDP parameter-sync audit after first nonzero-LR update: "
                            "max_abs_deviation=%.3e",
                            ddp_max_parameter_deviation,
                        )
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
                    "Epoch %d primary loss shares: det=%.2f%% final_mask=%.2f%% "
                    "shape=%.2f%%",
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
            _val_wall_t0 = time.monotonic()
            if use_ema_this_epoch:
                ema.apply_to(model)
                ema_applied_for_eval = True
            model.eval()
            if args.freeze_bn:
                _set_norm_eval(model)

            # Apply the TEST_MAX_PER_IMG model-side cap BEFORE the inference
            # loop and on EVERY rank (sharded validation runs predict() on all
            # ranks, and the cap doubles as the decoder memory-spike guard).
            # Previously this mutation ran inside the rank0-only eval section
            # after inference, so it only took effect from the second val.
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
            # (global_batch_idx, images_in_batch) for the sharded gather; the
            # global index is rank + local_idx * world_size because the shard
            # loader yields batches rank, rank+W, ... in order.
            shard_batch_spans: List[Tuple[int, int]] = []

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

                    if val_shard_enabled:
                        shard_batch_spans.append(
                            (
                                rank + batch_idx * val_shard_world,
                                len(outputs),
                            )
                        )

            if val_shard_enabled:
                _payload = {
                    "spans": shard_batch_spans,
                    "dt": all_dt,
                    "metas": all_img_metas,
                    "gt": all_gt if collect_gt_records else None,
                }
                _gathered = [None] * val_shard_world if is_main else None
                dist.gather_object(_payload, _gathered, dst=0, group=control_group)
                if is_main:
                    _merged: Dict[int, Dict[str, Any]] = {}
                    for _pl in _gathered:
                        if _pl is None:
                            continue
                        _pos = 0
                        for _gb, _cnt in _pl["spans"]:
                            _merged[_gb] = {
                                "dt": _pl["dt"][_pos : _pos + _cnt],
                                "metas": _pl["metas"][_pos : _pos + _cnt],
                                "gt": (
                                    _pl["gt"][_pos : _pos + _cnt]
                                    if _pl["gt"] is not None
                                    else None
                                ),
                            }
                            _pos += _cnt
                    _ordered = sorted(_merged)
                    _total_imgs = sum(len(_merged[g]["dt"]) for g in _ordered)
                    if _total_imgs != val_shard_total:
                        raise RuntimeError(
                            "Sharded validation gather assembled "
                            f"{_total_imgs} images, expected {val_shard_total}"
                        )
                    all_dt = [rec for g in _ordered for rec in _merged[g]["dt"]]
                    all_img_metas = [
                        m for g in _ordered for m in _merged[g]["metas"]
                    ]
                    if collect_gt_records:
                        all_gt = [rec for g in _ordered for rec in _merged[g]["gt"]]
                    logger.info(
                        "Validation shard gather: ranks=%d batches=%d images=%d "
                        "wall=%.1fs",
                        val_shard_world,
                        len(_ordered),
                        _total_imgs,
                        time.monotonic() - _val_wall_t0,
                    )
                else:
                    # Non-main ranks must not feed shard-local records into
                    # the COCO section below; empty lists skip it naturally
                    # (total_gt == 0), matching the unsharded contract.
                    all_gt, all_dt, all_img_metas = [], [], []

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

            if is_main and collect_gt_records:
                cached_val_gt_records = all_gt
                logger.info(
                    "Cached validation GT as RLE records: images=%d",
                    len(cached_val_gt_records),
                )
            elif is_main and len(all_gt) != len(all_dt):
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
                # (The model-side cap itself is now applied at validation
                # entry on every rank — see the _val_max_dets block above the
                # inference loop; only the eval maxDets list is consumed here.)
                # The segmentation score contract is stored in cfg.model and
                # consumed identically by validation and checkpoint inference.
                # Bbox candidate selection/ranking always remains detector-score based.
                eval_categories = VHR10_CATEGORIES if args.use_vhr10_coco else None
                coco_gt, coco_bbox_dt = build_coco_gt_and_dt(
                    all_gt, all_dt, all_img_metas, score_key="scores",
                    categories=eval_categories,
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
                        categories=eval_categories,
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

        # Restore training weights after EMA-based eval/save.  This must run
        # on EVERY rank: with VAL_SHARD_ACROSS_RANKS=1 all ranks apply EMA
        # inside the validation block, and a rank that keeps the shadow
        # weights would silently desync DDP training.  It stays after rank
        # 0's checkpoint saves and epoch logging (unchanged rank-0 ordering)
        # and before the stop broadcast below.
        if ema_applied_for_eval:
            ema.restore(model)
            ema_applied_for_eval = False

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
