"""Metric-dict reading for best selection, forward-telemetry logging,
dense-residual monitors and the DDP parameter-sync audit."""

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

logger = logging.getLogger("portable_sam_fusion")


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
                              "TAIL", "CONTEXT", "BOX", "UAV"}


def _log_forward_telemetry(
    model: torch.nn.Module,
    epoch: int,
    step: int,
    enabled_prefixes: Optional[set] = None,
) -> None:
    model_core = model.module if hasattr(model, "module") else model
    getter = getattr(model_core, "get_telemetry", None)
    if getter is None:
        getter = getattr(model_core, "get_forward_telemetry", None)
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
    getter = getattr(model_core, "get_telemetry", None)
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
