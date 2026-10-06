"""Dense prompt transform and paste utilities.

Standalone from the mask head so tests need not instantiate SAM2. Order:
ROI-local transform -> resize to the box's prompt-grid cells -> paste onto the
full canvas -> outside fill (-> clamp if configured).
"""
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def transform_coarse_prompt(
    coarse_logits: Tensor,
    transform: str,
    gamma: float,
    strength: float,
    clamp_range,
    temperature: float = 1.0,
    detach_input: bool = False,
) -> Tensor:
    """ROI-local coarse logits -> prompt logits (same shape).

    raw_logits: identity (optionally clamped). gamma/strength belong to the
    removed confidence_signed transform and are accepted but unused.
    """
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if detach_input:
        coarse_logits = coarse_logits.detach()
    calibrated_logits = coarse_logits / temperature
    if transform != "raw_logits":
        raise ValueError(
            f"Unknown dense_prompt transform: {transform!r}. "
            "Only 'raw_logits' survives the a3sgm-only cleanup."
        )
    prompt = calibrated_logits
    if clamp_range is not None:
        prompt = prompt.clamp(float(clamp_range[0]), float(clamp_range[1]))
    return prompt


def replace_invalid_dense_with_base(
    dense_embeddings: Tensor,
    base_embeddings: Tensor,
    valid: Tensor,
) -> Tensor:
    """Restore invalid per-ROI mask prompts to the exact no-mask base."""
    if dense_embeddings.shape != base_embeddings.shape:
        raise ValueError(
            "dense/base embedding shapes differ: "
            f"{tuple(dense_embeddings.shape)} vs {tuple(base_embeddings.shape)}"
        )
    if valid.dim() != 1 or valid.shape[0] != dense_embeddings.shape[0]:
        raise ValueError(
            f"valid must be [N], got {tuple(valid.shape)} for "
            f"N={dense_embeddings.shape[0]}"
        )
    return torch.where(
        valid.to(device=dense_embeddings.device, dtype=torch.bool)[:, None, None, None],
        dense_embeddings,
        base_embeddings,
    )


def paste_roi_to_full_canvas(
    roi_prompt_logits: Tensor,
    boxes: Tensor,
    mask_h: int,
    mask_w: int,
    image_size,
    outside_fill_logit: float = 0.0,
) -> Tensor:
    """Paste transformed ROI-local prompt logits onto the PE mask canvas.

    Args:
        roi_prompt_logits: [N,1,h,w] transformed ROI-local prompt logits.
        boxes: [N,4] xyxy pixel coordinates.
        mask_h, mask_w: PE mask canvas size (read from PE.mask_input_size).
        image_size: prompt_encoder_image_size (box scaling).
        outside_fill_logit: fill outside the bbox (0 = historical value).

    Returns:
        [N,1,mask_h,mask_w] full-image prompt canvas.
    """
    n = roi_prompt_logits.shape[0]
    if roi_prompt_logits.dim() != 4 or roi_prompt_logits.shape[1] != 1:
        raise ValueError(
            f"roi_prompt_logits must be [N,1,h,w], got {tuple(roi_prompt_logits.shape)}"
        )
    full_masks = roi_prompt_logits.new_full(
        (n, 1, mask_h, mask_w), fill_value=float(outside_fill_logit),
    )
    if n == 0:
        return full_masks

    if isinstance(image_size, (tuple, list)):
        image_h, image_w = float(image_size[0]), float(image_size[1])
    else:
        image_h = image_w = float(image_size)
    scale_xy = boxes.new_tensor(
        [mask_w / image_w, mask_h / image_h, mask_w / image_w, mask_h / image_h],
        dtype=roi_prompt_logits.dtype,
    )
    scaled_boxes = boxes.to(dtype=roi_prompt_logits.dtype) * scale_xy
    for i in range(n):
        x1, y1, x2, y2 = scaled_boxes[i]
        x1i = int(torch.floor(x1).clamp(0, mask_w - 1).item())
        y1i = int(torch.floor(y1).clamp(0, mask_h - 1).item())
        x2i = int(torch.ceil(x2).clamp(x1i + 1, mask_w).item())
        y2i = int(torch.ceil(y2).clamp(y1i + 1, mask_h).item())
        roi_h = max(1, y2i - y1i)
        roi_w = max(1, x2i - x1i)
        resized = F.interpolate(
            roi_prompt_logits[i : i + 1],
            size=(roi_h, roi_w),
            mode="bilinear",
            align_corners=False,
        )
        full_masks[i : i + 1, :, y1i:y2i, x1i:x2i] = resized
    return full_masks
