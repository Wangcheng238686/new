"""Dense prompt transform 工具（批次2 子任务3）。

独立于 mask_head，避免测试时实例化整个 SAM2 模型。
处理顺序（审查 3.1）：ROI-local transform → resize → paste → outside_fill → clamp。
"""
from typing import Optional, Tuple

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
) -> Tensor:
    """ROI-local coarse logits → prompt logits（独立 transform）。

    Args:
        coarse_logits: [N,1,h,w] ROI-local coarse mask logits。
        transform: "raw_logits"（直接返回，clamp_range 非 None 时 clamp）或
                   "confidence_signed"（prob=sigmoid; signed=2*prob-1;
                   confidence=signed.abs()^gamma; prompt=strength*signed*confidence; clamp）。
        gamma: confidence_signed 的指数（<1 放大不确定区域，>1 抑制）。
        strength: confidence_signed 的强度系数。
        clamp_range: None=不 clamp（raw_logits 默认兼容）；(min,max)=clamp。

    Returns:
        [N,1,h,w] prompt logits（与输入同 shape）。
    """
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    calibrated_logits = coarse_logits / temperature
    if transform == "raw_logits":
        prompt = calibrated_logits
        if clamp_range is not None:
            prompt = prompt.clamp(float(clamp_range[0]), float(clamp_range[1]))
        return prompt
    elif transform == "confidence_signed":
        prob = torch.sigmoid(calibrated_logits)
        signed = 2.0 * prob - 1.0
        confidence = signed.abs().pow(float(gamma))
        prompt = float(strength) * signed * confidence
        if clamp_range is not None:
            prompt = prompt.clamp(float(clamp_range[0]), float(clamp_range[1]))
        return prompt
    else:
        raise ValueError(
            f"Unknown dense_prompt transform: {transform!r}. "
            "Expected 'raw_logits' or 'confidence_signed'."
        )


def paste_roi_to_full_canvas(
    roi_prompt_logits: Tensor,
    boxes: Tensor,
    mask_h: int,
    mask_w: int,
    image_size,
    outside_fill_logit: float = 0.0,
) -> Tensor:
    """把 ROI-local prompt logits 粘到 full-image PE mask canvas。

    处理顺序（审查 3.1）：已 transform 的 ROI logits → resize 到 bbox 对应 prompt-grid
    → paste 到 canvas（bbox 外用 outside_fill_logit）。

    Args:
        roi_prompt_logits: [N,1,h,w] 已 transform 的 ROI-local prompt logits。
        boxes: [N,4] xyxy 像素坐标。
        mask_h, mask_w: PE mask canvas 尺寸（从 PE.mask_input_size 读，不硬编码）。
        image_size: prompt_encoder_image_size（用于 box 缩放）。
        outside_fill_logit: bbox 外填充值（0=旧实现；-6=强背景）。

    Returns:
        [N,1,mask_h,mask_w] full-image prompt canvas。
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
