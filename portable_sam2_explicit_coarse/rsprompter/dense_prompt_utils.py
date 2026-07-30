"""Dense prompt transform and excavation utilities.

独立于 mask_head，避免测试时实例化整个 SAM2 模型。
处理顺序（审查 3.1）：ROI-local transform → resize → paste → outside_fill → clamp。
"""
from typing import Dict, Tuple

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
    if detach_input:
        coarse_logits = coarse_logits.detach()
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


def gaussian_prompt_from_roi_coarse(
    coarse_logits: Tensor,
    boxes: Tensor,
    *,
    mask_h: int,
    mask_w: int,
    image_size,
    foreground_threshold: float = 0.5,
    omega: float = 15.0,
    gamma: float = 4.0,
) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
    """Excavate a detached SAMRefiner-style Gaussian on a full-image canvas.

    The hard foreground, exact Euclidean distance transform (EDT) centre and
    foreground area are measured on the ROI-local coarse grid.  Centre and
    area are then mapped analytically through the proposal box to the native
    PromptEncoder mask canvas, where an isotropic image-space Gaussian is
    generated.  This avoids both stretching an ROI-local Gaussian and
    quantising small objects before centre excavation.

    The operation is intentionally non-differentiable: callers use the
    returned prompt as an excavated observation, while the coarse head remains
    supervised by its own BCE+Dice objective.
    """
    if coarse_logits.dim() != 4 or coarse_logits.shape[1] != 1:
        raise ValueError(
            "coarse_logits must be [N,1,H,W], got "
            f"{tuple(coarse_logits.shape)}"
        )
    if boxes.dim() != 2 or boxes.shape != (coarse_logits.shape[0], 4):
        raise ValueError(
            f"boxes must be [N,4], got {tuple(boxes.shape)} for "
            f"N={coarse_logits.shape[0]}"
        )
    if mask_h <= 0 or mask_w <= 0:
        raise ValueError(f"mask size must be positive, got {(mask_h, mask_w)}")
    if not 0.0 < float(foreground_threshold) < 1.0:
        raise ValueError("foreground_threshold must be in (0,1)")
    if float(omega) <= 0.0 or float(gamma) <= 0.0:
        raise ValueError("omega and gamma must be positive")

    if isinstance(image_size, (tuple, list)):
        image_h, image_w = float(image_size[0]), float(image_size[1])
    else:
        image_h = image_w = float(image_size)
    if image_h <= 0.0 or image_w <= 0.0:
        raise ValueError(f"image_size must be positive, got {image_size}")

    # scipy's exact EDT is deliberately used on the detached 64x64 masks.  A
    # one-pixel zero pad makes an all-foreground ROI measure distance to the
    # proposal boundary instead of relying on scipy's no-background corner
    # convention.
    import numpy as np
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as error:
        raise RuntimeError(
            "gaussian_edt dense prompts require scipy.ndimage; install scipy "
            "in the configured training environment"
        ) from error

    detached = coarse_logits.detach()
    foreground_cpu = (
        torch.sigmoid(detached.float()) >= float(foreground_threshold)
    ).squeeze(1).to(device="cpu", dtype=torch.bool).numpy()
    n, roi_h, roi_w = foreground_cpu.shape
    centres_y = np.zeros(n, dtype=np.float32)
    centres_x = np.zeros(n, dtype=np.float32)
    areas = np.zeros(n, dtype=np.float32)
    edt_maxima = np.zeros(n, dtype=np.float32)
    valid_cpu = np.zeros(n, dtype=np.bool_)
    for index, foreground in enumerate(foreground_cpu):
        area = int(foreground.sum())
        if area == 0:
            continue
        padded = np.pad(foreground, 1, mode="constant", constant_values=False)
        distance = distance_transform_edt(padded)[1:-1, 1:-1]
        flat_index = int(distance.argmax())
        centre_y, centre_x = divmod(flat_index, roi_w)
        centres_y[index] = float(centre_y) + 0.5
        centres_x[index] = float(centre_x) + 0.5
        areas[index] = float(area)
        edt_maxima[index] = float(distance[centre_y, centre_x])
        valid_cpu[index] = True

    device = detached.device
    work_dtype = torch.float32
    valid = torch.from_numpy(valid_cpu).to(device=device)
    centre_y = torch.from_numpy(centres_y).to(device=device, dtype=work_dtype)
    centre_x = torch.from_numpy(centres_x).to(device=device, dtype=work_dtype)
    roi_area = torch.from_numpy(areas).to(device=device, dtype=work_dtype)
    edt_max = torch.from_numpy(edt_maxima).to(device=device, dtype=work_dtype)
    boxes_work = boxes.detach().to(device=device, dtype=work_dtype)
    box_w = (boxes_work[:, 2] - boxes_work[:, 0]).clamp_min(1.0)
    box_h = (boxes_work[:, 3] - boxes_work[:, 1]).clamp_min(1.0)

    canvas_centre_x = (
        boxes_work[:, 0] + centre_x * box_w / float(roi_w)
    ) * (float(mask_w) / image_w)
    canvas_centre_y = (
        boxes_work[:, 1] + centre_y * box_h / float(roi_h)
    ) * (float(mask_h) / image_h)
    canvas_area = (
        roi_area
        * (box_w / float(roi_w))
        * (box_h / float(roi_h))
        * (float(mask_w) / image_w)
        * (float(mask_h) / image_h)
    )

    yy = torch.arange(mask_h, device=device, dtype=work_dtype).view(1, 1, mask_h, 1)
    xx = torch.arange(mask_w, device=device, dtype=work_dtype).view(1, 1, 1, mask_w)
    distance_sq = (
        (xx - canvas_centre_x[:, None, None, None]).square()
        + (yy - canvas_centre_y[:, None, None, None]).square()
    )
    denominator = (float(gamma) * canvas_area).clamp_min(1e-6)
    prompt = float(omega) * torch.exp(
        -distance_sq / denominator[:, None, None, None]
    )
    prompt = prompt * valid[:, None, None, None].to(prompt.dtype)
    prompt = prompt.to(dtype=coarse_logits.dtype)

    valid_float = valid.to(work_dtype)
    valid_count = valid_float.sum()
    safe_valid_count = valid_count.clamp_min(1.0)
    stats = {
        "roi_count": valid_float.new_tensor(float(n)),
        "valid_count": valid_count,
        "valid_ratio": valid_float.mean() if n else valid_float.new_zeros(()),
        "foreground_area_ratio": (
            (roi_area / float(roi_h * roi_w) * valid_float).sum()
            / safe_valid_count
        ),
        "edt_max": (edt_max * valid_float).sum() / safe_valid_count,
        "center_x_normalized": (
            (centre_x / float(roi_w) * valid_float).sum() / safe_valid_count
        ),
        "center_y_normalized": (
            (centre_y / float(roi_h) * valid_float).sum() / safe_valid_count
        ),
        "canvas_area": (canvas_area * valid_float).sum() / safe_valid_count,
    }
    return prompt, valid, stats


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
