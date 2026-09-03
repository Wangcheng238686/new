"""Training-time box-prompt jitter (plan D: prompt-consumption robustification).

Perturbs each proposal box by up to ``jitter_scale`` of its side lengths,
clamped to image bounds and to ``min_box_size``, and falls back to the
original box whenever the perturbed IoU drops below
``min_iou_with_original``. The SAME jittered box is returned for RoIAlign,
prompt construction and target cropping (the caller passes it through), so
features, prompts and supervision stay aligned under the perturbation.

Inference must never call this (training-only prompt augmentation).
"""
from typing import Dict, List, Tuple

import torch


def _xyxy_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU of aligned [N,4] xyxy boxes."""
    ix1 = torch.maximum(a[:, 0], b[:, 0])
    iy1 = torch.maximum(a[:, 1], b[:, 1])
    ix2 = torch.minimum(a[:, 2], b[:, 2])
    iy2 = torch.minimum(a[:, 3], b[:, 3])
    inter = (ix2 - ix1).clamp_min(0) * (iy2 - iy1).clamp_min(0)
    area_a = (a[:, 2] - a[:, 0]).clamp_min(0) * (a[:, 3] - a[:, 1]).clamp_min(0)
    area_b = (b[:, 2] - b[:, 0]).clamp_min(0) * (b[:, 3] - b[:, 1]).clamp_min(0)
    union = area_a + area_b - inter
    return inter / union.clamp_min(1e-6)


def jitter_rois(
    rois: torch.Tensor,
    batch_img_metas: List[Dict],
    jitter_prob: float,
    jitter_scale: float,
    min_box_size: float,
    min_iou_with_original: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Jitter proposal RoIs [N,5] (batch_idx, x1, y1, x2, y2) in place-safe.

    Returns the (possibly) jittered RoIs and a stats dict compatible with the
    disabled-branch keys consumed by the trainer's debug logger
    (JITTER/num_rois, num_jittered, mean_iou, fallback_count,
    area_ratio_sum, area_ratio_count).
    """
    out = rois.clone()
    n = rois.shape[0]
    device = rois.device
    zero = rois.new_zeros(())

    if n == 0 or jitter_prob <= 0.0:
        stats = {
            "JITTER/num_rois": rois.new_tensor(float(n)),
            "JITTER/num_jittered": zero,
            "JITTER/mean_iou": rois.new_tensor(1.0),
            "JITTER/fallback_count": zero,
            "JITTER/area_ratio_sum": zero,
            "JITTER/area_ratio_count": zero,
        }
        return out, stats

    # Per-image bounds (max valid x/y) from metas, indexed by batch_ind.
    bounds = []
    for meta in batch_img_metas:
        shape = meta.get("img_shape", meta.get("ori_shape", None))
        if shape is None:
            raise ValueError("box jitter requires img_shape in img_metas")
        bounds.append((float(shape[1]), float(shape[0])))  # (w, h)
    max_w = rois.new_tensor([b[0] for b in bounds])[rois[:, 0].long()]
    max_h = rois.new_tensor([b[1] for b in bounds])[rois[:, 0].long()]

    boxes = rois[:, 1:5]
    w = (boxes[:, 2] - boxes[:, 0]).clamp_min(1e-3)
    h = (boxes[:, 3] - boxes[:, 1]).clamp_min(1e-3)

    # Each side moves by up to jitter_scale of its own extent, independently.
    offs = (torch.rand_like(boxes) * 2.0 - 1.0) * torch.stack(
        [w, h, w, h], dim=1
    ) * float(jitter_scale)
    cand = boxes + offs
    min_size_t = max_w.new_full((), float(min_box_size))
    cand[:, 0] = cand[:, 0].clamp(torch.zeros_like(max_w), max_w - min_size_t)
    cand[:, 1] = cand[:, 1].clamp(torch.zeros_like(max_h), max_h - min_size_t)
    cand[:, 2] = cand[:, 2].clamp(torch.full_like(max_w, min_size_t), max_w)
    cand[:, 3] = cand[:, 3].clamp(torch.full_like(max_h, min_size_t), max_h)
    # Degenerate flips after clamping: fall back to the original side.
    flip = (cand[:, 2] <= cand[:, 0] + min_box_size) | (
        cand[:, 3] <= cand[:, 1] + min_box_size
    )
    cand[flip] = boxes[flip]

    iou = _xyxy_iou(cand, boxes)
    sel = (torch.rand(n, device=device) < float(jitter_prob)) & (~flip)
    ok = sel & (iou >= float(min_iou_with_original))
    fallback_count = int((sel & ~ok).sum())

    out[:, 1:5] = torch.where(ok[:, None], cand, boxes)

    cand_area = (
        (cand[:, 2] - cand[:, 0]).clamp_min(0) * (cand[:, 3] - cand[:, 1]).clamp_min(0)
    )
    orig_area = w * h
    area_ratio = cand_area / orig_area.clamp_min(1e-6)

    stats = {
        "JITTER/num_rois": rois.new_tensor(float(n)),
        "JITTER/num_jittered": rois.new_tensor(float(int(ok.sum()))),
        "JITTER/mean_iou": iou[ok].mean() if bool(ok.any()) else rois.new_tensor(1.0),
        "JITTER/fallback_count": rois.new_tensor(float(fallback_count)),
        "JITTER/area_ratio_sum": area_ratio[ok].sum() if bool(ok.any()) else zero,
        "JITTER/area_ratio_count": rois.new_tensor(float(int(ok.sum()))),
    }
    return out, stats
