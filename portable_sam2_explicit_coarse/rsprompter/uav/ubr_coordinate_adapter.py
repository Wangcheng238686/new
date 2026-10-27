"""Explicit native-RoI -> image adapter for the historical FCN paste contract.

The decoder's 32-channel upscaled embedding and logits share a native grid
(the mask hypernetwork is pointwise).  Although their inputs repeat global
features, the deployed historical head supervises/interprets that grid as
RoI-local.  Both must use the SAME bbox sampling grid here.  This does not
change the old training targets or establish a new full-image decoder target.

Probability is sigmoid-ed BEFORE sampling, matching FCNMaskHead. Inverse
logit is finite with explicit epsilon; it is not a bitwise inverse at 0/1.
Keep original pasted probabilities for exact identity deployment and scores.
This module is an adapter only, not a model-head integration.
"""
from dataclasses import dataclass
from typing import Mapping, Tuple

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F


def fcn_canvas(boxes: Tensor, img_meta: Mapping, rescale: bool = False):
    """FCN canvas and xyxy boxes, without FCN's in-place bbox mutation.

Input boxes are always in resized model-image pixels. ``rescale=True``
maps them to original-image pixels; else canvas follows ori_shape*scale,
including numpy's round-to-even, NOT padded input_shape.
"""
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError('boxes must be [N,4] xyxy in model-image pixels')
    factor = boxes.new_tensor(img_meta['scale_factor']).reshape(-1)
    if factor.numel() != 2 or not torch.isfinite(factor).all() or (factor <= 0).any():
        raise ValueError('scale_factor must contain positive finite (w,h)')
    h, w = img_meta['ori_shape'][:2]
    if rescale:
        return boxes / factor.repeat(2), (int(h), int(w))
    return boxes.clone(), (int(np.round(h * factor[1].item())),
                           int(np.round(w * factor[0].item())))


def paste_native_roi(values: Tensor, boxes: Tensor, image_size: Tuple[int, int],
                     cpu_bbox_clip: bool = False) -> Tensor:
    """Differentiable FCN grid_sample paste for N,C,H,W native RoI values.

Pixel centers are (x+.5,y+.5); bbox xyxy are image-edge coordinates;
align_corners=False and zero padding match mmdet _do_paste_mask.
Output is FP32, including AMP inputs. No clipping/rounding boxes before paste.
    ``cpu_bbox_clip`` matches FCN CPU's per-instance skip_empty bbox +/-1
    pixel extent. It differs from full GPU paste for very low native resolution
    and large boxes (bilinear zero-padding has a half-source-pixel halo).
    Degenerate/nonfinite boxes are rejected explicitly instead of manufacturing
plausible samples from invalid proposals. Out-of-image positive boxes work.
"""
    if values.ndim != 4 or boxes.shape != (values.shape[0], 4):
        raise ValueError('expected native [N,C,H,W] and boxes [N,4]')
    h, w = map(int, image_size)
    if h <= 0 or w <= 0:
        raise ValueError('image_size must be positive (h,w)')
    if values.device != boxes.device:
        raise ValueError('values and boxes must share device')
    if not torch.isfinite(boxes).all() or (boxes[:, 2:] <= boxes[:, :2]).any():
        raise ValueError('boxes must be finite with positive width and height')
    if values.shape[0] == 0:
        return values.float().sum() * 0 + values.new_empty((0, values.shape[1], h, w), dtype=torch.float32)
    x0, y0, x1, y1 = boxes.float().split(1, dim=1)
    yy = (torch.arange(h, device=values.device, dtype=torch.float32) + .5 - y0) / (y1 - y0) * 2 - 1
    xx = (torch.arange(w, device=values.device, dtype=torch.float32) + .5 - x0) / (x1 - x0) * 2 - 1
    grid = torch.stack((xx[:, None].expand(-1, h, -1), yy[:, :, None].expand(-1, -1, w)), dim=-1)
    # Explicit disable: a caller's CUDA autocast must not downcast this bridge.
    with torch.autocast(device_type=values.device.type, enabled=False):
        result = F.grid_sample(values.float(), grid, align_corners=False,
                               mode='bilinear', padding_mode='zeros')
    if cpu_bbox_clip:
        low = (boxes[:, :2].floor() - 1).clamp_min(0)
        high = boxes[:, 2:].ceil() + 1
        ix = torch.arange(w, device=values.device)[None, None, :]
        iy = torch.arange(h, device=values.device)[None, :, None]
        valid = ((ix >= low[:, 0, None, None]) & (ix < high[:, 0, None, None]) &
                 (iy >= low[:, 1, None, None]) & (iy < high[:, 1, None, None]))
        result = result * valid[:, None]
    return result


def finite_probability_logits(probabilities: Tensor, epsilon: float = 1e-6) -> Tensor:
    """Finite FP32 logits; round-trip error includes FP32 rounding.

    At the locked epsilon=1e-6, the gate tolerance is 1.1e-6.
    Reject tiny epsilons whose FP32 upper clamp would round back to one.
    """
    if not torch.finfo(torch.float32).eps <= epsilon < .5:
        raise ValueError('epsilon must lie in [float32.eps,.5)')
    p = probabilities.float()
    if not torch.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('probabilities must be finite in [0,1]')
    p = p.clamp(epsilon, 1 - epsilon)
    return p.log() - torch.log1p(-p)


@dataclass
class RoIImageState:
    probabilities: Tensor
    logits: Tensor
    decoder_features: Tensor
    boxes: Tensor
    image_size: Tuple[int, int]
    epsilon: float


def adapt_native_roi(logits: Tensor, decoder_features: Tensor, boxes: Tensor,
                     img_meta: Mapping, rescale: bool = False,
                     epsilon: float = 1e-6) -> RoIImageState:
    """Class-agnostic single-image RoI outputs -> common image pixel canvas.

Caller selects one mask channel per instance BEFORE this adapter and passes
the corresponding 32-channel decoder embedding in identical instance order.
No thresholding, scores, GT, coordinate-mode guessing or geometry rescaling.
For training, use rescale=False and align geometry to the returned canvas.
"""
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError('logits must be selected [N,1,H,W]')
    if decoder_features.shape != (logits.shape[0], 32, *logits.shape[-2:]):
        raise ValueError('decoder features must share native grid/order: [N,32,H,W]')
    canvas_boxes, image_size = fcn_canvas(boxes, img_meta, rescale)
    # Keep sigmoid in original incoming dtype, matching old FCN activation.
    cpu_clip = logits.device.type == 'cpu'
    probability = paste_native_roi(logits.sigmoid(), canvas_boxes, image_size, cpu_clip)
    features = paste_native_roi(decoder_features, canvas_boxes, image_size, cpu_clip)
    return RoIImageState(probability, finite_probability_logits(probability, epsilon),
                         features, canvas_boxes, image_size, epsilon)


def residual_probabilities(state: RoIImageState, refined_logits: Tensor) -> Tensor:
    """Deploy UBR changes but preserve old probabilities at untouched pixels.

UBR guarantees z'=z outside its detached band and at zero init. Selecting
the saved probability for exactly unchanged logits avoids epsilon changing
    binary decisions, visualization uint8, or downstream mask scores.
    Inference only: do NOT train through this identity selection; supervise
    refined_logits directly. This does not crop/refit the UBR output to RoI.
"""
    if refined_logits.shape != state.logits.shape:
        raise ValueError('refined logits must retain the image canvas shape')
    p = refined_logits.float().sigmoid()
    identity = refined_logits == state.logits
    return torch.where(identity, state.probabilities, p)
