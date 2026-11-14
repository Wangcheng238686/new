"""Native-grid evidence sampling and a single lightweight mask residual head.

Legacy decoder outputs are interpreted in proposal-local coordinates. For a
native cell (i,j), image-edge coordinates are bbox.xy0 + (j+.5,i+.5)*bbox.wh/HW.
This preserves that historical interpretation, not a new physical calibration.
Aligned RGB bins are floor(uv/4), centered at (4*j+2,4*i+2): see
s2db_geometry.py's cell indexing and relative-offset construction. Geometry
uses original image pixels centered at (j+.5,i+.5). No implicit ROI resize.

Main mask loss stays the existing 512-square RoI supervision downstream.
No SDT, neighboring masks, auxiliary loss, alpha, or second decoder call.
"""
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def roi_sample_grid(boxes: Tensor, output_size: Tuple[int, int], source_size: Tuple[int, int],
                    stride: Tuple[float, float], origin: Tuple[float, float]) -> Tensor:
    """Normalized grid; stride/origin ordered (x,y), sizes ordered (h,w).

    Source index zero is located at explicit image-edge ``origin``. With
    align_corners=False, source index q maps to normalized (q+.5)*2/size-1.
    Values outside source center extents retain normal zero-padding behavior.
    """
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError('boxes must be [N,4] image-edge xyxy')
    if not torch.isfinite(boxes).all() or (boxes[:, 2:] <= boxes[:, :2]).any():
        raise ValueError('finite positive-area boxes required')
    h, w = map(int, output_size)
    sh, sw = map(int, source_size)
    if min(h, w, sh, sw) < 1 or min(stride) <= 0:
        raise ValueError('positive source/output sizes and source stride required')
    parameters = boxes.new_tensor((*stride, *origin), dtype=torch.float32)
    if not torch.isfinite(parameters).all():
        raise ValueError('finite stride and origin required')
    b = boxes.float()
    x = b[:, 0, None] + (torch.arange(w, device=b.device).float()[None]+.5)/w*(b[:, 2]-b[:, 0])[:, None]
    y = b[:, 1, None] + (torch.arange(h, device=b.device).float()[None]+.5)/h*(b[:, 3]-b[:, 1])[:, None]
    x = ((x-origin[0])/stride[0]+.5)*2/sw-1
    y = ((y-origin[1])/stride[1]+.5)*2/sh-1
    return torch.stack((x[:, None].expand(-1, h, -1), y[:, :, None].expand(-1, -1, w)), -1)


def sample_roi_evidence(boxes: Tensor, roi_image_ids: Tensor, output_size: Tuple[int, int],
                        aligned_rgb: Optional[Tensor] = None, rgb_valid: Optional[Tensor] = None,
                        geometry: Optional[Tensor] = None, *,
                        rgb_stride=(4., 4.), rgb_origin=(2., 2.),
                        geometry_stride=(1., 1.), geometry_origin=(.5, .5),
                        image_size=(512, 512), weight_epsilon=1e-6):
    """Sample evidence on native RoI cell centers, returning FP32 tensors.

    RGB = bilinear(value*valid)/bilinear(valid), zero if weight==0. Independent
    binary validity MUST come from retained RGB candidate support, not geometry.
    Geometry uses nearest sampling; channel4 is binary support and gates all
    five channels. No geometry support does not disable valid RGB, or vice versa.
    Source tensors remain differentiable; validity and sampling positions are
    detached. Missing evidence produces zeros but never gates satellite output.
    """
    if roi_image_ids.shape != (len(boxes),) or roi_image_ids.dtype != torch.long:
        raise ValueError('one int64 image ID per RoI required')
    if roi_image_ids.device != boxes.device or (roi_image_ids < 0).any():
        raise ValueError('image IDs must be nonnegative and on boxes device')
    h, w = map(int, output_size)
    if weight_epsilon <= 0 or min(image_size) < 1:
        raise ValueError('positive image_size and weight_epsilon required')
    # Validate boxes and sizes even when no evidence is supplied.
    roi_sample_grid(boxes, output_size, (1, 1), (1., 1.), (.5, .5))
    image_grid = roi_sample_grid(boxes.detach(), output_size, image_size, (1., 1.), (.5, .5))
    in_image = ((image_grid >= -1) & (image_grid < 1)).all(-1)[:, None]
    out_rgb = boxes.new_zeros((len(boxes), 32, h, w), dtype=torch.float32)
    out_geo = boxes.new_zeros((len(boxes), 5, h, w), dtype=torch.float32)
    ids = roi_image_ids

    def check(source, channels, name, finite=True):
        if source.ndim != 4 or source.shape[1] != channels or source.device != boxes.device:
            raise ValueError(f'{name} must be [B,{channels},H,W] on boxes device')
        if ids.numel() and int(ids.max()) >= len(source):
            raise ValueError(f'{name} image ID outside source batch')
        if finite and not torch.isfinite(source).all():
            raise ValueError(f'{name} must be finite')

    with torch.autocast(device_type=boxes.device.type, enabled=False):
        if aligned_rgb is not None:
            check(aligned_rgb, 32, 'RGB', finite=False)
            if rgb_valid is None:
                raise ValueError('RGB requires independent rgb_valid mask')
            check(rgb_valid, 1, 'RGB validity')
            if rgb_valid.shape[0:1]+rgb_valid.shape[-2:] != aligned_rgb.shape[0:1]+aligned_rgb.shape[-2:]:
                raise ValueError('RGB validity must share RGB batch and grid')
            valid = rgb_valid.detach().float()
            if not ((valid == 0) | (valid == 1)).all():
                raise ValueError('RGB validity must be binary')
            grid = roi_sample_grid(boxes.detach(), output_size, aligned_rgb.shape[-2:], rgb_stride, rgb_origin)
            clean = torch.where(valid.bool(), aligned_rgb.float(), torch.zeros_like(aligned_rgb.float()))
            if not torch.isfinite(clean).all():
                raise ValueError('RGB must be finite at valid source cells')
            selected_valid = valid[ids]
            weights = F.grid_sample(selected_valid, grid, align_corners=False)
            values = F.grid_sample(clean[ids], grid, align_corners=False)
            out_rgb = torch.where((weights > weight_epsilon) & in_image,
                                  values / weights.clamp_min(weight_epsilon),
                                  torch.zeros_like(values))
        elif rgb_valid is not None:
            raise ValueError('rgb_valid without aligned RGB is ambiguous')
        if geometry is not None:
            check(geometry, 5, 'geometry')
            if not ((geometry[:, 4:5] == 0) | (geometry[:, 4:5] == 1)).all():
                raise ValueError('geometry support channel4 must be binary')
            grid = roi_sample_grid(boxes.detach(), output_size, geometry.shape[-2:], geometry_stride, geometry_origin)
            sampled = F.grid_sample(geometry[ids].float(), grid, mode='nearest', align_corners=False)
            out_geo = sampled * sampled[:, 4:5].detach() * in_image
    return out_rgb, out_geo


class _DepthwiseSpatialBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1, groups=32),
                                  nn.Conv2d(32, 32, 1), nn.GroupNorm(8, 32), nn.GELU())

    def forward(self, value):
        return value + self.body(value)


class NativeRoIRefiner(nn.Module):
    """decoder32 + alignedRGB32 + geometry5 + detached probability1 -> delta.

    Decoder/RGB each use no-affine channel LayerNorm. Three depthwise-separable
    residual spatial blocks have GN8+GELU. No evidence gate on final residual.
    Last convolution zero init guarantees exact native logit identity.
    """
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Conv2d(70, 32, 1)
        self.blocks = nn.Sequential(*[_DepthwiseSpatialBlock() for _ in range(3)])
        self.out = nn.Conv2d(32, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, decoder_features, logits, roi_rgb=None, roi_geometry=None):
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError('logits must be [N,1,H,W]')
        n, _, h, w = logits.shape
        if decoder_features.shape != (n, 32, h, w):
            raise ValueError('decoder features must share native grid [N,32,H,W]')
        if roi_rgb is None:
            roi_rgb = decoder_features.new_zeros(n, 32, h, w)
        if roi_geometry is None:
            roi_geometry = decoder_features.new_zeros(n, 5, h, w)
        for value, channels in ((roi_rgb, 32), (roi_geometry, 5)):
            if value.shape != (n, channels, h, w) or value.device != logits.device:
                raise ValueError('sampled evidence must share native logits grid/device')
        if n == 0:
            return logits + sum(p.sum()*0 for p in self.parameters())
        # The zero-init output learns weights around 5e-4 on its first update.
        # With 64 RoIs and 512-square mean mask loss, its hidden gradients can
        # be <FP16's minimum subnormal even at finite, usable scaler values.
        # Keep this SMALL head entirely FP32; changing only its output conv
        # would cast hidden activation gradients back to half and still lose
        # them. The original decoder/backbone AMP contract is unchanged.
        with torch.autocast(device_type=logits.device.type, enabled=False):
            def channel_norm(value):
                return F.layer_norm(value.float().permute(0, 2, 3, 1), (32,)).permute(0, 3, 1, 2)
            inputs = torch.cat((channel_norm(decoder_features), channel_norm(roi_rgb),
                                roi_geometry.float(), logits.detach().float().sigmoid()), dim=1)
            delta = self.out(self.blocks(F.gelu(self.input_proj(inputs))))
        return logits + delta.to(logits.dtype)
