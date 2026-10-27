"""Standalone UBR components; no detector/UDPR wiring or implicit loss weights.

All inputs share an explicitly aligned target pixel grid (normally satellite
512x512). This module never resizes geometry or decoder features. Signed
distances and band_width are target *pixels*, not metres. GroupNorm computes
statistics on the supplied window; crops are not equivalent to full images.
"""
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _tensor4(name: str, value: torch.Tensor, channels: int) -> None:
    if value.ndim != 4 or value.shape[1] != channels:
        raise ValueError(f'{name} must have shape [N,{channels},H,W]')


@torch.no_grad()
def signed_distance_transform(binary_mask: torch.Tensor) -> torch.Tensor:
    """Positive inside, negative outside; empty/full masks have +/-infinity.

    Infinity encodes no internal boundary, so band_mask returns an empty band
    even for very large widths. CPU scipy EDT is deliberate: offline/state
    preparation, not a differentiable depth/geometry transformation.
    """
    from scipy.ndimage import distance_transform_edt

    _tensor4('binary_mask', binary_mask, 1)
    masks = binary_mask.detach().bool().cpu().numpy()
    result = np.empty(masks.shape, dtype=np.float32)
    for i, mask in enumerate(masks[:, 0]):
        if not mask.any():
            result[i, 0] = -np.inf
        elif mask.all():
            result[i, 0] = np.inf
        else:
            result[i, 0] = distance_transform_edt(mask) - distance_transform_edt(~mask)
    return torch.from_numpy(result).to(binary_mask.device)


def band_mask(sdt: torch.Tensor, width: int = 8) -> torch.Tensor:
    _tensor4('sdt', sdt, 1)
    if not isinstance(width, int) or isinstance(width, bool) or width < 1:
        raise ValueError('band width must be a positive integer in target pixels')
    return torch.isfinite(sdt.detach()) & (sdt.detach().abs() <= width)


@torch.no_grad()
def prediction_boundary_state(logits: torch.Tensor, threshold: float = 0.4,
                              width: int = 8):
    """Use the deployment decision p>=threshold, with no gradients through state."""
    _tensor4('logits', logits, 1)
    if not 0 < threshold < 1:
        raise ValueError('threshold must be in (0,1)')
    sdt = signed_distance_transform(logits.detach().float().sigmoid() >= threshold)
    return sdt, band_mask(sdt, width)


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1),
                                  nn.GroupNorm(8, channels), nn.SiLU())

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.body(value)


class UnifiedBoundaryRefiner(nn.Module):
    """40-channel spatial UBR, outputting z + band*r with unbounded residual.

    State inputs probability/SDT/neighbors are detached. Decoder features and
    z's direct path remain differentiable. There is no alpha scalar and no
    support gate: zero geometry still allows satellite-side refinement.
    """
    def __init__(self, hidden: int = 48, depth: int = 7, band_width: int = 8,
                 checkpoint_blocks: bool = False):
        super().__init__()
        if hidden <= 0 or hidden % 8 or depth < 1:
            raise ValueError('hidden must be positive/divisible by 8 and depth >=1')
        band_mask(torch.zeros(0, 1, 1, 1), band_width)
        self.band_width = band_width
        self.checkpoint_blocks = checkpoint_blocks
        self.input_proj = nn.Conv2d(40, hidden, 3, padding=1)
        self.blocks = nn.ModuleList([_ResidualBlock(hidden) for _ in range(depth)])
        self.out = nn.Conv2d(hidden, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, decoder_features: torch.Tensor, logits: torch.Tensor,
                sdt: torch.Tensor, neighbors: torch.Tensor,
                geometry: Optional[torch.Tensor] = None) -> torch.Tensor:
        _tensor4('logits', logits, 1)
        _tensor4('decoder_features', decoder_features, 32)
        _tensor4('sdt', sdt, 1)
        _tensor4('neighbors', neighbors, 1)
        if geometry is None:
            geometry = logits.new_zeros(logits.shape[0], 5, *logits.shape[-2:])
        _tensor4('geometry', geometry, 5)
        for name, tensor in [('decoder_features', decoder_features), ('sdt', sdt),
                             ('neighbors', neighbors), ('geometry', geometry)]:
            if (tensor.shape[0], *tensor.shape[-2:]) != (logits.shape[0], *logits.shape[-2:]):
                raise ValueError(f'{name} must share logits batch and spatial grid')
            if tensor.device != logits.device:
                raise ValueError(f'{name} must share logits device')
        if logits.shape[0] == 0:
            # Keep every parameter in the empty-rank graph, without convolution.
            return logits + sum(parameter.sum() * 0 for parameter in self.parameters())
        state_sdt = torch.nan_to_num(sdt.detach(), nan=0., posinf=0., neginf=0.)
        inputs = [decoder_features, logits.detach().sigmoid(), state_sdt,
                  neighbors.detach(), geometry]
        value = self.input_proj(torch.cat([t.to(decoder_features.dtype) for t in inputs], dim=1))
        for block in self.blocks:
            if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
                value = checkpoint(block, value, use_reentrant=False)
            else:
                value = block(value)
        residual = self.out(value).to(logits.dtype)
        active = band_mask(sdt, self.band_width)
        # where preserves outside logits exactly, including their direct gradient.
        return torch.where(active, logits + residual, logits)


def boundary_loss_components(final_logits: torch.Tensor, target: torch.Tensor,
                             band: torch.Tensor, image_ids: torch.Tensor,
                             num_images: int, gamma: float = 2.,
                             eps: float = 1e-6) -> Dict[str, torch.Tensor]:
    """UNWEIGHTED focal BCE and soft Dice; caller applies lambda=0.1 once.

    Pixel mean within each instance's band -> mean over nonempty instances in
    each image -> mean over ALL num_images (empty images contribute zero).
    image_ids maps instances to local batch images. No all_reduce is performed;
    equal image counts per rank permit ordinary DDP rank-mean normalization.
    """
    _tensor4('final_logits', final_logits, 1)
    if target.shape != final_logits.shape or band.shape != final_logits.shape:
        raise ValueError('target and band must have exactly the logits shape')
    if band.dtype != torch.bool:
        raise ValueError('band must be boolean')
    if image_ids.ndim != 1 or image_ids.numel() != final_logits.shape[0]:
        raise ValueError('image_ids must contain one index per instance')
    if image_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError('image_ids must be integer')
    if not isinstance(num_images, int) or num_images < 1 or gamma < 0 or eps <= 0:
        raise ValueError('num_images>=1, gamma>=0 and eps>0 required')
    if image_ids.numel() and (image_ids.min() < 0 or image_ids.max() >= num_images):
        raise ValueError('image_ids outside local image batch')
    if target.device != final_logits.device or band.device != final_logits.device:
        raise ValueError('target and band must share logits device')
    if not torch.isfinite(target).all() or (target < 0).any() or (target > 1).any():
        raise ValueError('target must be finite in [0,1]')
    z, y = final_logits.float(), target.float()
    ids = image_ids.to(z.device)
    zero = z.sum() * 0
    focal_sum, dice_sum = zero, zero
    for image_index in range(num_images):
        focal_instances, dice_instances = [], []
        for instance_index in torch.where(ids == image_index)[0]:
            valid = band[instance_index]
            if not valid.any():
                continue
            zi, yi = z[instance_index][valid], y[instance_index][valid]
            probability = zi.sigmoid()
            pt = probability * yi + (1 - probability) * (1 - yi)
            focal_instances.append(((1 - pt).pow(gamma) *
                                    F.binary_cross_entropy_with_logits(zi, yi, reduction='none')).mean())
            dice_instances.append(1 - (2 * (probability * yi).sum()) /
                                  (probability.sum() + yi.sum() + eps))
        if focal_instances:
            focal_sum = focal_sum + torch.stack(focal_instances).mean()
            dice_sum = dice_sum + torch.stack(dice_instances).mean()
    return {'focal': focal_sum / num_images, 'dice': dice_sum / num_images}


def boundary_loss(*args, **kwargs) -> torch.Tensor:
    """Raw focal+Dice; no internal 0.1 multiplication."""
    components = boundary_loss_components(*args, **kwargs)
    return components['focal'] + components['dice']
