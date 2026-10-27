"""Unified Boundary Refiner (UBR) - production module.

Replaces UDPR in both single-stream and dual-stream models (design doc
docs/UBR-UNIFIED-ARCHITECTURE-DESIGN-20261026.md, section 2). Single
architecture, optional UAV geometry port:

  inputs (per pixel, native 512x512):
    decoder_feat  32ch  stride-4 decoder features, bilinearly upsampled
    mask_logits    1ch  current instance mask logits, upsampled
    sdt            1ch  signed distance transform of the binarized mask
    neighbors      1ch  sum of other instances' binary masks (pre-refinement)
    uav_geom       5ch  UAV geometry map [h, dx, dy, viewcount, support];
                        zero-filled when absent (single-stream / degraded)

  output: per-pixel binary logit, applied only inside |SDT| <= k bands via
  mask[band] = UBR[band] (band reclassification), largest connected component
  kept per instance. In-training loss: band focal BCE + Dice (loss_ubr).

All geometry helpers are pure functions (explicit inputs/outputs, no hidden
state) so probes and tests can exercise them independently.
"""
from typing import Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

UAV_GEOMETRY_CHANNELS = 5  # height, dx, dy, viewcount, support


def signed_distance_transform(mask: torch.Tensor) -> torch.Tensor:
    """SDT of a binary mask; positive inside, negative outside, in pixels.

    mask: [B, 1, H, W] float/bool; must be non-empty per sample (an empty
    mask has no boundary and its band would span the whole image). Returns
    float32 [B, 1, H, W].
    """
    from scipy import ndimage

    assert mask.ndim == 4 and mask.shape[1] == 1
    out = torch.empty(mask.shape, dtype=torch.float32)
    for b in range(mask.shape[0]):
        m = mask[b, 0].detach().cpu().numpy() > 0.5
        if not m.any():
            out[b, 0] = 0.0
            continue
        inside = ndimage.distance_transform_edt(m).astype(np.float32)
        outside = ndimage.distance_transform_edt(~m).astype(np.float32)
        out[b, 0] = torch.from_numpy(inside - outside)
    return out.to(mask.device)


def band_mask(sdt: torch.Tensor, k: int) -> torch.Tensor:
    """Boolean band |SDT| <= k around the current boundary. [B,1,H,W]."""
    return sdt.abs() <= float(k)


def signed_distance_transform_numpy(mask: np.ndarray) -> np.ndarray:
    """Batched-CPU SDT, bit-identical to signed_distance_transform.

    mask: [B,H,W] uint8/bool. Same per-sample arithmetic order as the torch
    version (each EDT cast to float32 BEFORE the subtraction), so band
    masks derived from either implementation agree bit-for-bit. Zero rows
    (empty masks / bucket padding) stay exactly 0 like the torch version.
    """
    from scipy import ndimage

    assert mask.ndim == 3
    out = np.zeros(mask.shape, dtype=np.float32)
    for b in range(mask.shape[0]):
        m = mask[b] > 0
        if not m.any():
            continue
        out[b] = (ndimage.distance_transform_edt(m).astype(np.float32)
                  - ndimage.distance_transform_edt(~m).astype(np.float32))
    return out


def build_constant_geometry(
    real_geometry: torch.Tensor, channel_means: torch.Tensor
) -> torch.Tensor:
    """Matched constant control for the UAV port.

    Fills train-only per-channel means where support == 1, zeros in holes, so
    the support channel distribution matches the real arm exactly.
    real_geometry: [B,5,H,W]; channel_means: [4] — train-split means of
    channels 0..3 under support == 1 (the support channel itself is copied).
    """
    assert real_geometry.shape[1] == UAV_GEOMETRY_CHANNELS
    assert channel_means.numel() == 4, 'means for channels 0..3 only'
    fill = channel_means.to(real_geometry.dtype).view(1, 4, 1, 1)
    filled = fill.expand(real_geometry.shape[0], 4, *real_geometry.shape[2:])
    const = torch.cat([filled, real_geometry[:, 4:5]], dim=1)
    return const * real_geometry[:, 4:5]


class ConvBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class UnifiedBoundaryRefiner(nn.Module):
    """Dense CNN producing in-band per-pixel binary logits.

    Capacity contract (UBR v1.1, docs/UBR-V1.1-PERF-AND-RELAUNCH-20261027.md
    item B1): hidden=32 x depth=6 (67,457 params) is the NEW UNIFIED
    definition for every single-/dual-stream arm from EXP-0028-R2 onward.
    The legacy (48, 7) capacity (163,537 params) remains constructible ONLY
    for the G1 old-capacity equivalence replay against the dead EXP-0028
    arm. Pre-v1.1 probes that relied on the ctor default pinned the legacy
    capacity implicitly.

    grad_checkpoint (B2): recompute the conv blocks' activations in
    backward via torch.utils.checkpoint (~8x UBR activation saving, forward
    recompute cost). Default OFF at the 512 protocol; required ON at 1024.
    """

    def __init__(self, base_ch: int = 35, hidden: int = 32, depth: int = 6,
                 grad_checkpoint: bool = False):
        super().__init__()
        assert base_ch == 32 + 1 + 1 + 1  # decoder feat + logits + sdt + nbr
        self.grad_checkpoint = bool(grad_checkpoint)
        self.input_proj = nn.Conv2d(
            base_ch + UAV_GEOMETRY_CHANNELS, hidden, 3, padding=1
        )
        self.blocks = nn.Sequential(*[ConvBlock(hidden) for _ in range(depth)])
        self.out = nn.Conv2d(hidden, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        decoder_feat: torch.Tensor,
        mask_logits: torch.Tensor,
        sdt: torch.Tensor,
        neighbors: torch.Tensor,
        uav_geom: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns [B,1,H,W] binary logits; zero-init output starts neutral."""
        target_hw = mask_logits.shape[-2:]
        if uav_geom is None:
            uav_geom = mask_logits.new_zeros(
                mask_logits.shape[0], UAV_GEOMETRY_CHANNELS, *target_hw
            )
        assert uav_geom.shape[1] == UAV_GEOMETRY_CHANNELS
        if uav_geom.shape[-2:] != target_hw:
            uav_geom = F.interpolate(
                uav_geom, target_hw, mode="bilinear", align_corners=False
            )
        if decoder_feat.shape[-2:] != target_hw:
            decoder_feat = F.interpolate(
                decoder_feat, target_hw, mode="bilinear", align_corners=False
            )
        x = torch.cat(
            [decoder_feat, mask_logits, sdt, neighbors, uav_geom], dim=1
        )
        x = self.input_proj(x)
        for block in self.blocks:
            if (self.grad_checkpoint and self.training
                    and torch.is_grad_enabled()):
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.out(x)


def refiner_param_count(hidden: int, depth: int) -> int:
    """Total trainable parameters of UnifiedBoundaryRefiner(hidden, depth).

    Pure function so the integration's fail-closed pin and the harnesses
    share one arithmetic source: input_proj + depth * (conv + groupnorm)
    + 1x1 readout, all bias-inclusive.
    """
    proj = (32 + 1 + 1 + 1 + UAV_GEOMETRY_CHANNELS) * hidden * 9 + hidden
    block = hidden * hidden * 9 + hidden + 2 * hidden  # conv / GroupNorm(8)
    return proj + depth * block + hidden + 1


def apply_band_reclassification(
    mask: torch.Tensor, ubr_logits: torch.Tensor, sdt: torch.Tensor, k: int
) -> torch.Tensor:
    """Write UBR's in-band decision back; inference-only (no gradient).

    Write rule per band pixel: logit > 0 -> 1, logit < 0 -> 0, logit == 0 ->
    original value (a zero-initialized UBR is an exact copy). Anti-speckle:
    the largest 8-connected component of the post-write mask is kept and
    smaller components' pixels revert to the ORIGINAL mask value; then all
    pixels OUTSIDE the band are restored to the original, so the result may
    differ from the input only inside |SDT| <= k (bit-exact containment).
    mask: [B,1,H,W] float (0/1); returns refined float mask of same shape.
    """
    from scipy import ndimage

    assert mask.shape == ubr_logits.shape == sdt.shape
    band = band_mask(sdt, k)
    decision = torch.where(
        ubr_logits > 0, torch.ones_like(mask),
        torch.where(ubr_logits < 0, torch.zeros_like(mask), mask),
    )
    refined = decision * band + mask * ~band
    out = refined.clone()
    structure = np.ones((3, 3))  # 8-connectivity: SAM2 masks are 8-connected
    for b in range(refined.shape[0]):
        m = refined[b, 0].detach().cpu().numpy() > 0.5
        if not m.any():
            out[b, 0] = 0.0
            continue
        labels, n = ndimage.label(m, structure=structure)
        if n > 1:
            sizes = ndimage.sum(m, labels, range(1, n + 1))
            keep = 1 + int(np.argmax(sizes))
            dropped = torch.from_numpy(labels != keep).to(refined.device)
            out[b, 0] = torch.where(
                dropped, mask[b, 0], out[b, 0]
            )
        band_b = band[b, 0]
        out[b, 0] = torch.where(band_b, out[b, 0], mask[b, 0])
    return out
