"""Proposal-conditioned multi-scale renderer for the dense prompt canvas.

R3 deliberately does not replace the ShapePriorInjector: the latter remains
the source of sparse 2P2N prompts and established coarse supervision. This
module only renders the ROI-local logits delivered through the existing
canvas paste/support/PromptEncoder path.
"""

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import RoIAlign
from torch import Tensor


def _group_norm(channels: int, preferred_groups: int = 32) -> nn.GroupNorm:
    groups = min(int(preferred_groups), int(channels))
    while groups > 1 and channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ProposalCanvasRenderer(nn.Module):
    """R3: detached P3/P4 RoI features to one 128x128 canvas-logit map."""

    def __init__(
        self,
        feature_indices: Sequence[int] = (1, 2),
        featmap_strides: Sequence[int] = (8, 16),
        in_channels: int = 512,
        branch_channels: int = 128,
        p3_roi_size: int = 64,
        p4_roi_size: int = 32,
        output_size: int = 128,
        sampling_ratio: int = 0,
        aligned: bool = True,
    ) -> None:
        super().__init__()
        if tuple(feature_indices) != (1, 2):
            raise ValueError("R3 v1 is fixed to P3/P4 feature indices (1, 2)")
        if len(featmap_strides) != 2 or any(int(v) <= 0 for v in featmap_strides):
            raise ValueError("featmap_strides must contain two positive entries")
        if min(in_channels, branch_channels, p3_roi_size, p4_roi_size, output_size) <= 0:
            raise ValueError("R3 channels and spatial sizes must be positive")
        self.feature_indices = tuple(int(v) for v in feature_indices)
        self.in_channels = int(in_channels)
        self.output_size = int(output_size)
        self.p3_roi_size = int(p3_roi_size)
        self.p4_roi_size = int(p4_roi_size)
        self.p3_align = RoIAlign(
            output_size=(self.p3_roi_size, self.p3_roi_size),
            spatial_scale=1.0 / float(featmap_strides[0]),
            sampling_ratio=int(sampling_ratio), pool_mode="avg", aligned=bool(aligned),
        )
        self.p4_align = RoIAlign(
            output_size=(self.p4_roi_size, self.p4_roi_size),
            spatial_scale=1.0 / float(featmap_strides[1]),
            sampling_ratio=int(sampling_ratio), pool_mode="avg", aligned=bool(aligned),
        )
        self.p3_project = nn.Sequential(nn.Conv2d(self.in_channels, branch_channels, 1), _group_norm(branch_channels), nn.GELU())
        self.p4_project = nn.Sequential(nn.Conv2d(self.in_channels, branch_channels, 1), _group_norm(branch_channels), nn.GELU())
        mid_channels = max(1, int(branch_channels) // 2)
        self.fusion = nn.Sequential(
            nn.Conv2d(branch_channels, mid_channels, 3, padding=1), _group_norm(mid_channels), nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1), _group_norm(mid_channels), nn.GELU(),
        )
        self.output = nn.Conv2d(mid_channels, 1, 1)
        # Stable initialization; this is not a no-mask equivalence claim.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: Sequence[Tensor], prompt_rois: Tensor) -> Tensor:
        if prompt_rois.dim() != 2 or prompt_rois.shape[1] != 5:
            raise ValueError(f"prompt_rois must be [N,5], got {tuple(prompt_rois.shape)}")
        if len(features) <= max(self.feature_indices):
            raise ValueError("R3 requires a full PAFPN feature tuple containing P3/P4")
        p3, p4 = (features[idx] for idx in self.feature_indices)
        for name, feature in (("P3", p3), ("P4", p4)):
            if feature.dim() != 4 or feature.shape[1] != self.in_channels:
                raise ValueError(f"R3 {name} must be [B,{self.in_channels},H,W], got {tuple(feature.shape)}")
        if prompt_rois.numel() and (not torch.isfinite(prompt_rois).all() or (prompt_rois[:, 3] <= prompt_rois[:, 1]).any() or (prompt_rois[:, 4] <= prompt_rois[:, 2]).any()):
            raise ValueError("R3 prompt_rois contain non-finite or invalid boxes")
        if prompt_rois.shape[0] == 0:
            return p3.new_empty((0, 1, self.output_size, self.output_size))
        # Do not let renderer loss update the shared detector/FPN features.
        p3_roi = self.p3_align(p3.detach(), prompt_rois)
        p4_roi = self.p4_align(p4.detach(), prompt_rois)
        fused = self.p3_project(p3_roi) + F.interpolate(self.p4_project(p4_roi), size=(self.p3_roi_size, self.p3_roi_size), mode="bilinear", align_corners=False)
        return self.output(F.interpolate(self.fusion(fused), size=(self.output_size, self.output_size), mode="bilinear", align_corners=False))
