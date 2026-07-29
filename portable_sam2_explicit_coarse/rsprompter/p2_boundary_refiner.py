"""P2-driven local boundary refinement for explicit coarse mask prompts.

ShapePrior predicts the complete ROI mask. This module only writes inside a
detached, raw-coarse-derived search band. Ground truth is accepted only by
``boundary_loss`` and never controls the training/inference forward support.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import RoIAlign
from torch import Tensor


def _group_norm(channels: int, preferred_groups: int = 8) -> nn.GroupNorm:
    groups = min(int(preferred_groups), int(channels))
    while groups > 1 and channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class P2BoundaryRefiner(nn.Module):
    """Bounded P2 high-frequency residual around a raw coarse boundary."""

    def __init__(
        self,
        p2_in_channels: int = 512,
        projected_channels: int = 64,
        mid_channels: int = 64,
        roi_output_size: int = 32,
        coarse_size: int = 64,
        spatial_scale: float = 0.25,
        sampling_ratio: int = 0,
        aligned: bool = True,
        beta: float = 0.20,
        delta_logit_max: float = 2.0,
        foreground_thr: float = 0.50,
        boundary_band_radius: int = 2,
        search_band_radius: int = 4,
        max_search_coverage: float = 0.50,
        boundary_loss_weight: float = 0.05,
        zero_init_residual: bool = True,
    ) -> None:
        super().__init__()
        for name, value in (
            ("p2_in_channels", p2_in_channels),
            ("projected_channels", projected_channels),
            ("mid_channels", mid_channels),
            ("roi_output_size", roi_output_size),
            ("coarse_size", coarse_size),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not 0.0 < float(beta) <= 1.0:
            raise ValueError(f"beta must be in (0, 1], got {beta}")
        if float(delta_logit_max) <= 0:
            raise ValueError("delta_logit_max must be positive")
        if not 0.0 < float(foreground_thr) < 1.0:
            raise ValueError("foreground_thr must be in (0, 1)")
        if int(boundary_band_radius) < 0:
            raise ValueError("boundary_band_radius must be non-negative")
        if int(search_band_radius) < int(boundary_band_radius):
            raise ValueError("search_band_radius must be >= boundary_band_radius")
        if not 0.0 < float(max_search_coverage) <= 1.0:
            raise ValueError("max_search_coverage must be in (0, 1]")
        if float(boundary_loss_weight) < 0:
            raise ValueError("boundary_loss_weight must be non-negative")

        self.p2_in_channels = int(p2_in_channels)
        self.projected_channels = int(projected_channels)
        self.mid_channels = int(mid_channels)
        self.roi_output_size = int(roi_output_size)
        self.coarse_size = int(coarse_size)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)
        self.aligned = bool(aligned)
        self.beta = float(beta)
        self.delta_logit_max = float(delta_logit_max)
        self.foreground_thr = float(foreground_thr)
        self.boundary_band_radius = int(boundary_band_radius)
        self.search_band_radius = int(search_band_radius)
        self.max_search_coverage = float(max_search_coverage)
        self.boundary_loss_weight = float(boundary_loss_weight)

        self.p2_projection = nn.Sequential(
            nn.Conv2d(self.p2_in_channels, self.projected_channels, 1),
            _group_norm(self.projected_channels),
            nn.GELU(),
        )
        self.roi_align = RoIAlign(
            output_size=(self.roi_output_size, self.roi_output_size),
            spatial_scale=self.spatial_scale,
            sampling_ratio=self.sampling_ratio,
            pool_mode="avg",
            aligned=self.aligned,
        )
        fused_channels = 2 * self.projected_channels + 3
        self.fusion = nn.Sequential(
            nn.Conv2d(fused_channels, self.mid_channels, 3, padding=1),
            _group_norm(self.mid_channels),
            nn.GELU(),
            nn.Conv2d(self.mid_channels, self.mid_channels, 3, padding=1),
            _group_norm(self.mid_channels),
            nn.GELU(),
        )
        self.residual_head = nn.Conv2d(self.mid_channels, 1, 1)
        if zero_init_residual:
            nn.init.zeros_(self.residual_head.weight)
            nn.init.zeros_(self.residual_head.bias)

    @staticmethod
    def _replicate_avg3(value: Tensor) -> Tensor:
        return F.avg_pool2d(
            F.pad(value, (1, 1, 1, 1), mode="replicate"), 3, stride=1
        )

    @staticmethod
    def _morphological_boundary(binary: Tensor) -> Tensor:
        padded = F.pad(binary.float(), (1, 1, 1, 1), value=0.0)
        dilated = F.max_pool2d(padded, 3, stride=1)
        eroded = -F.max_pool2d(-padded, 3, stride=1)
        return (dilated - eroded).gt(0).to(binary.dtype)

    @staticmethod
    def _dilate(binary: Tensor, radius: int) -> Tensor:
        if int(radius) <= 0:
            return binary
        kernel = 2 * int(radius) + 1
        return F.max_pool2d(
            binary.float(), kernel_size=kernel, stride=1, padding=int(radius)
        ).gt(0).to(binary.dtype)

    def _raw_support(self, raw_logits: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        with torch.no_grad():
            probability = torch.sigmoid(raw_logits.detach())
            uncertainty = (4.0 * probability * (1.0 - probability)).clamp_(0.0, 1.0)
            smoothed = self._replicate_avg3(probability)
            binary = smoothed.ge(self.foreground_thr).to(probability.dtype)
            boundary_core = self._morphological_boundary(binary)
            boundary_band = self._dilate(boundary_core, self.boundary_band_radius)
            search_band = self._dilate(boundary_core, self.search_band_radius)
            coverage = search_band.float().mean(dim=(1, 2, 3))
            has_support = search_band.flatten(1).any(dim=1)
            support_valid = has_support & coverage.le(self.max_search_coverage)
            search_support = search_band * support_valid[:, None, None, None]
        return probability, uncertainty, boundary_band, search_support, support_valid

    def forward(
        self, p2_feature: Tensor, prompt_rois: Tensor, raw_logits: Tensor
    ) -> Dict[str, Tensor]:
        if p2_feature.dim() != 4 or p2_feature.shape[1] != self.p2_in_channels:
            raise ValueError(
                f"expected P2 [B,{self.p2_in_channels},H,W], got {tuple(p2_feature.shape)}"
            )
        if prompt_rois.dim() != 2 or prompt_rois.shape[1] != 5:
            raise ValueError(f"prompt_rois must be [N,5], got {tuple(prompt_rois.shape)}")
        if raw_logits.dim() != 4 or raw_logits.shape[1] != 1:
            raise ValueError(f"raw_logits must be [N,1,H,W], got {tuple(raw_logits.shape)}")
        if raw_logits.shape[-2:] != (self.coarse_size, self.coarse_size):
            raise ValueError(
                f"expected raw {self.coarse_size}x{self.coarse_size}, got {tuple(raw_logits.shape[-2:])}"
            )
        if prompt_rois.shape[0] != raw_logits.shape[0]:
            raise ValueError("prompt_rois and raw_logits batch sizes differ")
        if prompt_rois.numel() and (
            not torch.isfinite(prompt_rois).all()
            or (prompt_rois[:, 3] <= prompt_rois[:, 1]).any()
            or (prompt_rois[:, 4] <= prompt_rois[:, 2]).any()
        ):
            raise ValueError("prompt_rois contain non-finite or invalid boxes")

        probability, uncertainty, boundary_band, search_support, support_valid = self._raw_support(raw_logits)
        if raw_logits.shape[0] == 0:
            zero = raw_logits.new_zeros(())
            return dict(
                refined_logits=raw_logits,
                delta_logits=torch.zeros_like(raw_logits),
                raw_boundary_band=boundary_band,
                search_support=search_support,
                support_valid=support_valid,
                search_coverage_sum=zero,
                projected_feature_norm_sum=zero,
                highpass_feature_norm_sum=zero,
            )

        projected_p2 = self.p2_projection(p2_feature.detach())
        roi_feature = self.roi_align(projected_p2, prompt_rois)
        low_frequency = self._replicate_avg3(roi_feature)
        high_frequency = roi_feature - low_frequency
        target_size = (self.roi_output_size, self.roi_output_size)
        probability_32 = F.interpolate(probability, target_size, mode="bilinear", align_corners=False)
        uncertainty_32 = F.interpolate(uncertainty, target_size, mode="bilinear", align_corners=False)
        support_32 = F.adaptive_max_pool2d(search_support, target_size)
        fused = self.fusion(
            torch.cat([roi_feature, high_frequency, probability_32, uncertainty_32, support_32], dim=1)
        )
        fused = F.interpolate(
            fused, (self.coarse_size, self.coarse_size), mode="bilinear", align_corners=False
        )
        bounded_source = self.delta_logit_max * torch.tanh(self.residual_head(fused))
        delta = self.beta * bounded_source * search_support.to(bounded_source.dtype)
        refined = raw_logits + delta

        with torch.no_grad():
            search_coverage = search_support.float().mean(dim=(1, 2, 3))
            projected_norm = roi_feature.float().flatten(1).norm(dim=1)
            highpass_norm = high_frequency.float().flatten(1).norm(dim=1)
        return dict(
            refined_logits=refined,
            delta_logits=delta,
            raw_boundary_band=boundary_band,
            search_support=search_support,
            support_valid=support_valid,
            search_coverage_sum=search_coverage.sum(),
            projected_feature_norm_sum=projected_norm.sum(),
            highpass_feature_norm_sum=highpass_norm.sum(),
        )

    def boundary_loss(
        self,
        raw_logits: Tensor,
        delta_logits: Tensor,
        raw_boundary_band: Tensor,
        search_support: Tensor,
        target_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """Return unweighted per-ROI loss sum and valid ROI count."""
        if target_mask.dim() == 3:
            target_mask = target_mask.unsqueeze(1)
        if target_mask.shape != raw_logits.shape:
            raise ValueError(
                f"target {tuple(target_mask.shape)} must match raw {tuple(raw_logits.shape)}"
            )
        with torch.no_grad():
            target_binary = target_mask.ge(0.5).to(raw_logits.dtype)
            target_boundary = self._dilate(
                self._morphological_boundary(target_binary), self.boundary_band_radius
            )
            loss_support = search_support * torch.maximum(raw_boundary_band, target_boundary)
            pixel_count = loss_support.flatten(1).sum(dim=1)
            valid_roi = pixel_count.gt(0)

        refined_for_boundary = raw_logits.detach() + delta_logits
        pixel_loss = F.binary_cross_entropy_with_logits(
            refined_for_boundary, target_binary, reduction="none"
        )
        per_roi = (pixel_loss * loss_support).flatten(1).sum(dim=1) / pixel_count.clamp_min(1.0)
        local_sum = per_roi[valid_roi].sum()
        if not valid_roi.any():
            local_sum = delta_logits.sum() * 0.0
        local_count = valid_roi.sum().to(dtype=raw_logits.dtype).detach()
        with torch.no_grad():
            foreground_sum = (target_binary * loss_support).sum()
            support_pixel_sum = loss_support.sum()
        return local_sum, local_count, dict(
            loss_support_pixel_sum=support_pixel_sum,
            loss_support_foreground_sum=foreground_sum,
            loss_valid_roi_count=local_count,
        )
