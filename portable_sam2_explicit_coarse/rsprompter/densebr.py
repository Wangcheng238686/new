"""Uncertainty-aware residual calibration for explicit dense mask prompts.

The canonical DenseBR lives entirely in ROI-local logit space:

    coarse logits -> bounded residual calibration -> refined coarse logits

The refined logits are the single source for coarse supervision, adaptive
2P2N mining and the full-image dense prompt canvas.  No embedding-space gain,
feedback decoder pass, directional controller or extra DenseBR loss is kept.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _group_norm(channels: int, preferred_groups: int = 8) -> nn.GroupNorm:
    groups = min(int(preferred_groups), int(channels))
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def _bounded_scale_parameter(initial: float, maximum: float) -> nn.Parameter:
    if not 0.0 < initial < maximum:
        raise ValueError(
            f"scale initial value must be in (0, {maximum}), got {initial}"
        )
    ratio = initial / maximum
    return nn.Parameter(
        torch.tensor(math.log(ratio / (1.0 - ratio)), dtype=torch.float32)
    )


def _logit(initial: float) -> float:
    if not 0.0 < initial < 1.0:
        raise ValueError(f"sigmoid initial value must be in (0, 1), got {initial}")
    return math.log(initial / (1.0 - initial))


class DenseBR(nn.Module):
    """Boundary-aware dense-prompt residual calibrator.

    Args:
        roi_in_channels: Channels of the mask RoI feature.
        roi_channels: Projected RoI feature channels.
        cue_channels: Projected probability/uncertainty/boundary channels.
        mid_channels: Hidden channels used by the residual and gate heads.
        beta_init: Initial global residual amplitude.
        beta_max: Maximum global residual amplitude.
        delta_logit_max: Bound of the learned per-pixel logit residual.
        quality_gate_init: Initial per-ROI quality gate before learning.
        image_size: Reference image size used to normalize box geometry.
        foreground_thr: Threshold used only for detached quality statistics.
        detach_prompt_cues: Stop gradients through geometric cue extraction.
        zero_init_residual: Make the initial forward exactly identity.
    """

    def __init__(
        self,
        roi_in_channels: int = 512,
        roi_channels: int = 64,
        cue_channels: int = 32,
        mid_channels: int = 64,
        beta_init: float = 0.05,
        beta_max: float = 0.20,
        delta_logit_max: float = 2.0,
        quality_gate_init: float = 0.50,
        image_size: int = 1024,
        foreground_thr: float = 0.50,
        detach_prompt_cues: bool = True,
        zero_init_residual: bool = True,
    ) -> None:
        super().__init__()
        for name, value in (
            ("roi_in_channels", roi_in_channels),
            ("roi_channels", roi_channels),
            ("cue_channels", cue_channels),
            ("mid_channels", mid_channels),
        ):
            if int(value) <= 0:
                raise ValueError(f"DenseBR {name} must be positive, got {value}")
        if delta_logit_max <= 0:
            raise ValueError("DenseBR delta_logit_max must be positive")
        if beta_max <= 0:
            raise ValueError("DenseBR beta_max must be positive")
        if not 0.0 < foreground_thr < 1.0:
            raise ValueError("DenseBR foreground_thr must be in (0, 1)")

        self.beta_max = float(beta_max)
        self.delta_logit_max = float(delta_logit_max)
        self.image_size = float(image_size)
        self.foreground_thr = float(foreground_thr)
        self.detach_prompt_cues = bool(detach_prompt_cues)

        self.roi_proj = nn.Sequential(
            nn.Conv2d(int(roi_in_channels), int(roi_channels), 1),
            _group_norm(int(roi_channels)),
            nn.GELU(),
            nn.Conv2d(int(roi_channels), int(roi_channels), 3, padding=1),
            _group_norm(int(roi_channels)),
            nn.GELU(),
        )
        self.cue_proj = nn.Sequential(
            nn.Conv2d(3, int(cue_channels), 3, padding=1),
            _group_norm(int(cue_channels)),
            nn.GELU(),
        )
        fused_channels = int(roi_channels) + int(cue_channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(fused_channels, int(mid_channels), 3, padding=1),
            _group_norm(int(mid_channels), 16),
            nn.GELU(),
            nn.Conv2d(int(mid_channels), int(mid_channels), 3, padding=1),
            _group_norm(int(mid_channels), 16),
            nn.GELU(),
        )

        # Learned spatial gate and residual share the same local evidence but
        # have independent output heads.  The residual is zero initialized so
        # enabling DenseBR starts as an exact identity function.
        self.spatial_gate_head = nn.Conv2d(int(mid_channels), 1, 1)
        self.residual_head = nn.Conv2d(int(mid_channels), 1, 1)
        nn.init.zeros_(self.spatial_gate_head.weight)
        nn.init.zeros_(self.spatial_gate_head.bias)
        if zero_init_residual:
            nn.init.zeros_(self.residual_head.weight)
            nn.init.zeros_(self.residual_head.bias)

        condition_channels = max(32, int(roi_channels))
        self.quality_roi_proj = nn.Linear(int(roi_in_channels), condition_channels)
        # Four box geometry terms + four detached coarse-mask statistics.
        self.quality_head = nn.Sequential(
            nn.Linear(condition_channels + 8, condition_channels),
            nn.GELU(),
            nn.Linear(condition_channels, 1),
        )
        nn.init.zeros_(self.quality_head[-1].weight)
        nn.init.constant_(self.quality_head[-1].bias, _logit(quality_gate_init))

        self.beta_logit = _bounded_scale_parameter(float(beta_init), self.beta_max)

    @property
    def beta(self) -> Tensor:
        return self.beta_max * torch.sigmoid(self.beta_logit)

    @staticmethod
    def _prompt_cues(prompt_logits: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        prob = torch.sigmoid(prompt_logits)
        uncertainty = (4.0 * prob * (1.0 - prob)).clamp(0.0, 1.0)
        # Explicit zero padding treats the ROI exterior as background, so an
        # all-foreground coarse mask still exposes its outer contour.
        padded = F.pad(prob, (1, 1, 1, 1), mode="constant", value=0.0)
        dilated = F.max_pool2d(padded, kernel_size=3, stride=1, padding=0)
        eroded = -F.max_pool2d(-padded, kernel_size=3, stride=1, padding=0)
        boundary = (dilated - eroded).clamp(0.0, 1.0)
        return prob, uncertainty, boundary

    def _geometry(self, boxes: Tensor) -> Tensor:
        width = (boxes[:, 2] - boxes[:, 0]).clamp(min=1.0)
        height = (boxes[:, 3] - boxes[:, 1]).clamp(min=1.0)
        width_norm = (width / self.image_size).clamp(max=1.0)
        height_norm = (height / self.image_size).clamp(max=1.0)
        area_norm = (width_norm * height_norm).clamp(min=1e-6)
        aspect = torch.log((width / height).clamp(min=1e-6))
        return torch.stack(
            [width_norm, height_norm, torch.log(area_norm), aspect], dim=1
        )

    def forward(
        self,
        roi_features: Tensor,
        prompt_logits: Tensor,
        boxes: Tensor,
        debug_stats: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        if prompt_logits.dim() != 4 or prompt_logits.shape[1] != 1:
            raise ValueError(
                "DenseBR expects prompt_logits [N,1,H,W], got "
                f"{tuple(prompt_logits.shape)}"
            )
        if roi_features.shape[0] != prompt_logits.shape[0]:
            raise ValueError("DenseBR ROI feature and prompt batch sizes do not match")
        if boxes is None or boxes.shape[0] != prompt_logits.shape[0]:
            raise ValueError("DenseBR requires one prompt box per ROI")
        if prompt_logits.shape[0] == 0:
            return prompt_logits

        cue_logits = prompt_logits.detach() if self.detach_prompt_cues else prompt_logits
        prob, uncertainty, boundary = self._prompt_cues(cue_logits)

        roi_feat = self.roi_proj(roi_features)
        roi_feat = F.interpolate(
            roi_feat,
            size=prompt_logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cue_feat = self.cue_proj(torch.cat([prob, uncertainty, boundary], dim=1))
        fused = self.fuse(torch.cat([roi_feat, cue_feat], dim=1))

        learned_gate = torch.sigmoid(self.spatial_gate_head(fused))
        # Boundary evidence is primary. Uncertainty contributes only where a
        # boundary is present, avoiding broad writes in p≈0.5 interiors.
        boundary_uncertainty = uncertainty * boundary
        boundary_support = (boundary + boundary_uncertainty).clamp(0.0, 1.0)
        spatial_gate = (
            0.50 * boundary
            + 0.25 * boundary_uncertainty
            + 0.25 * learned_gate * boundary_support
        ).clamp(0.0, 1.0)

        roi_pooled = roi_features.mean(dim=(2, 3))
        roi_condition = F.gelu(self.quality_roi_proj(roi_pooled))
        foreground_ratio = (prob >= self.foreground_thr).to(prob.dtype).mean(
            dim=(1, 2, 3)
        )
        prompt_stats = torch.stack(
            [
                prob.mean(dim=(1, 2, 3)),
                uncertainty.mean(dim=(1, 2, 3)),
                boundary.mean(dim=(1, 2, 3)),
                foreground_ratio,
            ],
            dim=1,
        )
        condition = torch.cat(
            [
                roi_condition,
                self._geometry(boxes).to(dtype=roi_condition.dtype),
                prompt_stats.to(dtype=roi_condition.dtype),
            ],
            dim=1,
        )
        quality_gate = torch.sigmoid(self.quality_head(condition))

        raw_delta = self.residual_head(fused)
        bounded_delta = self.delta_logit_max * torch.tanh(raw_delta)
        beta = self.beta.to(dtype=prompt_logits.dtype)
        delta = (
            beta
            * quality_gate[:, :, None, None].to(dtype=prompt_logits.dtype)
            * spatial_gate.to(dtype=prompt_logits.dtype)
            * bounded_delta.to(dtype=prompt_logits.dtype)
        )
        refined = prompt_logits + delta

        if debug_stats is not None:
            with torch.no_grad():
                base_norm = (
                    prompt_logits.detach().float().flatten(1).norm(dim=1).mean()
                    .clamp_min(1e-6)
                )
                delta_norm = delta.detach().float().flatten(1).norm(dim=1).mean()
                debug_stats.update(
                    {
                        "DENSEBR/active_rois": prompt_logits.new_tensor(
                            float(prompt_logits.shape[0])
                        ),
                        "DENSEBR/beta": self.beta.detach().float(),
                        "DENSEBR/quality_gate_mean": quality_gate.detach().float().mean(),
                        "DENSEBR/quality_gate_std": quality_gate.detach().float().std(
                            unbiased=False
                        ),
                        "DENSEBR/uncertainty_mean": uncertainty.detach().float().mean(),
                        "DENSEBR/boundary_mean": boundary.detach().float().mean(),
                        "DENSEBR/learned_gate_mean": learned_gate.detach().float().mean(),
                        "DENSEBR/write_mean": spatial_gate.detach().float().mean(),
                        "DENSEBR/write_coverage": (
                            spatial_gate.detach() > 0.10
                        ).float().mean(),
                        "DENSEBR/delta_abs": delta.detach().float().abs().mean(),
                        "DENSEBR/delta_ratio": delta_norm / base_norm,
                    }
                )
        return refined
