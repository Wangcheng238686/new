"""Confidence-gated full-channel residuals with observable magnitude."""

import torch
from torch import nn


class ConfidenceFusion(nn.Module):
    def __init__(self, satellite_channels, uav_channels=256, alpha_init=0.05):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(uav_channels, satellite_channels, 1, bias=False),
            nn.GroupNorm(32, satellite_channels),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(2 * satellite_channels + 2, 64, 1), nn.SiLU(), nn.Conv2d(64, 1, 1)
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, satellite, uav, confidence, coverage):
        residual = self.project(uav)
        gate = self.gate(
            torch.cat(
                (
                    satellite,
                    residual,
                    confidence.to(satellite.dtype),
                    coverage.to(satellite.dtype),
                ),
                dim=1,
            )
        ).sigmoid()
        valid = coverage > 0
        applied = self.alpha * valid * confidence * gate * residual
        supported = valid.detach()
        supported_count = supported.sum().clamp_min(1)
        stats = dict(
            alpha=self.alpha.detach(),
            gate=gate.detach().mean(),
            supported_fraction=supported.float().mean(),
            gate_supported=(gate.detach().float() * supported).sum() / supported_count,
            confidence_supported=(confidence.detach().float() * supported).sum()
            / supported_count,
            effective_gain_supported=(
                self.alpha.detach() * confidence.detach() * gate.detach() * supported
            ).sum()
            / supported_count,
            coverage=coverage.detach().mean(),
            confidence=confidence.detach().mean(),
            residual_ratio=applied.detach().float().norm()
            / satellite.detach().float().norm().clamp_min(1e-6),
        )
        return satellite + applied.to(satellite.dtype), stats
