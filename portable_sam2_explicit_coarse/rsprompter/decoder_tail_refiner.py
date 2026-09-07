"""Uncertainty-guided decoder-tail point refinement (UDPR)."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DecoderTailPointRefiner(nn.Module):
    """Local residual classifier over SAM2's least-confident native logits."""

    def __init__(self, *, num_points: int = 64, feature_channels: int = 32,
                 token_dim: int = 256, hidden_dim: int = 128,
                 point_loss_weight: float = 1.0, delta_logit_max: float = 2.0) -> None:
        super().__init__()
        if num_points < 0 or hidden_dim <= 0 or point_loss_weight < 0 or delta_logit_max <= 0:
            raise ValueError("UDPR requires non-negative K/loss and positive hidden/delta sizes")
        self.num_points, self.point_loss_weight = int(num_points), float(point_loss_weight)
        self.delta_logit_max = float(delta_logit_max)
        input_dim = int(feature_channels) + int(token_dim) + 4  # z, x, y, |z|
        self.mlp = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, int(hidden_dim)), nn.GELU(), nn.Linear(int(hidden_dim), 1))
        # Exact identity at initialization; unlike a zero gate + zero head,
        # this preserves a gradient for the final residual layer.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def _select(self, logits: Tensor) -> Tuple[Tensor, Tensor]:
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError(f"UDPR expects [N,1,H,W] logits, got {tuple(logits.shape)}")
        n, _, h, w = logits.shape
        flat = logits[:, 0].detach().float().abs().reshape(n, h * w)
        # Exact ties use native raster order, not implementation-dependent TopK order.
        indices = torch.argsort(flat, dim=1, stable=True)[:, :min(self.num_points, h * w)]
        return indices, flat.gather(1, indices)

    def forward(self, logits: Tensor, upscaled: Tensor, mask_tokens: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        if upscaled.ndim != 4 or upscaled.shape[0] != logits.shape[0] or tuple(upscaled.shape[-2:]) != tuple(logits.shape[-2:]):
            raise ValueError("UDPR upscaled embedding must align with the native logit grid")
        if mask_tokens is None or mask_tokens.ndim != 3 or mask_tokens.shape[0] != logits.shape[0]:
            raise ValueError("UDPR requires one selected SAM2 mask token per ROI")
        n, _, h, w = logits.shape
        indices, selected_abs = self._select(logits)
        k = indices.shape[1]
        if k == 0:
            return logits, {"indices": indices, "delta": logits.new_zeros((n, 0)), "selected_abs": selected_abs,
                            "refined_selected_logits": logits.new_zeros((n, 0)),
                            "base_selected_logits": logits.new_zeros((n, 0)),
                            "base_logits_detached": logits[:, 0].detach()}
        feat = upscaled.flatten(2).transpose(1, 2).gather(1, indices[..., None].expand(-1, -1, upscaled.shape[1]))
        token = mask_tokens[:, 0, :].unsqueeze(1).expand(-1, k, -1)
        flat_logits = logits[:, 0].reshape(n, -1)
        selected_logits = flat_logits.gather(1, indices)
        yy = (indices // w).to(dtype=logits.dtype) / max(1, h - 1)
        xx = (indices % w).to(dtype=logits.dtype) / max(1, w - 1)
        attrs = torch.stack((selected_logits, xx, yy, selected_logits.abs()), dim=-1)
        delta = self.delta_logit_max * torch.tanh(self.mlp(torch.cat((feat, token, attrs), dim=-1)).squeeze(-1))
        refined = flat_logits.scatter(1, indices, selected_logits + delta).reshape(n, 1, h, w)
        return refined, {"indices": indices, "delta": delta, "selected_abs": selected_abs,
                         "refined_selected_logits": selected_logits + delta,
                         "base_selected_logits": selected_logits.detach(),
                         # Training-only telemetry uses this detached copy to
                         # quantify how much of the original A0 error set the
                         # selector covers. It never controls selection/loss.
                         "base_logits_detached": logits[:, 0].detach()}

    @staticmethod
    def _global_tail_bce(
        pos_sum: Tensor, neg_sum: Tensor, pos_count: Tensor, neg_count: Tensor
    ) -> Tuple[Tensor, Tensor, float]:
        """Return detached global sums/counts for telemetry and loss scaling.

        This helper is also called by the empty-positive-RoI path.  Therefore
        every DDP rank executes the same two collectives once per mask-loss
        invocation, even when only a subset of ranks has trainable RoIs.
        """
        counts = torch.stack((pos_count, neg_count)).to(dtype=pos_sum.dtype)
        sums = torch.stack((pos_sum.detach(), neg_sum.detach()))
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            world = float(dist.get_world_size())
        else:
            world = 1.0
        return counts, sums, world

    def empty_point_loss(self, zero: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """DDP-collective-compatible zero loss for a rank with no positive RoIs."""
        global_counts, global_sums, _ = self._global_tail_bce(zero, zero, zero, zero)
        active = (global_counts > 0).to(dtype=zero.dtype)
        global_bce = (
            global_sums[0] / global_counts[0].clamp_min(1.0)
            + global_sums[1] / global_counts[1].clamp_min(1.0)
        ) / active.sum().clamp_min(1.0)
        return zero, {
            "TAIL/selected_count": zero,
            "TAIL/selected_pos": zero,
            "TAIL/selected_bce": global_bce,
            "TAIL/delta_abs": zero,
            "TAIL/delta_max": zero,
            "TAIL/selected_error_fraction": zero,
            "TAIL/selected_error_coverage": zero,
            "TAIL/changed_fraction": zero,
            "TAIL/error_direction_agreement": zero,
            "TAIL/correct_flip_fraction": zero,
            "TAIL/destroy_fraction": zero,
        }

    def point_loss(self, tail_outputs: Dict[str, Tensor], targets: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """DDP-correct, positive/negative-balanced BCE over selected points."""
        indices, delta = tail_outputs["indices"], tail_outputs["delta"]
        if indices.numel() == 0:
            zero = targets.sum() * 0.0
            return self.empty_point_loss(zero)
        selected_logits = tail_outputs["refined_selected_logits"]
        selected_targets = targets.reshape(targets.shape[0], -1).gather(1, indices).to(selected_logits.dtype)
        # Do not reconstruct base logits as refined-delta: selector points are
        # deliberately near zero, where AMP round-off can alter their sign.
        base_selected_logits = tail_outputs["base_selected_logits"]
        target_binary = selected_targets > 0.5
        base_binary = base_selected_logits >= 0.0
        refined_binary = selected_logits.detach() >= 0.0
        selected_error = base_binary != target_binary
        all_base_binary = tail_outputs["base_logits_detached"] >= 0.0
        all_error_count = (all_base_binary != (targets > 0.5)).sum()
        changed = delta.detach().abs() > 1e-6
        correct_direction = (delta.detach() > 0.0) == target_binary
        correct_flip = selected_error & (refined_binary == target_binary)
        destroy = (~selected_error) & (refined_binary != target_binary)
        bce = F.binary_cross_entropy_with_logits(selected_logits, selected_targets, reduction="none")
        pos = selected_targets > 0.5
        neg = ~pos
        pos_sum, neg_sum = (bce * pos).sum(), (bce * neg).sum()
        global_counts, global_sums, world = self._global_tail_bce(
            pos_sum, neg_sum, pos.sum(), neg.sum()
        )
        pos_term = pos_sum * (world / global_counts[0].clamp_min(1.0)) if global_counts[0] > 0 else pos_sum * 0.0
        neg_term = neg_sum * (world / global_counts[1].clamp_min(1.0)) if global_counts[1] > 0 else neg_sum * 0.0
        loss = (pos_term + neg_term) / (global_counts > 0).sum().to(selected_logits.dtype).clamp_min(1.0)
        # This is telemetry only.  Unlike the DDP-scaled local loss above,
        # reduce detached numerators so rank-0's logged BCE is globally
        # interpretable.
        active = (global_counts > 0).to(selected_logits.dtype)
        global_bce = (
            global_sums[0] / global_counts[0].clamp_min(1.0)
            + global_sums[1] / global_counts[1].clamp_min(1.0)
        ) / active.sum().clamp_min(1.0)
        return self.point_loss_weight * loss, {
            "TAIL/selected_count": selected_logits.new_tensor(float(indices.numel())),
            "TAIL/selected_pos": pos.sum().to(dtype=selected_logits.dtype),
            "TAIL/selected_bce": global_bce, "TAIL/delta_abs": delta.detach().abs().mean(),
            "TAIL/delta_max": delta.detach().abs().max(),
            # All fields below are detached mechanism telemetry.  Their
            # denominators are named explicitly in DEBUG_FIELDS.md.
            "TAIL/selected_error_fraction": selected_error.float().mean(),
            "TAIL/selected_error_coverage": selected_error.sum().to(dtype=selected_logits.dtype) /
            all_error_count.to(dtype=selected_logits.dtype).clamp_min(1.0),
            "TAIL/changed_fraction": changed.float().mean(),
            "TAIL/error_direction_agreement": (selected_error & changed & correct_direction).sum().to(dtype=selected_logits.dtype) /
            selected_error.sum().to(dtype=selected_logits.dtype).clamp_min(1.0),
            "TAIL/correct_flip_fraction": correct_flip.float().mean(),
            "TAIL/destroy_fraction": destroy.float().mean(),
        }
