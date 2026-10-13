"""UDPR decoder tail: margin-anchored residual write on uncertain logits.

The tail selects the K decoder-logit locations closest to the binarisation
boundary, predicts a bounded residual there, and writes it back.  Its loss is
a one-sided margin anchored at the DEPLOYED binarisation logit
(log(mask_thr/(1-mask_thr)), e.g. -0.4055 at thr 0.4) — the tail-write oracle
showed that writes anchored at logit 0 cannot delete false-positive area under
a 0.4 threshold.

This is the coupled "margin" container: point features, the mask token, and
the selected logits are NOT detached, so both the margin loss and the final
mask loss (which consumes the refined logits) reach the decoder trunk — the
co-adaptation channel that separates a3sgm from the isolated-tail variants.
"""
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
                 point_loss_weight: float = 1.0, delta_logit_max: float = 2.0,
                 mode: str = "residual_v1_margin", gate_init_prob: float = 0.1,
                 gate_loss_weight: float = 1.0, keep_loss_weight: float = 0.05,
                 crossing_margin: float = 0.10, boundary_logit: float = 0.0) -> None:
        super().__init__()
        if num_points < 0 or hidden_dim <= 0 or point_loss_weight < 0 or delta_logit_max <= 0:
            raise ValueError("UDPR requires non-negative K/loss and positive hidden/delta sizes")
        if mode != "residual_v1_margin":
            raise ValueError(
                f"Unsupported UDPR mode={mode!r}; the BCE (v1), tailstop, and "
                "confidence-gated containers were removed with their arms — "
                "only the coupled margin container survives"
            )
        if crossing_margin < 0:
            raise ValueError("UDPR crossing_margin must be non-negative")
        # gate_* params stay accepted because historical config dicts carry
        # them; the confidence-gated container that consumed them is gone.
        self.num_points, self.point_loss_weight = int(num_points), float(point_loss_weight)
        self.delta_logit_max = float(delta_logit_max)
        self.mode = str(mode)
        self.gate_loss_weight = float(gate_loss_weight)
        self.keep_loss_weight = float(keep_loss_weight)
        self.crossing_margin = float(crossing_margin)
        self.boundary_logit = float(boundary_logit)
        input_dim = int(feature_channels) + int(token_dim) + 4  # z, x, y, |z|
        # Keep the v1 module and parameter names byte-for-byte compatible
        # with existing A3 checkpoints (mlp.{0,1,3} in the state dict).
        self.mlp = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, int(hidden_dim)), nn.GELU(), nn.Linear(int(hidden_dim), 1))
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
            return logits, {
                "indices": indices, "delta": logits.new_zeros((n, 0)), "selected_abs": selected_abs,
                            "refined_selected_logits": logits.new_zeros((n, 0)),
                            "base_selected_logits": logits.new_zeros((n, 0)),
                            "base_logits_detached": logits[:, 0].detach()}
        feat = upscaled.flatten(2).transpose(1, 2).gather(1, indices[..., None].expand(-1, -1, upscaled.shape[1]))
        token = mask_tokens[:, 0, :].unsqueeze(1).expand(-1, k, -1)
        flat_logits = logits[:, 0].reshape(n, -1)
        selected_logits = flat_logits.gather(1, indices)
        yy = (indices // w).to(dtype=logits.dtype) / max(1, h - 1)
        xx = (indices % w).to(dtype=logits.dtype) / max(1, w - 1)
        # Coupled container: nothing is detached — the margin loss and the
        # final-mask loss both reach the decoder through this module.
        attrs = torch.stack((selected_logits, xx, yy, selected_logits.abs()), dim=-1)
        point_features = torch.cat((feat, token, attrs), dim=-1)
        delta = self.delta_logit_max * torch.tanh(self.mlp(point_features).squeeze(-1))
        refined = flat_logits.scatter(1, indices, selected_logits + delta).reshape(n, 1, h, w)
        return refined, {"indices": indices, "delta": delta,
                         "selected_abs": selected_abs,
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
        stats = {
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
        # Margin modes take TWO _global_tail_bce collectives on non-empty
        # ranks (violation terms, then the detached BCE diagnostic), and
        # report selected_bce from call-2 sums over call-1 counts while
        # boundary_margin_violation is call-1's normalized sum.  Mirror
        # both the collective count and that field semantics here so an
        # empty rank neither desyncs the group nor misreports the fields
        # (audited 2026-09-16; desync would be a loud crash).
        _bce_counts, bce_sums, _ = self._global_tail_bce(zero, zero, zero, zero)
        _active = (global_counts > 0).to(zero.dtype)
        stats["TAIL/selected_bce"] = (
            bce_sums[0] / global_counts[0].clamp_min(1.0)
            + bce_sums[1] / global_counts[1].clamp_min(1.0)
        ) / _active.sum().clamp_min(1.0)
        stats["TAIL/boundary_margin_violation"] = global_bce
        return zero, stats

    def point_loss(self, tail_outputs: Dict[str, Tensor], targets: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Threshold-anchored one-sided margin over the selected points.

        Loss is zero for a point already beyond boundary +/- margin on the
        correct side and pushes wrong-side points toward crossing the DEPLOYED
        binarisation logit.  Positive/negative points are DDP-globally
        normalized then equally weighted.
        """
        indices, delta = tail_outputs["indices"], tail_outputs["delta"]
        if indices.numel() == 0:
            return self.empty_point_loss(targets.sum() * 0.0)
        selected_logits = tail_outputs["refined_selected_logits"]
        selected_targets = targets.reshape(targets.shape[0], -1).gather(1, indices).to(selected_logits.dtype)
        signed = selected_targets.mul(2.0).sub(1.0)
        violation = F.relu(self.crossing_margin - signed * (selected_logits - self.boundary_logit))
        pos = selected_targets > 0.5
        neg = ~pos
        pos_sum, neg_sum = (violation * pos).sum(), (violation * neg).sum()
        global_counts, global_sums, world = self._global_tail_bce(
            pos_sum, neg_sum, pos.sum(), neg.sum()
        )
        pos_term = pos_sum * (world / global_counts[0].clamp_min(1.0)) if global_counts[0] > 0 else pos_sum * 0.0
        neg_term = neg_sum * (world / global_counts[1].clamp_min(1.0)) if global_counts[1] > 0 else neg_sum * 0.0
        loss = (pos_term + neg_term) / (global_counts > 0).sum().to(selected_logits.dtype).clamp_min(1.0)
        active = (global_counts > 0).to(selected_logits.dtype)
        violation_global = (
            global_sums[0] / global_counts[0].clamp_min(1.0)
            + global_sums[1] / global_counts[1].clamp_min(1.0)
        ) / active.sum().clamp_min(1.0)
        # Mechanism telemetry mirrors the v1 contract (detached), plus the
        # boundary-anchored violation; BCE is retained as a comparability
        # diagnostic against the a3sgt era's logged scale.
        base_selected_logits = tail_outputs["base_selected_logits"]
        bce = F.binary_cross_entropy_with_logits(selected_logits.detach(), selected_targets, reduction="none")
        pos_bce, neg_bce = (bce * pos).sum(), (bce * neg).sum()
        _, bce_sums, _ = self._global_tail_bce(pos_bce, neg_bce, pos.sum() * 0.0, neg.sum() * 0.0)
        bce_global = (
            bce_sums[0] / global_counts[0].clamp_min(1.0)
            + bce_sums[1] / global_counts[1].clamp_min(1.0)
        ) / active.sum().clamp_min(1.0)
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
        return self.point_loss_weight * loss, {
            "TAIL/selected_count": selected_logits.new_tensor(float(indices.numel())),
            "TAIL/selected_pos": pos.sum().to(dtype=selected_logits.dtype),
            "TAIL/selected_bce": bce_global, "TAIL/delta_abs": delta.detach().abs().mean(),
            "TAIL/delta_max": delta.detach().abs().max(),
            "TAIL/selected_error_fraction": selected_error.float().mean(),
            "TAIL/selected_error_coverage": selected_error.sum().to(dtype=selected_logits.dtype) /
            all_error_count.to(dtype=selected_logits.dtype).clamp_min(1.0),
            "TAIL/changed_fraction": changed.float().mean(),
            "TAIL/error_direction_agreement": (selected_error & changed & correct_direction).sum().to(dtype=selected_logits.dtype) /
            selected_error.sum().to(dtype=selected_logits.dtype).clamp_min(1.0),
            "TAIL/correct_flip_fraction": correct_flip.float().mean(),
            "TAIL/destroy_fraction": destroy.float().mean(),
            "TAIL/boundary_margin_violation": violation_global,
            "TAIL/boundary_logit": selected_logits.new_tensor(self.boundary_logit),
        }
