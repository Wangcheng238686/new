"""Uncertainty-guided decoder-tail point refinement (UDPR)."""
from __future__ import annotations

import math
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
                 mode: str = "residual_v1", gate_init_prob: float = 0.1,
                 gate_loss_weight: float = 1.0, keep_loss_weight: float = 0.05) -> None:
        super().__init__()
        if num_points < 0 or hidden_dim <= 0 or point_loss_weight < 0 or delta_logit_max <= 0:
            raise ValueError("UDPR requires non-negative K/loss and positive hidden/delta sizes")
        if mode not in {"residual_v1", "confidence_gated"}:
            raise ValueError(f"Unsupported UDPR mode={mode!r}")
        if not 0.0 < gate_init_prob < 1.0 or gate_loss_weight < 0 or keep_loss_weight < 0:
            raise ValueError("UDPR confidence-gate probabilities/weights are invalid")
        self.num_points, self.point_loss_weight = int(num_points), float(point_loss_weight)
        self.delta_logit_max = float(delta_logit_max)
        self.mode = str(mode)
        self.gate_loss_weight = float(gate_loss_weight)
        self.keep_loss_weight = float(keep_loss_weight)
        input_dim = int(feature_channels) + int(token_dim) + 4  # z, x, y, |z|
        if self.mode == "residual_v1":
            # Keep the v1 module and parameter names byte-for-byte compatible
            # with existing A3 checkpoints.
            self.mlp = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, int(hidden_dim)), nn.GELU(), nn.Linear(int(hidden_dim), 1))
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)
        else:
            # DCR: uncertainty finds candidates; a separate correction-
            # confidence head authorizes the residual write strength.
            self.shared_mlp = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, int(hidden_dim)), nn.GELU())
            self.delta_head = nn.Linear(int(hidden_dim), 1)
            self.gate_head = nn.Linear(int(hidden_dim), 1)
            nn.init.zeros_(self.delta_head.weight)
            nn.init.zeros_(self.delta_head.bias)
            nn.init.zeros_(self.gate_head.weight)
            nn.init.constant_(self.gate_head.bias, math.log(gate_init_prob / (1.0 - gate_init_prob)))

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
        point_features = torch.cat((feat, token, attrs), dim=-1)
        if self.mode == "residual_v1":
            raw_delta = self.delta_logit_max * torch.tanh(self.mlp(point_features).squeeze(-1))
            gate = torch.ones_like(raw_delta)
            gate_logits = None
            delta = raw_delta
        else:
            hidden = self.shared_mlp(point_features)
            raw_delta = self.delta_logit_max * torch.tanh(self.delta_head(hidden).squeeze(-1))
            gate_logits = self.gate_head(hidden).squeeze(-1)
            gate = torch.sigmoid(gate_logits)
            delta = gate * raw_delta
        refined = flat_logits.scatter(1, indices, selected_logits + delta).reshape(n, 1, h, w)
        return refined, {"indices": indices, "delta": delta, "raw_delta": raw_delta, "gate": gate,
                         "gate_logits": gate_logits,
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

    @staticmethod
    def _global_gate_terms(
        error_sum: Tensor, correct_sum: Tensor, keep_sum: Tensor,
        error_count: Tensor, correct_count: Tensor,
    ) -> Tuple[Tensor, Tensor, float]:
        """DDP-normalized gate/keep loss terms for confidence-gated DCR."""
        counts = torch.stack((error_count, correct_count)).to(dtype=error_sum.dtype)
        sums = torch.stack((error_sum.detach(), correct_sum.detach(), keep_sum.detach()))
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            world = float(dist.get_world_size())
        else:
            world = 1.0
        return counts, sums, world

    @staticmethod
    def _global_cg_telemetry(values: Tensor) -> Tensor:
        """One global all-reduce for all A4 mechanism numerators/counts."""
        values = values.detach().clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
        return values

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
        if self.mode == "confidence_gated":
            gate_counts, gate_sums, _ = self._global_gate_terms(zero, zero, zero, zero, zero)
            # An empty local rank must expose the same DDP-global mechanism
            # telemetry as a rank holding positive RoIs; otherwise training
            # logs depend on which rank happened to be selected for output.
            telemetry = self._global_cg_telemetry(zero.new_zeros(16))
            total, pos_count, error_count, correct_count, all_error = telemetry[:5]
            changed_count, direction_count, flip_count, destroy_count, open_count = telemetry[5:10]
            gate_sum, gate_error_sum, gate_correct_sum = telemetry[10:13]
            applied_sum, applied_error_sum, applied_correct_sum = telemetry[13:16]
            total_safe = total.clamp_min(1.0)
            error_safe = error_count.clamp_min(1.0)
            correct_safe = correct_count.clamp_min(1.0)
            active_gate = (gate_counts > 0).to(dtype=zero.dtype)
            stats.update({
                "TAIL/selected_count": total,
                "TAIL/selected_pos": pos_count,
                "TAIL/delta_abs": applied_sum / total_safe,
                "TAIL/selected_error_fraction": error_count / total_safe,
                "TAIL/selected_error_coverage": error_count / all_error.clamp_min(1.0),
                "TAIL/changed_fraction": changed_count / total_safe,
                "TAIL/error_direction_agreement": direction_count / error_safe,
                "TAIL/correct_flip_fraction": flip_count / total_safe,
                "TAIL/destroy_fraction": destroy_count / total_safe,
                "TAIL/gate_mean": gate_sum / total_safe,
                "TAIL/gate_on_error": gate_error_sum / error_safe,
                "TAIL/gate_on_correct": gate_correct_sum / correct_safe,
                "TAIL/gate_bce": (
                    gate_sums[0] / gate_counts[0].clamp_min(1.0)
                    + gate_sums[1] / gate_counts[1].clamp_min(1.0)
                ) / active_gate.sum().clamp_min(1.0),
                "TAIL/applied_delta_abs": applied_sum / total_safe,
                "TAIL/applied_delta_error_abs": applied_error_sum / error_safe,
                "TAIL/applied_delta_correct_abs": applied_correct_sum / correct_safe,
                "TAIL/keep_loss": gate_sums[2] / gate_counts[1].clamp_min(1.0),
                "TAIL/net_flip_fraction": (flip_count - destroy_count) / total_safe,
                "TAIL/gate_open_fraction_q50": open_count / total_safe,
            })
        return zero, stats

    def point_loss(self, tail_outputs: Dict[str, Tensor], targets: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """DDP-correct, positive/negative-balanced BCE over selected points."""
        if self.mode == "confidence_gated":
            return self._confidence_gated_point_loss(tail_outputs, targets)
        return self._v1_point_loss(tail_outputs, targets)

    def _v1_point_loss(self, tail_outputs: Dict[str, Tensor], targets: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Original A3 loss path, retained exactly for checkpoint compatibility."""
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

    def _confidence_gated_point_loss(
        self, tail_outputs: Dict[str, Tensor], targets: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """DCR loss: refined BCE plus correction-confidence and keep terms."""
        indices = tail_outputs["indices"]
        if indices.numel() == 0:
            return self.empty_point_loss(targets.sum() * 0.0)
        selected_logits = tail_outputs["refined_selected_logits"]
        selected_targets = targets.reshape(targets.shape[0], -1).gather(1, indices).to(selected_logits.dtype)
        base_selected_logits = tail_outputs["base_selected_logits"]
        gate, gate_logits, applied_delta = (
            tail_outputs["gate"], tail_outputs["gate_logits"], tail_outputs["delta"]
        )
        target_binary = selected_targets > 0.5
        base_binary = base_selected_logits >= 0.0
        refined_binary = selected_logits.detach() >= 0.0
        selected_error = base_binary != target_binary
        selected_correct = ~selected_error
        all_base_binary = tail_outputs["base_logits_detached"] >= 0.0
        all_error_count = (all_base_binary != (targets > 0.5)).sum()
        changed = applied_delta.detach().abs() > 1e-6
        correct_direction = (applied_delta.detach() > 0.0) == target_binary
        correct_flip = selected_error & (refined_binary == target_binary)
        destroy = selected_correct & (refined_binary != target_binary)

        # Retain the v1 point-target objective, including DDP positive/negative
        # normalization, so A4 isolates correction authorization rather than
        # changing the class balance of the underlying refinement task.
        refined_bce = F.binary_cross_entropy_with_logits(selected_logits, selected_targets, reduction="none")
        pos = selected_targets > 0.5
        neg = ~pos
        pos_sum, neg_sum = (refined_bce * pos).sum(), (refined_bce * neg).sum()
        ref_counts, ref_sums, world = self._global_tail_bce(pos_sum, neg_sum, pos.sum(), neg.sum())
        pos_term = pos_sum * (world / ref_counts[0].clamp_min(1.0)) if ref_counts[0] > 0 else pos_sum * 0.0
        neg_term = neg_sum * (world / ref_counts[1].clamp_min(1.0)) if ref_counts[1] > 0 else neg_sum * 0.0
        ref_loss = (pos_term + neg_term) / (ref_counts > 0).sum().to(selected_logits.dtype).clamp_min(1.0)
        active_ref = (ref_counts > 0).to(selected_logits.dtype)
        ref_bce_global = (
            ref_sums[0] / ref_counts[0].clamp_min(1.0)
            + ref_sums[1] / ref_counts[1].clamp_min(1.0)
        ) / active_ref.sum().clamp_min(1.0)

        # The error target is formed only during training from detached
        # pre-tail logits.  It is never stored in, or read by, inference.
        error_target = selected_error.to(dtype=selected_logits.dtype)
        # BCEWithLogits is AMP-safe; sigmoid(gate_logits) remains only the
        # forward/telemetry value used to scale the residual.
        gate_bce = F.binary_cross_entropy_with_logits(gate_logits, error_target, reduction="none")
        error_gate_sum = (gate_bce * selected_error).sum()
        correct_gate_sum = (gate_bce * selected_correct).sum()
        keep_sum = (applied_delta.abs() * selected_correct).sum()
        gate_counts, gate_sums, gate_world = self._global_gate_terms(
            error_gate_sum, correct_gate_sum, keep_sum,
            selected_error.sum(), selected_correct.sum(),
        )
        error_gate_term = error_gate_sum * (gate_world / gate_counts[0].clamp_min(1.0)) if gate_counts[0] > 0 else error_gate_sum * 0.0
        correct_gate_term = correct_gate_sum * (gate_world / gate_counts[1].clamp_min(1.0)) if gate_counts[1] > 0 else correct_gate_sum * 0.0
        gate_loss = (error_gate_term + correct_gate_term) / (gate_counts > 0).sum().to(selected_logits.dtype).clamp_min(1.0)
        keep_loss = keep_sum * (gate_world / gate_counts[1].clamp_min(1.0)) if gate_counts[1] > 0 else keep_sum * 0.0
        active_gate = (gate_counts > 0).to(selected_logits.dtype)
        gate_bce_global = (
            gate_sums[0] / gate_counts[0].clamp_min(1.0)
            + gate_sums[1] / gate_counts[1].clamp_min(1.0)
        ) / active_gate.sum().clamp_min(1.0)
        keep_global = gate_sums[2] / gate_counts[1].clamp_min(1.0)

        telemetry = self._global_cg_telemetry(torch.stack((
            selected_logits.new_tensor(float(indices.numel())),
            pos.sum().to(dtype=selected_logits.dtype),
            selected_error.sum().to(dtype=selected_logits.dtype),
            selected_correct.sum().to(dtype=selected_logits.dtype),
            all_error_count.to(dtype=selected_logits.dtype),
            changed.sum().to(dtype=selected_logits.dtype),
            (selected_error & changed & correct_direction).sum().to(dtype=selected_logits.dtype),
            correct_flip.sum().to(dtype=selected_logits.dtype),
            destroy.sum().to(dtype=selected_logits.dtype),
            (gate.detach() >= 0.5).sum().to(dtype=selected_logits.dtype),
            gate.detach().sum(),
            (gate.detach() * selected_error).sum(),
            (gate.detach() * selected_correct).sum(),
            applied_delta.detach().abs().sum(),
            (applied_delta.detach().abs() * selected_error).sum(),
            (applied_delta.detach().abs() * selected_correct).sum(),
        )))
        total, pos_count, error_count, correct_count, all_error = telemetry[:5]
        changed_count, direction_count, flip_count, destroy_count, open_count = telemetry[5:10]
        gate_sum, gate_error_sum, gate_correct_sum = telemetry[10:13]
        applied_sum, applied_error_sum, applied_correct_sum = telemetry[13:16]
        total_safe = total.clamp_min(1.0)
        error_safe, correct_safe = error_count.clamp_min(1.0), correct_count.clamp_min(1.0)
        stats = {
            "TAIL/selected_count": total,
            "TAIL/selected_pos": pos_count,
            "TAIL/selected_bce": ref_bce_global,
            "TAIL/delta_abs": applied_sum / total_safe,
            "TAIL/delta_max": applied_delta.detach().abs().max(),
            "TAIL/selected_error_fraction": error_count / total_safe,
            "TAIL/selected_error_coverage": error_count / all_error.clamp_min(1.0),
            "TAIL/changed_fraction": changed_count / total_safe,
            "TAIL/error_direction_agreement": direction_count / error_safe,
            "TAIL/correct_flip_fraction": flip_count / total_safe,
            "TAIL/destroy_fraction": destroy_count / total_safe,
            "TAIL/gate_mean": gate_sum / total_safe,
            "TAIL/gate_on_error": gate_error_sum / error_safe,
            "TAIL/gate_on_correct": gate_correct_sum / correct_safe,
            "TAIL/gate_bce": gate_bce_global,
            "TAIL/applied_delta_abs": applied_sum / total_safe,
            "TAIL/applied_delta_error_abs": applied_error_sum / error_safe,
            "TAIL/applied_delta_correct_abs": applied_correct_sum / correct_safe,
            "TAIL/keep_loss": keep_global,
            "TAIL/net_flip_fraction": (flip_count - destroy_count) / total_safe,
            "TAIL/gate_open_fraction_q50": open_count / total_safe,
        }
        total_loss = (
            self.point_loss_weight * ref_loss
            + self.gate_loss_weight * gate_loss
            + self.keep_loss_weight * keep_loss
        )
        return total_loss, stats
