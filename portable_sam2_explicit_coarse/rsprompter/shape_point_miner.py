"""Adaptive hard 2P2N point mining from coarse logits (stop-gradient).

Split out of the original shape_prior.py (facade kept for import compat)."""

"""Explicit ROI-local coarse-mask prompt generation and point mining.

Canonical shape-point route:
- ROI feature -> lightweight coarse-mask decoder
- optional global SAM2 context + box-conditioned Spatial FiLM ablation
- ROI-local coarse mask
- adaptive hard 2P2N point mining (stop-gradient)

Only PyTorch is required.
"""
import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class ShapePointMiner(nn.Module):
    """Adaptive hard 2P2N mining from detached ROI-local coarse masks.

    Slot order is fixed as ``P1, P2, N1, N2``.  Labels are SAM-compatible:
    ``1`` positive, ``0`` negative, ``-1`` invalid/padding.
    """

    def __init__(
        self,
        foreground_thr: float = 0.50,
        max_distance_steps: Optional[int] = None,
        min_box_size: float = 2.0,
        max_positive_points: int = 2,
        max_negative_points: int = 2,
        adaptive_validity: bool = True,
        positive_conf_thr: float = 0.60,
        empty_positive_conf_thr: float = 0.35,
        negative_conf_thr: float = 0.30,
        min_inside_distance_ratio: float = 0.05,
        min_positive_distance_ratio: float = 0.15,
        min_negative_distance_ratio: float = 0.15,
        outer_ring_radius: int = 1,
        safe_background_radius_ratio: float = 0.10,
        point_jitter_prob: float = 0.0,
        point_jitter_radius: int = 1,
        point_warmup_cfg: Optional[Dict] = None,
    ):
        super().__init__()
        if int(max_positive_points) != 2 or int(max_negative_points) != 2:
            raise ValueError("Canonical ShapePointMiner requires max 2P2N")
        self.foreground_thr = float(foreground_thr)
        self.max_distance_steps = max_distance_steps
        self.min_box_size = float(min_box_size)
        self.adaptive_validity = bool(adaptive_validity)
        self.positive_conf_thr = float(positive_conf_thr)
        self.empty_positive_conf_thr = float(empty_positive_conf_thr)
        self.negative_conf_thr = float(negative_conf_thr)
        self.min_inside_distance_ratio = float(min_inside_distance_ratio)
        self.min_positive_distance_ratio = float(min_positive_distance_ratio)
        self.min_negative_distance_ratio = float(min_negative_distance_ratio)
        self.outer_ring_radius = max(1, int(outer_ring_radius))
        self.safe_background_radius_ratio = float(safe_background_radius_ratio)
        self.point_jitter_prob = float(point_jitter_prob)
        self.point_jitter_radius = max(1, int(point_jitter_radius))
        self.point_warmup_cfg = dict(point_warmup_cfg or {})
        self.point_warmup_cfg.setdefault("enabled", True)
        self.point_warmup_cfg.setdefault("no_point_epochs", 5)
        self.point_warmup_cfg.setdefault("one_pair_epochs", 5)
        _expected_full_start = (
            int(self.point_warmup_cfg["no_point_epochs"])
            + int(self.point_warmup_cfg["one_pair_epochs"])
            + 1
        )
        self.point_warmup_cfg.setdefault(
            "full_2p2n_start_epoch", _expected_full_start
        )
        if int(self.point_warmup_cfg["no_point_epochs"]) < 0:
            raise ValueError("no_point_epochs must be non-negative")
        if int(self.point_warmup_cfg["one_pair_epochs"]) < 0:
            raise ValueError("one_pair_epochs must be non-negative")
        if int(self.point_warmup_cfg["full_2p2n_start_epoch"]) != _expected_full_start:
            raise ValueError(
                "full_2p2n_start_epoch must equal "
                "no_point_epochs + one_pair_epochs + 1; got "
                f"{self.point_warmup_cfg['full_2p2n_start_epoch']} vs "
                f"{_expected_full_start}"
            )
        # Default to the full stage for standalone unit/inference calls.  The
        # trainer explicitly sets a 1-based epoch before every training epoch.
        self.current_epoch = int(
            self.point_warmup_cfg.get("full_2p2n_start_epoch", 11)
        )
        self._last_local_yx: Optional[Tensor] = None
        self._last_labels: Optional[Tensor] = None

    def set_current_epoch(self, epoch_number: int) -> None:
        self.current_epoch = max(1, int(epoch_number))

    def _warmup_stage(self) -> int:
        if not bool(self.point_warmup_cfg.get("enabled", True)):
            return 2
        no_point_end = int(self.point_warmup_cfg.get("no_point_epochs", 5))
        full_start = int(self.point_warmup_cfg.get("full_2p2n_start_epoch", 11))
        if self.current_epoch <= no_point_end:
            return 0
        if self.current_epoch < full_start:
            return 1
        return 2

    @staticmethod
    def erode(binary_mask: Tensor) -> Tensor:
        padded = F.pad(binary_mask, (1, 1, 1, 1), mode="constant", value=0.0)
        return 1.0 - F.max_pool2d(
            1.0 - padded, kernel_size=3, stride=1, padding=0
        )

    @staticmethod
    def dilate(binary_mask: Tensor) -> Tensor:
        padded = F.pad(binary_mask, (1, 1, 1, 1), mode="constant", value=0.0)
        return F.max_pool2d(padded, kernel_size=3, stride=1, padding=0)

    def _dilate_steps(self, binary_mask: Tensor, steps: int) -> Tensor:
        out = binary_mask
        for _ in range(max(0, int(steps))):
            out = self.dilate(out)
        return out

    def _inside_distance(self, foreground: Tensor) -> Tensor:
        h, w = foreground.shape[-2:]
        required_steps = (max(h, w) + 1) // 2
        num_steps = (
            required_steps
            if self.max_distance_steps is None
            else min(required_steps, int(self.max_distance_steps))
        )
        remaining = foreground.float()
        inside_distance = torch.zeros_like(remaining)
        for _ in range(max(1, num_steps)):
            inside_distance = inside_distance + remaining
            remaining = self.erode(remaining)
        return inside_distance

    @staticmethod
    def _grid(h: int, w: int, device: torch.device, dtype: torch.dtype):
        yy, xx = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype),
            torch.arange(w, device=device, dtype=dtype),
            indexing="ij",
        )
        return yy.reshape(1, -1), xx.reshape(1, -1)

    def _pick_platform_center(
        self, candidate: Tensor, score: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Pick the geometric center of each maximum-score plateau."""
        n, _, h, w = candidate.shape
        candidate_flat = candidate.flatten(1).bool()
        score_flat = score.flatten(1)
        valid = candidate_flat.any(dim=1)
        masked = torch.where(
            candidate_flat,
            score_flat,
            torch.full_like(score_flat, -torch.inf),
        )
        best = masked.max(dim=1, keepdim=True).values
        plateau = candidate_flat & (masked >= best - 1e-6)
        yy, xx = self._grid(h, w, score.device, score.dtype)
        weights = plateau.to(score.dtype)
        count = weights.sum(dim=1).clamp_min(1.0)
        center_y = (weights * yy).sum(dim=1) / count
        center_x = (weights * xx).sum(dim=1) / count
        dist2 = (yy - center_y[:, None]).square() + (
            xx - center_x[:, None]
        ).square()
        dist2 = torch.where(
            plateau, dist2, torch.full_like(dist2, torch.inf)
        )
        flat_idx = dist2.argmin(dim=1)
        yx = torch.stack([flat_idx // w, flat_idx % w], dim=1).long()
        return yx, valid

    @staticmethod
    def _distance_mask(
        reference_yx: Tensor, h: int, w: int, min_distance: float, dtype
    ) -> Tensor:
        yy, xx = torch.meshgrid(
            torch.arange(h, device=reference_yx.device, dtype=dtype),
            torch.arange(w, device=reference_yx.device, dtype=dtype),
            indexing="ij",
        )
        dy = yy[None] - reference_yx[:, 0, None, None].to(dtype)
        dx = xx[None] - reference_yx[:, 1, None, None].to(dtype)
        return (dy.square() + dx.square()).sqrt() >= float(min_distance)

    def forward(
        self,
        mask_logits: Tensor,
        boxes: Tensor,
        image_size,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        if mask_logits.ndim != 4 or mask_logits.shape[1] != 1:
            raise ValueError(
                f"mask_logits must be [N,1,H,W], got {tuple(mask_logits.shape)}"
            )
        if (
            boxes.ndim != 2
            or boxes.shape[1] != 4
            or boxes.shape[0] != mask_logits.shape[0]
        ):
            raise ValueError(
                f"boxes must be [N,4] aligned with masks, got {tuple(boxes.shape)}"
            )
        n, _, h, w = mask_logits.shape
        device = mask_logits.device
        if n == 0:
            zero = mask_logits.new_zeros(())
            coords = mask_logits.new_zeros((0, 4, 2))
            labels = torch.empty((0, 4), device=device, dtype=torch.long)
            self._last_local_yx = torch.empty((0, 4, 2), device=device, dtype=torch.long)
            self._last_labels = labels
            return coords, labels, {
                "p1_valid_count": zero,
                "p2_valid_count": zero,
                "n1_valid_count": zero,
                "n2_valid_count": zero,
                "roi_count": zero,
                "empty_count": zero,
                "full_count": zero,
                "positive_pair_distance_sum": zero,
                "positive_pair_count": zero,
                "negative_pair_distance_sum": zero,
                "negative_pair_count": zero,
                "pos_in_pred_fg_count": zero,
                "valid_pos_count": zero,
                "neg_in_pred_bg_count": zero,
                "valid_neg_count": zero,
                "point_distance_sum": zero,
                "point_pair_count": zero,
                "point_overlap_count": zero,
                "all_invalid_count": zero,
                "pjitter_count": zero,
            }

        with torch.no_grad():
            prob = torch.sigmoid(mask_logits.detach())
            foreground = prob >= self.foreground_thr
            flat_fg = foreground.flatten(1)
            has_fg = flat_fg.any(dim=1)
            full_fg = flat_fg.all(dim=1)
            has_strict_fg = (
                prob.flatten(1).max(dim=1).values > self.foreground_thr + 1e-6
            )
            inside = self._inside_distance(foreground.float())
            norm_inside = inside / float(max(1, min(h, w)))
            pos_score = norm_inside * prob

            p1_candidates = (
                foreground
                & (prob >= self.positive_conf_thr)
                & (norm_inside >= self.min_inside_distance_ratio)
            )
            p1_yx, p1_valid = self._pick_platform_center(
                p1_candidates.float(), pos_score
            )
            # A real foreground may be thin or below the high-confidence
            # threshold. Fall back to its best interior pixel instead of
            # invalidating the complete positive branch.
            fallback_p1_yx, fallback_p1_valid = self._pick_platform_center(
                foreground.float(), pos_score
            )
            use_p1_fallback = has_strict_fg & (~p1_valid) & fallback_p1_valid
            p1_yx = torch.where(
                use_p1_fallback[:, None], fallback_p1_yx, p1_yx
            )
            p1_valid = p1_valid | use_p1_fallback

            # Empty foreground: select the maximum-probability plateau only if
            # it reaches the explicit low-confidence fallback threshold.
            empty_candidates = prob >= prob.flatten(1).max(dim=1).values[:, None, None, None] - 1e-6
            empty_yx, _ = self._pick_platform_center(empty_candidates.float(), prob)
            empty_valid = (~has_fg) & (
                prob.flatten(1).max(dim=1).values
                >= self.empty_positive_conf_thr
            )
            p1_yx = torch.where((~has_fg)[:, None], empty_yx, p1_yx)
            p1_valid = torch.where(~has_fg, empty_valid, p1_valid)

            min_pos_dist = self.min_positive_distance_ratio * float(min(h, w))
            far_from_p1 = self._distance_mask(
                p1_yx, h, w, min_pos_dist, prob.dtype
            ).unsqueeze(1)
            p2_candidates = p1_candidates & far_from_p1 & p1_valid[:, None, None, None]
            p2_yx, p2_valid = self._pick_platform_center(
                p2_candidates.float(), pos_score
            )
            fallback_p2_candidates = foreground & far_from_p1
            fallback_p2_yx, fallback_p2_valid = self._pick_platform_center(
                fallback_p2_candidates.float(), pos_score
            )
            use_p2_fallback = p1_valid & (~p2_valid) & fallback_p2_valid
            p2_yx = torch.where(
                use_p2_fallback[:, None], fallback_p2_yx, p2_yx
            )
            p2_valid = p2_valid | use_p2_fallback

            outer = foreground.float()
            for _ in range(self.outer_ring_radius):
                outer = self.dilate(outer)
            outer_ring = (outer - foreground.float()).clamp(0, 1).bool()
            n1_candidates = outer_ring & (prob <= self.negative_conf_thr)

            # Empty masks have no outer ring. Use the ROI border as a safe
            # negative candidate set.
            border = torch.zeros((h, w), device=device, dtype=torch.bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            border = border[None, None].expand(n, 1, h, w)
            n1_candidates = torch.where(
                (~has_fg)[:, None, None, None],
                border & (prob <= self.negative_conf_thr),
                n1_candidates,
            )
            n1_candidates = n1_candidates & (~full_fg)[:, None, None, None]
            n1_score = 1.0 - prob
            n1_yx, n1_valid = self._pick_platform_center(
                n1_candidates.float(), n1_score
            )
            fallback_n1_candidates = (~foreground) & (~full_fg)[:, None, None, None]
            fallback_n1_yx, fallback_n1_valid = self._pick_platform_center(
                fallback_n1_candidates.float(), n1_score
            )
            use_n1_fallback = (~n1_valid) & fallback_n1_valid
            n1_yx = torch.where(
                use_n1_fallback[:, None], fallback_n1_yx, n1_yx
            )
            n1_valid = n1_valid | use_n1_fallback

            safe_radius = max(
                1,
                int(round(self.safe_background_radius_ratio * float(min(h, w)))),
            )
            expanded_fg = self._dilate_steps(foreground.float(), safe_radius).bool()
            safe_bg = (~expanded_fg) & (prob <= self.negative_conf_thr)
            min_neg_dist = self.min_negative_distance_ratio * float(min(h, w))
            far_from_n1 = self._distance_mask(
                n1_yx, h, w, min_neg_dist, prob.dtype
            ).unsqueeze(1)
            far_from_p1_n = self._distance_mask(
                p1_yx, h, w, min_neg_dist, prob.dtype
            ).unsqueeze(1)
            safe_bg = safe_bg & far_from_n1 & far_from_p1_n
            safe_bg = safe_bg & n1_valid[:, None, None, None]
            safe_bg = safe_bg & (~full_fg)[:, None, None, None]
            n2_yx, n2_valid = self._pick_platform_center(
                safe_bg.float(), 1.0 - prob
            )
            fallback_n2_candidates = (
                (~foreground)
                & far_from_n1
                & far_from_p1_n
                & n1_valid[:, None, None, None]
            )
            fallback_n2_yx, fallback_n2_valid = self._pick_platform_center(
                fallback_n2_candidates.float(), 1.0 - prob
            )
            use_n2_fallback = (~n2_valid) & fallback_n2_valid
            n2_yx = torch.where(
                use_n2_fallback[:, None], fallback_n2_yx, n2_yx
            )
            n2_valid = n2_valid | use_n2_fallback

            if not self.adaptive_validity:
                # Fixed-validity control: always provide two positive and two
                # negative extrema. Confidence/topology thresholds still define
                # the adaptive route above, but must not silently affect this
                # ablation. Distance masks keep the second point separated.
                full_canvas = torch.ones_like(foreground, dtype=torch.bool)
                if h * w < 4:
                    raise ValueError(
                        "fixed-validity 2P2N requires at least four coarse-mask pixels"
                    )
                p1_yx, _ = self._pick_platform_center(full_canvas, prob)
                fixed_far_from_p1 = self._distance_mask(
                    p1_yx, h, w, min_pos_dist, prob.dtype
                ).unsqueeze(1)
                p2_yx, p2_valid = self._pick_platform_center(
                    fixed_far_from_p1, prob
                )
                positive_exclusion = self._distance_mask(
                    p1_yx, h, w, max(1.0, min_neg_dist), prob.dtype
                ).unsqueeze(1)
                positive_exclusion = positive_exclusion & self._distance_mask(
                    p2_yx, h, w, max(1.0, min_neg_dist), prob.dtype
                ).unsqueeze(1)
                n1_yx, n1_valid = self._pick_platform_center(
                    full_canvas & positive_exclusion, 1.0 - prob
                )
                fixed_far_from_n1 = self._distance_mask(
                    n1_yx, h, w, min_neg_dist, prob.dtype
                ).unsqueeze(1)
                fixed_far_from_p1 = self._distance_mask(
                    p1_yx, h, w, min_neg_dist, prob.dtype
                ).unsqueeze(1)
                fixed_far_from_p2 = self._distance_mask(
                    p2_yx, h, w, min_neg_dist, prob.dtype
                ).unsqueeze(1)
                n2_candidates = (
                    full_canvas
                    & fixed_far_from_n1
                    & fixed_far_from_p1
                    & fixed_far_from_p2
                )
                n2_yx, n2_valid = self._pick_platform_center(
                    n2_candidates, 1.0 - prob
                )

                if not bool((p2_valid & n1_valid & n2_valid).all()):
                    raise RuntimeError(
                        "fixed-validity 2P2N could not find four mutually "
                        "separated points; reduce the configured distance ratios"
                    )

                p1_valid = torch.ones(n, device=device, dtype=torch.bool)
                p2_valid = torch.ones(n, device=device, dtype=torch.bool)
                n1_valid = torch.ones(n, device=device, dtype=torch.bool)
                n2_valid = torch.ones(n, device=device, dtype=torch.bool)

            local_yx = torch.stack([p1_yx, p2_yx, n1_yx, n2_yx], dim=1)
            labels = torch.stack(
                [
                    torch.where(
                        p1_valid,
                        torch.ones(n, device=device, dtype=torch.long),
                        torch.full((n,), -1, device=device, dtype=torch.long),
                    ),
                    torch.where(
                        p2_valid,
                        torch.ones(n, device=device, dtype=torch.long),
                        torch.full((n,), -1, device=device, dtype=torch.long),
                    ),
                    torch.where(
                        n1_valid,
                        torch.zeros(n, device=device, dtype=torch.long),
                        torch.full((n,), -1, device=device, dtype=torch.long),
                    ),
                    torch.where(
                        n2_valid,
                        torch.zeros(n, device=device, dtype=torch.long),
                        torch.full((n,), -1, device=device, dtype=torch.long),
                    ),
                ],
                dim=1,
            )

            stage = self._warmup_stage()
            if stage == 0:
                labels.fill_(-1)
            elif stage == 1:
                labels[:, 1] = -1
                labels[:, 3] = -1

            # Training-only positive-point jitter (plan D: prompt-consumption
            # robustification). Perturb P1/P2 by up to point_jitter_radius
            # canvas cells: positive candidates are gated by
            # min_inside_distance_ratio (>= ~3 cells inside the object at the
            # 64x64 canvas), so a 1-cell offset cannot cross the boundary.
            # Negatives are never jittered: N1 lives on a 1-cell outer ring
            # and any offset would flip its foreground/background semantics.
            pjitter_count = 0
            if (
                self.training
                and stage == 2
                and self.point_jitter_prob > 0.0
            ):
                radius = int(self.point_jitter_radius)
                local_yx = local_yx.clone()
                for slot in (0, 1):
                    sel = (
                        torch.rand(n, device=device) < self.point_jitter_prob
                    ) & (labels[:, slot] >= 0)
                    if bool(sel.any()):
                        off = torch.randint(
                            -radius, radius + 1, (int(sel.sum()), 2),
                            device=device,
                        )
                        local_yx[sel, slot, 0] = (
                            local_yx[sel, slot, 0] + off[:, 0]
                        ).clamp(0, h - 1)
                        local_yx[sel, slot, 1] = (
                            local_yx[sel, slot, 1] + off[:, 1]
                        ).clamp(0, w - 1)
                        pjitter_count += int(sel.sum())

        boxes_f = boxes.detach().to(mask_logits.dtype)
        x1, y1, x2, y2 = boxes_f.unbind(dim=1)
        box_w = (x2 - x1).clamp_min(self.min_box_size)
        box_h = (y2 - y1).clamp_min(self.min_box_size)

        def _to_image(yx: Tensor) -> Tensor:
            nx = (yx[..., 1].to(mask_logits.dtype) + 0.5) / float(max(w, 1))
            ny = (yx[..., 0].to(mask_logits.dtype) + 0.5) / float(max(h, 1))
            return torch.stack(
                [x1[:, None] + nx * box_w[:, None], y1[:, None] + ny * box_h[:, None]],
                dim=-1,
            )

        coords = _to_image(local_yx)
        if isinstance(image_size, (tuple, list)):
            image_h, image_w = int(image_size[0]), int(image_size[1])
        else:
            image_h = image_w = int(image_size)
        coords[..., 0].clamp_(0.0, max(float(image_w - 1), 0.0))
        coords[..., 1].clamp_(0.0, max(float(image_h - 1), 0.0))

        self._last_local_yx = local_yx.detach()
        self._last_labels = labels.detach()

        with torch.no_grad():
            valid_p1 = labels[:, 0] == 1
            valid_p2 = labels[:, 1] == 1
            valid_n1 = labels[:, 2] == 0
            valid_n2 = labels[:, 3] == 0
            pos_pair_valid = valid_p1 & valid_p2
            neg_pair_valid = valid_n1 & valid_n2
            pos_pair_dist = torch.norm(coords[:, 0] - coords[:, 1], dim=-1)
            neg_pair_dist = torch.norm(coords[:, 2] - coords[:, 3], dim=-1)
            all_pos_valid = torch.stack([valid_p1, valid_p2], dim=1)
            all_neg_valid = torch.stack([valid_n1, valid_n2], dim=1)
            batch_idx = torch.arange(n, device=device)[:, None].expand(-1, 4)
            sampled_fg = foreground[
                batch_idx,
                torch.zeros_like(batch_idx),
                local_yx[..., 0].clamp(0, h - 1),
                local_yx[..., 1].clamp(0, w - 1),
            ]
            pos_in_fg = sampled_fg[:, :2] & all_pos_valid
            neg_in_bg = (~sampled_fg[:, 2:]) & all_neg_valid
            overlaps = []
            for i in range(4):
                for j in range(i + 1, 4):
                    pair_valid = (labels[:, i] >= 0) & (labels[:, j] >= 0)
                    overlaps.append(
                        ((local_yx[:, i] == local_yx[:, j]).all(dim=1) & pair_valid)
                    )
            overlap_count = torch.stack(overlaps, dim=1).sum() if overlaps else mask_logits.new_zeros(())
            valid_pos_count = all_pos_valid.sum()
            valid_neg_count = all_neg_valid.sum()
            stats = {
                "p1_valid_count": valid_p1.sum().detach(),
                "p2_valid_count": valid_p2.sum().detach(),
                "n1_valid_count": valid_n1.sum().detach(),
                "n2_valid_count": valid_n2.sum().detach(),
                "roi_count": mask_logits.new_tensor(float(n)),
                "empty_count": (~has_fg).sum().detach(),
                "full_count": full_fg.sum().detach(),
                "positive_pair_distance_sum": (
                    pos_pair_dist * pos_pair_valid.to(pos_pair_dist.dtype)
                ).sum().detach(),
                "positive_pair_count": pos_pair_valid.sum().detach(),
                "negative_pair_distance_sum": (
                    neg_pair_dist * neg_pair_valid.to(neg_pair_dist.dtype)
                ).sum().detach(),
                "negative_pair_count": neg_pair_valid.sum().detach(),
                # Backward-compatible aggregate fields.
                "pos_in_pred_fg_count": pos_in_fg.sum().detach(),
                "valid_pos_count": valid_pos_count.detach(),
                "neg_in_pred_bg_count": neg_in_bg.sum().detach(),
                "valid_neg_count": valid_neg_count.detach(),
                "point_distance_sum": (
                    pos_pair_dist * pos_pair_valid.to(pos_pair_dist.dtype)
                ).sum().detach()
                + (
                    neg_pair_dist * neg_pair_valid.to(neg_pair_dist.dtype)
                ).sum().detach(),
                "point_pair_count": (
                    pos_pair_valid.sum() + neg_pair_valid.sum()
                ).detach(),
                "point_overlap_count": overlap_count.detach(),
                "all_invalid_count": (labels < 0).all(dim=1).sum().detach(),
                "warmup_stage": mask_logits.new_tensor(float(stage)),
                "pjitter_count": mask_logits.new_tensor(float(pjitter_count)),
            }
        return coords, labels, stats

    def get_last_local_points(self) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        return self._last_local_yx, self._last_labels
