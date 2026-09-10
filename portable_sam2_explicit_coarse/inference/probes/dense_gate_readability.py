"""Pure-tensor utilities for the pre-PromptEncoder dense-gate readability test.

No model is constructed here.  The eventual frozen-forward collector supplies
one row per proposal, while these functions define a deterministic, testable
feature/readout contract shared by raw-only, aligned, and row-permuted arms.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LinearGate:
    """Standardised logistic readout fitted only on the training split."""
    mean: torch.Tensor
    std: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor

    def score(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.mean.numel():
            raise ValueError("feature dimension differs from fitted readout")
        x = (features.float() - self.mean) / self.std
        return torch.sigmoid(x.matmul(self.weight) + self.bias)


def _map_stats(logits: torch.Tensor) -> torch.Tensor:
    """Six bounded/unbounded map summaries, one row per ROI."""
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("logits must be [N,1,H,W]")
    x = logits.float().squeeze(1)
    prob = x.sigmoid().clamp(1e-6, 1 - 1e-6)
    entropy = -(prob * prob.log() + (1 - prob) * (1 - prob).log())
    # Sign changes are a resolution-stable boundary-density proxy; no target
    # masks or decoded SAM output are involved.
    fg = x >= 0
    horizontal = fg[:, :, 1:] != fg[:, :, :-1]
    vertical = fg[:, 1:, :] != fg[:, :-1, :]
    boundary = (horizontal.float().mean((1, 2)) + vertical.float().mean((1, 2))) * .5
    return torch.stack((
        x.mean((1, 2)), x.std((1, 2), unbiased=False), x.abs().mean((1, 2)),
        prob.mean((1, 2)), entropy.mean((1, 2)), boundary,
    ), dim=1)


def pre_prompt_features(
    roi_feats: torch.Tensor,
    raw_coarse_logits: torch.Tensor,
    canvas_logits: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    image_hw: Tuple[int, int],
    labels: torch.Tensor,
    scores: torch.Tensor,
    num_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build raw-only and full pre-PE feature rows.

    The full arm contains per-channel spatial mean/std of the detector RoI
    feature map.  It is generated *before* ``PromptEncoder``.  The raw arm
    excludes this aligned appearance vector but keeps exactly the same coarse,
    canvas, geometry, class and detector-score scalar features.
    """
    if roi_feats.ndim != 4 or raw_coarse_logits.shape[0] != roi_feats.shape[0]:
        raise ValueError("ROI and raw coarse batches must align")
    n, channels = roi_feats.shape[:2]
    if canvas_logits.shape[0] != n or boxes_xyxy.shape != (n, 4):
        raise ValueError("canvas/boxes must align with ROI rows")
    if labels.shape != (n,) or scores.shape != (n,):
        raise ValueError("labels/scores must be [N]")
    if not bool(((labels >= 0) & (labels < num_classes)).all()):
        raise ValueError("labels outside explicit class contract")
    image_h, image_w = (float(image_hw[0]), float(image_hw[1]))
    if image_h <= 0 or image_w <= 0:
        raise ValueError("image_hw must be positive")
    x1, y1, x2, y2 = boxes_xyxy.float().unbind(1)
    width = (x2 - x1).clamp_min(1.0) / image_w
    height = (y2 - y1).clamp_min(1.0) / image_h
    geometry = torch.stack((
        ((x1 + x2) * .5) / image_w, ((y1 + y2) * .5) / image_h,
        width.log(), height.log(), (width / height).log(), (width * height).log(),
    ), dim=1)
    class_onehot = F.one_hot(labels.long(), num_classes=num_classes).float()
    scalar = torch.cat((_map_stats(raw_coarse_logits), _map_stats(canvas_logits),
                        geometry, class_onehot, scores.float()[:, None]), dim=1)
    appearance = torch.cat((roi_feats.float().mean((2, 3)),
                            roi_feats.float().std((2, 3), unbiased=False)), dim=1)
    if appearance.shape[1] != 2 * channels:
        raise AssertionError("unexpected ROI appearance feature width")
    return scalar, torch.cat((scalar, appearance), dim=1)


def fit_balanced_linear(
    features: torch.Tensor, labels: torch.Tensor, *, steps: int = 400,
    lr: float = .03, seed: int = 44,
) -> LinearGate:
    """Fit a deterministic class-balanced logistic diagnostic readout."""
    if features.ndim != 2 or labels.shape != (features.shape[0],):
        raise ValueError("features must be [N,D] and labels [N]")
    y = labels.float()
    if not bool(((y == 0) | (y == 1)).all()) or y.sum() in (0, len(y)):
        raise ValueError("readout needs both binary classes")
    mean = features.float().mean(0)
    std = features.float().std(0, unbiased=False).clamp_min(1e-5)
    x = ((features.float() - mean) / std)
    linear = torch.nn.Linear(x.shape[1], 1, device=x.device)
    torch.nn.init.zeros_(linear.weight); torch.nn.init.zeros_(linear.bias)
    positive_weight = (len(y) - y.sum()) / y.sum()
    opt = torch.optim.AdamW(linear.parameters(), lr=lr, weight_decay=1e-4)
    generator = torch.Generator(device=x.device).manual_seed(seed)
    for _ in range(steps):
        indices = torch.randint(len(x), (min(2048, len(x)),), generator=generator, device=x.device)
        loss = F.binary_cross_entropy_with_logits(
            linear(x[indices]).squeeze(1), y[indices], pos_weight=positive_weight
        )
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return LinearGate(mean.detach(), std.detach(), linear.weight.detach()[0], linear.bias.detach())


def row_permute(features: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Break instance-feature correspondence while preserving feature marginals."""
    if features.ndim != 2 or len(features) < 2:
        raise ValueError("row permutation needs at least two feature rows")
    generator = torch.Generator(device=features.device).manual_seed(seed)
    return features[torch.randperm(len(features), generator=generator, device=features.device)]


def binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Tie-aware Mann--Whitney AUC without optional sklearn dependency."""
    score = scores.detach().float().cpu().numpy(); y = labels.detach().cpu().numpy().astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if not n_pos or not n_neg:
        raise ValueError("AUC needs both classes")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and score[order[end]] == score[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) * .5
        start = end
    return float((ranks[y].sum() - n_pos * (n_pos + 1) * .5) / (n_pos * n_neg))
