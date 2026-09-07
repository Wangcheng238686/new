#!/usr/bin/env python
"""Unit tests for rsprompter.p2_boundary_refiner.saturation_telemetry.

Covers the audit requirements for the P2BR saturation fix:
  - dynamic threshold 0.99 * beta * delta_logit_max (golden-run value 0.396
    preserved; D2-style small envelope gets a meaningful threshold);
  - ge (>=) semantics at exactly the threshold;
  - statistics restricted to search_support;
  - pooled numerator/denominator semantics (NOT per-ROI mean of ratios:
    empty/rejected-support ROIs must not dilute the ratio, and empty total
    support must not produce NaN);
  - legacy delta_saturated_sum stays the all-pixel per-ROI-mean sum;
  - step/DDP aggregation: sums accumulate, ratio computed after reduction.

Run: python scripts/smoke/test_p2br_saturation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rsprompter.p2_boundary_refiner import saturation_telemetry


def _close(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


def test_threshold_values():
    for beta, cap, expected in (
        (0.20, 2.0, 0.396),  # golden run: threshold must stay 0.396
        (0.10, 0.50, 0.0495),  # D2 envelope
        (1.0, 1.0, 0.99),
    ):
        delta = torch.full((1, 1, 4, 4), 10.0)
        support = torch.ones(1, 1, 4, 4)
        out = saturation_telemetry(delta, support, beta=beta, delta_logit_max=cap)
        assert _close(out["saturation_threshold"], expected), (beta, cap, out)
        # |delta| >= threshold -> saturated; the legacy stat is the per-ROI
        # MEAN of the all-pixel saturated fraction (1.0 here), summed over ROIs.
        assert _close(out["delta_saturated_sum"], 1.0)
    print("PASS threshold values + ge saturation")


def test_ge_boundary():
    beta, cap = 0.1, 1.0
    threshold = 0.99 * beta * cap
    delta = torch.tensor([[[[threshold]]]])  # exactly at threshold
    support = torch.ones(1, 1, 1, 1)
    out = saturation_telemetry(delta, support, beta=beta, delta_logit_max=cap)
    assert _close(out["saturated_support_pixel_sum"], 1.0), "ge must count exactly-at-threshold"
    delta_below = torch.tensor([[[[threshold * 0.999]]]])
    out2 = saturation_telemetry(delta_below, support, beta=beta, delta_logit_max=cap)
    assert _close(out2["saturated_support_pixel_sum"], 0.0)
    print("PASS >= boundary semantics")


def test_support_restriction():
    # Large logits OUTSIDE the support must not enter the in-support stats.
    delta = torch.zeros(1, 1, 2, 2)
    delta[..., 0, 0] = 100.0  # outside support
    support = torch.zeros(1, 1, 2, 2)
    support[..., 1, 1] = 1.0
    delta[..., 1, 1] = 10.0  # inside support, saturated
    out = saturation_telemetry(delta, support, beta=0.2, delta_logit_max=2.0)
    assert _close(out["support_pixel_sum"], 1.0)
    assert _close(out["saturated_support_pixel_sum"], 1.0)
    # Legacy all-pixel stat counts BOTH saturated pixels (it is unrestricted).
    assert _close(out["delta_saturated_sum"], 2.0 / 4.0)
    print("PASS support restriction")


def test_pooled_not_mean_of_ratios():
    # ROI A: 100 support px all saturated; ROI B: 10 support px none saturated.
    # Pooled ratio must be 100/110, per-ROI mean of ratios would be 0.5.
    roi_a = torch.zeros(1, 1, 10, 10)
    roi_a[:, :, :10, :10] = 10.0
    support_a = torch.zeros(1, 1, 10, 10)
    support_a[:, :, :10, :10] = 1.0
    roi_b = torch.zeros(1, 1, 10, 10)
    support_b = torch.zeros(1, 1, 10, 10)
    support_b[:, :, :1, :10] = 1.0  # 10 px, delta stays 0 -> not saturated
    batch = torch.cat([roi_a, roi_b])
    support = torch.cat([support_a, support_b])
    out = saturation_telemetry(batch, support, beta=0.2, delta_logit_max=2.0)
    ratio = float(out["saturated_support_pixel_sum"]) / float(out["support_pixel_sum"])
    assert _close(ratio, 100.0 / 110.0), ratio
    assert not _close(ratio, 0.5)
    print("PASS pooled numerator/denominator semantics")


def test_rejected_and_empty_support():
    beta, cap = 0.2, 2.0
    # ROI with real support (saturated) + rejected ROI (support zeroed, delta
    # huge but unsupported) -> rejected ROI must not dilute the ratio.
    roi_ok = torch.full((1, 1, 4, 4), 10.0)
    support_ok = torch.ones(1, 1, 4, 4)
    roi_rejected = torch.full((1, 1, 4, 4), 10.0)
    support_rejected = torch.zeros(1, 1, 4, 4)
    alone = saturation_telemetry(roi_ok, support_ok, beta, cap)
    both = saturation_telemetry(
        torch.cat([roi_ok, roi_rejected]),
        torch.cat([support_ok, support_rejected]), beta, cap
    )
    assert _close(alone["support_pixel_sum"], both["support_pixel_sum"])
    assert _close(
        alone["saturated_support_pixel_sum"], both["saturated_support_pixel_sum"]
    )
    # All support empty -> denominator 0 must clamp to a 0-safe value, no NaN.
    empty = saturation_telemetry(roi_rejected, support_rejected, beta, cap)
    ratio = float(empty["saturated_support_pixel_sum"]) / max(
        float(empty["support_pixel_sum"]), 1.0
    )
    assert ratio == 0.0
    print("PASS rejected/empty support (no dilution, no NaN)")


def test_aggregation_semantics():
    """Epoch/DDP aggregation: sum num & den over parts, then divide."""
    beta, cap = 0.2, 2.0
    batches = [
        (torch.full((2, 1, 4, 4), 10.0), torch.ones(2, 1, 4, 4)),
        (torch.zeros(3, 1, 4, 4), torch.ones(3, 1, 4, 4)),
    ]
    joint = saturation_telemetry(
        torch.cat([b[0] for b in batches]), torch.cat([b[1] for b in batches]), beta, cap
    )
    num = sum(
        float(saturation_telemetry(d, s, beta, cap)["saturated_support_pixel_sum"])
        for d, s in batches
    )
    den = sum(float(saturation_telemetry(d, s, beta, cap)["support_pixel_sum"]) for d, s in batches)
    assert _close(num, float(joint["saturated_support_pixel_sum"]))
    assert _close(den, float(joint["support_pixel_sum"]))
    # Trainer epoch aggregation divides the SUMS (all-reduced across ranks),
    # never per-step ratios.
    epoch_ratio = num / max(den, 1.0)
    direct_ratio = float(joint["saturated_support_pixel_sum"]) / float(joint["support_pixel_sum"])
    assert _close(epoch_ratio, direct_ratio)
    print("PASS step/DDP aggregation semantics")


def main():
    torch.manual_seed(44)
    test_threshold_values()
    test_ge_boundary()
    test_support_restriction()
    test_pooled_not_mean_of_ratios()
    test_rejected_and_empty_support()
    test_aggregation_semantics()
    print("ALL P2BR saturation telemetry tests PASSED")


if __name__ == "__main__":
    main()
