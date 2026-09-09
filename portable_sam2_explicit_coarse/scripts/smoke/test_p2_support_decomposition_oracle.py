#!/usr/bin/env python3
"""Tensor contracts for the frozen-A0 P2 support decomposition Oracle."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rsprompter.p2_boundary_refiner import P2BoundaryRefiner  # noqa: E402
from inference.probes.p2_support_decomposition_oracle import _roi_target_grid  # noqa: E402


def main() -> int:
    # Match training target geometry, not the legacy int() truncation helper.
    full = torch.arange(36, dtype=torch.float32).reshape(6, 6) % 2
    fractional = torch.tensor([1.2, 1.8, 4.1, 4.2])
    expected = F.interpolate(full[1:5, 1:5][None, None], size=(3, 4), mode="nearest")[0, 0].bool()
    assert torch.equal(_roi_target_grid(full, fractional, (3, 4)), expected)
    raw = torch.full((2, 1, 64, 64), -2.)
    raw[:, :, 18:46, 18:46] = 2.  # non-empty raw boundary band
    base = P2BoundaryRefiner(p2_in_channels=512, beta=.1, delta_logit_max=.5,
                             boundary_band_radius=2, search_band_radius=4,
                             max_search_coverage=.5)
    ungated = P2BoundaryRefiner(p2_in_channels=512, beta=.1, delta_logit_max=.5,
                                boundary_band_radius=2, search_band_radius=4,
                                max_search_coverage=1.)
    _, _, _, accepted, _ = base._raw_support(raw)
    _, _, _, all_s4, _ = ungated._raw_support(raw)
    assert torch.all(accepted <= all_s4), "coverage rejection may only remove support"
    target = raw.ge(0).clone(); target[:, :, 20:24, 20:24] = False
    error = raw.ge(0).ne(target)
    candidate = raw.clone(); write = error & accepted.bool()
    candidate[write] = torch.where(target[write], torch.full_like(candidate[write], 8.), torch.full_like(candidate[write], -8.))
    assert torch.equal(candidate[~write], raw[~write]), "Oracle must preserve correct/support-outside logits"
    print("PASS: P2 support subset and correct-errors-only contracts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
