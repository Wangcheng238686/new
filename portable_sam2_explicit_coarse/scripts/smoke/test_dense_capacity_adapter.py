#!/usr/bin/env python
"""CPU tensor contract for the dense-only residual capacity adapter.

This is intentionally small: full model parity belongs to checkpoint smoke,
while this test makes the two non-negotiable local properties executable:
zero-init identity and no gradient through the detached RoI/source operands.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("RSPROMPTER_LIGHT_IMPORT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from rsprompter.shape_prior_injector import SmallMaskDecoder


def main() -> int:
    torch.manual_seed(44)
    roi = torch.randn(2, 512, 14, 14, requires_grad=True)
    point_logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    adapter = SmallMaskDecoder(in_ch=512, mid_ch=256, out_ch=1, out_size=64)
    final = adapter.net[-1]
    torch.nn.init.zeros_(final.weight)
    torch.nn.init.zeros_(final.bias)

    dense_logits = point_logits.detach() + adapter(roi.detach())
    assert torch.equal(dense_logits, point_logits.detach()), "zero-init is not identity"
    dense_logits.square().mean().backward()
    assert roi.grad is None, "adapter gradient leaked into RoI feature"
    assert point_logits.grad is None, "adapter gradient leaked into point source"
    assert final.weight.grad is not None and final.weight.grad.abs().sum() > 0, (
        "residual output layer received no gradient"
    )
    print("dense-capacity adapter tensor contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
