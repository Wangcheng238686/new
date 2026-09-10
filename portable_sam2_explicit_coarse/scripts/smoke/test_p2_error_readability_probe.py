#!/usr/bin/env python3
"""CPU contracts for the frozen-A0 P2 error-readability gate."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from inference.probes.p2_error_readability_probe import (  # noqa: E402
    _add_counts, _bootstrap, _safe_permutation, _verdict,
)


def _row(image_id: int, *, p2_hits: int) -> dict:
    row = {"image_id": image_id}
    for name, hits in (("raw", 2), ("p2", p2_hits), ("permuted", 2)):
        row.update({name + "_inter": 8 + hits, name + "_union": 30,
                    name + "_hits": hits, name + "_writes": 8,
                    name + "_errors": 8})
    return row


def main() -> int:
    p2 = torch.arange(2 * 3 * 4 * 4).reshape(2, 3, 4, 4)
    assert not torch.equal(_safe_permutation(p2)[0], p2[0])
    assert torch.equal(_safe_permutation(p2)[0], p2[0].roll((2, 2), dims=(-2, -1)))
    one = p2[:1]
    assert not torch.equal(_safe_permutation(one), one), "one-ROI control must still be spatially misaligned"

    # A fixed top-50% budget selects the two highest scores, flips only those
    # logits, and counts old-support-outside errors rather than all pixels.
    row = {key: 0.0 for name in ("raw", "p2", "permuted") for key in
           (name + "_inter", name + "_union", name + "_hits", name + "_writes", name + "_errors")}
    raw = torch.tensor([[3., -3.], [-3., 3.]])
    target = torch.tensor([[False, True], [False, True]])
    eligible = torch.ones_like(target)
    _add_counts(row, "p2", raw, target, eligible, torch.tensor([[.9, .8], [.1, .2]]), .5)
    assert row["p2_hits"] == 2 and row["p2_writes"] == 2 and row["p2_errors"] == 2
    raw_row = {key: 0.0 for key in row}
    _add_counts(raw_row, "raw", raw, target, eligible, torch.tensor([[.9, .8], [.1, .2]]), .5)
    # Raw selector must contribute only its post-write mask, never a second
    # unmodified baseline under the same raw_ prefix.
    assert raw_row["raw_inter"] == 2 and raw_row["raw_union"] == 2

    bootstrap = _bootstrap([_row(i, p2_hits=5) for i in range(10)], 100, 44)
    assert _verdict(bootstrap)["verdict"] == "pass"
    no_control = _bootstrap([_row(i, p2_hits=2) for i in range(10)], 100, 44)
    assert _verdict(no_control)["verdict"] == "fail"
    print("PASS: P2 readability selector / permutation / image-bootstrap contracts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
