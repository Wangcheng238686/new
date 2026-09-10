#!/usr/bin/env python3
"""CPU contract test for pre-PromptEncoder gate feature/readout utilities."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from inference.probes.dense_gate_readability import (  # noqa: E402
    binary_auc, fit_balanced_linear, pre_prompt_features, row_permute,
)


def main():
    torch.manual_seed(0)
    n = 12
    roi = torch.randn(n, 4, 3, 3)
    raw = torch.randn(n, 1, 4, 4)
    canvas = torch.randn(n, 1, 8, 8)
    boxes = torch.tensor([[0., 0., 10. + i, 12. + i] for i in range(n)])
    labels = torch.arange(n) % 3
    scores = torch.linspace(.1, .9, n)
    scalar, full = pre_prompt_features(roi, raw, canvas, boxes, (32, 32), labels, scores, 3)
    assert scalar.shape == (n, 6 + 6 + 6 + 3 + 1)
    assert full.shape == (n, scalar.shape[1] + 8)
    assert torch.equal(full[:, :scalar.shape[1]], scalar)
    permuted = row_permute(full, seed=44)
    assert sorted(map(tuple, permuted.tolist())) == sorted(map(tuple, full.tolist()))
    y = torch.tensor([0, 1] * (n // 2))
    # A deliberately readable synthetic scalar must achieve high AUC, while
    # the utility preserves arbitrary feature widths.
    x = torch.stack((y.float() * 2 - 1, torch.randn(n)), 1)
    model = fit_balanced_linear(x, y, steps=120, seed=44)
    assert binary_auc(model.score(x), y) > .99
    try:
        pre_prompt_features(roi, raw, canvas, boxes, (0, 32), labels, scores, 3)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid image geometry must fail")
    print("PASS dense gate readability feature/readout contracts")


if __name__ == "__main__":
    main()
