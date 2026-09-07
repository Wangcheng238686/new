"""Tensor-level regression checks for the independent UDPR module."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rsprompter.decoder_tail_refiner import DecoderTailPointRefiner


def main() -> None:
    module = DecoderTailPointRefiner(num_points=4)
    logits = torch.tensor([[[[-1.0, 0.1], [0.1, 2.0]]]], requires_grad=True)
    upscaled = torch.randn(1, 32, 2, 2)
    token = torch.randn(1, 1, 256)
    refined, outputs = module(logits, upscaled, token)
    # Zero-initialized residual must not perturb the A0 native logits.
    assert torch.equal(refined, logits)
    # Stable tie break is native raster order: two 0.1 cells -> indices 1,2.
    assert outputs["indices"].tolist() == [[1, 2, 0, 3]]
    loss, stats = module.point_loss(outputs, torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]))
    # With zero residual, the only base error is raster index 3.  It is
    # selected, but cannot yet be changed/flipped/destroyed.
    assert torch.isclose(stats["TAIL/selected_error_fraction"], torch.tensor(0.25))
    assert torch.isclose(stats["TAIL/selected_error_coverage"], torch.tensor(1.0))
    assert torch.isclose(stats["TAIL/changed_fraction"], torch.tensor(0.0))
    assert torch.isclose(stats["TAIL/correct_flip_fraction"], torch.tensor(0.0))
    assert torch.isclose(stats["TAIL/destroy_fraction"], torch.tensor(0.0))
    loss.backward()
    assert module.mlp[-1].weight.grad is not None
    assert float(module.mlp[-1].weight.grad.abs().sum()) > 0.0

    # A deterministic non-zero residual checks the target-aware telemetry,
    # including both a helpful flip and a destructive flip.  All four native
    # points are selected; two are initially wrong.
    moving = DecoderTailPointRefiner(num_points=4)
    with torch.no_grad():
        moving.mlp[-1].bias.fill_(1.0)
    moving_logits = torch.tensor([[[[-0.1, 0.1], [-0.5, 0.5]]]])
    _, moving_outputs = moving(moving_logits, upscaled, token)
    _, moving_stats = moving.point_loss(
        moving_outputs, torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    )
    assert torch.isclose(moving_stats["TAIL/selected_error_fraction"], torch.tensor(0.5))
    assert torch.isclose(moving_stats["TAIL/selected_error_coverage"], torch.tensor(1.0))
    assert torch.isclose(moving_stats["TAIL/changed_fraction"], torch.tensor(1.0))
    assert torch.isclose(moving_stats["TAIL/error_direction_agreement"], torch.tensor(0.5))
    assert torch.isclose(moving_stats["TAIL/correct_flip_fraction"], torch.tensor(0.25))
    assert torch.isclose(moving_stats["TAIL/destroy_fraction"], torch.tensor(0.25))
    print("UDPR tensor smoke: PASS")


if __name__ == "__main__":
    main()
