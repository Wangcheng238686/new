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

    # DCR/A4 starts as the exact identity despite a non-zero conservative
    # gate prior.  Its two output heads receive gradients on the first update;
    # the shared trunk receives one after those heads become non-zero.
    dcr = DecoderTailPointRefiner(num_points=4, mode="confidence_gated", gate_init_prob=0.1)
    dcr_logits = logits.detach().clone().requires_grad_(True)
    dcr_refined, dcr_outputs = dcr(dcr_logits, upscaled, token)
    assert torch.equal(dcr_refined, dcr_logits)
    dcr_loss, dcr_stats = dcr.point_loss(
        dcr_outputs, torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    )
    assert torch.isclose(dcr_stats["TAIL/gate_mean"], torch.tensor(0.1), atol=1e-6)
    assert torch.isclose(dcr_stats["TAIL/applied_delta_abs"], torch.tensor(0.0))
    assert torch.isclose(dcr_stats["TAIL/net_flip_fraction"], torch.tensor(0.0))
    opt = torch.optim.SGD(dcr.parameters(), lr=0.1)
    dcr_loss.backward()
    assert float(dcr.delta_head.weight.grad.abs().sum()) > 0.0
    assert float(dcr.gate_head.weight.grad.abs().sum()) > 0.0
    opt.step()
    opt.zero_grad(set_to_none=True)
    _, dcr_outputs_2 = dcr(dcr_logits.detach(), upscaled, token)
    dcr_loss_2, _ = dcr.point_loss(dcr_outputs_2, torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]))
    dcr_loss_2.backward()
    assert float(dcr.shared_mlp[1].weight.grad.abs().sum()) > 0.0

    # Production A4 runs with AMP.  Gate supervision must use logits-space
    # BCE, because sigmoid followed by BCE is unsafe under CUDA autocast.
    amp = DecoderTailPointRefiner(num_points=4, mode="confidence_gated")
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        _, amp_outputs = amp(logits.detach(), upscaled, token)
        amp_loss, _ = amp.point_loss(
            amp_outputs, torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
        )
    amp_loss.backward()
    assert amp.gate_head.weight.grad is not None
    assert float(amp.gate_head.weight.grad.abs().sum()) > 0.0
    print("UDPR tensor smoke: PASS")


if __name__ == "__main__":
    main()
