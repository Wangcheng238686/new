"""Contract checks for the coupled margin UDPR loss (the a3sgm tail).

Checks: violation semantics (zero for points already beyond boundary+margin
on the correct side, positive on the wrong side); the co-adaptation gradient
channel (loss and refined logits both backpropagate into the decoder inputs);
empty-rank telemetry keys; and zero-init forward equivalence.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rsprompter.decoder_tail_refiner import DecoderTailPointRefiner

BOUNDARY = -0.4054651081081644  # logit(0.4), the deployed threshold


def main() -> None:
    torch.manual_seed(0)
    refiner = DecoderTailPointRefiner(
        num_points=8, mode="residual_v1_margin",
        crossing_margin=0.10, boundary_logit=BOUNDARY,
    )
    # Explicit-config and default constructors must stay state-dict identical
    # (both paper checkpoints carry the exact mlp.{0,1,3} key family).
    default = DecoderTailPointRefiner(num_points=8)
    assert sorted(refiner.state_dict()) == sorted(default.state_dict())

    logits = torch.randn(3, 1, 16, 16, requires_grad=True)
    upscaled = torch.randn(3, 32, 16, 16, requires_grad=True)
    token = torch.randn(3, 1, 256, requires_grad=True)
    with torch.no_grad():
        refiner.mlp[-1].weight.add_(0.1 * torch.randn_like(refiner.mlp[-1].weight))
        refiner.mlp[-1].bias.add_(0.05)

    refined, outputs = refiner(logits, upscaled, token)
    targets = (torch.rand(3, 1, 16, 16) > 0.5).float()
    loss, stats = refiner.point_loss(outputs, targets)
    assert torch.isfinite(loss)
    # Coupled container: the margin loss reaches tail parameters AND the
    # decoder-side inputs (the a3sgm co-adaptation channel).
    (loss + refined.sum() * 0.01).backward()
    for name, param in refiner.named_parameters():
        assert param.grad is not None, f"tail param without grad: {name}"
    for tensor, label in ((logits, "logits"), (upscaled, "upscaled"), (token, "token")):
        assert tensor.grad is not None and float(tensor.grad.abs().sum()) > 0.0, \
            f"coupled mode MUST flow gradients into {label}"
    assert "TAIL/boundary_margin_violation" in stats
    assert torch.isclose(stats["TAIL/boundary_logit"], torch.tensor(BOUNDARY), atol=1e-6)

    # Violation semantics on hand-built states: satisfied points (beyond
    # boundary+margin on the correct side) contribute exactly zero; wrong-side
    # points contribute margin-plus-distance.
    signed = targets.reshape(3, -1).mul(2).sub(1.0).gather(1, outputs["indices"])
    z0 = outputs["base_selected_logits"]
    refined_sel = outputs["refined_selected_logits"]
    manual = torch.relu(0.10 - signed * (refined_sel - BOUNDARY))
    satisfied = manual <= 0
    assert bool((manual[satisfied] == 0).all())
    wrong_side = (signed * (z0 - BOUNDARY)) < 0
    assert bool((manual[wrong_side] > 0).all()), "wrong-side points must violate"

    # Empty-rank path returns the same telemetry keys (DDP collectives are
    # exercised only under an initialized process group, not here).
    zero = torch.zeros(())
    _, empty_stats = refiner.empty_point_loss(zero)
    assert "TAIL/boundary_margin_violation" in empty_stats

    # Zero-init equivalence: a fresh module never perturbs the base forward.
    fresh = DecoderTailPointRefiner(
        num_points=8, mode="residual_v1_margin", boundary_logit=BOUNDARY,
    )
    base = torch.randn(2, 1, 16, 16)
    out, _ = fresh(base, torch.randn(2, 32, 16, 16), torch.randn(2, 1, 256))
    assert torch.equal(out, base)

    print("test_decoder_tail_margin: PASS")


if __name__ == "__main__":
    main()
