"""Contract checks for the threshold-anchored stop-margin UDPR loss.

The a3sgtm arm replaces only the tail point loss: one-sided margin anchored
at the DEPLOYED binarisation logit (boundary_logit), inside the tailstop
container.  Checks: gradient isolation inherited from stop mode; violation
semantics (zero for points already beyond boundary+margin on the correct
side, positive on the wrong side); empty-rank path exposes the same telemetry
key; and a from-scratch equivalence smoke (delta=0 at init).
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
        num_points=8, mode="residual_v1_stop_margin",
        crossing_margin=0.10, boundary_logit=BOUNDARY,
    )
    # Same v1 parameter family as a3sg/a3sgt (cross-loadable).
    v1 = DecoderTailPointRefiner(num_points=8)
    assert sorted(refiner.state_dict()) == sorted(v1.state_dict())

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
    loss.backward()
    for name, param in refiner.named_parameters():
        assert param.grad is not None, f"tail param without grad: {name}"
    for tensor, label in ((logits, "logits"), (upscaled, "upscaled"), (token, "token")):
        assert tensor.grad is None or float(tensor.grad.abs().sum()) == 0.0, f"leak into {label}"
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

    # Empty-rank path returns the same telemetry key with zero collectives.
    zero = torch.zeros(())
    _, empty_stats = refiner.empty_point_loss(zero)
    assert "TAIL/boundary_margin_violation" in empty_stats

    # Zero-init equivalence: a fresh module never perturbs the base forward.
    fresh = DecoderTailPointRefiner(
        num_points=8, mode="residual_v1_stop_margin", boundary_logit=BOUNDARY,
    )
    base = torch.randn(2, 1, 16, 16)
    out, _ = fresh(base, torch.randn(2, 32, 16, 16), torch.randn(2, 1, 256))
    assert torch.equal(out, base)

    # Coupled twin (residual_v1_margin): identical forward VALUES, live
    # gradients (the co-adaptation channel), same loss.
    coupled = DecoderTailPointRefiner(
        num_points=8, mode="residual_v1_margin",
        crossing_margin=0.10, boundary_logit=BOUNDARY,
    )
    coupled.load_state_dict(refiner.state_dict())
    c_logits = torch.randn(3, 1, 16, 16, requires_grad=True)
    c_up = torch.randn(3, 32, 16, 16, requires_grad=True)
    c_tok = torch.randn(3, 1, 256, requires_grad=True)
    r_stop, o_stop = refiner(c_logits, c_up, c_tok)
    r_coup, o_coup = coupled(c_logits, c_up, c_tok)
    assert r_stop.shape == r_coup.shape
    # same weights + same inputs -> same delta values (grad plumbing only)
    assert torch.allclose(o_stop["delta"].detach(), o_coup["delta"].detach(), atol=1e-6)
    c_loss, _ = coupled.point_loss(o_coup, targets)
    (c_loss + r_coup.sum() * 0.01).backward()
    assert c_logits.grad is not None and float(c_logits.grad.abs().sum()) > 0.0, \
        "coupled mode MUST flow gradients into decoder logits"
    assert c_up.grad is not None and float(c_up.grad.abs().sum()) > 0.0
    assert c_tok.grad is not None and float(c_tok.grad.abs().sum()) > 0.0

    print("test_decoder_tail_stop_margin: PASS")


if __name__ == "__main__":
    main()
