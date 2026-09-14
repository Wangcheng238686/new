"""Gradient-isolation contract for the tailstop (residual_v1_stop) UDPR mode.

The tailstop arm keeps the v1 module, selection and losses byte-identical; the
only contract is plumbing: NO gradient may flow from any tail-side loss into
the decoder outputs (logits / upscaled feature / mask token), while the tail's
own parameters must still receive gradients.  The base's final-mask loss is
routed to the unrefined logits in sam2_mask_head (asserted by
verify_p2v2_arms.py), so the checks here close the module-level guarantee.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rsprompter.decoder_tail_refiner import DecoderTailPointRefiner


def main() -> None:
    torch.manual_seed(0)
    stop = DecoderTailPointRefiner(num_points=8, mode="residual_v1_stop")
    v1 = DecoderTailPointRefiner(num_points=8)
    # Same parameter names (and shapes) as v1: an a3sg checkpoint stays
    # loadable for mode-switch probes.
    assert sorted(stop.state_dict()) == sorted(v1.state_dict())
    stop.load_state_dict(v1.state_dict())
    # The v1 MLP zero-initializes its last layer, so at step 0 only that layer
    # receives gradients (by design).  Perturb it so this test can assert that
    # gradients reach EVERY tail parameter through the stop-mode plumbing.
    with torch.no_grad():
        for module in (stop, v1):
            module.mlp[-1].weight.add_(0.1 * torch.randn_like(module.mlp[-1].weight))
            module.mlp[-1].bias.add_(0.05)
    stop.load_state_dict(v1.state_dict())

    logits = torch.randn(3, 1, 16, 16, requires_grad=True)
    upscaled = torch.randn(3, 32, 16, 16, requires_grad=True)
    token = torch.randn(3, 1, 256, requires_grad=True)

    refined_stop, outputs_stop = stop(logits, upscaled, token)
    refined_v1, _ = v1(logits, upscaled, token)
    # Stop mode changes gradient plumbing only: forward values must be
    # bit-identical to v1 under the same weights (same selection, same delta).
    assert torch.equal(refined_stop, refined_v1)
    assert torch.equal(outputs_stop["indices"].detach(), v1._select(logits)[0])

    targets = (torch.rand(3, 1, 16, 16) > 0.5).float()
    loss, stats = stop.point_loss(outputs_stop, targets)
    assert torch.isfinite(loss)
    # Adversarial extra term: a loss placed DIRECTLY on the refiner's output
    # must still not reach the decoder state (this is the channel that made
    # v1 pollute the shared trunk).
    total = loss + refined_stop.sum() * 0.01
    total.backward()

    for name, param in stop.named_parameters():
        assert param.grad is not None, f"tail param without grad: {name}"
        assert float(param.grad.abs().sum()) > 0.0, f"tail grad vanished: {name}"
    for tensor, label in ((logits, "decoder logits"), (upscaled, "upscaled feature"),
                          (token, "mask token")):
        assert tensor.grad is None or float(tensor.grad.abs().sum()) == 0.0, (
            f"gradient leaked into {label}"
        )

    # v1 (non-stop) control on identical inputs: the same losses DO reach the
    # decoder state, so this test can distinguish the modes rather than
    # trivially passing on an inert module.
    v1_logits = logits.detach().clone().requires_grad_(True)
    v1_upscaled = upscaled.detach().clone().requires_grad_(True)
    v1_token = token.detach().clone().requires_grad_(True)
    refined_control, outputs_control = v1(v1_logits, v1_upscaled, v1_token)
    loss_control, _ = v1.point_loss(outputs_control, targets)
    (loss_control + refined_control.sum() * 0.01).backward()
    assert float(v1_logits.grad.abs().sum()) > 0.0
    assert float(v1_upscaled.grad.abs().sum()) > 0.0
    assert float(v1_token.grad.abs().sum()) > 0.0

    # Zero-init contract is inherited from v1: at initialization delta == 0,
    # so an a3sgt run starts from the exact A0-SG forward.
    fresh = DecoderTailPointRefiner(num_points=8, mode="residual_v1_stop")
    base = torch.randn(2, 1, 16, 16)
    out, _ = fresh(base, torch.randn(2, 32, 16, 16), torch.randn(2, 1, 256))
    assert torch.equal(out, base)

    print("test_decoder_tail_stop: PASS")


if __name__ == "__main__":
    main()
