"""Regression: UDPR/DCR selected-point DDP reductions tolerate an empty rank.

Run with ``torchrun --standalone --nproc_per_node=2``.  Rank 0 supplies one
positive RoI; rank 1 follows the no-positive-RoI branch.  Both must complete
the identical collectives and report the same global telemetry for both A3
and A4/DCR modes.
"""
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rsprompter.decoder_tail_refiner import DecoderTailPointRefiner


def main() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    for mode in ("residual_v1", "confidence_gated"):
        module = DecoderTailPointRefiner(num_points=4, mode=mode)
        if rank == 0:
            logits = torch.tensor([[[[-1.0, 0.1], [0.1, 2.0]]]])
            refined, outputs = module(logits, torch.randn(1, 32, 2, 2), torch.randn(1, 1, 256))
            del refined
            _, stats = module.point_loss(outputs, torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]))
        else:
            _, stats = module.empty_point_loss(torch.zeros(()))
        fields = ["TAIL/selected_bce"]
        if mode == "confidence_gated":
            fields += ["TAIL/gate_bce", "TAIL/keep_loss", "TAIL/gate_mean"]
        for field in fields:
            gathered = [torch.zeros_like(stats[field]) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, stats[field])
            assert torch.allclose(gathered[0], gathered[1]), (mode, field, gathered)

    # A4's one-class edge case must use the same active-class denominator on
    # an empty rank as on the rank owning all selected points.
    dcr = DecoderTailPointRefiner(num_points=4, mode="confidence_gated")
    if rank == 1:
        logits = torch.full((1, 1, 2, 2), -1.0)
        _, outputs = dcr(logits, torch.randn(1, 32, 2, 2), torch.randn(1, 1, 256))
        _, stats = dcr.point_loss(outputs, torch.zeros((1, 2, 2)))
    else:
        _, stats = dcr.empty_point_loss(torch.zeros(()))
    gathered = [torch.zeros_like(stats["TAIL/gate_bce"]) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, stats["TAIL/gate_bce"])
    assert torch.allclose(gathered[0], gathered[1]), ("one-class", gathered)
    if rank == 0:
        print("UDPR/DCR DDP empty-rank collective smoke: PASS")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
