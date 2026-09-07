"""Regression: UDPR selected-point DDP reductions tolerate an empty rank.

Run with ``torchrun --standalone --nproc_per_node=2``.  Rank 0 supplies one
positive RoI; rank 1 follows the no-positive-RoI branch.  Both must complete
the identical two collectives and report the same global BCE telemetry.
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
    module = DecoderTailPointRefiner(num_points=4)
    if rank == 0:
        logits = torch.tensor([[[[-1.0, 0.1], [0.1, 2.0]]]])
        refined, outputs = module(logits, torch.randn(1, 32, 2, 2), torch.randn(1, 1, 256))
        del refined
        _, stats = module.point_loss(outputs, torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]))
    else:
        _, stats = module.empty_point_loss(torch.zeros(()))
    gathered = [torch.zeros_like(stats["TAIL/selected_bce"]) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, stats["TAIL/selected_bce"])
    assert torch.allclose(gathered[0], gathered[1]), gathered
    if rank == 0:
        print("UDPR DDP empty-rank collective smoke: PASS")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
