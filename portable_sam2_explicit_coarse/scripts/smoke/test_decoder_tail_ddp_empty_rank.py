"""Regression: the margin tail's DDP reductions tolerate an empty rank.

Run with ``torchrun --standalone --nproc_per_node=2``.  Rank 0 supplies one
positive RoI; rank 1 follows the no-positive-RoI branch.  Both must execute
the identical collectives — margin mode calls ``_global_tail_bce`` exactly
twice on every rank (violation terms, then the detached BCE diagnostic) — and
report identical global telemetry.  A rank that skips or adds one collective
deadlocks here instead of in a training run.
"""
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rsprompter.decoder.decoder_tail_refiner import DecoderTailPointRefiner

# Fields produced on BOTH branches (key-set contract).  Only the DDP-global
# normalised fields are expected to be EQUAL across ranks; the rest are
# rank-local by design (counts, per-rank means).
SHARED_FIELDS = (
    "TAIL/selected_count", "TAIL/selected_pos", "TAIL/selected_bce",
    "TAIL/delta_abs", "TAIL/delta_max",
    "TAIL/selected_error_fraction", "TAIL/selected_error_coverage",
    "TAIL/changed_fraction", "TAIL/error_direction_agreement",
    "TAIL/correct_flip_fraction", "TAIL/destroy_fraction",
    "TAIL/boundary_margin_violation",
)
# Global normalised values: every rank must report the SAME number.
GLOBAL_FIELDS = ("TAIL/selected_bce", "TAIL/boundary_margin_violation")


def main() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()

    # Count collectives so a future edit cannot silently desynchronise ranks.
    collective_calls = {"count": 0}
    original_tail_bce = DecoderTailPointRefiner._global_tail_bce

    def counting_tail_bce(*args, **kwargs):
        collective_calls["count"] += 1
        return original_tail_bce(*args, **kwargs)

    DecoderTailPointRefiner._global_tail_bce = staticmethod(counting_tail_bce)
    try:
        module = DecoderTailPointRefiner(num_points=4, mode="residual_v1_margin")
        if rank == 0:
            logits = torch.tensor([[[[-1.0, 0.1], [0.1, 2.0]]]])
            refined, outputs = module(
                logits, torch.randn(1, 32, 2, 2), torch.randn(1, 1, 256)
            )
            del refined
            _, stats = module.point_loss(
                outputs, torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
            )
        else:
            _, stats = module.empty_point_loss(torch.zeros(()))

        assert collective_calls["count"] == 2, collective_calls
        counts = torch.tensor([collective_calls["count"]])
        dist.all_reduce(counts, op=dist.ReduceOp.MAX)
        assert int(counts.item()) == 2, counts

        for field in SHARED_FIELDS:
            assert field in stats, (rank, field)
        for field in GLOBAL_FIELDS:
            gathered = [torch.zeros_like(stats[field]) for _ in range(world)]
            dist.all_gather(gathered, stats[field])
            assert torch.allclose(gathered[0], gathered[1]), (field, gathered)
    finally:
        DecoderTailPointRefiner._global_tail_bce = staticmethod(original_tail_bce)

    if rank == 0:
        print("margin-tail DDP empty-rank collective smoke: PASS")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
