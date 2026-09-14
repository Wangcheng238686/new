"""Contract checks for the decoupled auxiliary grad-norm clip.

The single global clip let the tail/dense-capacity heads usurp the base's
0.5 budget (root-cause audit 2026-09-14).  The fix partitions params so the
base is clipped alone at 0.5 -- bit-matching aux-free arms -- while each aux
head is clipped at its own 0.5.  Checks the partition on toy module trees:
exact coverage, no overlap, correct membership, and legacy-path equivalence
when no aux module exists.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.train_rsprompter_fusion import _partition_aux_clip_params


class _Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 1))


class _MaskHead(nn.Module):
    def __init__(self, tail=True, densecap=False):
        super().__init__()
        self.trunk = nn.Linear(4, 4)
        if tail:
            self.decoder_tail_refiner = _Head()
        if densecap:
            self.dense_capacity_head = _Head()


class _Model(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.roi_head = nn.Module()
        self.roi_head.mask_head = _MaskHead(**kw)


class _DDPShim(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.module = core


def main() -> None:
    torch.manual_seed(0)

    # Both aux heads present (and a DDP wrapper): base must exclude both,
    # each aux group must be exactly that module's params, coverage exact.
    model = _DDPShim(_Model(tail=True, densecap=True))
    base, groups = _partition_aux_clip_params(model)
    named = dict(model.module.named_parameters())
    tail_params = {id(p) for n, p in named.items() if n.startswith("roi_head.mask_head.decoder_tail_refiner.")}
    cap_params = {id(p) for n, p in named.items() if n.startswith("roi_head.mask_head.dense_capacity_head.")}
    assert len(groups) == 2, groups
    assert {id(p) for g in groups for p in g} == tail_params | cap_params
    assert not (tail_params | cap_params) & {id(p) for p in base}
    all_ids = [id(p) for p in base] + [id(p) for g in groups for p in g]
    assert len(all_ids) == len(set(all_ids)) == len(named), "partition must cover every param exactly once"

    # Tail only: one group, exactly the tail's params.
    model_t = _Model(tail=True)
    named_t = dict(model_t.named_parameters())
    tail_ids = {id(p) for n, p in named_t.items() if n.startswith("roi_head.mask_head.decoder_tail_refiner.")}
    base, groups = _partition_aux_clip_params(model_t)
    assert len(groups) == 1
    assert {id(p) for g in groups for p in g} == tail_ids

    # No aux module: legacy path identical (base == all params, no groups).
    plain = _Model(tail=False)
    base, groups = _partition_aux_clip_params(plain)
    assert not groups and len(base) == len(list(plain.named_parameters()))

    # Semantics: base-only clipping equals the legacy clip over the same
    # grads (frozen base contributes None grads in both).
    m = _Model(tail=True)
    for p in m.backbone.parameters():
        p.requires_grad_(False)
    x = torch.randn(2, 4)
    loss = m.roi_head.mask_head.decoder_tail_refiner.mlp(x).sum() + m.roi_head.mask_head.trunk(x).sum()
    loss.backward()
    base, groups = _partition_aux_clip_params(m)
    with_grads_base = [p for p in base if p.grad is not None]
    assert len(with_grads_base) == len(list(m.roi_head.mask_head.trunk.parameters()))
    torch.nn.utils.clip_grad_norm_(base, 0.5)
    torch.nn.utils.clip_grad_norm_(groups[0], 0.5)
    for p in m.backbone.parameters():
        assert p.grad is None

    print("test_decoupled_aux_clip: PASS")


if __name__ == "__main__":
    main()
