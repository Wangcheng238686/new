#!/usr/bin/env python3
"""CPU invariants for the frozen tail-input readability probe."""
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from inference.probes.tail_input_readability_probe import _auc, _point_inputs, _target_for_final_mask_contract


def main():
    logits = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4) - 8
    upscaled = torch.ones(1, 32, 4, 4); tokens = torch.ones(1, 1, 256)
    base, patches = _point_inputs(logits, upscaled, tokens, torch.tensor([[5, 10]]))
    assert base.shape == (1, 2, 292), base.shape
    assert patches.shape == (1, 2, 18), patches.shape
    assert not torch.equal(patches[..., :9], patches[..., 9:])
    assert abs(_auc(torch.tensor([.1, .9]).numpy(), torch.tensor([0, 1]).numpy()) - 1.) < 1e-9
    class Head: final_mask_coordinate_mode = "full_image"
    gt = torch.tensor([[1., 0.], [0., 0.]])
    target = _target_for_final_mask_contract(Head(), gt, torch.tensor([0., 0., 1., 1.]), (4, 4))
    assert target.shape == (4, 4) and target[:2, :2].all() and not target[2:, 2:].any()
    print("PASS tail input readability probe invariants")


if __name__ == "__main__": main()
