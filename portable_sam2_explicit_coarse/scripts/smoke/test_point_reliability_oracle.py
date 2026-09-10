"""Membership/sign/invalid-slot tests for the frozen reliability probe."""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from inference.probes.point_reliability_oracle_probe import unreliable_points


def main():
    target = torch.zeros(10, 10)
    target[2:7, 2:7] = 1
    coords = torch.tensor([[3., 3.], [8., 8.], [4., 4.], [9., 9.]])
    labels = torch.tensor([1, 1, 0, 0])
    saved = coords.clone()
    assert unreliable_points(coords, labels, target).tolist() == [False, True, True, False]
    labels[1] = -1
    assert unreliable_points(coords, labels, target).tolist() == [False, False, True, False]
    assert torch.equal(saved, coords)
    coords[0] = torch.tensor([6.9, 6.9])
    assert not unreliable_points(coords, labels, target)[0]
    coords[0, 0] = 10
    try:
        unreliable_points(coords, labels, target)
    except RuntimeError:
        pass
    else:
        raise AssertionError("out-of-image coordinate accepted")
    print("PASS membership polarity, invalid preservation, no coordinate mutation, floor and bounds")


if __name__ == "__main__":
    main()
