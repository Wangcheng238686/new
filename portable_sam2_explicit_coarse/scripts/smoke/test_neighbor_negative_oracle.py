"""Selector accounting tests; no model, GPU or checkpoint required."""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from inference.probes.neighbor_negative_oracle_probe import select_neighbor_point


def main():
    masks = torch.zeros(2, 64, 64)
    masks[0, 20:40, 20:40] = 1
    masks[1, 20:40, 40:50] = 1
    box = torch.tensor([20., 20., 40., 40.])
    coords = torch.tensor([[25., 25.], [30., 30.], [19., 30.], [19., 19.]])
    labels = torch.tensor([1, 1, 0, 0])
    original = coords.clone()
    point = select_neighbor_point(coords, labels, box, masks, 0)
    assert point is not None
    x, y = point.floor().long()
    assert masks[1, y, x] and not masks[0, y, x]
    assert torch.equal(coords, original)
    assert torch.equal(point, select_neighbor_point(coords, labels, box, masks, 0))
    labels[3] = -1
    assert select_neighbor_point(coords, labels, box, masks, 0) is None
    labels[3] = 0
    masks[1] = masks[0]
    assert select_neighbor_point(coords, labels, box, masks, 0) is None
    masks[1].zero_()
    assert select_neighbor_point(coords, labels, box, masks, 0) is None
    masks[1, 60:64, 60:64] = 1
    assert select_neighbor_point(coords, labels, box, masks, 0) is None
    print("PASS: neighbor membership, target exclusion, immutable inputs, deterministic tie, invalid slot, overlap, absent/distant neighbor")


if __name__ == "__main__":
    main()
