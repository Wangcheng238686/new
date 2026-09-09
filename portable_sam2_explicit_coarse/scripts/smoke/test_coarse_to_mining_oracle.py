"""Pure tensor checks for the frozen coarse-to-mining Oracle replacement rule."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from inference.probes.coarse_to_mining_oracle_probe import (
    build_gt_coarse_candidate,
    select_consumer_outputs,
)


def main() -> None:
    raw = torch.tensor([
        [[[0.1, -0.2], [0.3, -0.4]]],
        [[[0.5, 0.6], [0.7, 0.8]]],
    ])
    ids = torch.tensor([0, 0])
    boxes = torch.tensor([[0.0, 0.0, 4.0, 4.0], [0.0, 0.0, 4.0, 4.0]])
    labels = torch.tensor([1, 2])
    gt_masks = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])
    candidate, stats = build_gt_coarse_candidate(
        raw, ids, boxes, labels,
        [(torch.tensor([[0.0, 0.0, 4.0, 4.0]]), torch.tensor([1]), gt_masks)],
        match_iou=0.5, logit_span=8.0,
    )
    # Same-label, IoU-qualified row becomes a signed exact GT coarse target.
    assert torch.equal(candidate[0, 0], torch.tensor([[8.0, -8.0], [-8.0, -8.0]]))
    # Different-label row is exactly raw: unmatched proposals never borrow GT.
    assert torch.equal(candidate[1], raw[1])
    assert stats == {"matched_rois": 1, "unmatched_rois": 1, "replaced_pixels": 4}

    # An overlapping GT from the wrong class cannot create a replacement.
    unchanged, wrong_stats = build_gt_coarse_candidate(
        raw[:1], ids[:1], boxes[:1], torch.tensor([9]),
        [(torch.tensor([[0.0, 0.0, 4.0, 4.0]]), torch.tensor([1]), gt_masks)],
        match_iou=0.5, logit_span=8.0,
    )
    assert torch.equal(unchanged, raw[:1])
    assert wrong_stats["matched_rois"] == 0 and wrong_stats["unmatched_rois"] == 1

    # The pre-registered cells are consumer-exclusive: points never replaces
    # the dense input, dense never replaces the point-miner input, both does
    # both, and the coarse loss side output remains exactly the original object.
    coarse_outputs = {"raw_logits": raw}
    raw_canvas, candidate_canvas = torch.zeros(2, 1, 3, 3), torch.ones(2, 1, 3, 3)
    raw_valid, candidate_valid = torch.zeros(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool)
    for mode, expected_points, expected_canvas, expected_valid in (
        ("raw", raw, raw_canvas, raw_valid),
        ("points", candidate, raw_canvas, raw_valid),
        ("dense", raw, candidate_canvas, candidate_valid),
        ("both", candidate, candidate_canvas, candidate_valid),
    ):
        point_input, returned_outputs, dense_input, dense_valid = select_consumer_outputs(
            mode, raw, coarse_outputs, raw_canvas, raw_valid,
            candidate, candidate_canvas, candidate_valid,
        )
        assert point_input is expected_points
        assert returned_outputs is coarse_outputs
        assert dense_input is expected_canvas
        assert dense_valid is expected_valid
    print("coarse-to-mining Oracle tensor smoke: PASS")


if __name__ == "__main__":
    main()
