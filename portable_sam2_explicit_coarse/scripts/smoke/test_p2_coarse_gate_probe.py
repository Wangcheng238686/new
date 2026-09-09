#!/usr/bin/env python3
"""CPU regression checks for the first-gate accounting and bootstrap scope."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from inference.probes.p2_coarse_gate_probe import (  # noqa: E402
    _box_support, _counts, _gate, _metric_from_counts, _roi_target_grid,
    paired_image_bootstrap,
)
from rsprompter.dense_prompt_utils import paste_roi_to_full_canvas  # noqa: E402


def record(image_id: int, raw_inter: int, refined_inter: int) -> dict:
    row = {"image_id": image_id, "matched_rois": 1, "unmatched_rois": 0,
           "support_valid_rois": 1, "rejected_support_rois": 0,
           "canvas_changed_pixels": 2, "canvas_support_pixels": 4}
    # fixed foreground/reference cardinalities: refined IoU wins on every image
    for prefix, inter in (("coarse_raw_", raw_inter), ("coarse_refined_", refined_inter),
                          ("canvas_raw_", raw_inter), ("canvas_refined_", refined_inter)):
        row.update({prefix + "inter": inter, prefix + "pred": 4, prefix + "target": 4,
                    prefix + "union": 8 - inter, prefix + "pixels": 4})
    return row


def main() -> int:
    # Fractional box: probe target must match training's floor/ceil crop.
    mask = torch.arange(36, dtype=torch.float32).reshape(6, 6) % 2
    box = torch.tensor([1.2, 1.8, 4.1, 4.2])
    actual = _roi_target_grid(mask, box, (3, 4))
    expected = F.interpolate(mask[1:5, 1:5][None, None], size=(3, 4), mode="nearest")[0, 0].bool()
    assert torch.equal(actual, expected)
    # Fractional support must equal the production floor/ceil paste domain,
    # rather than a pixel-centre approximation.
    support = _box_support(box, 6, 6, 6, torch.device("cpu"))
    canvas = paste_roi_to_full_canvas(torch.full((1, 1, 2, 2), 2.0), box[None], 6, 6, 6, outside_fill_logit=0.0)[0, 0]
    assert torch.equal(support, canvas.ne(0)), (support, canvas)
    counts = _counts(torch.tensor([[1, 0]], dtype=torch.bool), torch.tensor([[1, 1]], dtype=torch.bool))
    assert _metric_from_counts(counts, "iou") == .5 and _metric_from_counts(counts, "dice") == 2 / 3
    records = [record(1, 1, 3), record(2, 1, 3), record(3, 1, 3)]
    boot = paired_image_bootstrap(records, resamples=100, seed=44)
    assert boot["ci"]["coarse_delta_iou"]["low"] > 0
    assert _gate(boot)["verdict"] == "pass"
    no_change = paired_image_bootstrap([{**records[0], "canvas_changed_pixels": 0}], resamples=10, seed=44)
    assert _gate(no_change)["verdict"] == "fail"
    assert _gate(paired_image_bootstrap(records, resamples=0, seed=44))["verdict"] == "not_decided"
    print("PASS: p2 coarse gate accounting / image bootstrap / gate rule")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
