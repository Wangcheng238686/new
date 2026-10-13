#!/usr/bin/env python
"""raw_logits transform and canvas-restore tests (a3sgm-only cleanup)."""
import os
import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("RSPROMPTER_LIGHT_IMPORT", "1")

from rsprompter.cspm.dense_prompt_utils import (  # noqa: E402
    paste_roi_to_full_canvas,
    replace_invalid_dense_with_base,
    transform_coarse_prompt,
)


class DensePromptUtilsTest(unittest.TestCase):
    def test_raw_detach_is_an_explicit_gradient_control(self):
        logits = torch.randn(2, 1, 4, 4, requires_grad=True)
        attached = transform_coarse_prompt(
            logits, "raw_logits", 1.0, 1.0, (-8.0, 8.0)
        )
        detached = transform_coarse_prompt(
            logits,
            "raw_logits",
            1.0,
            1.0,
            (-8.0, 8.0),
            detach_input=True,
        )
        self.assertTrue(attached.requires_grad)
        self.assertFalse(detached.requires_grad)

    def test_paste_places_box_region_and_fills_outside(self):
        logits = torch.full((1, 1, 4, 4), 3.0)
        boxes = torch.tensor([[256.0, 256.0, 768.0, 768.0]])
        canvas = paste_roi_to_full_canvas(
            logits, boxes, mask_h=64, mask_w=64, image_size=1024,
            outside_fill_logit=0.0,
        )
        self.assertEqual(tuple(canvas.shape), (1, 1, 64, 64))
        # 中心格来自 ROI（3.0），画布角落是 outside fill（0.0）
        self.assertAlmostEqual(float(canvas[0, 0, 32, 32]), 3.0, places=5)
        self.assertEqual(float(canvas[0, 0, 0, 0]), 0.0)

    def test_invalid_roi_restores_exact_no_mask_base(self):
        base = torch.randn(2, 256, 4, 4)
        encoded = torch.randn_like(base)
        mixed = replace_invalid_dense_with_base(
            encoded, base, torch.tensor([True, False])
        )
        self.assertTrue(torch.equal(mixed[0], encoded[0]))
        self.assertTrue(torch.equal(mixed[1], base[1]))


if __name__ == "__main__":
    unittest.main()
