#!/usr/bin/env python
import os
import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("RSPROMPTER_LIGHT_IMPORT", "1")

from rsprompter.dense_prompt_utils import (  # noqa: E402
    gaussian_prompt_from_roi_coarse,
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

    def test_gaussian_maps_roi_center_and_area_to_full_canvas(self):
        logits = torch.full((1, 1, 8, 8), -20.0, requires_grad=True)
        with torch.no_grad():
            logits[:, :, 2:6, 2:6] = 20.0
        boxes = torch.tensor([[256.0, 128.0, 768.0, 640.0]])
        prompt, valid, stats = gaussian_prompt_from_roi_coarse(
            logits,
            boxes,
            mask_h=256,
            mask_w=256,
            image_size=(1024, 1024),
            foreground_threshold=0.5,
            omega=15.0,
            gamma=4.0,
        )
        self.assertFalse(prompt.requires_grad)
        self.assertEqual(valid.tolist(), [True])
        flat_index = int(prompt[0, 0].argmax().item())
        peak_y, peak_x = divmod(flat_index, 256)
        self.assertEqual((peak_y, peak_x), (88, 120))
        self.assertAlmostEqual(float(prompt.max()), 15.0, places=5)
        self.assertAlmostEqual(float(stats["foreground_area_ratio"]), 0.25, places=6)
        self.assertAlmostEqual(float(stats["canvas_area"]), 4096.0, places=4)

    def test_anisotropic_box_still_produces_image_space_isotropic_gaussian(self):
        logits = torch.full((1, 1, 8, 8), -20.0)
        logits[:, :, 2:6, 2:6] = 20.0
        boxes = torch.tensor([[256.0, 256.0, 768.0, 512.0]])
        prompt, valid, _ = gaussian_prompt_from_roi_coarse(
            logits,
            boxes,
            mask_h=256,
            mask_w=256,
            image_size=1024,
        )
        self.assertTrue(bool(valid[0]))
        peak = prompt[0, 0]
        peak_index = int(peak.argmax())
        peak_y, peak_x = divmod(peak_index, 256)
        self.assertAlmostEqual(
            float(peak[peak_y, peak_x + 4]),
            float(peak[peak_y + 4, peak_x]),
            places=5,
        )

    def test_empty_coarse_returns_invalid_zero_prompt(self):
        logits = torch.full((2, 1, 8, 8), -20.0, requires_grad=True)
        boxes = torch.tensor(
            [[0.0, 0.0, 512.0, 512.0], [512.0, 512.0, 1024.0, 1024.0]]
        )
        prompt, valid, stats = gaussian_prompt_from_roi_coarse(
            logits,
            boxes,
            mask_h=256,
            mask_w=256,
            image_size=1024,
        )
        self.assertFalse(prompt.requires_grad)
        self.assertEqual(valid.tolist(), [False, False])
        self.assertEqual(int(torch.count_nonzero(prompt)), 0)
        self.assertEqual(float(stats["valid_ratio"]), 0.0)

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
