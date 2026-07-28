#!/usr/bin/env python
import os
import sys
from pathlib import Path

os.environ.setdefault("RSPROMPTER_LIGHT_IMPORT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from rsprompter.dense_prompt_utils import paste_roi_to_full_canvas
from rsprompter.densebr import DenseBR
from rsprompter.shape_prior import ShapePointMiner, ShapePriorInjector


def main():
    torch.manual_seed(44)
    n = 3
    roi = torch.randn(n, 512, 14, 14, requires_grad=True)
    image = torch.randn(2, 256, 32, 32, requires_grad=True)
    boxes = torch.tensor(
        [[20.0, 30.0, 300.0, 350.0], [400.0, 100.0, 800.0, 600.0],
         [50.0, 500.0, 250.0, 900.0]]
    )
    roi_img_ids = torch.tensor([0, 1, 1])

    shape = ShapePriorInjector(
        img_size=1024,
        visual_context_dim=256,
        roi_feat_channels=512,
        fusion_type="gated_spatial_film",
        coarse_mask_output_size=64,
    )
    ctx = shape.forward_context_visual(image)
    raw, _, _, _ = shape.forward_roi(ctx, roi, boxes, roi_img_ids)
    assert raw.shape == (n, 1, 64, 64)

    densebr = DenseBR(roi_in_channels=512, image_size=1024)
    refined = densebr(roi, raw, boxes)
    assert refined.shape == raw.shape
    assert torch.equal(refined, raw), "zero-init DenseBR must start as identity"

    miner = ShapePointMiner(
        point_warmup_cfg=dict(
            enabled=False,
            no_point_epochs=0,
            one_pair_epochs=0,
            full_2p2n_start_epoch=1,
        )
    )
    coords, labels, _ = miner(refined, boxes, 1024)
    assert coords.shape == (n, 4, 2)
    assert labels.shape == (n, 4)

    canvas = paste_roi_to_full_canvas(refined, boxes, 128, 128, 1024)
    assert canvas.shape == (n, 1, 128, 128)
    canvas.mean().backward()
    assert roi.grad is not None
    assert image.grad is not None
    print("component smoke test: OK")


if __name__ == "__main__":
    main()
