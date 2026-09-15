#!/usr/bin/env python
"""Wall-clock benchmark of the full-image target/support construction pair.

Isolates the two functions the fi contract (full_image + roi_balanced_dice)
runs every micro-step: `_MaskHeadTargetsMixin.get_full_image_targets` and
`get_final_mask_roi_support`.  The WHU A3 launch of 2026-09-15 measured
~9.5-10.9 s per bs2 micro-step against ~0.9 s/sample on NWPU matrix300; this
bench attributes how much of that gap these two CPU/serial sections own at
WHU densities (~48 instances/image, 64 positive RoIs/image).

Usage (cvt2 env):
    python scripts/probes/bench_full_image_targets.py [--iters 20] [--device cuda]
No model, data, or checkpoint is touched; inputs are synthetic.
"""
import argparse
import time
from types import SimpleNamespace

import numpy as np
import torch

from mmdet.structures.mask import BitmapMasks
from rsprompter.sam2_mask_head_targets import _MaskHeadTargetsMixin

MIXIN = _MaskHeadTargetsMixin
CLS = _MaskHeadTargetsMixin


def synthetic_image(num_gt: int, img_size: int, rng: np.random.Generator):
    """~WHU-density GT: many small building rectangles on a 1024 canvas."""
    masks = np.zeros((num_gt, img_size, img_size), dtype=np.uint8)
    boxes = []
    for i in range(num_gt):
        w = int(rng.integers(8, 60))
        h = int(rng.integers(8, 60))
        x1 = int(rng.integers(0, img_size - w))
        y1 = int(rng.integers(0, img_size - h))
        masks[i, y1 : y1 + h, x1 : x1 + w] = 1
        boxes.append([x1, y1, x1 + w, y1 + h])
    return BitmapMasks(masks, height=img_size, width=img_size), torch.as_tensor(
        boxes, dtype=torch.float32
    )


def build_inputs(device: torch.device, num_images: int, num_gt: int, num_pos: int):
    rng = np.random.default_rng(0)
    sampling_results, gt_instances = [], []
    for _ in range(num_images):
        gt_masks, gt_boxes = synthetic_image(num_gt, 1024, rng)
        # proposals: random perturbations of (a subset of) GT boxes
        idx = torch.randint(0, num_gt, (num_pos,))
        priors = gt_boxes[idx] + torch.randn(num_pos, 4) * 4.0
        priors[:, 2] = torch.clamp(priors[:, 2], min=1.0)
        priors[:, 3] = torch.clamp(priors[:, 3], min=1.0)
        priors[:, 2] = torch.max(priors[:, 2], priors[:, 0] + 2)
        priors[:, 3] = torch.max(priors[:, 3], priors[:, 1] + 2)
        priors = priors.to(device)
        sampling_results.append(
            SimpleNamespace(pos_assigned_gt_inds=idx.to(device), pos_priors=priors)
        )
        gt_instances.append(SimpleNamespace(masks=gt_masks))
    return sampling_results, gt_instances


def time_fn(fn, iters: int = 20, warmup: int = 3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--num-images", type=int, default=2, help="bs2 micro-batch")
    parser.add_argument("--num-gt", type=int, default=48, help="WHU avg instances")
    parser.add_argument("--num-pos", type=int, default=64, help="pos RoIs per image")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    sampling_results, gt_instances = build_inputs(
        device, args.num_images, args.num_gt, args.num_pos
    )
    dummy_self = SimpleNamespace(
        final_mask_loss_cfg=SimpleNamespace(roi_expand_ratio=1.2)
    )

    for grid in (256, 1024):
        target_size = (grid, grid)
        t_targets = time_fn(
            lambda: CLS.get_full_image_targets(
                dummy_self, sampling_results, gt_instances, target_size, device
            ),
            iters=args.iters,
        )
        t_support = time_fn(
            lambda: CLS.get_final_mask_roi_support(
                dummy_self, sampling_results, gt_instances, target_size, device
            ),
            iters=args.iters,
        )
        total = t_targets + t_support
        print(
            f"grid={grid:4d}  get_full_image_targets={t_targets*1e3:8.1f} ms  "
            f"get_final_mask_roi_support={t_support*1e3:8.1f} ms  "
            f"sum={total*1e3:8.1f} ms/micro-step"
        )


if __name__ == "__main__":
    main()
