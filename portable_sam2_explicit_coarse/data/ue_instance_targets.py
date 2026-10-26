"""Satellite-box center targets; never read measured geometry or UAV caches."""
import math

import numpy as np
import torch


def center_targets(boxes, device='cuda'):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    heat = np.zeros((1, 128, 128), np.float32)
    reg = np.zeros((4, 128, 128), np.float32)
    valid = np.zeros((1, 128, 128), np.float32)
    order = np.argsort(-np.prod(boxes[:, 2:] - boxes[:, :2], axis=1), kind='stable')
    for box in boxes[order]:
        wh = (box[2:] - box[:2]) / 4
        center = (box[:2] + box[2:]) / 8
        ix, iy = np.clip(np.floor(center).astype(int), 0, 127)
        sigma = float(np.clip(min(wh) / 6, 1, 4))
        radius = int(math.ceil(3 * sigma))
        x0, x1 = max(0, ix - radius), min(128, ix + radius + 1)
        y0, y1 = max(0, iy - radius), min(128, iy + radius + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        gaussian = np.exp(-((xx - ix)**2 + (yy - iy)**2) / (2 * sigma**2))
        heat[0, y0:y1, x0:x1] = np.maximum(heat[0, y0:y1, x0:x1], gaussian)
        reg[:2, iy, ix] = center - [ix, iy]
        reg[2:, iy, ix] = np.log(np.maximum(wh, 1e-3))
        valid[0, iy, ix] = 1
    return {name: torch.from_numpy(value)[None].to(device)
            for name, value in (('heat', heat), ('reg', reg), ('valid', valid))}
