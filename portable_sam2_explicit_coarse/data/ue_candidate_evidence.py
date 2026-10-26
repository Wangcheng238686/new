"""Retained per-surface geometry, before any target-raster surface collapse."""
import json
from pathlib import Path

import numpy as np
import torch

from rsprompter.uav.local_adapter_probe import YAWS, camera_arrays


def candidate_geometry(cameras, index, anchor):
    """Seven attributes; original pixels and camera-Z are FP64 cache anchors.

    Channels: target offsets from stride4 SAT feature center /4 (two),
    world-Z minus sampling-origin Z /100m, source camera-Z /1000m,
    view yaw cosine/sine, SAT camera-Z above the cell's nearest anchor /10m.
    This preserves retained surfaces, not all source depth pixels or visibility.
    """
    index = np.asarray(index)
    anchor = np.asarray(anchor, dtype=np.float64)
    assert index.shape == (16384, 8) and anchor.shape == (16384, 8, 6)
    valid = index >= 0
    out = np.zeros((16384, 8, 7), np.float64)
    cells = np.arange(16384)
    origin = np.stack((cells % 128 * 4, cells // 128 * 4), -1)
    out[..., :2] = (anchor[..., 3:5] - origin[:, None]) / 4
    nearest = np.min(np.where(valid, anchor[..., 5], np.inf), axis=1)
    nearest[~valid.any(1)] = 0
    out[..., 6] = (anchor[..., 5] - nearest[:, None]) / 10
    for view, yaw in enumerate(YAWS):
        take = valid & (index // 16384 == view)
        if not take.any():
            continue
        u, v, z = anchor[take, :3].T
        k, _, transform = camera_arrays(cameras['uav'][f'yaw_{yaw:03d}'])
        point = np.stack(((u-k[0, 2])*z/k[0, 0], (v-k[1, 2])*z/k[1, 1], z), -1)
        world_z = point @ transform[2, :3] + transform[2, 3]
        out[take, 2] = (world_z - cameras['center_ue_cm'][2]/100) / 100
        out[take, 3] = z / 1000
        out[take, 4] = np.cos(np.deg2rad(yaw))
        out[take, 5] = np.sin(np.deg2rad(yaw))
        token = index[take] % 16384
        assert np.array_equal(token, (v//4).astype(int)*128+(u//4).astype(int))
    out[~valid] = 0
    assert np.isfinite(out).all()
    return out.astype(np.float32)


def load_candidate_geometry(data_root, ident, index, anchor):
    path = Path(data_root) / 'points' / f'{int(ident):06d}' / 'cameras.json'
    value = candidate_geometry(json.loads(path.read_text()), index.detach().cpu().numpy(),
                               anchor.detach().cpu().numpy())
    return torch.from_numpy(value).to(device=anchor.device)
