"""Metric camera transforms with immutable physical BEV coordinates."""

import json
from pathlib import Path

import numpy as np


def load_camera(folder):
    folder = Path(folder)
    intr = json.loads((folder / "camera_intrinsics.json").read_text())
    extr = json.loads((folder / "camera_extrinsics.json").read_text())
    contract = extr.get("extrinsics_contract", {})
    expected = dict(
        version=2,
        vector_convention="column",
        camera_axes="x_right_y_down_z_forward",
        image_origin="top_left",
        world_units="cm",
        image_flip="none",
    )
    if any(contract.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Unsupported camera contract: {folder}")
    k = np.array(
        [[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]],
        dtype=np.float64,
    )
    e = np.array(extr["world_to_camera_cv"], dtype=np.float64)
    t = np.array(extr["camera_to_world_cv"], dtype=np.float64)
    if (
        e.shape != (4, 4)
        or t.shape != (4, 4)
        or not all(np.isfinite(a).all() for a in (k, e, t))
    ):
        raise ValueError(f"Invalid camera matrices: {folder}")
    if k[0, 0] <= 0 or k[1, 1] <= 0 or min(intr["height"], intr["width"]) <= 0:
        raise ValueError(f"Invalid camera intrinsics: {folder}")
    e[:3, 3] /= 100.0
    t[:3, 3] /= 100.0
    if not np.allclose(e @ t, np.eye(4), atol=1e-6):
        raise ValueError(f"Camera transforms are not inverses: {folder}")
    return k, e, t, (int(intr["height"]), int(intr["width"]))


def localize_camera(e, t, origin):
    e, t = e.copy(), t.copy()
    e[:3, 3] += e[:3, :3] @ origin
    t[:3, 3] -= origin
    return e, t


def backproject(depth_m, u, v, k, t):
    camera = np.column_stack(
        ((u - k[0, 2]) * depth_m / k[0, 0], (v - k[1, 2]) * depth_m / k[1, 1], depth_m)
    )
    return camera @ t[:3, :3].T + t[:3, 3]


def project(points, k, e):
    camera = points @ e[:3, :3].T + e[:3, 3]
    pixels = camera @ k.T
    return pixels[:, :2] / pixels[:, 2:3], camera[:, 2]
