"""Cache only immutable depth/camera geometry, never trainable features."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np

from rsprompter.uav.cameras import load_camera
from rsprompter.uav.contract import geometry_signature
from rsprompter.uav.geometry import build_geometry

YAWS = (0, 72, 144, 216, 288)


def read_depth(path, size):
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"Unreadable depth: {path}")
    depth = raw[..., 2] if raw.ndim == 3 else raw
    if depth.dtype != np.float32 or depth.shape != (size, size):
        raise ValueError(f"Unexpected camera-Z representation: {path}")
    return depth.astype(np.float64) / 100


def validate_depth_floor(depth, camera, floor_m):
    """Verify the lower world-Z bound used for conservative split frustums."""
    if floor_m is None or not np.isfinite(floor_m):
        raise ValueError("UE spatial split requires a finite world-Z floor")
    k, _, t, _ = camera
    y, x = np.mgrid[: depth.shape[0], : depth.shape[1]]
    valid = np.isfinite(depth) & (depth > 0) & (depth < 10000)
    z = (
        depth[valid]
        * (
            t[2, 0] * (x[valid] - k[0, 2]) / k[0, 0]
            + t[2, 1] * (y[valid] - k[1, 2]) / k[1, 1]
            + t[2, 2]
        )
        + t[2, 3]
    )
    if len(z) and z.min() < floor_m - 1e-3:
        raise ValueError(
            "Observed depth is below the spatial split floor; rebuild the manifest"
        )


def geometry_for_sample(satellite, uav, origin, config, cache_root, floor_m=None):
    views = [uav / "oblique" / f"view_yaw_{yaw:03d}" for yaw in YAWS]
    identity = [geometry_signature(config), list(map(float, origin)), floor_m]
    for folder in [satellite, *views]:
        for filename in ("camera_intrinsics.json", "camera_extrinsics.json"):
            identity.append((folder / filename).read_text())
    for folder in views:
        path = folder / "depth_camera_z.0000.exr"
        stat = path.stat()
        identity.append([str(path.resolve()), stat.st_size, stat.st_mtime_ns])
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = Path(cache_root) / f"{key}.npz" if cache_root else None
    if cache is not None and cache.is_file():
        with np.load(cache, allow_pickle=False) as data:
            return {name: data[name].copy() for name in data.files}
    cameras = [load_camera(folder) for folder in views]
    depths = [
        read_depth(folder / "depth_camera_z.0000.exr", config["image_size"])
        for folder in views
    ]
    for depth, camera in zip(depths, cameras):
        validate_depth_floor(depth, camera, floor_m)
    arrays, _ = build_geometry(depths, cameras, load_camera(satellite), origin, config)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            dir=cache.parent, suffix=".npz", delete=False
        )
        temporary = Path(handle.name)
        try:
            with handle:
                np.savez_compressed(handle, **arrays)
            os.replace(temporary, cache)
        finally:
            temporary.unlink(missing_ok=True)
    return arrays
