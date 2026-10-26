"""CPU geometry precomputation shared by training and checkpoint inference."""

import numpy as np

from .cameras import backproject, localize_camera
from .candidates import sparse_candidates, token_anchors
from .contract import geometry_config
from .footprints import satellite_footprints
from .surfaces import observed_surfaces


def build_geometry(depths, cameras, satellite_camera, origin, config):
    config = geometry_config(config)
    points, point_views, anchors, tokens = [], [], [], []
    size, step = config["image_size"], config["depth_sample_stride"]
    if len(depths) != 5 or len(cameras) != 5:
        raise ValueError("Exactly five calibrated RGB-D views are required")
    for view, (depth, camera) in enumerate(zip(depths, cameras)):
        k, e, t, shape = camera
        if shape != (size, size) or depth.shape != shape:
            raise ValueError("Depth/RGB/camera size differs from the geometry contract")
        _, t = localize_camera(e, t, origin)
        y, x = np.mgrid[step // 2 : size : step, step // 2 : size : step]
        z = depth[y, x]
        valid = np.isfinite(z) & (z > 0) & (z < 10000)
        world = backproject(z[valid], x[valid], y[valid], k, t)
        points.append(world)
        point_views.append(np.full(len(world), view, dtype=np.int64))
        position, identity = token_anchors(depth, k, t, view, config)
        anchors.append(position)
        tokens.append(identity)
    reference, observed, height_valid, confidence = observed_surfaces(
        np.concatenate(points), np.concatenate(point_views), config
    )
    candidates, distance = sparse_candidates(reference, anchors, tokens, config)
    confidence[:, 3] = distance
    k, e, t, shape = satellite_camera
    if shape != (size, size):
        raise ValueError("Satellite camera size differs from the geometry contract")
    e, t = localize_camera(e, t, origin)
    available = (candidates >= 0).any(axis=1)
    projection, stats = satellite_footprints(
        reference, observed & height_valid & available, confidence, k, e, t, config
    )
    payload = dict(
        anchor_xyz=np.concatenate(anchors).astype(np.float32),
        anchor_token=np.concatenate(tokens),
        candidate_index=candidates,
        reference_xyz=reference.astype(np.float32),
        observation_support=observed,
        height_valid=height_valid,
        bev_confidence=confidence,
        **projection
    )
    stats.update(
        candidate_fraction=float((candidates >= 0).any(axis=1).mean()),
        observation_fraction=float(observed.mean()),
        height_fraction=float(height_valid.mean()),
    )
    return payload, stats
