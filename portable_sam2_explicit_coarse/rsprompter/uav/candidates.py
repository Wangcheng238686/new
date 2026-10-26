"""Sparse world-space candidates, unique by (view, image token)."""

import numpy as np
from scipy.spatial import cKDTree

from .cameras import backproject


def token_anchors(depth, k, t, view, config):
    stride, size = config["source_stride"], config["image_size"]
    h = size // stride
    y, x = np.mgrid[:size, :size]
    valid = np.isfinite(depth) & (depth > 0) & (depth < 10000)
    world_z = (
        depth
        * (
            t[2, 0] * (x - k[0, 2]) / k[0, 0]
            + t[2, 1] * (y - k[1, 2]) / k[1, 1]
            + t[2, 2]
        )
        + t[2, 3]
    )
    blocks = (
        depth.reshape(h, stride, h, stride).transpose(0, 2, 1, 3).reshape(h * h, -1)
    )
    zblocks = (
        world_z.reshape(h, stride, h, stride).transpose(0, 2, 1, 3).reshape(h * h, -1)
    )
    vblocks = (
        valid.reshape(h, stride, h, stride).transpose(0, 2, 1, 3).reshape(h * h, -1)
    )
    ids = np.flatnonzero(vblocks.any(axis=1))
    if not len(ids):
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64)
    clean = np.where(vblocks[ids], zblocks[ids], np.nan)
    median, upper = np.nanmedian(clean, axis=1), np.nanpercentile(clean, 90, axis=1)
    anchors, tokens = [], []
    for target in (median, upper):
        distance = np.where(
            vblocks[ids], np.abs(zblocks[ids] - target[:, None]), np.inf
        )
        pick = distance.argmin(axis=1)
        u, v = (ids % h) * stride + pick % stride, (ids // h) * stride + pick // stride
        anchors.append(backproject(blocks[ids, pick], u, v, k, t))
        tokens.append(view * h * h + ids)
    return np.concatenate(anchors), np.concatenate(tokens)


def sparse_candidates(reference, anchor_groups, token_groups, config):
    q, budget = len(reference), config["candidates_per_view"]
    selected = np.full((q, 5 * budget), -1, dtype=np.int64)
    offset = 0
    nearest = np.full(q, config["radius_m"], dtype=np.float64)
    cell = config["side_m"] / config["grid_size"]
    for view, (anchors, tokens) in enumerate(zip(anchor_groups, token_groups)):
        if not len(anchors):
            continue
        tree = cKDTree(anchors[:, :2])
        # Retrieve the whole radius before ranking. A nearest-XY pre-cap can
        # exclude roof tokens behind a dense set of facade anchors.
        neighborhoods = tree.query_ball_point(reference[:, :2], config["radius_m"])
        chosen = np.full((q, budget), -1, dtype=np.int64)
        for row, neighborhood in enumerate(neighborhoods):
            if not neighborhood:
                continue
            indices = np.asarray(neighborhood, dtype=np.int64)
            delta = anchors[indices] - reference[row]
            distance = np.linalg.norm(delta[:, :2], axis=1)
            nearest[row] = min(nearest[row], distance.min())
            core = (np.abs(delta[:, :2]) <= cell / 2).all(axis=1)
            core &= np.abs(delta[:, 2]) <= config["height_bin_m"]
            score = distance / config["radius_m"] + 0.1 * np.minimum(
                np.abs(delta[:, 2]) / 10, 3
            )
            score -= core.astype(float) * 10
            order = np.lexsort((indices, score))
            ranked = indices[order]
            # The best compatible anchor owns its token's single vote.
            _, first = np.unique(tokens[ranked], return_index=True)
            unique = ranked[np.sort(first)[:budget]]
            chosen[row, : len(unique)] = unique + offset
        selected[:, view * budget : (view + 1) * budget] = chosen
        offset += len(anchors)
    return selected, nearest / config["radius_m"]
