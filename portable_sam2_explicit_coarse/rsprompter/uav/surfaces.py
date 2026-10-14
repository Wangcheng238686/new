"""Observed single-surface statistics; attention context is not coverage."""

import numpy as np


def observed_surfaces(points, views, config):
    n = config["grid_size"]
    cell_m = config["side_m"] / n
    ij = np.floor((points[:, :2] + config["side_m"] / 2) / cell_m).astype(np.int64)
    keep = ((ij >= 0) & (ij < n)).all(axis=1)
    points, views, ij = points[keep], views[keep], ij[keep]
    flat = ij[:, 1] * n + ij[:, 0]
    count = np.bincount(flat, minlength=n * n)
    axis = (np.arange(n) + 0.5) * cell_m - config["side_m"] / 2
    x, y = np.meshgrid(axis, axis)
    reference = np.column_stack((x.ravel(), y.ravel(), np.zeros(n * n)))
    height_valid = np.zeros(n * n, dtype=bool)
    confidence = np.zeros((n * n, 4), dtype=np.float32)
    if not len(points):
        return reference, count > 0, height_valid, confidence
    zbin = np.floor(points[:, 2] / config["height_bin_m"]).astype(np.int64)
    pairs, sizes = np.unique(np.column_stack((flat, zbin)), axis=0, return_counts=True)
    top = np.full(n * n, np.iinfo(np.int64).min, dtype=np.int64)
    good = sizes >= config["min_height_points"]
    np.maximum.at(top, pairs[good, 0], pairs[good, 1])
    in_top = zbin == top[flat]
    upper_count = np.bincount(flat[in_top], minlength=n * n)
    upper_sum = np.bincount(flat[in_top], weights=points[in_top, 2], minlength=n * n)
    upper_square = np.bincount(
        flat[in_top], weights=points[in_top, 2] ** 2, minlength=n * n
    )
    height_valid = upper_count >= config["min_height_points"]
    reference[:, 2] = upper_sum / np.maximum(upper_count, 1)
    variance = upper_square / np.maximum(upper_count, 1) - reference[:, 2] ** 2
    bits = np.zeros(n * n, dtype=np.uint8)
    np.bitwise_or.at(bits, flat[in_top], (1 << views[in_top]).astype(np.uint8))
    view_count = np.array([bin(int(v)).count("1") for v in bits])
    confidence[:, 0] = view_count / 5.0
    confidence[:, 1] = upper_count / np.maximum(count, 1)
    confidence[:, 2] = np.sqrt(np.maximum(variance, 0)) / config["height_bin_m"]
    return reference, count > 0, height_valid, confidence
