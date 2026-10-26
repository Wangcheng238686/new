"""Fine-pixel observed-surface competition before sparse coarse pooling."""

import numpy as np

from .cameras import project


def empty_projection():
    return dict(
        project_source=np.zeros(0, dtype=np.int64),
        project_pixel=np.zeros(0, dtype=np.int64),
        project_mass=np.zeros(0, dtype=np.float32),
        project_quality=np.zeros(0, dtype=np.float32),
    )


def satellite_footprints(reference, supported, confidence, k, e, t, config):
    size = config["image_size"]
    cell = config["side_m"] / config["grid_size"]
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * cell / 2
    source, pixels, depths = [], [], []
    inverse_k = np.linalg.inv(k)
    for index in np.flatnonzero(supported):
        quad = np.tile(reference[index], (4, 1))
        quad[:, :2] += corners
        polygon, z = project(quad, k, e)
        if (z <= 0).any() or not np.isfinite(polygon).all():
            continue
        lower = np.maximum(np.ceil(polygon.min(axis=0)).astype(int), 0)
        upper = np.minimum(np.floor(polygon.max(axis=0)).astype(int), size - 1)
        if (upper < lower).any():
            continue
        yy, xx = np.mgrid[lower[1] : upper[1] + 1, lower[0] : upper[0] + 1]
        xy = np.column_stack((xx.ravel(), yy.ravel()))
        edge = np.roll(polygon, -1, axis=0) - polygon
        delta = xy[:, None] - polygon[None]
        cross = edge[None, :, 0] * delta[..., 1] - edge[None, :, 1] * delta[..., 0]
        inside = (cross >= -1e-7).all(axis=1) | (cross <= 1e-7).all(axis=1)
        xy = xy[inside]
        if not len(xy):
            continue
        rays = np.column_stack((xy, np.ones(len(xy)))) @ inverse_k.T
        direction = rays @ t[:3, :3].T
        camera_z = (reference[index, 2] - t[2, 3]) / direction[:, 2]
        valid = np.isfinite(camera_z) & (camera_z > 0)
        source.append(np.full(valid.sum(), index, dtype=np.int64))
        pixels.append(xy[valid, 1] * size + xy[valid, 0])
        depths.append(camera_z[valid])
    if not source:
        return empty_projection(), dict(projected_pixels=0, rejected_fraction=0.0)
    source, pixels, depths = (
        np.concatenate(source),
        np.concatenate(pixels),
        np.concatenate(depths),
    )
    nearest = np.full(size * size, np.inf)
    np.minimum.at(nearest, pixels, depths)
    keep = depths <= nearest[pixels] + config["collision_band_m"]
    rejected = float(1 - keep.mean())
    source, pixels = source[keep], pixels[keep]
    multiplicity = np.bincount(pixels, minlength=size * size)
    mass = 1.0 / multiplicity[pixels]
    view_quality = np.minimum(confidence[source, 0] * 5 / 2, 1)
    height_quality = np.minimum(2 * confidence[source, 1], 1) / (
        1 + confidence[source, 2]
    )
    distance_quality = np.exp(-0.5 * np.square(confidence[source, 3]))
    projection = dict(
        project_source=source,
        project_pixel=pixels,
        project_mass=mass.astype(np.float32),
        project_quality=(view_quality * height_quality * distance_quality).astype(
            np.float32
        ),
    )
    return projection, dict(
        projected_pixels=int((multiplicity > 0).sum()), rejected_fraction=rejected
    )
