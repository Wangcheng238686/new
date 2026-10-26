"""Pure camera/depth to candidate and native-raster conversion.

No file I/O. A reference-plane control must recompute BOTH products, rather
than retaining measured-depth correspondences, support, or visibility counts.
"""
import numpy as np

from .local_adapter_probe import YAWS, camera_arrays


def reference_plane_depth(cameras, relative_z_m=0.):
    """Intersect source rays with a fixed horizontal sampling-reference plane.

    relative_z_m is declared before results; zero is the rendering sampling
    origin, not a ground surface inferred from labels or measured depth.
    """
    assert np.isfinite(relative_z_m)
    plane = cameras['center_ue_cm'][2] / 100 + relative_z_m
    yy, xx = np.mgrid[:512, :512]
    depths = []
    for yaw in YAWS:
        k, _, t = camera_arrays(cameras['uav'][f'yaw_{yaw:03d}'])
        rays = np.stack(((xx - k[0, 2]) / k[0, 0],
                         (yy - k[1, 2]) / k[1, 1], np.ones_like(xx)), -1)
        world_rays = rays @ t[:3, :3].T
        divisor = world_rays[..., 2]
        z = np.full((512, 512), np.nan, dtype=np.float64)
        np.divide(plane - t[2, 3], divisor, out=z, where=np.abs(divisor) > 1e-12)
        z[~np.isfinite(z) | (z <= 0) | (z >= 10000)] = np.nan
        depths.append(z)
    return np.stack(depths)


def projected(cameras, depths, step):
    assert depths.shape == (5, 512, 512)
    sk, se, _ = camera_arrays(cameras['satellite'])
    yy, xx = np.mgrid[(1 if step == 2 else 0):512:step,
                      (1 if step == 2 else 0):512:step]
    xx, yy = xx.ravel(), yy.ravel()
    for view, yaw in enumerate(YAWS):
        z = depths[view, yy, xx].astype(np.float64)
        good = np.isfinite(z) & (z > 0) & (z < 10000)
        u, v, z = xx[good], yy[good], z[good]
        k, _, t = camera_arrays(cameras['uav'][f'yaw_{yaw:03d}'])
        cam = np.stack(((u - k[0, 2]) * z / k[0, 0],
                        (v - k[1, 2]) * z / k[1, 1], z), 1)
        world = cam @ t[:3, :3].T + t[:3, 3]
        sat = world @ se[:3, :3].T + se[:3, 3]
        pix = sat @ sk.T
        uv = pix[:, :2] / pix[:, 2:]
        ok = (sat[:, 2] > 0) & np.isfinite(uv).all(1) & (uv >= 0).all(1) & (uv < 512).all(1)
        yield view, yaw, *(a[ok] for a in (u, v, z, world, sat, uv))


def candidate_arrays(cameras, depths):
    cells, tokens, priorities, metas, anchors = [], [], [], [], []
    for view, yaw, u, v, z, world, sat, uv in projected(cameras, depths, 2):
        cell = (uv[:, 1] // 4).astype(np.int64) * 128 + (uv[:, 0] // 4).astype(np.int64)
        token = view * 16384 + (v // 4) * 128 + u // 4
        order = np.lexsort((sat[:, 2], token, cell))
        ordered_cell, ordered_token = cell[order], token[order]
        keep = np.r_[True, (ordered_cell[1:] != ordered_cell[:-1]) | (ordered_token[1:] != ordered_token[:-1])] if len(order) else np.zeros(0, bool)
        order = order[keep]
        cell, token, uv, world, z, sat, u, v = (a[order] for a in (cell, token, uv, world, z, sat, u, v))
        order = np.lexsort((sat[:, 2], cell))
        cell, token, uv, world, z, sat, u, v = (a[order] for a in (cell, token, uv, world, z, sat, u, v))
        if not len(cell):
            continue
        starts = np.maximum.accumulate(np.where(np.r_[True, cell[1:] != cell[:-1]], np.arange(len(cell)), 0))
        rank = np.arange(len(cell)) - starts
        keep = rank < 2
        cell, token, uv, world, z, sat, u, v, rank = (a[keep] for a in (cell, token, uv, world, z, sat, u, v, rank))
        meta = np.stack(((uv[:, 0] - (cell % 128 * 4 + 2)) / 4,
                         (uv[:, 1] - (cell // 128 * 4 + 2)) / 4,
                         (world[:, 2] - cameras['center_ue_cm'][2] / 100) / 100,
                         z / 1000, np.full(len(cell), np.cos(np.deg2rad(yaw))),
                         np.full(len(cell), np.sin(np.deg2rad(yaw)))), 1)
        cells.append(cell); tokens.append(token); priorities.append(rank * 5 + view)
        metas.append(meta); anchors.append(np.stack((u, v, z, uv[:, 0], uv[:, 1], sat[:, 2]), 1))
    index = np.full((16384, 8), -1, np.int32)
    geom = np.zeros((16384, 8, 6), np.float16)
    anchor = np.zeros((16384, 8, 6), np.float64)
    if cells:
        cell, token, priority, meta, extra = map(np.concatenate, (cells, tokens, priorities, metas, anchors))
        order = np.lexsort((priority, cell))
        cell, token, meta, extra = (a[order] for a in (cell, token, meta, extra))
        starts = np.maximum.accumulate(np.where(np.r_[True, cell[1:] != cell[:-1]], np.arange(len(cell)), 0))
        rank = np.arange(len(cell)) - starts
        keep = rank < 8
        index[cell[keep], rank[keep]] = token[keep]
        geom[cell[keep], rank[keep]] = meta[keep]
        anchor[cell[keep], rank[keep]] = extra[keep]
    return index, geom, anchor


def native_arrays(cameras, depths):
    pixels, satellite_z, heights, views = [], [], [], []
    for view, _, u, v, z, world, sat, uv in projected(cameras, depths, 1):
        pixels.append(uv[:, 1].astype(np.int64) * 512 + uv[:, 0].astype(np.int64))
        satellite_z.append(sat[:, 2])
        heights.append(world[:, 2] - cameras['center_ue_cm'][2] / 100)
        views.append(np.full(len(uv), view, np.int8))
    pix, z, h, view = map(np.concatenate, (pixels, satellite_z, heights, views))
    order = np.lexsort((z, pix))
    sorted_pix = pix[order]
    first = np.r_[True, sorted_pix[1:] != sorted_pix[:-1]] if len(order) else np.zeros(0, bool)
    selected = order[first]
    zbuf = np.full(512 * 512, np.inf)
    height = np.zeros(512 * 512, np.float32)
    zbuf[pix[selected]] = z[selected]
    height[pix[selected]] = h[selected]
    support = np.isfinite(zbuf).reshape(512, 512)
    count = np.zeros(512 * 512, np.float32)
    for vi in range(5):
        near = (view == vi) & (z <= zbuf[pix] + 1)
        count[np.unique(pix[near])] += 1
    height = height.reshape(512, 512)
    dx, dy = np.zeros_like(height), np.zeros_like(height)
    dx[:, :-1] = np.where(support[:, :-1] & support[:, 1:], np.diff(height, axis=1), 0)
    dy[:-1] = np.where(support[:-1] & support[1:], np.diff(height, axis=0), 0)
    fields = np.stack((height / 100, np.clip(dx / 10, -5, 5), np.clip(dy / 10, -5, 5), count.reshape(512, 512) / 5))
    fields[:, ~support] = 0
    return fields.astype(np.float16), support
