"""RGB-augmented native cache builder (20261024 prereg, part 1).

Mirrors rsprompter/uav/native_geometry_probe.prepare_native math EXACTLY
(same z-buffer winner, same support, same 4 fields); additionally stores the
winning surface's source-pixel UAV RGB (3ch, 0-255 -> float16). Independent
cache dir; the reproduced native cache is never touched.
"""
import json
import os
from pathlib import Path
import cv2
import numpy as np

from rsprompter.uav.local_adapter_probe import camera_arrays, file_sha, YAWS

VERSION = 'native512_rgb_v1'


def prepare_rgb(folder, cache):
    path = Path(cache) / f'{folder.name}.npz'
    if path.is_file():
        with np.load(path) as a:
            assert str(a['version']) == VERSION
        return path
    c = json.loads((folder / 'cameras.json').read_text())
    assert c['world_units'] == 'cm' and c['camera_axes'] == 'x_right_y_down_z_forward'
    sk, se, _ = camera_arrays(c['satellite'])
    yy, xx = np.mgrid[:512, :512]
    u, v = xx.ravel(), yy.ravel()
    pixels, depths, heights, views, srcidx = [], [], [], [], []
    rgbs = []
    hashes = {'cameras.json': file_sha(folder / 'cameras.json')}
    for view, yaw in enumerate(YAWS):
        dname = f'uav/yaw_{yaw:03d}/depth_camera_z.exr'
        hashes[dname] = file_sha(folder / dname)
        z = cv2.imread(str(folder / dname), cv2.IMREAD_UNCHANGED)[:, :, 2].ravel().astype(np.float64) / 100
        rgb = cv2.imread(str(folder / f'uav/yaw_{yaw:03d}/rgb.png'))
        assert rgb is not None and rgb.shape == (512, 512, 3)
        rgbs.append(rgb[:, :, ::-1].reshape(-1, 3).astype(np.float32))  # BGR->RGB
        good = np.isfinite(z) & (z > 0) & (z < 10000)
        k, e, t = camera_arrays(c['uav'][f'yaw_{yaw:03d}'])
        x = np.stack(((u[good] - k[0, 2]) * z[good] / k[0, 0], (v[good] - k[1, 2]) * z[good] / k[1, 1], z[good]), 1)
        world = x @ t[:3, :3].T + t[:3, 3]
        target = world @ se[:3, :3].T + se[:3, 3]
        proj = target @ sk.T
        uv = proj[:, :2] / proj[:, 2:]
        valid = (target[:, 2] > 0) & np.isfinite(uv).all(1) & (uv >= 0).all(1) & (uv < 512).all(1)
        uv = uv[valid]
        gi = np.where(good)[0][valid]
        pix = (uv[:, 1].astype(np.int64) * 512 + uv[:, 0].astype(np.int64))
        pixels.append(pix)
        depths.append(target[valid, 2])
        heights.append(world[valid, 2] - c['center_ue_cm'][2] / 100)
        views.append(np.full(len(pix), view, np.int8))
        srcidx.append(gi + view * (512 * 512))       # global index into stacked rgbs
    pix, z, h, view, gsrc = map(np.concatenate, (pixels, depths, heights, views, srcidx))
    order = np.lexsort((z, pix))
    sorted_pix = pix[order]
    first = np.r_[True, sorted_pix[1:] != sorted_pix[:-1]]
    selected = order[first]
    zbuf = np.full(512 * 512, np.inf)
    height = np.zeros(512 * 512, np.float32)
    rgbwin = np.zeros((512 * 512, 3), np.float32)
    zbuf[pix[selected]] = z[selected]
    height[pix[selected]] = h[selected]
    srcview = np.full(512 * 512, -1, np.int64)
    srcview[pix[selected]] = gsrc[selected] // (512 * 512)
    srcflat = gsrc[selected]
    allrgb = np.concatenate(rgbs, 0)
    rgbwin[pix[selected]] = allrgb[srcflat]
    support = np.isfinite(zbuf).reshape(512, 512)
    count = np.zeros(512 * 512, np.float32)
    for vi in range(5):
        near = (view == vi) & (z <= zbuf[pix] + 1)
        count[np.unique(pix[near])] += 1
    height = height.reshape(512, 512)
    dx = np.zeros_like(height)
    dy = np.zeros_like(height)
    validx = support[:, :-1] & support[:, 1:]
    validy = support[:-1] & support[1:]
    dx[:, :-1] = np.where(validx, np.diff(height, axis=1), 0)
    dy[:-1] = np.where(validy, np.diff(height, axis=0), 0)
    fields = np.stack((height / 100, np.clip(dx / 10, -5, 5), np.clip(dy / 10, -5, 5), count.reshape(512, 512) / 5))
    fields[:, ~support] = 0
    rgbmap = rgbwin.reshape(512, 512, 3).transpose(2, 0, 1)   # [3,512,512] 0-255
    rgbmap[:, ~support] = 0
    assert np.isfinite(fields).all() and np.isfinite(rgbmap).all()
    report = dict(version=VERSION, source_hashes=hashes,
                  support_pixels=int(support.sum()),
                  fields=['height_m/100', 'dx_m/10_clipped5', 'dy_m/10_clipped5', 'views_nearest1m/5'],
                  rgb='zbuffer-winner source pixel, RGB order, 0-255')
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, fields=fields.astype(np.float16),
                            rgb=rgbmap.astype(np.float16),
                            support=support, version=VERSION,
                            report=json.dumps(report))
    tmp.replace(path)
    return path


if __name__ == '__main__':
    import sys
    from concurrent.futures import ThreadPoolExecutor
    root = Path(sys.argv[1])
    cache = Path(sys.argv[2])
    cache.mkdir(parents=True, exist_ok=True)
    folders = sorted((root / 'points').iterdir())
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, p in enumerate(pool.map(lambda f: prepare_rgb(f, cache), folders)):
            if i % 200 == 0:
                print('PREPARED', i, len(folders), flush=True)
    print('DONE', len(folders))
