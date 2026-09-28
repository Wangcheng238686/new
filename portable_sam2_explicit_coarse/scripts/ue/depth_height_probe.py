"""Depth height-feature extractor + contract validator for UE UAV depth renders.

Implements the documented contract (UE_UAV_DEPTH_RENDERING_AND_MODEL_LABEL_GUIDE.md):
camera-space Z in cm, EXR channel R, backprojection via camera_to_world_cv,
UE world Z of the hit = object height above the z=0 ground plane.

Sanity mode validates on the existing 159-point backfill (no GT needed):
ground cells should yield world-Z ~ 0, building roofs positive heights.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2
import numpy as np


def read_depth_exr(path):
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise OSError(f"cannot read EXR: {path}")
    # single data channel (contract: R); cv2 may expose 3 channels with only
    # one populated -- pick the channel containing finite positive values
    if im.ndim == 3:
        best = None
        for ch in range(im.shape[2]):
            c = im[:, :, ch]
            ok = np.isfinite(c) & (c > 0)
            if best is None or ok.sum() > best[0]:
                best = (ok.sum(), c)
        z = best[1].astype(np.float32)
    else:
        z = im.astype(np.float32)
    valid = np.isfinite(z) & (z > 0)
    return z, valid


def load_camera(view_dir):
    intr = json.load(open(Path(view_dir) / "camera_intrinsics.json"))
    j = json.load(open(Path(view_dir) / "camera_extrinsics.json"))
    K = np.array([[intr["fx"], 0.0, intr["cx"]],
                  [0.0, intr["fy"], intr["cy"]],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    w2c = np.asarray(j["world_to_camera_cv"], dtype=np.float64)
    c2w = np.asarray(j["camera_to_world_cv"], dtype=np.float64)
    return K, w2c, c2w


def depth_world_height(z_cm, valid, K, c2w):
    """Backproject valid depth pixels to world (UE cm); return world Z per pixel
    plus the hit count map on an arbitrary target grid handled by the caller."""
    h, w = z_cm.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64),
                       np.arange(h, dtype=np.float64), indexing="xy")
    x = (u - K[0, 2]) / K[0, 0] * z_cm
    y = (v - K[1, 2]) / K[1, 1] * z_cm
    pc = np.stack([x, y, z_cm], -1)                       # [H,W,3] camera cm
    pw = pc @ c2w[:3, :3].T + c2w[:3, 3]                  # [H,W,3] world cm
    return pw, valid


def scene_height_features(point_dir, res=128, tile_m=360.0, center_xy_cm=None):
    view_dirs = sorted((Path(point_dir) / "oblique").iterdir())
    h_maps, vcnt = [], None
    world_xy_all = []
    for vd in view_dirs:
        z, valid = read_depth_exr(Path(vd) / "depth_camera_z.0000.exr")
        K, w2c, c2w = load_camera(vd)
        pw, v = depth_world_height(z, valid, K, c2w)
        hz = pw[:, :, 2] / 100.0                          # world Z in meters
        h_maps.append((hz, v))
        xy = pw[:, :, :2][v]                              # world XY cm of hits
        world_xy_all.append(xy)
        if vcnt is None:
            vcnt = v.astype(np.int32)
        else:
            vcnt += v.astype(np.int32)
    H, W = h_maps[0][0].shape
    hz_stack = np.stack([hm for hm, _ in h_maps])          # [V,H,W] meters
    vstack = np.stack([v for _, v in h_maps])
    valid_any = vstack > 0
    h_med = np.median(np.where(valid_any, hz_stack, np.nan), axis=0)
    h_med = np.nan_to_num(h_med, nan=0.0)
    h_std = np.nanstd(np.where(valid_any, hz_stack, np.nan), axis=0)
    h_std = np.nan_to_num(h_std, nan=0.0)
    h_max = np.nanmax(np.where(valid_any, hz_stack, np.nan), axis=0)
    h_max = np.nan_to_num(h_max, nan=0.0)
    return dict(h_med=h_med.astype(np.float32), h_std=h_std.astype(np.float32),
                h_max=h_max.astype(np.float32), valid=valid_any.astype(np.uint8),
                world_xy=world_xy_all)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--point-dir", required=True)
    args = ap.parse_args()
    feats = scene_height_features(args.point_dir)
    hm = feats["h_med"]
    print("height map:", hm.shape)
    print("percentiles 5/25/50/75/95:", np.round(np.percentile(hm, [5, 25, 50, 75, 95]), 2))
    print("frac <0.5m (ground):", round(float((hm < 0.5).mean()), 3),
          "| frac 0.5-40m (structures):", round(float(((hm >= 0.5) & (hm < 40)).mean()), 3),
          "| frac >=40m:", round(float((hm >= 40).mean()), 3))
    out = Path(args.point_dir) / "height_check.npz"
    np.savez_compressed(out, **feats)
    print("saved", out)
