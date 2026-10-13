#!/usr/bin/env python3
"""SURFBEV-D1-Alignment-R1: read-only geometry/label alignment probe.

See docs/SURFBEV-D1-Alignment-R1_preregistration.md.  This intentionally
contains no neural model.  It asks whether calibrated UAV surface geometry
aligns with the strict satellite visible-roof raster more than a 90-degree
world-XY geometry sham does.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = "portable_sam2_explicit_coarse/data/ue_raw_hero_uav_pairset_20261011"
OUT = "portable_sam2_explicit_coarse/outputs/SURFBEV-D1-Alignment-R1"
BOOT_DRAWS, BOOT_SEED = 10000, 20261011


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def k_and_ext(view_dir):
    i, e = jload(view_dir / "camera_intrinsics.json"), jload(view_dir / "camera_extrinsics.json")
    k = np.array([[i["fx"], 0.0, i["cx"]], [0.0, i["fy"], i["cy"]], [0.0, 0.0, 1.0]], np.float64)
    return k, np.asarray(e["world_to_camera_cv"], np.float64), np.asarray(e["camera_to_world_cv"], np.float64)


def read_ids(path):
    a = np.asarray(Image.open(path))
    if a.ndim == 2:
        return a.astype(np.int32)
    rgb = a[..., :3]
    if not (np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2])):
        raise ValueError("strict clean ID PNG is not grayscale encoded")
    return rgb[..., 0].astype(np.int32)


def device_z(path):
    # Producer document's normalized RGB radix decode.
    a = np.asarray(Image.open(path).convert("RGB"), np.float64) / 255.0
    return 1.0 - (a[..., 0] + a[..., 1] / 255.0 + a[..., 2] / 65025.0)


def dilate_bool(x, radius=2):
    y = x.copy()
    h, w = x.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            src_y0, src_y1 = max(0, -dy), min(h, h - dy)
            src_x0, src_x1 = max(0, -dx), min(w, w - dx)
            dst_y0, dst_y1 = max(0, dy), min(h, h + dy)
            dst_x0, dst_x1 = max(0, dx), min(w, w + dx)
            y[dst_y0:dst_y1, dst_x0:dst_x1] |= x[src_y0:src_y1, src_x0:src_x1]
    return y


def project_rasterize(pw, sat_k, sat_w2c, shape):
    """Satellite-view z-buffer followed by fixed 2px occupancy splat."""
    pc = pw @ sat_w2c[:3, :3].T + sat_w2c[:3, 3]
    good = np.isfinite(pc).all(1) & (pc[:, 2] > 1e-6)
    u = np.rint(sat_k[0, 0] * pc[:, 0] / pc[:, 2] + sat_k[0, 2]).astype(np.int64)
    v = np.rint(sat_k[1, 1] * pc[:, 1] / pc[:, 2] + sat_k[1, 2]).astype(np.int64)
    h, w = shape
    good &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z = u[good], v[good], pc[good, 2]
    flat = v * w + u
    zmin = np.full(h * w, np.inf, np.float64)
    np.minimum.at(zmin, flat, z)
    # A point that loses its exact satellite pixel z-buffer cannot contribute.
    keep = z <= zmin[flat] + 1e-3
    occ = np.zeros((h, w), dtype=bool)
    occ[v[keep], u[keep]] = True
    return dilate_bool(occ, 2), dict(n_input_points=int(len(pw)), n_in_sat_fov=int(good.sum()), n_zbuffer_points=int(keep.sum()))


def metrics(occ, ids):
    fg = ids > 0
    support = occ & fg
    # two support pixels makes a one-pixel sampling accident insufficient.
    visible = np.unique(ids[fg])
    n_supported = sum(int((support & (ids == lid)).sum() >= 2) for lid in visible)
    return dict(
        visible_instances=int(len(visible)),
        supported_instances=int(n_supported),
        instance_supported_ratio=float(n_supported / len(visible)) if len(visible) else None,
        roof_pixel_coverage=float(support.sum() / fg.sum()) if fg.any() else None,
        surface_precision=float(support.sum() / occ.sum()) if occ.any() else None,
        surface_pixels=int(occ.sum()), roof_pixels=int(fg.sum()), support_pixels=int(support.sum()),
    )


def scene_points(scene, stride):
    chunks = []
    for vd in sorted((scene / "uav").glob("view_yaw_*")):
        k, _, c2w = k_and_ext(vd)
        dz = device_z(vd / "depth_devicez_packed.0000.png")
        v, u = np.mgrid[stride // 2:dz.shape[0]:stride, stride // 2:dz.shape[1]:stride]
        zdev = dz[v, u]
        ok = np.isfinite(zdev) & (zdev > 1e-8) & (zdev < 1.0)
        u, v, z = u[ok].astype(np.float64), v[ok].astype(np.float64), 10.0 / zdev[ok]
        pc = np.stack([(u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z], 1)
        chunks.append(pc @ c2w[:3, :3].T + c2w[:3, 3])
    if len(chunks) != 5:
        raise ValueError("not five UAV views")
    return np.concatenate(chunks, 0)


def cell(scene_name):
    m = re.search(r"_gx_(-?\d+)_gy_(-?\d+)_", scene_name)
    if not m:
        raise ValueError(f"cannot parse spatial cell: {scene_name}")
    return int(m.group(1)), int(m.group(2))


def bootstrap_delta(rows, key):
    a = np.asarray([x["correct"][key] for x in rows], np.float64)
    b = np.asarray([x["sham90"][key] for x in rows], np.float64)
    d = a - b
    rng = np.random.default_rng(BOOT_SEED + sum(map(ord, key)))
    draws = rng.integers(0, len(d), size=(BOOT_DRAWS, len(d)))
    means = d[draws].mean(1)
    return dict(correct_mean=float(a.mean()), sham_mean=float(b.mean()), delta_mean=float(d.mean()),
                ci95_percentile=[float(np.quantile(means, .025)), float(np.quantile(means, .975))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--per-cell", type=int, default=3)
    ap.add_argument("--stride", type=int, default=4)
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty {out}")
    out.mkdir(parents=True, exist_ok=True)
    manifest = root / "accepted_pairs.jsonl"
    recs = [json.loads(x) for x in manifest.read_text().splitlines() if x]
    by_cell = {}
    for r in recs:
        by_cell.setdefault(cell(r["scene_dir"]), []).append(r["scene_dir"])
    chosen = []
    for c in sorted(by_cell):
        chosen.extend(sorted(by_cell[c])[:args.per_cell])
    rows, errors = [], []
    for name in chosen:
        try:
            scene = root / "scenes" / name
            ids = read_ids(scene / "satellite" / "satellite_instance_id_clean.png")
            sat_k, sat_w2c, sat_c2w = k_and_ext(scene / "satellite")
            pw = scene_points(scene, args.stride)
            correct_occ, ct = project_rasterize(pw, sat_k, sat_w2c, ids.shape)
            # 90-degree yaw sham about the satellite optical-axis ground center.
            center = sat_c2w[:3, 3].copy(); center[2] = 0.0
            q = pw.copy(); rel = q[:, :2] - center[:2]
            q[:, 0] = center[0] - rel[:, 1]
            q[:, 1] = center[1] + rel[:, 0]
            sham_occ, st = project_rasterize(q, sat_k, sat_w2c, ids.shape)
            rows.append(dict(scene=name, cell=list(cell(name)), correct=metrics(correct_occ, ids), sham90=metrics(sham_occ, ids), correct_telemetry=ct, sham90_telemetry=st))
        except Exception as exc:
            errors.append(dict(scene=name, error=repr(exc)))
    keys = ("instance_supported_ratio", "roof_pixel_coverage", "surface_precision")
    stats = {k: bootstrap_delta(rows, k) for k in keys} if rows else {}
    gate = bool(not errors and all(stats[k]["ci95_percentile"][0] > 0 for k in keys))
    report = dict(probe="SURFBEV-D1-Alignment-R1", scope="read-only data geometry; not AP",
                  manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(), stride=args.stride,
                  per_cell=args.per_cell, cells=len(by_cell), selected_scenes=len(chosen), completed_scenes=len(rows),
                  errors=errors, paired_bootstrap=dict(draws=BOOT_DRAWS, seed=BOOT_SEED, metrics=stats),
                  verdict="PASS" if gate else "FAIL", gate="all three correct-sham CI lower bounds > 0")
    (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (out / "per_scene.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
