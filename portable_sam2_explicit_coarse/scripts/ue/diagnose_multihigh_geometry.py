#!/usr/bin/env python3
"""Read-only feasibility diagnostic for a multi-height UAV plane-sweep.

This does *not* load a model or create a feature volume.  It verifies whether
the calibrated UE assets can geometrically support one: coverage of height
hypotheses, height-induced pixel parallax, and satellite-instance ROI anchors.

Example (from portable_sam2_explicit_coarse):
  python scripts/ue/diagnose_multihigh_geometry.py --split val --scenes 40 \
      --height-bins 0,2,4,7,10,15,22,32,45 --json-out /tmp/ue_geometry.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path,
                   default="/data1/wangcheng/dataset/ue_coco_with_uav")
    p.add_argument("--split", choices=("train", "val", "test"), default="val")
    p.add_argument("--scenes", type=int, default=40,
                   help="Deterministic prefix of valid scenes; 0 means all.")
    p.add_argument("--grid", type=int, default=48, help="Horizontal samples per axis.")
    p.add_argument("--extent-m", type=float, default=360.0,
                   help="Square ground extent used by current GeometrySparseBEV.")
    p.add_argument("--height-bins", default="0,2,4,7,10,15,22,32,45",
                   help="World-Z hypotheses in metres (UE metadata is centimetres).")
    p.add_argument("--offset-m", type=float, default=20.0,
                   help="ROI anchor horizontal search radius; uses a 3x3 grid.")
    p.add_argument("--json-out", type=Path, default=None)
    return p.parse_args()


def project(world_xy_cm, height_m, center_cm, intrinsics, extrinsics):
    """Return UV and positive-depth mask for W2C UE camera matrices."""
    n = len(world_xy_cm)
    points = np.ones((n, 4), dtype=np.float64)
    points[:, :2] = world_xy_cm + center_cm[None]
    points[:, 2] = height_m * 100.0
    camera = np.einsum("vij,nj->vni", extrinsics, points)
    z = camera[..., 2]
    pixel = np.einsum("vij,vnj->vni", intrinsics, camera[..., :3])
    uv = pixel[..., :2] / np.maximum(pixel[..., 2:3], 1e-6)
    return uv, z > 1e-4


def valid_mask(uv, positive_z, image_sizes):
    widths = np.asarray([s[0] for s in image_sizes])[:, None]
    heights = np.asarray([s[1] for s in image_sizes])[:, None]
    return (positive_z & (uv[..., 0] >= 0) & (uv[..., 0] < widths)
            & (uv[..., 1] >= 0) & (uv[..., 1] < heights))


def local_grid(grid, extent_m):
    # Same target orientation as GeometrySparseBEV: X follows negative image-v;
    # Y follows image-u.  Units are centimetres before scene_center is added.
    u = (np.arange(grid) + .5) / grid
    v = (np.arange(grid) + .5) / grid
    vv, uu = np.meshgrid(v, u, indexing="ij")
    return np.stack((-(vv.ravel() - .5) * extent_m * 100,
                     (uu.ravel() - .5) * extent_m * 100), axis=1)


def instance_centers(satellite_json, image_size, extent_m):
    """LabelMe polygon group centres mapped by the current satellite affine."""
    shapes = json.loads(satellite_json.read_text()).get("shapes", [])
    groups = defaultdict(list)
    for i, shape in enumerate(shapes):
        if shape.get("shape_type") == "polygon" and len(shape.get("points", [])) >= 3:
            groups[shape.get("group_id", i)].extend(shape["points"])
    width, height = image_size
    out = []
    for points in groups.values():
        xy = np.asarray(points, dtype=np.float64)
        u, v = xy.mean(axis=0)
        out.append((-(v / height - .5) * extent_m * 100,
                    (u / width - .5) * extent_m * 100))
    return np.asarray(out, dtype=np.float64).reshape(-1, 2)


def mean_or_none(values):
    return float(np.mean(values)) if values else None


def main():
    args = parse_args()
    heights_m = [float(x) for x in args.height_bins.split(",")]
    if not heights_m or min(heights_m) < 0:
        raise ValueError("--height-bins must contain non-negative values")
    root, split = args.data_root, args.split
    centers = json.loads((root / "scene_centers.json").read_text())
    satellite_dir, drone_dir = root / split / "satellite", root / split / "drone"
    ids = [p.stem for p in sorted(satellite_dir.glob("*.png"))
           if (satellite_dir / f"{p.stem}.json").is_file()
           and (drone_dir / p.stem / "ue_render_metadata" / "camera_params.json").is_file()
           and p.stem in centers]
    if args.scenes:
        ids = ids[:args.scenes]
    if not ids:
        raise RuntimeError("No valid satellite/UAV/metadata triples were found")

    grid_xy = local_grid(args.grid, args.extent_m)
    offsets = np.array([-args.offset_m, 0.0, args.offset_m]) * 100.0
    offsets_xy = np.array([(dx, dy) for dx in offsets for dy in offsets])
    height_stats = {str(h): defaultdict(list) for h in heights_m}
    roi_stats = {str(h): defaultdict(list) for h in heights_m}
    errors = []

    for sid in ids:
        try:
            meta_dir = drone_dir / sid / "ue_render_metadata"
            meta = json.loads((meta_dir / "camera_params.json").read_text())
            K = np.asarray(meta["intrinsics"], dtype=np.float64)
            E = np.asarray(meta["extrinsics"], dtype=np.float64)
            names = meta["image_names"]
            if K.shape != (5, 3, 3) or E.shape != (5, 4, 4) or len(names) != 5:
                raise ValueError(f"expected 5-view K/E, got {K.shape}/{E.shape}/{len(names)}")
            image_sizes = []
            for name in names:
                path = drone_dir / sid / name
                if not path.is_file():
                    path = meta_dir / name
                img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise FileNotFoundError(path)
                image_sizes.append((int(img.shape[1]), int(img.shape[0])))
            sat = cv2.imread(str(satellite_dir / f"{sid}.png"), cv2.IMREAD_UNCHANGED)
            if sat is None:
                raise FileNotFoundError(satellite_dir / f"{sid}.png")
            roi_xy = instance_centers(satellite_dir / f"{sid}.json",
                                      (sat.shape[1], sat.shape[0]), args.extent_m)
            center = np.asarray(centers[sid], dtype=np.float64)

            base_uv, base_z = project(grid_xy, 0.0, center, K, E)
            base_valid = valid_mask(base_uv, base_z, image_sizes)
            for h in heights_m:
                uv, positive = project(grid_xy, h, center, K, E)
                valid = valid_mask(uv, positive, image_sizes)
                per_point = valid.sum(axis=0)
                stats = height_stats[str(h)]
                stats["per_view_valid_ratio"].append(float(valid.mean()))
                stats["any_view_coverage"].append(float((per_point > 0).mean()))
                stats["two_plus_view_coverage"].append(float((per_point >= 2).mean()))
                stats["mean_visible_views"].append(float(per_point.mean()))
                if h != 0:
                    common = valid & base_valid
                    shift = np.linalg.norm(uv - base_uv, axis=-1)[common]
                    if len(shift):
                        stats["common_view_parallax_px"].append(float(np.median(shift)))

                if len(roi_xy):
                    candidates = (roi_xy[:, None, :] + offsets_xy[None]).reshape(-1, 2)
                    ruv, rz = project(candidates, h, center, K, E)
                    rvalid = valid_mask(ruv, rz, image_sizes).reshape(5, len(roi_xy), 9)
                    rcov = rvalid.any(axis=0)
                    rstats = roi_stats[str(h)]
                    rstats["anchor_any_view_coverage"].append(float(rcov[:, 4].mean()))
                    rstats["search_any_candidate_coverage"].append(float(rcov.any(axis=1).mean()))
                    rstats["search_two_plus_view_coverage"].append(float((rvalid.sum(axis=0) >= 2).any(axis=1).mean()))
        except Exception as exc:  # Keep a dataset audit useful even with damaged scenes.
            errors.append({"scene_id": sid, "error": str(exc)})

    def reduce(stats):
        return {key: mean_or_none(value) for key, value in stats.items()}
    report = {
        "purpose": "Read-only geometric feasibility of multi-height plane-sweep + ROI height/offset retrieval",
        "contract": {"world_units": "UE centimetres", "extrinsics": "W2C", "views_expected": 5,
                     "extent_m": args.extent_m, "grid": args.grid, "height_bins_m": heights_m,
                     "roi_offsets_m": [-args.offset_m, 0.0, args.offset_m]},
        "split": split, "scenes_requested": args.scenes, "scenes_examined": len(ids) - len(errors),
        "scene_ids": ids, "errors": errors,
        "height_grid": {str(h): reduce(height_stats[str(h)]) for h in heights_m},
        "roi_anchor": {str(h): reduce(roi_stats[str(h)]) for h in heights_m},
        "interpretation": {
            "any_view_coverage": "fraction of grid points with at least one valid UAV observation",
            "two_plus_view_coverage": "fraction with multi-view support; a prerequisite for geometry-only consistency",
            "common_view_parallax_px": "median height-vs-ground pixel shift where both projections are valid; near zero means heights are not distinguishable in image space",
            "search_any_candidate_coverage": "fraction of labelled satellite instances where any height/3x3 offset candidate is seen by UAV",
        },
    }
    print(json.dumps(report, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
