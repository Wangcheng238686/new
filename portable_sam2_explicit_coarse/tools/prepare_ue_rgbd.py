#!/usr/bin/env python
"""Freeze paired samples with conservative view footprints and building leakage checks."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rsprompter.uav.cameras import load_camera
from rsprompter.uav.contract import PAYLOAD_SCHEMA
from data.ue_geometry_cache import YAWS, read_depth, validate_depth_floor


def footprint(camera, floor_m):
    k, _, t, (h, w) = camera
    pixels = np.array([[0, 0, 1], [w - 1, 0, 1], [w - 1, h - 1, 1], [0, h - 1, 1]])
    direction = (pixels @ np.linalg.inv(k).T) @ t[:3, :3].T
    if (direction[:, 2] >= -1e-6).any() or floor_m >= t[2, 3]:
        raise ValueError(
            "Footprint bound needs downward cameras above the conservative floor"
        )
    distance = (floor_m - t[2, 3]) / direction[:, 2]
    ground = t[:3, 3] + distance[:, None] * direction
    # Include the apex: high surfaces on tilted rays can precede the floor
    # section and fall outside its XY bounding box.
    vertices = np.vstack((ground[:, :2], t[None, :2, 3]))
    return np.r_[vertices.min(axis=0), vertices.max(axis=0)]


def inventory(satellite_root, uav_root, floor_m):
    points, excluded = [], []
    for folder in sorted(satellite_root.glob("Coord_*")):
        try:
            sat = folder / "satellite"
            uav = uav_root / folder.name
            status = json.loads((folder / "status.json").read_text())
            if not status.get("dataset_success"):
                raise ValueError("satellite not accepted")
            required = [
                sat / name
                for name in (
                    "satellite_rgb.0000.png",
                    "satellite_instance_id_clean.png",
                    "instances.json",
                )
            ]
            required += [
                uav / "oblique" / f"view_yaw_{yaw:03d}" / name
                for yaw in YAWS
                for name in ("oblique_rgb.0000.png", "depth_camera_z.0000.exr")
            ]
            if not all(path.is_file() for path in required):
                raise ValueError("incomplete paired files")
            views = [uav / "oblique" / f"view_yaw_{yaw:03d}" for yaw in YAWS]
            cameras = [load_camera(view) for view in views]
            for view, camera in zip(views, cameras):
                validate_depth_floor(
                    read_depth(view / "depth_camera_z.0000.exr", camera[3][0]),
                    camera,
                    floor_m,
                )
            bounds = np.array(
                [footprint(c, floor_m) for c in [load_camera(sat), *cameras]]
            )
            bound = np.r_[bounds[:, :2].min(axis=0), bounds[:, 2:].max(axis=0)]
            instances = json.loads((sat / "instances.json").read_text())
            p = instances["point"]
            origin = [p[axis] / 100 for axis in ("x", "y", "z")]
            # The RGB-D target is a flight datum, not a satellite prediction.
            targets = [
                json.loads((view / "camera_extrinsics.json").read_text()).get(
                    "target_ue_cm"
                )
                for view in views
            ]
            if all(target is not None for target in targets):
                coordinates = np.array(
                    [
                        [target[axis] / 100 for axis in ("x", "y", "z")]
                        for target in targets
                    ]
                )
                if not np.allclose(coordinates, coordinates[0], atol=1e-4):
                    raise ValueError("UAV targets are inconsistent")
                origin = coordinates[0].tolist()
            else:
                origin[:2] = np.mean([c[2][:2, 3] for c in cameras], axis=0).tolist()
            points.append(
                dict(
                    index=int(p["index"]),
                    satellite_dir=str(sat.relative_to(satellite_root)),
                    uav_dir=str(uav.relative_to(uav_root)),
                    origin_m=origin,
                    footprint_xy_m=bound.tolist(),
                    building_ids=sorted(
                        {
                            str(row["global_instance_id"])
                            for row in instances["instances"]
                        }
                    ),
                )
            )
        except (OSError, KeyError, ValueError) as exc:
            excluded.append(dict(point=folder.name, reason=str(exc)))
    return points, excluded


def split_points(points, axis, train_fraction, test_boundary=None):
    coordinate = 0 if axis == "x" else 1
    cut = float(
        np.quantile([p["origin_m"][coordinate] for p in points], train_fraction)
    )
    splits = dict(train=[], validation=[])
    if test_boundary is not None:
        if test_boundary <= cut:
            raise ValueError("Test boundary must be beyond the training boundary")
        splits["test"] = []
    rejected = []
    by_id = {p["index"]: p for p in points}
    for p in points:
        low, high = p["footprint_xy_m"][coordinate], p["footprint_xy_m"][coordinate + 2]
        if high < cut:
            splits["train"].append(p["index"])
        elif low > cut and (test_boundary is None or high < test_boundary):
            splits["validation"].append(p["index"])
        elif test_boundary is not None and low > test_boundary:
            splits["test"].append(p["index"])
        else:
            rejected.append(p["index"])
    used_buildings = set()
    for name, indices in splits.items():
        kept = []
        for index in indices:
            ids = set(by_id[index]["building_ids"])
            if ids & used_buildings:
                rejected.append(index)
            else:
                kept.append(index)
        splits[name] = kept
        used_buildings.update(b for index in kept for b in by_id[index]["building_ids"])
        if not kept:
            raise ValueError(
                f"Conservative split has no {name} samples; expand data or choose another axis/boundary"
            )
    return splits, dict(
        axis=axis,
        train_boundary_m=cut,
        test_boundary_m=test_boundary,
        boundary_or_shared_building_exclusions=rejected,
        cross_split_building_overlap=0,
        rule="whole satellite+five-UAV frustum bounds on one side; no shared building IDs",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--satellite-root", type=Path, required=True)
    parser.add_argument("--uav-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--axis", choices=("x", "y"), default="x")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--test-boundary-m", type=float)
    parser.add_argument("--floor-m", type=float, default=-100.0)
    args = parser.parse_args()
    if not 0 < args.train_fraction < 1:
        raise ValueError("Training fraction must be between zero and one")
    points, excluded = inventory(
        args.satellite_root.resolve(), args.uav_root.resolve(), args.floor_m
    )
    if not points:
        raise ValueError("No complete paired samples")
    splits, audit = split_points(
        points, args.axis, args.train_fraction, args.test_boundary_m
    )
    result = dict(
        schema=PAYLOAD_SCHEMA,
        satellite_root=str(args.satellite_root.resolve()),
        uav_root=str(args.uav_root.resolve()),
        points=points,
        splits=splits,
        split_audit={**audit, "floor_m": args.floor_m},
        inventory_excluded=excluded,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                accepted=len(points),
                splits={k: len(v) for k, v in splits.items()},
                audit=audit,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
