#!/usr/bin/env python3
"""Read-only acceptance audit for separately rendered satellite/UAV production roots.

No sample is accepted merely because its directory exists.  The script validates the
per-modality producer status, required payload, image dimensions, camera schema and
the point identity shared by the two renders, then writes a reproducible manifest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

VIEWS = ("view_yaw_000", "view_yaw_072", "view_yaw_144", "view_yaw_216", "view_yaw_288")
SAT_FILES = ("satellite_rgb.0000.png", "satellite_instance_id_clean.png", "instances.json",
             "camera_intrinsics.json", "camera_extrinsics.json", "view_metadata.json", "status.json",
             "strict_instance_exr_validation.json", "strict_instance_cleanup_report.json")
UAV_FILES = ("oblique_rgb.0000.png", "depth_devicez_packed.0000.png", "depth_camera_z.0000.exr",
             "camera_intrinsics.json", "camera_extrinsics.json", "view_metadata.json", "status.json")
VERIFY_PIXELS = True


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def truth(value: Any) -> bool:
    return value is True


def image_512(path: Path) -> bool:
    if not VERIFY_PIXELS:
        return path.is_file() and path.stat().st_size > 0
    try:
        with Image.open(path) as img:
            return img.size == (512, 512)
    except Exception:
        return False


def camera_matrix(path: Path, rows: int, cols: int, kind: str) -> bool:
    try:
        v = read(path)
        if kind == "K":
            return v.get("width") == 512 and v.get("height") == 512 and all(isinstance(v.get(x), (int, float)) for x in ("fx", "fy", "cx", "cy"))
        raw = v.get("world_to_camera_cv", v.get("matrix", v.get("data", v.get("transform", v))))
        if isinstance(raw, list) and len(raw) == rows and all(isinstance(r, list) and len(r) == cols for r in raw):
            return True
        if isinstance(raw, list) and len(raw) == rows * cols:
            return True
        return False
    except Exception:
        return False


def point_key(path: Path) -> tuple[Any, ...] | None:
    try:
        point = read(path / "point_metadata.json")
        point = point.get("point", point)
        return (point.get("index"), point.get("point_id"), point.get("x"), point.get("y"), point.get("z"))
    except Exception:
        return None


def sat_check(root: Path) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []
    sat = root / "satellite"
    for name in SAT_FILES:
        if not (sat / name).is_file(): issues.append(f"sat_missing:{name}")
    if not (root / "status.json").is_file(): issues.append("sat_missing:root_status")
    if issues: return issues, {}
    try:
        top, status = read(root / "status.json"), read(sat / "status.json")
        for k in ("dataset_success", "satellite_success", "satellite_rgb_success", "satellite_mask_success"):
            if not truth(top.get(k)): issues.append(f"sat_status:{k}")
        for k in ("rgb_success", "mask_success", "strict_instance_id_success"):
            if not truth(status.get(k)): issues.append(f"sat_status:{k}")
        if status.get("resolution") != {"width": 512, "height": 512}: issues.append("sat_status:resolution")
        exr = read(sat / "strict_instance_exr_validation.json")
        if not truth(exr.get("ok")): issues.append("sat_status:strict_exr")
        clean = read(sat / "strict_instance_cleanup_report.json")
        if not truth(clean.get("ok")): issues.append("sat_status:cleanup")
        if not image_512(sat / "satellite_rgb.0000.png"): issues.append("sat_payload:rgb_512")
        if not image_512(sat / "satellite_instance_id_clean.png"): issues.append("sat_payload:mask_512")
        if not camera_matrix(sat / "camera_intrinsics.json", 3, 3, "K"): issues.append("sat_camera:K")
        if not camera_matrix(sat / "camera_extrinsics.json", 4, 4, "W2C"): issues.append("sat_camera:W2C")
        instances = read(sat / "instances.json")
        records = instances.get("instances", instances) if isinstance(instances, dict) else instances
        if not isinstance(records, list) or not records: issues.append("sat_label:instances")
    except Exception as exc:
        issues.append(f"sat_parse:{type(exc).__name__}")
    return issues, {"sat_dir": str(sat)}


def uav_check(root: Path) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []
    if not (root / "status.json").is_file(): return ["uav_missing:root_status"], {}
    try:
        status = read(root / "status.json")
        for k in ("dataset_success", "oblique_success", "oblique_all_views_successful"):
            if not truth(status.get(k)): issues.append(f"uav_status:{k}")
        if status.get("oblique_expected_view_count") != 5 or status.get("oblique_success_view_count") != 5:
            issues.append("uav_status:view_count")
    except Exception as exc:
        return [f"uav_parse:{type(exc).__name__}"], {}
    for view in VIEWS:
        folder = root / "oblique" / view
        for name in UAV_FILES:
            if not (folder / name).is_file(): issues.append(f"uav_missing:{view}/{name}")
        if any(x.startswith(f"uav_missing:{view}/") for x in issues): continue
        try:
            s = read(folder / "status.json")
            for k in ("rgb_success", "rgb_depth_pair_success", "executor_success", "quality_ok", "scenecapture_depth_exists"):
                if not truth(s.get(k)): issues.append(f"uav_status:{view}:{k}")
            q = s.get("quality", {})
            if not truth(q.get("quality_ok")): issues.append(f"uav_quality:{view}")
            if not image_512(folder / "oblique_rgb.0000.png"): issues.append(f"uav_payload:{view}:rgb_512")
            if not image_512(folder / "depth_devicez_packed.0000.png"): issues.append(f"uav_payload:{view}:depth_512")
            if not camera_matrix(folder / "camera_intrinsics.json", 3, 3, "K"): issues.append(f"uav_camera:{view}:K")
            if not camera_matrix(folder / "camera_extrinsics.json", 4, 4, "W2C"): issues.append(f"uav_camera:{view}:W2C")
        except Exception as exc:
            issues.append(f"uav_parse:{view}:{type(exc).__name__}")
    return issues, {"uav_dir": str(root / "oblique")}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sat-root", type=Path, required=True)
    p.add_argument("--uav-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--producer-status-only", action="store_true", help="use producer's already-recorded 512/quality validation; still checks every payload exists")
    args = p.parse_args(); out = args.output_dir
    global VERIFY_PIXELS
    VERIFY_PIXELS = not args.producer_status_only
    if out.exists() and any(out.iterdir()): raise RuntimeError(f"refuse overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    satellites = {x.name: x for x in args.sat_root.iterdir() if x.is_dir()}
    uavs = {x.name: x for x in args.uav_root.iterdir() if x.is_dir()}
    rows, accepted = [], []
    for name in sorted(satellites):
        sat_issues, sat_meta = sat_check(satellites[name]); uav_path = uavs.get(name)
        uav_issues, uav_meta = (["uav_missing:scene"], {}) if uav_path is None else uav_check(uav_path)
        skey = point_key(satellites[name]); ukey = point_key(uav_path) if uav_path else None
        issues = sat_issues + uav_issues
        if skey is None: issues.append("pair_key:satellite_missing")
        elif ukey is None: issues.append("pair_key:uav_missing")
        elif skey != ukey: issues.append("pair_key:mismatch")
        row = {"scene_dir": name, "scene_index": skey[0] if skey else "", "point_id": skey[1] if skey else "", "accepted": not issues, "issues": "|".join(issues), **sat_meta, **uav_meta}
        rows.append(row)
        if not issues: accepted.append(row)
    extras = sorted(set(uavs) - set(satellites))
    fields = ["scene_dir", "scene_index", "point_id", "accepted", "issues", "sat_dir", "uav_dir"]
    with (out / "audit.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    with (out / "accepted_pairs.jsonl").open("w") as f:
        for row in accepted: f.write(json.dumps(row, sort_keys=True) + "\n")
    report = {"sat_root": str(args.sat_root), "uav_root": str(args.uav_root), "pixel_validation": "producer_status_and_payload_existence" if args.producer_status_only else "fresh_decode_and_payload_existence", "satellite_scene_dirs": len(satellites), "uav_scene_dirs": len(uavs), "named_intersection": len(set(satellites) & set(uavs)), "accepted_pairs": len(accepted), "rejected_satellite_dirs": len(rows) - len(accepted), "uav_only_dirs": len(extras), "rejection_reasons": dict(Counter(issue for row in rows for issue in row["issues"].split("|") if issue)), "accepted_manifest_sha256": hashlib.sha256((out / "accepted_pairs.jsonl").read_bytes()).hexdigest()}
    (out / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (out / "uav_only_scene_dirs.txt").write_text("\n".join(extras) + ("\n" if extras else ""))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__": main()
