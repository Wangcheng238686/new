#!/usr/bin/env python3
"""Materialize a non-copying, immutable raw pairset from an audit manifest.

The data payload remains in its producer roots.  Each accepted scene gets only two
relative symlinks (satellite and uav) plus a manifest entry, avoiding an untracked
copy of hundreds of GB while making the accepted contract explicit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def link(path: Path, target: Path) -> None:
    if path.exists() or path.is_symlink(): raise RuntimeError(f"refuse overwrite {path}")
    path.symlink_to(os.path.relpath(target, path.parent), target_is_directory=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--accepted-jsonl", type=Path, required=True)
    p.add_argument("--audit-summary", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args(); root = args.output_root.resolve()
    if root.exists() and any(root.iterdir()): raise RuntimeError(f"refuse overwrite {root}")
    root.mkdir(parents=True, exist_ok=True); scenes = root / "scenes"; scenes.mkdir()
    accepted = [json.loads(line) for line in args.accepted_jsonl.read_text().splitlines() if line]
    if not accepted: raise RuntimeError("refuse to materialize zero accepted scenes")
    required = {"scene_dir", "scene_index", "point_id", "sat_dir", "uav_dir"}
    for row in accepted:
        if required - set(row): raise RuntimeError(f"manifest row missing keys: {required-set(row)}")
        d = scenes / row["scene_dir"]; d.mkdir()
        link(d / "satellite", Path(row["sat_dir"]).resolve())
        link(d / "uav", Path(row["uav_dir"]).resolve())
    (root / "accepted_pairs.jsonl").write_text("\n".join(json.dumps(x, sort_keys=True) for x in accepted) + "\n")
    audit = json.loads(args.audit_summary.read_text())
    contract = {"schema_version": "ue_raw_satellite_uav_pairset_v1", "accepted_scene_count": len(accepted),
                "layout": {"scenes/<scene_dir>/satellite": "symlink to producer satellite payload", "scenes/<scene_dir>/uav": "symlink to producer oblique payload"},
                "required_satellite": ["satellite_rgb.0000.png", "satellite_instance_id_clean.png", "instances.json", "camera_intrinsics.json", "camera_extrinsics.json"],
                "required_uav_views": ["view_yaw_000", "view_yaw_072", "view_yaw_144", "view_yaw_216", "view_yaw_288"],
                "uav_per_view": ["oblique_rgb.0000.png", "depth_devicez_packed.0000.png", "depth_camera_z.0000.exr", "camera_intrinsics.json", "camera_extrinsics.json"],
                "audit_summary_sha256": sha256(args.audit_summary), "accepted_manifest_sha256": sha256(root / "accepted_pairs.jsonl"),
                "source_audit": audit}
    (root / "DATASET_CONTRACT.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_root": str(root), "accepted_scene_count": len(accepted), "manifest_sha256": contract["accepted_manifest_sha256"]}, indent=2))


if __name__ == "__main__": main()
