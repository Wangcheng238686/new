#!/usr/bin/env python3
"""Build or independently recheck the leakage-controlled raw SurfaceBEV split."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


PAIR_SHA = "f15cd8f5b8d10a69d721546d6cfa1f2cce58af2861d57b1357c5e9ff1e1fe62a"
V1_SHA = "894df4561b166d37d06d033a6fc326223062384b402f730591b4c76d41f926d5"
BUFFER_CM = 40000.0
SOURCE_FILES = (
    "satellite_rgb.0000.png", "satellite_instance_id_clean.png", "instances.json",
    "camera_intrinsics.json", "camera_extrinsics.json",
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def visible_ids(sat):
    meta = json.loads((sat / "instances.json").read_text())
    local_to_global = {
        int(row["local_instance_id"]): int(row["global_instance_id"])
        for row in meta["instances"]
    }
    raster = np.asarray(Image.open(sat / "satellite_instance_id_clean.png"))
    if raster.shape != (512, 512):
        raise ValueError(f"unexpected clean-ID shape: {sat}: {raster.shape}")
    labels = {int(x) for x in np.unique(raster)} - {0}
    if labels - local_to_global.keys():
        raise ValueError(f"unmapped visible local IDs: {sat}")
    return {local_to_global[x] for x in labels}


def conflicts(left, right, buffer_cm):
    return (abs(left["x"] - right["x"]) < buffer_cm and
            abs(left["y"] - right["y"]) < buffer_cm)


def build(root):
    accepted = root / "accepted_pairs.jsonl"
    old = root / "surfbev_b0_split.json"
    if sha256(accepted) != PAIR_SHA or sha256(old) != V1_SHA:
        raise ValueError("v1 input manifest hash mismatch")
    source_rows = [json.loads(line) for line in accepted.read_text().splitlines() if line]
    old_rows = json.loads(old.read_text())["scenes"]
    old_by_scene = {r["scene_dir"]: r for r in old_rows}
    if len(source_rows) != 1200 or len(old_by_scene) != 1200:
        raise ValueError("source/v1 scene count mismatch")
    records = []
    input_digest = hashlib.sha256()
    for row in source_rows:
        scene = row["scene_dir"]
        if scene not in old_by_scene:
            raise ValueError(f"scene absent from v1: {scene}")
        sat = root / "scenes" / scene / "satellite"
        if sat.resolve() != Path(row["sat_dir"]).resolve():
            raise ValueError(f"satellite path differs from accepted manifest: {scene}")
        file_hashes = {name: sha256(sat / name) for name in SOURCE_FILES}
        input_digest.update((scene + json.dumps(file_hashes, sort_keys=True)).encode())
        k = json.loads((sat / "camera_intrinsics.json").read_text())
        e = json.loads((sat / "camera_extrinsics.json").read_text())
        point = json.loads((sat / "instances.json").read_text())["point"]
        if (k["width"], k["height"], k["focal_length_mm"], k["sensor_width_mm"],
                k["sensor_height_mm"], e["location_ue_cm"]["z"],
                e["rotation_ue_deg"]["pitch"], e["rotation_ue_deg"]["yaw"],
                e["rotation_ue_deg"]["roll"]) != (
                512, 512, 200.0, 36.0, 36.0, 200000.0, -90.0, 0.0, 0.0):
            raise ValueError(f"camera contract drift: {scene}")
        if (point["x"], point["y"]) != (
                e["location_ue_cm"]["x"], e["location_ue_cm"]["y"]):
            raise ValueError(f"camera center and point disagree: {scene}")
        records.append({
            "scene_dir": scene, "scene_index": row["scene_index"],
            "gx": old_by_scene[scene]["gx"], "gy": old_by_scene[scene]["gy"],
            "old_split": old_by_scene[scene]["split"],
            "x": point["x"], "y": point["y"],
            "ids": visible_ids(sat), "rgb_sha256": file_hashes["satellite_rgb.0000.png"],
        })

    held = [r for r in records if r["old_split"] in ("val", "test")]
    held_ids = set().union(*(r["ids"] for r in held))
    result = []
    reasons_count = Counter()
    for rec in records:
        reasons = []
        if rec["old_split"] == "train":
            if rec["ids"] & held_ids:
                reasons.append("shared_visible_global_instance_id")
            if any(conflicts(rec, h, BUFFER_CM) for h in held):
                reasons.append("satellite_view_buffer_400m")
        assigned = "excluded" if reasons else rec["old_split"]
        reasons_count.update(reasons)
        result.append({
            "scene_dir": rec["scene_dir"], "scene_index": rec["scene_index"],
            "gx": rec["gx"], "gy": rec["gy"], "split": assigned,
            "exclusion_reasons": reasons,
        })
        rec["split"] = assigned

    active = {s: [r for r in records if r["split"] == s] for s in ("train", "val", "test")}
    image_hashes = {s: {r["rgb_sha256"] for r in active[s]} for s in active}
    cross = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        id_a = set().union(*(r["ids"] for r in active[a]))
        id_b = set().union(*(r["ids"] for r in active[b]))
        pair_count = sum(conflicts(x, y, BUFFER_CM) for x in active[a] for y in active[b])
        cross[f"{a}_{b}"] = {
            "shared_visible_global_ids": len(id_a & id_b),
            "view_buffer_conflicting_pairs": pair_count,
            "exact_rgb_sha256_duplicates": len(image_hashes[a] & image_hashes[b]),
        }
        if any(cross[f"{a}_{b}"].values()):
            raise ValueError(f"cross-split isolation failed: {a}/{b}: {cross[f'{a}_{b}']}")
    counts = dict(Counter(r["split"] for r in records))
    if counts != {"train": 694, "val": 100, "test": 151, "excluded": 255}:
        raise ValueError(f"unexpected v2 counts: {counts}")
    return {
        "schema_version": "surfacebev_b0_spatial_object_disjoint_v2",
        "accepted_manifest_sha256": PAIR_SHA,
        "parent_split_sha256": V1_SHA,
        "source_payload_fingerprint_sha256": input_digest.hexdigest(),
        "camera_ground_footprint_width_cm": 36000.0,
        "cross_split_view_buffer_cm": BUFFER_CM,
        "counts": counts,
        "exclusion_reasons_count": dict(reasons_count),
        "audit": cross,
        "scenes": result,
    }


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2] / "data/ue_raw_hero_uav_pairset_20261011"
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--verify", action="store_true", help="Recompute and compare with an existing v2 manifest")
    args = parser.parse_args()
    out = args.out or args.root / "surfbev_b0_split_v2.json"
    payload = build(args.root)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.verify:
        if out.read_text() != rendered:
            raise SystemExit(f"v2 manifest verification failed: {out}")
    else:
        if out.exists():
            raise SystemExit(f"refusing to overwrite immutable v2 split: {out}")
        out.write_text(rendered)
    print(json.dumps({"manifest": str(out), "sha256": sha256(out), "counts": payload["counts"],
                      "exclusion_reasons_count": payload["exclusion_reasons_count"],
                      "audit": payload["audit"], "verify": args.verify}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
