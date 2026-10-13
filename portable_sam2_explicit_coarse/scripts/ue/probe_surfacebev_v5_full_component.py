#!/usr/bin/env python3
"""Read-only full-component world-cell overlap for candidate ID corrections."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from build_surfacebev_v4_postsplit import to_world
from probe_surfacebev_v5_physical_instances import rasterize


PACKAGE = Path(__file__).resolve().parents[2]
V3 = PACKAGE / "outputs/SURFBEV-MASK-V3-20261012"
R4 = PACKAGE / "outputs/SURFBEV-V5-PHYSICAL-INSTANCE-R4-20261012"
OUT = PACKAGE / "outputs/SURFBEV-V5-PHYSICAL-INSTANCE-R5-20261012"
V3_SHA = "8d3da5addd302451337954191dccbf1b76f588a0790f1073415e30439015922f"
R4_SHA = "a5346fa10ed2cb26c37c25174e1efe83c6b628600764c4f8ef3a4734eae04975"
OFFSETS = tuple((dx, dy) for dx in range(-3, 4) for dy in range(-3, 4) if dx*dx + dy*dy <= 9)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def world_cell_set(points):
    return set(map(tuple, np.floor(points).astype(np.int64)))


def overlap(source, reference):
    raw = sum(point in reference for point in source)
    near = sum(any((point[0]+dx, point[1]+dy) in reference for dx, dy in OFFSETS) for point in source)
    return round(raw/len(source), 6), round(near/len(source), 6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3", type=Path, default=V3)
    parser.add_argument("--r4", type=Path, default=R4)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(args.out)
    if sha(args.v3 / "manifest.jsonl") != V3_SHA or sha(args.r4 / "identity_consensus.jsonl") != R4_SHA:
        raise ValueError("registered input hash changed")
    candidates = [json.loads(x) for x in (args.r4 / "identity_consensus.jsonl").read_text().splitlines()
                  if json.loads(x)["identity_correction_candidate"] is not None]
    if len(candidates) != 19:
        raise ValueError("expected 19 candidates")
    parents = {x["global_instance_id"] for x in candidates}
    targets = {x["identity_correction_candidate"] for x in candidates}
    source = [json.loads(x) for x in (args.v3 / "manifest.jsonl").read_text().splitlines()]
    if len(source) != 1200 or len({x["scene_dir"] for x in source}) != 1200:
        raise ValueError("expected 1200 unique scenes")
    parent_entries = collections.defaultdict(list)
    target_cells = collections.defaultdict(dict)
    for rec in source:
        if rec["split"] not in ("train", "val"):
            continue
        meta = json.loads((Path(rec["source_png"]).parent / "instances.json").read_text())
        relevant = [(int(x["local_instance_id"]), int(x["global_instance_id"])) for x in meta["instances"]
                    if int(x["global_instance_id"]) in parents | targets]
        if not relevant:
            continue
        if sha(Path(rec["v3_png"])) != rec["v3_sha256"]:
            raise ValueError(f"v3 label changed: {rec['scene_dir']}")
        with Image.open(rec["v3_png"]) as image:
            if image.mode != "L" or image.size != (512, 512):
                raise ValueError(f"invalid PNG: {rec['scene_dir']}")
            ids = np.asarray(image).copy()
        origin = to_world(rec, (0, 0))
        du = to_world(rec, (1, 0)) - origin
        dv = to_world(rec, (0, 1)) - origin
        for local, gid in relevant:
            ys, xs = np.where(ids == local)
            if not len(xs):
                continue
            points = origin + xs[:, None] * du + ys[:, None] * dv
            if gid in parents:
                parent_entries[gid].append(dict(scene_dir=rec["scene_dir"], split=rec["split"],
                                                local_id=local, origin=origin, points=points))
            if gid in targets:
                target_cells[gid][rec["scene_dir"]] = world_cell_set(points)
    rows = []
    raster_cache = {gid: rasterize(parent_entries[gid]) for gid in parents}
    for case in candidates:
        parent, target = case["global_instance_id"], case["identity_correction_candidate"]
        raster = raster_cache[parent]
        low = np.asarray(raster["origin_xy_m"], dtype=np.int64)
        label = case["component_label"]
        refs = {scene: cells for scene, cells in target_cells[target].items() if scene not in case["scene_dirs"]}
        ref_union = set().union(*refs.values()) if refs else set()
        scene_results = []
        for entry in parent_entries[parent]:
            if entry["scene_dir"] not in case["scene_dirs"]:
                continue
            points = entry["points"]
            xy = np.floor(points).astype(np.int64) - low
            selected = raster["labels"][xy[:, 1], xy[:, 0]] == label
            if not np.any(selected):
                raise ValueError(f"lost component pixels: {parent} {entry['scene_dir']}")
            cells = world_cell_set(points[selected])
            raw, near = overlap(cells, ref_union) if ref_union else (0.0, 0.0)
            scene_results.append(dict(scene_dir=entry["scene_dir"], split=entry["split"],
                                      local_instance_id=entry["local_id"],
                                      component_pixels=int(selected.sum()), world_cells=len(cells),
                                      raw_overlap=raw, overlap_3m=near))
        if len(scene_results) != len(case["scene_dirs"]):
            raise ValueError(f"source scene count mismatch: {parent} -> {target}")
        min_raw = min(x["raw_overlap"] for x in scene_results)
        min_near = min(x["overlap_3m"] for x in scene_results)
        gate = len(refs) >= 3 and min_raw >= .90 and min_near >= .98
        rows.append(dict(parent_global_id=parent, target_global_id=target,
                         source_component_label=label, reference_scenes=len(refs),
                         target_world_cells=len(ref_union), source_scenes=len(scene_results),
                         min_raw_overlap=min_raw, min_overlap_3m=min_near,
                         geometry_gate="PASS" if gate else "HOLD",
                         scene_results=scene_results))
    args.out.mkdir(parents=True)
    manifest = args.out / "component_overlap.jsonl"
    manifest.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows))
    counts = collections.Counter(x["geometry_gate"] for x in rows)
    summary = dict(protocol="SURFBEV-V5-PHYSICAL-INSTANCE-R5-20261012", status="DIAGNOSTIC_ONLY",
                   v3_manifest_sha256=V3_SHA, r4_manifest_sha256=R4_SHA,
                   candidate_cases=len(rows), geometry_gate_counts=dict(counts),
                   known_170002_gate=next(x["geometry_gate"] for x in rows if x["parent_global_id"] == 170002),
                   manifest_sha256=sha(manifest),
                   note="1m/3m world-grid pixel overlap, no labels changed. Geometry PASS is not dataset release.")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
