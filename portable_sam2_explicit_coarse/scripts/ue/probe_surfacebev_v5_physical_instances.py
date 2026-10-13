#!/usr/bin/env python3
"""Read-only cross-scene ground-plane mosaics for physical-instance split discovery."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from build_surfacebev_v4_postsplit import to_world


PACKAGE = Path(__file__).resolve().parents[2]
V3 = PACKAGE / "outputs/SURFBEV-MASK-V3-20261012"
CANDIDATES = PACKAGE / "outputs/SURFBEV-DISCONNECT-QA-20261012-R3/candidates.csv"
OUT = PACKAGE / "outputs/SURFBEV-V5-PHYSICAL-INSTANCE-R2-20261012"
V3_SHA = "8d3da5addd302451337954191dccbf1b76f588a0790f1073415e30439015922f"
KNOWN_POS = (1874325323, 170002)
KNOWN_NEG = (1571206454, 1508745546)
GRID_M = 1.0
DILATE_M = 3


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def read_ids(path):
    with Image.open(path) as image:
        if image.mode != "L" or image.size != (512, 512):
            raise ValueError(f"invalid v3 ID PNG: {path}")
        return np.asarray(image).copy()


def rasterize(entries, shuffled=False):
    if shuffled:
        rng = np.random.default_rng(20261012)
        origins = np.stack([e["origin"] for e in entries])
        order = rng.permutation(len(entries))
        point_sets = [e["points"] + origins[order[i]] - e["origin"] for i, e in enumerate(entries)]
    else:
        point_sets = [e["points"] for e in entries]
    all_points = np.concatenate(point_sets)
    low = np.floor(all_points.min(axis=0) / GRID_M).astype(np.int64) - 8
    high = np.ceil(all_points.max(axis=0) / GRID_M).astype(np.int64) + 9
    width, height = (high - low).astype(int)
    if width <= 0 or height <= 0 or width * height > 8_000_000:
        raise ValueError(f"invalid mosaic size {width}x{height}")
    seen = np.zeros((height, width), np.uint16)
    xy_indices = []
    for points in point_sets:
        xy = np.floor(points / GRID_M).astype(np.int64) - low
        unique = np.unique(xy[:, 1] * width + xy[:, 0])
        seen.flat[unique] += 1
        xy_indices.append(xy)
    occupancy = (seen > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*DILATE_M+1, 2*DILATE_M+1))
    dilated = cv2.dilate(occupancy, kernel)
    n, labels, stats, centers = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    comps = []
    for index in range(1, n):
        raw = int((labels[occupancy > 0] == index).sum())
        if raw == 0:
            continue
        supports = []
        for e, xy in zip(entries, xy_indices):
            subset = labels[xy[:, 1], xy[:, 0]] == index
            if np.any(subset):
                supports.append(dict(scene_dir=e["scene_dir"], split=e["split"], pixels=int(subset.sum()),
                                     fraction=round(float(subset.mean()), 6)))
        cx, cy = centers[index]
        comps.append(dict(label=index, raw_cells=raw,
                          dilated_cells=int(stats[index, cv2.CC_STAT_AREA]),
                          center_xy_m=[round(float((cx+low[0])*GRID_M), 3),
                                       round(float((cy+low[1])*GRID_M), 3)],
                          supporting_scenes=len(supports),
                          scene_support=supports))
    comps.sort(key=lambda x:(x["raw_cells"], x["supporting_scenes"]), reverse=True)
    multi = int((seen >= 2).sum())
    overlap_ratio = multi / max(1, int((seen > 0).sum()))
    return dict(occupancy=occupancy, labels=labels, seen=seen, components=comps,
                overlap_ratio=overlap_ratio, origin_xy_m=(low * GRID_M).tolist(),
                image_shape=[height, width])


def save_mosaic(path, result):
    occupancy = result["occupancy"]
    labels = result["labels"]
    canvas = np.zeros((*occupancy.shape, 3), np.uint8)
    palette = ((255, 55, 155), (30, 210, 235), (235, 190, 45), (140, 230, 80),
               (180, 110, 255), (255, 120, 60))
    for i, comp in enumerate(result["components"]):
        canvas[labels == comp["label"]] = (np.asarray(palette[i % len(palette)]) * .35).astype(np.uint8)
        canvas[(labels == comp["label"]) & (occupancy > 0)] = palette[i % len(palette)]
    if max(canvas.shape[:2]) > 2048:
        scale = 2048 / max(canvas.shape[:2])
        canvas = cv2.resize(canvas, (round(canvas.shape[1]*scale), round(canvas.shape[0]*scale)),
                            interpolation=cv2.INTER_AREA)
    Image.fromarray(canvas).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3", type=Path, default=V3)
    parser.add_argument("--candidates", type=Path, default=CANDIDATES)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing to overwrite {args.out}")
    if sha(args.v3 / "manifest.jsonl") != V3_SHA:
        raise ValueError("v3 manifest hash mismatch")
    candidate_rows = list(csv.DictReader(args.candidates.open()))
    candidate_ids = {int(r["global_instance_id"]) for r in candidate_rows}
    if len(candidate_ids) != 36 or not set(KNOWN_POS + KNOWN_NEG) <= candidate_ids:
        raise ValueError("candidate global ID registry changed")
    source = [json.loads(line) for line in (args.v3 / "manifest.jsonl").read_text().splitlines() if line]
    if len(source) != 1200:
        raise ValueError("expected 1200 scenes")
    by_global = collections.defaultdict(list)
    for rec in source:
        if rec["split"] not in ("train", "val"):
            continue
        if sha(Path(rec["v3_png"])) != rec["v3_sha256"]:
            raise ValueError(f"v3 label hash changed: {rec['scene_dir']}")
        meta = json.loads((Path(rec["source_png"]).parent / "instances.json").read_text())
        lookup = {int(x["local_instance_id"]): int(x["global_instance_id"]) for x in meta["instances"]}
        relevant = {lid: gid for lid, gid in lookup.items() if gid in candidate_ids}
        if not relevant:
            continue
        ids = read_ids(rec["v3_png"])
        origin = to_world(rec, (0, 0))
        du = to_world(rec, (1, 0)) - origin
        dv = to_world(rec, (0, 1)) - origin
        for lid, gid in relevant.items():
            ys, xs = np.where(ids == lid)
            if len(xs) == 0:
                continue
            points = origin + xs[:, None] * du + ys[:, None] * dv
            by_global[gid].append(dict(scene_dir=rec["scene_dir"], split=rec["split"],
                                       local_id=lid, origin=origin, points=points,
                                       visible_pixels=len(xs)))
    if set(by_global) != candidate_ids:
        raise ValueError("missing registered candidate global")
    (args.out / "mosaics").mkdir(parents=True)
    rows = []
    for gid in sorted(candidate_ids):
        entries = by_global[gid]
        correct = rasterize(entries)
        sham = rasterize(entries, shuffled=True)
        major = [c for c in correct["components"] if c["raw_cells"] >= 64 and c["supporting_scenes"] >= 2]
        row = dict(global_instance_id=gid, splits=sorted({e["split"] for e in entries}),
                   visible_scenes=len(entries), visible_pixels=sum(e["visible_pixels"] for e in entries),
                   mosaic_shape_yx_cells=correct["image_shape"],
                   correct_overlap_ratio=round(correct["overlap_ratio"], 6),
                   sham_overlap_ratio=round(sham["overlap_ratio"], 6),
                   all_component_count=len(correct["components"]), major_component_count=len(major),
                   components=[{k:v for k,v in c.items() if k != "scene_support"} for c in correct["components"]],
                   scene_assignments={e["scene_dir"]: [dict(component_label=c["label"],
                       pixel_fraction=next((s["fraction"] for s in c["scene_support"] if s["scene_dir"] == e["scene_dir"]), 0.0))
                       for c in correct["components"]] for e in entries})
        rows.append(row)
        save_mosaic(args.out / "mosaics" / f"global_{gid}.png", correct)
    manifest = args.out / "global_probe.jsonl"
    manifest.write_text("".join(json.dumps(row, sort_keys=True, default=json_scalar) + "\n" for row in rows))
    anchor = {str(g): next(x for x in rows if x["global_instance_id"] == g)["major_component_count"]
              for g in KNOWN_POS + KNOWN_NEG}
    summary = dict(protocol="SURFBEV-V5-PHYSICAL-INSTANCE-R2-20261012", status="DIAGNOSTIC_ONLY",
                   v3_manifest_sha256=V3_SHA, candidate_csv_sha256=sha(args.candidates),
                   input_candidate_globals=len(candidate_ids), output_global_rows=len(rows),
                   known_anchor_major_component_counts=anchor,
                   n_major_two_or_more=sum(x["major_component_count"] >= 2 for x in rows),
                   correct_overlap_mean=float(np.mean([x["correct_overlap_ratio"] for x in rows])),
                   sham_overlap_mean=float(np.mean([x["sham_overlap_ratio"] for x in rows])),
                   probe_sha256=sha(manifest),
                   note="World XY uses Z=0 approximation for candidate association only. No labels changed, no AP claim.")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
