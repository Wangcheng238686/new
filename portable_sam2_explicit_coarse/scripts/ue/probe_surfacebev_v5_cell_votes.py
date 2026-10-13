#!/usr/bin/env python3
"""Read-only per-world-cell, per-scene identity votes for R5 candidates."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from build_surfacebev_v4_postsplit import to_world
from probe_surfacebev_v5_physical_instances import rasterize


PACKAGE = Path(__file__).resolve().parents[2]
V3 = PACKAGE / "outputs/SURFBEV-MASK-V3-20261012"
R4 = PACKAGE / "outputs/SURFBEV-V5-PHYSICAL-INSTANCE-R4-20261012"
R5 = PACKAGE / "outputs/SURFBEV-V5-PHYSICAL-INSTANCE-R5-20261012"
OUT = PACKAGE / "outputs/SURFBEV-V5-PHYSICAL-INSTANCE-R6B-20261012"
V3_SHA = "8d3da5addd302451337954191dccbf1b76f588a0790f1073415e30439015922f"
R4_SHA = "a5346fa10ed2cb26c37c25174e1efe83c6b628600764c4f8ef3a4734eae04975"
R5_SHA = "74d1ebdf9551b4954aa0aec82b4d2ec3a3aa4a91ab63021cb72e772e15d3cb98"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3", type=Path, default=V3)
    parser.add_argument("--r4", type=Path, default=R4)
    parser.add_argument("--r5", type=Path, default=R5)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(args.out)
    if (sha(args.v3 / "manifest.jsonl") != V3_SHA or
            sha(args.r4 / "identity_consensus.jsonl") != R4_SHA or
            sha(args.r5 / "component_overlap.jsonl") != R5_SHA):
        raise ValueError("registered input hash changed")
    r4 = [json.loads(x) for x in (args.r4 / "identity_consensus.jsonl").read_text().splitlines()]
    r5 = [json.loads(x) for x in (args.r5 / "component_overlap.jsonl").read_text().splitlines()]
    candidate_lookup = {(x["global_instance_id"], x["component_label"]): x for x in r4}
    if len(r5) != 19:
        raise ValueError("expected 19 cases")
    parents = {x["parent_global_id"] for x in r5}
    source = [json.loads(x) for x in (args.v3 / "manifest.jsonl").read_text().splitlines()]
    if len(source) != 1200 or len({x["scene_dir"] for x in source}) != 1200:
        raise ValueError("expected 1200 unique scenes")
    parent_entries = collections.defaultdict(list)
    for rec in source:
        if rec["split"] not in ("train", "val"):
            continue
        meta = json.loads((Path(rec["source_png"]).parent / "instances.json").read_text())
        relevant = [(int(x["local_instance_id"]), int(x["global_instance_id"])) for x in meta["instances"]
                    if int(x["global_instance_id"]) in parents]
        if not relevant:
            continue
        if sha(Path(rec["v3_png"])) != rec["v3_sha256"]:
            raise ValueError(f"v3 label changed: {rec['scene_dir']}")
        with Image.open(rec["v3_png"]) as image:
            ids = np.asarray(image).copy()
        origin = to_world(rec, (0, 0))
        du = to_world(rec, (1, 0)) - origin
        dv = to_world(rec, (0, 1)) - origin
        for local, gid in relevant:
            ys, xs = np.where(ids == local)
            if len(xs):
                points = origin + xs[:, None] * du + ys[:, None] * dv
                parent_entries[gid].append(dict(scene_dir=rec["scene_dir"], split=rec["split"],
                                                local_id=local, origin=origin, points=points))
    raster_cache = {gid: rasterize(parent_entries[gid]) for gid in parents}
    scene_cases = []
    for item in r5:
        parent, target, label = item["parent_global_id"], item["target_global_id"], item["source_component_label"]
        case = candidate_lookup[(parent, label)]
        if case["identity_correction_candidate"] != target:
            raise ValueError("R4/R5 target mismatch")
        raster = raster_cache[parent]
        low = np.asarray(raster["origin_xy_m"], dtype=np.int64)
        for scene_record in item["scene_results"]:
            entry = next(x for x in parent_entries[parent] if x["scene_dir"] == scene_record["scene_dir"])
            xy = np.floor(entry["points"]).astype(np.int64) - low
            selected = raster["labels"][xy[:, 1], xy[:, 0]] == label
            cells = np.unique(np.floor(entry["points"][selected]).astype(np.int64), axis=0)
            if len(cells) != scene_record["world_cells"]:
                raise ValueError("R5 source cell mismatch")
            scene_cases.append(dict(parent=parent, target=target, component_label=label,
                                    source_scene=entry["scene_dir"], excluded_scenes=set(case["scene_dirs"]),
                                    split=entry["split"], cells=cells,
                                    in_bounds=np.zeros(len(cells), np.int32),
                                    exact=np.zeros(len(cells), np.int32),
                                    near=np.zeros(len(cells), np.int32),
                                    reference_scenes=[]))
    if len(scene_cases) != sum(x["source_scenes"] for x in r5):
        raise ValueError("source case count mismatch")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    for rec in source:
        if rec["split"] not in ("train", "val"):
            continue
        origin = to_world(rec, (0, 0))
        du = to_world(rec, (1, 0)) - origin
        dv = to_world(rec, (0, 1)) - origin
        inv = np.linalg.inv(np.column_stack((du, dv)))
        active = []
        for case in scene_cases:
            if case["split"] != rec["split"] or rec["scene_dir"] in case["excluded_scenes"]:
                continue
            uv = (case["cells"] + .5 - origin) @ inv.T
            pix = np.rint(uv).astype(np.int32)
            valid = (pix[:, 0] >= 0) & (pix[:, 0] < 512) & (pix[:, 1] >= 0) & (pix[:, 1] < 512)
            if np.any(valid):
                active.append((case, pix, valid))
        if not active:
            continue
        if sha(Path(rec["v3_png"])) != rec["v3_sha256"]:
            raise ValueError(f"v3 label changed: {rec['scene_dir']}")
        with Image.open(rec["v3_png"]) as image:
            if image.mode != "L" or image.size != (512, 512):
                raise ValueError(f"invalid PNG: {rec['scene_dir']}")
            ids = np.asarray(image).copy()
        meta = json.loads((Path(rec["source_png"]).parent / "instances.json").read_text())
        target_local = collections.defaultdict(list)
        for row in meta["instances"]:
            target_local[int(row["global_instance_id"])].append(int(row["local_instance_id"]))
        target_masks = {}
        for case, _, _ in active:
            if case["target"] not in target_masks:
                mask = np.isin(ids, target_local[case["target"]]).astype(np.uint8) if target_local[case["target"]] else np.zeros_like(ids)
                target_masks[case["target"]] = (mask, cv2.dilate(mask, kernel))
        for case, pix, valid in active:
            x, y = pix[valid, 0], pix[valid, 1]
            mask, widened = target_masks[case["target"]]
            exact = mask[y, x].astype(np.int32)
            near = widened[y, x].astype(np.int32)
            case["in_bounds"][valid] += 1
            case["exact"][valid] += exact
            case["near"][valid] += near
            case["reference_scenes"].append(dict(scene_dir=rec["scene_dir"], in_bounds_cells=int(valid.sum()),
                                                 exact_fraction=round(float(exact.mean()), 6),
                                                 near_fraction=round(float(near.mean()), 6)))
    rows = []
    for case in scene_cases:
        seen = case["in_bounds"]
        enough = seen >= 3
        adequate_fraction = float(enough.mean())
        near_fraction = np.divide(case["near"], seen, out=np.zeros(len(seen), np.float64), where=seen > 0)
        exact_fraction = np.divide(case["exact"], seen, out=np.zeros(len(seen), np.float64), where=seen > 0)
        consistent = float((near_fraction[enough] >= .90).mean()) if np.any(enough) else 0.0
        refs = case["reference_scenes"]
        gate = len(refs) >= 3 and adequate_fraction >= .98 and consistent >= .98
        rows.append(dict(parent_global_id=case["parent"], target_global_id=case["target"],
                         component_label=case["component_label"], source_scene=case["source_scene"],
                         split=case["split"], world_cells=len(seen), reference_scenes=len(refs),
                         fraction_cells_with_3_views=round(adequate_fraction, 6),
                         fraction_3view_cells_near_target_90pct=round(consistent, 6),
                         exact_vote_fraction_median=round(float(np.median(exact_fraction[enough])), 6) if np.any(enough) else 0.0,
                         near_vote_fraction_median=round(float(np.median(near_fraction[enough])), 6) if np.any(enough) else 0.0,
                         min_reference_scene_near_fraction=min(x["near_fraction"] for x in refs) if refs else None,
                         gate="PASS" if gate else "HOLD", reference_scene_results=refs))
    args.out.mkdir(parents=True)
    manifest = args.out / "cell_votes.jsonl"
    manifest.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows))
    case_counts = collections.Counter(x["gate"] for x in rows)
    pair_gates = collections.defaultdict(list)
    for row in rows:
        pair_gates[(row["parent_global_id"], row["target_global_id"])].append(row["gate"])
    summary = dict(protocol="SURFBEV-V5-PHYSICAL-INSTANCE-R6B-20261012", status="DIAGNOSTIC_ONLY",
                   v3_manifest_sha256=V3_SHA, r4_manifest_sha256=R4_SHA, r5_manifest_sha256=R5_SHA,
                   source_components=len(rows), gate_counts=dict(case_counts),
                   all_source_scenes_passing_pairs=sum(all(g == "PASS" for g in gates) for gates in pair_gates.values()),
                   held_parent_target_pairs=sum(not all(g == "PASS" for g in gates) for gates in pair_gates.values()),
                   manifest_sha256=sha(manifest), note="All 1m cells voted across same-split scenes; no labels changed.")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
