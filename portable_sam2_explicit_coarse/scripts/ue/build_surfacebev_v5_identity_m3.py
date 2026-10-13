#!/usr/bin/env python3
"""Materialize a 1200-scene, unreleased V5-M3 identity-correction candidate."""
import argparse
import collections
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from build_surfacebev_v4_postsplit import to_world
from probe_surfacebev_v5_physical_instances import rasterize


PACKAGE = Path(__file__).resolve().parents[2]
V3 = PACKAGE / "outputs/SURFBEV-MASK-V3-20261012"
V4 = PACKAGE / "outputs/SURFBEV-V4-POSTSPLIT-20261012-R3"
R4 = PACKAGE / "outputs/SURFBEV-V5-PHYSICAL-INSTANCE-R4-20261012"
R6B = PACKAGE / "outputs/SURFBEV-V5-PHYSICAL-INSTANCE-R6B-20261012"
OUT = PACKAGE / "outputs/SURFBEV-V5-IDENTITY-M3-20261012"
V3_SHA = "8d3da5addd302451337954191dccbf1b76f588a0790f1073415e30439015922f"
V4_SHA = "49b7f9268db3f767cef4d5cd56265d674da59c54808ca9650140d2d3c5e1884f"
R4_SHA = "a5346fa10ed2cb26c37c25174e1efe83c6b628600764c4f8ef3a4734eae04975"
R6B_SHA = "910a1525c0b04566936032d8233b7dc5e2fc9342a1e5a227073f2be649294bc7"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_ids(path):
    with Image.open(path) as image:
        if image.mode != "L" or image.size != (512, 512):
            raise ValueError(f"invalid ID PNG: {path}")
        return np.asarray(image).copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3", type=Path, default=V3)
    parser.add_argument("--v4", type=Path, default=V4)
    parser.add_argument("--r4", type=Path, default=R4)
    parser.add_argument("--r6b", type=Path, default=R6B)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(args.out)
    locked = ((args.v3 / "manifest.jsonl", V3_SHA), (args.v4 / "manifest.jsonl", V4_SHA),
              (args.r4 / "identity_consensus.jsonl", R4_SHA), (args.r6b / "cell_votes.jsonl", R6B_SHA))
    for path, expected in locked:
        if sha(path) != expected:
            raise ValueError(f"registered input changed: {path}")
    v3 = [json.loads(x) for x in (args.v3 / "manifest.jsonl").read_text().splitlines()]
    v4 = {x["scene_dir"]: x for x in map(json.loads, (args.v4 / "manifest.jsonl").read_text().splitlines())}
    if len(v3) != 1200 or len(v4) != 1200 or len({x["scene_dir"] for x in v3}) != 1200:
        raise ValueError("expected 1200 unique scenes in each manifest")
    r4 = [json.loads(x) for x in (args.r4 / "identity_consensus.jsonl").read_text().splitlines()]
    r6 = [json.loads(x) for x in (args.r6b / "cell_votes.jsonl").read_text().splitlines()]
    pair_gates = collections.defaultdict(list)
    for row in r6:
        pair_gates[(row["parent_global_id"], row["target_global_id"])].append(row["gate"])
    accepted = [x for x in r4 if x["identity_correction_candidate"] is not None and
                all(g == "PASS" for g in pair_gates[(x["global_instance_id"], x["identity_correction_candidate"])])]
    if len(accepted) != 18 or any(not pair_gates[(x["global_instance_id"], x["identity_correction_candidate"])]
                                  for x in accepted):
        raise ValueError("expected exactly 18 all-source-scene PASS identity pairs")
    if any(x["global_instance_id"] == 532000 for x in accepted):
        raise ValueError("registered HOLD pair leaked")
    parents = {x["global_instance_id"] for x in accepted}
    parent_entries = collections.defaultdict(list)
    original_meta = {}
    for rec in v3:
        meta_path = Path(rec["source_png"]).parent / "instances.json"
        meta = json.loads(meta_path.read_text())
        original_meta[rec["scene_dir"]] = (meta, sha(meta_path))
        if rec["split"] not in ("train", "val"):
            continue
        relevant = [(int(x["local_instance_id"]), int(x["global_instance_id"])) for x in meta["instances"]
                    if int(x["global_instance_id"]) in parents]
        if not relevant:
            continue
        if sha(Path(rec["v3_png"])) != rec["v3_sha256"]:
            raise ValueError(f"v3 label changed: {rec['scene_dir']}")
        ids = read_ids(Path(rec["v3_png"]))
        origin = to_world(rec, (0, 0))
        du = to_world(rec, (1, 0)) - origin
        dv = to_world(rec, (0, 1)) - origin
        for local, gid in relevant:
            ys, xs = np.where(ids == local)
            if len(xs):
                parent_entries[gid].append(dict(scene_dir=rec["scene_dir"], split=rec["split"],
                                                local_id=local, origin=origin,
                                                points=origin + xs[:, None]*du + ys[:, None]*dv,
                                                xy=np.stack((xs, ys), axis=-1)))
    raster_cache = {gid: rasterize(parent_entries[gid]) for gid in parents}
    corrections = collections.defaultdict(list)
    for case in accepted:
        parent, target, label = case["global_instance_id"], case["identity_correction_candidate"], case["component_label"]
        raster = raster_cache[parent]
        low = np.asarray(raster["origin_xy_m"], dtype=np.int64)
        for scene in case["scene_dirs"]:
            entry = next((x for x in parent_entries[parent] if x["scene_dir"] == scene), None)
            if entry is None:
                raise ValueError(f"source scene absent: {parent} {scene}")
            xy = np.floor(entry["points"]).astype(np.int64) - low
            selected = raster["labels"][xy[:, 1], xy[:, 0]] == label
            if not np.any(selected):
                raise ValueError(f"empty component: {parent} {scene}")
            mask = np.zeros((512, 512), bool)
            mask[entry["xy"][selected, 1], entry["xy"][selected, 0]] = True
            corrections[scene].append(dict(parent_global_id=parent, target_global_id=target,
                                            source_local_id=entry["local_id"], component_label=label,
                                            mask=mask, pixels=int(mask.sum())))
    if sum(map(len, corrections.values())) != 24:
        raise ValueError("expected 24 source-scene components")

    (args.out / "labels").mkdir(parents=True)
    (args.out / "metadata").mkdir()
    rows = []
    splits_for_global = collections.defaultdict(set)
    changed = 0
    tiny_drop_count = 0
    tiny_drop_scenes = []
    for rec in v3:
        scene = rec["scene_dir"]
        v4rec = v4[scene]
        if v4rec["split"] != rec["split"] or v4rec["v3_sha256"] != rec["v3_sha256"]:
            raise ValueError(f"v3/v4 manifest mismatch: {scene}")
        v3path, v4path = Path(rec["v3_png"]), Path(v4rec["v4_png"])
        if sha(v3path) != rec["v3_sha256"] or sha(v4path) != v4rec["v4_sha256"]:
            raise ValueError(f"source label changed: {scene}")
        original = read_ids(v3path)
        base = read_ids(v4path)
        if not np.array_equal(original > 0, base > 0):
            raise ValueError(f"v4 foreground changed: {scene}")
        meta, meta_sha = original_meta[scene]
        local_to_global = {int(x["local_instance_id"]): int(x["global_instance_id"]) for x in meta["instances"]}
        if len(local_to_global) != len(meta["instances"]):
            raise ValueError(f"duplicate source local ID: {scene}")
        local_to_global.update({int(k): int(v) for k, v in v4rec["child_local_to_global"].items()})
        used = set(local_to_global) | set(map(int, np.unique(base)))
        output = base.copy()
        actions = []
        occupied = np.zeros((512, 512), bool)
        initially_visible_globals = {local_to_global[int(i)] for i in np.unique(base[base > 0])}
        target_to_new_local = {}
        for correction in sorted(corrections[scene], key=lambda x:(x["parent_global_id"], x["target_global_id"])):
            mask = correction["mask"]
            local = correction["source_local_id"]
            target = correction["target_global_id"]
            if (np.any(occupied & mask) or not np.all(original[mask] == local) or
                    not np.all(base[mask] == local) or local_to_global.get(local) != correction["parent_global_id"]):
                raise ValueError(f"correction source mismatch/overlap: {scene}")
            if target in initially_visible_globals:
                raise ValueError(f"target already visible in source scene: {scene} {target}")
            free = target_to_new_local.get(target)
            if free is None:
                free = next((i for i in range(1, 256) if i not in used), None)
                if free is None:
                    raise ValueError(f"no free local ID: {scene}")
                used.add(free)
                target_to_new_local[target] = free
            output[mask] = free
            local_to_global[free] = target
            occupied |= mask
            actions.append(dict(action="IDENTITY_CORRECTION", parent_global_id=correction["parent_global_id"],
                                target_global_id=target, source_local_id=local, new_local_id=free,
                                component_label=correction["component_label"], pixels=correction["pixels"]))
        removed = np.zeros((512, 512), bool)
        for target, new_local in target_to_new_local.items():
            ys, xs = np.where(output == new_local)
            border_distance = min(int(xs.min()), int(ys.min()), 511-int(xs.max()), 511-int(ys.max()))
            if len(xs) < 16 and border_distance <= 1:
                removed[ys, xs] = True
                output[ys, xs] = 0
                tiny_drop_count += len(xs)
                tiny_drop_scenes.append(scene)
                for action in actions:
                    if action["new_local_id"] == new_local:
                        action["final_status"] = "DROPPED_TINY_NEAR_EDGE_FRAGMENT"
                        action["dropped_pixels"] = len(xs)
            else:
                for action in actions:
                    if action["new_local_id"] == new_local:
                        action["final_status"] = "KEPT"
                        action["dropped_pixels"] = 0
        if not np.array_equal((original > 0) & ~removed, output > 0) or not np.array_equal(base[~occupied], output[~occupied]):
            raise ValueError(f"pixel conservation failed: {scene}")
        dest = args.out / "labels" / (scene + ".png")
        if actions:
            Image.fromarray(output, "L").save(dest)
        else:
            shutil.copyfile(v4path, dest)
        instances = []
        for local in np.unique(output):
            local = int(local)
            if local == 0:
                continue
            if local not in local_to_global:
                raise ValueError(f"unmapped visible local ID {local}: {scene}")
            ys, xs = np.where(output == local)
            gid = local_to_global[local]
            instances.append(dict(local_instance_id=local, global_instance_id=gid,
                                  area_px=int(len(xs)), bbox_xyxy=[int(xs.min()), int(ys.min()),
                                                                   int(xs.max())+1, int(ys.max())+1]))
            if rec["split"] != "excluded":
                splits_for_global[gid].add(rec["split"])
        metadata = args.out / "metadata" / (scene + ".json")
        metadata.write_text(json.dumps(dict(scene_dir=scene, split=rec["split"],
                                            source_instances_json_sha256=meta_sha,
                                            visible_instances=instances), sort_keys=True) + "\n")
        changed += int((output != base).sum())
        rows.append(dict(scene_dir=scene, split=rec["split"], gx=rec["gx"], gy=rec["gy"],
                         satellite_rgb=rec["satellite_rgb"], v3_png=str(v3path), v3_sha256=rec["v3_sha256"],
                         v4_png=str(v4path), v4_sha256=v4rec["v4_sha256"],
                         v5_png=str(dest.resolve()), v5_sha256=sha(dest),
                         v5_metadata=str(metadata.resolve()), v5_metadata_sha256=sha(metadata),
                         v4_action=v4rec["action"], identity_actions=actions,
                         visible_instances=len(instances), foreground_pixels=int((output > 0).sum())))
    cross = {str(k): sorted(v) for k, v in splits_for_global.items() if len(v) > 1}
    if cross:
        raise ValueError(f"global ID crosses supervised splits: {list(cross.items())[:3]}")
    expected_tiny_scene = "Coord_002894_pt_gx_-7_gy_-2_s_020_gx_-7_gy_-2_s_20_X_-137500_Y_-22500"
    if tiny_drop_count != 1 or tiny_drop_scenes != [expected_tiny_scene]:
        raise ValueError(f"unexpected tiny fragment drops: {tiny_drop_count} {tiny_drop_scenes}")
    manifest = args.out / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows))
    summary = dict(protocol="SURFBEV-V5-IDENTITY-M3-20261012", status="BUILT_UNREVIEWED",
                   reader_connected=False, scenes=len(rows),
                   split_counts=dict(collections.Counter(x["split"] for x in rows)),
                   v3_manifest_sha256=V3_SHA, v4_manifest_sha256=V4_SHA,
                   r4_identity_sha256=R4_SHA, r6b_cell_votes_sha256=R6B_SHA,
                   all_source_scenes_passing_pairs=len(accepted),
                   identity_source_scene_components=sum(len(x["identity_actions"]) for x in rows),
                   identity_scene_count=sum(bool(x["identity_actions"]) for x in rows),
                   identity_pixels_changed_from_v4=changed,
                   tiny_new_instance_fragment_pixels_dropped=tiny_drop_count,
                   tiny_new_instance_fragment_scenes=tiny_drop_scenes,
                   v4_split_scenes=sum(x["v4_action"] == "SPLIT" for x in rows),
                   known_hold=dict(identity_pair="532000 -> 402005", v4_excluded_scenes=6,
                                   connected_risk_globals=95),
                   manifest_sha256=sha(manifest),
                   note="Full 1200-scene candidate with metadata; not semantically released, not used by training.")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
