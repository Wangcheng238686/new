#!/usr/bin/env python3
"""Materialize immutable v3 satellite ID rasters by one frozen data-side rule.

No model ignore list is created. Source PNGs remain read-only; every output
pixel change is audited against the complete set of removed local IDs.
"""
import argparse
import collections
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image


PACKAGE = Path(__file__).resolve().parents[2]
ROOT = PACKAGE / "data/ue_raw_hero_uav_pairset_20261011"
OUT = PACKAGE / "outputs/SURFBEV-MASK-V3-20261012"
PAIR_SHA = "f15cd8f5b8d10a69d721546d6cfa1f2cce58af2861d57b1357c5e9ff1e1fe62a"
SPLIT_SHA = "17f49430b15037c8da25c63a1bf7fe4d510cf5ff153263d2feba62c4393441d5"
RULE = "touch_border AND (area_px_512 < 16 OR min_bbox_side_px_512 <= 2)"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_ids(path):
    image = Image.open(path)
    if image.mode != "L" or image.size != (512, 512):
        raise ValueError(f"expected 512x512 grayscale L ID PNG: {path}, got {image.mode} {image.size}")
    return np.asarray(image, dtype=np.uint8)


def candidate_details(ids):
    removed = []
    for raw_id in np.unique(ids):
        lid = int(raw_id)
        if lid == 0:
            continue
        ys, xs = np.where(ids == lid)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        area = int(len(xs))
        border = x0 == 0 or y0 == 0 or x1 == 512 or y1 == 512
        if border and (area < 16 or min(x1-x0, y1-y0) <= 2):
            removed.append(dict(local_instance_id=lid, area_px_512=area,
                                bbox=[x0, y0, x1, y1], touches_border=True))
    return removed


def clean(ids, removed):
    out = ids.copy()
    for record in removed:
        out[ids == record["local_instance_id"]] = 0
    expected_diff = np.isin(ids, [r["local_instance_id"] for r in removed])
    actual_diff = ids != out
    if not np.array_equal(actual_diff, expected_diff):
        raise RuntimeError("changed pixels do not equal the selected ID pixels")
    if np.any(out[actual_diff] != 0):
        raise RuntimeError("removed ID pixel did not become background")
    if candidate_details(out):
        raise RuntimeError("v3 raster still contains rule-matching IDs")
    return out, int(actual_diff.sum())


def load_contract(root):
    accepted_file = root / "accepted_pairs.jsonl"
    split_file = root / "surfbev_b0_split_v2.json"
    if sha(accepted_file) != PAIR_SHA or sha(split_file) != SPLIT_SHA:
        raise RuntimeError("input pair/split manifest SHA differs from registered v2")
    accepted = {json.loads(line)["scene_dir"] for line in accepted_file.read_text().splitlines() if line}
    split = json.loads(split_file.read_text())
    records = split["scenes"]
    if len(accepted) != 1200 or len(records) != 1200:
        raise RuntimeError("expected 1200 accepted/split records")
    if {r["scene_dir"] for r in records} != accepted:
        raise RuntimeError("split and accepted scenes differ")
    if collections.Counter(r["split"] for r in records) != dict(train=694, val=100, test=151, excluded=255):
        raise RuntimeError("unexpected split sizes")
    return records


def select_spotcheck(rows, n_each=12, seed=20261012):
    """Pick scene-level changed/unchanged controls without using test content."""
    rng = random.Random(seed)
    chosen = []
    for split in ("train", "val"):
        for changed in (True, False):
            candidates = [r for r in rows if r["split"] == split and bool(r["removed_ids"]) == changed]
            rng.shuffle(candidates)
            cell_count = collections.Counter()
            for _ in range(min(n_each, len(candidates))):
                best = min(enumerate(candidates), key=lambda ir: (cell_count[(ir[1]["gx"], ir[1]["gy"])], ir[0]))
                _, row = best
                candidates.remove(row)
                chosen.append(dict(split=split, changed=int(changed), scene_dir=row["scene_dir"],
                                   gx=row["gx"], gy=row["gy"], removed_ids="|".join(str(x["local_instance_id"]) for x in row["removed_ids"]),
                                   source_png=row["source_png"], v3_png=row["v3_png"],
                                   satellite_rgb=row["satellite_rgb"]))
                cell_count[(row["gx"], row["gy"])] += 1
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing v3 dataset: {args.out}")
    records = load_contract(args.root)
    labels = args.out / "labels"
    labels.mkdir(parents=True)
    manifest = []
    summary = collections.defaultdict(lambda: dict(scenes=0, changed_scenes=0,
                                                   input_instances=0, removed_instances=0,
                                                   removed_pixels=0, output_instances=0))
    for record in records:
        scene, split = record["scene_dir"], record["split"]
        sat = args.root / "scenes" / scene / "satellite"
        src = sat / "satellite_instance_id_clean.png"
        rgb = sat / "satellite_rgb.0000.png"
        meta = json.loads((sat / "instances.json").read_text())
        declared = [int(x["local_instance_id"]) for x in meta["instances"]]
        if len(set(declared)) != len(declared):
            raise RuntimeError(f"duplicate declared local ID: {scene}")
        source_sha = sha(src)
        ids = read_ids(src)
        source_ids = set(np.unique(ids).astype(int)) - {0}
        if not source_ids <= set(declared):
            raise RuntimeError(f"raster has undeclared ID: {scene}")
        removed = candidate_details(ids)
        out_ids, changed_pixels = clean(ids, removed)
        dest = labels / f"{scene}.png"
        Image.fromarray(out_ids, mode="L").save(dest)
        if not np.array_equal(read_ids(dest), out_ids):
            raise RuntimeError(f"written v3 ID PNG differs: {scene}")
        if sha(src) != source_sha:
            raise RuntimeError(f"source ID PNG changed during build: {scene}")
        output_ids = set(np.unique(out_ids).astype(int)) - {0}
        removed_ids = {x["local_instance_id"] for x in removed}
        if output_ids != source_ids - removed_ids or changed_pixels != sum(x["area_px_512"] for x in removed):
            raise RuntimeError(f"v3 instance or pixel accounting failed: {scene}")
        row = dict(scene_dir=scene, split=split, gx=record["gx"], gy=record["gy"],
                   source_png=str(src.resolve()), source_sha256=source_sha,
                   v3_png=str(dest.resolve()), v3_sha256=sha(dest),
                   satellite_rgb=str(rgb.resolve()), input_instances=len(source_ids),
                   output_instances=len(output_ids), removed_ids=removed,
                   removed_pixels=changed_pixels)
        manifest.append(row)
        s = summary[split]
        s["scenes"] += 1
        s["changed_scenes"] += bool(removed)
        s["input_instances"] += len(source_ids)
        s["removed_instances"] += len(removed)
        s["removed_pixels"] += changed_pixels
        s["output_instances"] += len(output_ids)
    if len(manifest) != 1200:
        raise RuntimeError("incomplete v3 label materialization")
    spot = select_spotcheck(manifest)
    if len(spot) != 48 or len({x["scene_dir"] for x in spot}) != 48:
        raise RuntimeError(f"expected 48 distinct train/val spotcheck scenes, got {len(spot)}")
    manifest_path = args.out / "manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in manifest))
    with (args.out / "spotcheck.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(spot[0]))
        writer.writeheader()
        writer.writerows(spot)
    report = dict(protocol="SURFBEV-MASK-V3-20261012", rule=RULE,
                  pair_manifest_sha256=PAIR_SHA, split_manifest_sha256=SPLIT_SHA,
                  source_scenes=1200, output_scenes=1200, split_summary=summary,
                  manifest_sha256=sha(manifest_path), spotcheck_sha256=sha(args.out / "spotcheck.csv"),
                  n_spotcheck=len(spot), status="BUILT_UNREVIEWED",
                  note="Data-side derived raster. No model-side ignore. AP is not comparable to v2 by itself.")
    (args.out / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
