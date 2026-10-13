#!/usr/bin/env python3
"""Reproducible, read-only train/val mask-QA triage for the v2 raw pairset.

This samples *instances*, not pixels. Shape flags are review cues, never deletion
rules. No model output, UAV imagery, or test scene is opened.
"""
import argparse
import collections
import csv
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2] / "data/ue_raw_hero_uav_pairset_20261011"
OUT = Path(__file__).resolve().parents[2] / "outputs/SURFBEV-MASK-QA-20261012-R1"
PAIR_SHA = "f15cd8f5b8d10a69d721546d6cfa1f2cce58af2861d57b1357c5e9ff1e1fe62a"
SPLIT_SHA = "17f49430b15037c8da25c63a1bf7fe4d510cf5ff153263d2feba62c4393441d5"
STRATA = ("micro_1_4", "tiny_5_15", "thin", "major_fragment", "sparse", "border", "control")
FIELDS = ("sample_id", "split", "stratum", "scene_dir", "gx", "gy", "local_instance_id",
          "global_instance_id", "area_px_512", "x0", "y0", "x1_exclusive", "y1_exclusive",
          "bbox_width", "bbox_height", "bbox_fill", "components_8", "largest_component_frac",
          "touches_border", "satellite_rgb", "clean_id_png", "instances_json",
          "clean_id_png_sha256")
REVIEW_FIELDS = FIELDS + ("verdict", "error_type", "evidence", "reviewer")
ADJUDICATION_FIELDS = ("sample_id", "split", "stratum", "scene_dir", "local_instance_id",
                       "global_instance_id", "reviewer_a_verdict", "reviewer_b_verdict",
                       "adjudicated_verdict", "adjudicated_error_type", "rationale", "adjudicator")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ids_from_png(path):
    a = np.asarray(Image.open(path))
    if a.ndim == 3:
        if a.shape[2] < 3 or not (np.array_equal(a[:, :, 0], a[:, :, 1])
                                      and np.array_equal(a[:, :, 0], a[:, :, 2])):
            raise ValueError(f"non-grayscale-coded ID raster: {path}")
        a = a[:, :, 0]
    if a.shape != (512, 512):
        raise ValueError(f"unexpected ID shape {a.shape}: {path}")
    return a


def measure(mask):
    ys, xs = np.where(mask)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    crop = mask[y0:y1, x0:x1].astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(crop, connectivity=8)
    area = int(mask.sum())
    largest = int(stats[1:, cv2.CC_STAT_AREA].max())
    return dict(area_px_512=area, x0=x0, y0=y0, x1_exclusive=x1, y1_exclusive=y1,
                bbox_width=x1-x0, bbox_height=y1-y0,
                bbox_fill=round(area / ((x1-x0) * (y1-y0)), 6),
                components_8=n-1, largest_component_frac=round(largest / area, 6),
                touches_border=int(x0 == 0 or y0 == 0 or x1 == 512 or y1 == 512))


def stratum(m):
    area = m["area_px_512"]
    if area <= 4:
        return "micro_1_4"
    if area <= 15:
        return "tiny_5_15"
    if min(m["bbox_width"], m["bbox_height"]) <= 2:
        return "thin"
    if m["components_8"] > 1 and m["largest_component_frac"] < .8:
        return "major_fragment"
    if m["bbox_fill"] < .15:
        return "sparse"
    if m["touches_border"]:
        return "border"
    return "control"


def load_candidates(root):
    manifest = root / "accepted_pairs.jsonl"
    split_file = root / "surfbev_b0_split_v2.json"
    if sha256(manifest) != PAIR_SHA or sha256(split_file) != SPLIT_SHA:
        raise RuntimeError("pair/split SHA differs from registered v2 contract")
    split = json.loads(split_file.read_text())
    accepted = {json.loads(line)["scene_dir"] for line in manifest.read_text().splitlines() if line}
    if len(accepted) != 1200 or len(split["scenes"]) != 1200:
        raise RuntimeError("expected exactly 1200 registered scenes")
    rows = collections.defaultdict(list)
    seen = set()
    counts = collections.Counter()
    for rec in split["scenes"]:
        scene = rec["scene_dir"]
        if scene not in accepted or scene in seen:
            raise RuntimeError(f"foreign/duplicate scene: {scene}")
        seen.add(scene)
        part = rec["split"]
        counts[part] += 1
        if part in ("test", "excluded"):
            continue  # Do not open test labels or use excluded scenes to set policy.
        if part not in ("train", "val"):
            raise RuntimeError(f"unknown split: {part}")
        sat = root / "scenes" / scene / "satellite"
        ids = ids_from_png(sat / "satellite_instance_id_clean.png")
        meta = json.loads((sat / "instances.json").read_text())
        by_local = {int(x["local_instance_id"]): int(x["global_instance_id"])
                    for x in meta["instances"]}
        if len(by_local) != len(meta["instances"]):
            raise RuntimeError(f"duplicate local ID: {scene}")
        for local in np.unique(ids):
            local = int(local)
            if local == 0:
                continue
            if local not in by_local:
                raise RuntimeError(f"raster ID absent from instances.json: {scene}, {local}")
            m = measure(ids == local)
            row = dict(split=part, scene_dir=scene, gx=rec["gx"], gy=rec["gy"],
                       local_instance_id=local, global_instance_id=by_local[local], **m)
            row["stratum"] = stratum(m)
            rows[(part, row["stratum"])].append(row)
    if counts != {"train": 694, "val": 100, "test": 151, "excluded": 255}:
        raise RuntimeError(f"unexpected split counts: {counts}")
    return rows, counts


def select(rows, n_train, n_val, seed):
    rng = random.Random(seed)
    picked = []
    used_global = set()
    cell_count = collections.Counter()
    scene_count = collections.Counter()
    diagnostics = {}
    for part in ("train", "val"):
        for group in STRATA:
            candidates = list(rows[(part, group)])
            rng.shuffle(candidates)
            target = n_train if part == "train" else n_val
            unique_global = {r["global_instance_id"] for r in candidates}
            already_picked = unique_global & used_global
            eligible_unique = unique_global - used_global
            before = len(picked)
            for _ in range(min(target, len(candidates))):
                eligible = ((i, r) for i, r in enumerate(candidates)
                            if r["global_instance_id"] not in used_global)
                best = min(eligible, key=lambda ir: (
                    cell_count[(part, ir[1]["gx"], ir[1]["gy"])],
                    scene_count[ir[1]["scene_dir"]], ir[0]), default=None)
                if best is None:
                    break
                index, row = best
                candidates.pop(index)
                picked.append(row)
                used_global.add(row["global_instance_id"])
                cell_count[(part, row["gx"], row["gy"])] += 1
                scene_count[row["scene_dir"]] += 1
            actual = len(picked) - before
            diagnostics[f"{part}/{group}"] = dict(population_rows=len(rows[(part, group)]),
                                                     unique_global=len(unique_global),
                                                     already_picked_global=len(already_picked),
                                                     remaining_unique_before_sampling=len(eligible_unique),
                                                     target=target, selected=actual,
                                                     shortfall_reason=("insufficient distinct unpicked global IDs"
                                                                       if actual < target else None))
    return picked, diagnostics


def choose_double_review(picked, seed, target):
    """Freeze a spatial/stratum-spread second-review subset before any verdict."""
    rng = random.Random(seed + 1)
    groups = collections.defaultdict(list)
    for row in picked:
        groups[(row["split"], row["stratum"])].append(row)
    for candidates in groups.values():
        rng.shuffle(candidates)
    selected = []
    while len(selected) < target:
        progressed = False
        for key in sorted(groups):
            if groups[key] and len(selected) < target:
                selected.append(groups[key].pop())
                progressed = True
        if not progressed:
            break
    if len(selected) != target:
        raise RuntimeError(f"cannot select {target} double-review samples")
    return selected


def write_csv(path, fields, records):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--train-per-stratum", type=int, default=12)
    parser.add_argument("--val-per-stratum", type=int, default=6)
    args = parser.parse_args()
    if (args.seed, args.train_per_stratum, args.val_per_stratum) != (20261012, 12, 6):
        raise ValueError("sampling protocol is frozen; change protocol/version before parameters")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing sample: {args.out}")
    rows, split_counts = load_candidates(args.root)
    picked, diagnostics = select(rows, 12, 6, args.seed)
    for row in picked:
        sat = args.root / "scenes" / row["scene_dir"] / "satellite"
        key = f'{row["scene_dir"]}:{row["local_instance_id"]}:{row["global_instance_id"]}'
        row["sample_id"] = hashlib.sha256(key.encode()).hexdigest()[:16]
        row.update(satellite_rgb=str((sat / "satellite_rgb.0000.png").resolve()),
                   clean_id_png=str((sat / "satellite_instance_id_clean.png").resolve()),
                   instances_json=str((sat / "instances.json").resolve()),
                   clean_id_png_sha256=sha256(sat / "satellite_instance_id_clean.png"))
    if len({r["sample_id"] for r in picked}) != len(picked):
        raise RuntimeError("duplicate sample ID")
    double_review = choose_double_review(picked, args.seed, (len(picked) + 4) // 5)
    summary = dict(protocol="SURFBEV-MASK-QA-20261012", seed=args.seed,
                   pair_manifest_sha256=PAIR_SHA, split_manifest_sha256=SPLIT_SHA,
                   source_splits=split_counts,
                   population={f"{part}/{group}": len(rows[(part, group)])
                               for part in ("train", "val") for group in STRATA},
                   selected={f"{part}/{group}": sum(r["split"] == part and r["stratum"] == group
                                                     for r in picked)
                             for part in ("train", "val") for group in STRATA},
                   stratum_diagnostics=diagnostics,
                   n_selected=len(picked), n_unique_global_ids=len({r["global_instance_id"] for r in picked}),
                   n_unique_scenes=len({r["scene_dir"] for r in picked}),
                   n_double_review=len(double_review),
                   double_review_sample_ids=[r["sample_id"] for r in double_review],
                   note="Risk-enriched QA sample, not a population error-rate estimate; no test labels read.")
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "sample_manifest.csv", FIELDS,
              [{k: row[k] for k in FIELDS} for row in picked])
    write_csv(args.out / "reviewer_A.csv", REVIEW_FIELDS,
              [{**row, "verdict": "", "error_type": "", "evidence": "", "reviewer": ""} for row in picked])
    write_csv(args.out / "reviewer_B_blind.csv", REVIEW_FIELDS,
              [{**row, "verdict": "", "error_type": "", "evidence": "", "reviewer": ""} for row in double_review])
    write_csv(args.out / "adjudication.csv", ADJUDICATION_FIELDS,
              [{**{k: row[k] for k in ADJUDICATION_FIELDS[:6]},
                **{k: "" for k in ADJUDICATION_FIELDS[6:]}} for row in double_review])
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
