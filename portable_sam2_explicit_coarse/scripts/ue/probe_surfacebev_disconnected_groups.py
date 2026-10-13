#!/usr/bin/env python3
"""Read-only v3 train/val inventory of substantial disconnected instance masks."""
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


PACKAGE = Path(__file__).resolve().parents[2]
V3 = PACKAGE / "outputs/SURFBEV-MASK-V3-20261012"
OUT = PACKAGE / "outputs/SURFBEV-DISCONNECT-QA-20261012-R3"
KNOWN = ("Coord_002917_pt_gx_-5_gy_-2_s_002_gx_-5_gy_-2_s_2_X_-90000_Y_-37500", 16)
KNOWN_GLOBAL = 1874325323
FIELDS = ("split", "scene_dir", "gx", "gy", "local_instance_id", "global_instance_id", "anchor_label",
          "grouping_mode", "n_related_actors", "area", "components_8", "largest_area", "second_area",
          "second_frac", "min_gap_px", "bbox_fill", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1",
          "major_x0", "major_y0", "major_x1", "major_y1", "second_x0", "second_y0", "second_x1", "second_y1",
          "touches_border", "m2", "m2_strong", "rgb", "v3_png")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(mask):
    ys, xs = np.where(mask)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1
    binary = mask[y0:y1, x0:x1].astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    area = int(len(xs))
    order = sorted(range(1, n), key=lambda i: int(stats[i, cv2.CC_STAT_AREA]), reverse=True)
    result = dict(area=area, components_8=n-1, largest_area=int(stats[order[0], cv2.CC_STAT_AREA]),
                  second_area=0, second_frac=0., min_gap_px=0., bbox_fill=round(area / ((x1-x0)*(y1-y0)), 6),
                  bbox_x0=x0, bbox_y0=y0, bbox_x1=x1, bbox_y1=y1,
                  touches_border=int(x0 == 0 or y0 == 0 or x1 == 512 or y1 == 512),
                  major_x0="", major_y0="", major_x1="", major_y1="",
                  second_x0="", second_y0="", second_x1="", second_y1="")
    if n < 3:
        return result
    first, second = order[:2]
    result["second_area"] = int(stats[second, cv2.CC_STAT_AREA])
    result["second_frac"] = round(result["second_area"]/area, 6)
    gap = cv2.distanceTransform((labels != first).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    result["min_gap_px"] = round(float(gap[labels == second].min()), 4)
    for prefix, index in (("major", first), ("second", second)):
        bx, by, bw, bh, _ = (int(x) for x in stats[index])
        result.update({f"{prefix}_x0":x0+bx, f"{prefix}_y0":y0+by,
                       f"{prefix}_x1":x0+bx+bw, f"{prefix}_y1":y0+by+bh})
    return result


def pick(rows, n, rng, used):
    # Select uniformly among global IDs, then show their clearest two-block scene.
    representative = {}
    for row in rows:
        global_id = row["global_instance_id"]
        if global_id == KNOWN_GLOBAL:
            continue
        old = representative.get(global_id)
        if old is None or (row["second_area"], row["largest_area"], row["scene_dir"]) > (old["second_area"], old["largest_area"], old["scene_dir"]):
            representative[global_id] = row
    by_split = {split: sorted((row for row in representative.values() if row["split"] == split), key=lambda x:x["global_instance_id"]) for split in ("train", "val")}
    for values in by_split.values():
        rng.shuffle(values)
    chosen = []
    for split in ("train", "val"):
        for row in by_split[split]:
            if len([x for x in chosen if x["split"] == split]) >= n//2:
                break
            if row["global_instance_id"] not in used:
                chosen.append(row); used.add(row["global_instance_id"])
    for split in ("train", "val"):
        for row in by_split[split]:
            if len(chosen) >= n:
                break
            if row["global_instance_id"] not in used:
                chosen.append(row); used.add(row["global_instance_id"])
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3", type=Path, default=V3)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing to overwrite {args.out}")
    summary = json.loads((args.v3/"summary.json").read_text())
    if summary["manifest_sha256"] != sha(args.v3/"manifest.jsonl"):
        raise ValueError("v3 manifest SHA mismatch")
    if summary["pair_manifest_sha256"] != "f15cd8f5b8d10a69d721546d6cfa1f2cce58af2861d57b1357c5e9ff1e1fe62a" or summary["split_manifest_sha256"] != "17f49430b15037c8da25c63a1bf7fe4d510cf5ff153263d2feba62c4393441d5":
        raise ValueError("unexpected v3 pair/split contract")
    manifest = [json.loads(line) for line in (args.v3/"manifest.jsonl").read_text().splitlines() if line]
    if len(manifest) != 1200 or len({r["scene_dir"] for r in manifest}) != 1200:
        raise ValueError("expected 1200 unique scenes")
    rows, inventory = [], collections.Counter()
    thresholds = collections.Counter()
    scenes = collections.defaultdict(set)
    globals_by_tier = collections.defaultdict(set)
    for rec in manifest:
        split = rec["split"]
        if split not in ("train", "val"):
            continue
        label_path = Path(rec["v3_png"])
        source_path = Path(rec["source_png"])
        if sha(label_path) != rec["v3_sha256"] or sha(source_path) != rec["source_sha256"]:
            raise ValueError(f"source/v3 SHA mismatch: {rec['scene_dir']}")
        with Image.open(label_path) as image:
            if image.mode != "L" or image.size != (512,512):
                raise ValueError(f"invalid label: {label_path}")
            ids = np.asarray(image).copy()
        metadata = json.loads((source_path.parent/"instances.json").read_text())
        lookup = {int(x["local_instance_id"]): x for x in metadata["instances"]}
        inventory[(split,"scenes")] += 1
        for local in np.unique(ids[ids > 0]):
            local = int(local)
            meta = lookup.get(local)
            if meta is None:
                raise ValueError(f"missing metadata: {rec['scene_dir']} local {local}")
            stats = measure(ids == local)
            inventory[(split,"instances")] += 1
            if stats["components_8"] > 1:
                inventory[(split,"any_disconnected")] += 1
            second = stats["second_area"]
            fraction = stats["second_frac"]
            gap = stats["min_gap_px"]
            for area_cut in (32,64,128,256):
                for frac_cut in (.1,.2,.3):
                    for gap_cut in (0,4,8,16):
                        if second >= area_cut and fraction >= frac_cut and gap >= gap_cut:
                            thresholds[(split,area_cut,frac_cut,gap_cut)] += 1
            m2 = int(second >= 64 and fraction >= .1)
            strong = int(second >= 128 and fraction >= .2 and gap >= 8)
            if not m2:
                continue
            row = dict(split=split, scene_dir=rec["scene_dir"], gx=rec["gx"], gy=rec["gy"],
                       local_instance_id=local, global_instance_id=int(meta["global_instance_id"]),
                       anchor_label=meta.get("anchor_label", ""), grouping_mode=meta.get("grouping_mode", ""),
                       n_related_actors=len(meta.get("related_actors", [])),
                       m2=m2, m2_strong=strong, rgb=rec["satellite_rgb"], v3_png=str(label_path), **stats)
            rows.append(row)
            for tier in ("m2", "strong" if strong else "weak"):
                scenes[(split,tier)].add(rec["scene_dir"])
                globals_by_tier[(split,tier)].add(int(meta["global_instance_id"]))
            inventory[(split,"m2")] += 1
            if strong:
                inventory[(split,"strong")] += 1
    if inventory[("train","scenes")] != 694 or inventory[("val","scenes")] != 100:
        raise ValueError(f"split scene counts changed: {inventory}")
    known = next((r for r in rows if (r["scene_dir"],r["local_instance_id"]) == KNOWN), None)
    if known is None or not known["m2_strong"]:
        raise ValueError(f"known positive missing from strong tier: {known}")
    rng = random.Random(20261012)
    used_globals = set()
    strong_globals = {r["global_instance_id"] for r in rows if r["m2_strong"]}
    sampled = pick([r for r in rows if r["global_instance_id"] in strong_globals and r["m2_strong"]], 32, rng, used_globals) + pick([r for r in rows if r["global_instance_id"] not in strong_globals], 16, rng, used_globals)
    if len({r["global_instance_id"] for r in sampled}) != len(sampled):
        raise ValueError("visual sample reused a global instance ID")
    args.out.mkdir(parents=True)
    for name, selected in (("candidates.csv", rows), ("visual_sample.csv", sampled)):
        with (args.out/name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader(); writer.writerows(selected)
    out_summary = dict(protocol="SURFBEV-DISCONNECT-QA-20261012", v3_manifest_sha256=summary["manifest_sha256"],
                       n_scenes=794, inventory={split:{key:inventory[(split,key)] for key in ("scenes","instances","any_disconnected","m2","strong")} for split in ("train","val")},
                       scene_counts={f"{split}_{tier}":len(scenes[(split,tier)]) for split in ("train","val") for tier in ("m2","strong")},
                       global_counts={f"{split}_{tier}":len(globals_by_tier[(split,tier)]) for split in ("train","val") for tier in ("m2","strong")},
                       threshold_grid=[dict(split=split, second_area_ge=area, second_frac_ge=frac, gap_ge=gap, instances=count) for (split,area,frac,gap),count in sorted(thresholds.items())],
                       known_positive={k:known[k] for k in ("split","scene_dir","local_instance_id","global_instance_id","area","second_area","second_frac","min_gap_px","m2_strong")},
                       n_visual_sample=len(sampled), candidates_sha256=sha(args.out/"candidates.csv"), sample_sha256=sha(args.out/"visual_sample.csv"),
                       interpretation="Morphology candidates, not confirmed physical building merges. No labels modified.")
    (args.out/"summary.json").write_text(json.dumps(out_summary,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"inventory":out_summary["inventory"],"scene_counts":out_summary["scene_counts"],"global_counts":out_summary["global_counts"],"known_positive":out_summary["known_positive"],"n_visual_sample":len(sampled)},indent=2))


if __name__ == "__main__":
    main()
