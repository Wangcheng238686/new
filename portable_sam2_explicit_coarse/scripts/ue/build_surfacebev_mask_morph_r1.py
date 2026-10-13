#!/usr/bin/env python3
"""Conservative data-side fill of fully enclosed <=4px background holes."""
import argparse
import collections
import hashlib
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


PACKAGE = Path(__file__).resolve().parents[2]
V3 = PACKAGE / "outputs/SURFBEV-MASK-V3-20261012"
OUT = PACKAGE / "outputs/SURFBEV-MASK-MORPH-R1-20261012"
V3_SHA = "8d3da5addd302451337954191dccbf1b76f588a0790f1073415e30439015922f"
SPLIT_SHA = "17f49430b15037c8da25c63a1bf7fe4d510cf5ff153263d2feba62c4393441d5"
MAX_HOLE = 4


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_ids(path):
    with Image.open(path) as image:
        if image.mode != "L" or image.size != (512, 512):
            raise ValueError(f"unexpected ID raster: {path}")
        return np.asarray(image).copy()


def enclosed_holes(ids):
    """Yield disjoint global pixel coordinates, owner local ID, and hole size."""
    holes = []
    for lid in np.unique(ids):
        if lid == 0:
            continue
        ys, xs = np.where(ids == lid)
        y0, y1 = max(0, int(ys.min()) - 1), min(512, int(ys.max()) + 2)
        x0, x1 = max(0, int(xs.min()) - 1), min(512, int(xs.max()) + 2)
        roi = ids[y0:y1, x0:x1]
        inverse = (roi != lid).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=8)
        for component in range(1, n):
            area = int(stats[component, cv2.CC_STAT_AREA])
            if area > MAX_HOLE:
                continue
            x, y, w, h = (int(stats[component, field]) for field in
                          (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
            if x == 0 or y == 0 or x + w == roi.shape[1] or y + h == roi.shape[0]:
                continue
            crop = labels[y:y+h, x:x+w] == component
            if np.any(roi[y:y+h, x:x+w][crop] != 0):
                continue
            dy, dx = np.where(crop)
            gy, gx = y0 + y + dy, x0 + x + dx
            if np.any(gy == 0) or np.any(gy == 511) or np.any(gx == 0) or np.any(gx == 511):
                continue
            holes.append((int(lid), gy, gx, area))
    return holes


def select_review(rows):
    rng = random.Random(20261012)
    selected = []
    for split in ("train", "val"):
        pool = [r for r in rows if r["split"] == split]
        for band in ("1", "2", "3-4", "unchanged"):
            candidates = [r for r in pool if
                          (not r["hole_sizes"] if band == "unchanged" else
                           any((str(x) if x <= 2 else "3-4") == band for x in r["hole_sizes"]))]
            rng.shuffle(candidates)
            seen_cells = set()
            chosen = []
            for row in candidates:
                cell = (row["gx"], row["gy"])
                if cell not in seen_cells:
                    chosen.append(row)
                    seen_cells.add(cell)
                if len(chosen) == 8:
                    break
            if len(chosen) < 8:
                chosen += [r for r in candidates if r not in chosen][:8-len(chosen)]
            for row in chosen:
                selected.append(dict(scene_dir=row["scene_dir"], split=split, band=band,
                                     rgb=row["satellite_rgb"], v3_png=row["v3_png"],
                                     cleaned_png=row["cleaned_png"], changed_pixels=row["changed_pixels"]))
    if len({(r["scene_dir"], r["band"]) for r in selected}) != len(selected):
        raise RuntimeError("duplicate review sample in same stratum")
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3", type=Path, default=V3)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing to overwrite {args.out}")
    if sha(args.v3 / "manifest.jsonl") != V3_SHA:
        raise ValueError("unregistered v3 manifest")
    v3_summary = json.loads((args.v3 / "summary.json").read_text())
    if v3_summary["split_manifest_sha256"] != SPLIT_SHA:
        raise ValueError("unregistered split")
    inputs = [json.loads(line) for line in (args.v3 / "manifest.jsonl").read_text().splitlines() if line]
    if len(inputs) != 1200 or len({r["scene_dir"] for r in inputs}) != 1200:
        raise ValueError("expected 1200 unique scenes")
    (args.out / "labels").mkdir(parents=True)
    rows = []
    summary = collections.defaultdict(lambda: dict(scenes=0, changed_scenes=0, holes=0, pixels_added=0))
    for rec in inputs:
        src = Path(rec["v3_png"])
        if sha(src) != rec["v3_sha256"]:
            raise ValueError(f"v3 source hash changed: {rec['scene_dir']}")
        ids = read_ids(src)
        holes = enclosed_holes(ids)
        out = ids.copy()
        for lid, ys, xs, _ in holes:
            if np.any(out[ys, xs] != 0):
                raise RuntimeError(f"overlapping hole proposals: {rec['scene_dir']}")
            out[ys, xs] = lid
        changed = out != ids
        expected = sum(area for _, _, _, area in holes)
        if int(changed.sum()) != expected or np.any(ids[changed] != 0):
            raise RuntimeError(f"pixel provenance mismatch: {rec['scene_dir']}")
        if not np.array_equal(np.unique(ids[ids > 0]), np.unique(out[out > 0])):
            raise RuntimeError(f"instance ID set changed: {rec['scene_dir']}")
        dest = args.out / "labels" / f"{rec['scene_dir']}.png"
        if holes:
            Image.fromarray(out, mode="L").save(dest)
        else:
            shutil.copyfile(src, dest)
        if not np.array_equal(read_ids(dest), out):
            raise RuntimeError(f"write parity failed: {rec['scene_dir']}")
        row = dict(scene_dir=rec["scene_dir"], split=rec["split"], gx=rec["gx"], gy=rec["gy"],
                   satellite_rgb=rec["satellite_rgb"], v3_png=str(src), v3_sha256=rec["v3_sha256"],
                   cleaned_png=str(dest.resolve()), cleaned_sha256=sha(dest),
                   changed_pixels=expected, hole_count=len(holes),
                   hole_sizes=[area for _, _, _, area in holes],
                   holes=[dict(local_id=lid, size=area, pixels=[[int(y), int(x)] for y, x in zip(ys, xs)])
                          for lid, ys, xs, area in holes])
        rows.append(row)
        cell = summary[rec["split"]]
        cell["scenes"] += 1
        cell["changed_scenes"] += bool(holes)
        cell["holes"] += len(holes)
        cell["pixels_added"] += expected
    review = select_review(rows)
    manifest = args.out / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    review_path = args.out / "review_sample.jsonl"
    review_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in review))
    result = dict(protocol="SURFBEV-MASK-MORPH-R1-20261012", status="BUILT_UNREVIEWED",
                  v3_manifest_sha256=V3_SHA, split_manifest_sha256=SPLIT_SHA,
                  rule="fully enclosed 8-connected background component of size <=4 pixels within one instance",
                  output_scenes=len(rows), split_summary=summary,
                  review_strata=collections.Counter((r["split"], r["band"]) for r in review).__str__(),
                  review_count=len(review), manifest_sha256=sha(manifest), review_sha256=sha(review_path),
                  note="Candidate label version only; no reader switch, model training, or AP claim.")
    (args.out / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
