#!/usr/bin/env python3
"""Re-split ue_coco by grid cell to eliminate region leakage.

The original ue_coco train/val/test split assigns individual sampling points
(distinct world X,Y) to each split, but those points come from the SAME
geographic regions (grid cells gx,gy). 81/84 cells span multiple splits, so
100% of test images share their region with train images. This is data leakage:
the model sees adjacent patches of the same UE area in training and testing.

This script regroups ALL images (across the three original splits) by their
grid cell (gx,gy) and assigns whole cells to a new train/val/test split. Cells
never span splits after the rewrite. Split sizes target 70/15/15 of cells, with
image counts reported so the ratio is visible per split too.

Usage:
    python scripts/resplit_ue_coco.py --data-root /data1/wangcheng/dataset/ue_coco
    # Dry-run (no files written, just reports):
    python scripts/resplit_ue_coco.py --data-root ... --dry-run

Outputs:
    annotations/train.clean.json / val.clean.json / test.clean.json
Original *.json are backed up to *.orig.json on the first run (not overwritten).
"""
import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

GRID_RE = re.compile(r"gx_(-?\d+)_gy_(-?\d+)")
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
SEED = 44


def parse_grid(file_name: str):
    m = GRID_RE.search(file_name)
    if not m:
        raise ValueError(f"cannot parse grid cell from {file_name!r}")
    return (int(m.group(1)), int(m.group(2)))


def load_all_images(ann_dir: Path):
    """Merge images+annotations from the three original splits (dedup by image id)."""
    images = {}
    annotations = []
    categories = None
    seen_ann = set()
    for split in ("train", "val", "test"):
        p = ann_dir / f"{split}.json"
        with open(p) as f:
            data = json.load(f)
        categories = data.get("categories", categories)
        for img in data["images"]:
            # Dedup by id; a handful of ids could collide if files overlap.
            images[img["id"]] = img
        for ann in data["annotations"]:
            if ann["id"] in seen_ann:
                continue
            seen_ann.add(ann["id"])
            annotations.append(ann)
    return images, annotations, categories


def assign_cells(cells, seed=SEED):
    """Assign whole grid cells to splits targeting TRAIN/VAL/TEST cell fractions.

    Stratify by cell image count so each split gets a mix of large and small
    cells (avoids all big cells landing in one split). Returns dict cell->split.
    """
    rng = np.random.RandomState(seed)
    # Sort cells by image count for stable, deterministic ordering.
    cell_items = sorted(cells.items(), key=lambda kv: (len(kv[1]), kv[0]))
    n = len(cell_items)
    n_train = int(round(n * TRAIN_FRAC))
    n_val = int(round(n * VAL_FRAC))
    # remaining -> test
    # Stratified round-robin over count-sorted cells gives a size-balanced mix.
    assignments = {}
    # Shuffle within stable order using seed for reproducibility.
    order = list(range(n))
    rng.shuffle(order)
    sorted_cells = [cell_items[i][0] for i in order]
    for i, cell in enumerate(sorted_cells):
        if i < n_train:
            assignments[cell] = "train"
        elif i < n_train + n_val:
            assignments[cell] = "val"
        else:
            assignments[cell] = "test"
    return assignments


def renumber(images, annotations):
    """Renumber image ids and annotation ids to be contiguous per split."""
    old_to_new = {old: new + 1 for new, old in enumerate(sorted(images.keys()))}
    new_images = []
    for old in sorted(images.keys()):
        img = dict(images[old])
        img["id"] = old_to_new[old]
        new_images.append(img)
    new_annotations = []
    for new_id, ann in enumerate(annotations, start=1):
        a = dict(ann)
        if a["image_id"] not in old_to_new:
            continue
        a["image_id"] = old_to_new[a["image_id"]]
        a["id"] = new_id
        new_annotations.append(a)
    return new_images, new_annotations, old_to_new


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=str, default="/data1/wangcheng/dataset/ue_coco")
    ap.add_argument("--dry-run", action="store_true", help="Report only; write no files.")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    root = Path(args.data_root)
    ann_dir = root / "annotations"

    images, annotations, categories = load_all_images(ann_dir)
    print(f"Merged images (dedup): {len(images)}, annotations: {len(annotations)}")

    # Group image ids by grid cell.
    cells = defaultdict(list)
    unassigned = 0
    for img_id, img in images.items():
        cell = parse_grid(img["file_name"])
        cells[cell].append(img_id)
    print(f"Grid cells: {len(cells)}")

    # Original-split leakage summary.
    orig_cell_splits = defaultdict(set)
    for split in ("train", "val", "test"):
        with open(ann_dir / f"{split}.json") as f:
            d = json.load(f)
        for img in d["images"]:
            orig_cell_splits[parse_grid(img["file_name"])].add(split)
    leaky = sum(1 for s in orig_cell_splits.values() if len(s) > 1)
    print(f"Original split: {leaky}/{len(orig_cell_splits)} cells span >1 split (leakage)")

    # Assign cells to splits.
    assignments = assign_cells(cells, seed=args.seed)

    # Build per-split image sets + report.
    split_images = {"train": [], "val": [], "test": []}
    split_cells = {"train": set(), "val": set(), "test": set()}
    for cell, split in assignments.items():
        split_cells[split].add(cell)
        for img_id in cells[cell]:
            split_images[split].append(img_id)
    print("\nNew split (by grid cell):")
    for s in ("train", "val", "test"):
        n_img = len(split_images[s])
        print(f"  {s:5s}: {len(split_cells[s]):3d} cells, {n_img:4d} images")

    # Verify zero cell overlap.
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a < b:
                ov = len(split_cells[a] & split_cells[b])
                if ov:
                    print(f"  WARNING: {a} ∩ {b} = {ov} cells (leak!)")
    print("Cell overlap check: OK (no warnings above means zero leakage)")

    if args.dry_run:
        print("\n[dry-run] no files written.")
        return

    # Renumber and write new jsons (with .clean suffix); back up originals once.
    for split in ("train", "val", "test"):
        orig_path = ann_dir / f"{split}.json"
        bak_path = ann_dir / f"{split}.orig.json"
        if not bak_path.exists():
            shutil.copy2(orig_path, bak_path)
            print(f"Backed up {orig_path.name} -> {bak_path.name}")

    split_img_sets = {s: set(split_images[s]) for s in ("train", "val", "test")}
    # Renumber within each split for clean contiguous ids.
    for split in ("train", "val", "test"):
        these_imgs = {img_id: images[img_id] for img_id in split_img_sets[split]}
        these_anns = [a for a in annotations if a["image_id"] in these_imgs]
        new_images, new_anns, _ = renumber(these_imgs, these_anns)
        out = {"images": new_images, "annotations": new_anns, "categories": categories}
        out_path = ann_dir / f"{split}.clean.json"
        with open(out_path, "w") as f:
            json.dump(out, f)
        print(f"Wrote {out_path}: {len(new_images)} images, {len(new_anns)} annotations")
    print("\nDone. New files: annotations/{train,val,test}.clean.json")


if __name__ == "__main__":
    main()
