#!/usr/bin/env python3
"""Build the immutable, cell-disjoint SURFBEV B0 split manifest."""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

MANIFEST_SHA = "f15cd8f5b8d10a69d721546d6cfa1f2cce58af2861d57b1357c5e9ff1e1fe62a"
TEST = {(x, y) for x in range(2, 5) for y in range(-7, -4)}
VAL = {(x, y) for x in range(-7, -4) for y in range(-3, 0)}
GRID = re.compile(r"_gx_(-?\d+)_gy_(-?\d+)_")


def grid(scene):
    m = GRID.search(scene)
    if not m:
        raise ValueError(f"missing gx/gy in {scene}")
    return int(m.group(1)), int(m.group(2))


def main():
    ap = argparse.ArgumentParser()
    project = Path(__file__).resolve().parents[2]
    default_root = project / "data/ue_raw_hero_uav_pairset_20261011"
    ap.add_argument("--root", default=str(default_root))
    ap.add_argument("--out", default=str(default_root / "surfbev_b0_split.json"))
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    source = root / "accepted_pairs.jsonl"
    raw = source.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != MANIFEST_SHA:
        raise RuntimeError(f"accepted manifest SHA mismatch: {got}")
    if out.exists():
        raise SystemExit(f"refusing to overwrite immutable split: {out}")
    rows = [json.loads(x) for x in raw.decode().splitlines() if x]
    assigned, counts, cells = [], Counter(), {"train": set(), "val": set(), "test": set()}
    for r in rows:
        g = grid(r["scene_dir"])
        split = "test" if g in TEST else "val" if g in VAL else "train"
        assigned.append({"scene_dir": r["scene_dir"], "scene_index": r["scene_index"], "gx": g[0], "gy": g[1], "split": split})
        counts[split] += 1; cells[split].add(g)
    if counts != Counter(train=949, val=100, test=151):
        raise RuntimeError(f"unexpected split counts: {counts}")
    if [len(cells[x]) for x in ("train", "val", "test")] != [63, 9, 9]:
        raise RuntimeError("unexpected cell allocation")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": "surfacebev_b0_spatial_split_v1", "accepted_manifest_sha256": got,
               "test_block": {"gx": [2, 4], "gy": [-7, -5]}, "val_block": {"gx": [-7, -5], "gy": [-3, -1]},
               "counts": dict(counts), "cell_counts": {k: len(v) for k, v in cells.items()}, "scenes": assigned}
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"out": str(out), "counts": dict(counts), "cells": payload["cell_counts"]}, indent=2))


if __name__ == "__main__":
    main()
