#!/usr/bin/env python3
"""SURFBEV-D1b-InstanceDensity-R1: saturated-D1 follow-up, no model involved."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from probe_surfacebev_sat_alignment import (
    ROOT, cell, device_z, k_and_ext, project_rasterize, read_ids, scene_points,
)

OUT = "portable_sam2_explicit_coarse/outputs/SURFBEV-D1b-InstanceDensity-R1"
BOOT_DRAWS, BOOT_SEED = 10000, 20261012


def density_metrics(occ, ids):
    labels = np.unique(ids[ids > 0])
    cov = np.asarray([(occ & (ids == lid)).sum() / (ids == lid).sum() for lid in labels], np.float64)
    fg = ids > 0
    support = occ & fg
    return {
        "visible_instances": int(len(labels)),
        "instance_macro_coverage": float(cov.mean()),
        "instance_coverage_q25": float(np.quantile(cov, .25)),
        "roof_pixel_coverage": float(support.sum() / fg.sum()),
        "surface_precision": float(support.sum() / occ.sum()),
    }


def bootstrap(rows, key):
    c = np.asarray([x["correct"][key] for x in rows], np.float64)
    s = np.asarray([x["sham90"][key] for x in rows], np.float64)
    d = c - s
    rng = np.random.default_rng(BOOT_SEED + sum(map(ord, key)))
    means = d[rng.integers(0, len(d), size=(BOOT_DRAWS, len(d)))].mean(1)
    return {
        "correct_mean": float(c.mean()), "sham_mean": float(s.mean()), "delta_mean": float(d.mean()),
        "ci95_percentile": [float(np.quantile(means, .025)), float(np.quantile(means, .975))],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--per-cell", type=int, default=3)
    ap.add_argument("--stride", type=int, default=4)
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty {out}")
    out.mkdir(parents=True, exist_ok=True)
    manifest = root / "accepted_pairs.jsonl"
    by_cell = {}
    for line in manifest.read_text().splitlines():
        rec = json.loads(line)
        by_cell.setdefault(cell(rec["scene_dir"]), []).append(rec["scene_dir"])
    chosen = [name for c in sorted(by_cell) for name in sorted(by_cell[c])[:args.per_cell]]
    rows, errors = [], []
    for name in chosen:
        try:
            scene = root / "scenes" / name
            ids = read_ids(scene / "satellite" / "satellite_instance_id_clean.png")
            sat_k, sat_w2c, sat_c2w = k_and_ext(scene / "satellite")
            pw = scene_points(scene, args.stride)
            occ, ct = project_rasterize(pw, sat_k, sat_w2c, ids.shape)
            center = sat_c2w[:3, 3].copy(); center[2] = 0.0
            sham = pw.copy(); rel = sham[:, :2] - center[:2]
            sham[:, 0] = center[0] - rel[:, 1]
            sham[:, 1] = center[1] + rel[:, 0]
            socc, st = project_rasterize(sham, sat_k, sat_w2c, ids.shape)
            rows.append({"scene": name, "cell": list(cell(name)), "correct": density_metrics(occ, ids),
                         "sham90": density_metrics(socc, ids), "correct_telemetry": ct,
                         "sham90_telemetry": st})
        except Exception as exc:
            errors.append({"scene": name, "error": repr(exc)})
    keys = ("instance_macro_coverage", "instance_coverage_q25", "roof_pixel_coverage")
    stats = {k: bootstrap(rows, k) for k in keys} if rows else {}
    gate = bool(not errors and all(stats[k]["ci95_percentile"][0] > 0 for k in keys))
    summary = {
        "probe": "SURFBEV-D1b-InstanceDensity-R1", "scope": "read-only geometry; not AP",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(), "stride": args.stride,
        "per_cell": args.per_cell, "cells": len(by_cell), "selected_scenes": len(chosen),
        "completed_scenes": len(rows), "errors": errors,
        "paired_bootstrap": {"draws": BOOT_DRAWS, "seed": BOOT_SEED, "metrics": stats},
        "verdict": "PASS" if gate else "FAIL", "gate": "all three correct-sham CI lower bounds > 0",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (out / "per_scene.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
