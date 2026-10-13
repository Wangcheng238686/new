#!/usr/bin/env python3
"""Read-only train/val inventory of possible physical-building ID merges."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


PACKAGE = Path(__file__).resolve().parents[2]
V3 = PACKAGE / "outputs/SURFBEV-MASK-V3-20261012"
R2 = PACKAGE / "outputs/SURFBEV-V5-PHYSICAL-INSTANCE-R2-20261012"
OUT = PACKAGE / "outputs/SURFBEV-V5-PHYSICAL-INSTANCE-R3-20261012"
V3_SHA = "8d3da5addd302451337954191dccbf1b76f588a0790f1073415e30439015922f"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mask_shape(mask):
    area = int(mask.sum())
    ys, xs = np.where(mask)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    box_area = (x1 - x0) * (y1 - y0)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    contour_points = np.stack((xs, ys), axis=-1).astype(np.int32)
    hull_area = float(cv2.contourArea(cv2.convexHull(contour_points))) if len(contour_points) >= 3 else 0.0
    return dict(area_px=area, bbox_xyxy=[x0, y0, x1, y1],
                bbox_fill=round(area / box_area, 6),
                hull_fill=round(min(1.0, area / hull_area), 6) if hull_area else None,
                component_areas=sorted((int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)), reverse=True),
                touches_border=(x0 == 0 or y0 == 0 or x1 == 512 or y1 == 512))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3", type=Path, default=V3)
    parser.add_argument("--r2", type=Path, default=R2)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(args.out)
    if sha(args.v3 / "manifest.jsonl") != V3_SHA:
        raise ValueError("v3 manifest changed")
    r2_summary = json.loads((args.r2 / "summary.json").read_text())
    if r2_summary["probe_sha256"] != sha(args.r2 / "global_probe.jsonl"):
        raise ValueError("R2 probe changed")
    source = [json.loads(x) for x in (args.v3 / "manifest.jsonl").read_text().splitlines()]
    if len(source) != 1200 or len({x["scene_dir"] for x in source}) != 1200:
        raise ValueError("expected 1200 unique scenes")
    connected = []
    counters = collections.Counter()
    for rec in source:
        if rec["split"] not in ("train", "val"):
            continue
        if sha(Path(rec["v3_png"])) != rec["v3_sha256"]:
            raise ValueError(f"v3 label changed: {rec['scene_dir']}")
        with Image.open(rec["v3_png"]) as image:
            if image.mode != "L" or image.size != (512, 512):
                raise ValueError(f"invalid v3 PNG: {rec['scene_dir']}")
            ids = np.asarray(image).copy()
        meta_path = Path(rec["source_png"]).parent / "instances.json"
        meta = json.loads(meta_path.read_text())
        global_for_local = {int(x["local_instance_id"]): int(x["global_instance_id"]) for x in meta["instances"]}
        if len(global_for_local) != len(meta["instances"]):
            raise ValueError(f"duplicate local ID: {rec['scene_dir']}")
        counters["scenes"] += 1
        for local in np.unique(ids):
            local = int(local)
            if local == 0:
                continue
            if local not in global_for_local:
                raise ValueError(f"missing global for local {local}: {rec['scene_dir']}")
            metrics = mask_shape(ids == local)
            counters["visible_instances"] += 1
            if metrics["area_px"] < 128 or len(metrics["component_areas"]) != 1:
                counters["not_connected_risk_eligible"] += 1
                continue
            flagged = metrics["bbox_fill"] <= 0.40 or (metrics["hull_fill"] is not None and metrics["hull_fill"] <= 0.65)
            counters["connected_eligible"] += 1
            if flagged:
                counters["connected_risk"] += 1
                connected.append(dict(scene_dir=rec["scene_dir"], split=rec["split"],
                                      local_instance_id=local, global_instance_id=global_for_local[local],
                                      satellite_rgb=rec["satellite_rgb"], v3_png=rec["v3_png"],
                                      v3_sha256=rec["v3_sha256"], instances_json_sha256=sha(meta_path),
                                      **metrics))
    r2_rows = [json.loads(x) for x in (args.r2 / "global_probe.jsonl").read_text().splitlines()]
    orphan = []
    for row in r2_rows:
        if not row["components"]:
            continue
        main = row["components"][0]
        mx, my = main["center_xy_m"]
        for comp in row["components"][1:]:
            cx, cy = comp["center_xy_m"]
            separation = float(np.hypot(cx - mx, cy - my))
            support_scenes = [scene for scene, assigns in row["scene_assignments"].items()
                              if any(a["component_label"] == comp["label"] and a["pixel_fraction"] > 0 for a in assigns)]
            main_in_same_scene = any(any(a["component_label"] == main["label"] and a["pixel_fraction"] > 0
                                         for a in row["scene_assignments"][scene]) for scene in support_scenes)
            if comp["raw_cells"] < 64 or separation < 20 or not main_in_same_scene:
                continue
            orphan.append(dict(global_instance_id=row["global_instance_id"],
                               component_label=comp["label"], raw_cells=comp["raw_cells"],
                               supporting_scenes=comp["supporting_scenes"],
                               separation_from_main_m=round(separation, 3),
                               scene_dirs=support_scenes, split=row["splits"][0]))
    args.out.mkdir(parents=True)
    for name, rows in (("connected_risk.jsonl", connected), ("isolated_world_components.jsonl", orphan)):
        (args.out / name).write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    summary = dict(protocol="SURFBEV-V5-PHYSICAL-INSTANCE-R3-20261012", status="INVENTORY_ONLY",
                   v3_manifest_sha256=V3_SHA, r2_probe_sha256=r2_summary["probe_sha256"],
                   counts=dict(counters), isolated_world_components=len(orphan),
                   connected_risk_global_ids=len({x["global_instance_id"] for x in connected}),
                   isolated_world_global_ids=len({x["global_instance_id"] for x in orphan}),
                   known_positive_orphan_170002=any(x["global_instance_id"] == 170002 for x in orphan),
                   connected_risk_sha256=sha(args.out / "connected_risk.jsonl"),
                   isolated_world_sha256=sha(args.out / "isolated_world_components.jsonl"),
                   note="Candidate inventory only; no labels or reader changed, no automatic split decision.")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
