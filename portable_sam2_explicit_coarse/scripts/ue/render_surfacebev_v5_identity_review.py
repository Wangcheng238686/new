#!/usr/bin/env python3
"""Render source RGB with parent remainder and proposed ID correction, without editing labels."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from build_surfacebev_v4_postsplit import to_world
from probe_surfacebev_v5_physical_instances import rasterize


PACKAGE = Path(__file__).resolve().parents[2]
V3 = PACKAGE / "outputs/SURFBEV-MASK-V3-20261012"
R4 = PACKAGE / "outputs/SURFBEV-V5-PHYSICAL-INSTANCE-R4-20261012"
R6B = PACKAGE / "outputs/SURFBEV-V5-PHYSICAL-INSTANCE-R6B-20261012"
OUT = PACKAGE / "outputs/SURFBEV-V5-PHYSICAL-INSTANCE-IDENTITY-VIS-20261012"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3", type=Path, default=V3)
    parser.add_argument("--r4", type=Path, default=R4)
    parser.add_argument("--r6b", type=Path, default=R6B)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(args.out)
    source = [json.loads(x) for x in (args.v3 / "manifest.jsonl").read_text().splitlines()]
    by_scene = {x["scene_dir"]: x for x in source}
    r4 = [json.loads(x) for x in (args.r4 / "identity_consensus.jsonl").read_text().splitlines()]
    r6 = [json.loads(x) for x in (args.r6b / "cell_votes.jsonl").read_text().splitlines()]
    pair_gates = collections.defaultdict(list)
    for row in r6:
        pair_gates[(row["parent_global_id"], row["target_global_id"])].append(row["gate"])
    accepted = [x for x in r4 if x["identity_correction_candidate"] is not None and
                all(g == "PASS" for g in pair_gates[(x["global_instance_id"], x["identity_correction_candidate"])])]
    if len(accepted) != 18:
        raise ValueError("expected 18 whole-pair PASS candidates")
    parents = {x["global_instance_id"] for x in accepted}
    entries = collections.defaultdict(list)
    for rec in source:
        if rec["split"] not in ("train", "val"):
            continue
        meta = json.loads((Path(rec["source_png"]).parent / "instances.json").read_text())
        relevant = [(int(x["local_instance_id"]), int(x["global_instance_id"])) for x in meta["instances"]
                    if int(x["global_instance_id"]) in parents]
        if not relevant:
            continue
        with Image.open(rec["v3_png"]) as image:
            ids = np.asarray(image).copy()
        origin = to_world(rec, (0, 0))
        du = to_world(rec, (1, 0)) - origin
        dv = to_world(rec, (0, 1)) - origin
        for local, gid in relevant:
            ys, xs = np.where(ids == local)
            if len(xs):
                entries[gid].append(dict(scene_dir=rec["scene_dir"], split=rec["split"],
                                         local_id=local, origin=origin,
                                         points=origin + xs[:, None]*du + ys[:, None]*dv,
                                         xy=np.stack((xs, ys), axis=-1)))
    rasters = {gid: rasterize(entries[gid]) for gid in parents}
    tiles = []
    records = []
    for case in sorted(accepted, key=lambda x:(x["global_instance_id"], x["component_label"])):
        parent, target, label = case["global_instance_id"], case["identity_correction_candidate"], case["component_label"]
        raster = rasters[parent]
        low = np.asarray(raster["origin_xy_m"], dtype=np.int64)
        for scene in case["scene_dirs"]:
            entry = next(e for e in entries[parent] if e["scene_dir"] == scene)
            rec = by_scene[scene]
            grid_xy = np.floor(entry["points"]).astype(np.int64) - low
            selected = raster["labels"][grid_xy[:, 1], grid_xy[:, 0]] == label
            orphan = np.zeros((512, 512), bool)
            orphan[entry["xy"][selected, 1], entry["xy"][selected, 0]] = True
            with Image.open(rec["v3_png"]) as image:
                ids = np.asarray(image).copy()
            parent_mask = ids == entry["local_id"]
            if not np.any(orphan) or not np.all(orphan <= parent_mask):
                raise ValueError(f"invalid selected component {scene}")
            with Image.open(rec["satellite_rgb"]) as image:
                canvas = np.asarray(image.convert("RGB")).copy().astype(np.float32)
            remainder = parent_mask & ~orphan
            canvas[remainder] = .55*canvas[remainder] + .45*np.asarray([255, 55, 160])
            canvas[orphan] = .45*canvas[orphan] + .55*np.asarray([20, 235, 225])
            tile = Image.new("RGB", (512, 552), (15, 17, 22))
            draw = ImageDraw.Draw(tile)
            draw.text((8, 5), f"{parent} -> {target} | {scene[:31]}", fill="white")
            draw.text((8, 22), f"cyan candidate {int(orphan.sum())} px | magenta parent remainder {int(remainder.sum())} px", fill="white")
            tile.paste(Image.fromarray(canvas.astype(np.uint8)), (0, 40))
            tiles.append(tile)
            records.append(dict(parent_global_id=parent, target_global_id=target,
                                component_label=label, scene_dir=scene,
                                candidate_pixels=int(orphan.sum()), parent_remainder_pixels=int(remainder.sum())))
    args.out.mkdir(parents=True)
    for start in range(0, len(tiles), 4):
        sheet = Image.new("RGB", (1024, 1104), (15, 17, 22))
        for i, tile in enumerate(tiles[start:start+4]):
            sheet.paste(tile, ((i%2)*512, (i//2)*552))
        sheet.save(args.out / f"identity_review_{start//4+1:02d}.png")
    (args.out / "selected.jsonl").write_text("".join(json.dumps(x, sort_keys=True)+"\n" for x in records))
    print(json.dumps(dict(source_scene_components=len(tiles), sheets=(len(tiles)+3)//4)))


if __name__ == "__main__":
    main()
