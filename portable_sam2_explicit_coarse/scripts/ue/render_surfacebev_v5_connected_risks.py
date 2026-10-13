#!/usr/bin/env python3
"""Render all R3 connected-risk globals for visual semantic QA; no label edits."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PACKAGE = Path(__file__).resolve().parents[2]
R3 = PACKAGE / "outputs/SURFBEV-V5-PHYSICAL-INSTANCE-R3-20261012"
OUT = PACKAGE / "outputs/SURFBEV-V5-PHYSICAL-INSTANCE-CONNECTED-VIS-20261012"
OUT_MAX = PACKAGE / "outputs/SURFBEV-V5-PHYSICAL-INSTANCE-CONNECTED-VIS-MAXAREA-20261012"


def thumbnail(row):
    with Image.open(row["satellite_rgb"]) as image:
        rgb = image.convert("RGB")
    with Image.open(row["v3_png"]) as image:
        ids = np.asarray(image).copy()
    if rgb.size != (512, 512) or ids.shape != (512, 512):
        raise ValueError(row["scene_dir"])
    mask = ids == row["local_instance_id"]
    x0, y0, x1, y1 = row["bbox_xyxy"]
    cx, cy = (x0+x1)//2, (y0+y1)//2
    side = max(96, int(max(x1-x0, y1-y0)*1.35))
    side = min(side, 512)
    left = max(0, min(512-side, cx-side//2))
    top = max(0, min(512-side, cy-side//2))
    box = (left, top, left+side, top+side)
    crop = np.asarray(rgb.crop(box)).copy().astype(np.float32)
    highlight = mask[top:top+side, left:left+side]
    crop[highlight] = crop[highlight]*.48 + np.asarray([30, 235, 225])*.52
    tile = Image.fromarray(crop.astype(np.uint8)).resize((320, 320), Image.Resampling.NEAREST)
    title = Image.new("RGB", (320, 52), (16, 18, 22))
    draw = ImageDraw.Draw(title)
    draw.text((8, 5), f"global {row['global_instance_id']}  local {row['local_instance_id']}", fill=(255, 255, 255))
    draw.text((8, 26), f"area {row['area_px']}  box {row['bbox_fill']:.2f}  hull {row['hull_fill']:.2f}", fill=(198, 217, 225))
    canvas = Image.new("RGB", (320, 372), (16, 18, 22))
    canvas.paste(title, (0, 0))
    canvas.paste(tile, (0, 52))
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r3", type=Path, default=R3)
    parser.add_argument("--selection", choices=("min_fill", "max_area"), default="min_fill")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.out is None:
        args.out = OUT if args.selection == "min_fill" else OUT_MAX
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(args.out)
    rows = [json.loads(x) for x in (args.r3 / "connected_risk.jsonl").read_text().splitlines()]
    representative = {}
    for row in rows:
        gid = row["global_instance_id"]
        if args.selection == "min_fill":
            priority = (-row["bbox_fill"], -row["hull_fill"], row["scene_dir"])
            previous = None if gid not in representative else (-representative[gid]["bbox_fill"],
                -representative[gid]["hull_fill"], representative[gid]["scene_dir"])
        else:
            priority = (not row["touches_border"], row["area_px"], row["scene_dir"])
            previous = None if gid not in representative else (not representative[gid]["touches_border"],
                representative[gid]["area_px"], representative[gid]["scene_dir"])
        if previous is None or priority > previous:
            representative[gid] = row
    if len(representative) != 95:
        raise ValueError("R3 global count changed")
    args.out.mkdir(parents=True)
    selected = [representative[gid] for gid in sorted(representative)]
    for start in range(0, len(selected), 16):
        sheet = Image.new("RGB", (4*320, 4*372), (10, 12, 16))
        for i, row in enumerate(selected[start:start+16]):
            sheet.paste(thumbnail(row), ((i%4)*320, (i//4)*372))
        sheet.save(args.out / f"connected_risk_{start//16+1:02d}.png")
    (args.out / "selected.jsonl").write_text("".join(json.dumps(dict(global_instance_id=x["global_instance_id"],
                                 scene_dir=x["scene_dir"], local_instance_id=x["local_instance_id"],
                                 bbox_fill=x["bbox_fill"], hull_fill=x["hull_fill"])) + "\n" for x in selected))
    print(json.dumps(dict(globals=len(selected), sheets=(len(selected)+15)//16,
                          note="diagnostic visuals only; no mask edited")))


if __name__ == "__main__":
    main()
