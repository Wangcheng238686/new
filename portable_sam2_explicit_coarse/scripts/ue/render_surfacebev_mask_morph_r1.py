#!/usr/bin/env python3
"""Render fixed train/val stratified samples of micro-hole filling."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


PACKAGE = Path(__file__).resolve().parents[2]
ROOT = PACKAGE / "outputs/SURFBEV-MASK-MORPH-R1-20261012"


def render(row, dest):
    rgb = np.asarray(Image.open(row["rgb"]).convert("RGB"))
    old = np.asarray(Image.open(row["v3_png"]).convert("L"))
    new = np.asarray(Image.open(row["cleaned_png"]).convert("L"))
    if rgb.shape != (512, 512, 3) or old.shape != (512, 512) or new.shape != (512, 512):
        raise ValueError(f"unexpected dimensions: {row['scene_dir']}")
    changed = old != new
    if int(changed.sum()) != int(row["changed_pixels"]):
        raise ValueError(f"changed pixel count mismatch: {row['scene_dir']}")
    overlay = rgb.copy()
    if np.any(changed):
        overlay[changed] = [255, 40, 180]
        yy, xx = np.where(changed)
        cy, cx = int(np.median(yy)), int(np.median(xx))
    else:
        cy, cx = 256, 256
    cv2.rectangle(overlay, (max(0, cx-32), max(0, cy-32)),
                  (min(511, cx+31), min(511, cy+31)), (255, 255, 255), 1)
    y0, x0 = max(0, min(448, cy-32)), max(0, min(448, cx-32))
    zoom = Image.fromarray(overlay[y0:y0+64, x0:x0+64]).resize((512, 512), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (1536, 548), (32, 35, 41))
    for index, panel in enumerate((Image.fromarray(rgb), Image.fromarray(overlay), zoom)):
        canvas.paste(panel, (512*index, 36))
    draw = ImageDraw.Draw(canvas)
    for index, label in enumerate(("Satellite RGB", "R1 changed pixels (pink)", "64x64 crop, enlarged")):
        draw.text((512*index+8, 10), label, fill=(240, 240, 240))
    path = dest / f"{row['split']}_{row['band']}_{row['scene_dir']}.jpg"
    canvas.save(path, quality=91, subsampling=0)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    dest = args.root / "visual_review"
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"refusing to overwrite {dest}")
    dest.mkdir(parents=True)
    rows = [json.loads(line) for line in (args.root / "review_sample.jsonl").read_text().splitlines() if line]
    index = []
    for row in rows:
        path = render(row, dest)
        index.append(dict(scene_dir=row["scene_dir"], split=row["split"], band=row["band"],
                          changed_pixels=row["changed_pixels"], image=str(path.resolve())))
    (dest / "index.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in index))
    print(f"rendered {len(index)} fixed review images in {dest}")


if __name__ == "__main__":
    main()
