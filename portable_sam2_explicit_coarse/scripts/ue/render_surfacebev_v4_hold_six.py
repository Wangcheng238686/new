#!/usr/bin/env python3
"""Standalone PNGs for all six V4 excluded HOLD scenes; labels are read-only."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


PACKAGE = Path(__file__).resolve().parents[2]
V4 = PACKAGE / "outputs/SURFBEV-V4-POSTSPLIT-20261012-R3"
OUT = PACKAGE / "outputs/SURFBEV-V4-HOLD-SIX-VIS-20261012"


def crop_box(mask):
    ys, xs = np.where(mask)
    left, right, top, bottom = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    side = min(512, max(144, int(max(right-left, bottom-top) * 1.6)))
    cx, cy = (left+right)//2, (top+bottom)//2
    x0 = max(0, min(512-side, cx-side//2))
    y0 = max(0, min(512-side, cy-side//2))
    return x0, y0, x0+side, y0+side


def render(row, out):
    with Image.open(row["satellite_rgb"]) as image:
        rgb = np.asarray(image.convert("RGB")).copy()
    with Image.open(row["v3_png"]) as image:
        ids = np.asarray(image.convert("L")).copy()
    with Image.open(row["v4_png"]) as image:
        newer = np.asarray(image.convert("L")).copy()
    if rgb.shape != (512,512,3) or ids.shape != (512,512) or not np.array_equal(ids, newer):
        raise ValueError(f"unexpected HOLD label parity: {row['scene_dir']}")
    local = int(row["parent_local_id"])
    mask = ids == local
    if not mask.any():
        raise ValueError(f"parent absent: {row['scene_dir']}")
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    components = sorted((int(stats[i, cv2.CC_STAT_AREA]) for i in range(1,n)), reverse=True)
    x0,y0,x1,y1 = crop_box(mask)
    raw = rgb[y0:y1,x0:x1].copy()
    selected = mask[y0:y1,x0:x1]
    overlay = raw.copy()
    overlay[selected] = (.5*overlay[selected] + .5*np.asarray([255,45,170])).astype(np.uint8)
    contours,_ = cv2.findContours(selected.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255,45,170), 2)
    raw_image = Image.fromarray(raw).resize((460,460),Image.Resampling.NEAREST)
    overlay_image = Image.fromarray(overlay).resize((460,460),Image.Resampling.NEAREST)
    full = Image.fromarray(rgb).resize((230,230),Image.Resampling.BILINEAR)
    small_box = [round(v*230/512) for v in (x0,y0,x1,y1)]
    ImageDraw.Draw(full).rectangle(small_box, outline=(255,45,170), width=3)
    canvas = Image.new("RGB", (1160,510), (28,30,35))
    canvas.paste(raw_image,(0,50))
    canvas.paste(overlay_image,(460,50))
    canvas.paste(full,(930,50))
    draw = ImageDraw.Draw(canvas)
    short = row["scene_dir"].split("_")[1]
    draw.text((8,8), f"{short}  |  excluded / HOLD  |  global 1874325323  |  local {local}",fill=(255,255,255))
    draw.text((8,30), f"visible {int(mask.sum())} px  |  8-connected parts {len(components)}  |  largest {components[:3]}",fill=(225,225,225))
    draw.text((8,493),"RGB zoom",fill=(230,230,230))
    draw.text((468,493),"same ID highlighted (unchanged)",fill=(230,230,230))
    draw.text((936,288),"full scene / zoom box",fill=(230,230,230))
    path = out / f"hold_{short}.png"
    canvas.save(path)
    return dict(scene_dir=row["scene_dir"], png=str(path.resolve()), visible_pixels=int(mask.sum()),
                component_areas=components, crop_xyxy=[x0,y0,x1,y1], v3_png=row["v3_png"], v4_png=row["v4_png"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v4", type=Path, default=V4)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(args.out)
    rows = [json.loads(x) for x in (args.v4/"manifest.jsonl").read_text().splitlines()]
    holds = sorted((x for x in rows if x["split"] == "excluded" and x["action"] == "HOLD"),key=lambda x:x["scene_dir"])
    if len(holds) != 6 or any(x["parent_global_id"] != 1874325323 for x in holds):
        raise ValueError("expected exactly six registered excluded HOLD scenes")
    args.out.mkdir(parents=True)
    result = [render(row,args.out) for row in holds]
    (args.out/"index.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps([(r["scene_dir"],r["visible_pixels"],r["component_areas"][:3]) for r in result]))


if __name__ == "__main__":
    main()
