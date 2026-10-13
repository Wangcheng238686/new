#!/usr/bin/env python3
"""Render frozen v3 train/val spot checks without editing any label raster."""
import argparse
import csv
import html
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


PACKAGE = Path(__file__).resolve().parents[2]
ROOT = PACKAGE / "outputs/SURFBEV-MASK-V3-20261012"


def read_label(path):
    with Image.open(path) as image:
        if image.mode != "L" or image.size != (512, 512):
            raise ValueError(f"unexpected label contract: {path}")
        return np.asarray(image).copy()


def tint(image, mask, color):
    result = image.copy()
    result[mask] = (result[mask].astype(np.float32) * .45 + np.asarray(color) * .55).astype(np.uint8)
    return result


def outline(image, mask, color, thickness=1):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, thickness)


def crop_box(mask):
    ys, xs = np.where(mask)
    if not len(xs):
        return (0, 0, 512)
    side = min(512, max(48, int(xs.max()-xs.min()+25), int(ys.max()-ys.min()+25)))
    cx, cy = (int(xs.min())+int(xs.max()))/2, (int(ys.min())+int(ys.max()))/2
    x0 = max(0, min(512-side, int(round(cx-side/2))))
    y0 = max(0, min(512-side, int(round(cy-side/2))))
    return x0, y0, side


def render(row, dest):
    old = read_label(row["source_png"])
    new = read_label(row["v3_png"])
    rgb = np.asarray(Image.open(row["satellite_rgb"]).convert("RGB"))
    if rgb.shape != (512, 512, 3):
        raise ValueError(f"unexpected RGB shape: {row['satellite_rgb']}")
    removed = (old != 0) & (new == 0)
    if not np.array_equal(old[~removed], new[~removed]) or int(removed.any()) != int(row["changed"]):
        raise ValueError(f"v3 diff disagrees with sample manifest: {row['scene_dir']}")
    declared = {int(value) for value in row["removed_ids"].split("|") if value}
    if set(np.unique(old[removed]).tolist()) != declared:
        raise ValueError(f"removed IDs disagree: {row['scene_dir']}")

    # Changed scenes zoom the largest removed instance. Controls zoom the largest kept instance.
    if declared:
        focus_id = max(declared, key=lambda instance_id: int((old == instance_id).sum()))
        focus = old == focus_id
    else:
        ids, areas = np.unique(old[old > 0], return_counts=True)
        if not len(ids):
            raise ValueError(f"empty control label: {row['scene_dir']}")
        focus_id = int(ids[np.argmax(areas)])
        focus = old == focus_id
    x0, y0, side = crop_box(focus)

    full = rgb.copy()
    # A dilated marker makes one-pixel fragments discoverable at full-image scale.
    if declared:
        marker = cv2.dilate(removed.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        outline(full, marker, (255, 30, 170), 2)
    before = rgb.copy()
    after = rgb.copy()
    if declared:
        before = tint(before, removed, (255, 30, 170))
        outline(before, removed, (255, 235, 0), 1)
    else:
        before = tint(before, focus, (30, 220, 255))
        after = tint(after, new == focus_id, (30, 220, 255))
    crop = (slice(y0, y0+side), slice(x0, x0+side))
    before_zoom = Image.fromarray(before[crop]).resize((512, 512), Image.Resampling.NEAREST)
    after_zoom = Image.fromarray(after[crop]).resize((512, 512), Image.Resampling.NEAREST)
    image = Image.new("RGB", (1536, 546), (34, 37, 43))
    for index, panel in enumerate((Image.fromarray(full), before_zoom, after_zoom)):
        image.paste(panel, (512*index, 34))
    draw = ImageDraw.Draw(image)
    for index, label in enumerate(("RGB / removed markers", "v2 / before (zoom)", "v3 / after (zoom)")):
        draw.text((512*index+8, 9), label, fill=(245, 245, 245))
    image.save(dest / "full" / f"{row['scene_dir']}.jpg", quality=90, subsampling=0)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, default=ROOT / "spotcheck.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "spotcheck_visual")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing to overwrite: {args.out}")
    rows = list(csv.DictReader(args.sample.open()))
    counts = Counter((r["split"], r["changed"]) for r in rows)
    if len(rows) != 48 or len({r["scene_dir"] for r in rows}) != 48 or counts != Counter({("train", "1"): 12, ("train", "0"): 12, ("val", "1"): 12, ("val", "0"): 12}):
        raise ValueError(f"unexpected frozen sample: {counts}")
    (args.out / "full").mkdir(parents=True)
    for row in rows:
        render(row, args.out)
    parts = ["<!doctype html><html lang=zh-CN><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>",
             "<title>SurfaceBEV v3 标签抽样对照</title>",
             "<style>:root{color-scheme:light dark;font-family:system-ui,sans-serif}body{max-width:1600px;margin:auto;padding:20px;line-height:1.5}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:18px}.sample{min-width:0;border:1px solid color-mix(in srgb,currentColor 20%,transparent);padding:8px}.sample img{width:100%;height:auto}.sample code{overflow-wrap:anywhere}</style>",
             "<h1>v3 数据侧清洗：固定分层抽样 48 场</h1>",
             "<p>左：完整卫星 RGB，粉色圈标记被删除像素。中：v2 清洗前局部，粉色为被删除标签。右：v3 清洗后同一局部。无变更场景的蓝色为保留实例，前后应一致。仅抽 train/val，不含 test。</p><main>"]
    for row in rows:
        scene = html.escape(row["scene_dir"])
        parts.append(f'<article class="sample"><a href="full/{scene}.jpg"><img src="full/{scene}.jpg" loading="lazy" alt="{scene} 的清洗前后对照"></a><div>{html.escape(row["split"])} / {"已变更" if row["changed"] == "1" else "未变更"} · 删除 ID：{html.escape(row["removed_ids"] or "无")}<br><code>{scene}</code></div></article>')
    parts.append("</main></html>")
    (args.out / "index.html").write_text("\n".join(parts), encoding="utf-8")
    print(f"rendered {len(rows)} samples: {args.out / 'index.html'}")


if __name__ == "__main__":
    main()
