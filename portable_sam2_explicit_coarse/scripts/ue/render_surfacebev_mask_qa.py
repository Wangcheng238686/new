#!/usr/bin/env python3
"""Render exact R1 mask overlays; never change the source RGB or ID raster."""
import argparse
import csv
import hashlib
import html
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


PACKAGE = Path(__file__).resolve().parents[2]
SAMPLE = PACKAGE / "outputs/SURFBEV-MASK-QA-20261012-R1/sample_manifest.csv"
OUTPUT = PACKAGE / "outputs/SURFBEV-MASK-QA-20261012-R1/visual_review"
STRATA = ("micro_1_4", "tiny_5_15", "thin", "major_fragment", "sparse", "border", "control")


def read_ids(path):
    a = np.asarray(Image.open(path))
    if a.ndim == 3:
        if not (np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 0], a[..., 2])):
            raise ValueError(f"non-grayscale ID raster: {path}")
        a = a[..., 0]
    if a.shape != (512, 512):
        raise ValueError(f"unexpected ID shape: {path}: {a.shape}")
    return a


def overlay(rgb, mask, bbox):
    canvas = rgb.copy()
    tint = np.array([255, 40, 160], dtype=np.float32)
    canvas[mask] = np.clip(canvas[mask].astype(np.float32) * .62 + tint * .38, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (255, 235, 0), 1)
    x0, y0, x1, y1 = bbox
    cv2.rectangle(canvas, (x0, y0), (x1-1, y1-1), (20, 255, 120), 1)
    return Image.fromarray(canvas)


def render_one(row, out):
    src_rgb, src_ids = Path(row["satellite_rgb"]), Path(row["clean_id_png"])
    if hashlib.sha256(src_ids.read_bytes()).hexdigest() != row["clean_id_png_sha256"]:
        raise RuntimeError(f"ID PNG changed since sampling: {src_ids}")
    rgb = np.asarray(Image.open(src_rgb).convert("RGB"))
    if rgb.shape != (512, 512, 3):
        raise ValueError(f"unexpected RGB shape: {src_rgb}")
    mask = read_ids(src_ids) == int(row["local_instance_id"])
    if int(mask.sum()) != int(row["area_px_512"]):
        raise RuntimeError(f"mask area changed: {row['sample_id']}")
    x0, y0, x1, y1 = (int(row[k]) for k in ("x0", "y0", "x1_exclusive", "y1_exclusive"))
    box = (x0, y0, x1, y1)
    side = min(512, max(64, x1-x0+32, y1-y0+32))
    cx, cy = (x0+x1)/2, (y0+y1)/2
    left = max(0, min(512-side, int(round(cx-side/2))))
    top = max(0, min(512-side, int(round(cy-side/2))))
    zoom = overlay(rgb[top:top+side, left:left+side], mask[top:top+side, left:left+side],
                   (x0-left, y0-top, x1-left, y1-top)).resize((512, 512), Image.Resampling.NEAREST)
    view = Image.new("RGB", (1536, 512))
    view.paste(Image.fromarray(rgb), (0, 0))
    view.paste(overlay(rgb, mask, box), (512, 0))
    view.paste(zoom, (1024, 0))
    sid = row["sample_id"]
    view.save(out / "full" / f"{sid}.jpg", "JPEG", quality=88, subsampling=0)
    view.resize((768, 256), Image.Resampling.LANCZOS).save(out / "thumb" / f"{sid}.jpg", "JPEG", quality=72)
    return view


def write_overview(rows, rendered, out):
    chosen = []
    for part in ("train", "val"):
        for group in STRATA:
            row = next((r for r in rows if r["split"] == part and r["stratum"] == group), None)
            if row:
                chosen.append(row)
    item_w, item_h, ncols = 512, 215, 3
    sheet = Image.new("RGB", (item_w*ncols, item_h*((len(chosen)+ncols-1)//ncols)), (35, 38, 43))
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(chosen):
        x, y = (i%ncols)*item_w, (i//ncols)*item_h
        sheet.paste(rendered[row["sample_id"]].resize((item_w, 171), Image.Resampling.LANCZOS), (x, y+30))
        draw.text((x+6, y+6), f'{row["split"]} / {row["stratum"]} | ID {row["local_instance_id"]} | {row["area_px_512"]} px', fill=(245, 245, 245))
    sheet.save(out / "overview.jpg", "JPEG", quality=85)


def write_index(rows, out):
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>SurfaceBEV 掩码审阅图</title>',
             '<style>:root{color-scheme:light dark;font-family:system-ui,sans-serif}body{max-width:1500px;margin:auto;padding:20px;line-height:1.5}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(450px,1fr));gap:20px}.sample{border:1px solid color-mix(in srgb,currentColor 22%,transparent);padding:10px;min-width:0}.sample img{display:block;width:100%;height:auto}.sample small{display:block;overflow-wrap:anywhere;opacity:.7}@media(max-width:520px){main{grid-template-columns:1fr}}</style>',
             '<h1>建筑实例掩码复核</h1>',
             '<p>每项从左至右：原始卫星 RGB、当前实例叠加、局部放大。粉色=实例像素；黄色=边界；绿色=外接框。点击图片看全分辨率。形态异常仅供排查，不等于错标。</p>',
             '<main>']
    for row in rows:
        sid = html.escape(row["sample_id"])
        part, group = html.escape(row["split"]), html.escape(row["stratum"])
        scene = html.escape(row["scene_dir"])
        parts.append(f'<article class="sample"><a href="full/{sid}.jpg" target="_blank"><img src="thumb/{sid}.jpg" loading="lazy" alt="原图、实例叠加和局部放大：{sid}"></a><div><strong>{part} · {group}</strong> · <code>{sid}</code> · local {row["local_instance_id"]} / global {row["global_instance_id"]} · {row["area_px_512"]} px</div><small>{scene}</small></article>')
    parts.append('</main></html>')
    (out / "index.html").write_text('\n'.join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, default=SAMPLE)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing visual review: {args.out}")
    rows = list(csv.DictReader(args.sample.open()))
    if len(rows) != 113 or len({r["sample_id"] for r in rows}) != 113:
        raise RuntimeError("expected the frozen 113-instance R1 sample")
    if any(r["split"] not in ("train", "val") for r in rows):
        raise RuntimeError("test/excluded sample is prohibited")
    (args.out / "full").mkdir(parents=True)
    (args.out / "thumb").mkdir()
    rendered = {r["sample_id"]: render_one(r, args.out) for r in rows}
    write_overview(rows, rendered, args.out)
    write_index(rows, args.out)
    print(f"rendered {len(rows)} exact overlays to {args.out}")


if __name__ == "__main__":
    main()
