#!/usr/bin/env python3
"""Render fixed R2 micro-hole review with owner-mask context and band-specific crop."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


PACKAGE = Path(__file__).resolve().parents[2]
ROOT = PACKAGE / "outputs/SURFBEV-MASK-MORPH-R2-20261012"


def zoom(rgb, ids, lid, cy, cx, changed=None):
    y0 = max(0, min(448, cy - 32))
    x0 = max(0, min(448, cx - 32))
    crop = rgb[y0:y0+64, x0:x0+64].copy()
    owner = ids[y0:y0+64, x0:x0+64] == lid
    crop[owner] = (crop[owner].astype(np.float32) * .6 + np.array([20, 220, 240]) * .4).astype(np.uint8)
    contours, _ = cv2.findContours(owner.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(crop, contours, -1, (20, 220, 240), 1)
    if changed is not None:
        mask = changed[y0:y0+64, x0:x0+64]
        crop[mask] = [255, 35, 180]
    return Image.fromarray(crop).resize((512, 512), Image.Resampling.NEAREST)


def render(sample, record, out):
    rgb = np.asarray(Image.open(sample["rgb"]).convert("RGB"))
    old = np.asarray(Image.open(sample["v3_png"]).convert("L"))
    new = np.asarray(Image.open(sample["cleaned_png"]).convert("L"))
    changed = old != new
    if int(changed.sum()) != sample["changed_pixels"]:
        raise ValueError("sample pixel count mismatch")
    band = sample["band"]
    choices = [h for h in record["holes"] if
               (str(h["size"]) if h["size"] <= 2 else "3-4") == band]
    if band != "unchanged" and not choices:
        raise ValueError(f"no hole for review band: {sample['scene_dir']} {band}")
    if choices:
        target = choices[0]
        lid = int(target["local_id"])
        cy, cx = target["pixels"][0]
    else:
        lid = int(np.unique(old[old > 0])[0]) if np.any(old > 0) else 0
        cy, cx = 256, 256
    full = rgb.copy()
    full[changed] = [255, 35, 180]
    cv2.rectangle(full, (max(0, cx-32), max(0, cy-32)),
                  (min(511, cx+31), min(511, cy+31)), (255, 255, 255), 1)
    panels = (Image.fromarray(full), zoom(rgb, old, lid, cy, cx),
              zoom(rgb, new, lid, cy, cx, changed))
    canvas = Image.new("RGB", (1536, 548), (33, 36, 42))
    for i, panel in enumerate(panels):
        canvas.paste(panel, (512*i, 36))
    draw = ImageDraw.Draw(canvas)
    for i, label in enumerate(("Satellite RGB + target", "v3 owner mask (cyan)", "R2 owner + filled hole (pink)")):
        draw.text((512*i+8, 10), label, fill=(245, 245, 245))
    path = out / f"{sample['split']}_{band}_{sample['scene_dir']}.jpg"
    canvas.save(path, quality=91, subsampling=0)
    return path


def main():
    out = ROOT / "visual_review"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing overwrite {out}")
    out.mkdir(parents=True)
    manifest = {r["scene_dir"]: r for r in
                (json.loads(line) for line in (ROOT / "manifest.jsonl").read_text().splitlines() if line)}
    samples = [json.loads(line) for line in (ROOT / "review_sample.jsonl").read_text().splitlines() if line]
    result = []
    for sample in samples:
        path = render(sample, manifest[sample["scene_dir"]], out)
        result.append(dict(split=sample["split"], band=sample["band"],
                           scene_dir=sample["scene_dir"], image=str(path.resolve())))
    (out / "index.jsonl").write_text("".join(json.dumps(row) + "\n" for row in result))
    print(f"rendered {len(result)} images to {out}")


if __name__ == "__main__":
    main()
