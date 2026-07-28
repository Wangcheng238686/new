#!/usr/bin/env python3
"""Convert WHU-512 binary mask labels to COCO instance segmentation JSON.

Usage:
    python tools/mask_to_coco_whu512.py \
        --data-root /data/wangcheng/dataset/WHU-512 \
        --output-dir /data/wangcheng/dataset/WHU-512/annotations
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage


def mask_to_polygons(binary_mask: np.ndarray, min_area: int = 10):
    """Extract polygon contours from a single connected component mask."""
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1
    )
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        contour = contour.flatten().tolist()
        if len(contour) >= 6:
            polygons.append(contour)
    return polygons


def convert_split(data_root: Path, split: str, output_dir: Path, min_area: int = 10):
    """Convert one split (train/val/test) to COCO JSON."""
    image_dir = data_root / split / "image"
    label_dir = data_root / split / "label"

    if not image_dir.exists() or not label_dir.exists():
        print(f"[SKIP] {split}: {image_dir} or {label_dir} not found")
        return 0

    images = []
    annotations = []
    ann_id = 1

    filenames = sorted(f for f in os.listdir(image_dir) if f.lower().endswith((".tif", ".tiff", ".png", ".jpg")))
    print(f"[INFO] {split}: {len(filenames)} images")

    for idx, fname in enumerate(filenames):
        img_path = image_dir / fname
        img = Image.open(img_path)
        w, h = img.size

        images.append({
            "id": idx,
            "file_name": fname,
            "width": w,
            "height": h,
        })

        label_path = label_dir / fname
        if not label_path.exists():
            # Try common extensions
            stem = Path(fname).stem
            for ext in [".tif", ".tiff", ".png"]:
                candidate = label_dir / (stem + ext)
                if candidate.exists():
                    label_path = candidate
                    break

        if label_path.exists():
            label = np.array(Image.open(label_path))
            if label.dtype == bool:
                label = label.astype(np.uint8)

            labeled, num_instances = ndimage.label(label)
            for inst_id in range(1, num_instances + 1):
                inst_mask = (labeled == inst_id).astype(np.uint8)
                area = int(inst_mask.sum())
                if area < min_area:
                    continue

                polygons = mask_to_polygons(inst_mask, min_area=min_area)
                if not polygons:
                    continue

                ys, xs = np.where(inst_mask > 0)
                x_min, x_max = int(xs.min()), int(xs.max())
                y_min, y_max = int(ys.min()), int(ys.max())
                bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]

                annotations.append({
                    "id": ann_id,
                    "image_id": idx,
                    "category_id": 1,
                    "segmentation": polygons,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                })
                ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "building", "supercategory": "building"}],
    }

    output_path = output_dir / f"{split}.json"
    with open(output_path, "w") as f:
        json.dump(coco, f)

    print(f"[DONE] {split}: {len(images)} images, {len(annotations)} annotations -> {output_path}")
    return len(annotations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="/data/wangcheng/dataset/WHU-512")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--min-area", type=int, default=10, help="Min instance area in pixels")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir) if args.output_dir else data_root / "annotations"
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for split in ["train", "val", "test"]:
        total += convert_split(data_root, split, output_dir, args.min_area)

    print(f"\n[SUMMARY] Total annotations: {total}")


if __name__ == "__main__":
    main()
