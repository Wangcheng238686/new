#!/usr/bin/env python3
"""Build COCO-format 800x800 patches for iSAID from DOTA images + iSAID json.

Inputs (under --src):
    {set}/images/P####.png                    original DOTA images
    {set}/Annotations/iSAID_{set}.json        official json (polygons, 15 cats)
    {set}/Instance_masks/images/images/P####_instance_id_RGB.png   (QC only)

Outputs (under --tar):
    {set}/images/P####_y0_y1_x0_x1.png        patches (devkit naming convention)
    {set}/instances_isaid_{set}.json          COCO annotations for the patches
    _records/{set}/P####.json                 per-image sidecar records (resume)

Tiling mirrors iSAID_Devkit/preprocess/split.py: sliding window with
stride = patch - overlap, window shifted back to fit at right/bottom edges;
images with either side <= patch are copied whole. P1527/P1530 are skipped
(same as the devkit). Annotations entirely inside a window keep their original
polygons (shifted); clipped annotations are rasterized -> cropped ->
re-polygonized, with RLE fallback when contours degenerate.

Usage:
    python scripts/build_isaid_patches.py \
        --src /data1/wangcheng/dataset/iSAID \
        --tar /data1/wangcheng/dataset/iSAID/isaid_patches_800 \
        --sets train,val --workers 24
    python scripts/build_isaid_patches.py ... --qc 20   # after build
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_isaid")

# Known-corrupt iSAID train images, skipped by the official devkit split.py.
DEVKIT_SKIP_IMAGES = {"P1527", "P1530"}

# iSAID official category order (from iSAID_train.json).
CATEGORIES = [
    "storage_tank", "Large_Vehicle", "Small_Vehicle", "plane", "ship",
    "Swimming_pool", "Harbor", "tennis_court", "Ground_Track_Field",
    "Soccer_ball_field", "baseball_diamond", "Bridge", "basketball_court",
    "Roundabout", "Helicopter",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build iSAID 800x800 COCO patches")
    parser.add_argument("--src", required=True, help="iSAID root (train/ val/ with images + Annotations)")
    parser.add_argument("--tar", required=True, help="output root, e.g. .../isaid_patches_800")
    parser.add_argument("--sets", default="train,val", help="comma separated: train,val")
    parser.add_argument("--patch", type=int, default=800, help="patch width/height")
    parser.add_argument("--overlap", type=int, default=200, help="overlap between patches")
    parser.add_argument("--min-area", type=int, default=25,
                        help="drop clipped fragments smaller than this many pixels")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true", help="statistics only, no writes")
    parser.add_argument("--qc", type=int, default=0,
                        help="run QC on N random annotated patches against official instance_id PNGs")
    parser.add_argument("--qc-seed", type=int, default=0)
    parser.add_argument("--rebuild-json", action="store_true",
                        help="force re-merging per-image records into the final json")
    return parser.parse_args()


def compute_windows(img_w: int, img_h: int, patch: int, overlap: int) -> List[Tuple[int, int, int, int]]:
    """(x0, x1, y0, y1) windows; mirrors devkit split.py edge handling.
    Tiling only happens when both sides exceed the patch size (devkit gate),
    otherwise the whole image is a single window."""
    if img_h <= patch or img_w <= patch:
        return [(0, img_w, 0, img_h)]
    windows: List[Tuple[int, int, int, int]] = []
    xs = []
    for x in range(0, img_w, patch - overlap):
        x_str, x_end = x, x + patch
        if x_end > img_w:
            x_str -= x_end - img_w
            x_end = img_w
        xs.append((x_str, x_end))
    ys = []
    for y in range(0, img_h, patch - overlap):
        y_str, y_end = y, y + patch
        if y_end > img_h:
            y_str -= y_end - img_h
            y_end = img_h
        ys.append((y_str, y_end))
    for (x_str, x_end) in xs:
        for (y_str, y_end) in ys:
            windows.append((x_str, x_end, y_str, y_end))
    return windows


def polys_bbox(polys: List[List[float]]) -> Tuple[float, float, float, float]:
    xs, ys = [], []
    for p in polys:
        xs.extend(p[0::2])
        ys.extend(p[1::2])
    return min(xs), min(ys), max(xs), max(ys)


def rasterize_local(polys: List[List[float]], pad: int = 1) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
    """Rasterize polygons onto their own bbox canvas (hole-free union fill,
    matching WHUCocoInstanceDataset._decode_mask semantics)."""
    x0, y0, x1, y1 = polys_bbox(polys)
    x0i, y0i = max(int(np.floor(x0)) - pad, 0), max(int(np.floor(y0)) - pad, 0)
    x1i, y1i = int(np.ceil(x1)) + pad, int(np.ceil(y1)) + pad
    canvas = np.zeros((y1i - y0i, x1i - x0i), dtype=np.uint8)
    shifted = [np.round(np.asarray(p, dtype=np.float64).reshape(-1, 2) - [x0i, y0i]).astype(np.int32) for p in polys]
    cv2.fillPoly(canvas, shifted, 1)
    return (x0i, y0i, x1i, y1i), canvas


def mask_to_polygons(mask: np.ndarray) -> Optional[List[List[float]]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: List[List[float]] = []
    for cnt in contours:
        cnt = cnt.reshape(-1, 2)
        if cnt.shape[0] < 3:
            continue
        polys.append(cnt.astype(float).flatten().tolist())
    return polys or None


def mask_to_rle(mask: np.ndarray) -> Dict[str, object]:
    from pycocotools import mask as mask_utils

    rle = mask_utils.encode(np.asfortranarray(mask))
    return {"counts": rle["counts"].decode("utf-8"), "size": list(mask.shape)}


def clip_annotation(
    ann: Dict,
    local_box: Tuple[int, int, int, int],
    local_mask: np.ndarray,
    window: Tuple[int, int, int, int],
    min_area: int,
) -> Optional[Dict]:
    """Clip one pre-rasterized instance to a window and rebuild its COCO fields."""
    x0i, y0i, _, _ = local_box
    wx0, wx1, wy0, wy1 = window
    # window rect in local-canvas coords
    sx0, sy0 = wx0 - x0i, wy0 - y0i
    sx1, sy1 = sx0 + (wx1 - wx0), sy0 + (wy1 - wy0)
    sx0c, sy0c = max(sx0, 0), max(sy0, 0)
    sx1c, sy1c = min(sx1, local_mask.shape[1]), min(sy1, local_mask.shape[0])
    if sx0c >= sx1c or sy0c >= sy1c:
        return None
    crop = local_mask[sy0c:sy1c, sx0c:sx1c]
    area = int(crop.sum())
    if area < min_area:
        return None
    # crop origin in patch coords
    dx, dy = sx0c - sx0, sy0c - sy0
    polys = mask_to_polygons(crop)
    if polys is not None:
        seg: object = [[v + (dx if i % 2 == 0 else dy) for i, v in enumerate(p)] for p in polys]
    else:
        # embed the crop in a patch-sized canvas so RLE decodes at the right place
        canvas = np.zeros((wy1 - wy0, wx1 - wx0), dtype=np.uint8)
        canvas[dy:dy + crop.shape[0], dx:dx + crop.shape[1]] = crop
        seg = mask_to_rle(canvas)
    ys, xs = np.nonzero(crop)
    bx, by = float(xs.min()) + dx, float(ys.min()) + dy
    bw, bh = float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)
    return {
        "segmentation": seg,
        "area": area,
        "bbox": [bx, by, bw, bh],
        "category_id": ann["category_id"],
        "iscrowd": 0,
    }


def shift_polys(polys: List[List[float]], dx: float, dy: float) -> List[List[float]]:
    return [[v + (dx if i % 2 == 0 else dy) for i, v in enumerate(p)] for p in polys]


def process_image(task: Dict) -> Dict:
    """Worker: split one source image into patches and clip its annotations."""
    set_name = task["set"]
    stem = task["stem"]
    src_img = Path(task["src"]) / set_name / "images" / f"{stem}.png"
    tar_imgs = Path(task["tar"]) / set_name / "images"
    rec_path = Path(task["tar"]) / "_records" / set_name / f"{stem}.json"
    patch, overlap, min_area = task["patch"], task["overlap"], task["min_area"]

    img = cv2.imread(str(src_img), cv2.IMREAD_COLOR)
    if img is None:
        return {"stem": stem, "error": f"cannot read {src_img}"}
    img_h, img_w = img.shape[:2]

    windows = compute_windows(img_w, img_h, patch, overlap)
    patch_entries: List[Dict] = []
    ann_entries: List[Dict] = []

    anns = task["anns"]
    # Pre-rasterize only annotations that intersect at least one window (all of
    # them do; windows cover the image), lazily per annotation when first needed.
    raster_cache: Dict[int, Tuple[Tuple[int, int, int, int], np.ndarray]] = {}
    for idx, ann in enumerate(anns):
        bbox = ann["bbox"]  # xywh in source coords
        ax0, ay0 = bbox[0], bbox[1]
        ax1, ay1 = ax0 + bbox[2], ay0 + bbox[3]

        for wid, (wx0, wx1, wy0, wy1) in enumerate(windows):
            if ax0 >= wx1 or ax1 <= wx0 or ay0 >= wy1 or ay1 <= wy0:
                continue
            fully_inside = ax0 >= wx0 and ay0 >= wy0 and ax1 <= wx1 and ay1 <= wy1
            if fully_inside:
                seg = shift_polys(ann["segmentation"], -wx0, -wy0)
                patch_ann = {
                    "segmentation": seg,
                    "area": ann["area"],
                    "bbox": [bbox[0] - wx0, bbox[1] - wy0, bbox[2], bbox[3]],
                    "category_id": ann["category_id"],
                    "iscrowd": 0,
                }
            else:
                if idx not in raster_cache:
                    raster_cache[idx] = rasterize_local(ann["segmentation"])
                patch_ann = clip_annotation(
                    ann, raster_cache[idx][0], raster_cache[idx][1],
                    (wx0, wx1, wy0, wy1), min_area,
                )
                if patch_ann is None:
                    continue
            patch_ann["window"] = wid
            ann_entries.append(patch_ann)

    # Assign annotation to patch entries by window id.
    if not task.get("skip_writes"):
        tar_imgs.mkdir(parents=True, exist_ok=True)
    anns_by_window: Dict[int, List[Dict]] = {}
    for pa in ann_entries:
        anns_by_window.setdefault(pa.pop("window"), []).append(pa)

    for wid, (wx0, wx1, wy0, wy1) in enumerate(windows):
        fname = f"{stem}_{wy0}_{wy1}_{wx0}_{wx1}.png"
        patch_entries.append({
            "file_name": fname,
            "width": wx1 - wx0,
            "height": wy1 - wy0,
            "annotations": anns_by_window.get(wid, []),
        })
        if task.get("skip_writes"):
            continue
        out_path = tar_imgs / fname
        if not out_path.exists():
            cv2.imwrite(str(out_path), img[wy0:wy1, wx0:wx1])

    result = {
        "stem": stem,
        "source_size": [img_w, img_h],
        "patches": patch_entries,
    }
    if not task.get("skip_writes"):
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rec_path, "w") as f:
            json.dump(result, f)
    return result


def build_set(args, set_name: str) -> Dict:
    src = Path(args.src)
    tar = Path(args.tar)
    ann_path = src / set_name / "Annotations" / f"iSAID_{set_name}.json"
    logger.info("[%s] loading %s", set_name, ann_path)
    with open(ann_path) as f:
        coco = json.load(f)
    anns_by_img: Dict[int, List[Dict]] = {}
    # The official train/val jsons use DIFFERENT numeric category_id spaces
    # (e.g. val id 9 = Small_Vehicle while train id 9 = Ground_Track_Field).
    # Remap by category NAME to the canonical train order before anything else.
    canonical = {name: i + 1 for i, name in enumerate(CATEGORIES)}
    src_cats = {c["id"]: c["name"] for c in coco["categories"]}
    unknown_names = set()
    for ann in coco["annotations"]:
        name = ann.get("category_name") or src_cats.get(ann["category_id"])
        if name not in canonical:
            unknown_names.add(name)
            continue
        ann["category_id"] = canonical[name]
        anns_by_img.setdefault(ann["image_id"], []).append(ann)
    if unknown_names:
        raise ValueError(
            f"[{set_name}] category names not in canonical table: {sorted(unknown_names)}"
        )

    tasks = []
    missing = []
    for img_entry in coco["images"]:
        stem = Path(img_entry["file_name"]).stem
        if stem in DEVKIT_SKIP_IMAGES:
            logger.info("[%s] skip %s (devkit corrupt-image list)", set_name, stem)
            continue
        img_path = src / set_name / "images" / img_entry["file_name"]
        if not img_path.exists():
            missing.append(img_entry["file_name"])
            continue
        rec_path = tar / "_records" / set_name / f"{stem}.json"
        if rec_path.exists() and not args.rebuild_json:
            continue  # already processed; merged by record scan below
        tasks.append({
            "set": set_name,
            "stem": stem,
            "src": str(src),
            "tar": str(tar),
            "patch": args.patch,
            "overlap": args.overlap,
            "min_area": args.min_area,
            "anns": anns_by_img.get(img_entry["id"], []),
            "skip_writes": args.dry_run,
        })
    if missing:
        logger.warning("[%s] %d images missing on disk (first: %s)",
                       set_name, len(missing), missing[:5])

    logger.info("[%s] %d images to process (%d already recorded)",
                set_name, len(tasks), len(coco["images"]) - len(missing) - len(tasks))

    if tasks and not args.dry_run:
        workers = max(1, min(args.workers, len(tasks)))
        done = 0
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed([ex.submit(process_image, t) for t in tasks]):
                res = fut.result()
                if "error" in res:
                    logger.error("[%s] %s: %s", set_name, res["stem"], res["error"])
                    continue
                done += 1
                if done % 50 == 0 or done == len(tasks):
                    rate = done / max(time.time() - t0, 1e-6)
                    logger.info("[%s] %d/%d images (%.1f img/s)",
                                set_name, done, len(tasks), rate)
    elif tasks and args.dry_run:
        for t in tasks:
            process_image(t)

    # Merge per-image records (covers resumed runs too).
    records_dir = tar / "_records" / set_name
    record_files = sorted(records_dir.glob("*.json")) if records_dir.exists() else []
    images, annotations = [], []
    per_class = {i + 1: 0 for i in range(len(CATEGORIES))}
    for rec_file in record_files:
        with open(rec_file) as f:
            rec = json.load(f)
        img_id = len(images)
        for patch in rec["patches"]:
            images.append({
                "id": img_id,
                "file_name": patch["file_name"],
                "width": patch["width"],
                "height": patch["height"],
            })
            for pa in patch["annotations"]:
                per_class[pa["category_id"]] = per_class.get(pa["category_id"], 0) + 1
                annotations.append({
                    "id": len(annotations),
                    "image_id": img_id,
                    "segmentation": pa["segmentation"],
                    "category_id": pa["category_id"],
                    "iscrowd": pa["iscrowd"],
                    "area": pa["area"],
                    "bbox": pa["bbox"],
                })
            img_id += 1

    out = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": i + 1, "name": name} for i, name in enumerate(CATEGORIES)
        ],
    }
    if not args.dry_run:
        out_dir = tar / set_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"instances_isaid_{set_name}.json"
        with open(out_path, "w") as f:
            json.dump(out, f)
        logger.info("[%s] wrote %s", set_name, out_path)

    logger.info("[%s] summary: %d patches, %d annotations",
                set_name, len(images), len(annotations))
    per_class_str = ", ".join(
        f"{CATEGORIES[cid - 1]}={cnt}" for cid, cnt in sorted(per_class.items()) if cnt
    )
    logger.info("[%s] per-class: %s", set_name, per_class_str)
    return {"images": len(images), "annotations": len(annotations)}


def decode_official_instance_ids(png_path: Path) -> Optional[np.ndarray]:
    """iSAID instance id = R + G*256 (B unused)."""
    img = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        ids = img[:, :, 0].astype(np.int64) + img[:, :, 1].astype(np.int64) * 256
    else:
        ids = img.astype(np.int64)
    return ids


def decode_our_foreground(patch_anns: List[Dict], h: int, w: int) -> Tuple[np.ndarray, int]:
    mask = np.zeros((h, w), dtype=np.uint8)
    for pa in patch_anns:
        seg = pa["segmentation"]
        if isinstance(seg, dict):
            from pycocotools import mask as mask_utils

            rle = {"size": seg["size"], "counts": seg["counts"].encode("utf-8")}
            mask |= mask_utils.decode(rle)
        else:
            polys = [np.round(np.asarray(p, dtype=np.float64).reshape(-1, 2)).astype(np.int32) for p in seg]
            cv2.fillPoly(mask, polys, 1)
    return mask, len(patch_anns)


def run_qc(args, set_name: str, n_samples: int) -> None:
    import random

    src = Path(args.src)
    tar = Path(args.tar)
    with open(tar / set_name / f"instances_isaid_{set_name}.json") as f:
        coco = json.load(f)
    anns_by_img: Dict[int, List[Dict]] = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    candidates = [im for im in coco["images"] if anns_by_img.get(im["id"])]
    rng = random.Random(args.qc_seed)
    samples = rng.sample(candidates, min(n_samples, len(candidates)))
    logger.info("[qc:%s] %d/%d annotated patches sampled", set_name, len(samples), len(candidates))

    ious, count_agrees, count_ours_all, count_png_all = [], 0, 0, 0
    for im in samples:
        stem_parts = Path(im["file_name"]).stem.split("_")
        stem, wy0, wy1, wx0, wx1 = (
            "_".join(stem_parts[:-4]), int(stem_parts[-4]), int(stem_parts[-3]),
            int(stem_parts[-2]), int(stem_parts[-1]),
        )
        png = src / set_name / "Instance_masks" / "images" / "images" / f"{stem}_instance_id_RGB.png"
        if not png.exists():
            logger.warning("[qc] missing official png %s", png)
            continue
        ids = decode_official_instance_ids(png)
        if ids is None:
            continue
        crop = ids[wy0:wy1, wx0:wx1]
        official_mask = (crop > 0).astype(np.uint8)
        official_count = int(len(np.unique(crop[crop > 0])))
        our_mask, our_count = decode_our_foreground(
            anns_by_img[im["id"]], im["height"], im["width"]
        )
        inter = int(np.logical_and(our_mask, official_mask).sum())
        union = int(np.logical_or(our_mask, official_mask).sum())
        iou = inter / union if union else 1.0
        ious.append(iou)
        count_ours_all += our_count
        count_png_all += official_count
        if abs(our_count - official_count) <= max(2, int(0.05 * official_count)):
            count_agrees += 1
        logger.info("[qc] %s iou=%.4f ours=%d official=%d",
                    im["file_name"], iou, our_count, official_count)

    if ious:
        logger.info("[qc:%s] mean_fg_iou=%.4f median=%.4f min=%.4f | count_agree=%d/%d | anns ours=%d official=%d",
                    set_name, float(np.mean(ious)), float(np.median(ious)), float(np.min(ious)),
                    count_agrees, len(ious), count_ours_all, count_png_all)
        if float(np.mean(ious)) < 0.95:
            logger.warning("[qc:%s] mean IoU below 0.95 -- investigate before training", set_name)


def main() -> None:
    args = parse_args()
    for set_name in args.sets.split(","):
        stats = build_set(args, set_name)
        if args.qc > 0 and not args.dry_run:
            run_qc(args, set_name, args.qc)
        logger.info("[%s] final: %s", set_name, stats)


if __name__ == "__main__":
    main()
