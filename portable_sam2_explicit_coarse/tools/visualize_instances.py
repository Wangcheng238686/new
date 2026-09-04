#!/usr/bin/env python3
"""Visualize COCO-format GT annotations and model predictions.

Works for any COCO-style instance dataset in this repo (WHU, iSAID patches,
UE-COCO, ...). Coordinates are handled per source:

- ``gt``   : annotation json coordinates are in the stored-image frame; drawn
             directly on the stored image.
- ``pred`` : prediction records from inference/infer_from_checkpoint.py are in
             the MODEL frame (image_size from the checkpoint, rescale=False).
             Masks are RLE-encoded at model size; they are decoded and scaled
             back to the stored image automatically.
- ``merge``: tiled patch predictions (iSAID-style names ``stem_y0_y1_x0_x1``)
             are shifted back to the ORIGINAL image coordinate system,
             deduplicated with per-class box NMS over overlap regions, and
             drawn on the original image. Untiled files pass through as plain
             per-image overlays (so WHU works here too).

Examples:
    # GT overlay (WHU validation)
    python scripts/visualize_instances.py gt \
        --ann-json "/data1/wangcheng/dataset/WHU/2.4 annotation/annotation/validation.json" \
        --images-dir "/data1/wangcheng/dataset/WHU/2.3 valid/validation" \
        --out-dir logs/viz/whu_gt --limit 8

    # Prediction overlay (optionally with GT outlines behind)
    python scripts/visualize_instances.py pred \
        --pred-json runs/inference_val/predictions.json \
        --out-dir logs/viz/whu_pred --score-thr 0.3 --limit 20 \
        --gt-ann-json .../validation.json

    # Merge tiled patch predictions back onto original images
    python scripts/visualize_instances.py merge \
        --pred-json runs/inference_val/predictions.json \
        --orig-images-dir /data1/wangcheng/dataset/iSAID/val/images \
        --out-dir logs/viz/isaid_merged --nms-iou 0.5 --limit 4
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("viz")

# 16 visually distinct colors (BGR).
PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (70, 240, 240), (240, 50, 230), (210, 245, 60),
    (250, 190, 190), (0, 128, 128), (220, 190, 255), (170, 110, 40),
    (255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0),
]

# Fallback single-class name (id -> name); used when no categories source
# is available. WHU single-class maps id 1 -> building either way.
WHU_SINGLE_CLASS_NAMES = {1: "building"}

PATCH_NAME_RE = re.compile(r"^(?P<stem>.+)_(?P<y0>\d+)_(?P<y1>\d+)_(?P<x0>\d+)_(?P<x1>\d+)\.png$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out-dir", required=True)
    common.add_argument("--limit", type=int, default=10, help="max images to render")
    common.add_argument("--score-thr", type=float, default=0.0)
    common.add_argument("--line-thickness", type=int, default=2)
    common.add_argument("--mask-alpha", type=float, default=0.35)
    common.add_argument("--mask-only", action="store_true",
                        help="仅彩色 mask 叠加原图(每实例独立颜色, 无框/无文字/无置信度)")
    common.add_argument("--font-scale", type=float, default=0.5)
    common.add_argument("--categories-json", type=str, default="",
                        help="optional json with a categories list (id/name)")

    p_gt = sub.add_parser("gt", parents=[common], help="draw GT annotations")
    p_gt.add_argument("--ann-json", required=True)
    p_gt.add_argument("--images-dir", required=True)

    p_pred = sub.add_parser("pred", parents=[common], help="draw predictions")
    p_pred.add_argument("--pred-json", required=True)
    p_pred.add_argument("--gt-ann-json", default="",
                        help="optional GT json: outlines drawn beneath predictions")
    p_pred.add_argument("--images-dir", default="",
                        help="image dir override (records carry absolute paths by default)")

    p_merge = sub.add_parser("merge", parents=[common],
                             help="merge tiled patch predictions onto original images")
    p_merge.add_argument("--pred-json", required=True)
    p_merge.add_argument("--orig-images-dir", required=True,
                         help="directory with the untiled original images")
    p_merge.add_argument("--nms-iou", type=float, default=0.5)
    p_merge.add_argument("--max-side", type=int, default=4096,
                         help="downscale output canvas so max(H,W) <= this")

    return parser.parse_args()


# ---------------------------------------------------------------- helpers

def decode_segmentation(seg, h: int, w: int) -> Optional[np.ndarray]:
    """Decode COCO segmentation (polygon list or RLE dict) to uint8 (h,w)."""
    if not seg:
        return None
    if isinstance(seg, dict) and "counts" in seg:
        from pycocotools import mask as mask_utils

        counts = seg["counts"]
        if isinstance(counts, str):
            counts = counts.encode("utf-8")
        rle = {"size": seg["size"], "counts": counts}
        mask = mask_utils.decode(rle)
        if mask.ndim == 3:
            mask = mask[..., 0]
        return mask.astype(np.uint8)
    if isinstance(seg, list):
        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in seg:
            if not isinstance(poly, list) or len(poly) < 6:
                continue
            pts = np.round(np.asarray(poly, dtype=np.float64).reshape(-1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
        return mask
    return None


def resize_mask(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    if mask.shape[0] == h and mask.shape[1] == w:
        return mask
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def parse_patch_name(file_name: str) -> Optional[Tuple[str, int, int, int, int]]:
    m = PATCH_NAME_RE.match(Path(file_name).name)
    if not m:
        return None
    g = m.groupdict()
    return g["stem"], int(g["y0"]), int(g["y1"]), int(g["x0"]), int(g["x1"])


def load_categories(args, ann: Optional[dict] = None,
                    observed_ids: Optional[set] = None) -> Dict[int, str]:
    """Resolve category names.

    Priority: --categories-json > ann json categories > heuristic on the ids
    actually present (single-class id=1 datasets like WHU -> "building",
    multi-class -> canonical iSAID names) > generic classN.
    """
    names: Dict[int, str] = {}
    if args.categories_json:
        with open(args.categories_json) as f:
            for cat in json.load(f).get("categories", []):
                names[int(cat["id"])] = cat["name"]
    elif ann and ann.get("categories"):
        for cat in ann["categories"]:
            names[int(cat["id"])] = cat["name"]
    if not names and observed_ids is not None:
        if observed_ids and observed_ids <= {1}:
            names = dict(WHU_SINGLE_CLASS_NAMES)

    def lookup(cat_id: int) -> str:
        if cat_id in names:
            return names[cat_id]
        return f"class{cat_id}"
    return {cid: lookup(cid) for cid in sorted(set(names) | set(observed_ids or set()))}


def color_for(cat_id: int) -> Tuple[int, int, int]:
    return PALETTE[int(cat_id - 1) % len(PALETTE)]


def draw_instances(img: np.ndarray, records: List[dict], cat_names: Dict[int, str],
                   args, with_scores: bool, outline_only: bool = False) -> np.ndarray:
    """records: bbox xyxy float, category_id int, optional 'mask' uint8 HxW."""
    canvas = img.copy()
    overlay = img.copy()
    h, w = canvas.shape[:2]
    font = max(0.4, args.font_scale * max(h, w) / 800.0)
    thickness = max(1, args.line_thickness * max(h, w) // 1000)

    if getattr(args, "mask_only", False):
        # 仅彩色 mask 叠加原图: 每实例独立颜色, 无框/无文字/无置信度
        # (与对比算法 whu_test_*.py 的可视化口径一致)
        for i, rec in enumerate(records):
            m = rec.get("mask")
            if m is None or not m.any():
                continue
            overlay[m > 0] = PALETTE[i % len(PALETTE)]
        return cv2.addWeighted(overlay, args.mask_alpha, canvas,
                               1.0 - args.mask_alpha, 0)

    for rec in records:
        cat_id = int(rec["category_id"])
        color = color_for(cat_id)
        x1, y1, x2, y2 = rec["bbox"]
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        if outline_only:
            cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 215, 255), max(1, thickness // 2))
            continue
        mask = rec.get("mask")
        if mask is not None and mask.any():
            overlay[mask > 0] = color
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, color, thickness)
        else:
            cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        label = cat_names.get(cat_id, f"class{cat_id}")
        if with_scores:
            label = f"{label} {float(rec.get('score', 0)):.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font, 1)
        ty = max(th + 4, cy)
        cv2.rectangle(canvas, (cx - tw // 2 - 2, ty - th - 4), (cx + tw // 2 + 2, ty), color, -1)
        cv2.putText(canvas, label, (cx - tw // 2, ty - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, font, (255, 255, 255), 1, cv2.LINE_AA)

    if outline_only:
        return canvas
    return cv2.addWeighted(overlay, args.mask_alpha, canvas, 1.0 - args.mask_alpha, 0)


def read_image(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path))
    if img is None:
        logger.warning("cannot read %s", path)
    return img


# ---------------------------------------------------------------- modes

def mode_gt(args) -> None:
    with open(args.ann_json) as f:
        ann = json.load(f)
    cat_names = load_categories(args, ann)
    images_dir = Path(args.images_dir)
    anns_by_img: Dict[int, List[dict]] = defaultdict(list)
    for a in ann.get("annotations", []):
        anns_by_img[a["image_id"]].append(a)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered = 0
    for img_entry in ann.get("images", []):
        if rendered >= args.limit:
            break
        file_name = img_entry["file_name"]
        img = read_image(images_dir / file_name)
        if img is None:
            continue
        h, w = img.shape[:2]
        records = []
        for a in anns_by_img.get(img_entry["id"], []):
            mask = decode_segmentation(a.get("segmentation"), h, w)
            x, y, bw, bh = a["bbox"]
            records.append({"bbox": [x, y, x + bw, y + bh],
                            "category_id": a["category_id"], "mask": mask})
        if not records:
            continue
        vis = draw_instances(img, records, cat_names, args, with_scores=False)
        cv2.imwrite(str(out_dir / f"gt_{Path(file_name).stem}.png"), vis)
        rendered += 1
    logger.info("gt: rendered %d -> %s", rendered, out_dir)


def load_predictions(pred_json: str, score_thr: float) -> List[dict]:
    with open(pred_json) as f:
        preds = json.load(f)
    kept = [p for p in preds if float(p.get("score", 0.0)) >= score_thr]
    logger.info("predictions: %d total, %d after score>=%.3f", len(preds), len(kept), score_thr)
    return kept


def predictions_for_image(preds: List[dict], img_h: int, img_w: int) -> List[dict]:
    """Convert model-frame prediction records to the stored-image frame.

    RLE size reveals the frame the mask lives in; anything not matching the
    image is rescaled (masks nearest, boxes by scale factors).
    """
    records = []
    for p in preds:
        bx, by, bw, bh = p["bbox"]
        seg = p.get("segmentation")
        mask = None
        sx = sy = 1.0
        if isinstance(seg, dict) and "size" in seg:
            mh, mw = int(seg["size"][0]), int(seg["size"][1])
            mask = decode_segmentation(seg, mh, mw)
            sx, sy = img_w / mw, img_h / mh
        elif isinstance(seg, list):
            mask = decode_segmentation(seg, img_h, img_w)
        if mask is not None:
            mask = resize_mask(mask, img_h, img_w)
        records.append({
            "bbox": [bx * sx, by * sy, (bx + bw) * sx, (by + bh) * sy],
            "category_id": int(p["category_id"]),
            "score": float(p.get("score", 0.0)),
            "mask": mask,
        })
    return records


def mode_pred(args) -> None:
    preds = load_predictions(args.pred_json, args.score_thr)
    if not preds:
        logger.warning("no predictions pass the score threshold")
        return
    gt_anns_by_file: Dict[str, List[dict]] = {}
    gt_w_h: Dict[str, Tuple[int, int]] = {}
    cat_names = None
    if args.gt_ann_json:
        with open(args.gt_ann_json) as f:
            ann = json.load(f)
        cat_names = load_categories(args, ann)
        ids = {im["id"]: im for im in ann.get("images", [])}
        for a in ann.get("annotations", []):
            entry = ids.get(a["image_id"])
            if entry:
                gt_anns_by_file.setdefault(entry["file_name"], []).append(a)
                gt_w_h[entry["file_name"]] = (entry["height"], entry["width"])
    if cat_names is None:
        cat_names = load_categories(args, observed_ids={int(p["category_id"]) for p in preds})

    by_file: Dict[str, List[dict]] = defaultdict(list)
    for p in preds:
        name = p.get("file_name") or ""
        by_file[name].append(p)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered = 0
    for name, plist in by_file.items():
        if rendered >= args.limit:
            break
        path = Path(args.images_dir or "/") / Path(name).name if args.images_dir else Path(name)
        img = read_image(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        records = predictions_for_image(plist, h, w)
        if args.gt_ann_json:
            rel = path.name
            gt_list = gt_anns_by_file.get(rel)
            if gt_list is not None:
                gh, gw = gt_w_h[rel]
                gt_records = []
                for a in gt_list:
                    m = decode_segmentation(a.get("segmentation"), gh, gw)
                    m = resize_mask(m, h, w) if m is not None else None
                    x, y, bw2, bh2 = a["bbox"]
                    sx, sy = w / gw, h / gh
                    gt_records.append({"bbox": [x * sx, y * sy, (x + bw2) * sx, (y + bh2) * sy],
                                       "category_id": a["category_id"], "mask": m})
                img = draw_instances(img, gt_records, cat_names, args,
                                     with_scores=False, outline_only=True)
        vis = draw_instances(img, records, cat_names, args, with_scores=True)
        cv2.imwrite(str(out_dir / f"pred_{path.stem}.png"), vis)
        rendered += 1
    logger.info("pred: rendered %d -> %s", rendered, out_dir)


def box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def class_nms(records: List[dict], iou_thr: float) -> List[dict]:
    """Greedy per-category box NMS; records carry xyxy 'bbox' + 'category_id'."""
    kept: List[dict] = []
    by_cat: Dict[int, List[dict]] = defaultdict(list)
    for r in records:
        by_cat[r["category_id"]].append(r)
    for _, items in by_cat.items():
        items.sort(key=lambda r: -r["score"])
        for item in items:
            if all(box_iou(item["bbox"], k["bbox"]) < iou_thr for k in
                   [x for x in kept if x["category_id"] == item["category_id"]]):
                kept.append(item)
    return kept


def find_original(stem: str, orig_dir: Path) -> Optional[Path]:
    for ext in (".png", ".jpg", ".tif", ".jpeg"):
        p = orig_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def mode_merge(args) -> None:
    preds = load_predictions(args.pred_json, args.score_thr)
    if not preds:
        logger.warning("no predictions pass the score threshold")
        return
    cat_names = load_categories(args, observed_ids={int(p["category_id"]) for p in preds})
    orig_dir = Path(args.orig_images_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # group by source stem; each record carries its own window (patches of the
    # same source image have different offsets — per-record, never per-stem)
    groups: Dict[str, List[dict]] = defaultdict(list)
    for p in preds:
        parsed = parse_patch_name(p.get("file_name") or "")
        if parsed is None:
            stem = Path(p.get("file_name") or "unknown").stem
            p["_win"] = None
            groups[stem].append(p)
        else:
            stem, y0, y1, x0, x1 = parsed
            p["_win"] = (x0, y0, x1, y1)
            groups[stem].append(p)

    rendered = 0
    for stem, plist in groups.items():
        if rendered >= args.limit:
            break
        orig_path = find_original(stem, orig_dir)
        if orig_path is None:
            logger.warning("original image not found for stem %s", stem)
            continue
        img = read_image(orig_path)
        if img is None:
            continue
        H, W = img.shape[:2]
        tiled = any(p["_win"] is not None for p in plist)
        merged_dir = out_dir / "merged_json"
        records_orig: List[dict] = []

        if not tiled:
            # untiled (e.g. WHU): predictions already in image frame
            canvas = img
            h, w = H, W
            for p in plist:
                bx, by, bw, bh = p["bbox"]
                records_orig.append({"bbox": [bx, by, bx + bw, by + bh],
                                     "category_id": int(p["category_id"]),
                                     "score": float(p.get("score", 0.0)),
                                     "patch": Path(p["file_name"]).name})
            nms_records = class_nms(records_orig, args.nms_iou)
            vis_records = []
            for r in nms_records:
                seg = next(p.get("segmentation") for p in plist
                           if Path(p["file_name"]).name == r["patch"])
                mask = None
                if isinstance(seg, dict) and "size" in seg:
                    mh, mw = int(seg["size"][0]), int(seg["size"][1])
                    mask = resize_mask(decode_segmentation(seg, mh, mw), H, W)
                r["mask"] = mask
                vis_records.append(r)
        else:
            for p in plist:
                x0, y0, x1, y1 = p["_win"]
                win_w, win_h = x1 - x0, y1 - y0
                seg = p.get("segmentation")
                sx = sy = 1.0
                if isinstance(seg, dict) and "size" in seg:
                    mh, mw = int(seg["size"][0]), int(seg["size"][1])
                    sx, sy = win_w / mw, win_h / mh
                bx, by, bw, bh = p["bbox"]
                ox1, oy1 = x0 + bx * sx, y0 + by * sy
                ox2, oy2 = x0 + (bx + bw) * sx, y0 + (by + bh) * sy
                records_orig.append({
                    "bbox": [ox1, oy1, ox2, oy2],
                    "category_id": int(p["category_id"]),
                    "score": float(p.get("score", 0.0)),
                    "patch": Path(p["file_name"]).name,
                    "_seg": seg, "_win": (x0, y0, win_w, win_h),
                })
            nms_records = class_nms(records_orig, args.nms_iou)

            # draw at (possibly downscaled) canvas resolution; masks are
            # decoded per instance at patch scale to avoid huge allocations
            scale = min(1.0, args.max_side / max(H, W))
            canvas = cv2.resize(img, (int(W * scale), int(H * scale))) if scale < 1.0 else img
            ch, cw = canvas.shape[:2]
            vis_records = []
            for r in nms_records:
                seg = r.pop("_seg")
                wx0, wy0, win_w2, win_h2 = r.pop("_win")
                mask = None
                if isinstance(seg, dict) and "size" in seg:
                    mh, mw = int(seg["size"][0]), int(seg["size"][1])
                    m = decode_segmentation(seg, mh, mw)
                    # model frame -> patch frame -> canvas window
                    m = cv2.resize(m, (win_w2, win_h2), interpolation=cv2.INTER_NEAREST)
                    cx0, cy0 = int(wx0 * scale), int(wy0 * scale)
                    cw2, ch2 = max(1, int(round(win_w2 * scale))), max(1, int(round(win_h2 * scale)))
                    m = cv2.resize(m, (cw2, ch2), interpolation=cv2.INTER_NEAREST)
                    full = np.zeros((ch, cw), dtype=np.uint8)
                    full[cy0:cy0 + ch2, cx0:cx0 + cw2] = m
                    mask = full
                # keep original-frame bbox in r for the merged json; the draw
                # record gets the canvas-scaled copy
                bbox = r["bbox"]
                vis_records.append({
                    **{k: v for k, v in r.items() if k != "bbox"},
                    "bbox": [bbox[0] * scale, bbox[1] * scale,
                             bbox[2] * scale, bbox[3] * scale],
                    "mask": mask,
                })

        vis = draw_instances(canvas, vis_records,
                             cat_names, args, with_scores=True)
        cv2.imwrite(str(out_dir / f"merged_{stem}.png"), vis)

        merged_dir.mkdir(exist_ok=True)
        merged = {
            "file_name": orig_path.name,
            "width": W,
            "height": H,
            "nms_iou": args.nms_iou,
            "instances": [
                {"bbox_xyxy": [float(v) for v in r["bbox"]],
                 "category_id": r["category_id"],
                 "score": r["score"],
                 "patch": r["patch"]}
                for r in nms_records
            ],
        }
        with open(merged_dir / f"{stem}.json", "w") as f:
            json.dump(merged, f, indent=1)
        logger.info("merged %s: %d raw -> %d after NMS", stem, len(plist), len(nms_records))
        rendered += 1
    logger.info("merge: rendered %d -> %s", rendered, out_dir)


def main() -> None:
    args = parse_args()
    if args.mode == "gt":
        mode_gt(args)
    elif args.mode == "pred":
        mode_pred(args)
    else:
        mode_merge(args)


if __name__ == "__main__":
    main()
