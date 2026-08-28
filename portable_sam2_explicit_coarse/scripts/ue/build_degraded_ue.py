"""Build a degraded-satellite copy of the UE paired dataset.

Only the satellite PNGs are regenerated (real files); annotations, UAV views
and scene metadata are symlinked back to the source dataset, so the directory
contract is unchanged and every existing loader / third-party codebase works
by switching ``--data-root`` alone.

Degradation is applied on the native 512x512 satellite tile:

  down2   512 ->(INTER_AREA)-> 256 ->(INTER_LINEAR)-> 512   ~1.4 m/px GSD
  down4   512 ->(INTER_AREA)-> 128 ->(INTER_LINEAR)-> 512   ~2.8 m/px GSD
  blur    Gaussian blur, sigma configurable (mild atmospheric)
  cloud   soft white cloud patches covering --cloud-ratio of the area

Usage:
  python scripts/ue/build_degraded_ue.py --mode down2 --out /data1/.../ue_coco_with_uav_d2
  python scripts/ue/build_degraded_ue.py --mode cloud --cloud-ratio 0.30 \
      --out /data1/.../ue_coco_with_uav_c30

The script is idempotent (existing degraded PNGs are skipped unless --force)
and writes a DEGRADATION.md manifest with the recipe, seed and file counts.
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

SPLITS = ("train", "val", "test")


def degrade_down(img: np.ndarray, factor: int) -> np.ndarray:
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // factor, h // factor), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def degrade_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    k = int(sigma * 3) * 2 + 1
    return cv2.GaussianBlur(img, (k, k), sigma)


def degrade_cloud(img: np.ndarray, ratio: float, rng: np.random.RandomState) -> np.ndarray:
    """Soft elliptical cloud patches covering ~ratio of the image area."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.float32)
    target = ratio * h * w
    # seed the field with a few large clouds, then grow until area is reached
    while mask.sum() < target:
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        ax, ay = rng.uniform(0.15, 0.45) * w, rng.uniform(0.12, 0.35) * h
        angle = rng.uniform(0, 180)
        patch = np.zeros((h, w), np.float32)
        cv2.ellipse(patch, (int(cx), int(cy)), (int(ax), int(ay)), angle, 0, 360, 1.0, -1)
        patch = cv2.GaussianBlur(patch, (0, 0), min(h, w) / 24.0)
        mask = np.clip(mask + patch, 0, 1)
    mask3 = mask[..., None]
    cloud_rgb = np.full_like(img, 235, dtype=np.uint8)
    noise = rng.randint(-12, 12, size=img.shape, dtype=np.int16)
    cloud_rgb = np.clip(cloud_rgb.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    out = (img.astype(np.float32) * (1 - mask3) + cloud_rgb.astype(np.float32) * mask3)
    return out.astype(np.uint8)


def apply_mode(img: np.ndarray, mode: str, sigma: float, cloud_ratio: float,
               rng: np.random.RandomState) -> np.ndarray:
    if mode == "down2":
        return degrade_down(img, 2)
    if mode == "down4":
        return degrade_down(img, 4)
    if mode == "blur":
        return degrade_blur(img, sigma)
    if mode == "cloud":
        return degrade_cloud(img, cloud_ratio, rng)
    raise ValueError(f"unknown mode {mode}")


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="/data1/wangcheng/dataset/ue_coco_with_uav")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", required=True, choices=["down2", "down4", "blur", "cloud"])
    ap.add_argument("--sigma", type=float, default=1.5, help="blur mode gaussian sigma")
    ap.add_argument("--cloud-ratio", type=float, default=0.30, help="cloud mode area ratio")
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--splits", nargs="*", default=list(SPLITS))
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    if not src.is_dir():
        sys.exit(f"source dataset not found: {src}")
    if out.resolve() == src.resolve():
        sys.exit("refusing to write the degraded copy onto the source dataset")

    out.mkdir(parents=True, exist_ok=True)

    # scene metadata: real copy (tiny)
    meta = out / "scene_centers.json"
    if not meta.exists():
        shutil.copy2(src / "scene_centers.json", meta)

    counts, skipped = {}, 0
    t0 = time.time()
    for split in args.splits:
        sat_src = src / split / "satellite"
        sat_out = out / split / "satellite"
        sat_out.mkdir(parents=True, exist_ok=True)

        # UAV views and anything else in the split: symlink the whole entry so
        # the degraded copy stays in sync with future source additions.
        for entry in (src / split).iterdir():
            if entry.name == "satellite":
                continue
            link = out / split / entry.name
            if not link.exists():
                link.symlink_to(entry.resolve())

        pngs = sorted(sat_src.glob("*.png"))
        for i, png in enumerate(pngs):
            dst = sat_out / png.name
            ann = sat_src / (png.stem + ".json")
            ann_link = sat_out / ann.name
            if not ann_link.exists():
                ann_link.symlink_to(ann.resolve())
            if dst.exists() and not args.force:
                skipped += 1
                continue
            img = cv2.imread(str(png))
            rng = np.random.RandomState(args.seed + int(png.stem) if png.stem.isdigit()
                                        else args.seed + i)
            cv2.imwrite(str(dst), apply_mode(img, args.mode, args.sigma,
                                             args.cloud_ratio, rng))
        counts[split] = len(pngs)
        print(f"[{split}] satellite PNGs: {len(pngs)} (skipped existing: see total)",
              flush=True)

    recipe = dict(
        source=str(src), mode=args.mode, sigma=args.sigma,
        cloud_ratio=args.cloud_ratio, seed=args.seed, splits=args.splits,
        op=("INTER_AREA down -> INTER_LINEAR up to native 512"
            if args.mode.startswith("down") else
            f"gaussian blur sigma={args.sigma}" if args.mode == "blur" else
            f"soft elliptical clouds, target area ratio {args.cloud_ratio}"),
        created=time.strftime("%Y-%m-%d %H:%M:%S"), counts=counts,
        note=("satellite PNGs are real degraded files; annotations and UAV "
              "views are symlinks to the source. GT masks/labels unchanged."),
    )
    (out / "DEGRADATION.md").write_text(
        "# Degraded UE dataset\n\n```json\n" + json.dumps(recipe, indent=2) + "\n```\n")
    print(f"done in {time.time()-t0:.1f}s -> {out} (degraded-new files, "
          f"skipped-existing={skipped})")


if __name__ == "__main__":
    main()
