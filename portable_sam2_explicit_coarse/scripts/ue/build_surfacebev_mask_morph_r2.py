#!/usr/bin/env python3
"""R2: retain only R1 micro-holes at least 4 px inside their own instance."""
import json
from pathlib import Path

import cv2
import numpy as np

import build_surfacebev_mask_morph_r1 as core


R2_OUT = core.PACKAGE / "outputs/SURFBEV-MASK-MORPH-R2-20261012"
R1_HOLES = core.enclosed_holes


def interior_holes(ids):
    keep = []
    cache = {}
    for lid, ys, xs, area in R1_HOLES(ids):
        if lid not in cache:
            cache[lid] = (ids == lid).astype(np.uint8)
        filled = cache[lid].copy()
        filled[ys, xs] = 1
        distance = cv2.distanceTransform(filled, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        border = np.minimum.reduce((ys, xs, 511 - ys, 511 - xs))
        if np.all(distance[ys, xs] >= 4.0) and np.all(border >= 4):
            keep.append((lid, ys, xs, area))
    return keep


def main():
    core.OUT = R2_OUT
    core.enclosed_holes = interior_holes
    core.main()
    summary_path = R2_OUT / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["protocol"] = "SURFBEV-MASK-MORPH-R2-20261012"
    summary["rule"] = "R1 enclosed <=4px background hole AND every pixel >=4px from other-label/background and image border"
    summary["supersedes_failed_candidate"] = "SURFBEV-MASK-MORPH-R1-20261012"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
