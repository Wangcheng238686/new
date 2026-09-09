#!/usr/bin/env python
"""Compare the RCNN_SAMPLER_NUM 128-vs-256 screening arms.

Parses both per-arm tee logs and prints, at matched epochs:
  - segm/mAP and bbox/mAP trajectories (from per-epoch validation),
  - debug_mask_pos_rois (realized positive-RoI count per image),
  - mask-head training loss (loss_mask) and total loss,
  - window means (E1-10 / E11-20 / E21-30) for a noise-robust readout.

Usage:
  python scripts/smoke/compare_roi_screen.py <num128.log> <num256.log>
"""
from __future__ import annotations

import re
import statistics
import sys
from collections import defaultdict


def parse(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    segm = {}
    bbox = {}
    for m in re.finditer(
        r"Epoch (\d+)/\d+ \|.*?segm/mAP=([0-9.]+)", text
    ):
        segm[int(m.group(1))] = float(m.group(2))
    for m in re.finditer(
        r"Validation bbox/mAP: ([0-9.]+)", text
    ):
        pass  # paired below with epoch lines
    # pair bbox prints with the following epoch summary
    lines = text.splitlines()
    pend = None
    for ln in lines:
        m = re.search(r"Validation bbox/mAP: ([0-9.]+)", ln)
        if m:
            pend = float(m.group(1))
            continue
        if pend is not None:
            m2 = re.search(r"Epoch (\d+)/\d+ \|", ln)
            if m2:
                bbox[int(m2.group(1))] = pend
                pend = None
    pos_rois = {}
    for m in re.finditer(
        r"Epoch (\d+) detailed losses: .*?debug_mask_pos_rois=([0-9.]+)", text
    ):
        pos_rois[int(m.group(1))] = float(m.group(2))
    loss_mask = {}
    for m in re.finditer(
        r"Epoch (\d+) detailed losses: .*?\bloss_mask=([0-9.]+)", text
    ):
        loss_mask[int(m.group(1))] = float(m.group(2))
    return dict(segm=segm, bbox=bbox, pos_rois=pos_rois, loss_mask=loss_mask)


def window(d, lo, hi):
    vals = [v for e, v in d.items() if lo <= e <= hi]
    return (statistics.mean(vals), len(vals)) if vals else (float("nan"), 0)


def main():
    a = parse(sys.argv[1])
    b = parse(sys.argv[2])
    epochs = sorted(set(a["segm"]) | set(b["segm"]))
    print("ep   segm128 segm256 | bbox128 bbox256 | posR128 posR256 | lmask128 lmask256")
    for e in epochs:
        print(
            f"{e:3d}  {a['segm'].get(e, float('nan')):.4f}  {b['segm'].get(e, float('nan')):.4f}"
            f" | {a['bbox'].get(e, float('nan')):.4f}  {b['bbox'].get(e, float('nan')):.4f}"
            f" | {a['pos_rois'].get(e, float('nan')):6.1f} {b['pos_rois'].get(e, float('nan')):6.1f}"
            f" | {a['loss_mask'].get(e, float('nan')):.4f}  {b['loss_mask'].get(e, float('nan')):.4f}"
        )
    print("\nwindow means (segm / bbox / pos_rois):")
    for lo, hi in ((1, 10), (11, 20), (21, 30), (1, 30)):
        row = []
        for k in ("segm", "bbox", "pos_rois"):
            ma, na = window(a[k], lo, hi)
            mb, nb = window(b[k], lo, hi)
            row.append(f"{k}: {ma:.4f}(n{na}) vs {mb:.4f}(n{nb}) d={ma-mb:+.4f}")
        print(f"  E{lo}-{hi}: " + " | ".join(row))


if __name__ == "__main__":
    main()
