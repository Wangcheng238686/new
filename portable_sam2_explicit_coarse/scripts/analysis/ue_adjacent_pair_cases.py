"""Deterministically selected engineering overlays; not publication statistics."""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from pycocotools import mask as mu

from ue_adjacent_pair_audit import group, load, rles


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    pairs = sorted(
        load(a.output / "pairs.json"), key=lambda x: (x["image_id"], x["slots"])
    )
    manifest = load(a.manifest)
    points = {x["index"]: x for x in manifest["points"]}
    preds = {
        n: group(load(a.source / n / "dt_records.json"))
        for n in ["single_best", "dual_correct"]
    }
    gt = group(load(a.source / "single_best/gt_records.json"))

    def flags(row):
        s = row["modes"]["single_best"]["0.5"]
        d = row["modes"]["dual_correct"]["0.5"]
        return s, d

    categories = {
        "dual_improved": lambda s, d: not s["mask_pair_success"]
        and d["mask_pair_success"],
        "dual_regressed": lambda s, d: s["mask_pair_success"]
        and not d["mask_pair_success"],
        "both_fail_dual_box_ok": lambda s, d: not s["mask_pair_success"]
        and d["mask_fail_box_success"],
        "both_fail_dual_nms_loss": lambda s, d: not s["mask_pair_success"]
        and not d["mask_pair_success"]
        and d["nms_coverage_loss"],
        "both_fail_dual_rpn_lack": lambda s, d: not s["mask_pair_success"]
        and not d["mask_pair_success"]
        and d["rpn_lacks_coverage"],
        "both_success": lambda s, d: s["mask_pair_success"] and d["mask_pair_success"],
    }
    selected = []
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    for category, predicate in categories.items():
        row = next((r for r in pairs if predicate(*flags(r))), None)
        if row is None:
            selected.append(dict(category=category, pair=None))
            continue
        image_id = row["image_id"]
        i, j = row["slots"]
        truth = mu.decode(rles(gt[image_id])).transpose(2, 0, 1).astype(bool)
        union = truth[i] | truth[j]
        yy, xx = np.where(union)
        x0, x1 = max(0, xx.min() - 12), min(512, xx.max() + 13)
        y0, y1 = max(0, yy.min() - 12), min(512, yy.max() + 13)
        sat = Path(manifest["satellite_root"]) / points[image_id]["satellite_dir"]
        rgb = cv2.cvtColor(
            cv2.imread(str(sat / "satellite_rgb.0000.png")), cv2.COLOR_BGR2RGB
        )
        fig, axes = plt.subplots(1, 4, figsize=(14, 4.6), layout="constrained")
        fig.suptitle(
            f"{category} | image {image_id}, GT slots {i}/{j}, global {'/'.join(row['physical_pair'])}\n"
            f"{row['stratum']}, center distance {row['distance_pixel_centers_px']:.2f}px | success threshold IoU=0.5",
            fontsize=11,
        )
        axes[0].imshow(rgb)
        axes[0].add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor="yellow",
                linewidth=1.5,
            )
        )
        axes[0].set_title("Satellite RGB / crop extent")
        for ax in axes[1:]:
            ax.imshow(rgb)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)
        for k, color in [(i, "#E69F00"), (j, "#0072B2")]:
            axes[1].contour(truth[k], levels=[0.5], colors=[color], linewidths=1.5)
            cy, cx = np.argwhere(truth[k]).mean(axis=0)
            cx = np.clip(cx, x0 + (x1 - x0) * 0.12, x1 - (x1 - x0) * 0.16)
            cy = np.clip(cy, y0 + (y1 - y0) * 0.10, y1 - (y1 - y0) * 0.10)
            axes[1].text(
                cx,
                cy,
                f"GT{k}",
                color="white",
                fontsize=9,
                bbox=dict(facecolor="black", alpha=0.7, edgecolor="none"),
            )
        axes[1].set_title("GT visible-instance contours")
        details = {}
        for ax, name in zip(axes[2:], ["single_best", "dual_correct"]):
            dt = preds[name][image_id]
            masks = mu.decode(rles(dt)).transpose(2, 0, 1).astype(bool)
            area = truth[[i, j]].sum(axis=(1, 2))
            cover = np.array(
                [
                    (masks & truth[k]).sum(axis=(1, 2)) / ar
                    for k, ar in zip([i, j], area)
                ]
            )
            relevant = np.flatnonzero(np.max(cover, axis=0) >= 0.1)
            for index, pred in enumerate(relevant):
                color = palette[index % len(palette)]
                ax.contour(masks[pred], levels=[0.5], colors=[color], linewidths=1.5)
                local = masks[pred, y0:y1, x0:x1]
                if local.any():
                    cy, cx = np.argwhere(local).mean(axis=0)
                    cx = np.clip(cx, (x1 - x0) * 0.10, (x1 - x0) * 0.82)
                    cy = np.clip(cy, (y1 - y0) * 0.10, (y1 - y0) * 0.90)
                    ax.text(
                        cx + x0,
                        cy + y0,
                        f"P{pred}",
                        color="white",
                        fontsize=9,
                        bbox=dict(facecolor="black", alpha=0.7, edgecolor="none"),
                    )
            for k in [i, j]:
                ax.contour(
                    truth[k],
                    levels=[0.5],
                    colors=["white"],
                    linestyles="dashed",
                    linewidths=0.7,
                )
            data = row["modes"][name]["0.5"]
            ax.set_title(
                f"{name}\nmask pair={data['mask_pair_success']}, box pair={data['box_pair_success']}",
                fontsize=10,
            )
            details[name] = dict(
                displayed_prediction_indices=relevant.tolist(), pair_flags=data
            )
        for ax in axes:
            ax.set_axis_off()
        path = a.output / f"case_{category}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        selected.append(
            dict(category=category, pair=row, details=details, figure=str(path))
        )
    (a.output / "cases.json").write_text(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
