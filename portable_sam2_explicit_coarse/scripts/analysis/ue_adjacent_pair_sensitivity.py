"""Explicitly post-hoc descriptive checks motivated by case inspection."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

from ue_adjacent_pair_audit import group, load, rles


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    pairs = load(a.output / "pairs.json")
    instances = load(a.output / "instances.json")
    records = {(r["image_id"], r["gt_slot"]): r for r in instances}
    gt = group(load(a.source / "single_best/gt_records.json"))
    border = {}
    fragment = {}
    for name in ["single_best", "dual_correct"]:
        dt = group(load(a.source / name / "dt_records.json"))
        for image_id, truth in gt.items():
            gs, ds = rles(truth), rles(dt[image_id])
            garea, darea = mu.area(gs).astype(float), mu.area(ds).astype(float)
            ious = np.asarray(mu.iou(gs, ds, [0] * len(ds)))
            inter = ious * (garea[:, None] + darea[None, :]) / (1 + ious)
            coverage = inter / garea[:, None]
            purity = inter / np.maximum(darea[None, :], 1)
            dd_iou = np.asarray(mu.iou(ds, ds, [0] * len(ds)))
            for i, row in enumerate(truth):
                x, y, w, h = row["bbox"]
                border[(image_id, i)] = bool(
                    x == 0 or y == 0 or x + w == 512 or y + h == 512
                )
                candidates = np.flatnonzero((coverage[i] >= 0.25) & (purity[i] >= 0.5))
                found = []
                # Require a failed instance; a successfully matched GT never gets this proxy.
                if not records[(image_id, i)]["modes"][name]["0.5"]["mask_matched"]:
                    for j, k in itertools.combinations(candidates, 2):
                        if dd_iou[j, k] > 0.1:
                            continue
                        gm = mu.decode(gs[i]).astype(bool)
                        combined = mu.decode(ds[j]).astype(bool) | mu.decode(
                            ds[k]
                        ).astype(bool)
                        if (gm & combined).sum() / garea[i] >= 0.5:
                            found = [int(j), int(k)]
                            break
                fragment[(name, image_id, i)] = found
    out = {
        "scope": "Post-hoc descriptive sensitivity; primary protocol unchanged",
        "strata": {},
        "fragment_instances": [],
    }
    for stratum in [
        "all",
        "any_border",
        "neither_border",
        "any_small",
        "neither_small",
        "interior_nonsmall",
    ]:
        selected = []
        for pair in pairs:
            image_id = pair["image_id"]
            idx = [(image_id, i) for i in pair["slots"]]
            b = any(border[k] for k in idx)
            s = any(records[k]["area"] < 32**2 for k in idx)
            include = {
                "all": True,
                "any_border": b,
                "neither_border": not b,
                "any_small": s,
                "neither_small": not s,
                "interior_nonsmall": not b and not s,
            }[stratum]
            if include:
                selected.append(pair)
        sub = dict(
            appearances=len(selected),
            unique_physical_pairs=len({tuple(x["physical_pair"]) for x in selected}),
            modes={},
        )
        for name in ["single_best", "dual_correct"]:
            stats = {}
            for t in ["0.5", "0.75"]:
                rows = [x["modes"][name][t] for x in selected]
                stats[t] = {
                    k: sum(x[k] for x in rows)
                    for k in [
                        "mask_pair_success",
                        "box_pair_success",
                        "mask_fail_box_success",
                        "mask_fail_box_fail",
                    ]
                }
                stats[t]["mask_fail_gtbox_both_pass"] = sum(
                    not x["mask_pair_success"] and x["gt_box_both_masks_pass"]
                    for x in rows
                )
            stats["mask05_failure_with_fragment_proxy"] = sum(
                not x["modes"][name]["0.5"]["mask_pair_success"]
                and any(fragment[(name, x["image_id"], i)] for i in x["slots"])
                for x in selected
            )
            sub["modes"][name] = stats
        out["strata"][stratum] = sub
    for (name, image_id, i), found in fragment.items():
        if found:
            out["fragment_instances"].append(
                dict(
                    mode=name,
                    image_id=image_id,
                    gt_slot=i,
                    global_id=records[(image_id, i)]["global_id"],
                    area=records[(image_id, i)]["area"],
                    border=border[(image_id, i)],
                    prediction_indices=found,
                )
            )
    (a.output / "posthoc_sensitivity.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
