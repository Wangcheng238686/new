"""CPU-only, fixed-protocol audit of existing predictions. No model imports."""

import argparse
import collections
import hashlib
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment

NAMES = ["single_best", "dual_correct", "dual_zero", "dual_shift"]
THRESHOLDS = [0.5, 0.75]


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rles(rows):
    return [
        dict(
            size=x["segmentation"]["size"], counts=x["segmentation"]["counts"].encode()
        )
        for x in rows
    ]


def match(ious, threshold):
    """GT x prediction matrix. Cardinality precedes total IoU, dummy allowed."""
    n, p = ious.shape
    weights = np.full((n, p + n), -1e6, dtype=float)
    weights[:, p:] = 0
    weights[:, :p] = np.where(ious >= threshold, n + 1 + ious, -1e6)
    rows, cols = linear_sum_assignment(weights, maximize=True)
    assignment = np.full(n, -1, dtype=int)
    for i, j in zip(rows, cols):
        if j < p and ious[i, j] >= threshold:
            assignment[i] = j
    used = assignment[assignment >= 0]
    assert len(set(used)) == len(used)
    assert all(ious[i, j] >= threshold for i, j in enumerate(assignment) if j >= 0)
    return assignment


def pair_feasible(ious, i, j, threshold):
    a = np.flatnonzero(ious[i] >= threshold)
    b = np.flatnonzero(ious[j] >= threshold)
    return bool(len(a) and len(b) and (len(a) > 1 or len(b) > 1 or a[0] != b[0]))


def self_check():
    assert list(match(np.array([[0.8], [0.7]]), 0.5)).count(-1) == 1
    assert not pair_feasible(np.array([[0.8], [0.7]]), 0, 1, 0.5)
    # Greedy highest-IoU would lose one instance here.
    assert np.array_equal(match(np.array([[0.99, 0.51], [0.51, 0.0]]), 0.5), [1, 0])
    assert np.array_equal(match(np.zeros((2, 0)), 0.5), [-1, -1])


def group(rows):
    result = collections.defaultdict(list)
    for row in rows:
        result[row["image_id"]].append(row)
    return result


def largest(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    assert n > 1
    return labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def aggregate(pairs, gt_rows, ids):
    summary = dict(
        images=len(ids),
        gt_appearances=len(gt_rows),
        unique_buildings=len({x["global_id"] for x in gt_rows}),
        strata={},
    )
    for stratum in ["contact", "near", "all_adjacent", "largest_components_near"]:
        rows = [
            x
            for x in pairs
            if (
                stratum == "all_adjacent"
                or (
                    stratum == "largest_components_near"
                    and x["largest_components_near"]
                )
                or x["stratum"] == stratum
            )
        ]
        unique = collections.defaultdict(list)
        for row in rows:
            unique[tuple(row["physical_pair"])].append(row)
        sub = dict(
            appearances=len(rows),
            unique_physical_pairs=len(unique),
            images=len({x["image_id"] for x in rows}),
            modes={},
            transitions={},
        )
        for name in NAMES:
            ms = {}
            for t in THRESHOLDS:
                key = str(t)
                data = [x["modes"][name][key] for x in rows]
                counts = (
                    {
                        k: sum(bool(x[k]) for x in data)
                        for k in data[0]
                        if isinstance(data[0][k], bool)
                    }
                    if data
                    else {}
                )
                physical = [
                    np.mean([x["modes"][name][key]["mask_pair_success"] for x in v])
                    for v in unique.values()
                ]
                invasion = [
                    v for x in data for v in x["neighbor_coverage_given_box_match"]
                ]
                ms[key] = dict(
                    counts=counts,
                    mask_success_rate=(
                        counts.get("mask_pair_success", 0) / len(rows) if rows else None
                    ),
                    physical_pair_equal_mean=(
                        float(np.mean(physical)) if physical else None
                    ),
                    neighbor_coverage_assignment_count=len(invasion),
                    neighbor_coverage_mean=(
                        float(np.mean(invasion)) if invasion else None
                    ),
                    neighbor_coverage_ge10pct=sum(x >= 0.1 for x in invasion),
                    neighbor_coverage_ge50pct=sum(x >= 0.5 for x in invasion),
                )
            sub["modes"][name] = ms
        for t in THRESHOLDS:
            key = str(t)
            a = np.array(
                [x["modes"]["single_best"][key]["mask_pair_success"] for x in rows], int
            )
            b = np.array(
                [x["modes"]["dual_correct"][key]["mask_pair_success"] for x in rows],
                int,
            )
            sums = np.zeros((len(ids), 2))
            positions = {image_id: i for i, image_id in enumerate(ids)}
            for x, delta in zip(rows, b - a):
                sums[positions[x["image_id"]]] += (delta, 1)
            rng = np.random.default_rng(20261016)
            boot = sums[rng.integers(0, len(ids), size=(2000, len(ids)))].sum(axis=1)
            rates = boot[boot[:, 1] > 0, 0] / boot[boot[:, 1] > 0, 1]
            sub["transitions"][key] = dict(
                improved=int(((a == 0) & (b == 1)).sum()),
                regressed=int(((a == 1) & (b == 0)).sum()),
                both_success=int(((a == 1) & (b == 1)).sum()),
                both_fail=int(((a == 0) & (b == 0)).sum()),
                delta_rate=float((b - a).mean()) if len(rows) else None,
                exploratory_image_bootstrap_ci95=(
                    np.quantile(rates, [0.025, 0.975]).tolist() if len(rates) else None
                ),
            )
        summary["strata"][stratum] = sub
    summary["instance_strata"] = {}
    for adjacent in [False, True]:
        rows = [x for x in gt_rows if x["adjacent"] == adjacent]
        summary["instance_strata"][str(adjacent)] = dict(
            appearances=len(rows),
            unique_buildings=len({x["global_id"] for x in rows}),
            modes={
                name: {
                    str(t): {
                        kind: sum(x["modes"][name][str(t)][kind] for x in rows)
                        for kind in ["mask_matched", "box_matched", "gt_box_mask_pass"]
                    }
                    for t in THRESHOLDS
                }
                for name in NAMES
            },
        )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    self_check()
    cv2.setNumThreads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = load(args.manifest)
    points = {x["index"]: x for x in manifest["points"]}
    inputs = [args.manifest]
    artifacts = {}
    for name in NAMES:
        base = args.source / name
        paths = {
            k: base / f"{k}.json"
            for k in ["gt_records", "dt_records", "images", "metrics"]
        }
        paths["stage"] = args.source / "full_stage_audit" / f"{name}.json"
        paths["fixed"] = args.source / "full_fixed_box_audit" / f"{name}.json"
        inputs.extend(paths.values())
        data = {k: load(p) for k, p in paths.items()}
        data["gt"] = group(data["gt_records"])
        data["dt"] = group(data["dt_records"])
        data["stages"] = group(data["stage"]["instances"])
        data["fixed_gt"] = {
            x["image_id"]: x for x in data["fixed"]["records"] if x["box_type"] == "gt"
        }
        artifacts[name] = data
    ref = artifacts["single_best"]
    ids = [x["image_id"] for x in ref["images"]]
    assert len(ids) == 153 and len(ref["gt_records"]) == 4496
    for data in artifacts.values():
        assert data["images"] == ref["images"]
        assert data["gt_records"] == ref["gt_records"]
        assert data["stage"]["processed_image_ids"] == ids
        assert data["stage"]["manifest_sha256"] == sha(args.manifest)
    pairs, gt_rows, scales = [], [], []
    max_predictions = collections.Counter()
    for image_id in ids:
        gt = ref["gt"][image_id]
        n = len(gt)
        gs = rles(gt)
        masks = mu.decode(gs).transpose(2, 0, 1).astype(bool)
        assert np.max(masks.sum(axis=0)) <= 1
        areas = masks.sum(axis=(1, 2))
        assert np.all(areas > 0)
        boxes = np.array([x["bbox"] for x in gt])
        stages = ref["stages"][image_id]
        assert [x["gt_slot"] for x in stages] == list(range(n))
        globals_ = [x["global_id"] for x in stages]
        assert len(set(globals_)) == n
        sat = Path(manifest["satellite_root"]) / points[image_id]["satellite_dir"]
        label_path = sat / "satellite_instance_id_clean.png"
        labels = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        inputs.extend(
            [
                label_path,
                sat / "instances.json",
                sat / "camera_intrinsics.json",
                sat / "camera_extrinsics.json",
            ]
        )
        mapping = {
            x["local_instance_id"]: str(x["global_instance_id"])
            for x in load(sat / "instances.json")["instances"]
        }
        for i, row in enumerate(stages):
            assert np.array_equal(masks[i], labels == row["local_id"])
            assert mapping[row["local_id"]] == globals_[i]
            b = row["bbox_xyxy"]
            assert np.array_equal(boxes[i], [b[0], b[1], b[2] - b[0], b[3] - b[1]])
        k = load(sat / "camera_intrinsics.json")
        e = load(sat / "camera_extrinsics.json")
        assert k["fx"] == k["fy"] and abs(e["rotation_ue_deg"]["pitch"] + 90) < 1e-6
        scale = e["location_ue_cm"]["z"] / 100 / k["fx"]
        scales.append(scale)
        near = []
        big = [largest(x) for x in masks]
        for i in range(n):
            distance = distance_transform_edt(~masks[i])
            big_distance = None
            for j in range(i + 1, n):
                # Bounding-box lower bound avoids scanning remote masks.
                x, y, w, h = boxes[i]
                u, v, a, b = boxes[j]
                dx = max(x - u - a, u - x - w, 0)
                dy = max(y - v - b, v - y - h, 0)
                if dx * dx + dy * dy > 25:
                    continue
                d = float(distance[masks[j]].min())
                if d > 5:
                    continue
                if big_distance is None:
                    big_distance = distance_transform_edt(~big[i])
                near.append(
                    dict(
                        image_id=image_id,
                        slots=[i, j],
                        physical_pair=sorted([globals_[i], globals_[j]]),
                        distance_pixel_centers_px=d,
                        nominal_ground_distance_m=d * scale,
                        stratum="contact" if d <= np.sqrt(2) else "near",
                        largest_components_near=bool(big_distance[big[j]].min() <= 5),
                        modes={},
                    )
                )
        adjacency = {i for row in near for i in row["slots"]}
        per_instance = [
            dict(
                image_id=image_id,
                gt_slot=i,
                global_id=globals_[i],
                adjacent=i in adjacency,
                area=int(areas[i]),
                modes={},
            )
            for i in range(n)
        ]
        for name in NAMES:
            data = artifacts[name]
            dt = data["dt"][image_id]
            max_predictions[name] = max(max_predictions[name], len(dt))
            assert len(dt) <= 100
            ds = rles(dt)
            mask_iou = np.asarray(mu.iou(gs, ds, [0] * len(ds))).reshape(n, len(dt))
            box_iou = np.asarray(
                mu.iou([x["bbox"] for x in gt], [x["bbox"] for x in dt], [0] * len(dt))
            ).reshape(n, len(dt))
            p_area = np.array(mu.area(ds), dtype=float)
            inter = mask_iou * (areas[:, None] + p_area[None, :]) / (1 + mask_iou)
            coverage = inter / areas[:, None]
            stage = data["stages"][image_id]
            assert [x["global_id"] for x in stage] == globals_
            fixed = data["fixed_gt"][image_id]
            assert np.array_equal(fixed["boxes_xyxy"], [x["bbox_xyxy"] for x in stages])
            for t in THRESHOLDS:
                key = str(t)
                ma, ba = match(mask_iou, t), match(box_iou, t)
                for i in range(n):
                    per_instance[i]["modes"].setdefault(name, {})[key] = dict(
                        mask_matched=bool(ma[i] >= 0),
                        box_matched=bool(ba[i] >= 0),
                        gt_box_mask_pass=bool(fixed["mask_iou"][i] >= t),
                        mask_prediction=int(ma[i]),
                        box_prediction=int(ba[i]),
                    )
                for row in near:
                    i, j = row["slots"]
                    ms = bool(ma[i] >= 0 and ma[j] >= 0)
                    bs = bool(ba[i] >= 0 and ba[j] >= 0)
                    invasion = [float(coverage[j, ba[i]])] if ba[i] >= 0 else []
                    if ba[j] >= 0:
                        invasion.append(float(coverage[i, ba[j]]))
                    s = [stage[i], stage[j]]
                    labels_ = dict(
                        mask_pair_success=ms,
                        box_pair_success=bs,
                        mask_fail_box_success=bool(not ms and bs),
                        mask_fail_box_fail=bool(not ms and not bs),
                        mask_success_box_fail=bool(ms and not bs),
                        pair_local_mask_feasible=pair_feasible(mask_iou, i, j, t),
                        pair_local_box_feasible=pair_feasible(box_iou, i, j, t),
                        cross_building_mask_coverage=bool(
                            np.any((coverage[i] >= 0.5) & (coverage[j] >= 0.5))
                        ),
                        box_assigned_mask_failure=bool(
                            bs and (mask_iou[i, ba[i]] < t or mask_iou[j, ba[j]] < t)
                        ),
                        gt_box_both_masks_pass=bool(
                            fixed["mask_iou"][i] >= t and fixed["mask_iou"][j] >= t
                        ),
                        rpn_lacks_coverage=any(x["rpn_best_iou"] < t for x in s),
                        regression_coverage_loss=any(
                            x["rpn_best_iou"] >= t > x["raw_best_iou"] for x in s
                        ),
                        score_coverage_loss=any(
                            x["raw_best_iou"] >= t > x["eligible_best_iou"] for x in s
                        ),
                        nms_coverage_loss=any(
                            x["eligible_best_iou"] >= t > x["final_best_iou"] for x in s
                        ),
                        final_independent_coverage_but_global_conflict=bool(
                            not bs and all(x["final_best_iou"] >= t for x in s)
                        ),
                        neighbor_coverage_given_box_match=invasion,
                    )
                    # These columns have denominator all adjacent pairs, and failure-only companions.
                    for k_ in [
                        "rpn_lacks_coverage",
                        "regression_coverage_loss",
                        "score_coverage_loss",
                        "nms_coverage_loss",
                        "cross_building_mask_coverage",
                    ]:
                        labels_[k_ + "_among_box_fail"] = bool(labels_[k_] and not bs)
                        labels_[k_ + "_among_mask_fail"] = bool(labels_[k_] and not ms)
                    row["modes"].setdefault(name, {})[key] = labels_
        pairs.extend(near)
        gt_rows.extend(per_instance)
        print(image_id, "GT", n, "adjacent", len(near), flush=True)
    summary = aggregate(pairs, gt_rows, ids)
    summary.update(
        nominal_ground_m_per_px_range=[min(scales), max(scales)],
        max_predictions_per_image=dict(max_predictions),
        original_metrics={name: artifacts[name]["metrics"] for name in NAMES},
        checks="153 IDs, 4496 GT, four-arm identical GT, clean labels/global IDs/bboxes exact; unique assignments; synthetic checks PASS",
    )
    (args.output / "pairs.json").write_text(json.dumps(pairs))
    (args.output / "instances.json").write_text(json.dumps(gt_rows))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output / "source_identity.json").write_text(
        json.dumps({str(p): sha(p) for p in inputs}, indent=2)
    )


if __name__ == "__main__":
    main()
