#!/usr/bin/env python3
"""T1: frozen UAV-aided visible-rooftop occupancy semantic segmentation probe.

Registered in EXPERIMENT_GOVERNANCE.md Appendix T.  This is deliberately a
read-only feature experiment: it never modifies detector weights, proposals or
predictions.  It is not an instance-AP experiment.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from probe_fusion_boundary_confirmation import (
    FIT_SCENES, ROOT, TRAIN_RATIO, TRAIN_SEED, Tee, attach_dual_sample,
    data_hash, dual_predict_digest, feature_digest, frozen_features, git_value,
    mapping_hash, parameter_hash, sha256_bytes, sha256_file, typed_hash,
    world_grid,
)

sys.path.insert(0, str(ROOT))
from mmengine.config import Config
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmdet.registry import MODELS
from mmengine.registry import MODELS as MMENGINE_MODELS
from data.ue_dual_stream_dataset import UEDualStreamDataset
import mmdet.models  # noqa: F401
import rsprompter  # noqa: F401

if "DetDataPreprocessor" not in MMENGINE_MODELS:
    MMENGINE_MODELS.register_module(name="DetDataPreprocessor", module=DetDataPreprocessor)

SAMPLE_PER_CLASS = 750
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260934
SAMPLE_SEED_BASE = 20260934
ARMS = ("SAT", "SAT_DBEV_correct", "SAT_DBEV_mis")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--single-ckpt", type=Path, required=True)
    p.add_argument("--dual-ckpt", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/ue_dual_stream_p2p3_sparse.py"))
    p.add_argument("--data-root", type=Path, default=Path("/data1/wangcheng/dataset/ue_coco_with_uav"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--experiment-id", default="MHBEV-FUS-20260921-T1")
    return p.parse_args()


def occupancy(feature: dict[str, np.ndarray]) -> np.ndarray:
    return feature["occ"].reshape(-1).astype(np.uint8)


def fit_arm(x: np.ndarray, y: np.ndarray) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    clf = LogisticRegression(C=0.1, penalty="l2", solver="lbfgs", max_iter=2000,
                             random_state=SAMPLE_SEED_BASE, n_jobs=1)
    clf.fit(scaler.fit_transform(x), y)
    return scaler, clf


def score(scaler: StandardScaler, clf: LogisticRegression, x: np.ndarray) -> np.ndarray:
    return clf.predict_proba(scaler.transform(x))[:, 1]


def metrics(probability: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pred = probability >= 0.5
    yb = y.astype(bool)
    tp = int(np.logical_and(pred, yb).sum()); fp = int(np.logical_and(pred, ~yb).sum())
    fn = int(np.logical_and(~pred, yb).sum()); tn = int(np.logical_and(~pred, ~yb).sum())
    building_iou = tp / max(tp + fp + fn, 1)
    background_iou = tn / max(tn + fp + fn, 1)
    return {"miou": (building_iou + background_iou) / 2,
            "building_iou": building_iou,
            "building_f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "pr_auc": float(average_precision_score(y, probability)),
            "prevalence": float(y.mean())}


def bootstrap(delta: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    draws = rng.integers(0, len(delta), size=(BOOTSTRAP_SAMPLES, len(delta)))
    values = delta[draws].mean(axis=1)
    return {"mean": float(delta.mean()), "ci95_percentile": [float(v) for v in np.percentile(values, [2.5, 97.5])],
            "bootstrap_samples": BOOTSTRAP_SAMPLES}


def main() -> None:
    args = parse_args(); out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"refuse to overwrite existing artifact: {out}")
    out.mkdir(parents=True, exist_ok=True)
    tee = Tee(sys.stdout, out / "stdout.log"); previous_stdout = sys.stdout; sys.stdout = tee
    try:
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("T1 requires CUDA")
        for path in (args.single_ckpt, args.dual_ckpt, args.config, args.data_root / "scene_centers.json"):
            if not path.is_file(): raise FileNotFoundError(path)
        cfg = Config.fromfile(str(args.config)); model = MODELS.build(cfg.model)
        dual = torch.load(args.dual_ckpt, map_location="cpu", weights_only=False); dual = dual.get("model", dual.get("state_dict", dual))
        dload = model.load_state_dict(dual, strict=False)
        if dload.missing_keys or dload.unexpected_keys: raise RuntimeError(f"dual checkpoint contract: {dload}")
        single = torch.load(args.single_ckpt, map_location="cpu", weights_only=False); single = single.get("model", single.get("state_dict", single))
        sload = model.load_state_dict(single, strict=False)
        allowed = ("drone_encoder.", "geometry_bev.", "p2p3_gate.", "embedding_gate.")
        bad = [key for key in sload.missing_keys if not key.startswith(allowed)]
        if bad or sload.unexpected_keys: raise RuntimeError(f"single overlay contract: bad={bad}, unexpected={sload.unexpected_keys}")
        model = model.to(device).eval(); initial_state = parameter_hash(model)
        train = UEDualStreamDataset(args.data_root, "train", image_size=(1024, 1024), train_subset=TRAIN_RATIO, seed=TRAIN_SEED, load_uav=True)
        val = UEDualStreamDataset(args.data_root, "val", image_size=(1024, 1024), load_uav=True)
        if len(train) != 128 or len(val) != 259: raise RuntimeError("fixed dataset cardinality failed")
        fit_records = [(i, sid) for i, sid in enumerate(train.ids) if len(train[i]["gt_masks"]) > 0][:FIT_SCENES]
        if len(fit_records) != FIT_SCENES: raise RuntimeError("insufficient nonempty fit scenes")
        train_ids = [sid for _, sid in fit_records]; val_ids = list(val.ids)
        manifest = {"experiment_id": args.experiment_id, "status": "running_read_only_frozen_probe", "argv": sys.argv,
            "cwd": str(Path.cwd()), "python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "git_commit": git_value(["git", "rev-parse", "HEAD"]), "git_dirty": git_value(["git", "status", "--porcelain"]),
            "contract": {"fit_scenes": FIT_SCENES, "val_scenes": 259, "sample_per_class": SAMPLE_PER_CLASS, "C": .1, "threshold": .5,
                "projection": "z0/360m/W2C-fp32-autocast-disabled/5-view/4-FPN/bilinear/valid-view-average"},
            "inputs": {"single_checkpoint_sha256": sha256_file(args.single_ckpt), "dual_checkpoint_sha256": sha256_file(args.dual_ckpt),
                "config_sha256": sha256_file(args.config), "scene_centers_sha256": sha256_file(args.data_root / "scene_centers.json"),
                "resolved_config_sha256": sha256_bytes(cfg.pretty_text.encode()), "train_data_sha256": data_hash(args.data_root, train_ids, "train"),
                "val_data_sha256": data_hash(args.data_root, val_ids, "val")}, "train_ids_fit_80": train_ids, "val_ids": val_ids,
            "initial_model_state_typed_sha256": initial_state}
        (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        fit = {name: [] for name in ARMS}; labels = []; telemetry = []; val_features = []
        def scan(ds, records, split, collect):
            for progress, (index, sid) in enumerate(records):
                item = ds[index]; before, pb = dual_predict_digest(model, item, device)
                first = frozen_features(model, item, device); second = frozen_features(model, item, device)
                hashes, hashes2 = feature_digest(first), feature_digest(second)
                after, pa = dual_predict_digest(model, item, device)
                if hashes != hashes2 or before != after or parameter_hash(model) != initial_state or model.training:
                    raise RuntimeError(f"{sid}: frozen invariant drift")
                telemetry.append({"split": split, "scene_id": sid, "prediction_before": mapping_hash(before), "prediction_after": mapping_hash(after),
                    "feature_hash": mapping_hash(hashes), "feature_repeat_hash": mapping_hash(hashes2), "model_state_hash": initial_state,
                    "before_proposals_absent": pb["before_proposals_absent"], "after_proposals_absent": pb["after_proposals_absent"],
                    "repeat_before_proposals_absent": pa["before_proposals_absent"], "repeat_after_proposals_absent": pa["after_proposals_absent"],
                    "correct_valid_mean": float(first["correct_count"].mean()), "mis_valid_mean": float(first["mis_count"].mean())})
                if collect:
                    y = occupancy(first); rng = np.random.default_rng(SAMPLE_SEED_BASE + index)
                    pos, neg = np.flatnonzero(y), np.flatnonzero(~y.astype(bool)); n = min(SAMPLE_PER_CLASS, len(pos), len(neg))
                    if not n: raise RuntimeError(f"{sid}: occupancy class missing")
                    chosen = np.concatenate((rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)))
                    fit["SAT"].append(first["sat"][chosen]); fit["SAT_DBEV_correct"].append(np.concatenate((first["sat"][chosen], first["correct"][chosen]), 1))
                    fit["SAT_DBEV_mis"].append(np.concatenate((first["sat"][chosen], first["mis"][chosen]), 1)); labels.append(y[chosen])
                else: val_features.append((sid, first))
                if (progress + 1) % 10 == 0 or progress + 1 == len(records): print(f"{split}: {progress + 1}/{len(records)}", flush=True)
        with torch.inference_mode(): scan(train, fit_records, "train", True); scan(val, list(enumerate(val.ids)), "val", False)
        yfit = np.concatenate(labels); classifiers = {name: fit_arm(np.concatenate(values).astype(np.float32), yfit) for name, values in fit.items()}
        np.savez_compressed(out / "fitted_artifacts.npz", train_label=yfit,
            **{f"{name}_mean": pair[0].mean_ for name, pair in classifiers.items()}, **{f"{name}_scale": pair[0].scale_ for name, pair in classifiers.items()},
            **{f"{name}_coef": pair[1].coef_ for name, pair in classifiers.items()}, **{f"{name}_intercept": pair[1].intercept_ for name, pair in classifiers.items()})
        rows = []
        for sid, feat in val_features:
            values = {"SAT": feat["sat"], "SAT_DBEV_correct": np.concatenate((feat["sat"], feat["correct"]), 1), "SAT_DBEV_mis": np.concatenate((feat["sat"], feat["mis"]), 1)}
            y = occupancy(feat); row = {"scene_id": sid}
            for name in ARMS: row.update({f"{name}_{key}": value for key, value in metrics(score(*classifiers[name], values[name]), y).items()})
            rows.append(row)
        fields = list(rows[0]);
        with (out / "per_scene_semantic_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        arrays = {name: {metric: np.asarray([row[f"{name}_{metric}"] for row in rows]) for metric in ("miou", "building_iou", "building_f1", "pr_auc")} for name in ARMS}
        rng = np.random.default_rng(BOOTSTRAP_SEED); deltas = {"correct_minus_sat_miou": arrays["SAT_DBEV_correct"]["miou"] - arrays["SAT"]["miou"], "correct_minus_mis_miou": arrays["SAT_DBEV_correct"]["miou"] - arrays["SAT_DBEV_mis"]["miou"]}
        boot = {name: bootstrap(value, rng) for name, value in deltas.items()}
        invariants = {"train_val_scene_disjoint": not bool(set(train_ids) & set(val_ids)), "all_scene_prediction_typed_parity": all(x["prediction_before"] == x["prediction_after"] for x in telemetry), "repeat_feature_typed_hash": all(x["feature_hash"] == x["feature_repeat_hash"] for x in telemetry), "no_proposal_write": all(x["before_proposals_absent"] and x["after_proposals_absent"] and x["repeat_before_proposals_absent"] and x["repeat_after_proposals_absent"] for x in telemetry), "model_state_unchanged": all(x["model_state_hash"] == initial_state for x in telemetry), "fitted_artifact_exists": (out / "fitted_artifacts.npz").is_file()}
        correct_iou_gain = float(arrays["SAT_DBEV_correct"]["building_iou"].mean() - arrays["SAT"]["building_iou"].mean())
        passed = all(invariants.values()) and boot["correct_minus_sat_miou"]["ci95_percentile"][0] > 0 and boot["correct_minus_mis_miou"]["ci95_percentile"][0] > 0 and correct_iou_gain >= .002
        summary = {"experiment_id": args.experiment_id, "status": "complete_read_only_frozen_probe", "decision": "T1_SEMANTIC_PASS" if passed else "T1_NO_SEMANTIC_OCCUPANCY_GAIN", "decision_rule": "Appendix T", "arms": {name: {metric: {"mean": float(value.mean()), "std": float(value.std()), "n": len(value)} for metric, value in metrics_.items()} for name, metrics_ in arrays.items()}, "deltas": {name: {"mean": float(value.mean()), "improved": int((value > 0).sum()), "worse": int((value < 0).sum())} for name, value in deltas.items()}, "bootstrap": boot, "correct_building_iou_gain": correct_iou_gain, "invariants": invariants, "interpretation": "T1 measures a derived visible-rooftop occupancy semantic task only; it is not detector or instance-AP evidence."}
        (out / "telemetry.csv").write_text("\n".join(",".join(str(row[key]) for key in telemetry[0]) for row in ([{key: key for key in telemetry[0]}] + telemetry)) + "\n")
        (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True)); manifest["status"] = "complete"; manifest["final_model_state_typed_sha256"] = parameter_hash(model); manifest["summary_sha256"] = sha256_file(out / "summary.json"); (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)); print(json.dumps(summary, indent=2, sort_keys=True))
    except Exception as error:
        (out / "summary.json").write_text(json.dumps({"experiment_id": args.experiment_id, "status": "invalid", "decision": "INVALID", "reason": f"{type(error).__name__}: {error}"}, indent=2)); raise
    finally:
        sys.stdout = previous_stdout; tee.close()


if __name__ == "__main__": main()
