#!/usr/bin/env python3
"""U1: frozen Depth-Anything-V2 relative-depth occupancy probe (Appendix U)."""
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
import torch.nn.functional as F
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from torchvision.transforms import Compose

from probe_fusion_boundary_confirmation import (
    FIT_SCENES, ROOT, TRAIN_RATIO, TRAIN_SEED, Tee, data_hash, dual_predict_digest,
    feature_digest, frozen_features, git_value, mapping_hash, parameter_hash,
    sha256_bytes, sha256_file, typed_hash,
)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "legacy_baseline" / "portable_sam2_fusion_new" / "depth_anything_v2"))
from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.util.transform import NormalizeImage, PrepareForNet, Resize
from mmengine.config import Config
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmdet.registry import MODELS
from mmengine.registry import MODELS as MMENGINE_MODELS
from data.ue_dual_stream_dataset import UEDualStreamDataset
import mmdet.models  # noqa: F401
import rsprompter  # noqa: F401

if "DetDataPreprocessor" not in MMENGINE_MODELS:
    MMENGINE_MODELS.register_module(name="DetDataPreprocessor", module=DetDataPreprocessor)

SAMPLE_PER_CLASS, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, SAMPLE_SEED = 750, 10_000, 20260935, 20260935
ARMS = ("SAT", "SAT_RGB", "SAT_RGB_MDE_correct", "SAT_RGB_MDE_mis")
DEPTH_CKPT_SHA256 = "715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--single-ckpt", type=Path, required=True)
    p.add_argument("--dual-ckpt", type=Path, required=True)
    p.add_argument("--depth-ckpt", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/ue_dual_stream_p2p3_sparse.py"))
    p.add_argument("--data-root", type=Path, default=Path("/data1/wangcheng/dataset/ue_coco_with_uav"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--experiment-id", default="MHBEV-FUS-20260921-U1")
    return p.parse_args()


def occupancy(feat: dict[str, np.ndarray]) -> np.ndarray:
    return feat["occ"].reshape(-1).astype(np.uint8)


def fit_arm(x: np.ndarray, y: np.ndarray) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    clf = LogisticRegression(C=.1, penalty="l2", solver="lbfgs", max_iter=2000, random_state=SAMPLE_SEED, n_jobs=1)
    clf.fit(scaler.fit_transform(x), y)
    return scaler, clf


def score(pair: tuple[StandardScaler, LogisticRegression], x: np.ndarray) -> np.ndarray:
    return pair[1].predict_proba(pair[0].transform(x))[:, 1]


def metrics(probability: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pred, truth = probability >= .5, y.astype(bool)
    tp = int(np.logical_and(pred, truth).sum()); fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum()); tn = int(np.logical_and(~pred, ~truth).sum())
    building_iou, background_iou = tp / max(tp + fp + fn, 1), tn / max(tn + fp + fn, 1)
    return {"miou": (building_iou + background_iou) / 2, "building_iou": building_iou,
            "building_f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "pr_auc": float(average_precision_score(y, probability)), "prevalence": float(y.mean())}


def bootstrap(delta: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    values = delta[rng.integers(0, len(delta), size=(BOOTSTRAP_SAMPLES, len(delta)))].mean(axis=1)
    return {"mean": float(delta.mean()), "ci95_percentile": [float(v) for v in np.percentile(values, [2.5, 97.5])],
            "bootstrap_samples": BOOTSTRAP_SAMPLES}


def make_depth_model(path: Path, device: torch.device) -> DepthAnythingV2:
    if sha256_file(path) != DEPTH_CKPT_SHA256:
        raise RuntimeError("U1 requires the pre-registered official DAv2-Small checkpoint SHA256")
    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    loaded = model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError(f"DAv2 checkpoint contract {loaded}")
    return model.to(device).eval()


def infer_official_depth(model: DepthAnythingV2, raw_bgr: np.ndarray, device: torch.device) -> np.ndarray:
    """Official DAv2 preprocessing, with the tensor placed on the explicit U1 device."""
    if raw_bgr.shape != (512, 512, 3) or raw_bgr.dtype != np.uint8:
        raise RuntimeError(f"U1 requires uint8 512x512 BGR UAV input, got {raw_bgr.shape}/{raw_bgr.dtype}")
    transform = Compose([Resize(width=518, height=518, resize_target=False, keep_aspect_ratio=True,
                                ensure_multiple_of=14, resize_method="lower_bound", image_interpolation_method=cv2.INTER_CUBIC),
                         NormalizeImage(mean=[.485, .456, .406], std=[.229, .224, .225]), PrepareForNet()])
    image = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB) / 255.0
    image = torch.from_numpy(transform({"image": image})["image"]).unsqueeze(0).to(device)
    if tuple(image.shape) != (1, 3, 518, 518):
        raise RuntimeError(f"official U1 transform contract failed: {tuple(image.shape)}")
    depth = model(image)
    depth = F.interpolate(depth[:, None], (512, 512), mode="bilinear", align_corners=True)[0, 0]
    if tuple(depth.shape) != (512, 512) or not torch.isfinite(depth).all():
        raise RuntimeError("DAv2 output contract failed")
    return depth.cpu().numpy().astype(np.float32)


def project_depth(depth: torch.Tensor, points_xyz: np.ndarray, intr: torch.Tensor, extr: torch.Tensor, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Sample normalized per-view MDE at the same z=0 world lattice as direct DBEV."""
    xyz = torch.from_numpy(points_xyz).to(device=device, dtype=torch.float32)
    points = torch.cat((xyz, torch.ones((len(xyz), 1), device=device)), dim=1)
    total, count = torch.zeros(len(points), device=device), torch.zeros(len(points), device=device)
    with torch.autocast(device_type=device.type, enabled=False):
        for view in range(5):
            camera = points @ extr[view].T
            z = camera[:, 2]; pixel = camera[:, :3] @ intr[view].T
            uv = pixel[:, :2] / pixel[:, 2:3].clamp_min(1e-6)
            valid = (z > 1e-4) & (uv[:, 0] >= 0) & (uv[:, 0] < 512) & (uv[:, 1] >= 0) & (uv[:, 1] < 512)
            grid = (uv / 512.0) * 2.0 - 1.0; grid[~valid] = 0
            value = F.grid_sample(depth[view:view + 1].float(), grid[None, :, None, :], mode="bilinear", padding_mode="zeros", align_corners=False)[0, 0, :, 0]
            total += value * valid.float(); count += valid.float()
    return (total / count.clamp_min(1)).cpu().numpy().astype(np.float32), count.cpu().numpy().astype(np.int32)


def depth_features(model: DepthAnythingV2, item: dict[str, Any], base: dict[str, np.ndarray], cache_file: Path, device: torch.device, force: bool = False) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if cache_file.is_file() and not force:
        cached = np.load(cache_file, allow_pickle=False)
        return cached["correct"], cached["mis"], {"cache": "hit", "nonconstant_views": int(cached["nonconstant_views"][0]), "depth_sha256": str(cached["depth_sha256"][0])}
    raw = item["drone_images"].detach().cpu().numpy()
    if raw.shape != (5, 3, 512, 512):
        raise RuntimeError(f"U1 requires five 3x512x512 UAV tensors, got {raw.shape}")
    maps, nonconstant = [], 0
    for image in raw:
        rgb = np.clip(np.transpose(image, (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)
        bgr = rgb[:, :, ::-1].copy()
        depth = infer_official_depth(model, bgr, device)
        lo, hi = np.quantile(depth, [.01, .99])
        if not np.isfinite(depth).all() or not hi > lo:
            depth = np.zeros_like(depth); nonconstant += 0
        else:
            depth = np.clip((depth - lo) / (hi - lo), 0, 1); nonconstant += 1
        maps.append(depth)
    stack = np.stack(maps)[:, None]
    intr, extr = item["intrinsics"].to(device).float(), item["extrinsics"].to(device).float()
    depth_t = torch.from_numpy(stack).to(device)
    if tuple(depth_t.shape) != (5, 1, 512, 512): raise RuntimeError("U1 normalized depth shape contract failed")
    correct, count_c = project_depth(depth_t, base["world"], intr, extr, device)
    mis, count_m = project_depth(depth_t, base["world_mis"], intr, extr, device)
    if not np.array_equal(count_c, base["correct_count"]) or not np.array_equal(count_m, base["mis_count"]):
        raise RuntimeError("MDE and RGB DBEV projection valid masks disagree")
    digest = typed_hash(stack)
    np.savez_compressed(cache_file, correct=correct, mis=mis, nonconstant_views=np.asarray([nonconstant], dtype=np.int16), depth_sha256=np.asarray([digest]))
    return correct, mis, {"cache": "miss", "nonconstant_views": nonconstant, "depth_sha256": digest}


def main() -> None:
    args = parse_args(); out = args.output_dir.resolve(); cache = out / "mde_p3_cache"
    if out.exists() and any(out.iterdir()): raise RuntimeError(f"refuse to overwrite existing artifact: {out}")
    out.mkdir(parents=True); cache.mkdir()
    tee, previous = Tee(sys.stdout, out / "stdout.log"), sys.stdout; sys.stdout = tee
    try:
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available(): raise RuntimeError("U1 requires CUDA")
        for path in (args.single_ckpt, args.dual_ckpt, args.depth_ckpt, args.config, args.data_root / "scene_centers.json"):
            if not path.is_file(): raise FileNotFoundError(path)
        cfg = Config.fromfile(str(args.config)); model = MODELS.build(cfg.model)
        dual = torch.load(args.dual_ckpt, map_location="cpu", weights_only=False); dual = dual.get("model", dual.get("state_dict", dual))
        dl = model.load_state_dict(dual, strict=False)
        if dl.missing_keys or dl.unexpected_keys: raise RuntimeError(f"dual contract {dl}")
        single = torch.load(args.single_ckpt, map_location="cpu", weights_only=False); single = single.get("model", single.get("state_dict", single))
        sl = model.load_state_dict(single, strict=False); allowed = ("drone_encoder.", "geometry_bev.", "p2p3_gate.", "embedding_gate.")
        if [x for x in sl.missing_keys if not x.startswith(allowed)] or sl.unexpected_keys: raise RuntimeError(f"single overlay contract {sl}")
        model = model.to(device).eval(); initial_state = parameter_hash(model); mde = make_depth_model(args.depth_ckpt, device); initial_mde_state = parameter_hash(mde)
        if mde.training: raise RuntimeError("DAv2 eval contract failed")
        train = UEDualStreamDataset(args.data_root, "train", image_size=(1024, 1024), train_subset=TRAIN_RATIO, seed=TRAIN_SEED, load_uav=True)
        val = UEDualStreamDataset(args.data_root, "val", image_size=(1024, 1024), load_uav=True)
        if len(train) != 128 or len(val) != 259: raise RuntimeError("fixed dataset cardinality failed")
        fit_records = [(i, sid) for i, sid in enumerate(train.ids) if len(train[i]["gt_masks"]) > 0][:FIT_SCENES]
        if len(fit_records) != FIT_SCENES: raise RuntimeError("insufficient nonempty fit scenes")
        train_ids, val_ids = [sid for _, sid in fit_records], list(val.ids)
        manifest = {"experiment_id": args.experiment_id, "status": "running_read_only_frozen_probe", "argv": sys.argv, "cwd": str(Path.cwd()), "python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda, "git_commit": git_value(["git", "rev-parse", "HEAD"]), "git_dirty": git_value(["git", "status", "--porcelain"]), "contract": {"fit_scenes": FIT_SCENES, "val_scenes": 259, "sample_per_class": SAMPLE_PER_CLASS, "C": .1, "threshold": .5, "projection": "T1 z0/360m/W2C/5-view; DA-v2 per-view 1p-99p normalized scalar"}, "inputs": {"single_checkpoint_sha256": sha256_file(args.single_ckpt), "dual_checkpoint_sha256": sha256_file(args.dual_ckpt), "depth_checkpoint_sha256": sha256_file(args.depth_ckpt), "config_sha256": sha256_file(args.config), "scene_centers_sha256": sha256_file(args.data_root / "scene_centers.json"), "resolved_config_sha256": sha256_bytes(cfg.pretty_text.encode()), "train_data_sha256": data_hash(args.data_root, train_ids, "train"), "val_data_sha256": data_hash(args.data_root, val_ids, "val")}, "train_ids_fit_80": train_ids, "val_ids": val_ids, "initial_model_state_typed_sha256": initial_state, "initial_mde_state_typed_sha256": initial_mde_state}
        (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        fit = {name: [] for name in ARMS}; labels, telemetry, validation = [], [], []
        def scan(ds, records, split: str, collect: bool) -> None:
            for progress, (index, sid) in enumerate(records):
                item = ds[index]; before, pb = dual_predict_digest(model, item, device)
                base = frozen_features(model, item, device); dc, dm, meta = depth_features(mde, item, base, cache / f"{split}_{sid}.npz", device)
                repeat = frozen_features(model, item, device); dc2, dm2, meta2 = depth_features(mde, item, repeat, cache / f"{split}_{sid}.npz", device, force=(progress == 0))
                base_hash = feature_digest(base); repeat_hash = feature_digest(repeat)
                if base_hash != repeat_hash or typed_hash(dc) != typed_hash(dc2) or typed_hash(dm) != typed_hash(dm2) or parameter_hash(model) != initial_state or model.training or mde.training:
                    raise RuntimeError(f"{sid}: frozen invariant drift")
                after, pa = dual_predict_digest(model, item, device)
                telemetry.append({"split": split, "scene_id": sid, "prediction_before": mapping_hash(before), "prediction_after": mapping_hash(after), "feature_hash": mapping_hash({**base_hash, "mde_correct": typed_hash(dc), "mde_mis": typed_hash(dm)}), "feature_repeat_hash": mapping_hash({**repeat_hash, "mde_correct": typed_hash(dc2), "mde_mis": typed_hash(dm2)}), "model_state_hash": initial_state, "before_proposals_absent": pb["before_proposals_absent"], "after_proposals_absent": pb["after_proposals_absent"], "repeat_before_proposals_absent": pa["before_proposals_absent"], "repeat_after_proposals_absent": pa["after_proposals_absent"], "mde_cache_first": meta["cache"], "mde_cache_repeat": meta2["cache"], "mde_true_repeat": progress == 0, "mde_nonconstant_views": meta["nonconstant_views"], "mde_depth_sha256": meta["depth_sha256"]})
                if before != after: raise RuntimeError(f"{sid}: standard prediction parity failed")
                values = {"SAT": base["sat"], "SAT_RGB": np.concatenate((base["sat"], base["correct"]), 1), "SAT_RGB_MDE_correct": np.concatenate((base["sat"], base["correct"], dc[:, None]), 1), "SAT_RGB_MDE_mis": np.concatenate((base["sat"], base["mis"], dm[:, None]), 1)}
                y = occupancy(base)
                if collect:
                    rng = np.random.default_rng(SAMPLE_SEED + index); pos, neg = np.flatnonzero(y), np.flatnonzero(~y.astype(bool)); n = min(SAMPLE_PER_CLASS, len(pos), len(neg))
                    if not n: raise RuntimeError(f"{sid}: occupancy class missing")
                    chosen = np.concatenate((rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)))
                    for name in ARMS: fit[name].append(values[name][chosen])
                    labels.append(y[chosen]); telemetry[-1]["fit_per_class"] = n
                else: validation.append((sid, y, values))
                if (progress + 1) % 10 == 0 or progress + 1 == len(records): print(f"{split}: {progress + 1}/{len(records)}", flush=True)
        with torch.inference_mode(): scan(train, fit_records, "train", True); scan(val, list(enumerate(val.ids)), "val", False)
        yfit = np.concatenate(labels); classifiers = {name: fit_arm(np.concatenate(value).astype(np.float32), yfit) for name, value in fit.items()}
        np.savez_compressed(out / "fitted_artifacts.npz", train_label=yfit, **{f"{n}_mean": p[0].mean_ for n, p in classifiers.items()}, **{f"{n}_scale": p[0].scale_ for n, p in classifiers.items()}, **{f"{n}_coef": p[1].coef_ for n, p in classifiers.items()}, **{f"{n}_intercept": p[1].intercept_ for n, p in classifiers.items()})
        rows = []
        for sid, y, values in validation:
            row = {"scene_id": sid}
            for name in ARMS: row.update({f"{name}_{key}": value for key, value in metrics(score(classifiers[name], values[name]), y).items()})
            rows.append(row)
        with (out / "per_scene_semantic_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        arrays = {name: {metric: np.asarray([row[f"{name}_{metric}"] for row in rows]) for metric in ("miou", "building_iou", "building_f1", "pr_auc")} for name in ARMS}
        deltas = {"mde_minus_rgb_miou": arrays["SAT_RGB_MDE_correct"]["miou"] - arrays["SAT_RGB"]["miou"], "mde_minus_sat_miou": arrays["SAT_RGB_MDE_correct"]["miou"] - arrays["SAT"]["miou"], "mde_minus_mis_miou": arrays["SAT_RGB_MDE_correct"]["miou"] - arrays["SAT_RGB_MDE_mis"]["miou"]}
        rng = np.random.default_rng(BOOTSTRAP_SEED); boot = {name: bootstrap(value, rng) for name, value in deltas.items()}
        invariants = {"train_val_scene_disjoint": not bool(set(train_ids) & set(val_ids)), "all_scene_prediction_typed_parity": all(x["prediction_before"] == x["prediction_after"] for x in telemetry), "repeat_feature_typed_hash": all(x["feature_hash"] == x["feature_repeat_hash"] for x in telemetry), "no_proposal_write": all(x["before_proposals_absent"] and x["after_proposals_absent"] and x["repeat_before_proposals_absent"] and x["repeat_after_proposals_absent"] for x in telemetry), "model_state_unchanged": all(x["model_state_hash"] == initial_state for x in telemetry), "mde_state_unchanged": parameter_hash(mde) == initial_mde_state, "mde_eval_mode": not mde.training, "mde_true_repeat_per_split": sum(bool(x["mde_true_repeat"]) for x in telemetry if x["split"] == "train") == 1 and sum(bool(x["mde_true_repeat"]) for x in telemetry if x["split"] == "val") == 1, "all_mde_views_nonconstant": all(x["mde_nonconstant_views"] == 5 for x in telemetry), "fitted_artifact_exists": (out / "fitted_artifacts.npz").is_file()}
        iou_gain = float(arrays["SAT_RGB_MDE_correct"]["building_iou"].mean() - arrays["SAT"]["building_iou"].mean())
        passed = all(invariants.values()) and all(boot[n]["ci95_percentile"][0] > 0 for n in deltas) and iou_gain >= .002
        summary = {"experiment_id": args.experiment_id, "status": "complete_read_only_frozen_probe", "decision": "U1_MDE_SEMANTIC_PASS" if passed else "U1_NO_MDE_SEMANTIC_GAIN", "decision_rule": "Appendix U", "arms": {name: {metric: {"mean": float(value.mean()), "std": float(value.std()), "n": len(value)} for metric, value in metric_values.items()} for name, metric_values in arrays.items()}, "deltas": {name: {"mean": float(value.mean()), "improved": int((value > 0).sum()), "worse": int((value < 0).sum())} for name, value in deltas.items()}, "bootstrap": boot, "mde_correct_building_iou_gain_over_sat": iou_gain, "invariants": invariants, "interpretation": "U1 measures frozen relative-depth representation on derived visible-rooftop occupancy only; it is neither metric depth accuracy nor detector/instance-AP evidence."}
        with (out / "telemetry.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(telemetry[0])); writer.writeheader(); writer.writerows(telemetry)
        (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True)); manifest.update({"status": "complete", "final_model_state_typed_sha256": parameter_hash(model), "final_mde_state_typed_sha256": parameter_hash(mde), "summary_sha256": sha256_file(out / "summary.json")}); (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)); print(json.dumps(summary, indent=2, sort_keys=True))
    except Exception as error:
        (out / "summary.json").write_text(json.dumps({"experiment_id": args.experiment_id, "status": "invalid", "decision": "INVALID", "reason": f"{type(error).__name__}: {error}"}, indent=2)); raise
    finally:
        sys.stdout = previous; tee.close()


if __name__ == "__main__":
    main()
