#!/usr/bin/env python3
"""E5: pre-registered, read-only confirmation of DBEV boundary information.

This is deliberately a *frozen* feature probe.  It never calls loss/backward,
never changes a detector parameter, and never supplies or writes proposals.  See
EXPERIMENT_GOVERNANCE.md Appendix K for the registered decision rule.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

# Keep numerical linear algebra reproducible across the repeat-feature check.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("EXPLICIT_PROMPT_MODE", "points_box_dense")
os.environ.setdefault("P2_BOUNDARY_REFINER_ENABLED", "1")
os.environ.setdefault("SAM_IMAGE_EMBED_STRIDE", "16")
os.environ.setdefault("SHAPE_DENSE_TRANSFORM", "raw_logits")
os.environ.setdefault("SHAPE_DENSE_DETACH", "1")
os.environ.setdefault("FINAL_MASK_COORDINATE_MODE", "roi_local")
os.environ.setdefault("SAM2_CKPT", "/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt")
os.environ.setdefault("UE_DUAL_INJECT_EMBEDDING", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import mmdet.models  # noqa: F401
import rsprompter  # noqa: F401
from data.ue_dual_stream_dataset import UEDualStreamDataset
from train.train_rsprompter_fusion import _attach_dual_stream_metainfo

if "DetDataPreprocessor" not in MMENGINE_MODELS:
    MMENGINE_MODELS.register_module(name="DetDataPreprocessor", module=DetDataPreprocessor)

TRAIN_RATIO = 0.1797752808988764
TRAIN_SEED = 44
FIT_SCENES = 80
SAMPLE_PER_SCENE = 1500
PCA_DIM = 512
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260926
SAMPLE_SEED_BASE = 20260925
ARMS = ("SAT", "SAT_DBEV_correct", "SAT_DBEV_mis", "SAT_PCA512_correct", "SAT_PCA512_mis")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def typed_hash(value: Any) -> str:
    """Hash dtype, shape and bytes; do not silently equate dtype-coerced arrays."""
    if hasattr(value, "to_ndarray"):
        value = value.to_ndarray()
    elif hasattr(value, "masks"):
        value = value.masks
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous().numpy()
    array = np.ascontiguousarray(value)
    return sha256_bytes(
        f"dtype={array.dtype};shape={tuple(array.shape)};".encode("utf-8") + array.tobytes()
    )


def mapping_hash(values: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(values), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(payload)


def parameter_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        arr = value.detach().cpu().contiguous().numpy()
        h.update(f"{name};dtype={arr.dtype};shape={tuple(arr.shape)};".encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


def digest_prediction(outputs) -> dict[str, str]:
    instance = outputs[0].pred_instances
    return {name: typed_hash(getattr(instance, name)) for name in ("bboxes", "scores", "labels", "masks")}


class Tee:
    def __init__(self, stream, path: Path):
        self.stream, self.file = stream, path.open("w", encoding="utf-8", buffering=1)

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--single-ckpt", type=Path,
                   default=Path("work_dirs/ue/ue_single_c5v2_control_tr128_va259_s44_r1/best_model_epoch86.pth"))
    p.add_argument("--dual-ckpt", type=Path,
                   default=Path("work_dirs/ue/ue_dual_c5v2_p2p3_emb_tr128_va259_s44_r1/best_model_epoch85.pth"))
    p.add_argument("--config", type=Path, default=Path("configs/ue_dual_stream_p2p3_sparse.py"))
    p.add_argument("--data-root", type=Path, default=Path("/data1/wangcheng/dataset/ue_coco_with_uav"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--experiment-id", default="MHBEV-FUS-20260919-E5")
    return p.parse_args()


def attach_dual_sample(item: Mapping[str, Any], device: torch.device) -> DetDataSample:
    """Use the trainer's exact metadata key contract for the standard predict path."""
    sample = DetDataSample()
    sample.set_metainfo(dict(item["img_metas"]))
    batch = {
        "drone_images": item["drone_images"][None],
        "intrinsics": item["intrinsics"][None],
        "extrinsics": item["extrinsics"][None],
        "scene_centers": item["scene_center"][None],
    }
    _attach_dual_stream_metainfo(sample, batch, 0)
    return sample.to(device)


def dual_predict_digest(model, item: Mapping[str, Any], device: torch.device) -> tuple[dict[str, str], dict[str, bool]]:
    """Normal deployed prediction with fail-closed no-proposal-write checks."""
    sample = attach_dual_sample(item, device)
    data = model.data_preprocessor({"inputs": [item["img"].to(device)], "data_samples": [sample]}, training=False)
    live = data["data_samples"][0]
    before_absent = live.get("proposals", None) is None
    if not before_absent:
        raise RuntimeError("E5 requires no pre-supplied proposals")
    outputs = model.predict(data["inputs"], data["data_samples"], rescale=False)
    after_absent = live.get("proposals", None) is None
    if not after_absent:
        raise RuntimeError("model.predict wrote proposals into the E5 data sample")
    return digest_prediction(outputs), {"before_proposals_absent": before_absent, "after_proposals_absent": after_absent}


def world_grid(h: int, w: int, center_xy_cm: torch.Tensor, device: torch.device) -> torch.Tensor:
    """P3 lattice normalized over the audited full 360m tile, z=0 in UE cm."""
    with torch.autocast(device_type=device.type, enabled=False):
        u = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * (512.0 / w)
        v = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * (512.0 / h)
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        x = -(vv / 512.0 - 0.5) * 360.0 * 100.0 + center_xy_cm[0].float()
        y = (uu / 512.0 - 0.5) * 360.0 * 100.0 + center_xy_cm[1].float()
        return torch.stack((x, y, torch.zeros_like(x), torch.ones_like(x)), dim=-1).reshape(-1, 4)


def rotate_points_90(points: torch.Tensor, center_xy_cm: torch.Tensor) -> torch.Tensor:
    out = points.clone()
    dx, dy = points[:, 0] - center_xy_cm[0], points[:, 1] - center_xy_cm[1]
    out[:, 0] = center_xy_cm[0] - dy
    out[:, 1] = center_xy_cm[1] + dx
    return out


def signed_boundary_label(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    occ = np.zeros((128, 128), dtype=np.uint8)
    if getattr(masks, "ndim", 0) == 3:
        for mask in masks:
            occ = np.maximum(occ, cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST))
    if not occ.any() or occ.all():
        raise RuntimeError("E5 requires a non-degenerate union mask for every selected scene")
    outside = cv2.distanceTransform(1 - occ, cv2.DIST_L2, 3)
    inside = cv2.distanceTransform(occ, cv2.DIST_L2, 3)
    return ((outside - inside).reshape(-1).astype(np.float32) / 8.0), occ


def has_non_degenerate_boundary_label(masks: np.ndarray) -> bool:
    """The registered deterministic fit-scene eligibility predicate."""
    occ = np.zeros((128, 128), dtype=np.uint8)
    if getattr(masks, "ndim", 0) == 3:
        for mask in masks:
            occ = np.maximum(occ, cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST))
    return bool(occ.any()) and not bool(occ.all())


@torch.no_grad()
def frozen_features(model, item: Mapping[str, Any], device: torch.device) -> dict[str, np.ndarray]:
    """Read SAT P3 and direct UAV-FPN BEV; no detector predict/proposal is involved."""
    # Preprocess satellite input exactly as deployed.  Fresh data sample prevents feature
    # extraction from sharing the prediction samples used for no-write parity checks.
    sample = attach_dual_sample(item, device)
    data = model.data_preprocessor({"inputs": [item["img"].to(device)], "data_samples": [sample]}, training=False)
    if data["data_samples"][0].get("proposals", None) is not None:
        raise RuntimeError("feature extraction received proposals")
    inputs = data["inputs"]
    if isinstance(inputs, (tuple, list)):
        inputs = inputs[0]
    satellite_outputs = model.backbone(inputs)
    pyramids = model.neck(satellite_outputs[1])
    sat = pyramids[1][0].detach().float()
    if tuple(sat.shape) != (512, 128, 128):
        raise RuntimeError(f"unexpected P3 contract: {tuple(sat.shape)}")

    drone = item["drone_images"][None].to(device)
    drone = (drone - drone.new_tensor([0.485, 0.456, 0.406])[None, None, :, None, None]) / \
            drone.new_tensor([0.229, 0.224, 0.225])[None, None, :, None, None]
    _, fpn = model.drone_encoder(drone.reshape(5, *drone.shape[2:]))
    fpn = tuple(sorted(tuple(fpn[-4:]), key=lambda x: x.shape[-2] * x.shape[-1], reverse=True))
    if len(fpn) != 4 or any(x.shape[0] != 5 or x.shape[1] != 256 for x in fpn):
        raise RuntimeError("unexpected frozen UAV FPN contract")

    intr = item["intrinsics"].to(device).float()
    extr = item["extrinsics"].to(device).float()
    center = item["scene_center"].to(device).float()
    if tuple(intr.shape) != (5, 3, 3) or tuple(extr.shape) != (5, 4, 4):
        raise RuntimeError("E5 requires five-view K/E shapes [5,3,3]/[5,4,4]")
    points = world_grid(128, 128, center, device)

    def sample(points_cm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        per_level, counts = [], None
        with torch.autocast(device_type=device.type, enabled=False):
            for feat in fpn:
                total = torch.zeros(256, len(points_cm), device=device, dtype=torch.float32)
                level_count = torch.zeros(len(points_cm), device=device, dtype=torch.float32)
                for view in range(5):
                    camera = points_cm @ extr[view].T
                    z = camera[:, 2]
                    pixel = camera[:, :3] @ intr[view].T
                    uv = pixel[:, :2] / pixel[:, 2:3].clamp_min(1e-6)
                    valid = ((z > 1e-4) & (uv[:, 0] >= 0) & (uv[:, 0] < 512) &
                             (uv[:, 1] >= 0) & (uv[:, 1] < 512))
                    grid = (uv / 512.0) * 2.0 - 1.0
                    grid[~valid] = 0.0
                    values = F.grid_sample(feat[view:view + 1].float(), grid[None, :, None, :],
                                           mode="bilinear", padding_mode="zeros", align_corners=False)[0, :, :, 0]
                    total += values * valid[None].float()
                    level_count += valid.float()
                per_level.append(total / level_count.clamp_min(1.0))
                # Validity is geometric and level-independent, but check it rather than assume it.
                if counts is None:
                    counts = level_count.to(torch.int32)
                elif not torch.equal(counts, level_count.to(torch.int32)):
                    raise RuntimeError("UAV FPN levels disagree on geometric valid-view counts")
        return torch.cat(per_level, dim=0), counts

    correct, count_correct = sample(points)
    mis_points = rotate_points_90(points, center)
    mis, count_mis = sample(mis_points)
    y, occ = signed_boundary_label(item["gt_masks"])
    return {
        "sat": sat.cpu().numpy().transpose(1, 2, 0).reshape(-1, 512).astype(np.float32),
        "correct": correct.cpu().numpy().T.astype(np.float32),
        "mis": mis.cpu().numpy().T.astype(np.float32),
        "label": y,
        "occ": occ,
        "correct_count": count_correct.cpu().numpy(),
        "mis_count": count_mis.cpu().numpy(),
        "world": points[:, :3].cpu().numpy().astype(np.float32),
        "world_mis": mis_points[:, :3].cpu().numpy().astype(np.float32),
    }


def feature_digest(feat: Mapping[str, np.ndarray]) -> dict[str, str]:
    return {name: typed_hash(feat[name]) for name in ("sat", "correct", "mis", "label", "correct_count", "mis_count", "world", "world_mis")}


def data_hash(root: Path, scene_ids: list[str], split: str) -> str:
    """Content fingerprint of every fixed input file used by E5 for one split."""
    h = hashlib.sha256()
    for sid in scene_ids:
        satellite = root / split / "satellite"
        meta = root / split / "drone" / sid / "ue_render_metadata" / "camera_params.json"
        files = [satellite / f"{sid}.png", satellite / f"{sid}.json", meta]
        camera = json.loads(meta.read_text())
        drone_dir = root / split / "drone" / sid
        for name in camera["image_names"]:
            image = drone_dir / name
            if not image.is_file():
                image = drone_dir / "ue_render_metadata" / name
            files.append(image)
        for path in files:
            if not path.is_file():
                raise FileNotFoundError(path)
            h.update(str(path.relative_to(root)).encode("utf-8"))
            h.update(sha256_file(path).encode("ascii"))
    return h.hexdigest()


@dataclass
class Arm:
    name: str
    scaler: StandardScaler
    ridge: Ridge

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.ridge.predict(self.scaler.transform(x)).astype(np.float64)


def fit_arm(name: str, x: np.ndarray, y: np.ndarray, alpha: float) -> Arm:
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    # scipy LSQR has no random initialization; the one-thread environment above makes
    # its numerical path deterministic while avoiding an unbounded dense SVD workspace.
    ridge = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr", tol=1e-10, max_iter=20_000)
    ridge.fit(scaler.fit_transform(x), y)
    return Arm(name=name, scaler=scaler, ridge=ridge)


def pearson(prediction: np.ndarray, target: np.ndarray) -> float:
    if not np.isfinite(prediction).all() or not np.isfinite(target).all() or prediction.std() == 0 or target.std() == 0:
        raise RuntimeError("non-finite or constant per-scene Pearson input")
    return float(np.corrcoef(prediction, target)[0, 1])


def bootstrap(delta: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    draws = rng.integers(0, len(delta), size=(BOOTSTRAP_SAMPLES, len(delta)))
    means = delta[draws].mean(axis=1)
    return {
        "mean": float(delta.mean()), "ci95_percentile": [float(x) for x in np.percentile(means, [2.5, 97.5])],
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
    }


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"refuse to overwrite existing E5 artifact directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    tee = Tee(sys.stdout, out / "stdout.log")
    old_stdout = sys.stdout
    sys.stdout = tee
    try:
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("E5 requires a CUDA device for the frozen dual-stream extraction")
        if args.ridge_alpha <= 0:
            raise ValueError("ridge alpha must be positive")
        for path in (args.single_ckpt, args.dual_ckpt, args.config, args.data_root / "scene_centers.json"):
            if not path.is_file():
                raise FileNotFoundError(path)

        cfg = Config.fromfile(str(args.config))
        model = MODELS.build(cfg.model)
        dual_state = torch.load(args.dual_ckpt, map_location="cpu", weights_only=False)
        dual_state = dual_state.get("model", dual_state.get("state_dict", dual_state))
        dual_load = model.load_state_dict(dual_state, strict=False)
        if dual_load.missing_keys or dual_load.unexpected_keys:
            raise RuntimeError(f"dual checkpoint strict contract failed: {dual_load}")
        single_state = torch.load(args.single_ckpt, map_location="cpu", weights_only=False)
        single_state = single_state.get("model", single_state.get("state_dict", single_state))
        single_load = model.load_state_dict(single_state, strict=False)
        allowed_missing = ("drone_encoder.", "geometry_bev.", "p2p3_gate.", "embedding_gate.")
        bad_missing = [x for x in single_load.missing_keys if not x.startswith(allowed_missing)]
        if bad_missing or single_load.unexpected_keys:
            raise RuntimeError(f"single overlay contract failed: bad_missing={bad_missing}, unexpected={single_load.unexpected_keys}")
        model = model.to(device).eval()
        if model.training:
            raise RuntimeError("model.eval() did not take effect")
        initial_state_hash = parameter_hash(model)

        train = UEDualStreamDataset(args.data_root, "train", image_size=(1024, 1024),
                                    train_subset=TRAIN_RATIO, seed=TRAIN_SEED, load_uav=True)
        val = UEDualStreamDataset(args.data_root, "val", image_size=(1024, 1024), load_uav=True)
        if len(train) != 128 or len(val) != 259:
            raise RuntimeError(f"fixed E5 split cardinality failed: train={len(train)} val={len(val)}")
        fit_records, skipped_train_records = [], []
        for source_index, sid in enumerate(train.ids):
            if has_non_degenerate_boundary_label(train[source_index]["gt_masks"]):
                fit_records.append((source_index, sid))
                if len(fit_records) == FIT_SCENES:
                    break
            else:
                skipped_train_records.append({"scene_id": sid, "source_index": source_index,
                                              "reason": "degenerate_p3_union_mask"})
        train_ids = [sid for _, sid in fit_records]
        val_records = list(enumerate(val.ids))
        val_ids = [sid for _, sid in val_records]
        if len(train_ids) != FIT_SCENES or set(train_ids) & set(val_ids):
            raise RuntimeError("E5 train/val ID contract failed")

        # Exact launch/provenance is written before any expensive scan.
        manifest = {
            "experiment_id": args.experiment_id, "status": "running_read_only_frozen_probe",
            "argv": sys.argv, "cwd": str(Path.cwd()), "python": sys.version, "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device),
            "git_commit": git_value(["git", "rev-parse", "HEAD"]), "git_dirty": git_value(["git", "status", "--porcelain"]),
            "seeds": {"train_subset": TRAIN_SEED, "cell_sample_base": SAMPLE_SEED_BASE, "pca": SAMPLE_SEED_BASE, "bootstrap": BOOTSTRAP_SEED},
            "contract": {"fit_scenes": FIT_SCENES, "fit_cells_per_scene": SAMPLE_PER_SCENE, "val_scenes": 259,
                         "p3": [512, 128, 128], "dbev": 1024, "pca_components": PCA_DIM, "ridge_alpha": args.ridge_alpha,
                         "projection": "z0/360m/W2C-fp32-autocast-disabled/5-view/4-FPN/bilinear/valid-view-average"},
            "inputs": {"single_checkpoint": str(args.single_ckpt.resolve()), "dual_checkpoint": str(args.dual_ckpt.resolve()),
                       "config": str(args.config.resolve()), "data_root": str(args.data_root.resolve()),
                       "single_checkpoint_sha256": sha256_file(args.single_ckpt), "dual_checkpoint_sha256": sha256_file(args.dual_ckpt),
                       "config_sha256": sha256_file(args.config), "resolved_config_sha256": sha256_bytes(cfg.pretty_text.encode("utf-8")),
                       "scene_centers_sha256": sha256_file(args.data_root / "scene_centers.json"),
                       "dual_load": {"missing": list(dual_load.missing_keys), "unexpected": list(dual_load.unexpected_keys)},
                       "single_overlay_load": {"missing": list(single_load.missing_keys), "unexpected": list(single_load.unexpected_keys)}},
            "train_subset_ids_all_128": list(train.ids), "train_ids_fit_80": train_ids,
            "train_fit_source_indices": [index for index, _ in fit_records],
            "train_skipped_degenerate_boundary": skipped_train_records, "val_ids": val_ids,
            "initial_model_state_typed_sha256": initial_state_hash,
        }
        (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        print(json.dumps({"E5": args.experiment_id, "train": len(train_ids), "val": len(val_ids), "device": str(device)}, sort_keys=True))
        print("hashing fixed input data ...", flush=True)
        manifest["inputs"]["train_data_sha256"] = data_hash(args.data_root, train_ids, "train")
        manifest["inputs"]["val_data_sha256"] = data_hash(args.data_root, val_ids, "val")
        (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

        # Archive lattice formula and all scene centers; together these exactly reproduce
        # the per-scene W2C world-point mapping without retaining 339 duplicate grids.
        template = world_grid(128, 128, torch.zeros(2, device=device), device).cpu().numpy().astype(np.float32)
        np.savez_compressed(out / "p3_world_lattice_contract.npz", local_world_cm=template,
                            train_scene_ids=np.asarray(train_ids), val_scene_ids=np.asarray(val_ids),
                            train_scene_centers=np.asarray([train[index]["scene_center"].numpy() for index, _ in fit_records], dtype=np.float32),
                            val_scene_centers=np.asarray([val[i]["scene_center"].numpy() for i in range(len(val))], dtype=np.float32))

        fit_sat, fit_correct, fit_mis, fit_y, sample_rows, telemetry = [], [], [], [], [], []
        val_features: list[tuple[str, dict[str, np.ndarray]]] = []

        def scan(dataset, records: list[tuple[int, str]], split: str, collect_fit: bool) -> None:
            for progress, (source_index, sid) in enumerate(records):
                item = dataset[source_index]
                before, proposal_before = dual_predict_digest(model, item, device)
                first = frozen_features(model, item, device)
                second = frozen_features(model, item, device)
                first_hash, second_hash = feature_digest(first), feature_digest(second)
                if first_hash != second_hash:
                    raise RuntimeError(f"{sid}: repeated frozen feature typed hash drift")
                after, proposal_after = dual_predict_digest(model, item, device)
                if before != after:
                    raise RuntimeError(f"{sid}: standard dual prediction parity drift")
                if parameter_hash(model) != initial_state_hash or model.training:
                    raise RuntimeError(f"{sid}: model state/training state drift")
                telemetry.append({
                    "split": split, "scene_id": sid, "prediction_before": mapping_hash(before), "prediction_after": mapping_hash(after),
                    **{f"prediction_before_{name}_typed_sha256": value for name, value in before.items()},
                    **{f"prediction_after_{name}_typed_sha256": value for name, value in after.items()},
                    "feature_hash": mapping_hash(first_hash), "feature_repeat_hash": mapping_hash(second_hash),
                    **{f"feature_{name}_typed_sha256": value for name, value in first_hash.items()},
                    "feature_repeat_equal": True, "model_state_hash": initial_state_hash, "model_training": False,
                    "before_proposals_absent": proposal_before["before_proposals_absent"], "before_after_proposals_absent": proposal_before["after_proposals_absent"],
                    "after_proposals_absent": proposal_after["before_proposals_absent"], "after_after_proposals_absent": proposal_after["after_proposals_absent"],
                    "correct_valid_mean": float(first["correct_count"].mean()), "mis_valid_mean": float(first["mis_count"].mean()),
                    "correct_valid_min": int(first["correct_count"].min()), "mis_valid_min": int(first["mis_count"].min()),
                    "world_hash": first_hash["world"], "world_mis_hash": first_hash["world_mis"], "label_hash": first_hash["label"],
                })
                if collect_fit:
                    rng = np.random.default_rng(SAMPLE_SEED_BASE + source_index)
                    chosen = rng.choice(128 * 128, size=SAMPLE_PER_SCENE, replace=False)
                    fit_sat.append(first["sat"][chosen]); fit_correct.append(first["correct"][chosen]); fit_mis.append(first["mis"][chosen]); fit_y.append(first["label"][chosen])
                    sample_rows.append({"scene_id": sid, "scene_index": source_index, "sample_seed": SAMPLE_SEED_BASE + source_index,
                                        "indices_typed_sha256": typed_hash(chosen.astype(np.int64)), "label_typed_sha256": typed_hash(first["label"][chosen])})
                else:
                    val_features.append((sid, first))
                if (progress + 1) % 10 == 0 or progress + 1 == len(records):
                    print(f"{split}: {progress + 1}/{len(records)} scenes", flush=True)

        with torch.inference_mode():
            scan(train, fit_records, "train", True)
            scan(val, val_records, "val", False)

        x_sat = np.concatenate(fit_sat).astype(np.float32)
        x_correct = np.concatenate(fit_correct).astype(np.float32)
        x_mis = np.concatenate(fit_mis).astype(np.float32)
        y = np.concatenate(fit_y).astype(np.float32)
        if x_sat.shape != (FIT_SCENES * SAMPLE_PER_SCENE, 512) or x_correct.shape != (len(y), 1024) or x_mis.shape != (len(y), 1024):
            raise RuntimeError("E5 fitted feature cardinality/dimension contract failed")
        # Independent, train-only deterministic PCAs.  Each arm has the same output
        # dimension and each transform is fitted only on that arm's train descriptor.
        print("fitting train-only PCA/ridge readouts ...", flush=True)
        pca_correct = PCA(n_components=PCA_DIM, svd_solver="randomized", random_state=SAMPLE_SEED_BASE)
        pca_mis = PCA(n_components=PCA_DIM, svd_solver="randomized", random_state=SAMPLE_SEED_BASE)
        z_correct, z_mis = pca_correct.fit_transform(x_correct), pca_mis.fit_transform(x_mis)
        fit_inputs = {
            "SAT": x_sat,
            "SAT_DBEV_correct": np.concatenate((x_sat, x_correct), axis=1),
            "SAT_DBEV_mis": np.concatenate((x_sat, x_mis), axis=1),
            "SAT_PCA512_correct": np.concatenate((x_sat, z_correct.astype(np.float32)), axis=1),
            "SAT_PCA512_mis": np.concatenate((x_sat, z_mis.astype(np.float32)), axis=1),
        }
        arms = {name: fit_arm(name, fit_inputs[name], y, args.ridge_alpha) for name in ARMS}
        np.savez_compressed(out / "fitted_artifacts.npz", train_label=y,
                            pca_correct_components=pca_correct.components_, pca_correct_mean=pca_correct.mean_,
                            pca_mis_components=pca_mis.components_, pca_mis_mean=pca_mis.mean_,
                            **{f"{name}_scale_mean": arm.scaler.mean_ for name, arm in arms.items()},
                            **{f"{name}_scale_scale": arm.scaler.scale_ for name, arm in arms.items()},
                            **{f"{name}_ridge_coef": np.asarray(arm.ridge.coef_) for name, arm in arms.items()},
                            **{f"{name}_ridge_intercept": np.asarray(arm.ridge.intercept_) for name, arm in arms.items()})
        write_csv(out / "train_samples.csv", sample_rows, ["scene_id", "scene_index", "sample_seed", "indices_typed_sha256", "label_typed_sha256"])

        per_scene = []
        for sid, feature in val_features:
            zc = pca_correct.transform(feature["correct"])
            zm = pca_mis.transform(feature["mis"])
            values = {
                "SAT": feature["sat"], "SAT_DBEV_correct": np.concatenate((feature["sat"], feature["correct"]), axis=1),
                "SAT_DBEV_mis": np.concatenate((feature["sat"], feature["mis"]), axis=1),
                "SAT_PCA512_correct": np.concatenate((feature["sat"], zc.astype(np.float32)), axis=1),
                "SAT_PCA512_mis": np.concatenate((feature["sat"], zm.astype(np.float32)), axis=1),
            }
            row = {"scene_id": sid, **{name: pearson(arms[name].predict(values[name]), feature["label"]) for name in ARMS}}
            per_scene.append(row)
        write_csv(out / "per_scene_boundary_r.csv", per_scene, ["scene_id", *ARMS])
        write_csv(out / "telemetry.csv", telemetry, list(telemetry[0].keys()))

        arrays = {name: np.asarray([row[name] for row in per_scene], dtype=np.float64) for name in ARMS}
        deltas = {
            "correct_minus_sat": arrays["SAT_DBEV_correct"] - arrays["SAT"],
            "correct_minus_mis": arrays["SAT_DBEV_correct"] - arrays["SAT_DBEV_mis"],
            "pca512_correct_minus_mis": arrays["SAT_PCA512_correct"] - arrays["SAT_PCA512_mis"],
        }
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        boot = {name: bootstrap(delta, rng) for name, delta in deltas.items()}
        paired_t = {name: {"statistic": float(stats.ttest_1samp(delta, 0.0).statistic), "p_two_sided": float(stats.ttest_1samp(delta, 0.0).pvalue)} for name, delta in deltas.items()}
        summary_arms = {name: {"mean": float(x.mean()), "std": float(x.std()), "n": int(len(x))} for name, x in arrays.items()}
        invariants = {
            "train_val_scene_disjoint": not bool(set(train_ids) & set(val_ids)), "all_scene_prediction_typed_parity": all(x["prediction_before"] == x["prediction_after"] for x in telemetry),
            "repeat_feature_typed_hash": all(x["feature_hash"] == x["feature_repeat_hash"] for x in telemetry),
            "no_proposal_write": all(x["before_proposals_absent"] and x["before_after_proposals_absent"] and x["after_proposals_absent"] and x["after_after_proposals_absent"] for x in telemetry),
            "model_state_unchanged": all(x["model_state_hash"] == initial_state_hash and not x["model_training"] for x in telemetry),
            "fitted_artifact_exists": (out / "fitted_artifacts.npz").is_file(), "manifest_exists": (out / "run_manifest.json").is_file(),
        }
        ci = {name: values["ci95_percentile"] for name, values in boot.items()}
        passed = all(invariants.values()) and ci["correct_minus_sat"][0] >= .02 and ci["correct_minus_mis"][0] >= .02 and ci["pca512_correct_minus_mis"][0] >= .01
        decision = "E5_PASS" if passed else "E5_NO_MARGINAL_BOUNDARY_SIGNAL"
        summary = {
            "experiment_id": args.experiment_id, "status": "complete_read_only_frozen_probe", "decision": decision,
            "decision_rule": "E5_PASS iff all invariants and CI lower(correct-SAT)>=.02, CI lower(correct-mis)>=.02, CI lower(PCA512 correct-mis)>=.01; else E5_NO_MARGINAL_BOUNDARY_SIGNAL. Any contract/manifest/parity/hash failure raises INVALID.",
            "arms": summary_arms, "deltas": {name: {"mean": float(value.mean()), "std": float(value.std()), "improved": int((value > 0).sum()), "worse": int((value < 0).sum())} for name, value in deltas.items()},
            "bootstrap": boot, "paired_t": paired_t, "invariants": invariants,
            "interpretation": "E5 only tests frozen linear boundary information. It does not report detector AP or authorize implementation/training; PASS only authorizes a separate heat-start proposal.",
        }
        (out / "bootstrap.json").write_text(json.dumps({"seed": BOOTSTRAP_SEED, "metrics": boot}, indent=2, sort_keys=True))
        (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
        manifest["final_model_state_typed_sha256"] = parameter_hash(model)
        manifest["fitted_artifacts_sha256"] = sha256_file(out / "fitted_artifacts.npz")
        manifest["summary_sha256"] = sha256_file(out / "summary.json")
        manifest["status"] = "complete"
        (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        print(json.dumps({"decision": decision, "ci": ci, "invariants": invariants}, indent=2, sort_keys=True), flush=True)
    except Exception as exc:
        # A fail-closed terminal artifact makes a contract break auditable even if the
        # process is launched under screen/tee and the caller only inspects the output dir.
        invalid = {
            "experiment_id": args.experiment_id,
            "status": "invalid",
            "decision": "INVALID",
            "reason": f"{type(exc).__name__}: {exc}",
            "interpretation": "No E5 decision may be made from an invalid run.",
        }
        (out / "summary.json").write_text(json.dumps(invalid, indent=2, sort_keys=True))
        print(json.dumps(invalid, sort_keys=True), flush=True)
        raise
    finally:
        sys.stdout = old_stdout
        tee.close()


if __name__ == "__main__":
    main()
