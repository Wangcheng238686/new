#!/usr/bin/env python
"""A1 frozen gate: calibrated P2 z=0 UAV sampling with detector bypass.

This is deliberately a *probe*, not a trainable dual-stream model.  It proves
that the proposed A1 coordinate path is numerically viable while a passive
hook leaves the published single-stream detector bitwise unchanged.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping

os.environ.setdefault("SAM2_CKPT", "/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config, ConfigDict
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample

from data.ue_dual_stream_dataset import UEDualStreamDataset
from inference.infer_from_checkpoint import (
    _load_checkpoint, _load_model_state, _register_and_build, _resolve_model_config,
    _resolve_sam2_repo, _select_state_dict, _snapshot,
)
from rsprompter.dual_stream import GeometrySparseBEV


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="/data1/wangcheng/dataset/ue_coco_with_uav")
    parser.add_argument("--scenes", type=int, default=16,
                        help="Deterministic prefix used for frozen feature/parity probe.")
    parser.add_argument("--parity-scenes", type=int, default=0,
                        help="0 means all --scenes. The formal gate uses the full 259-scene val set.")
    parser.add_argument("--grid", type=int, default=256, help="A1 P2 target resolution.")
    parser.add_argument("--feature-points", type=int, default=2048,
                        help="Uniform grid subset used only for descriptive affinity telemetry.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sam2-ckpt", default=os.environ["SAM2_CKPT"])
    parser.add_argument("--experiment-id", default="MHBEV-ROI-20260914-A1-Frozen-R2")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def build_sample(item: Mapping[str, Any], device: torch.device) -> DetDataSample:
    sample = DetDataSample()
    sample.set_metainfo(dict(item["img_metas"]))
    return sample.to(device)


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def instance_digest(outputs) -> Dict[str, str]:
    """Hash every prediction field used by bbox and mask evaluation."""
    result: Dict[str, str] = {}
    instance = outputs[0].pred_instances
    for name in ("bboxes", "scores", "labels", "masks"):
        value = getattr(instance, name, None)
        if hasattr(value, "to_ndarray"):
            value = value.to_ndarray()
        elif hasattr(value, "masks"):
            value = value.masks
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().contiguous().numpy()
        result[name] = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
    return result


def predict(model, item: Mapping[str, Any], device: torch.device):
    sample = build_sample(item, device)
    data = model.data_preprocessor(
        {"inputs": [item["img"].to(device)], "data_samples": [sample]}, training=False
    )
    return model.predict(data["inputs"], data["data_samples"], rescale=False)


def project(points: torch.Tensor, intrinsics: torch.Tensor, extrinsics: torch.Tensor):
    """Project [Q,4] UE-world cm coordinates with the fixed W2C contract."""
    with torch.autocast(device_type=points.device.type, enabled=False):
        camera = torch.einsum("vij,qj->vqi", extrinsics.float(), points.float())
        pixel = torch.einsum("vij,vqj->vqi", intrinsics.float(), camera[..., :3])
        uv = pixel[..., :2] / pixel[..., 2:3].clamp_min(1e-6)
    valid = ((camera[..., 2] > 1e-4) & (uv[..., 0] >= 0) & (uv[..., 0] < 512) &
             (uv[..., 1] >= 0) & (uv[..., 1] < 512))
    return uv, valid, camera


def project_fp64(points: torch.Tensor, intrinsics: torch.Tensor, extrinsics: torch.Tensor):
    """Audit-only reference projection; deployed projection remains FP32."""
    camera = torch.einsum("vij,qj->vqi", extrinsics.double(), points.double())
    pixel = torch.einsum("vij,vqj->vqi", intrinsics.double(), camera[..., :3])
    uv = pixel[..., :2] / pixel[..., 2:3].clamp_min(1e-12)
    valid = ((camera[..., 2] > 1e-4) & (uv[..., 0] >= 0) & (uv[..., 0] < 512) &
             (uv[..., 1] >= 0) & (uv[..., 1] < 512))
    return uv, valid


def sample_feature(fmap: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Sample [V,C,H,W] map at [V,Q,2] UAV-pixel coordinates."""
    grid = ((uv / 512.0) * 2.0 - 1.0).to(fmap.dtype)
    values = F.grid_sample(fmap, grid[:, None], mode="bilinear",
                           padding_mode="zeros", align_corners=False)
    return values[:, :, 0].transpose(1, 2)  # [V,Q,C]


def pair_cosine(values: torch.Tensor, valid: torch.Tensor) -> Dict[str, float]:
    values = F.normalize(values.float(), dim=-1, eps=1e-6)
    stats = []
    for left in range(values.shape[0]):
        for right in range(left + 1, values.shape[0]):
            keep = valid[left] & valid[right]
            if keep.any():
                stats.append((values[left, keep] * values[right, keep]).sum(-1))
    if not stats:
        return {"pair_cosine": float("nan"), "pair_points": 0}
    joined = torch.cat(stats)
    return {"pair_cosine": float(joined.mean().cpu()), "pair_points": int(joined.numel())}


def make_uav_encoder(args: argparse.Namespace, device: torch.device):
    config = Config.fromfile(str(ROOT / "configs/ue_dual_stream_p2p3_sparse.py"))
    backbone = copy.deepcopy(config.model.drone_backbone)
    backbone["checkpoint_path"] = args.sam2_ckpt
    backbone["init_cfg"]["checkpoint"] = args.sam2_ckpt
    encoder = MODELS.build(ConfigDict(backbone)).to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def encode_p2(encoder, drone_images: torch.Tensor) -> torch.Tensor:
    # The highest-resolution UAV SAM FPN level is the A1 source; this has
    # native 128x128 support and is resampled into the 256x256 satellite P2 grid.
    views = drone_images.shape[0]
    # Exact deployed dual-stream preprocessing (RGB [0,1] -> ImageNet).
    mean = drone_images.new_tensor([.485, .456, .406])[:, None, None]
    std = drone_images.new_tensor([.229, .224, .225])[:, None, None]
    _, fpn = encoder((drone_images - mean) / std)
    level = max(fpn[-4:], key=lambda x: x.shape[-2] * x.shape[-1])
    if level.shape[0] != views:
        raise RuntimeError(f"UAV encoder batch/view mismatch: {tuple(level.shape)}, views={views}")
    return level


def a1_candidate(fmap: torch.Tensor, uv: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Frozen P2 candidate and its required >=2-view hard mask.

    The identity channel projection is intentional for A1-Frozen: a learned
    1x1 projection belongs to hot-start.  This lets the probe test the actual
    P2 residual wiring without creating trainable parameters.
    """
    values = sample_feature(fmap, uv)  # [V,Q,C]
    support = valid.sum(0) >= 2
    weighted = (values * valid[..., None].to(values.dtype)).sum(0)
    candidate = weighted / valid.sum(0).clamp_min(1)[..., None].to(values.dtype)
    candidate = candidate * support[:, None].to(candidate.dtype)
    return candidate.transpose(0, 1).reshape(1, fmap.shape[1], 256, 256), support


def frozen_p2_projection(candidate: torch.Tensor, target_channels: int) -> torch.Tensor:
    """Channel-shape adapter for the zero-gate wiring check, not a learned head."""
    if candidate.shape[1] > target_channels:
        raise RuntimeError("A1 source has more channels than detector P2")
    return F.pad(candidate, (0, 0, 0, 0, 0, target_channels - candidate.shape[1]))


def main() -> None:
    args = arguments()
    if args.grid != 256:
        raise ValueError("A1 is pre-registered at P2=256; do not change --grid.")
    if not Path(args.sam2_ckpt).is_file():
        raise FileNotFoundError(args.sam2_ckpt)
    device = torch.device("cuda:0" if args.device == "cuda" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the A1 frozen gate")
    torch.manual_seed(20260914)
    torch.backends.cudnn.benchmark = False

    checkpoint = _load_checkpoint(Path(args.checkpoint))
    snapshot = _snapshot(checkpoint)
    loader_args = SimpleNamespace(config=None, sam2_ckpt=args.sam2_ckpt, sam2_repo=None)
    _resolve_sam2_repo(loader_args, snapshot)
    model_config, source = _resolve_model_config(loader_args, snapshot)
    detector = _register_and_build(model_config).to(device).eval()
    load_report = _load_model_state(detector, _select_state_dict(checkpoint, "model"), False)
    if load_report["missing_keys"] or load_report["unexpected_keys"]:
        raise RuntimeError(f"strict detector loading failed: {load_report}")
    uav_encoder = make_uav_encoder(args, device)

    dataset = UEDualStreamDataset(args.data_root, "val", image_size=(1024, 1024), load_uav=True)
    n = min(args.scenes, len(dataset))
    parity_n = n if args.parity_scenes == 0 else min(args.parity_scenes, n)
    if n < 2:
        raise RuntimeError("Need at least two scenes for a scene-permuted sham")
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)

    # P2 z=0 grid in the audited 360m satellite-to-world convention.
    def scene_points(item: Mapping[str, Any], shift_x_m: float = 0.0) -> torch.Tensor:
        points = GeometrySparseBEV._ground_points(
            args.grid, args.grid, item["scene_center"].to(device)[None], device, torch.float32
        )[0]
        points[:, 0] += shift_x_m * 100.0
        return points
    feature_indices = torch.linspace(0, args.grid * args.grid - 1,
                                     min(args.feature_points, args.grid * args.grid),
                                     device=device).long()

    parity = []
    telemetry = {key: [] for key in ("correct", "xy_plus_80m", "scene_permuted")}
    calibration = []
    with torch.inference_mode():
        for index in range(n):
            item = dataset[index]
            if tuple(item["intrinsics"].shape) != (5, 3, 3) or tuple(item["extrinsics"].shape) != (5, 4, 4):
                raise RuntimeError(f"{dataset.ids[index]} violates 5-view K/E contract")
            fmap = encode_p2(uav_encoder, item["drone_images"].to(device))
            if fmap.shape[0] != 5 or fmap.shape[1] != 256:
                raise RuntimeError(f"{dataset.ids[index]} UAV P2 source has unexpected shape {tuple(fmap.shape)}")
            intrinsics, extrinsics = item["intrinsics"].to(device), item["extrinsics"].to(device)
            # Audit only: float64 avoids inverse cancellation at UE cm scale.
            E64 = extrinsics.double()
            cam_center = torch.linalg.inv(E64)[:, :, 3]
            residual = torch.einsum("vij,vj->vi", E64, cam_center)
            centre_world = torch.tensor([item["scene_center"][0], item["scene_center"][1], 0.0, 1.0], device=device)
            centre_uv, centre_valid, _ = project(centre_world[None], intrinsics, extrinsics)
            if not torch.isfinite(centre_uv).all() or not centre_valid.all():
                raise RuntimeError(f"{dataset.ids[index]} tile centre fails finite/in-frame 5-view calibration")
            runtime_points = scene_points(item)
            runtime_uv, runtime_valid, _ = project(runtime_points, intrinsics, extrinsics)
            audit_uv, audit_valid = project_fp64(runtime_points, intrinsics, extrinsics)
            jointly_valid = runtime_valid & audit_valid
            max_delta = (runtime_uv.double()[jointly_valid] - audit_uv[jointly_valid]).abs().max() if jointly_valid.any() else torch.tensor(float("inf"), device=device)
            # R2: exact equality is meaningful away from pixel-grid edges.
            runtime_uv64 = runtime_uv.double()
            runtime_margin = torch.stack((runtime_uv64[..., 0].abs(), (runtime_uv64[..., 0] - 512.0).abs(), runtime_uv64[..., 1].abs(), (runtime_uv64[..., 1] - 512.0).abs())).amin(0)
            audit_margin = torch.stack((audit_uv[..., 0].abs(), (audit_uv[..., 0] - 512.0).abs(), audit_uv[..., 1].abs(), (audit_uv[..., 1] - 512.0).abs())).amin(0)
            boundary_band = (runtime_margin.double() <= 0.5) | (audit_margin <= 0.5)
            valid_mismatch = runtime_valid != audit_valid
            interior = ~boundary_band
            support_runtime = runtime_valid.sum(0) >= 2
            support_audit = audit_valid.sum(0) >= 2
            calibration.append({"scene_id": dataset.ids[index], "w2c_camera_center_max_abs": float(residual.sub(torch.tensor([0.,0.,0.,1.], device=device, dtype=torch.float64)).abs().max().cpu()), "tile_center_valid_views": int(centre_valid.sum().cpu()), "tile_center_uv": centre_uv[:, 0].detach().cpu().tolist(), "fp32_fp64_valid_equal": bool(torch.equal(runtime_valid, audit_valid)), "fp32_fp64_max_uv_delta_px": float(max_delta.cpu()), "boundary_band_view_queries": int(boundary_band.sum().cpu()), "valid_mismatch_view_queries": int(valid_mismatch.sum().cpu()), "valid_mismatch_outside_boundary": int((valid_mismatch & interior).sum().cpu()), "interior_valid_equal": bool(torch.equal(runtime_valid[interior], audit_valid[interior])), "two_view_support_equal": bool(torch.equal(support_runtime, support_audit))})

            conditions = {
                "correct": (scene_points(item), fmap),
                "xy_plus_80m": (scene_points(item, 80.0), fmap),
                # Reuse current K/E: visual scene is wrong but geometric field and
                # candidate cardinality stay identical to the correct condition.
                "scene_permuted": (scene_points(item), None),
            }
            for name, (points, source_map) in conditions.items():
                if name == "scene_permuted":
                    # delayed lookup is resolved after the first pass below
                    continue
                uv, valid, _ = project(points, intrinsics, extrinsics)
                candidate, support = a1_candidate(source_map, uv, valid)
                # This is the A1 safety invariant, not merely a reported ratio.
                if candidate[:, :, ~support.reshape(256, 256)].abs().max().item() != 0.0:
                    raise RuntimeError(f"{dataset.ids[index]} {name}: <2-view candidate is nonzero")
                subset_uv, subset_valid = uv[:, feature_indices], valid[:, feature_indices]
                values = sample_feature(source_map, subset_uv) * (subset_valid.sum(0) >= 2)[None, :, None]
                record = {"scene_id": dataset.ids[index], "grid_candidates": args.grid * args.grid,
                          "valid_view_ratio": float(valid.float().mean().cpu()),
                          "any_view_ratio": float(valid.any(0).float().mean().cpu()),
                          "two_view_ratio": float((valid.sum(0) >= 2).float().mean().cpu()),
                          "feature_abs_mean": float(values.abs().mean().cpu()),
                          "hard_mask_zero": True,
                          "candidate_abs_mean": float(candidate.abs().mean().cpu())}
                record.update(pair_cosine(values, subset_valid)); telemetry[name].append(record)
            if index < parity_n:
                reference = instance_digest(predict(detector, item, device))
                # Actual proposed A1 connection: frozen P2 candidate enters a
                # residual P2 injection whose scalar gate is exactly zero.
                correct_uv, correct_valid, _ = project(scene_points(item), intrinsics, extrinsics)
                candidate_p2, _ = a1_candidate(fmap, correct_uv, correct_valid)
                gamma_p2 = candidate_p2.new_zeros(())
                captured = []
                def inject_zero(_module, _inputs, outputs):
                    if not isinstance(outputs, (tuple, list)):
                        raise RuntimeError("detector neck must return a feature sequence")
                    residual_p2 = frozen_p2_projection(candidate_p2, outputs[0].shape[1]).to(outputs[0].dtype)
                    if outputs[0].shape != residual_p2.shape:
                        raise RuntimeError(f"P2 injection shape mismatch {tuple(outputs[0].shape)} vs {tuple(residual_p2.shape)}")
                    injected = outputs[0] + gamma_p2 * residual_p2.to(outputs[0].dtype)
                    if not torch.equal(injected, outputs[0]):
                        raise RuntimeError("zero-gated A1 injection is not bitwise identity")
                    captured.append(tensor_sha256(outputs[0]))
                    return (injected, *outputs[1:])
                hook = detector.neck.register_forward_hook(inject_zero)
                observed = instance_digest(predict(detector, item, device))
                hook.remove()
                parity.append({"scene_id": dataset.ids[index], "prediction_equal": reference == observed,
                               "fields": reference, "p2_hash": captured[0] if captured else None,
                               "gamma_p2": float(gamma_p2.cpu()), "residual_abs_mean": float(candidate_p2.abs().mean().cpu())})

        for index in range(n):
            item = dataset[index]
            intrinsics, extrinsics = item["intrinsics"].to(device), item["extrinsics"].to(device)
            uv, valid, _ = project(scene_points(item), intrinsics, extrinsics)
            # Deliberately stream the next-scene map: retaining all 259 FPNs
            # would turn a read-only gate into a GPU-memory stress test.
            permuted_item = dataset[(index + 1) % n]
            permuted_fmap = encode_p2(uav_encoder, permuted_item["drone_images"].to(device))
            candidate, support = a1_candidate(permuted_fmap, uv, valid)
            if candidate[:, :, ~support.reshape(256, 256)].abs().max().item() != 0.0:
                raise RuntimeError(f"{dataset.ids[index]} scene_permuted: <2-view candidate is nonzero")
            subset_uv, subset_valid = uv[:, feature_indices], valid[:, feature_indices]
            values = sample_feature(permuted_fmap, subset_uv) * (subset_valid.sum(0) >= 2)[None, :, None]
            record = {"scene_id": dataset.ids[index], "grid_candidates": args.grid * args.grid,
                      "valid_view_ratio": float(valid.float().mean().cpu()),
                      "any_view_ratio": float(valid.any(0).float().mean().cpu()),
                      "two_view_ratio": float((valid.sum(0) >= 2).float().mean().cpu()),
                      "feature_abs_mean": float(values.abs().mean().cpu()), "hard_mask_zero": True,
                      "candidate_abs_mean": float(candidate.abs().mean().cpu())}
            record.update(pair_cosine(values, subset_valid)); telemetry["scene_permuted"].append(record)

    def aggregate(records):
        keys = ("valid_view_ratio", "any_view_ratio", "two_view_ratio", "feature_abs_mean", "candidate_abs_mean", "pair_cosine")
        return {key: float(np.nanmean([row[key] for row in records])) for key in keys} | {"scenes": len(records)}
    summary = {
        "experiment_id": args.experiment_id, "arm": "A1-Frozen-R2", "status": "probe_only_no_training",
        "checkpoint": str(Path(args.checkpoint).resolve()), "model_config_source": source,
        "data": {"root": args.data_root, "split": "val", "scene_ids": dataset.ids[:n]},
        "contract": {"grid": args.grid, "height_m": 0.0, "target_meters": 360.0, "uav_views": 5,
                     "extrinsics": "W2C column-vector UE centimetres", "autocast_projection": "disabled_fp32"},
        "parity": {"scenes": parity, "all_prediction_fields_bitwise_equal": bool(parity) and all(row["prediction_equal"] for row in parity)},
        "calibration": calibration,
        "telemetry": {name: {"aggregate": aggregate(rows), "per_scene": rows} for name, rows in telemetry.items()},
        "gates": {"a0_bypass_bitwise": bool(parity) and all(row["prediction_equal"] for row in parity),
                  "all_conditions_feature_nonzero": all(bool(rows) and all(row["feature_abs_mean"] > 0 for row in rows) for rows in telemetry.values()),
                  "candidate_counts_equal": all(len({row["grid_candidates"] for rows in telemetry.values() for row in rows}) == 1 for _ in [0]),
                  "hard_mask_zero": all(row["hard_mask_zero"] for rows in telemetry.values() for row in rows),
                  "w2c_camera_center_fp64": max(row["w2c_camera_center_max_abs"] for row in calibration) < 1e-6,
                  "fp32_fp64_projection_r2": all(row["interior_valid_equal"] and row["two_view_support_equal"] and row["valid_mismatch_outside_boundary"] == 0 and row["fp32_fp64_max_uv_delta_px"] <= 0.5 for row in calibration),
                  "tile_center_five_view": all(row["tile_center_valid_views"] == 5 for row in calibration),
                  "note": "Affinity is descriptive only; it is not a semantic-effectiveness gate for frozen SAM features."},
    }
    with (output_dir / "a1_frozen_probe.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({"output": str(output_dir / "a1_frozen_probe.json"), "gates": summary["gates"],
                      "correct": summary["telemetry"]["correct"]["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
