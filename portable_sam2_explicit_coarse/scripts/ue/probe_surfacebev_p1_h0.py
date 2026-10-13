#!/usr/bin/env python3
"""SURFBEV-P1-V2-H0: frozen SurfaceBEV harvesting and parity gate.

This deliberately has no labels and refuses the test split.  It is the
pre-readout wiring gate defined in docs/SURFBEV-P1-V2-H0_preregistration.md.
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

os.environ.setdefault("SAM2_CKPT", "/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from mmengine.config import Config, ConfigDict
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample

from data.ue_raw_pair_dataset import PAIR_SHA256, SPLIT_SHA256, UERawPairDataset
from inference.infer_from_checkpoint import (
    _load_checkpoint, _load_model_state, _register_and_build, _resolve_model_config,
    _resolve_sam2_repo, _select_state_dict, _snapshot,
)
from rsprompter.surfacebev import devicez_to_camera_z, rasterize_surfacebev

EXPECTED_B0_PATH = Path("/data1/wangcheng/checkpoint/surfbev_b0/surfacebev_b0_v2_raw1024_noaug_ddp4_s44_EXP-0006/last_model_epoch100.pth")
EXPECTED_B0_SHA256 = "fc3ead0e9490ffd9249cdbc4ce2ee9556bed5252efc3e475dfcc0f2aace65145"
EXPECTED_RUN_TAG = "surfacebev_b0_v2_raw1024_noaug_ddp4_s44_EXP-0006"


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default=str(ROOT / "data/ue_raw_hero_uav_pairset_20261011"))
    p.add_argument("--split-manifest", default="surfbev_b0_split_v2.json")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--sam2-ckpt", default=os.environ["SAM2_CKPT"])
    p.add_argument("--train-scenes", type=int, default=0, help="0 means all 694; smaller is smoke only.")
    p.add_argument("--val-scenes", type=int, default=0, help="0 means all 100; smaller is smoke only.")
    p.add_argument("--height-samples-per-view", type=int, default=0,
                   help="0 means every descriptor centre; nonzero is smoke only.")
    p.add_argument("--experiment-id", default="SURFBEV-P1-V2-H0-20261012")
    return p.parse_args()


def sha_tensor(x):
    return hashlib.sha256(x.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_predictions(outputs):
    instance = outputs[0].pred_instances
    result = {}
    for key in ("bboxes", "scores", "labels", "masks"):
        value = getattr(instance, key)
        if hasattr(value, "to_ndarray"):
            value = value.to_ndarray()
        elif hasattr(value, "masks"):
            value = value.masks
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        result[key] = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
    return result


def model_input(model, item, device):
    sample = DetDataSample(); sample.set_metainfo(dict(item["img_metas"]))
    return model.data_preprocessor({"inputs": [item["img"].to(device)],
                                    "data_samples": [sample.to(device)]}, training=False)


@torch.inference_mode()
def passive_p3_and_parity(model, item, device):
    before = model_input(model, item, device)
    pred_before = model.predict(before["inputs"], before["data_samples"], rescale=False)
    digest_before = digest_predictions(pred_before)
    feature_data = model_input(model, item, device)
    pyramids, _, _, _ = model.extract_feat(feature_data["inputs"], feature_data["data_samples"])
    p3 = pyramids[1][0].float()
    after = model_input(model, item, device)
    pred_after = model.predict(after["inputs"], after["data_samples"], rescale=False)
    digest_after = digest_predictions(pred_after)
    if digest_before != digest_after:
        raise RuntimeError("passive P3 harvesting changed B0 predictions")
    if p3.shape[-2:] != (128, 128) or not torch.isfinite(p3).all():
        raise RuntimeError(f"unexpected/nonfinite B0 P3: {tuple(p3.shape)}")
    return p3, digest_before


def make_uav_encoder(args, device):
    cfg = Config.fromfile(str(ROOT / "configs/ue_dual_stream_p2p3_sparse.py"))
    backbone = copy.deepcopy(cfg.model.drone_backbone)
    backbone["checkpoint_path"] = args.sam2_ckpt
    backbone["init_cfg"]["checkpoint"] = args.sam2_ckpt
    encoder = MODELS.build(ConfigDict(backbone)).to(device).eval()
    encoder.requires_grad_(False)
    return encoder


@torch.inference_mode()
def uav_descriptors(encoder, rgb):
    mean = rgb.new_tensor([.485, .456, .406])[:, None, None]
    std = rgb.new_tensor([.229, .224, .225])[:, None, None]
    _, fpn = encoder((rgb - mean) / std)
    source = max(tuple(fpn[-4:]), key=lambda f: f.shape[-2] * f.shape[-1]).float()
    if source.shape[0] != 5 or source.ndim != 4 or not torch.isfinite(source).all():
        raise RuntimeError(f"invalid frozen UAV descriptor tensor {tuple(source.shape)}")
    return source


def cameras(item, device):
    def k_and_w2c(pair):
        intr, extr = pair
        # This is an audit, so retain float64 through the inverse product.  At
        # UE centimetre world scales float32 cancellation is not a geometry
        # failure (D0-R1 registered the float64 <=1e-6 cm criterion).
        k = torch.tensor([[intr["fx"], 0., intr["cx"]], [0., intr["fy"], intr["cy"]], [0., 0., 1.]], device=device)
        w2c64 = torch.tensor(extr["world_to_camera_cv"], device=device, dtype=torch.float64)
        c2w64 = torch.tensor(extr["camera_to_world_cv"], device=device, dtype=torch.float64)
        if not torch.allclose(w2c64 @ c2w64, torch.eye(4, device=device, dtype=torch.float64), atol=1e-6, rtol=0):
            raise RuntimeError("UAV/satellite camera C2W-W2C contract failed")
        # SurfaceBEV's deployed projection remains explicitly FP32.
        return k.float(), w2c64.float(), c2w64.float()
    uk, uw, uc = zip(*(k_and_w2c(x) for x in item["uav_cameras"]))
    sk, sw, _ = k_and_w2c(item["satellite_camera"])
    if len(uk) != 5:
        raise RuntimeError("raw pair does not have five UAV cameras")
    return torch.stack(uk), torch.stack(uc), sk, sw


@torch.inference_mode()
def scene_height_samples(item, device, fh, fw, limit):
    uk, uc, _, _ = cameras(item, device)
    yy, xx = torch.meshgrid(torch.arange(fh, device=device), torch.arange(fw, device=device), indexing="ij")
    px = ((xx.float() + .5) * 512. / fw).long().clamp(max=511)
    py = ((yy.float() + .5) * 512. / fh).long().clamp(max=511)
    z = devicez_to_camera_z(item["devicez_rgb"].to(device))[:, py, px].reshape(5, -1)
    u = ((xx.reshape(-1).float() + .5) * 512. / fw)[None]
    v = ((yy.reshape(-1).float() + .5) * 512. / fh)[None]
    pc = torch.stack(((u-uk[:,None,0,2])*z/uk[:,None,0,0], (v-uk[:,None,1,2])*z/uk[:,None,1,1], z), -1)
    world = pc @ uc[:, :3, :3].transpose(1, 2) + uc[:, None, :3, 3]
    values = world[..., 2].reshape(-1)
    values = values[torch.isfinite(values)]
    if limit and values.numel() > limit:
        ids = torch.linspace(0, values.numel() - 1, limit, device=device).long()
        values = values[ids]
    return values.cpu().numpy()


def main():
    args = arguments()
    if Path(args.split_manifest).name != "surfbev_b0_split_v2.json":
        raise ValueError("H0 is only registered for the leakage-controlled v2 split")
    if not Path(args.sam2_ckpt).is_file():
        raise FileNotFoundError(args.sam2_ckpt)
    root, out = Path(args.data_root).resolve(), Path(args.output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"refusing to overwrite nonempty output: {out}")
    out.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("H0 needs a CUDA device")
    checkpoint_path = Path(args.checkpoint).resolve()
    if checkpoint_path != EXPECTED_B0_PATH or checkpoint_path.name != "last_model_epoch100.pth":
        raise RuntimeError("H0 is locked to the registered EXP-0006 epoch-100 last checkpoint")
    checkpoint_sha256 = sha_file(checkpoint_path)
    if checkpoint_sha256 != EXPECTED_B0_SHA256:
        raise RuntimeError("registered EXP-0006 last checkpoint SHA256 mismatch")
    checkpoint = _load_checkpoint(checkpoint_path); snapshot = _snapshot(checkpoint)
    training_args = snapshot.get("training_args", {})
    if (checkpoint.get("epoch") != 99 or training_args.get("run_tag") != EXPECTED_RUN_TAG or
            training_args.get("seed") != 44 or training_args.get("epochs") != 100):
        raise RuntimeError("checkpoint metadata is not EXP-0006 seed-44 epoch-100 last")
    if snapshot.get("training_args", {}).get("raw_split_manifest") != "surfbev_b0_split_v2.json":
        raise RuntimeError("checkpoint is not a v2 B0 checkpoint")
    loader_args = SimpleNamespace(config=None, sam2_ckpt=args.sam2_ckpt, sam2_repo=None)
    _resolve_sam2_repo(loader_args, snapshot)
    cfg, cfg_source = _resolve_model_config(loader_args, snapshot)
    detector = _register_and_build(cfg).to(device).eval()
    report = _load_model_state(detector, _select_state_dict(checkpoint, "model"), False)
    if report["missing_keys"] or report["unexpected_keys"]:
        raise RuntimeError(f"strict B0 checkpoint loading failed: {report}")
    uav_encoder = make_uav_encoder(args, device)
    train = UERawPairDataset(root, "train", load_uav=True, split_manifest=args.split_manifest, load_targets=False)
    val = UERawPairDataset(root, "val", load_uav=True, split_manifest=args.split_manifest, load_targets=False)
    ntrain = len(train) if args.train_scenes == 0 else min(args.train_scenes, len(train))
    nval = len(val) if args.val_scenes == 0 else min(args.val_scenes, len(val))
    smoke = ntrain != len(train) or nval != len(val) or bool(args.height_samples_per_view)
    # Obtain the actual frozen UAV descriptor resolution once, then determine
    # height bounds exclusively from train data before reading validation.
    train_first = train[0]
    if any(name.startswith("gt_") for name in train_first):
        raise RuntimeError("H0 label-free reader constructed train target fields")
    first = uav_descriptors(uav_encoder, train_first["drone_images"].to(device))
    heights = []
    for i in range(ntrain):
        train_item = train[i]
        if any(name.startswith("gt_") for name in train_item):
            raise RuntimeError("H0 label-free reader constructed train target fields")
        heights.append(scene_height_samples(train_item, device, first.shape[-2], first.shape[-1], args.height_samples_per_view))
        if (i + 1) % 50 == 0: print(f"height pass {i+1}/{ntrain}", flush=True)
    heights = np.concatenate(heights)
    low, high = np.quantile(heights, [.005, .995])
    if not (np.isfinite(low) and np.isfinite(high) and high > low):
        raise RuntimeError("invalid train-only SurfaceBEV height bounds")
    edges = torch.linspace(float(low), float(high), 13, device=device)
    rows = []
    for i in range(nval):
        item = val[i]
        if any(name.startswith("gt_") for name in item):
            raise RuntimeError("H0 label-free reader constructed target fields")
        p3, pred_hash = passive_p3_and_parity(detector, item, device)
        desc = uav_descriptors(uav_encoder, item["drone_images"].to(device))
        descriptor_hash = sha_tensor(desc)
        uk, uc, sk, sw = cameras(item, device)
        correct, ct = rasterize_surfacebev(desc, item["devicez_rgb"].to(device), uk, uc, sk, sw, edges, p3.shape[-2:], False)
        if sha_tensor(desc) != descriptor_hash:
            raise RuntimeError("correct rasterizer mutated frozen UAV descriptors")
        sham, st = rasterize_surfacebev(desc, item["devicez_rgb"].to(device), uk, uc, sk, sw, edges, p3.shape[-2:], True)
        if correct.shape[-2:] != p3.shape[-2:] or sham.shape != correct.shape:
            raise RuntimeError("SurfaceBEV/P3 shape contract failed")
        if not (torch.isfinite(correct).all() and torch.isfinite(sham).all()):
            raise RuntimeError("nonfinite SurfaceBEV")
        if ct["input_points"] != st["input_points"] or sha_tensor(desc) != descriptor_hash:
            raise RuntimeError("correct/sham raw descriptor or candidate-count invariant failed")
        for stat in (ct, st):
            if stat["zbuffer_points"] <= 0 or stat["covered_cells"] <= 0 or not np.isfinite(stat["zbuffer_residual_mean"]):
                raise RuntimeError(f"surface telemetry invalid: {stat}")
        rows.append(dict(scene_id=item["img_metas"]["scene_id"], b0_prediction_hash=pred_hash,
                         p3_shape=list(p3.shape), p3_sha256=sha_tensor(p3), descriptor_sha256=descriptor_hash,
                         descriptor_shape=list(desc.shape), correct=ct, sham90=st,
                         raw_input_invariant=dict(correct_input_points=ct["input_points"], sham_input_points=st["input_points"],
                                                  correct_descriptor_sha256=descriptor_hash, sham_descriptor_sha256=descriptor_hash,
                                                  equal=True),
                         correct_nonmissing_bins=int((correct[:, -2] == 0).sum().cpu()),
                         sham_nonmissing_bins=int((sham[:, -2] == 0).sum().cpu())))
        if (i + 1) % 10 == 0: print(f"harvest {i+1}/{nval}", flush=True)
    summary = dict(experiment_id=args.experiment_id, arm="P1-V2-H0", status="smoke" if smoke else "formal_harvest",
                   checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint_sha256, checkpoint_epoch_0based=checkpoint.get("epoch"),
                   checkpoint_epoch_1based=int(checkpoint.get("epoch", -1)) + 1, config_source=cfg_source,
                   data=dict(pair_manifest_sha256=PAIR_SHA256, split_sha256=SPLIT_SHA256["surfbev_b0_split_v2.json"],
                             train_total=len(train), val_total=len(val), train_used=ntrain, val_used=nval, test_read=False),
                   height_edges_cm=edges.cpu().tolist(), height_quantile_source="train_only_0.5_99.5",
                   invariant=dict(passive_prediction_hash_before_after=True, strict_checkpoint_load=True,
                                  label_free_reader=True, checkpoint_identity_locked=True,
                                  raw_descriptor_equal_by_construction=True, p3_hw=[128, 128], surface_height_bins=12),
                   verdict="H0_PASS" if (not smoke and len(rows) == len(val)) else "SMOKE_ONLY",
                   decision_rule="H0_PASS requires all 694 train height scenes and all 100 val harvest scenes; smoke is not a gate decision.")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (out / "per_scene.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
