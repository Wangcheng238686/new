"""Geometry perturbation probe for the trained UE dual-stream checkpoints.

Answers two questions the fusion monitors cannot:
  1. Orientation: rotate the UAV extrinsics around the world Z axis by
     90/180/270 degrees at inference -- if the trained model actually uses
     BEV geometry, predictions must react; if outputs are unchanged, the
     fusion path is effectively closed regardless of geometry correctness.
  2. Magnitude: zero the injection gates (gamma -> 0 recovers the exact
     single-stream model) as the reference floor.

Usage (from portable_sam2_explicit_coarse/):
  python scripts/ue/probe_geometry.py \
      --checkpoint work_dirs/ue/<run>/best_model_epochNN.pth [--scenes 40]

Also dumps per-scene prediction-agreement stats between settings.
"""
import argparse
import math
import os
import sys
from pathlib import Path

# Env contract must match scripts/ue/run_ue_dual_stream.sh.
os.environ.setdefault("EXPLICIT_PROMPT_MODE", "points_box_dense")
os.environ.setdefault("P2_BOUNDARY_REFINER_ENABLED", "1")
os.environ.setdefault("SAM_IMAGE_EMBED_STRIDE", "16")
os.environ.setdefault("SHAPE_DENSE_TRANSFORM", "raw_logits")
os.environ.setdefault("SHAPE_DENSE_DETACH", "1")
os.environ.setdefault("FINAL_MASK_COORDINATE_MODE", "roi_local")
os.environ.setdefault("SAM2_CKPT", "/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt")
os.environ.setdefault("UE_DUAL_INJECT_EMBEDDING", "1")

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from mmengine.config import Config

import mmdet.models  # noqa: F401  (registry side effects)
import rsprompter  # noqa: F401  (registers DualStreamRSPrompterAnchor)
from mmdet.registry import MODELS
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmdet.structures import DetDataSample
from mmdet.structures.mask import BitmapMasks
from scipy.optimize import linear_sum_assignment

from mmengine.registry import MODELS as MMENGINE_MODELS
if "DetDataPreprocessor" not in MMENGINE_MODELS:
    MMENGINE_MODELS.register_module(
        name="DetDataPreprocessor", module=DetDataPreprocessor
    )

from data.ue_dual_stream_dataset import UEDualStreamDataset


def rot_z(deg: float) -> torch.Tensor:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    T = torch.tensor([
        [c, -s, 0.0, 0.0],
        [s, c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    return T


def build_sample(item, device):
    ds = DetDataSample()
    ds.set_metainfo(item["img_metas"])
    ds.set_metainfo({
        "drone_images": item["drone_images"],
        "intrinsics": item["intrinsics"],
        "extrinsics": item["extrinsics"],
        "scene_center": item["scene_center"],
    })
    return ds.to(device)


def pred_masks_and_scores(outputs):
    """Flatten a predict() output into (masks [N,H,W] bool, scores [N])."""
    masks, scores = [], []
    for res in outputs:
        inst = getattr(res, "pred_instances", None)
        if inst is None or inst.bboxes is None or len(inst.bboxes) == 0:
            continue
        m = inst.masks
        if isinstance(m, BitmapMasks):
            arr = m.to_ndarray()
        else:
            arr = m.cpu().numpy()
        masks.append(arr.astype(bool))
        scores.append(inst.scores.detach().float().cpu().numpy())
    if not masks:
        return np.zeros((0, 1024, 1024), bool), np.zeros((0,), np.float32)
    return masks[0], scores[0]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    A = a.reshape(len(a), -1).astype(np.float32)
    B = b.reshape(len(b), -1).astype(np.float32)
    inter = A @ B.T
    union = A.sum(1)[:, None] + B.sum(1)[None, :] - inter
    return inter / np.clip(union, 1.0, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/ue_dual_stream_p2p3_sparse.py")
    ap.add_argument("--data-root", default="/data1/wangcheng/dataset/ue_coco_with_uav")
    ap.add_argument("--scenes", type=int, default=40)
    ap.add_argument("--image-size", type=int, default=1024)
    args = ap.parse_args()

    device = torch.device("cuda")
    cfg = Config.fromfile(args.config)
    model = MODELS.build(cfg.model)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    msg = model.load_state_dict(state, strict=False)
    print(f"loaded {args.checkpoint}: missing={len(msg.missing_keys)} "
          f"unexpected={len(msg.unexpected_keys)}")
    model = model.to(device).eval()

    gammas = {
        "gamma_p2": float(model.p2p3_gate.gamma_p2),
        "gamma_p3": float(model.p2p3_gate.gamma_p3),
        "gamma_emb": float(model.embedding_gate.gamma)
        if getattr(model, "embedding_gate", None) is not None else None,
    }
    print("checkpoint gate values:", gammas)

    dataset = UEDualStreamDataset(args.data_root, "val",
                                  image_size=(args.image_size, args.image_size),
                                  load_uav=True)
    n = min(args.scenes, len(dataset))
    print(f"probing {n} / {len(dataset)} val scenes")

    settings = ["rot0", "rot90", "rot180", "rot270", "gates_zero"]
    results = {s: [] for s in settings}  # per-scene (masks, scores)

    with torch.no_grad():
        for idx in range(n):
            item = dataset[idx]
            for setting in settings:
                ds = build_sample(item, device)
                if setting.startswith("rot"):
                    deg = float(setting[3:])
                    T = rot_z(deg).to(device)
                    E = ds.metainfo["extrinsics"].to(device) @ T
                    ds.set_metainfo({"extrinsics": E})
                imgs = item["img"][None].to(device)
                # mirror the trainer: run through the model's own preprocessor
                proc = model.data_preprocessor(
                    {"inputs": [imgs[0]], "data_samples": [ds]}, training=False)
                if setting == "gates_zero":
                    model.p2p3_gate.gamma_p2.data.zero_()
                    model.p2p3_gate.gamma_p3.data.zero_()
                    if getattr(model, "embedding_gate", None) is not None:
                        model.embedding_gate.gamma.data.zero_()
                out = model.predict(proc["inputs"], proc["data_samples"], rescale=False)
                if setting == "gates_zero":  # restore
                    model.p2p3_gate.gamma_p2.data.fill_(gammas["gamma_p2"])
                    model.p2p3_gate.gamma_p3.data.fill_(gammas["gamma_p3"])
                    if getattr(model, "embedding_gate", None) is not None:
                        model.embedding_gate.gamma.data.fill_(gammas["gamma_emb"])
                masks, scores = pred_masks_and_scores(out)
                results[setting].append((masks, scores))

    # ---- agreement analysis: each rotation vs rot0, and gates_zero vs rot0 ----
    print("\n=== prediction agreement vs rot0 ===")
    print(f"{'setting':>10} | {'mean IoU(matched)':>17} | {'score L1':>9} | "
          f"{'#inst diff':>10} | {'identical':>9}")
    for s in settings[1:]:
        ious, sl1, idiff, identical = [], [], [], 0
        for (m0, s0), (m1, s1) in zip(results["rot0"], results[s]):
            if len(m0) == 0 and len(m1) == 0:
                identical += 1
                continue
            if np.array_equal(m0, m1):
                identical += 1
                continue
            M = iou_matrix(m0, m1)
            r, c = linear_sum_assignment(-M)
            ok = M[r, c] > 0.3
            ious.extend(M[r[ok], c[ok]].tolist())
            idiff.append(len(m1) - len(m0))
            k = min(len(s0), len(s1))
            if k:
                sl1.append(float(np.abs(s0[:k] - s1[:k]).mean()))
        miou = float(np.mean(ious)) if ious else float("nan")
        sl = float(np.mean(sl1)) if sl1 else 0.0
        di = float(np.mean(idiff)) if idiff else 0.0
        print(f"{s:>10} | {miou:>17.4f} | {sl:>9.5f} | {di:>10.2f} | "
              f"{identical:>6}/{n}")

    # ---- GT-matched quality per setting (mean matched IoU vs GT) ----
    print("\n=== GT-matched mean IoU per setting ===")
    for s in settings:
        matched = []
        for idx in range(n):
            gt = dataset[idx]["gt_masks"]  # [N,1024,102] uint8
            if gt.ndim != 3 or len(gt) == 0:
                continue
            masks, _ = results[s][idx]
            if len(masks) == 0:
                continue
            # gt masks are at dataset image_size
            M = iou_matrix(gt.astype(bool), masks)
            r, c = linear_sum_assignment(-M)
            matched.extend(M[r, c][M[r, c] > 0.3].tolist())
        print(f"{s:>10}: matched-IoU={np.mean(matched) if matched else float('nan'):.4f} "
              f"(matches={len(matched)})")


if __name__ == "__main__":
    main()
