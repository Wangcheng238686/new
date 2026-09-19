"""P5: does the frozen detector's mask pathway respond to boundary-aligned
feature changes, and does that response reach AP75 / boundary F1?

Causal perturbation probe on the trained single-stream checkpoint (fully
frozen, no training). For each val scene we add a controlled, spatially
structured perturbation to the P2 pyramid map (x[0]) AFTER the neck:

    x0' = x0 + alpha * sigma_c * sdf_z * s_c

where sdf_z is the z-scored GT signed-distance-to-boundary field (256 res),
sigma_c the per-channel std of x0, and s_c = sign of the per-channel
correlation between x0[c] and sdf_z estimated once on 20 TRAIN scenes (so the
perturbation pushes each channel along its own boundary-encoding direction).

Settings (pre-registered):
  alpha ladder, aligned field : 0 / 0.1 / 0.25 / 0.5 / 1.0
  alpha=0.25, random field    : smoothed gaussian noise, z-scored, same s_c
  alpha=0.25, mismatched field: sdf of a different scene

Pre-registered interpretation (written before results):
  Responsiveness : union-mask XOR fraction vs alpha=0 grows with alpha
                   (decoder output is not invariant to boundary features).
  Efficacy       : at alpha=0.25 the ALIGNED field improves AP75 and boundary
                   F1 vs alpha=0, and by more than the random-field control;
                   mismatched field must not improve (sign/specificity check).
  Outcome table  : monotone gain -> mechanism viable as-is;
                   inverted-U with positive peak -> viable with tuned alpha;
                   flat or negative -> feature-to-metric chain broken, the
                   BEV->refiner recommendation must be reconsidered.

This probe does NOT train anything and does NOT modify any checkpoint or data.
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

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

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.registry import MODELS as MMENGINE_MODELS

import mmdet.models  # noqa: F401
import rsprompter  # noqa: F401
from mmdet.registry import MODELS
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmdet.structures import DetDataSample
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

if "DetDataPreprocessor" not in MMENGINE_MODELS:
    MMENGINE_MODELS.register_module(name="DetDataPreprocessor", module=DetDataPreprocessor)

from data.ue_dual_stream_dataset import UEDualStreamDataset


def sdf_field(occ: np.ndarray) -> np.ndarray:
    d_out = cv2.distanceTransform(1 - occ, cv2.DIST_L2, 3)
    d_in = cv2.distanceTransform(occ, cv2.DIST_L2, 3)
    sdf = d_out - d_in
    return (sdf - sdf.mean()) / (sdf.std() + 1e-6)


def occ256(item) -> np.ndarray:
    occ = np.zeros((256, 256), np.uint8)
    for m in item["gt_masks"]:
        occ = np.maximum(occ, cv2.resize(m, (256, 256), interpolation=cv2.INTER_NEAREST))
    return occ


def random_field(rng) -> np.ndarray:
    f = rng.standard_normal((256, 256)).astype(np.float32)
    f = cv2.GaussianBlur(f, (0, 0), 4)
    return (f - f.mean()) / (f.std() + 1e-6)


def masks_to_rle(masks: np.ndarray) -> list:
    out = []
    for m in masks:
        rle = maskUtils.encode(np.asfortranarray(m.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode()
        out.append(rle)
    return out


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tol: int = 3) -> float:
    k = np.ones((3, 3), np.uint8)
    def bnd(m):
        return (m.astype(np.uint8) ^ cv2.erode(m.astype(np.uint8), k)) > 0
    pb, gb = bnd(pred), bnd(gt)
    if pb.sum() == 0 and gb.sum() == 0:
        return 1.0
    pd = cv2.dilate(pb.astype(np.uint8), np.ones((2 * tol + 1,) * 2, np.uint8)) > 0
    gd = cv2.dilate(gb.astype(np.uint8), np.ones((2 * tol + 1,) * 2, np.uint8)) > 0
    prec = (pb & gd).sum() / max(pb.sum(), 1)
    rec = (gb & pd).sum() / max(gb.sum(), 1)
    return 2 * prec * rec / max(prec + rec, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single-ckpt", default="work_dirs/ue/ue_single_c5v2_control_tr128_va259_s44_r1/best_model_epoch86.pth")
    ap.add_argument("--config", default="configs/ue_dual_stream_p2p3_sparse.py")
    ap.add_argument("--data-root", default="/data1/wangcheng/dataset/ue_coco_with_uav")
    ap.add_argument("--n-calib", type=int, default=20)
    ap.add_argument("--n-val", type=int, default=40)
    ap.add_argument("--out", default="outputs/probe_decoder_boundary_response")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    cfg = Config.fromfile(args.config)
    model = MODELS.build(cfg.model)
    ck = torch.load(args.single_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model", ck.get("state_dict")), strict=False)
    model = model.to(device).eval()

    train_ds = UEDualStreamDataset(args.data_root, "train", image_size=(1024, 1024),
                                   train_subset=0.1797752808988764, seed=44, load_uav=True)
    val_ds = UEDualStreamDataset(args.data_root, "val", image_size=(1024, 1024), load_uav=True)
    n_val = min(args.n_val, len(val_ds))
    print(f"calib scenes={args.n_calib} val scenes={n_val}")

    # ---- calibration: per-channel sign of (x0 channel, sdf) correlation on train scenes ----
    corr_acc, n_c = None, 0
    with torch.no_grad():
        for idx in range(args.n_calib):
            item = train_ds[idx]
            if len(item["gt_labels"]) == 0:
                continue
            ds = DetDataSample(); ds.set_metainfo(item["img_metas"])
            proc = model.data_preprocessor(
                {"inputs": [item["img"].to(device)], "data_samples": [ds.to(device)]}, training=False)
            inputs = proc["inputs"][0] if isinstance(proc["inputs"], (list, tuple)) else proc["inputs"]
            vo = model.backbone(inputs)
            x0 = model.neck(vo[1])[0][0].float()
            sdf = sdf_field(occ256(item))
            a = x0.reshape(512, -1).cpu().numpy()
            b = (sdf - sdf.mean()).reshape(-1)
            a = a - a.mean(1, keepdims=True)
            c = (a @ b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b) + 1e-9)
            corr_acc = c if corr_acc is None else corr_acc + c
            n_c += 1
    sgn = np.sign(corr_acc / n_c).astype(np.float32)
    sgn[sgn == 0] = 1.0
    torch_sgn = torch.from_numpy(sgn).to(device)[:, None, None]
    print(f"calibration done on {n_c} scenes; "
          f"sign balance +:{(sgn>0).sum()} -:{(sgn<0).sum()}")

    # ---- neck patching ----
    state = dict(field=None, alpha=0.0)
    orig_forward = model.neck.forward
    def patched_forward(hidden):
        x = orig_forward(hidden)
        if state["field"] is not None and state["alpha"] > 0:
            x0 = x[0].float()
            sigma = x0.flatten(2).std(dim=2).mean()[None, None, None]
            delta = state["alpha"] * sigma * state["field"][None].to(x0) * torch_sgn[None]
            x = list(x)
            x[0] = (x[0].float() + delta).to(x[0].dtype)
            x = tuple(x)
        return x
    model.neck.forward = patched_forward

    # ---- GT coco ground truth (masks at 1024) ----
    cats = [{"id": 1, "name": "building"}]
    imgs, anns = [], []
    aid = 1
    for i in range(n_val):
        item = val_ds[i]
        imgs.append(dict(id=i, file_name=item["img_metas"]["scene_id"], height=1024, width=1024))
        for m in item["gt_masks"]:
            rle = maskUtils.encode(np.asfortranarray(m))
            rle["counts"] = rle["counts"].decode()
            xs, ys = np.where(m)
            anns.append(dict(id=aid, image_id=i, category_id=1, segmentation=rle,
                             bbox=[float(xs.min()), float(ys.min()),
                                   float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)],
                             area=int(m.sum()), iscrowd=0))
            aid += 1
    gt = dict(images=imgs, annotations=anns, categories=cats)
    gtjson = out / "gt_subset.json"
    json.dump(gt, gtjson.open("w"))
    coco_gt = COCO(str(gtjson))
    print(f"gt subset: {len(imgs)} images, {len(anns)} instances")

    settings = [("aligned", a) for a in (0.0, 0.1, 0.25, 0.5, 1.0)] + \
               [("random", 0.25), ("mismatch", 0.25)]
    results_by_setting, sens, bf1_by_setting = {}, {}, {}

    t0 = time.time()
    for si, (kind, alpha) in enumerate(settings):
        model.neck.forward = patched_forward
        coco_dt_list = []
        xor_fracs, bf1s = [], []
        for i in range(n_val):
            item = val_ds[i]
            ds = DetDataSample(); ds.set_metainfo(item["img_metas"])
            ds.set_metainfo({"drone_images": item["drone_images"], "intrinsics": item["intrinsics"],
                             "extrinsics": item["extrinsics"], "scene_center": item["scene_center"]})
            proc = model.data_preprocessor(
                {"inputs": [item["img"].to(device)], "data_samples": [ds.to(device)]}, training=False)
            inputs = proc["inputs"][0] if isinstance(proc["inputs"], (list, tuple)) else proc["inputs"]
            if kind == "aligned":
                state["field"] = torch.from_numpy(sdf_field(occ256(item)).astype(np.float32))
            elif kind == "random":
                state["field"] = torch.from_numpy(random_field(np.random.RandomState(1000 + i)).astype(np.float32))
            else:
                j = (i + 7) % n_val
                state["field"] = torch.from_numpy(sdf_field(occ256(val_ds[j])).astype(np.float32))
            state["alpha"] = alpha
            with torch.no_grad():
                outs = model.predict(inputs, proc["data_samples"], rescale=False)
            inst = outs[0].pred_instances
            if inst is None or len(inst.bboxes) == 0:
                pm = np.zeros((1024, 1024), bool)
            else:
                msk = inst.masks
                pmasks = msk.to_ndarray() if isinstance(msk, BitmapMasks__) else msk.cpu().numpy()
                keep = inst.scores.cpu().numpy() >= 0.5
                pm = pmasks[keep].any(0) if keep.any() else np.zeros((1024, 1024), bool)
            gtocc = np.zeros((1024, 1024), np.uint8)
            for m in item["gt_masks"]:
                gtocc = np.maximum(gtocc, m)
            if alpha > 0:
                xor_fracs.append(float((pm ^ (gtocc > 0)).mean()))
            bf1s.append(boundary_f1(pm, gtocc > 0))
            # coco results: all instances with scores
            if inst is not None and len(inst.bboxes):
                msk = inst.masks
                pmasks = msk.to_ndarray() if isinstance(msk, BitmapMasks__) else msk.cpu().numpy()
                rles = masks_to_rle(pmasks)
                scs = inst.scores.cpu().numpy()
                for r, s in zip(rles, scs):
                    coco_dt_list.append(dict(image_id=i, category_id=1, segmentation=r, score=float(s)))
        results_by_setting[f"{kind}@{alpha}"] = coco_dt_list
        bf1_by_setting[f"{kind}@{alpha}"] = float(np.mean(bf1s))
        sens[f"{kind}@{alpha}"] = float(np.mean(xor_fracs)) if xor_fracs else 0.0
        print(f"[{si+1}/{len(settings)}] {kind}@{alpha}: boundaryF1={bf1_by_setting[f'{kind}@{alpha}']:.4f} "
              f"sensitivity={sens[f'{kind}@{alpha}']:.5f} ({time.time()-t0:.0f}s)")

    # ---- COCO eval per setting ----
    print("\n===== COCO segm metrics per setting =====")
    metrics = {}
    for key, dt in results_by_setting.items():
        if not dt:
            metrics[key] = None
            continue
        coco_dt = coco_gt.loadRes(dt)
        ev = COCOeval(coco_gt, coco_dt, "segm")
        ev.params.imgIds = list(range(n_val))
        ev.evaluate(); ev.accumulate(); ev.summarize()
        m = dict(AP=float(ev.stats[0]), AP50=float(ev.stats[1]), AP75=float(ev.stats[2]))
        metrics[key] = m
        print(f"  {key:12s} AP={m['AP']:.4f} AP50={m['AP50']:.4f} AP75={m['AP75']:.4f}")

    summary = dict(settings=list(results_by_setting.keys()), metrics=metrics,
                   boundary_f1=bf1_by_setting, sensitivity=sens,
                   lineage=dict(single_ckpt=args.single_ckpt, data_root=args.data_root,
                                n_val=n_val, n_calib=args.n_calib,
                                perturbation="P2 map, alpha*sigma_c*sdf_z*s_c"))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---- pre-registered readout ----
    print("\n===== pre-registered readout =====")
    base, al = metrics.get("aligned@0.0"), metrics.get("aligned@0.25")
    rnd, mis = metrics.get("random@0.25"), metrics.get("mismatch@0.25")
    if base and al:
        print(f"sensitivity(0.25) = {sens.get('aligned@0.25', 0):.5f} "
              f"({'RESPONDS' if sens.get('aligned@0.25',0) > 0.002 else 'weak/no response'})")
        print(f"aligned dAP75 = {al['AP75']-base['AP75']:+.4f} | "
              f"random dAP75 = {(rnd['AP75']-base['AP75']) if rnd else 0:+.4f} | "
              f"mismatch dAP75 = {(mis['AP75']-base['AP75']) if mis else 0:+.4f}")
        print(f"aligned dBF1 = {bf1_by_setting['aligned@0.25']-bf1_by_setting['aligned@0.0']:+.4f}")
        peak = max((metrics[k]['AP75'] - base['AP75'], k) for k in metrics
                   if k.startswith('aligned@') and metrics[k])
        print(f"best aligned AP75 gain: {peak[0]:+.4f} at {peak[1]}")


from mmdet.structures.mask import BitmapMasks as BitmapMasks__  # noqa

if __name__ == "__main__":
    main()
