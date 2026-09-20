"""P6: injection-site causal footprint audit (frozen single-stream checkpoint).

For each candidate consumption point of boundary information, apply the SAME
GT-derived oracle boundary field (per-site calibrated) and measure how much
AP75 / AP / boundary F1 respond. No training; ranks injection sites by causal
metric footprint, to decide where the E5-confirmed BEV boundary signal can
actually be consumed.

Sites:
  P2      neck P2 map (256x256x512)      -- P5 reference site
  EMB     image embedding (64x64x256)    -- SAM2 decoder canvas + shape ctx
  COARSE  post-refiner shape logits      -- point mining + dense prompt driver
  COARSEmis  mismatched sdf (control)

Perturbation: x' = x + alpha * sigma_site * sdf_z(res) * s_c, where s_c is the
per-channel sign of corr(x_c, sdf_z) estimated on 20 train scenes (P2/EMB), and
for COARSE the sdf is RoI-cropped per proposal (alpha * logits-std * sdf_z).
"""
import argparse
import io
import contextlib
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
from mmdet.structures.mask import BitmapMasks
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

if "DetDataPreprocessor" not in MMENGINE_MODELS:
    MMENGINE_MODELS.register_module(name="DetDataPreprocessor", module=DetDataPreprocessor)

from data.ue_dual_stream_dataset import UEDualStreamDataset


def sdf_field(occ: np.ndarray) -> np.ndarray:
    d_out = cv2.distanceTransform(1 - occ, cv2.DIST_L2, 3)
    d_in = cv2.distanceTransform(occ, cv2.DIST_L2, 3)
    s = d_out - d_in
    return (s - s.mean()) / (s.std() + 1e-6)


def occ_at(item, res: int) -> np.ndarray:
    occ = np.zeros((res, res), np.uint8)
    for m in item["gt_masks"]:
        occ = np.maximum(occ, cv2.resize(m, (res, res), interpolation=cv2.INTER_NEAREST))
    return occ


def random_field(rng):
    f = cv2.GaussianBlur(rng.standard_normal((256, 256)).astype(np.float32), (0, 0), 4)
    return (f - f.mean()) / (f.std() + 1e-6)


def boundary_f1(pred, gt, tol=3):
    k = np.ones((3, 3), np.uint8)
    def bnd(m): return (m.astype(np.uint8) ^ cv2.erode(m.astype(np.uint8), k)) > 0
    pb, gb = bnd(pred), bnd(gt)
    if pb.sum() == 0 and gb.sum() == 0: return 1.0
    d = np.ones((2 * tol + 1,) * 2, np.uint8)
    pd = cv2.dilate(pb.astype(np.uint8), d) > 0
    gd = cv2.dilate(gb.astype(np.uint8), d) > 0
    p = (pb & gd).sum() / max(pb.sum(), 1)
    r = (gb & pd).sum() / max(gb.sum(), 1)
    return 2 * p * r / max(p + r, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single-ckpt", default="work_dirs/ue/ue_single_c5v2_control_tr128_va259_s44_r1/best_model_epoch86.pth")
    ap.add_argument("--config", default="configs/ue_dual_stream_p2p3_sparse.py")
    ap.add_argument("--data-root", default="/data1/wangcheng/dataset/ue_coco_with_uav")
    ap.add_argument("--n-calib", type=int, default=20)
    ap.add_argument("--n-val", type=int, default=40)
    ap.add_argument("--out", default="outputs/probe_p6_footprint")
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

    def forward_inputs(item):
        ds = DetDataSample(); ds.set_metainfo(item["img_metas"])
        ds.set_metainfo({"drone_images": item["drone_images"], "intrinsics": item["intrinsics"],
                         "extrinsics": item["extrinsics"], "scene_center": item["scene_center"]})
        proc = model.data_preprocessor(
            {"inputs": [item["img"].to(device)], "data_samples": [ds.to(device)]}, training=False)
        inputs = proc["inputs"][0] if isinstance(proc["inputs"], (list, tuple)) else proc["inputs"]
        return inputs, proc["data_samples"]

    # ---- calibrate per-channel signs for P2 and EMB on train scenes ----
    p2_sign = None; emb_sign = None; n_c = 0
    with torch.no_grad():
        for idx in range(args.n_calib):
            item = train_ds[idx]
            if len(item["gt_labels"]) == 0: continue
            inputs, dss = forward_inputs(item)
            vo = model.backbone(inputs)
            x = model.neck(vo[1])
            emb = model._select_sam_image_embeddings(vo, inputs)[0].float()
            s256 = sdf_field(occ_at(item, 256)); s64 = sdf_field(occ_at(item, 64))
            a2 = x[0][0].float().reshape(512, -1).cpu().numpy(); b2 = s256.reshape(-1)
            c2 = (a2 - a2.mean(1, keepdims=True)) @ b2 / (np.linalg.norm(a2, axis=1) * np.linalg.norm(b2) + 1e-9)
            ae = emb.reshape(256, -1).cpu().numpy(); be = s64.reshape(-1)
            ce = (ae - ae.mean(1, keepdims=True)) @ be / (np.linalg.norm(ae, axis=1) * np.linalg.norm(be) + 1e-9)
            p2_sign = c2 if p2_sign is None else p2_sign + c2
            emb_sign = ce if emb_sign is None else emb_sign + ce
            n_c += 1
    p2_sign = np.sign(p2_sign / n_c).astype(np.float32); p2_sign[p2_sign == 0] = 1
    emb_sign = np.sign(emb_sign / n_c).astype(np.float32); emb_sign[emb_sign == 0] = 1
    p2_sign_t = torch.from_numpy(p2_sign).to(device)[:, None, None]
    emb_sign_t = torch.from_numpy(emb_sign).to(device)[:, None, None]
    print(f"calibration on {n_c} scenes done", flush=True)

    # ---- perturbation hooks ----
    state = dict(site=None, alpha=0.0, field256=None, field64=None)
    orig_neck = model.neck.forward
    def neck_forward(hidden):
        x = orig_neck(hidden)
        if state["site"] == "P2" and state["alpha"] > 0:
            x0 = x[0].float()
            sigma = x0.flatten(2).std(dim=2).mean()[None, None, None]
            x = list(x)
            x[0] = (x0 + state["alpha"] * sigma * state["field256"][None].to(x0) * p2_sign_t[None]).to(x[0].dtype)
            x = tuple(x)
        return x
    model.neck.forward = neck_forward

    orig_extract = model.extract_feat
    def extract_forward(batch_inputs, batch_data_samples=None):
        x, emb, pos, hrf = orig_extract(batch_inputs, batch_data_samples)
        if state["site"] == "EMB" and state["alpha"] > 0:
            sigma = emb.float().flatten(2).std(dim=2).mean()[None, None, None]
            emb = emb.float() + state["alpha"] * sigma * state["field64"][None].to(emb) * emb_sign_t[None]
            emb = emb.to(x[0].dtype)
        return x, emb, pos, hrf
    model.extract_feat = extract_forward

    refiner = model.roi_head.mask_head.p2_boundary_refiner
    orig_ref_forward = refiner.forward
    def ref_forward(p2_feature, prompt_rois, raw_logits):
        out = orig_ref_forward(p2_feature, prompt_rois, raw_logits)
        if state["site"] == "COARSE" and state["alpha"] > 0 and prompt_rois.shape[0]:
            sdf_img = state["field256"][None, None].to(out["refined_logits"])
            crops = refiner.roi_align(sdf_img, prompt_rois)
            crops64 = F.interpolate(crops, (64, 64), mode="bilinear", align_corners=False)
            z = (crops64 - crops64.mean()) / (crops64.std() + 1e-6)
            s = out["refined_logits"].float().std() + 1e-6
            out["refined_logits"] = out["refined_logits"] + (
                state["alpha"] * s * z[0]).to(out["refined_logits"].dtype)
        return out
    refiner.forward = ref_forward

    # ---- GT cache (reuse P5's if present) ----
    gt_path = Path("outputs/probe_decoder_boundary_response/gt_subset.json")
    if gt_path.exists():
        coco_gt = COCO(str(gt_path))
    else:
        gt_imgs, gt_anns = [], []
        aid = 1
        for i in range(n_val):
            item = val_ds[i]
            gt_imgs.append(dict(id=i, file_name=item["img_metas"]["scene_id"], height=1024, width=1024))
            for m in item["gt_masks"]:
                rle = maskUtils.encode(np.asfortranarray(m)); rle["counts"] = rle["counts"].decode()
                ys, xs = np.where(m)
                gt_anns.append(dict(id=aid, image_id=i, category_id=1, segmentation=rle,
                                    bbox=[float(xs.min()), float(ys.min()), float(xs.max()-xs.min()+1), float(ys.max()-ys.min()+1)],
                                    area=int(m.sum()), iscrowd=0))
                aid += 1
        gp = out / "gt_subset.json"
        json.dump(dict(images=gt_imgs, annotations=gt_anns, categories=[dict(id=1, name="building")]), gp.open("w"))
        coco_gt = COCO(str(gp))

    settings = [("P2", 1.0), ("EMB", 0.5), ("EMB", 1.0),
                ("COARSE", 0.5), ("COARSE", 1.0), ("COARSEmis", 1.0)]

    def run_setting(kind, alpha):
        coco_dt, bf1s = [], []
        with torch.no_grad():
            for i in range(n_val):
                item = val_ds[i]
                if kind in ("P2", "EMB"):
                    state["site"], state["alpha"] = kind, alpha
                    state["field256"] = torch.from_numpy(sdf_field(occ_at(item, 256)).astype(np.float32))
                    state["field64"] = torch.from_numpy(sdf_field(occ_at(item, 64)).astype(np.float32))
                elif kind in ("COARSE", "COARSEmis"):
                    state["site"], state["alpha"] = "COARSE", alpha
                    j = (i + 7) % n_val if kind == "COARSEmis" else i
                    state["field256"] = torch.from_numpy(sdf_field(occ_at(val_ds[j], 256)).astype(np.float32))
                inputs, dss = forward_inputs(item)
                outs = model.predict(inputs, dss, rescale=False)
                state["site"], state["alpha"] = None, 0.0
                inst = outs[0].pred_instances
                if inst is not None and len(inst.bboxes):
                    msk = inst.masks
                    pmasks = msk.to_ndarray() if isinstance(msk, BitmapMasks) else msk.cpu().numpy()
                    scs = inst.scores.cpu().numpy()
                    keep = scs >= 0.5
                    pm = pmasks[keep].any(0) if keep.any() else np.zeros((1024, 1024), bool)
                    for m, s in zip(pmasks, scs):
                        rle = maskUtils.encode(np.asfortranarray(m.astype(np.uint8)))
                        rle["counts"] = rle["counts"].decode()
                        coco_dt.append(dict(image_id=i, category_id=1, segmentation=rle, score=float(s)))
                else:
                    pm = np.zeros((1024, 1024), bool)
                gtocc = np.zeros((1024, 1024), np.uint8)
                for m in item["gt_masks"]:
                    gtocc = np.maximum(gtocc, m)
                bf1s.append(boundary_f1(pm, gtocc > 0))
        metrics = dict(boundary_f1=float(np.mean(bf1s)))
        if coco_dt:
            dt = coco_gt.loadRes(coco_dt)
            ev = COCOeval(coco_gt, dt, "segm"); ev.params.imgIds = list(range(n_val))
            ev.evaluate(); ev.accumulate()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf): ev.summarize()
            metrics.update(AP=float(ev.stats[0]), AP50=float(ev.stats[1]), AP75=float(ev.stats[2]))
        return metrics

    base = run_setting("P2", 0.0)
    print(f"baseline (site=None): {json.dumps(base)}", flush=True)
    rows = {}
    for kind, alpha in settings:
        m = run_setting(kind, alpha)
        rows[f"{kind}@{alpha}"] = m
        print(f"{kind}@{alpha}: AP={m.get('AP', float('nan')):.4f} AP75={m.get('AP75', float('nan')):.4f} "
              f"AP50={m.get('AP50', float('nan')):.4f} BF1={m['boundary_f1']:.4f}", flush=True)

    print("\n===== footprint summary (delta vs baseline) =====")
    for k, m in rows.items():
        print(f"{k:14s} dAP={m.get('AP',0)-base['AP']:+.4f} dAP75={m.get('AP75',0)-base['AP75']:+.4f} "
              f"dBF1={m['boundary_f1']-base['boundary_f1']:+.4f}")
    json.dump(dict(baseline=base, rows=rows, n_val=n_val,
                   lineage=dict(ckpt=args.single_ckpt)),
              (out / "summary.json").open("w"), indent=1)
    print("saved", out / "summary.json")


if __name__ == "__main__":
    main()
