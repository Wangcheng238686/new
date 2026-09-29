"""MHBEV-FUS-20260929-SZAB: small-instance-retasked P2-injection A/B.

MHBEV-FUS-20260920-A/B framework (finetune_p2_inject_ab.py) with the preregistered
small-building task redefinition applied to BOTH arms (Appendix V):
  - small-instance loss weighting w_k = 1/sqrt(mask pixel area @1024^2), applied to
    (i) the final instance mask loss (per-ROI weighted mean replacing the plain
    all-pixel mean; numerically identical to it when all w are equal) and
    (ii) the bbox head cls/box losses (positive proposals' label/bbox weights scaled
    by the assigned GT's w_k; negatives untouched; mmdet avg_factor untouched).
    RPN / coarse-mask / P2-boundary-refiner losses stay unweighted.
  - primary readout: last-epoch 259-val segm AP_s; per-scene small-instance
    mask-F1@0.5 (both sides area<1024px^2, pred score>=0.5, IoU>=0.5 greedy
    one-to-one) feeds the paired bootstrap (10k, seed 20260936).
Arms identical except the injection (--inject 1 = B, --inject 0 = A).
Registered gates: EXPERIMENT_GOVERNANCE.md Appendix V (SZAB_INJECT_PASS).
"""
import argparse
import hashlib
import io
import contextlib
import json
import math
import os
import subprocess
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
os.environ.setdefault("UE_DUAL_INJECT_EMBEDDING", "0")

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
from mmengine.structures import InstanceData
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

if "DetDataPreprocessor" not in MMENGINE_MODELS:
    MMENGINE_MODELS.register_module(name="DetDataPreprocessor", module=DetDataPreprocessor)

from data.ue_dual_stream_dataset import UEDualStreamDataset, ue_dual_collate
from torch.utils.data import DataLoader
from train.train_rsprompter_fusion import (
    _attach_dual_stream_metainfo,
    _materialize_p2_boundary_refiner_loss,
)

SINGLE_CKPT = "work_dirs/ue/ue_single_c5v2_control_tr128_va259_s44_r1/best_model_epoch86.pth"
DUAL_CKPT = "work_dirs/ue/ue_dual_c5v2_p2p3_emb_tr128_va259_s44_r1/best_model_epoch85.pth"
EPOCHS = 6
LR = 1e-4
INJ_LR = 1e-3
WD = 0.01
GRAD_ACCUM = 8
WARMUP_STEPS = 40
FIT_SEED = 20260925
TAU = 1.0  # full fitted field at init for the fine-tune arm


def sdf_field(occ: np.ndarray) -> np.ndarray:
    d_out = cv2.distanceTransform(1 - occ, cv2.DIST_L2, 3)
    d_in = cv2.distanceTransform(occ, cv2.DIST_L2, 3)
    s = d_out - d_in
    return (s - s.mean()) / (s.std() + 1e-6)


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


def git_info():
    root = str(project_root)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, cwd=root).stdout.strip()
    dirty = subprocess.run(["git", "status", "--short"], capture_output=True,
                           text=True, cwd=root).stdout
    return commit, dirty


def sha256(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inject", type=int, choices=[0, 1], required=True)
    ap.add_argument("--config", default="configs/ue_dual_stream_p2p3_sparse.py")
    ap.add_argument("--data-root", default="/data1/wangcheng/dataset/ue_coco_with_uav")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--single-ckpt", default=SINGLE_CKPT)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-train-batches", type=int, default=0,
                    help="smoke only: cap train batches per epoch (0 = full)")
    ap.add_argument("--limit-val-scenes", type=int, default=0,
                    help="smoke only: cap val scenes per eval (0 = full 259)")
    ap.add_argument("--mem-fraction", type=float, default=0.0,
                    help="optional torch.cuda.set_per_process_memory_fraction "
                         "cap when co-locating on a shared GPU (0 = uncapped)")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    if args.mem_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction, device.index or 0)
        print(f"cuda mem fraction capped at {args.mem_fraction}", flush=True)

    import random
    SEED = 20260928
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    cfg = Config.fromfile(args.config)
    model = MODELS.build(cfg.model)
    dual = torch.load(DUAL_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(dual.get("model", dual.get("state_dict")), strict=False)
    single = torch.load(args.single_ckpt, map_location="cpu", weights_only=False)
    single_sd = single.get("model", single.get("state_dict"))
    model_sd = model.state_dict()
    filtered = {k: v for k, v in single_sd.items()
                if k in model_sd and model_sd[k].shape == v.shape}
    dropped = [k for k in single_sd if k not in filtered]
    model.load_state_dict(filtered, strict=False)
    assert dropped == [], f"shape-mismatched single keys: {dropped}"
    model = model.to(device)
    assert model.roi_head.mask_head.p2_boundary_refiner.bev_channels == 0
    model.p2p3_gate.gamma_p2.data.zero_()
    model.p2p3_gate.gamma_p3.data.zero_()

    train_ds = UEDualStreamDataset(args.data_root, "train", image_size=(1024, 1024),
                                   train_subset=1.0, seed=44, load_uav=True)
    val_ds = UEDualStreamDataset(args.data_root, "val", image_size=(1024, 1024), load_uav=True)
    assert not (set(train_ds.ids) & set(val_ds.ids))
    print(f"train={len(train_ds)} val={len(val_ds)} inject={args.inject}", flush=True)

    orig_bev_forward = model.geometry_bev.forward
    def bev_forward(*a, **kw):
        b = orig_bev_forward(*a, **kw)
        model.geometry_bev._last_out = b
        return b
    model.geometry_bev.forward = bev_forward
    model.p2p3_gate.gamma_p2.data.zero_()
    model.p2p3_gate.gamma_p3.data.zero_()

    def forward_inputs(item):
        ds = DetDataSample(); ds.set_metainfo(item["img_metas"])
        ds.set_metainfo({"drone_images": item["drone_images"], "intrinsics": item["intrinsics"],
                         "extrinsics": item["extrinsics"], "scene_center": item["scene_center"]})
        proc = model.data_preprocessor(
            {"inputs": [item["img"].to(device)], "data_samples": [ds.to(device)]}, training=False)
        inputs = proc["inputs"][0] if isinstance(proc["inputs"], (list, tuple)) else proc["inputs"]
        return inputs, proc["data_samples"]

    def compute_feats(item):
        inputs, dss = forward_inputs(item)
        vo = model.backbone(inputs)
        x = model.neck(vo[1])
        drone = item["drone_images"][None].to(device)
        drone = (drone - drone.new_tensor([.485, .456, .406])[None, None, :, None, None]) / \
                drone.new_tensor([.229, .224, .225])[None, None, :, None, None]
        _, fpn = model.drone_encoder(drone.reshape(drone.shape[0] * drone.shape[1], *drone.shape[2:]))
        fpn = sorted(tuple(fpn[-4:]), key=lambda t: t.shape[-2] * t.shape[-1], reverse=True)
        bev = model.geometry_bev(x[1], fpn, item["intrinsics"][None].to(device),
                                 item["extrinsics"][None].to(device),
                                 item["scene_center"][None].to(device))[0].float()
        return x[0][0].float(), bev

    # ---- full detector trainable (drone encoder + geometry BEV stay frozen) ----
    frozen_prefixes = ("drone_encoder.", "geometry_bev.")
    for n, p in model.named_parameters():
        p.requires_grad_(not n.startswith(frozen_prefixes))

    injector = None; gamma = None
    if args.inject:
        injector = torch.nn.Conv2d(128, 512, 1, bias=False).to(device)
        gamma = torch.nn.Parameter(torch.tensor(0.25, device=device))
        model.injector = injector
        model.injector_gamma = gamma
        injector.weight.requires_grad_(True)
        gamma.requires_grad_(True)

    # ---- warm fit: calibrate on first 20 scenes, collect fit cells on next 80 ----
    if args.inject:
        p2_sign = None; sigma_bar = None; n_c = 0
        X, T_rows = [], []
        n_fit = 0
        with torch.no_grad():
            for idx in range(len(train_ds)):
                item = train_ds[idx]
                if len(item["gt_labels"]) == 0: continue
                x0, bev = compute_feats(item)
                occ = np.zeros((256, 256), np.uint8)
                for m in item["gt_masks"]:
                    occ = np.maximum(occ, cv2.resize(m, (256, 256), interpolation=cv2.INTER_NEAREST))
                if occ.sum() == 0: continue
                sdf256 = sdf_field(occ)
                sdf128 = cv2.resize(sdf256, (128, 128), interpolation=cv2.INTER_LINEAR)
                sdf128 = (sdf128 - sdf128.mean()) / (sdf128.std() + 1e-6)
                if n_c < 20:
                    a = x0.flatten(1).cpu().numpy(); b = sdf256.reshape(-1)
                    a = a - a.mean(1, keepdims=True)
                    c = (a @ b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b) + 1e-9)
                    p2_sign = c if p2_sign is None else p2_sign + c
                    s = x0.reshape(x0.shape[0], -1).std(dim=1)
                    sigma_bar = s if sigma_bar is None else sigma_bar + s
                    n_c += 1
                    continue
                if n_fit >= 80:
                    if n_c >= 20:
                        break   # remaining scenes only waste the feature forward;
                    continue    # the fit set (first 80) is already complete
                p2s = np.sign(p2_sign / n_c).astype(np.float32); p2s[p2s == 0] = 1
                sb = (sigma_bar / n_c).cpu().numpy() if torch.is_tensor(sigma_bar) else (sigma_bar / n_c)
                T = (sdf128.reshape(-1, 1) * (sb * p2s)[None, :]).astype(np.float32)  # [cells,512]
                B = bev.reshape(bev.shape[0], -1).T.cpu().numpy()  # [cells,128]
                rng = np.random.RandomState(FIT_SEED + idx)
                sel = rng.choice(T.shape[0], size=1500, replace=False)
                X.append(B[sel]); T_rows.append(T[sel]); n_fit += 1
        p2s = np.sign(p2_sign / n_c).astype(np.float32); p2s[p2s == 0] = 1
        sb = (sigma_bar / n_c).cpu().numpy() if torch.is_tensor(sigma_bar) else (sigma_bar / n_c)
        Xa = np.concatenate(X).astype(np.float64)
        Ta = np.concatenate(T_rows).astype(np.float64)
        lam = 1e-2 * float((Xa.T @ Xa).diagonal().mean())
        Wls = np.linalg.solve(Xa.T @ Xa + lam * np.eye(Xa.shape[1]), Xa.T @ Ta)
        injector.weight.data = torch.from_numpy(Wls.T.astype(np.float32)).view(512, 128, 1, 1).to(device)
        pred = Xa @ Wls
        r2 = 1 - ((Ta - pred) ** 2).sum() / ((Ta - Ta.mean(0)) ** 2).sum()
        print(f"warm fit: cells={len(Xa)} R2={r2:.4f}", flush=True)

    inject = dict(enabled=bool(args.inject))
    sz_engagement = dict(delta_sq=0.0, p2_sq=0.0, count=0)

    orig_extract = model.extract_feat
    def extract_with_inject(batch_inputs, batch_data_samples=None):
        x, emb, pos, hrf = orig_extract(batch_inputs, batch_data_samples)
        bev = getattr(model.geometry_bev, "_last_out", None)
        if inject["enabled"] and bev is not None:
            delta = gamma * injector(bev.float())
            delta = F.interpolate(delta, size=x[0].shape[-2:], mode="bilinear", align_corners=False)
            with torch.no_grad():
                if model.training:
                    sz_engagement["delta_sq"] += float(delta.detach().float().pow(2).sum())
                    sz_engagement["p2_sq"] += float(x[0].detach().float().pow(2).sum())
                    sz_engagement["count"] += 1
            x = list(x)
            x[0] = (x[0].float() + delta).to(x[0].dtype)
            x = tuple(x)
        return x, emb, pos, hrf
    model.extract_feat = extract_with_inject

    # ---- Appendix V small-instance loss reweighting (both arms) ----
    # w_k = 1/sqrt(GT mask pixel area at 1024^2). Per-image vector stashed on the
    # model in make_batch (batch_size=1); mask hook re-derives per-ROI weights from
    # batch_gt_instances, bbox hook uses the side channel (sampling_results carry
    # no GT references in this mmdet version).
    class SZWeightedMaskCE(torch.nn.Module):
        """Per-ROI weighted mean of the per-ROI pixel BCE, replacing mmdet's
        mask_cross_entropy plain all-pixel mean. Identical to it when all w equal."""

        def forward(self, mask_preds, mask_targets, pos_labels):
            n = mask_preds.size(0)
            if n == 0:
                return mask_preds.sum()[None]
            inds = torch.arange(n, device=mask_preds.device)
            pred_slice = mask_preds[inds, pos_labels].squeeze(1)
            per_roi = F.binary_cross_entropy_with_logits(
                pred_slice, mask_targets, reduction="none").mean(dim=(1, 2))
            w = self._sz_roi_w.to(device=per_roi.device, dtype=per_roi.dtype)
            if w.numel() != n:
                raise RuntimeError(f"sz weight/ROI mismatch: {w.numel()} vs {n}")
            denom = w.sum().clamp_min(1e-12)
            self._last_stats = dict(
                n_roi=n, sum_w=float(w.sum()),
                weighted=float((per_roi * w).sum() / denom),
                plain=float(per_roi.mean()))
            return ((per_roi * w).sum() / denom)[None]

    def sz_gt_area_w(gt_masks_np):
        areas = gt_masks_np.sum(axis=(1, 2)).astype(np.float64)
        return torch.from_numpy(1.0 / np.sqrt(np.maximum(areas, 1.0))).float()

    def sz_pos_roi_weights(sampling_results, batch_gt_instances):
        ws = []
        for res, gt in zip(sampling_results, batch_gt_instances):
            w_gt = gt.sz_gt_w.to(torch.float32)
            pos = res.pos_assigned_gt_inds.to(torch.long)
            ws.append(w_gt[pos] if pos.numel() else w_gt.new_zeros(0))
        return torch.cat(ws)

    mask_head = model.roi_head.mask_head
    sz_mask_ce = SZWeightedMaskCE()
    mask_head.loss_mask = sz_mask_ce
    sz_orig_mask_lat = mask_head.loss_and_target

    def sz_loss_and_target(mask_preds, sampling_results, batch_gt_instances,
                           rcnn_train_cfg, final_mask_pos_priors=None):
        sz_mask_ce._sz_roi_w = sz_pos_roi_weights(sampling_results, batch_gt_instances)
        return sz_orig_mask_lat(mask_preds=mask_preds,
                                sampling_results=sampling_results,
                                batch_gt_instances=batch_gt_instances,
                                rcnn_train_cfg=rcnn_train_cfg,
                                final_mask_pos_priors=final_mask_pos_priors)
    mask_head.loss_and_target = sz_loss_and_target

    bbox_head = model.roi_head.bbox_head
    sz_orig_get_targets = bbox_head.get_targets

    def sz_get_targets(sampling_results, rcnn_train_cfg, concat=True):
        labels, label_weights, bbox_targets, bbox_weights = sz_orig_get_targets(
            sampling_results, rcnn_train_cfg, concat=concat)
        ws = []
        for res in sampling_results:
            w_gt = model._sz_gt_w.to(torch.float32)
            pos = res.pos_assigned_gt_inds.to(torch.long)
            ws.append(w_gt[pos] if pos.numel() else w_gt.new_zeros(0))
        w_roi = torch.cat(ws).to(label_weights.device, label_weights.dtype)
        pos_mask = (labels >= 0) & (labels < bbox_head.num_classes)
        if int(pos_mask.sum()) != int(w_roi.numel()):
            raise RuntimeError(
                f"sz pos/weight mismatch: {int(pos_mask.sum())} pos vs {int(w_roi.numel())} w")
        scale = torch.ones_like(label_weights)
        scale[pos_mask] = w_roi
        return (labels, label_weights * scale,
                bbox_targets, bbox_weights * scale[:, None])
    bbox_head.get_targets = sz_get_targets

    sz_loss_stats = dict(weighted=0.0, plain=0.0, sum_w=0.0, n_roi=0, batches=0)

    def sz_reset_epoch_stats():
        sz_loss_stats.update(weighted=0.0, plain=0.0, sum_w=0.0, n_roi=0, batches=0)
        sz_engagement.update(delta_sq=0.0, p2_sq=0.0, count=0)

    if args.inject:
        base_groups = [
            dict(params=[p for n, p in model.named_parameters()
                         if p.requires_grad and not n.startswith(("injector", "injector_gamma"))],
                 lr=args.lr, weight_decay=WD, base=args.lr),
            dict(params=[p for n, p in model.named_parameters()
                         if p.requires_grad and n.startswith(("injector", "injector_gamma"))],
                 lr=INJ_LR, weight_decay=WD, base=INJ_LR),
        ]
    else:
        base_groups = [dict(params=[p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=WD, base=args.lr)]
    optimizer = torch.optim.AdamW(base_groups)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2,
                              collate_fn=ue_dual_collate, drop_last=False)
    print(f"optimizer groups: {len(base_groups)}, "
          f"trainable tensors: {sum(1 for p in model.parameters() if p.requires_grad)}", flush=True)

    def make_batch(batch, device):
        data_samples = []
        for i in range(len(batch["imgs"])):
            ds = DetDataSample()
            ds.set_metainfo(batch["img_metas"][i])
            gi = InstanceData()
            labels = batch["gt_labels"][i]
            valid = labels >= 0
            gi.bboxes = batch["gt_bboxes"][i][valid].to(device)
            gi.labels = labels[valid].to(device)
            masks = batch["gt_masks"][i][valid.cpu().numpy()]
            gi.masks = BitmapMasks(masks, *batch["img_metas"][i]["img_shape"])
            gi.sz_gt_w = sz_gt_area_w(masks)
            ds.gt_instances = gi
            _attach_dual_stream_metainfo(ds, batch, i)
            data_samples.append(ds.to(device))
        if len(data_samples) == 1:
            model._sz_gt_w = data_samples[0].gt_instances.sz_gt_w.to(device)
        imgs = batch["imgs"].to(device)
        proc = model.data_preprocessor({"inputs": imgs, "data_samples": data_samples},
                                       training=True)
        return (proc["inputs"][0] if isinstance(proc["inputs"], (list, tuple))
                else proc["inputs"]), proc["data_samples"]

    _coco = {}
    def coco_handle():
        if "gt" not in _coco:
            gp = out / "gt_val259.json"
            if not gp.exists():
                gt_imgs, gt_anns = [], []
                aid = 1
                for i in range(len(val_ds)):
                    item = val_ds[i]
                    gt_imgs.append(dict(id=i, file_name=item["img_metas"]["scene_id"],
                                        height=1024, width=1024))
                    for m in item["gt_masks"]:
                        rle = maskUtils.encode(np.asfortranarray(m))
                        rle["counts"] = rle["counts"].decode()
                        ys, xs = np.where(m)
                        gt_anns.append(dict(id=aid, image_id=i, category_id=1, segmentation=rle,
                                            bbox=[float(xs.min()), float(ys.min()),
                                                  float(xs.max()-xs.min()+1), float(ys.max()-ys.min()+1)],
                                            area=int(m.sum()), iscrowd=0))
                        aid += 1
                json.dump(dict(images=gt_imgs, annotations=gt_anns,
                               categories=[dict(id=1, name="building")]), gp.open("w"))
            _coco["gt"] = COCO(str(gp))
        return _coco["gt"]

    SMALL_AREA_PX = 32 * 32  # COCO small, in 1024^2-canvas pixels

    def small_instance_f1(pred_masks, pred_scores, gt_masks):
        """Appendix V per-scene small-instance readout: both sides restricted to
        area < 1024 px^2 (32x32), preds additionally score >= 0.5 (caller filters),
        greedy one-to-one IoU>=0.5 matching by descending score. Both-empty -> 1,
        one-side-empty -> 0."""
        gs = [m for m in gt_masks if int(m.sum()) < SMALL_AREA_PX]
        ps = [(m, s) for m, s in zip(pred_masks, pred_scores)
              if int(m.sum()) < SMALL_AREA_PX]
        ps.sort(key=lambda t: -float(t[1]))
        matched = 0
        used = [False] * len(gs)
        for pm, _ in ps:
            best, best_iou = -1, 0.0
            for k, gm in enumerate(gs):
                if used[k]:
                    continue
                inter = int(np.logical_and(pm, gm).sum())
                if inter == 0:
                    continue
                union = int(pm.sum()) + int(gm.sum()) - inter
                iou = inter / max(union, 1)
                if iou > best_iou:
                    best_iou, best = iou, k
            if best >= 0 and best_iou >= 0.5:
                used[best] = True
                matched += 1
        if not gs and not ps:
            f1 = 1.0
        elif not gs or not ps:
            f1 = 0.0
        else:
            prec = matched / len(ps)
            rec = matched / len(gs)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        return dict(f1=float(f1), n_gt_small=len(gs), n_pred_small=len(ps),
                    matched=matched)

    def eval_epoch():
        model.eval()
        coco_dt, bf1s, sf1s = [], [], []
        n_val = len(val_ds)
        if args.limit_val_scenes:
            n_val = min(args.limit_val_scenes, n_val)
        with torch.no_grad():
            for i in range(n_val):
                item = val_ds[i]
                ds = DetDataSample(); ds.set_metainfo(item["img_metas"])
                ds.set_metainfo({"drone_images": item["drone_images"],
                                 "intrinsics": item["intrinsics"],
                                 "extrinsics": item["extrinsics"],
                                 "scene_center": item["scene_center"]})
                proc = model.data_preprocessor(
                    {"inputs": [item["img"].to(device)], "data_samples": [ds.to(device)]},
                    training=False)
                inputs = (proc["inputs"][0] if isinstance(proc["inputs"], (list, tuple))
                          else proc["inputs"])
                outs = model.predict(inputs, proc["data_samples"], rescale=False)
                inst = outs[0].pred_instances
                kept_masks, kept_scores = [], []
                if inst is not None and len(inst.bboxes):
                    msk = inst.masks
                    pmasks = msk.to_ndarray() if isinstance(msk, BitmapMasks) else msk.cpu().numpy()
                    scs = inst.scores.cpu().numpy()
                    for m, s in zip(pmasks, scs):
                        rle = maskUtils.encode(np.asfortranarray(m.astype(np.uint8)))
                        rle["counts"] = rle["counts"].decode()
                        coco_dt.append(dict(image_id=i, category_id=1, segmentation=rle,
                                            score=float(s)))
                    keep = scs >= 0.5
                    if keep.any():
                        kept_masks = [mm.astype(bool) for mm in pmasks[keep]]
                        kept_scores = scs[keep].tolist()
                        pm = pmasks[keep].any(0)
                    else:
                        pm = np.zeros((1024, 1024), bool)
                else:
                    pm = np.zeros((1024, 1024), bool)
                gtocc = np.zeros((1024, 1024), np.uint8)
                for m in item["gt_masks"]:
                    gtocc = np.maximum(gtocc, m)
                bf1s.append(boundary_f1(pm, gtocc > 0))
                sf1s.append(dict(scene=i, **small_instance_f1(kept_masks, kept_scores,
                                                              list(item["gt_masks"]))))
        metrics = dict(boundary_f1=float(np.mean(bf1s)),
                       small_f1=float(np.mean([r["f1"] for r in sf1s])))
        if coco_dt:
            dt = coco_handle().loadRes(coco_dt)
            ev = COCOeval(coco_handle(), dt, "segm")
            ev.params.imgIds = list(range(n_val))
            ev.evaluate(); ev.accumulate()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf): ev.summarize()
            metrics.update(AP=float(ev.stats[0]), AP50=float(ev.stats[1]), AP75=float(ev.stats[2]),
                           AP_s=float(ev.stats[3]), AP_m=float(ev.stats[4]), AP_l=float(ev.stats[5]))
        model.train()
        return metrics, bf1s, sf1s

    commit, dirty = git_info()
    # Appendix V.5.2 calibration: hash of the initial state over keys shared by
    # both arms (injector/gamma excluded) — must be identical across A and B.
    _shared_state = {k: v for k, v in model.state_dict().items()
                     if not k.startswith(("injector", "injector_gamma"))}
    _h = hashlib.sha256()
    for k in sorted(_shared_state):
        _h.update(k.encode())
        _h.update(np.ascontiguousarray(_shared_state[k].detach().cpu().numpy()).tobytes())
    manifest = dict(
        inject=bool(args.inject), git_commit=commit, git_dirty=dirty,
        experiment="MHBEV-FUS-20260929-SZAB", arm=("B" if args.inject else "A"),
        init_state_hash_shared=_h.hexdigest(),
        sz_reweight=dict(formula="w_k = 1/sqrt(max(gt_mask_pixel_area@1024^2, 1))",
                         scope=["final_instance_mask_loss", "bbox_head_cls_box"],
                         unweighted=["rpn", "coarse_mask_aux", "p2_boundary_refiner"]),
        boot_seed=20260936, small_area_px=SMALL_AREA_PX,
        seeds=dict(env=SEED, train_subset=44, fit=FIT_SEED),
        single_ckpt_sha256=sha256(args.single_ckpt), dual_ckpt_sha256=sha256(DUAL_CKPT),
        single_ckpt=args.single_ckpt, epochs=args.epochs, lr=args.lr, inj_lr=INJ_LR,
        tau=TAU, warm_fit_r2=(float(r2) if args.inject else None),
        train_scenes=len(train_ds), val_scenes=len(val_ds),
        limit_train_batches=args.limit_train_batches,
        limit_val_scenes=args.limit_val_scenes)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))

    history = []
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        inject["enabled"] = bool(args.inject)
        sz_reset_epoch_stats()
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        skipped = 0
        n_batches = len(train_loader)
        if args.limit_train_batches:
            n_batches = min(args.limit_train_batches, n_batches)
        for bi, batch in enumerate(train_loader):
            if bi >= n_batches:
                break
            inputs, data_samples = make_batch(batch, device)
            sz_mask_ce._last_stats = None   # no stale carry-over across batches
            with torch.amp.autocast("cuda"):
                loss_dict = model.loss(inputs, data_samples)
                loss_dict = _materialize_p2_boundary_refiner_loss(loss_dict, model, False)
            if getattr(sz_mask_ce, "_last_stats", None):
                st = sz_mask_ce._last_stats
                sz_loss_stats["weighted"] += st["weighted"]
                sz_loss_stats["plain"] += st["plain"]
                sz_loss_stats["sum_w"] += st["sum_w"]
                sz_loss_stats["n_roi"] += st["n_roi"]
                sz_loss_stats["batches"] += 1
            parts = [v for k, v in loss_dict.items()
                     if "loss" in k and isinstance(v, torch.Tensor)]
            loss = sum(parts) / GRAD_ACCUM
            if not loss.requires_grad:
                skipped += 1
                continue
            scaler.scale(loss).backward()
            if (bi + 1) % GRAD_ACCUM == 0 or (bi + 1) == n_batches:
                step += 1
                total_steps = args.epochs * math.ceil(n_batches / GRAD_ACCUM)
                warm = min(1.0, (step + 1) / (WARMUP_STEPS + 1))
                cosine = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(step / total_steps, 1.0)))
                for g in base_groups:
                    g["lr"] = g["base"] * warm * cosine
                for i, g in enumerate(base_groups):
                    optimizer.param_groups[i]["lr"] = g["lr"]
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        ev, bf1s, sf1s = eval_epoch()
        ev.update(epoch=epoch, train_seconds=round(time.time() - t0, 1), skipped=skipped)
        if sz_loss_stats["batches"]:
            ev["sz_mask_weighted_over_plain"] = (
                sz_loss_stats["weighted"] / sz_loss_stats["plain"])
            ev["sz_sum_w"] = sz_loss_stats["sum_w"]
            ev["sz_n_roi"] = sz_loss_stats["n_roi"]
            ev["sz_mean_pos_roi_w"] = (
                sz_loss_stats["sum_w"] / max(sz_loss_stats["n_roi"], 1))
        if args.inject:
            if gamma is not None:
                ev["inject_gamma"] = float(gamma.detach())
            if sz_engagement["count"]:
                # epoch mean-square ratio mean|delta|^2 / mean|P2|^2
                ev["inject_engagement_ms_ratio"] = (
                    (sz_engagement["delta_sq"] / sz_engagement["count"])
                    / max(sz_engagement["p2_sq"] / sz_engagement["count"], 1e-12))
        history.append(ev)
        json.dump(history, (out / "history.json").open("w"), indent=1)
        (out / f"per_scene_bf1_epoch{epoch}.json").write_text(json.dumps(bf1s))
        (out / f"per_scene_smallf1_epoch{epoch}.json").write_text(json.dumps(sf1s))
        torch.save(dict(model=model.state_dict(), epoch=epoch,
                        metrics={k: v for k, v in ev.items() if not k.startswith("per_scene")}),
                   out / "last.pth")
        print(f"epoch {epoch}: BF1={ev['boundary_f1']:.4f} AP={ev['AP']:.4f} "
              f"AP75={ev['AP75']:.4f} AP_s={ev.get('AP_s', float('nan')):.4f} "
              f"smallF1={ev['small_f1']:.4f} "
              f"sz_ratio={ev.get('sz_mask_weighted_over_plain', float('nan')):.3f} "
              f"skipped={skipped} ({ev['train_seconds']}s)", flush=True)

    print("arm complete", flush=True)


if __name__ == "__main__":
    main()
