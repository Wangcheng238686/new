"""MHBEV-FUS-20260920-A/B: full-data, full-co-adaptation P2-injection fine-tune.

Direct causal metric test for the two largest confounds of the prior
screenings (frozen detector in H1/H2, starved zero-gate in D2/C30):
  - FULL train data (712 scenes),
  - FULL detector co-adaptation (LoRA + neck + RPN/ROI heads unfrozen),
  - P2 injection, least-squares warm-fitted conv (E5-style), warm gamma 0.25,
  - arms identical except the injection (--inject 1 = B, --inject 0 = A).

Per-epoch 259-val segm mAP/AP75/boundary-F1; B-vs-A paired comparison is the
metric-gain readout.
"""
import argparse
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
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

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
                if n_fit >= 80: continue
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

    orig_extract = model.extract_feat
    def extract_with_inject(batch_inputs, batch_data_samples=None):
        x, emb, pos, hrf = orig_extract(batch_inputs, batch_data_samples)
        bev = getattr(model.geometry_bev, "_last_out", None)
        if inject["enabled"] and bev is not None:
            delta = gamma * injector(bev.float())
            delta = F.interpolate(delta, size=x[0].shape[-2:], mode="bilinear", align_corners=False)
            x = list(x)
            x[0] = (x[0].float() + delta).to(x[0].dtype)
            x = tuple(x)
        return x, emb, pos, hrf
    model.extract_feat = extract_with_inject

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
            ds.gt_instances = gi
            _attach_dual_stream_metainfo(ds, batch, i)
            data_samples.append(ds.to(device))
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

    def eval_epoch():
        model.eval()
        coco_dt, bf1s = [], []
        with torch.no_grad():
            for i in range(len(val_ds)):
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
                    pm = pmasks[keep].any(0) if keep.any() else np.zeros((1024, 1024), bool)
                else:
                    pm = np.zeros((1024, 1024), bool)
                gtocc = np.zeros((1024, 1024), np.uint8)
                for m in item["gt_masks"]:
                    gtocc = np.maximum(gtocc, m)
                bf1s.append(boundary_f1(pm, gtocc > 0))
        metrics = dict(boundary_f1=float(np.mean(bf1s)))
        if coco_dt:
            dt = coco_handle().loadRes(coco_dt)
            ev = COCOeval(coco_handle(), dt, "segm")
            ev.params.imgIds = list(range(len(val_ds)))
            ev.evaluate(); ev.accumulate()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf): ev.summarize()
            metrics.update(AP=float(ev.stats[0]), AP50=float(ev.stats[1]), AP75=float(ev.stats[2]),
                           AP_s=float(ev.stats[3]), AP_m=float(ev.stats[4]), AP_l=float(ev.stats[5]))
        model.train()
        return metrics, bf1s

    commit, dirty = git_info()
    manifest = dict(
        inject=bool(args.inject), git_commit=commit, git_dirty=dirty,
        seeds=dict(env=SEED, train_subset=44, fit=FIT_SEED),
        single_ckpt_sha256=sha256(args.single_ckpt), dual_ckpt_sha256=sha256(DUAL_CKPT),
        single_ckpt=args.single_ckpt, epochs=args.epochs, lr=args.lr, inj_lr=INJ_LR,
        tau=TAU, warm_fit_r2=(float(r2) if args.inject else None),
        train_scenes=len(train_ds), val_scenes=len(val_ds))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))

    history = []
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        inject["enabled"] = bool(args.inject)
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        skipped = 0
        for bi, batch in enumerate(train_loader):
            inputs, data_samples = make_batch(batch, device)
            with torch.amp.autocast("cuda"):
                loss_dict = model.loss(inputs, data_samples)
                loss_dict = _materialize_p2_boundary_refiner_loss(loss_dict, model, False)
            parts = [v for k, v in loss_dict.items()
                     if "loss" in k and isinstance(v, torch.Tensor)]
            loss = sum(parts) / GRAD_ACCUM
            if not loss.requires_grad:
                skipped += 1
                continue
            scaler.scale(loss).backward()
            if (bi + 1) % GRAD_ACCUM == 0 or (bi + 1) == len(train_loader):
                step += 1
                total_steps = args.epochs * math.ceil(len(train_loader) / GRAD_ACCUM)
                warm = min(1.0, (step + 1) / (WARMUP_STEPS + 1))
                cosine = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(step / total_steps, 1.0)))
                for g in base_groups:
                    g["lr"] = g["base"] * warm * cosine
                for i, g in enumerate(base_groups):
                    optimizer.param_groups[i]["lr"] = g["lr"]
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        ev, bf1s = eval_epoch()
        ev.update(epoch=epoch, train_seconds=round(time.time() - t0, 1), skipped=skipped)
        history.append(ev)
        json.dump(history, (out / "history.json").open("w"), indent=1)
        (out / f"per_scene_bf1_epoch{epoch}.json").write_text(json.dumps(bf1s))
        torch.save(dict(model=model.state_dict(), epoch=epoch,
                        metrics={k: v for k, v in ev.items() if k != "per_scene_bf1"}),
                   out / "last.pth")
        print(f"epoch {epoch}: BF1={ev['boundary_f1']:.4f} AP={ev['AP']:.4f} "
              f"AP75={ev['AP75']:.4f} skipped={skipped} ({ev['train_seconds']}s)", flush=True)

    print("arm complete", flush=True)


if __name__ == "__main__":
    main()
