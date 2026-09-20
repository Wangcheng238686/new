"""MHBEV-FUS-20260919-H2: BEV-to-P2 warm-fitted injection screening.

Preregistered in EXPERIMENT_GOVERNANCE.md Appendix M. Evidence chain:
  P6   -- P2 is the ONLY injection site with a positive boundary response;
  E2   -- zero gates cannot bootstrap (gradient starvation);
  H1   -- a random-init branch under detector loss does not discover the
          oracle direction (refiner site, flat);
  E4/E5-- the marginal boundary information exists and is linearly readable.

H2 injects at P2 with a least-squares warm-fitted conv: the injector weights
are fitted so that gamma*conv(BEV) reproduces the calibrated boundary field
(sigma_c * s_c * sdf_z) from step 0. Only the injector (conv 128->512 1x1 +
scalar gamma, init 1.0) is trainable; the detector is frozen.

Epoch-0 records two references: (a) pure baseline (gamma=0 forward = exact
single-stream) and (b) warm-injected forward. Gates at e in {4,8}: paired
bootstrap BF1(e)-BF1(a) CI lower > 0; AP75 delta >= -0.003; gamma deviates
>= 20% with active injection; invariants pass.
"""
import argparse
import io
import contextlib
import json
import math
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
os.environ.setdefault("UE_DUAL_INJECT_EMBEDDING", "0")  # H2: no embedding gate

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import torch
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
EPOCHS = 8
LR = 1e-3
WD = 0.01
GRAD_ACCUM = 8
WARMUP_STEPS = 20
BOOT_SEED = 20260927
FIT_SEED = 20260925
BOOT_DRAWS = 10000
TAU = 0.25  # warm-fit scale: init perturbation ~ P5 alpha=0.25 response zone


def sdf_field(occ: np.ndarray) -> np.ndarray:
    d_out = cv2.distanceTransform(1 - occ, cv2.DIST_L2, 3)
    d_in = cv2.distanceTransform(occ, cv2.DIST_L2, 3)
    s = d_out - d_in
    return (s - s.mean()) / (s.std() + 1e-6)


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tol: int = 3) -> float:
    k = np.ones((3, 3), np.uint8)
    def bnd(m):
        return (m.astype(np.uint8) ^ cv2.erode(m.astype(np.uint8), k)) > 0
    pb, gb = bnd(pred), bnd(gt)
    if pb.sum() == 0 and gb.sum() == 0:
        return 1.0
    d = np.ones((2 * tol + 1,) * 2, np.uint8)
    pd = cv2.dilate(pb.astype(np.uint8), d) > 0
    gd = cv2.dilate(gb.astype(np.uint8), d) > 0
    prec = (pb & gd).sum() / max(pb.sum(), 1)
    rec = (gb & pd).sum() / max(gb.sum(), 1)
    return 2 * prec * rec / max(prec + rec, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ue_dual_stream_p2p3_sparse.py")
    ap.add_argument("--data-root", default="/data1/wangcheng/dataset/ue_coco_with_uav")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--out", default="/data1/wangcheng/checkpoint/worktrees_dual-stream-dev_probe/MHBEV-FUS-20260919-H2-screen")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    import random
    SEED = 20260928
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    cfg = Config.fromfile(args.config)
    model = MODELS.build(cfg.model)
    dual = torch.load(DUAL_CKPT, map_location="cpu", weights_only=False)
    m1 = model.load_state_dict(dual.get("model", dual.get("state_dict")), strict=False)
    single = torch.load(SINGLE_CKPT, map_location="cpu", weights_only=False)
    single_sd = single.get("model", single.get("state_dict"))
    model_sd = model.state_dict()
    filtered = {k: v for k, v in single_sd.items()
                if k in model_sd and model_sd[k].shape == v.shape}
    dropped = [k for k in single_sd if k not in filtered]
    m2 = model.load_state_dict(filtered, strict=False)
    print(f"dual load: missing={sorted(m1.missing_keys)}")
    print(f"dual load unexpected={sorted(m1.unexpected_keys)}")
    print(f"single overwrite: matched={len(filtered)} dropped(shape)={dropped}")
    assert dropped == [], f"shape-mismatched single keys: {dropped}"
    model = model.to(device)
    refiner = model.roi_head.mask_head.p2_boundary_refiner
    assert refiner.bev_channels == 0, "H2 must not enable the refiner BEV branch"
    assert not hasattr(model, "embedding_gate") or model.embedding_gate is None

    train_ds = UEDualStreamDataset(args.data_root, "train", image_size=(1024, 1024),
                                   train_subset=0.1797752808988764, seed=44, load_uav=True)
    val_ds = UEDualStreamDataset(args.data_root, "val", image_size=(1024, 1024), load_uav=True)
    assert not (set(train_ds.ids) & set(val_ds.ids)), "train/val scene overlap"

    # stash the BEV computed inside the dual extract_feat
    orig_bev_forward = model.geometry_bev.forward
    def bev_forward(*a, **kw):
        b = orig_bev_forward(*a, **kw)
        model.geometry_bev._last_out = b
        return b
    model.geometry_bev.forward = bev_forward
    # pyramid gates stay exactly zero and frozen
    model.p2p3_gate.gamma_p2.data.zero_()
    model.p2p3_gate.gamma_p3.data.zero_()

    # ---- H2 injector ----
    injector = torch.nn.Conv2d(128, 512, 1, bias=False).to(device)
    gamma = torch.nn.Parameter(torch.ones((), device=device))
    model.injector = injector
    model.injector_gamma = gamma

    def forward_inputs(item):
        ds = DetDataSample(); ds.set_metainfo(item["img_metas"])
        ds.set_metainfo({"drone_images": item["drone_images"], "intrinsics": item["intrinsics"],
                         "extrinsics": item["extrinsics"], "scene_center": item["scene_center"]})
        proc = model.data_preprocessor(
            {"inputs": [item["img"].to(device)], "data_samples": [ds.to(device)]}, training=False)
        inputs = proc["inputs"][0] if isinstance(proc["inputs"], (list, tuple)) else proc["inputs"]
        return inputs, proc["data_samples"]

    def compute_feats(item):
        """Backbone + neck + frozen drone encoder + BEV. Returns (x0, bev)."""
        inputs, dss = forward_inputs(item)
        vo = model.backbone(inputs)
        x = model.neck(vo[1])
        drone = item["drone_images"][None].to(device)  # [1,5,3,512,512] RGB 0-1
        drone = (drone - drone.new_tensor([.485, .456, .406])[None, None, :, None, None]) / \
                drone.new_tensor([.229, .224, .225])[None, None, :, None, None]
        _, fpn = model.drone_encoder(drone.reshape(drone.shape[0] * drone.shape[1],
                                                   *drone.shape[2:]))
        fpn = sorted(tuple(fpn[-4:]), key=lambda t: t.shape[-2] * t.shape[-1], reverse=True)
        bev = model.geometry_bev(x[1], fpn, item["intrinsics"][None].to(device),
                                 item["extrinsics"][None].to(device),
                                 item["scene_center"][None].to(device))[0].float()
        return x[0][0].float(), bev  # [512,256,256], [128,128,128]

    # ---- pass 1: channel signs + sigma_bar on 20 train scenes ----
    p2_sign = None; sigma_bar = None; n_c = 0
    with torch.no_grad():
        for idx in range(len(train_ds)):
            if n_c >= 20: break
            item = train_ds[idx]
            if len(item["gt_labels"]) == 0: continue
            x0, _ = compute_feats(item)
            occ = np.zeros((256, 256), np.uint8)
            for m in item["gt_masks"]:
                occ = np.maximum(occ, cv2.resize(m, (256, 256), interpolation=cv2.INTER_NEAREST))
            if occ.sum() == 0: continue
            sdf = sdf_field(occ)
            a = x0.flatten(1).cpu().numpy(); b = sdf.reshape(-1)
            a = a - a.mean(1, keepdims=True)
            c = (a @ b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b) + 1e-9)
            p2_sign = c if p2_sign is None else p2_sign + c
            s = x0.flatten(2).std(dim=1)
            sigma_bar = s if sigma_bar is None else sigma_bar + s
            n_c += 1
    p2_sign = np.sign(p2_sign / n_c).astype(np.float32); p2_sign[p2_sign == 0] = 1
    sigma_bar = (sigma_bar / n_c).cpu().numpy()  # [512]
    print(f"calibration on {n_c} scenes done", flush=True)

    # ---- pass 2: warm-fit cells on up to 80 train scenes (1,500 cells/scene) ----
    X, T_rows = [], []
    n_fit = 0
    with torch.no_grad():
        for idx in range(len(train_ds)):
            if n_fit >= 80: break
            item = train_ds[idx]
            if len(item["gt_labels"]) == 0: continue
            x0, bev = compute_feats(item)
            occ = np.zeros((256, 256), np.uint8)
            for m in item["gt_masks"]:
                occ = np.maximum(occ, cv2.resize(m, (256, 256), interpolation=cv2.INTER_NEAREST))
            if occ.sum() == 0: continue
            # grid choice (fixed, recorded): the fit runs on the NATIVE BEV grid
            # 128x128 (= satellite P3 space); the sdf field is resampled to it.
            sdf128 = cv2.resize(sdf_field(occ), (128, 128), interpolation=cv2.INTER_LINEAR)
            sdf128 = (sdf128 - sdf128.mean()) / (sdf128.std() + 1e-6)
            T = (sigma_bar[:, None] * p2_sign[:, None] * sdf128.reshape(1, -1)).astype(np.float32)  # [cells,512]
            B = bev.reshape(bev.shape[0], -1).T.cpu().numpy()  # [cells,128]
            rng = np.random.RandomState(FIT_SEED + idx)
            sel = rng.choice(T.shape[0], size=1500, replace=False)
            X.append(B[sel]); T_rows.append(T[sel])
            n_fit += 1
    X = np.concatenate(X).astype(np.float64)
    T_all = np.concatenate(T_rows).astype(np.float64)
    lam = 1e-2 * float((X.T @ X).diagonal().mean())
    Wls = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ T_all)  # [128,512]
    r2_unscaled = 1 - ((T_all - X @ Wls) ** 2).sum() / ((T_all - T_all.mean(0)) ** 2).sum()
    r2_scaled = 1 - ((T_all - X @ Wls * TAU) ** 2).sum() / ((T_all - T_all.mean(0)) ** 2).sum()
    print(f"warm fit: cells={len(X)} R2_unscaled={r2_unscaled:.4f} R2_scaled(tau={TAU})={r2_scaled:.4f}",
          flush=True)

    injector.weight.data = torch.from_numpy(
        (Wls * TAU).T.astype(np.float32)).view(512, 128, 1, 1).to(device)
    gamma = torch.nn.Parameter(torch.ones((), device=device))
    model.injector = injector
    model.injector_gamma = gamma

    # freeze everything, train only injector.weight + gamma
    for p in model.parameters():
        p.requires_grad_(False)
    model.injector.weight.requires_grad_(True)
    model.injector_gamma.requires_grad_(True)
    trainable = [model.injector.weight, model.injector_gamma]
    n_train_par = sum(p.numel() for p in trainable)
    print(f"trainable params: {n_train_par} in {len(trainable)} tensors", flush=True)

    # injection hook: applied to P2 after each extract_feat
    inject = dict(enabled=False)
    orig_extract = model.extract_feat
    def extract_forward(batch_inputs, batch_data_samples=None):
        x, emb, pos, hrf = orig_extract(batch_inputs, batch_data_samples)
        bev = getattr(model.geometry_bev, "_last_out", None)
        if inject["enabled"] and bev is not None:
            delta = gamma * injector(bev.float())                     # [1,512,128,128]
            delta = F.interpolate(delta, size=x[0].shape[-2:], mode="bilinear",
                                  align_corners=False)                # -> P2 256 res
            if model.training and torch.is_grad_enabled():
                ratio = float(delta.float().flatten(1).norm(dim=1).mean()
                              / x[0].float().flatten(1).norm(dim=1).mean().clamp_min(1e-9))
                getattr(model, "_engagement", []).append(ratio)
            x = list(x)
            x[0] = (x[0].float() + delta).to(x[0].dtype)
            x = tuple(x)
        return x, emb, pos, hrf
    model.extract_feat = extract_forward

    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WD)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    model._engagement = []
    import subprocess, hashlib
    def sha256(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        return h.hexdigest()
    manifest = dict(
        git_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                  text=True, cwd=project_root).stdout.strip(),
        git_dirty=subprocess.run(["git", "status", "--short"], capture_output=True,
                                 text=True, cwd=project_root).stdout,
        seeds=dict(global_env=SEED, train_subset=44, fit=FIT_SEED, bootstrap=BOOT_SEED),
        single_ckpt_sha256=sha256(SINGLE_CKPT), dual_ckpt_sha256=sha256(DUAL_CKPT),
        train_ids=list(train_ds.ids), val_ids=list(val_ds.ids),
        trainable_params=[n for n, _ in model.named_parameters()
                          if n.startswith(("injector", "injector_gamma"))],
        warm_fit=dict(tau=TAU, lam_rel=1e-2, cells=len(X), fit_seed=FIT_SEED,
                      r2_unscaled=float(r2_unscaled), r2_scaled=float(r2_scaled),
                      grid="fit on native BEV 128 grid; sdf resampled 256->128"),
        epochs=args.epochs, lr=LR, wd=WD, grad_accum=GRAD_ACCUM, warmup_steps=WARMUP_STEPS)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2,
                              collate_fn=ue_dual_collate, drop_last=False)
    print(f"train={len(train_ds)} val={len(val_ds)} epochs={args.epochs}", flush=True)

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

    _coco_gt = None
    def coco_gt_handle():
        nonlocal _coco_gt
        if _coco_gt is None:
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
            _coco_gt = COCO(str(gp))
        return _coco_gt

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
            dt = coco_gt_handle().loadRes(coco_dt)
            ev = COCOeval(coco_gt_handle(), dt, "segm")
            ev.params.imgIds = list(range(len(val_ds)))
            ev.evaluate(); ev.accumulate()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf): ev.summarize()
            metrics.update(AP=float(ev.stats[0]), AP50=float(ev.stats[1]), AP75=float(ev.stats[2]))
        return metrics, bf1s

    # ---- epoch-0 references ----
    print("===== epoch-0 (a): pure baseline (gamma=0 forward) =====", flush=True)
    saved_gamma = float(gamma)
    gamma.data.fill_(0.0)
    m_a, bf1_a = eval_epoch()
    print(f"pure baseline: BF1={m_a['boundary_f1']:.4f} AP={m_a['AP']:.4f} AP75={m_a['AP75']:.4f}",
          flush=True)
    base_bf1 = list(bf1_a)
    print("===== epoch-0 (b): warm-injected forward =====", flush=True)
    inject["enabled"] = True
    m_b, bf1_b = eval_epoch()
    (out / "per_scene_bf1_warm_injected.json").write_text(json.dumps(bf1_b))
    print(f"warm-injected: BF1={m_b['boundary_f1']:.4f} AP={m_b['AP']:.4f} AP75={m_b['AP75']:.4f}",
          flush=True)
    if m_b["boundary_f1"] == m_a["boundary_f1"] and bf1_b == bf1_a:
        raise RuntimeError("reference (b) identical to (a): warm injection is dead -- abort")
    history = [dict(epoch=0, mode="pure_baseline", **m_a, gamma=0.0),
               dict(epoch=0, mode="warm_injected", **m_b, gamma=float(gamma))]
    inject["enabled"] = True  # stay on for training
    json.dump(history, (out / "history.json").open("w"), indent=1)
    (out / "per_scene_bf1_pure_baseline.json").write_text(json.dumps(base_bf1))
    model.train()

    # ---- screening training ----
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        model._engagement = []
        skipped_no_grad = 0
        inject["enabled"] = True
        for bi, batch in enumerate(train_loader):
            inputs, data_samples = make_batch(batch, device)
            with torch.amp.autocast("cuda"):
                loss_dict = model.loss(inputs, data_samples)
                loss_dict = _materialize_p2_boundary_refiner_loss(loss_dict, model, False)
            parts = [v for k, v in loss_dict.items()
                     if "loss" in k and isinstance(v, torch.Tensor)]
            loss = sum(parts) / GRAD_ACCUM
            if not loss.requires_grad:
                # empty-GT batch (degenerate scene 002646): constant loss under
                # the frozen detector -- nothing to train on.
                skipped_no_grad += 1
                continue
            scaler.scale(loss).backward()
            if (bi + 1) % GRAD_ACCUM == 0 or (bi + 1) == len(train_loader):
                if not any(p.grad is not None for p in trainable):
                    continue
                step += 1
                total_steps = args.epochs * math.ceil(len(train_loader) / GRAD_ACCUM)
                warm = min(1.0, (step + 1) / (WARMUP_STEPS + 1))
                cosine = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(step / total_steps, 1.0)))
                for g in optimizer.param_groups:
                    g["lr"] = LR * warm * cosine
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        ev, bf1s = eval_epoch()
        ev.update(epoch=epoch, train_seconds=round(time.time() - t0, 1),
                  skipped_no_grad_batches=skipped_no_grad, gamma=float(gamma),
                  engagement_delta_over_p2=(float(np.mean(model._engagement))
                                             if model._engagement else 0.0))
        d = np.array(bf1s) - np.array(base_bf1)
        rng = np.random.RandomState(BOOT_SEED)
        idx = rng.randint(0, len(d), size=(BOOT_DRAWS, len(d)))
        ci_lo = float(np.percentile(d[idx].mean(1), 2.5))
        ev["bf1_delta_vs_pure_baseline"] = float(d.mean())
        ev["bf1_delta_ci95"] = [ci_lo, float(np.percentile(d[idx].mean(1), 97.5))]
        history.append(ev)
        json.dump(history, (out / "history.json").open("w"), indent=1)
        (out / f"per_scene_bf1_epoch{epoch}.json").write_text(json.dumps(bf1s))
        print(f"epoch {epoch}: BF1={ev['boundary_f1']:.4f} AP={ev['AP']:.4f} AP75={ev['AP75']:.4f} "
              f"gamma={ev['gamma']:.4f} dBF1={ev['bf1_delta_vs_pure_baseline']:+.4f} "
              f"CI_lo={ci_lo:+.4f} skipped={skipped_no_grad} ({ev['train_seconds']}s)", flush=True)

    print("H2 screening complete; gates evaluated in result review", flush=True)


if __name__ == "__main__":
    main()
