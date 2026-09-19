"""Frozen-feature fusion probes for the UE dual-stream design.

Three questions decide the next fusion mechanism, all answered WITHOUT
detector training:

P1 (information): per-cell building-occupancy linear readouts on frozen
   features, fit on train scenes / evaluated per-scene on val scenes --
     SAT        satellite P3 (512-d, trained single-stream checkpoint)
     LBEV       learned GeometrySparseBEV (128-d, dual r3 checkpoint)
     DBEV       training-free direct geometric BEV: sample the frozen UAV FPN
                at calibrated projections of every satellite cell (1024-d)
     DBEV-mis   DBEV with ground points rotated 90 deg about the scene center
                (geometry-content mismatch negative control)
     SAT+DBEV   concat -- the marginal UAV information over satellite alone
   If SAT+DBEV > SAT by a clear margin, UAV information exists and the
   injection mechanism is the bottleneck; if not, the UAV content itself is
   saturated at this resolution.

P2 (gate dynamics): dL/dgamma at the deployed checkpoint over real training
   batches -- SNR of the gate gradient vs its weight-decay equilibrium.

P3 (spatial structure): is the injected residual concentrated on building
   cells already, and does it align with the base pyramid features?
"""
import argparse
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
os.environ.setdefault("UE_DUAL_INJECT_EMBEDDING", "1")

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from mmengine.config import Config
from mmengine.registry import MODELS as MMENGINE_MODELS

import mmdet.models  # noqa: F401
import rsprompter  # noqa: F401
from mmdet.registry import MODELS
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmdet.structures import DetDataSample
from mmdet.structures.mask import BitmapMasks

if "DetDataPreprocessor" not in MMENGINE_MODELS:
    MMENGINE_MODELS.register_module(name="DetDataPreprocessor", module=DetDataPreprocessor)

from data.ue_dual_stream_dataset import UEDualStreamDataset, ue_dual_collate
from train.train_rsprompter_fusion import _attach_dual_stream_metainfo, _materialize_p2_boundary_refiner_loss


def rot_ground_points(points_cm: torch.Tensor, center_cm: torch.Tensor, deg: float) -> torch.Tensor:
    """Rotate ground points about the scene center (2-D, world XY in cm)."""
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    rel = points_cm[..., :2] - center_cm[:2][None, :]
    rot = torch.tensor([[c, -s], [s, c]], device=points_cm.device, dtype=points_cm.dtype)
    out = points_cm.clone()
    out[..., :2] = rel @ rot.T + center_cm[:2][None, :]
    return out


@torch.no_grad()
def scene_features(model, item, device, rot_deg=0.0):
    """Extract SAT P3, learned BEV, direct BEV (+delta stats) for one scene."""
    ds = DetDataSample()
    ds.set_metainfo(item["img_metas"])
    proc = model.data_preprocessor(
        {"inputs": [item["img"].to(device)], "data_samples": [ds.to(device)]}, training=False)
    inputs = proc["inputs"]
    if isinstance(inputs, (list, tuple)):
        inputs = inputs[0]

    vo = model.backbone(inputs)
    x = model.neck(vo[1])
    sat_p3 = x[1][0].float()                      # [512,128,128]
    sat_p2 = x[0][0].float()                      # [512,256,256]

    drone = item["drone_images"][None].to(device)  # [1,5,3,512,512] RGB 0-1
    drone = (drone - drone.new_tensor([.485, .456, .406])[None, None, :, None, None]) / \
            drone.new_tensor([.229, .224, .225])[None, None, :, None, None]
    _, fpn = model.drone_encoder(drone.reshape(drone.shape[0] * drone.shape[1], *drone.shape[2:]))
    fpn = sorted(tuple(fpn[-4:]), key=lambda t: t.shape[-2] * t.shape[-1], reverse=True)

    intr = item["intrinsics"][None].to(device)
    extr = item["extrinsics"][None].to(device)
    center2 = item["scene_center"][None].to(device)

    bev = model.geometry_bev(x[1], fpn, intr, extr, center2)[0].float()  # [128,128,128]

    # ---- training-free direct BEV ----
    h = w = sat_p3.shape[-1]
    dev, dt = device, torch.float32
    u = (torch.arange(w, device=dev, dtype=dt) + .5) * (512.0 / w)
    v = (torch.arange(h, device=dev, dtype=dt) + .5) * (512.0 / h)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    gx = -(vv / 512.0 - .5) * 360.0
    gy = (uu / 512.0 - .5) * 360.0
    center3 = torch.cat([center2[0], center2[0].new_zeros(1)])
    pts = torch.stack((gx * 100.0, gy * 100.0, torch.zeros_like(gx), torch.ones_like(gx)), -1)
    pts = pts.reshape(-1, 4)
    pts[..., :2] = pts[..., :2] + center3[None, :2]
    if rot_deg:
        pts = rot_ground_points(pts, center3, rot_deg)
    Q = pts.shape[0]

    per_level = []
    valid_any = torch.zeros(Q, device=dev, dtype=dt)
    for feat in fpn:                              # each [5,256,hl,wl]
        acc = torch.zeros(256, Q, device=dev, dtype=dt)
        cnt = torch.zeros(Q, device=dev, dtype=dt)
        for vi in range(extr.shape[1]):
            camc = pts @ extr[0, vi].T
            z = camc[:, 2]
            pix = camc[:, :3] @ intr[0, vi].T
            uvv = pix[:, :2] / pix[:, 2:3].clamp_min(1e-6)
            ok = (z > 0) & (uvv[:, 0] >= 0) & (uvv[:, 0] < 512) & (uvv[:, 1] >= 0) & (uvv[:, 1] < 512)
            grid = (uvv / 512.0) * 2.0 - 1.0
            grid[~ok] = 0.0
            g = grid[None, :, None, :]
            smp = F.grid_sample(feat[vi:vi + 1].float(), g, mode="bilinear",
                                padding_mode="zeros", align_corners=False)[0, :, :, 0]
            smp = smp * ok[None].float()
            acc += smp
            cnt += ok.float()
        per_level.append(acc / cnt.clamp_min(1.0))
        valid_any += cnt
    direct = torch.cat(per_level, 0)              # [1024, Q]
    direct = direct.reshape(direct.shape[0], h, w)
    coverage = (valid_any / float(extr.shape[1])).reshape(h, w)

    # ---- injection delta stats (P3) ----
    gate = model.p2p3_gate
    delta3 = (gate.gamma_p3 * gate.p3(bev[None]))[0].float()
    ratio3 = (delta3.flatten(1).norm(dim=0) + 1e-9) / (sat_p3.flatten(1).norm(dim=0) + 1e-9)
    cos3 = F.cosine_similarity(delta3.flatten(1), sat_p3.flatten(1), dim=0)

    # ---- occupancy ----
    masks = item["gt_masks"]
    occ = np.zeros((128, 128), np.uint8)
    if getattr(masks, "ndim", 0) == 3 and len(masks):
        for m in masks:
            occ = np.maximum(occ, cv2.resize(m, (128, 128), interpolation=cv2.INTER_NEAREST))
    occ_t = torch.from_numpy(occ).to(device)

    return dict(
        sat=sat_p3.cpu().numpy().astype(np.float16),
        bev=bev.cpu().numpy().astype(np.float16),
        direct=direct.cpu().numpy().astype(np.float16),
        occ=occ,
        ratio3=ratio3.cpu().numpy(), cos3=cos3.cpu().numpy(),
        coverage=coverage.cpu().numpy(),
    )


def p2_delta_stats(model, item, device):
    """Per-cell injection ratio at P2 vs 256-res occupancy."""
    ds = DetDataSample()
    ds.set_metainfo(item["img_metas"])
    proc = model.data_preprocessor(
        {"inputs": [item["img"].to(device)], "data_samples": [ds.to(device)]}, training=False)
    inputs = proc["inputs"][0] if isinstance(proc["inputs"], (list, tuple)) else proc["inputs"]
    with torch.no_grad():
        vo = model.backbone(inputs)
        x = model.neck(vo[1])
        drone = item["drone_images"][None].to(device)
        drone = (drone - drone.new_tensor([.485, .456, .406])[None, None, :, None, None]) / \
                drone.new_tensor([.229, .224, .225])[None, None, :, None, None]
        _, fpn = model.drone_encoder(drone.reshape(5, *drone.shape[2:]))
        fpn = sorted(tuple(fpn[-4:]), key=lambda t: t.shape[-2] * t.shape[-1], reverse=True)
        bev = model.geometry_bev(x[1], fpn, item["intrinsics"][None].to(device),
                                 item["extrinsics"][None].to(device),
                                 item["scene_center"][None].to(device))
        gate = model.p2p3_gate
        delta2 = (gate.gamma_p2 * gate.p2(F.interpolate(bev, size=x[0].shape[-2:],
                  mode="bilinear", align_corners=False)))[0].float()
        base2 = x[0][0].float()
        ratio2 = (delta2.flatten(1).norm(dim=0) + 1e-9) / (base2.flatten(1).norm(dim=0) + 1e-9)
    masks = item["gt_masks"]
    occ = np.zeros((256, 256), np.uint8)
    if getattr(masks, "ndim", 0) == 3 and len(masks):
        for m in masks:
            occ = np.maximum(occ, cv2.resize(m, (256, 256), interpolation=cv2.INTER_NEAREST))
    return ratio2.cpu().numpy(), occ


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra = (ra - ra.mean()) / (ra.std() + 1e-9)
    rb = (rb - rb.mean()) / (rb.std() + 1e-9)
    return float((ra * rb).mean())


def fit_probe(X: np.ndarray, y: np.ndarray, device, iters=500, lr=0.05, wd=1e-4, target="occupancy"):
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True) + 1e-4
    Xt = torch.tensor(((X - mu) / sd), dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    lin = nn.Linear(X.shape[1], 1).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=wd)
    for _ in range(iters):
        opt.zero_grad()
        out = lin(Xt).squeeze(-1)
        if target == "boundary":
            loss = F.mse_loss(out, yt)
        else:
            loss = F.binary_cross_entropy_with_logits(
                out, yt, pos_weight=torch.tensor(max(float(yt.sum()), 1.0) / max(float(len(yt) - yt.sum()), 1.0), device=yt.device))
        loss.backward()
        opt.step()
    return dict(w=lin.weight.detach()[0].cpu().numpy(), b=float(lin.bias.item()),
                mu=mu[0], sd=sd[0])


def score_probe(pr, X: np.ndarray) -> np.ndarray:
    return (X / pr["sd"] - pr["mu"] / pr["sd"]) @ pr["w"] + pr["b"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single-ckpt", default="work_dirs/ue/ue_single_c5v2_control_tr128_va259_s44_r1/best_model_epoch86.pth")
    ap.add_argument("--dual-ckpt", default="work_dirs/ue/ue_dual_c5v2_p2p3_emb_tr128_va259_s44_r1/best_model_epoch85.pth")
    ap.add_argument("--config", default="configs/ue_dual_stream_p2p3_sparse.py")
    ap.add_argument("--data-root", default="/data1/wangcheng/dataset/ue_coco_with_uav")
    ap.add_argument("--n-train", type=int, default=80)
    ap.add_argument("--n-val", type=int, default=60)
    ap.add_argument("--neg-per-scene", type=int, default=400)
    ap.add_argument("--grad-batches", type=int, default=10)
    ap.add_argument("--target", choices=["occupancy", "boundary"], default="occupancy")
    ap.add_argument("--out", default="outputs/probe_fusion_information")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    cfg = Config.fromfile(args.config)
    model = MODELS.build(cfg.model)
    dual = torch.load(args.dual_ckpt, map_location="cpu", weights_only=False)
    msg = model.load_state_dict(dual.get("model", dual.get("state_dict")), strict=False)
    print(f"dual ckpt loaded: missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    single = torch.load(args.single_ckpt, map_location="cpu", weights_only=False)
    msg2 = model.load_state_dict(single.get("model", single.get("state_dict")), strict=False)
    print(f"single ckpt overwrote satellite branch: missing={len(msg2.missing_keys)} unexpected={len(msg2.unexpected_keys)}")
    model = model.to(device)

    gam = dict(g2=float(model.p2p3_gate.gamma_p2), g3=float(model.p2p3_gate.gamma_p3),
               ge=float(model.embedding_gate.gamma) if getattr(model, "embedding_gate", None) else None)
    print("gate values:", gam)

    train_ds = UEDualStreamDataset(args.data_root, "train", image_size=(1024, 1024),
                                   train_subset=0.1797752808988764, seed=44, load_uav=True)
    val_ds = UEDualStreamDataset(args.data_root, "val", image_size=(1024, 1024), load_uav=True)
    print(f"train scenes={len(train_ds)} val scenes={len(val_ds)}")

    # ================= P2: gate-gradient SNR (model.train, real batches) =================
    print("\n===== P2 gate-gradient probe =====")
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)
    gates = [model.p2p3_gate.gamma_p2, model.p2p3_gate.gamma_p3]
    for g in gates:
        g.requires_grad_(True)
    if getattr(model, "embedding_gate", None) is not None:
        model.embedding_gate.gamma.requires_grad_(True)
        gates.append(model.embedding_gate.gamma)
    gvals = [[] for _ in gates]
    nb = 0
    for idx in range(len(train_ds)):
        if nb >= args.grad_batches:
            break
        item = train_ds[idx]
        if len(item["gt_labels"]) == 0:
            continue
        ds = DetDataSample()
        ds.set_metainfo(item["img_metas"])
        from mmengine.structures import InstanceData
        gi = InstanceData()
        gi.bboxes = item["gt_bboxes"].to(device)
        gi.labels = item["gt_labels"].to(device)
        gi.masks = BitmapMasks(item["gt_masks"], *item["img_metas"]["img_shape"])
        ds.gt_instances = gi
        batch = dict(drone_images=item["drone_images"][None], intrinsics=item["intrinsics"][None],
                     extrinsics=item["extrinsics"][None], scene_centers=item["scene_center"][None])
        _attach_dual_stream_metainfo(ds, batch, 0)
        proc = model.data_preprocessor({"inputs": [item["img"].to(device)],
                                        "data_samples": [ds.to(device)]}, training=True)
        inputs = proc["inputs"][0] if isinstance(proc["inputs"], (list, tuple)) else proc["inputs"]
        loss_dict = model.loss(inputs, proc["data_samples"])
        loss_dict = _materialize_p2_boundary_refiner_loss(loss_dict, model, False)
        parts = []
        for k, v in loss_dict.items():
            if "loss" not in k:
                continue
            if isinstance(v, torch.Tensor):
                parts.append(v)
            elif isinstance(v, (list, tuple)):
                parts.extend(x for x in v if isinstance(x, torch.Tensor))
        loss = sum(parts) if parts else torch.tensor(0.0, device=device)
        gs = torch.autograd.grad(loss, gates, allow_unused=False)
        for k, g in enumerate(gs):
            gvals[k].append(float(g))
        nb += 1
        print(f"  batch {nb}: loss={float(loss):.4f} grads={[f'{g:.3e}' for g in gs]}")
    grad_stats = {}
    names = ["gamma_p2", "gamma_p3"] + (["gamma_emb"] if len(gates) == 3 else [])
    for name, vals in zip(names, gvals):
        v = np.array(vals)
        grad_stats[name] = dict(mean=float(v.mean()), std=float(v.std()),
                                snr=abs(float(v.mean())) / (float(v.std()) + 1e-12),
                                sign_consistency=float((np.sign(v) == np.sign(v.mean())).mean()),
                                wd_pull=float(0.05 * abs(dict(gamma_p2=gam["g2"],
                                gamma_p3=gam["g3"], gamma_emb=gam["ge"] or 0.0)[name])))
    print(json.dumps(grad_stats, indent=1))

    # ================= P1: information probes =================
    print("\n===== P1 feature extraction (train scenes for probe fit) =====")
    model.eval()
    t0 = time.time()
    sets = {"SAT": 512, "LBEV": 128, "DBEV": 1024, "DBEVmis": 1024}
    fit_X = {k: [] for k in sets}
    fit_y = []
    n_used = 0
    for idx in range(len(train_ds)):
        if n_used >= args.n_train:
            break
        item = train_ds[idx]
        if len(item["gt_labels"]) == 0:
            continue
        f = scene_features(model, item, device, rot_deg=0.0)
        if f["occ"].sum() == 0:
            continue
        rng = np.random.RandomState(idx)
        if args.target == "boundary":
            occm = f["occ"].astype(np.uint8)
            d_out = cv2.distanceTransform(1 - occm, cv2.DIST_L2, 3)
            d_in = cv2.distanceTransform(occm, cv2.DIST_L2, 3)
            signed = (d_out - d_in).reshape(-1).astype(np.float32) / 8.0
            sel = rng.choice(signed.shape[0], size=1500, replace=False)
        else:
            occ = f["occ"].reshape(-1).astype(bool)
            pos_id = np.where(occ)[0]
            neg_id = np.where(~occ)[0]
            n_half = min(args.neg_per_scene, len(neg_id), len(pos_id))
            pos_id = rng.choice(pos_id, size=n_half, replace=False)
            neg_id = rng.choice(neg_id, size=n_half, replace=False)
            sel = np.concatenate([pos_id, neg_id])
        sat = f["sat"].reshape(512, -1).T
        bev = f["bev"].reshape(128, -1).T
        dbv = f["direct"].reshape(1024, -1).T
        dbm = scene_features(model, item, device, rot_deg=90.0)["direct"].reshape(1024, -1).T
        fit_X["SAT"].append(sat[sel])
        fit_X["LBEV"].append(bev[sel])
        fit_X["DBEV"].append(dbv[sel])
        fit_X["DBEVmis"].append(dbm[sel])
        if args.target == "boundary":
            fit_y.append(signed[sel])
        else:
            fit_y.append(f["occ"].reshape(-1)[sel])
        n_used += 1
        if n_used % 10 == 0:
            print(f"  {n_used} scenes ({time.time()-t0:.0f}s)")
    Xs = {k: np.concatenate(v).astype(np.float32) for k, v in fit_X.items()}
    y = np.concatenate(fit_y).astype(np.float32)
    print(f"fit cells: {len(y)} pos={int(y.sum())}")
    # concat sets
    Xs["SAT+LBEV"] = np.concatenate([Xs["SAT"], Xs["LBEV"]], 1)
    Xs["SAT+DBEV"] = np.concatenate([Xs["SAT"], Xs["DBEV"]], 1)
    print("\n===== fitting linear probes =====")
    probes = {}
    for k, X in Xs.items():
        t = time.time()
        probes[k] = fit_probe(X, y, device, target=args.target)
        print(f"  {k:9s} dim={X.shape[1]} fit {time.time()-t:.0f}s")

    print("\n===== P1 evaluation on val scenes =====")
    rows = []
    per_scene = {k: [] for k in probes}
    p3stats = dict(ratio_occ=[], ratio_bg=[], spearman=[], cos_occ=[], cos_bg=[], coverage=[])
    p2stats = dict(ratio_occ=[], ratio_bg=[], spearman=[])
    from sklearn.metrics import average_precision_score
    for idx in range(min(args.n_val, len(val_ds))):
        item = val_ds[idx]
        if len(item["gt_labels"]) == 0:
            continue
        f = scene_features(model, item, device, rot_deg=0.0)
        if f["occ"].sum() == 0:
            continue
        occ = f["occ"].reshape(-1).astype(int)
        sat = f["sat"].reshape(512, -1).T.astype(np.float32)
        bev = f["bev"].reshape(128, -1).T.astype(np.float32)
        dbv = f["direct"].reshape(1024, -1).T.astype(np.float32)
        dbm = scene_features(model, item, device, rot_deg=90.0)["direct"].reshape(1024, -1).T.astype(np.float32)
        feats = dict(SAT=sat, LBEV=bev, DBEV=dbv, DBEVmis=dbm,
                     **{"SAT+LBEV": np.concatenate([sat, bev], 1),
                        "SAT+DBEV": np.concatenate([sat, dbv], 1)})
        occ = f["occ"].reshape(-1).astype(int)
        if args.target == "boundary":
            occm = f["occ"].astype(np.uint8)
            d_out = cv2.distanceTransform(1 - occm, cv2.DIST_L2, 3)
            d_in = cv2.distanceTransform(occm, cv2.DIST_L2, 3)
            ytrue = (d_out - d_in).reshape(-1).astype(np.float32) / 8.0
        aps = {}
        for k, pr in probes.items():
            sc = score_probe(pr, feats[k])
            if args.target == "boundary":
                r = float(np.corrcoef(sc, ytrue)[0, 1])
                aps[k] = r
            else:
                aps[k] = float(average_precision_score(occ, sc))
            per_scene[k].append(aps[k])
        rows.append(dict(scene=item["img_metas"]["scene_id"], **aps))
        occm = f["occ"].reshape(-1).astype(bool)
        p3stats["ratio_occ"].append(float(f["ratio3"][occm].mean()))
        p3stats["ratio_bg"].append(float(f["ratio3"][~occm].mean()))
        p3stats["cos_occ"].append(float(f["cos3"][occm].mean()))
        p3stats["cos_bg"].append(float(f["cos3"][~occm].mean()))
        p3stats["spearman"].append(rank_corr(f["ratio3"], f["occ"].reshape(-1).astype(np.float32)))
        p3stats["coverage"].append(float(f["coverage"].mean()))
        r2, occ2 = p2_delta_stats(model, item, device)
        o2 = occ2.reshape(-1).astype(bool)
        p2stats["ratio_occ"].append(float(r2[o2].mean()))
        p2stats["ratio_bg"].append(float(r2[~o2].mean()))
        p2stats["spearman"].append(rank_corr(r2, occ2.reshape(-1).astype(np.float32)))
        if (idx + 1) % 10 == 0:
            print(f"  {idx+1} val scenes ({time.time()-t0:.0f}s)")

    keys = list(next(iter(rows)).keys())[1:]
    print(f"\n===== per-scene {'Pearson-r' if args.target=='boundary' else 'AP'} summary =====")
    summary = {}
    base = np.array(per_scene["SAT"])
    for k in keys:
        v = np.array(per_scene[k])
        entry = dict(mean=float(v.mean()), std=float(v.std()), n=len(v))
        if k.startswith("SAT+"):
            d = v - base
            entry.update(delta_mean=float(d.mean()), delta_std=float(d.std()),
                         improved=int((d > 0).sum()), worse=int((d < 0).sum()))
        summary[k] = entry
        print(f"  {k:9s}: {v.mean():.4f} ± {v.std():.4f}" +
              (f"   ΔvsSAT={entry['delta_mean']:+.4f} (improved {entry['improved']}/{len(v)})" if "delta_mean" in entry else ""))
    dm = np.array(per_scene["SAT+DBEV"]) - base
    if dm.std() > 0:
        from scipy import stats as st
        t, p = st.ttest_rel(np.array(per_scene["SAT+DBEV"]), base)
        summary["SAT+DBEV"]["paired_t"] = float(t); summary["SAT+DBEV"]["p"] = float(p)
        print(f"  paired t-test SAT+DBEV vs SAT: t={t:.2f} p={p:.4g}")
    out_summary = dict(
        target=args.target, gates=gam, grad_stats=grad_stats, ap_summary=summary,
        p3=dict((k, float(np.mean(v))) for k, v in p3stats.items()),
        p2=dict((k, float(np.mean(v))) for k, v in p2stats.items()),
        n_train_used=n_used, n_val=len(rows),
    )
    (out / "summary.json").write_text(json.dumps(out_summary, indent=2))
    with (out / "per_scene_val_ap.csv").open("w") as f:
        f.write("scene," + ",".join(keys) + "\n")
        for r in rows:
            f.write(r["scene"] + "," + ",".join(f"{r[k]:.5f}" for k in keys) + "\n")
    print(f"\nsaved {out}/summary.json and per_scene_val_ap.csv")


if __name__ == "__main__":
    main()
