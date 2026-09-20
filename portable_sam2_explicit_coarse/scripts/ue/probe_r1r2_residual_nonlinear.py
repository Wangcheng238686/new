"""MHBEV-FUS-20260919-R1/R2: frozen-feature probes for the two remaining live
questions on the ORIGINAL UE dataset. Read-only; no training of the detector.

R1 (model-residual conditioning): within the boundary search band, can DBEV
  predict where the frozen single-stream model's prediction DISAGREES with GT,
  beyond SAT? Band cells, per-scene AUC, paired bootstrap vs SAT.
  R1_MARGINAL_PASS iff CI lower of (SAT+DBEV - SAT) > 0.

R2 (nonlinear field readout): can an MLP extract more boundary-field
  information from DBEV than the linear ridge? Target = 128-res sdf.
  R2_UNLOCK iff CI lower of (MLP SAT+DBEV - MLP SAT) > 0 AND
  MLP(SAT+DBEV) - ridge(SAT+DBEV) >= +0.02 r.
"""
import argparse
import io
import contextlib
import json
import math
import os
import sys
import time
import subprocess
import copy
from pathlib import Path

os.environ.setdefault("EXPLICIT_PROMPT_MODE", "points_box_dense")
os.environ.setdefault("P2_BOUNDARY_REFINER_ENABLED", "1")
os.environ.setdefault("SAM_IMAGE_EMBED_STRIDE", "16")
os.environ.setdefault("SHAPE_DENSE_TRANSFORM", "raw_logits")
os.environ.setdefault("SHAPE_DENSE_DETACH", "1")
os.environ.setdefault("FINAL_MASK_COORDINATE_MODE", "roi_local")
os.environ.setdefault("SAM2_CKPT", "/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt")
os.environ.setdefault("UE_DUAL_INJECT_EMBEDDING", "0")  # single-stream predictions

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

from data.ue_dual_stream_dataset import UEDualStreamDataset
from sklearn.metrics import roc_auc_score

SINGLE_CKPT = "work_dirs/ue/ue_single_c5v2_control_tr128_va259_s44_r1/best_model_epoch86.pth"
DUAL_CKPT = "work_dirs/ue/ue_dual_c5v2_p2p3_emb_tr128_va259_s44_r1/best_model_epoch85.pth"
BOOT_SEED = 20260930
BOOT_DRAWS = 10000
RES = 128


def subprocess_git():
    root = str(project_root)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, cwd=root).stdout.strip()
    dirty = subprocess.run(["git", "status", "--short"], capture_output=True,
                           text=True, cwd=root).stdout
    return commit, dirty


def sdf128_of(item):
    occ = np.zeros((RES, RES), np.uint8)
    for m in item["gt_masks"]:
        occ = np.maximum(occ, cv2.resize(m, (RES, RES), interpolation=cv2.INTER_NEAREST))
    if occ.sum() == 0:
        return None, None
    d_out = cv2.distanceTransform(1 - occ, cv2.DIST_L2, 3)
    d_in = cv2.distanceTransform(occ, cv2.DIST_L2, 3)
    sdf = cv2.blur((d_out - d_in).astype(np.float32), (3, 3))
    sdf = (sdf - sdf.mean()) / (sdf.std() + 1e-6)
    boundary = cv2.morphologyEx(occ, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    band = cv2.dilate(boundary.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0  # radius 4
    return occ, (sdf, band.reshape(-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ue_dual_stream_p2p3_sparse.py")
    ap.add_argument("--data-root", default="/data1/wangcheng/dataset/ue_coco_with_uav")
    ap.add_argument("--n-fit", type=int, default=80)
    ap.add_argument("--n-val", type=int, default=259)
    ap.add_argument("--r1-cells-per-scene", type=int, default=2000)
    ap.add_argument("--r2-cells-per-scene", type=int, default=1500)
    ap.add_argument("--out", default="outputs/probe_r1r2")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    import random
    SEED = 20260929
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    cfg = Config.fromfile(args.config)
    model = MODELS.build(cfg.model)
    dual = torch.load(DUAL_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(dual.get("model", dual.get("state_dict")), strict=False)
    single = torch.load(SINGLE_CKPT, map_location="cpu", weights_only=False)
    single_sd = single.get("model", single.get("state_dict"))
    model_sd = model.state_dict()
    filtered = {k: v for k, v in single_sd.items()
                if k in model_sd and model_sd[k].shape == v.shape}
    dropped = [k for k in single_sd if k not in filtered]
    model.load_state_dict(filtered, strict=False)
    assert dropped == [], dropped
    model = model.to(device).eval()
    # dual ckpt carries TRAINED gate values; the single overlay lacks gate keys.
    # Zero them BEFORE any predict, else predictions are not the pure
    # single-stream baseline and R1 labels are contaminated.
    gates_before = dict(g2=float(model.p2p3_gate.gamma_p2), g3=float(model.p2p3_gate.gamma_p3))
    model.p2p3_gate.gamma_p2.data.zero_()
    model.p2p3_gate.gamma_p3.data.zero_()
    assert (not hasattr(model, "embedding_gate")) or model.embedding_gate is None
    assert float(model.p2p3_gate.gamma_p2) == 0.0 and float(model.p2p3_gate.gamma_p3) == 0.0
    print(f"gates zeroed: before={gates_before} after=0/0 "
          "(predictions = exact single-stream)", flush=True)
    print("gates zero-verified: predictions = exact single-stream", flush=True)

    train_ds = UEDualStreamDataset(args.data_root, "train", image_size=(1024, 1024),
                                   train_subset=0.1797752808988764, seed=44, load_uav=True)
    val_ds = UEDualStreamDataset(args.data_root, "val", image_size=(1024, 1024), load_uav=True)

    def forward_inputs(item):
        ds = DetDataSample(); ds.set_metainfo(item["img_metas"])
        ds.set_metainfo({"drone_images": item["drone_images"], "intrinsics": item["intrinsics"],
                         "extrinsics": item["extrinsics"], "scene_center": item["scene_center"]})
        proc = model.data_preprocessor(
            {"inputs": [item["img"].to(device)], "data_samples": [ds.to(device)]}, training=False)
        inputs = proc["inputs"][0] if isinstance(proc["inputs"], (list, tuple)) else proc["inputs"]
        return inputs, proc["data_samples"]

    def extract_scene(item):
        """Single predict + both correct and rotated-geometry DBEV sampling."""
        inputs, dss = forward_inputs(item)
        with torch.no_grad():
            vo = model.backbone(inputs)
            x = model.neck(vo[1])
            sat = x[1][0].float()                       # [512,128,128]
            drone = item["drone_images"][None].to(device)
            drone = (drone - drone.new_tensor([.485, .456, .406])[None, None, :, None, None]) / \
                    drone.new_tensor([.229, .224, .225])[None, None, :, None, None]
            _, fpn = model.drone_encoder(drone.reshape(drone.shape[0] * drone.shape[1],
                                                       *drone.shape[2:]))
            fpn = sorted(tuple(fpn[-4:]), key=lambda t: t.shape[-2] * t.shape[-1], reverse=True)
            bev_learned = model.geometry_bev(x[1], fpn, item["intrinsics"][None].to(device),
                                             item["extrinsics"][None].to(device),
                                             item["scene_center"][None].to(device))[0].float()
            h = w = RES
            u = (torch.arange(w, device=device) + .5) * (512.0 / w)
            v = (torch.arange(h, device=device) + .5) * (512.0 / h)
            vv, uu = torch.meshgrid(v, u, indexing="ij")
            gx = -(vv / 512.0 - .5) * 360.0
            gy = (uu / 512.0 - .5) * 360.0
            c = item["scene_center"].to(device)
            base = torch.stack((gx.reshape(-1) * 100 + c[0], gy.reshape(-1) * 100 + c[1],
                                torch.zeros(h * w, device=device),
                                torch.ones(h * w, device=device)), -1)
            rot = torch.tensor([[0., -1.], [1., 0.]], device=device)
            rel = base[:, :2] - c[None, :2]
            pts_mis = base.clone()
            pts_mis[:, :2] = rel @ rot.T + c[None, :2]

            def sample(pts):
                per_level = []
                for feat in fpn:                        # [5,256,h,w]
                    acc = torch.zeros(256, h * w, device=device)
                    cnt = torch.zeros(h * w, device=device)
                    for vi in range(5):
                        camc = pts @ item["extrinsics"][vi].cuda().T
                        z = camc[:, 2]
                        pix = camc[:, :3] @ item["intrinsics"][vi].cuda().T
                        uv = pix[:, :2] / pix[:, 2:3].clamp_min(1e-6)
                        ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < 512) & \
                             (uv[:, 1] >= 0) & (uv[:, 1] < 512)
                        grid = ((uv / 512.0) * 2.0 - 1.0)
                        grid[~ok] = 0.0
                        smp = F.grid_sample(feat[vi:vi + 1].float(), grid[None, :, None, :],
                                            mode="bilinear", padding_mode="zeros",
                                            align_corners=False)[0, :, :, 0]
                        acc += smp * ok[None].float()
                        cnt += ok.float()
                    per_level.append(acc / cnt.clamp_min(1.0))
                return torch.cat(per_level, 0)          # [1024, Q]

            direct = sample(base).reshape(1024, h, w)
            directmis = sample(pts_mis).reshape(1024, h, w)
            # deployed-path prediction (single forward; exact single-stream:
            # gates zeroed and verified at startup)
            outs = model.predict(inputs, dss, rescale=False)
            inst = outs[0].pred_instances
            pm = np.zeros((1024, 1024), np.uint8)
            if inst is not None and len(inst.bboxes):
                msk = inst.masks
                pmasks = msk.to_ndarray() if isinstance(msk, BitmapMasks) else msk.cpu().numpy()
                keep = inst.scores.cpu().numpy() >= 0.5
                if keep.any():
                    for m in pmasks[keep]:
                        pm = np.maximum(pm, m.astype(np.uint8))
            pred128 = cv2.resize(pm, (RES, RES), interpolation=cv2.INTER_NEAREST)
        occ, sdfpack = sdf128_of(item)
        if occ is None:
            return None
        sdf, band = sdfpack
        pred128_bin = (pred128 > 0).reshape(-1)
        gt_bin = occ.reshape(-1) > 0
        err = (pred128_bin != gt_bin).astype(np.float32)
        return dict(
            sat=sat.reshape(512, -1).T.cpu().numpy().astype(np.float16),
            bev=bev_learned.reshape(128, -1).T.cpu().numpy().astype(np.float16),
            direct=direct.reshape(1024, -1).T.cpu().numpy().astype(np.float16),
            directmis=directmis.reshape(1024, -1).T.cpu().numpy().astype(np.float16),
            band=band, sdf=sdf, err=err, gt=gt_bin, pred=pred128_bin)

    # ================= fit pass (train scenes) =================
    print("=== fit pass (train scenes) ===", flush=True)
    r1 = {k: [] for k in ("sat", "bevlearned", "direct", "err")}
    r2 = {k: [] for k in ("sat", "direct", "directmis", "sdf")}
    n_fit = 0
    t0 = time.time()
    for idx in range(len(train_ds)):
        if n_fit >= args.n_fit: break
        item = train_ds[idx]
        if len(item["gt_labels"]) == 0: continue
        f = extract_scene(item)
        if f is None: continue
        rng = np.random.RandomState(7000 + idx)
        bi = np.where(f["band"] & ((f["err"] > 0) | (f["pred"] > 0)))[0]
        if len(bi) < 50: continue
        sel = rng.choice(bi, size=min(args.r1_cells_per_scene, len(bi)), replace=False)
        r1["sat"].append(f["sat"][sel]); r1["bevlearned"].append(f["bev"][sel])
        r1["direct"].append(f["direct"][sel]); r1["err"].append(f["err"][sel])
        ri = rng.choice(f["sdf"].shape[0], size=args.r2_cells_per_scene, replace=False)
        r2["sat"].append(f["sat"][ri]); r2["direct"].append(f["direct"][ri])
        r2["directmis"].append(f["directmis"][ri]); r2["sdf"].append(f["sdf"][ri])
        n_fit += 1
        if n_fit % 10 == 0: print(f"  fit {n_fit} scenes ({time.time()-t0:.0f}s)", flush=True)
    Xs = {k: np.concatenate(v).astype(np.float32) for k, v in r1.items()}
    y_err = Xs.pop("err")
    F2 = {k: np.concatenate(v).astype(np.float32) for k, v in r2.items()}
    y_sdf = F2.pop("sdf")
    print(f"R1 fit cells={len(y_err)} err_rate={y_err.mean():.3f}; R2 fit cells={len(y_sdf)}", flush=True)

    # ---- R1 probes: standardized logistic per arm ----
    def fit_logistic(X, y, iters=300, lr=0.05, C=0.1):
        mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True) + 1e-4
        Xt = torch.tensor((X - mu) / sd, dtype=torch.float32, device=device)
        yt = torch.tensor(y, dtype=torch.float32, device=device)
        lin = nn.Linear(X.shape[1], 1).to(device)
        opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=C)
        pos = float(yt.sum()) / max(len(yt), 1)
        pw = torch.tensor((1 - pos) / max(pos, 1e-6), device=device).clamp(0.2, 5)
        for _ in range(iters):
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(lin(Xt).squeeze(-1), yt, pos_weight=pw)
            loss.backward(); opt.step()
        return dict(w=lin.weight.detach()[0].cpu().numpy(), b=float(lin.bias.item()), mu=mu[0], sd=sd[0])

    def score_logistic(pr, X):
        return (X / pr["sd"] - pr["mu"] / pr["sd"]) @ pr["w"] + pr["b"]

    r1_probes, r1_pca = {}, {}
    for name, mat in (("SAT", Xs["sat"]), ("BEVlearned", Xs["bevlearned"]),
                      ("DBEVd", Xs["direct"]),
                      ("SAT+DBEVd", np.concatenate([Xs["sat"], Xs["direct"]], 1))):
        mu = mat.mean(0, keepdims=True); sd = mat.std(0, keepdims=True) + 1e-4
        r1_pca[name] = (mu[0], sd[0])
        r1_probes[name] = fit_logistic((mat - mu) / sd, y_err)
    print("R1 probes fitted", flush=True)

    # ---- R2 models: ridge + MLP ----
    def fit_ridge(X, y):
        mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True) + 1e-4
        Xs_ = (X - mu) / sd
        W = np.linalg.solve(Xs_.T @ Xs_ + 1.0 * np.eye(X.shape[1]), Xs_.T @ (y - y.mean()))
        return dict(w=W, b=float(y.mean()), mu=mu[0], sd=sd[0])

    def score_ridge(pr, X):
        return (X / pr["sd"] - pr["mu"] / pr["sd"]) @ pr["w"] + pr["b"] + 0 * pr["b"]

    class MLP(nn.Module):
        def __init__(self, din):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(din, 256), nn.GELU(),
                                     nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))
        def forward(self, x): return self.net(x).squeeze(-1)

    def fit_mlp(X, y, device, iters=400, lr=1e-3, copy=None):
        mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True) + 1e-4
        Xt = torch.tensor((X - mu) / sd, dtype=torch.float32, device=device)
        yt = torch.tensor(y, dtype=torch.float32, device=device)
        net = MLP(X.shape[1]).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
        for _ in range(iters):
            opt.zero_grad()
            loss = F.mse_loss(net(Xt), yt)
            loss.backward(); opt.step()
        sd_np = sd[0]
        return dict(net=net, mu=mu[0], sd=sd_np)

    def score_mlp(pr, X):
        Xt = torch.tensor((X / pr["sd"] - pr["mu"] / pr["sd"]), dtype=torch.float32, device=device)
        with torch.no_grad():
            return pr["net"](Xt).squeeze(-1).cpu().numpy()

    print("=== R2 models fitting ===", flush=True)
    r2_models = {}
    X2_cat = np.concatenate([F2["sat"], F2["direct"]], 1)
    r2_models[("ridge", "SAT")] = fit_ridge(F2["sat"], y_sdf)
    r2_models[("ridge", "DBEVd")] = fit_ridge(F2["direct"], y_sdf)
    r2_models[("ridge", "SAT+DBEVd")] = fit_ridge(X2_cat, y_sdf)
    r2_models[("ridge", "DBEVmis")] = fit_ridge(F2["directmis"], y_sdf)
    for name, mat in (("SAT", F2["sat"]), ("DBEVd", F2["direct"]), ("SAT+DBEVd", X2_cat),
                      ("DBEVmis", F2["directmis"])):
        r2_models[("mlp", name)] = fit_mlp(mat, y_sdf, device, copy=copy)
    print("R2 models fitted", flush=True)

    # ================= eval pass (val scenes) =================
    print("=== eval pass (val scenes) ===", flush=True)
    per_scene = {}
    n_val = 0
    skipped_eval_scenes = 0
    for idx in range(args.n_val):
        item = val_ds[idx]
        if len(item["gt_labels"]) == 0: continue
        f = extract_scene(item)
        if f is None: continue
        band = f["band"]
        res = dict(scene_index=idx)
        if band.sum() < 50:
            skipped_eval_scenes += 1
            per_scene[item["img_metas"]["scene_id"]] = dict(scene_index=idx, skipped=True)
            continue
        if len(np.unique(f["err"][band])) < 2:
            skipped_eval_scenes += 1
            per_scene[item["img_metas"]["scene_id"]] = dict(scene_index=idx, skipped=True)
            continue
        # R1 AUC on band cells
        allmats = dict(SAT=f["sat"].astype(np.float32), BEVlearned=f["bev"].astype(np.float32),
                       DBEVd=f["direct"].astype(np.float32),
                       DBEVmis=f["directmis"].astype(np.float32))
        allmats["SAT+DBEVd"] = np.concatenate([allmats["SAT"], allmats["DBEVd"]], 1)
        for name, pr in r1_probes.items():
            sc = score_logistic(pr, allmats[name])
            res[f"R1_{name}"] = float(roc_auc_score(f["err"][band], sc[band]))
        # R2 Pearson on all cells
        allmats["SAT+DBEVd"] = np.concatenate([allmats["SAT"], allmats["DBEVd"]], 1)
        for (kind, name), pr in r2_models.items():
            sc = (score_ridge(pr, allmats[name]) if kind == "ridge"
                  else score_mlp(pr, allmats[name]))
            if np.std(sc) < 1e-9: continue
            res[f"R2_{kind}_{name}"] = float(np.corrcoef(sc, f["sdf"])[0, 1])
        per_scene[item["img_metas"]["scene_id"]] = res
        n_val += 1
        if n_val % 10 == 0: print(f"  eval {n_val} scenes", flush=True)

    keys = sorted({k for r in per_scene.values() for k in r
                   if k.startswith(("R1_", "R2_"))})
    skipped_eval = sum(1 for r in per_scene.values() if r.get("skipped"))
    commit, dirty = subprocess_git()
    manifest = dict(git_commit=commit, git_dirty=dirty,
                    seeds=dict(env=SEED, boot=BOOT_SEED, fit_cell="7000+idx"),
                    gates_before=gates_before,
                    single_ckpt=SINGLE_CKPT, dual_ckpt=DUAL_CKPT,
                    n_fit_scenes=n_fit, n_val_requested=args.n_val,
                    r2_fit_protocol_deviation=(
                        "80 scenes x 1500 cells, Adam 400 iters full-batch, no early stop "
                        "(preregistered: 60 fit + 20 early-stop split, 200 iters); "
                        "both MLP arms symmetric, eval on held-out scenes"),
                    r1_fit_sampling_note="band cells intersected with (err>0 | pred>0), <=2000/scene",
                    pos_weight_clamp="0.2..5 (affects fit balance, AUC is rank-based)",
                    naming_note="BEVlearned = learned geometry_bev 128-d; DBEVd = direct geometric 1024-d")
    print("\n===== per-scene summary (mean±std over val scenes) =====")
    summary = {}
    for k in keys:
        v = np.array([r[k] for r in per_scene.values() if k in r])
        summary[k] = dict(mean=float(v.mean()), std=float(v.std()), n=len(v))
        print(f"  {k:22s}: {v.mean():+.4f} ± {v.std():.4f} (n={len(v)})")
    # paired marginals
    def paired(a, b):
        keys_common = [s for s in per_scene if a in per_scene[s] and b in per_scene[s]]
        va = np.array([per_scene[s][a] for s in keys_common])
        vb = np.array([per_scene[s][b] for s in keys_common])
        d = va - vb
        rng = np.random.RandomState(BOOT_SEED)
        idx = rng.randint(0, len(d), size=(BOOT_DRAWS, len(d)))
        lo = float(np.percentile(d[idx].mean(1), 2.5))
        hi = float(np.percentile(d[idx].mean(1), 97.5))
        return dict(mean=float(d.mean()), ci=[lo, hi], improved=int((d > 0).sum()), n=len(d))
    def paired_cohort(a, b, cohort):
        if cohort == "first60":
            sub = {s: r for s, r in per_scene.items() if r.get("scene_index", 999) < 60}
        else:
            sub = per_scene
        keys_common = [s for s in sub if a in sub[s] and b in sub[s]]
        va = np.array([sub[s][a] for s in keys_common])
        vb = np.array([sub[s][b] for s in keys_common])
        d = va - vb
        rng = np.random.RandomState(BOOT_SEED)
        idx = rng.randint(0, len(d), size=(BOOT_DRAWS, len(d)))
        lo = float(np.percentile(d[idx].mean(1), 2.5))
        hi = float(np.percentile(d[idx].mean(1), 97.5))
        return dict(mean=float(d.mean()), ci=[lo, hi], improved=int((d > 0).sum()), n=len(d))

    for cohort in ("first60", "full259"):
        print(f"===== paired marginals on cohort={cohort} =====")
        for a, b in (("R1_SAT+DBEVd", "R1_SAT"), ("R1_SAT+DBEVd", "R1_DBEVd"),
                     ("R2_mlp_SAT+DBEVd", "R2_mlp_SAT"),
                     ("R2_mlp_SAT+DBEVd", "R2_ridge_SAT+DBEVd"),
                     ("R2_ridge_SAT+DBEVd", "R2_ridge_SAT")):
            try:
                p = paired_cohort(a, b, cohort)
                summary[f"paired[{cohort}] {a} - {b}"] = p
                print(f"  {a} - {b}: {p['mean']:+.4f} CI[{p['ci'][0]:+.4f},{p['ci'][1]:+.4f}] "
                      f"improved {p['improved']}/{p['n']}")
            except Exception as ex:
                print(f"  paired {a} - {b}: failed {ex}")
    for a, b in (("R1_SAT+DBEVd", "R1_SAT"), ("R1_SAT+DBEVd", "R1_DBEVd"),
                 ("R2_mlp_SAT+DBEVd", "R2_mlp_SAT"), ("R2_mlp_SAT+DBEVd", "R2_ridge_SAT+DBEVd"),
                 ("R2_ridge_SAT+DBEVd", "R2_ridge_SAT")):
        try:
            summary[f"paired {a} - {b}"] = paired(a, b)
            p = summary[f"paired {a} - {b}"]
            print(f"  paired {a} - {b}: {p['mean']:+.4f} CI[{p['ci'][0]:+.4f},{p['ci'][1]:+.4f}] "
                  f"improved {p['improved']}/{p['n']}")
        except Exception as ex:
            print(f"  paired {a} - {b}: failed {ex}")
    summary["manifest"] = manifest
    summary["skipped_eval_scenes"] = skipped_eval
    json.dump(summary, (out / "summary.json").open("w"), indent=1)
    json.dump(per_scene, (out / "per_scene.json").open("w"), indent=1)
    print("saved", out / "summary.json")


if __name__ == "__main__":
    main()
