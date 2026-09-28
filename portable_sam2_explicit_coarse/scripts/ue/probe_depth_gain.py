"""MHBEV-FUS-20260921-DEPTH: does UAV-rendered DEPTH (height) carry marginal
information for satellite segmentation beyond SAT features?

Scenes: the new-standard strict-instance renders (3502-3664). The UE instance
EXR gives a binary building-foreground GT (per-instance IDs are not exposed);
5-view rendered depth gives per-cell world height.

Fit on the first 80 valid scenes, evaluate per-scene on the rest.
Readouts:
  R1  band-error AUC     (band cells, label = model prediction != GT)
  R2  sdf-field Pearson  (all cells, target = signed distance /8)
Arms: SAT (satellite P3 512-d) / HEIGHT (depth world-height stats 4-d) /
      SAT+HEIGHT (516-d).
Gates: paired bootstrap (10k, seed 20260930) CI lower > 0 for
  HEIGHT-SAT and (SAT+HEIGHT)-SAT marginals on each readout.
Read-only; no detector training.
"""
import argparse
import io
import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
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
from torch import nn
from mmengine.config import Config
from mmengine.registry import MODELS as MMENGINE_MODELS

import mmdet.models  # noqa: F401
import rsprompter  # noqa: F401
from mmdet.registry import MODELS
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmdet.structures import DetDataSample
from mmdet.structures.mask import BitmapMasks
from sklearn.metrics import roc_auc_score

if "DetDataPreprocessor" not in MMENGINE_MODELS:
    MMENGINE_MODELS.register_module(name="DetDataPreprocessor", module=DetDataPreprocessor)

SINGLE_CKPT = "work_dirs/ue/ue_single_c5v2_control_tr128_va259_s44_r1/best_model_epoch86.pth"
RES = 128
BOOT_SEED = 20260930
BOOT_DRAWS = 10000


def read_depth_exr(path):
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise OSError(f"cannot read EXR {path}")
    if im.ndim == 3:   # pick the channel with the most finite-positive values
        best = None
        for c in range(im.shape[2]):
            ok = np.isfinite(im[:, :, c]) & (im[:, :, c] > 0)
            if best is None or int(ok.sum()) > best[0]:
                best = (int(ok.sum()), im[:, :, c])
        z = best[1].astype(np.float32)
    else:
        z = im.astype(np.float32)
    valid = np.isfinite(z) & (z > 0)
    return z, valid


def load_camera(view_dir):
    intr = json.load(open(Path(view_dir) / "camera_intrinsics.json"))
    j = json.load(open(Path(view_dir) / "camera_extrinsics.json"))
    K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]],
                 dtype=np.float64)
    c2w = np.asarray(j["camera_to_world_cv"], dtype=np.float64)
    return K, c2w


def sdf_band_of(occ128):
    d_out = cv2.distanceTransform(1 - occ128, cv2.DIST_L2, 3)
    d_in = cv2.distanceTransform(occ128, cv2.DIST_L2, 3)
    sdf = cv2.blur((d_out - d_in).astype(np.float32), (3, 3))
    sdf = (sdf - sdf.mean()) / (sdf.std() + 1e-6)
    boundary = cv2.morphologyEx(occ128, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    band = cv2.dilate(boundary.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    return sdf.reshape(-1), band.reshape(-1)


def scene_record(point_dir, model, device, res=128):
    """Decode strict-instance GT + height features + satellite P3 + model pred.

    World->cell binning uses the 360m contract directly:
      i = floor((0.5 - x_m/360) * res)   (satellite row, from world -x)
      j = floor((y_m/360 + 0.5) * res)   (satellite col, from world +y)
    """
    exr_p = Path(point_dir) / "satellite" / \
        "satellite_instance_id.FinalImageM_PP_CustomStencil_GrayRaw.0000.exr"
    rgb_p = Path(point_dir) / "satellite" / "satellite_rgb.0000.png"
    inst_p = Path(point_dir) / "satellite" / "instances.json"
    if not (exr_p.exists() and rgb_p.exists() and inst_p.exists()):
        return None
    exr = cv2.imread(str(exr_p), cv2.IMREAD_UNCHANGED)
    if exr is None:
        return None
    ch_scores = [len(np.unique(exr[:, :, c])) for c in range(exr.shape[2])]
    ids = np.rint(exr[:, :, int(np.argmax(ch_scores))]).astype(np.int32)
    inst = json.load(open(inst_p))["instances"]
    occ = np.zeros((res, res), np.uint8)
    n_inst = 0
    for i in inst:
        lid = int(i["local_instance_id"])
        m512 = (ids == lid).astype(np.uint8)
        if m512.sum() < 25:
            continue
        m128 = cv2.resize(m512, (res, res), interpolation=cv2.INTER_NEAREST)
        if m128.sum() == 0:
            continue
        occ = np.maximum(occ, m128)
        n_inst += 1
    if occ.sum() < 100 or n_inst == 0:
        return None
    sdf, band = sdf_band_of(occ)

    # ---- per-cell height stats: per-view cell mean, then across views ----
    oblique = Path(point_dir) / "oblique"
    cell_means = []
    n_views = 0
    Q = res * res
    for vd in sorted(oblique.iterdir()):
        exr_d = vd / "depth_camera_z.0000.exr"
        if not exr_d.exists():
            continue
        z, valid = read_depth_exr(vd)
        K, c2w = load_camera(vd)
        H, W = z.shape
        u, v = np.meshgrid(np.arange(W, dtype=np.float64),
                           np.arange(H, dtype=np.float64), indexing="xy")
        x = (u - K[0, 2]) / K[0, 0] * z
        y = (v - K[1, 2]) / K[1, 1] * z
        pc = np.stack([x, y, z], -1)
        pw = pc @ c2w[:3, :3].T + c2w[:3, 3]            # world cm
        wz = np.clip(pw[:, :, 2] / 100.0, -50, 500)     # world Z, meters
        xi = np.clip(((0.5 - pw[:, :, 0] / 100.0 / 360.0) * res).astype(np.int64),
                     0, res - 1)
        yi = np.clip(((pw[:, :, 1] / 100.0 / 360.0 + 0.5) * res).astype(np.int64),
                     0, res - 1)
        flat = yi * res + xi
        okv = valid.reshape(-1)
        wzf = wz.reshape(-1)
        s = np.bincount(flat[okv], weights=wzf[okv], minlength=Q)
        c = np.bincount(flat[okv], minlength=Q)
        cell_means.append(np.where(c > 0, s / np.maximum(c, 1), np.nan))  # [Q]
        n_views += 1
    if n_views < 3:
        return None
    hz = np.stack(cell_means)                            # [V,Q], nan = no valid hit
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(hz, axis=0)
        std = np.nanstd(hz, axis=0)
        vmax = np.nanmax(hz, axis=0)
        vfrac = (~np.isnan(hz)).mean(0)
    med = np.nan_to_num(med, nan=0.0)
    std = np.nan_to_num(std, nan=0.0)
    vmax = np.nan_to_num(vmax, nan=0.0)
    height = np.stack([med, std, vmax, vfrac], 1).astype(np.float32)  # [Q,4]

    # ---- satellite forward: P3 features + deployed-path prediction ----
    sat512 = cv2.imread(str(rgb_p))
    sat1024 = cv2.resize(sat512, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    ds = DetDataSample()
    ds.set_metainfo(dict(img_shape=(1024, 1024), ori_shape=(512, 512),
                         pad_shape=(1024, 1024), scale_factor=(2.0, 2.0)))
    views = sorted(oblique.glob("view_yaw_*"))
    rgb_views, intrs, extrs = [], [], []
    for vd in views:
        ij = json.load(open(vd / "camera_intrinsics.json"))
        rgb_views.append(cv2.cvtColor(cv2.imread(str(vd / "oblique_rgb.0000.png")),
                                      cv2.COLOR_BGR2RGB))
        intrs.append(np.array([[ij["fx"], 0, ij["cx"]], [0, ij["fy"], ij["cy"]],
                               [0, 0, 1]], dtype=np.float32))
        extrs.append(np.asarray(json.load(open(vd / "camera_extrinsics.json"))
                                ["world_to_camera_cv"], dtype=np.float32))
    drone = torch.stack([torch.from_numpy(r).permute(2, 0, 1).float() / 255.0
                         for r in rgb_views])[None].to(device)
    batch = dict(drone_images=drone,
                 intrinsics=torch.tensor(np.stack(intrs))[None].to(device),
                 extrinsics=torch.tensor(np.stack(extrs))[None].to(device),
                 scene_centers=torch.zeros(1, 2).to(device))
    _attach_dual_stream_metainfo(ds, batch, 0)
    rgb_t = torch.from_numpy(cv2.cvtColor(sat1024, cv2.COLOR_BGR2RGB)
                             ).permute(2, 0, 1).float()[None].to(device)
    proc = model.data_preprocessor({"inputs": rgb_t, "data_samples": [ds.to(device)]},
                                   training=False)
    inputs = proc["inputs"][0] if isinstance(proc["inputs"], (list, tuple)) else proc["inputs"]
    with torch.no_grad():
        vo = model.backbone(inputs)
        x = model.neck(vo[1])
        satF = x[1][0].float().reshape(512, -1).T.cpu().numpy()   # [Q,512]
        outs = model.predict(inputs, proc["data_samples"], rescale=False)
    inst_p = outs[0].pred_instances
    pm = np.zeros((1024, 1024), np.uint8)
    if inst_p is not None and len(inst_p.bboxes):
        msk = inst_p.masks
        pmasks = msk.to_ndarray() if isinstance(msk, BitmapMasks) else msk.cpu().numpy()
        keep = inst_p.scores.cpu().numpy() >= 0.5
        if keep.any():
            for m in pmasks[keep]:
                pm = np.maximum(pm, m.astype(np.uint8))
    pred128 = cv2.resize(pm, (RES, RES), interpolation=cv2.INTER_NEAREST).reshape(-1) > 0

    return dict(satF=satF.astype(np.float16), height=height.astype(np.float16),
                occ=occ.reshape(-1) > 0, occ_s=occ_s, occ_l=occ_l,
                sdf=sdf, band=band, err=(occ.reshape(-1) > 0) != pred128,
                scene=Path(point_dir).name, n_inst=n_inst, n_views=n_views,
                hmap_med=med.astype(np.float32), hmap_std=std.astype(np.float32),
                pred128=pm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single-ckpt", default=SINGLE_CKPT)
    ap.add_argument("--config", default="configs/ue_dual_stream_p2p3_sparse.py")
    ap.add_argument("--depth-root", default="/data1/wangcheng/dataset/UAV_DepthBackfill_3502_3664")
    ap.add_argument("--n-fit", type=int, default=80)
    ap.add_argument("--r1-cells", type=int, default=2000)
    ap.add_argument("--r2-cells", type=int, default=1500)
    ap.add_argument("--out", default="outputs/probe_depth_gain")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing to overwrite non-empty out dir: {out}")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    import random
    SEED = 20260929
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    cfg = Config.fromfile(args.config)
    model = MODELS.build(cfg.model)
    single = torch.load(args.single_ckpt, map_location="cpu", weights_only=False)
    sd = single.get("model", single.get("state_dict"))
    model_sd = model.state_dict()
    filtered = {k: v for k, v in sd.items() if k in model_sd and model_sd[k].shape == v.shape}
    dropped = [k for k in sd if k not in filtered]
    model.load_state_dict(filtered, strict=False)
    assert dropped == [], dropped
    model.p2p3_gate.gamma_p2.data.zero_()
    model.p2p3_gate.gamma_p3.data.zero_()
    model = model.to(device).eval()
    print("single-stream baseline (gates zeroed)", flush=True)

    depth_root = Path(args.depth_root)
    scenes = sorted(d for d in depth_root.iterdir()
                    if d.is_dir() and (d / "satellite" / "satellite_rgb.0000.png").exists()
                    and (d / "satellite" / "satellite_instance_id.FinalImageM_PP_CustomStencil_GrayRaw.0000.exr").exists())
    print(f"complete scenes: {len(scenes)}", flush=True)

    fit_rows, val_rows = [], []
    n_used = 0
    t0 = time.time()
    for pd_ in scenes:
        rec = scene_record(pd_, model, device, RES)
        if rec is None:
            continue
        if n_used < args.n_fit:
            fit_rows.append(rec)
        else:
            val_rows.append(rec)
        n_used = len(fit_rows) + len(val_rows)
        if n_used % 20 == 0:
            print(f"  scenes {n_used} ({time.time()-t0:.0f}s)", flush=True)
    print(f"usable scenes: fit={len(fit_rows)} val={len(val_rows)}", flush=True)

    # ---- fit matrices ----
    sat_fit = np.concatenate([r["satF"] for r in fit_rows]).astype(np.float32)
    hgt_fit = np.concatenate([r["height"] for r in fit_rows]).astype(np.float32)
    cat_fit = np.concatenate([sat_fit, hgt_fit], 1)
    band_fit = np.concatenate([r["band"] for r in fit_rows])
    err_fit = np.concatenate([r["err"] for r in fit_rows]).astype(np.float32)
    sdf_fit = np.concatenate([r["sdf"] for r in fit_rows]).astype(np.float32)

    bi = np.where(band_fit)[0]
    if len(bi) < 1000:
        sys.exit("too few band cells in fit set")
    rng = np.random.RandomState(7000)
    sel = rng.choice(bi, size=min(40000, len(bi)), replace=False)
    print(f"R1 fit cells (band): {len(sel)} err_rate={err_fit[sel].mean():.3f}", flush=True)

    probes_auc = {}
    for arm, X in (("SAT", sat_fit), ("HEIGHT", hgt_fit), ("SAT+HEIGHT", cat_fit)):
        probes_auc[arm] = fit_logistic(X[sel], err_fit[sel], device=device)

    # sdf fit cells: 1500 per scene
    sdf_sel, off = [], 0
    for r in fit_rows:
        rng2 = np.random.RandomState(8000 + off)
        s = rng2.choice(r["sdf"].shape[0], size=args.r2_cells, replace=False)
        sdf_sel.append(off + s)
        off += r["sdf"].shape[0]
    sdf_sel = np.concatenate(sdf_sel)
    sdf_y = sdf_fit[sdf_sel]
    r2_models = {}
    for arm, X in (("SAT", sat_fit), ("HEIGHT", hgt_fit), ("SAT+HEIGHT", cat_fit)):
        r2_models[arm] = fit_ridge(X[sdf_sel], sdf_y[sdf_sel])
    r2_mlp = {}
    for arm, X in (("SAT", sat_fit), ("HEIGHT", hgt_fit), ("SAT+HEIGHT", cat_fit)):
        r2_mlp[arm] = fit_mlp(X[sdf_sel], sdf_y[sdf_sel], device=device)
    print("probes/models fitted", flush=True)

    # ---- eval pass ----
    per_scene = {}
    n_eval = 0
    for row in val_rows:
        mats = dict(SAT=row["satF"].astype(np.float32),
                    HEIGHT=row["height"].astype(np.float32))
        mats["SAT+HEIGHT"] = np.concatenate([mats["SAT"], mats["HEIGHT"]], 1)
        res = dict(scene_index=n_eval, n_inst=row["n_inst"], n_views=row["n_views"])
        band = row["band"]
        occb = row["occ"]
        if band.sum() >= 50 and len(np.unique(row["err"][band])) >= 2:
            for arm in ("SAT", "HEIGHT", "SAT+HEIGHT"):
                sc = score_logistic(probes_auc[arm], mats[arm])
                res[f"R1_{arm}"] = float(roc_auc_score(row["err"][band], sc[band]))
        for arm in ("SAT", "HEIGHT", "SAT+HEIGHT"):
            sc = score_ridge(r2_models[arm], mats[arm])
            if np.std(sc) >= 1e-9:
                res[f"R2_ridge_{arm}"] = float(np.corrcoef(sc, row["sdf"])[0, 1])
            sc = score_mlp(r2_mlp[arm], mats[arm])
            if np.std(sc) >= 1e-9:
                res[f"R2_mlp_{arm}"] = float(np.corrcoef(sc, row["sdf"])[0, 1])
        per_scene[row["scene"]] = res
        n_eval += 1
        if n_eval % 10 == 0:
            print(f"  eval {n_eval} ({time.time()-t0:.0f}s)", flush=True)

    keys = sorted({k for r in per_scene.values() for k in r if k.startswith(("R1_", "R2_"))})
    print("\n===== per-scene summary (mean±std) =====")
    summary = {}
    for k in keys:
        v = np.array([r[k] for r in per_scene.values() if k in r])
        summary[k] = dict(mean=float(v.mean()), std=float(v.std()), n=len(v))
        print(f"  {k:22s}: {v.mean():+.4f} ± {v.std():.4f} (n={len(v)})")

    def paired(a, b):
        common = [s for s in per_scene if a in per_scene[s] and b in per_scene[s]]
        d = np.array([per_scene[s][a] - per_scene[s][b] for s in common])
        rng = np.random.RandomState(BOOT_SEED)
        idx = rng.randint(0, len(d), size=(BOOT_DRAWS, len(d)))
        lo = float(np.percentile(d[idx].mean(1), 2.5))
        hi = float(np.percentile(d[idx].mean(1), 97.5))
        return dict(mean=float(d.mean()), ci=[lo, hi], improved=int((d > 0).sum()), n=len(d))

    for a, b in (("R1_SAT+HEIGHT", "R1_SAT"), ("R1_SAT+HEIGHT", "R1_HEIGHT"),
                 ("R2_ridge_SAT+HEIGHT", "R2_ridge_SAT"),
                 ("R2_mlp_SAT+HEIGHT", "R2_mlp_SAT")):
        try:
            p = paired(a, b)
            summary[f"paired {a} - {b}"] = p
            print(f"  paired {a} - {b}: {p['mean']:+.4f} "
                  f"CI[{p['ci'][0]:+.4f},{p['ci'][1]:+.4f}] improved {p['improved']}/{p['n']}")
        except Exception as ex:
            print(f"  paired {a} - {b}: failed {ex}")

    commit, dirty = subprocess_git()
    summary["manifest"] = dict(git_commit=commit, git_dirty=dirty, seed=SEED,
                               boot_seed=BOOT_SEED, single_ckpt=args.single_ckpt,
                               n_fit=len(fit_rows), n_eval=n_eval,
                               note=("depth-set scenes only; no overlap with the "
                                     "ue_coco_with_uav labeled benchmark"))
    json.dump(summary, (out / "summary.json").open("w"), indent=1)
    json.dump(per_scene, (out / "per_scene.json").open("w"), indent=1)
    print("saved", out / "summary.json", flush=True)


def subprocess_git():
    root = str(project_root)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, cwd=root).stdout.strip()
    dirty = subprocess.run(["git", "status", "--short"], capture_output=True,
                           text=True, cwd=root).stdout
    return commit, dirty


if __name__ == "__main__":
    main()
