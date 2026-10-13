"""Size-stratified marginal-info probe: does DBEV add occupancy information for
SMALL buildings specifically? (Task-pivot evidence for small-instance
segmentation.) Frozen single-stream features; read-only.
"""
import argparse
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
os.environ.setdefault("UE_DUAL_INJECT_EMBEDDING", "0")

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

if "DetDataPreprocessor" not in MMENGINE_MODELS:
    MMENGINE_MODELS.register_module(name="DetDataPreprocessor", module=DetDataPreprocessor)

from data.ue_dual_stream_dataset import UEDualStreamDataset
from sklearn.metrics import roc_auc_score

RES = 128
SMALL_AREA = 5000  # px^2 at 1024 (~70x70 px, ~600 m^2 footprint)


def extract(item, model, device):
    ds = DetDataSample(); ds.set_metainfo(item["img_metas"])
    ds.set_metainfo({"drone_images": item["drone_images"], "intrinsics": item["intrinsics"],
                     "extrinsics": item["extrinsics"], "scene_center": item["scene_center"]})
    proc = model.data_preprocessor(
        {"inputs": [item["img"].to(device)], "data_samples": [ds.to(device)]}, training=False)
    inputs = proc["inputs"][0] if isinstance(proc["inputs"], (list, tuple)) else proc["inputs"]
    with torch.no_grad():
        vo = model.backbone(inputs)
        x = model.neck(vo[1])
        sat = x[1][0].float()
        drone = item["drone_images"][None].to(device)
        drone = (drone - drone.new_tensor([.485, .456, .406])[None, None, :, None, None]) / \
                drone.new_tensor([.229, .224, .225])[None, None, :, None, None]
        _, fpn = model.drone_encoder(drone.reshape(drone.shape[0] * drone.shape[1], *drone.shape[2:]))
        fpn = sorted(tuple(fpn[-4:]), key=lambda t: t.shape[-2] * t.shape[-1], reverse=True)
        h = w = RES
        u = (torch.arange(w, device=device) + .5) * (512.0 / w)
        v = (torch.arange(h, device=device) + .5) * (512.0 / h)
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        gx = -(vv / 512.0 - .5) * 360.0
        gy = (uu / 512.0 - .5) * 360.0
        c = item["scene_center"].to(device)
        pts = torch.stack((gx.reshape(-1) * 100 + c[0], gy.reshape(-1) * 100 + c[1],
                           torch.zeros(h * w, device=device), torch.ones(h * w, device=device)), -1)
        per_level = []
        for feat in fpn:
            acc = torch.zeros(256, h * w, device=device)
            cnt = torch.zeros(h * w, device=device)
            for vi in range(5):
                camc = pts @ item["extrinsics"][vi].cuda().T
                z = camc[:, 2]
                pix = camc[:, :3] @ item["intrinsics"][vi].cuda().T
                uv = pix[:, :2] / pix[:, 2:3].clamp_min(1e-6)
                ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < 512) & (uv[:, 1] >= 0) & (uv[:, 1] < 512)
                grid = ((uv / 512.0) * 2.0 - 1.0)
                grid[~ok] = 0.0
                smp = F.grid_sample(feat[vi:vi + 1].float(), grid[None, :, None, :],
                                    mode="bilinear", padding_mode="zeros",
                                    align_corners=False)[0, :, :, 0]
                acc += smp * ok[None].float()
                cnt += ok.float()
            per_level.append(acc / cnt.clamp_min(1.0))
        direct = torch.cat(per_level, 0).reshape(1024, h, w)
    return (sat.float().cpu().numpy().astype(np.float16),
            direct.cpu().numpy().astype(np.float16))


import torch.nn.functional as F  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single-ckpt", default="work_dirs/ue/ue_single_c5v2_control_tr128_va259_s44_r1/best_model_epoch86.pth")
    ap.add_argument("--config", default="configs/ue_dual_stream_p2p3_sparse.py")
    ap.add_argument("--data-root", default="/data1/wangcheng/dataset/ue_coco_with_uav")
    ap.add_argument("--n-fit", type=int, default=80)
    ap.add_argument("--n-val", type=int, default=80)
    ap.add_argument("--small-area", type=int, default=SMALL_AREA)
    ap.add_argument("--out", default="outputs/probe_size_strat")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    cfg = Config.fromfile(args.config)
    model = MODELS.build(cfg.model)
    dual = torch.load(DUAL_FALLBACK := "work_dirs/ue/ue_dual_c5v2_p2p3_emb_tr128_va259_s44_r1/best_model_epoch85.pth",
                      map_location="cpu", weights_only=False)
    model.load_state_dict(dual.get("model", dual.get("state_dict")), strict=False)
    single = torch.load(args.single_ckpt, map_location="cpu", weights_only=False)
    single_sd = single.get("model", single.get("state_dict"))
    model_sd = model.state_dict()
    filtered = {k: v for k, v in single_sd.items() if k in model_sd and model_sd[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)
    model = model.to(device).eval()
    model.p2p3_gate.gamma_p2.data.zero_()
    model.p2p3_gate.gamma_p3.data.zero_()

    train_ds = UEDualStreamDataset(args.data_root, "train", image_size=(1024, 1024),
                                   train_subset=0.1797752808988764, seed=44, load_uav=True)
    val_ds = UEDualStreamDataset(args.data_root, "val", image_size=(1024, 1024), load_uav=True)

    def strata_masks(item):
        """Per-instance small/large occupancy maps at 128."""
        occ_s = np.zeros((RES, RES), np.uint8)
        occ_l = np.zeros((RES, RES), np.uint8)
        n_s = n_l = 0
        for m in item["gt_masks"]:
            area = int(m.sum())
            r128 = cv2.resize(m, (RES, RES), interpolation=cv2.INTER_NEAREST)
            if area < args.small_area:
                occ_s = np.maximum(occ_s, r128); n_s += 1
            else:
                occ_l = np.maximum(occ_l, r128); n_l += 1
        return occ_s, occ_l, n_s, n_l

    # ---- fit pass ----
    print("=== fit pass ===", flush=True)
    fits = {("SAT", "small"): [], ("SAT", "large"): [],
            ("DBEV", "small"): [], ("DBEV", "large"): [],
            ("SAT+DBEV", "small"): [], ("SAT+DBEV", "large"): []}
    yblocks = {"small": [], "large": []}
    n_fit = 0
    t0 = time.time()
    for idx in range(len(train_ds)):
        if n_fit >= args.n_fit: break
        item = train_ds[idx]
        if len(item["gt_labels"]) == 0: continue
        sat, dbv = extract(item, model, device)
        occ_s, occ_l, ns, nl = strata_masks(item)
        if occ_s.sum() == 0 or occ_l.sum() == 0: continue
        satF = sat.reshape(512, -1).T.astype(np.float32)
        dbvF = dbv.reshape(1024, -1).T.astype(np.float32)
        catF = np.concatenate([satF, dbvF], 1)
        rng = np.random.RandomState(9000 + idx)
        pos_s = np.where(occ_s.reshape(-1) > 0)[0]
        pos_l = np.where(occ_l.reshape(-1) > 0)[0]
        neg = np.where((occ_s.reshape(-1) == 0) & (occ_l.reshape(-1) == 0))[0]
        if len(pos_s) < 400 or len(pos_l) < 400 or len(neg) < 400: continue
        sel_s = np.concatenate([rng.choice(pos_s, 400, replace=False),
                                rng.choice(neg, 400, replace=False)])
        sel_l = np.concatenate([rng.choice(pos_l, 400, replace=False),
                                rng.choice(neg, 400, replace=False)])
        for arm, F_ in (("SAT", satF), ("DBEV", dbvF), ("SAT+DBEV", catF)):
            fits[(arm, "small")].append(F_[sel_s]); fits[(arm, "large")].append(F_[sel_l])
        yblocks["small"].append(occ_s.reshape(-1)[sel_s])
        yblocks["large"].append(occ_l.reshape(-1)[sel_l])
        n_fit += 1
        if n_fit % 10 == 0: print(f"  fit {n_fit} ({time.time()-t0:.0f}s)", flush=True)

    import torch.nn.functional as F
    from torch import nn
    def fit_logistic(X, y, device="cuda", iters=300, lr=0.05):
        mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True) + 1e-4
        Xt = torch.tensor((X - mu) / sd, dtype=torch.float32, device=device)
        yt = torch.tensor(y, dtype=torch.float32, device=device)
        lin = nn.Linear(X.shape[1], 1).to(device)
        opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=0.1)
        pos = float(yt.mean())
        pw = torch.tensor((1 - pos) / max(pos, 1e-6), device=device).clamp(0.2, 5)
        for _ in range(iters):
            opt.zero_grad()
            F.binary_cross_entropy_with_logits(lin(Xt).squeeze(-1), yt, pos_weight=pw).backward()
            opt.step()
        return dict(w=lin.weight.detach()[0].cpu().numpy(), b=float(lin.bias.item()), mu=mu[0], sd=sd[0])

    probes = {}
    for (arm, strat), blocks in fits.items():
        X = np.concatenate(blocks)
        y = np.concatenate(yblocks[strat]).astype(np.float32)
        # ys[strat] list already aligned per-scene with blocks
        probes[(arm, strat)] = fit_logistic(X, y, device)
    print("probes fitted", flush=True)

    # ---- eval pass ----
    print("=== eval pass ===", flush=True)
    per_scene = {}
    n_val = 0
    for idx in range(min(args.n_val, len(val_ds))):
        item = val_ds[idx]
        if len(item["gt_labels"]) == 0: continue
        sat, dbv = extract(item, model, device)
        occ_s, occ_l, ns, nl = strata_masks(item)
        if occ_s.sum() == 0 and occ_l.sum() == 0: continue
        satF = sat.reshape(512, -1).T.astype(np.float32)
        dbvF = dbv.reshape(1024, -1).T.astype(np.float32)
        catF = np.concatenate([satF, dbvF], 1)
        mats = dict(SAT=satF, DBEV=dbvF, **{"SAT+DBEV": catF})
        res = dict(scene_index=idx)
        for strat, occ in (("small", occ_s), ("large", occ_l)):
            yb = occ.reshape(-1)
            if len(np.unique(yb)) < 2: continue
            for arm in ("SAT", "DBEV", "SAT+DBEV"):
                pr = probes[(arm, strat)]
                sc = (mats[arm] / pr["sd"] - pr["mu"] / pr["sd"]) @ pr["w"] + pr["b"]
                if np.std(sc) < 1e-9: continue
                res[f"{arm}_{strat}"] = float(roc_auc_score(yb, sc))
        per_scene[item["img_metas"]["scene_id"]] = res
        n_val += 1
        if n_val % 10 == 0: print(f"  eval {n_val}", flush=True)

    keys = sorted({k for r in per_scene.values() for k in r if not r.get("skipped")})
    print("\n===== per-scene AUC (mean±std) =====")
    summary = {}
    for k in keys:
        v = np.array([r[k] for r in per_scene.values() if k in r])
        summary[k] = dict(mean=float(v.mean()), std=float(v.std()), n=len(v))
        print(f"  {k:16s}: {v.mean():.4f} ± {v.std():.4f}")
    for strat in ("small", "large"):
        d = np.array([r[f"SAT+DBEV_{strat}"] - r[f"SAT_{strat}"]
                      for r in per_scene.values()
                      if f"SAT+DBEV_{strat}" in r and f"SAT_{strat}" in r])
        rng = np.random.RandomState(20260930)
        idx = rng.randint(0, len(d), size=(10000, len(d)))
        lo = float(np.percentile(d[idx].mean(1), 2.5))
        hi = float(np.percentile(d[idx].mean(1), 97.5))
        summary[f"marginal_{strat}"] = dict(mean=float(d.mean()), ci=[lo, hi],
                                            improved=int((d > 0).sum()), n=len(d))
        print(f"  marginal(SAT+DBEV - SAT) {strat}: {d.mean():+.4f} CI[{lo:+.4f},{hi:+.4f}] "
              f"improved {int((d>0).sum())}/{len(d)}")
    summary["manifest"] = dict(ckpt=args.single_ckpt, small_area=args.small_area,
                               n_fit=n_fit, n_val=n_val)
    json.dump(summary, (out / "summary.json").open("w"), indent=1)
    json.dump(per_scene, (out / "per_scene.json").open("w"), indent=1)
    print("saved", out / "summary.json")


from data.ue_dual_stream_dataset import UEDualStreamDataset  # noqa: E402

if __name__ == "__main__":
    main()
