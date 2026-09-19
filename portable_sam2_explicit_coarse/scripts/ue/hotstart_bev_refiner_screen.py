"""MHBEV-FUS-20260919-H1: BEV-to-boundary-refiner hot-start screening.

Preregistered in EXPERIMENT_GOVERNANCE.md Appendix L. The whole detector is
frozen (single-stream r1 lineage for satellite/roi_head, r3 lineage for the
frozen drone encoder + geometry BEV); only the new BEV branch inside
P2BoundaryRefiner is trained, supervised by the full materialized detector
loss. Boundary F1 (259-val, paired vs the epoch-0 model which is the exact
single-stream baseline because the BEV head is zero-initialized) is the
co-primary endpoint; AP75 delta >= -0.003 is a non-degeneration gate; the
delta_bev / delta_p2br norm ratio is the engagement monitor.

Usage (from portable_sam2_explicit_coarse/):
  CUDA_VISIBLE_DEVICES=2 python scripts/ue/hotstart_bev_refiner_screen.py
"""
import argparse
import copy
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
os.environ.setdefault("UE_DUAL_INJECT_EMBEDDING", "0")  # H1: no embedding gate

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
BOOT_DRAWS = 10000


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


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ue_dual_stream_p2p3_sparse.py")
    ap.add_argument("--data-root", default="/data1/wangcheng/dataset/ue_coco_with_uav")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--out", default="/data1/wangcheng/checkpoint/worktrees_dual-stream-dev_probe/MHBEV-FUS-20260919-H1-screen")
    return ap


def main():
    args = build_arg_parser().parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    cfg = Config.fromfile(args.config)
    # H1: disable pyramid/embedding injection paths; enable the refiner BEV branch.
    cfg.model.roi_head.mask_head.p2_boundary_refiner.bev_channels = 128
    cfg.model.roi_head.mask_head.p2_boundary_refiner.bev_spatial_scale = 0.125
    model = MODELS.build(cfg.model)

    dual = torch.load(DUAL_CKPT, map_location="cpu", weights_only=False)
    m1 = model.load_state_dict(dual.get("model", dual.get("state_dict")), strict=False)
    single = torch.load(SINGLE_CKPT, map_location="cpu", weights_only=False)
    single_sd = single.get("model", single.get("state_dict"))
    # drop the refiner fusion conv (input-width changed by the BEV branch concat
    # is NOT used in this design -- fusion untouched; only guard anyway) and any
    # dual-only keys; everything else overwrites with single-stream weights.
    filtered = {k: v for k, v in single_sd.items()
                if model.state_dict()[k].shape == v.shape}
    dropped = [k for k in single_sd if k not in filtered]
    m2 = model.load_state_dict(filtered, strict=False)
    print(f"dual load: missing={len(m1.missing_keys)} unexpected={len(m1.unexpected_keys)}")
    print(f"single overwrite: matched={len(filtered)} dropped(shape)={dropped}")
    model = model.to(device)

    # freeze everything, train only the new BEV branch
    for p in model.parameters():
        p.requires_grad_(False)
    refiner = model.roi_head.mask_head.p2_boundary_refiner
    trainable = [p for n, p in refiner.named_parameters() if n.startswith(("bev_",))]
    for p in trainable:
        p.requires_grad_(True)
    n_train_par = sum(p.numel() for p in trainable)
    print(f"trainable params: {n_train_par} in {len(trainable)} tensors "
          f"({[n for n, _ in refiner.named_parameters() if n.startswith('bev_')]}")

    # provider: refiner pulls the current-batch BEV from the dual model
    model_core = model
    refiner.bev_provider = lambda: getattr(model_core, "_last_bev", None)
    orig_extract = model_core.extract_feat
    def extract_feat_with_bev(batch_inputs, batch_data_samples=None):
        x, emb, pos, hrf = orig_extract(batch_inputs, batch_data_samples)
        model_core._last_bev = getattr(model_core.geometry_bev, "_last_bev_out", None)
        return x, emb, pos, hrf
    model_core.extract_feat = extract_feat_with_bev
    # wrap geometry_bev to expose its output
    orig_bev_forward = model_core.geometry_bev.forward
    def bev_forward(*a, **kw):
        b = orig_bev_forward(*a, **kw)
        model_core.geometry_bev._last_bev_out = b
        return b
    model_core.geometry_bev.forward = bev_forward
    # pyramid gates stay exactly zero and frozen
    model_core.p2p3_gate.gamma_p2.data.zero_()
    model_core.p2p3_gate.gamma_p3.data.zero_()

    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WD)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    train_ds = UEDualStreamDataset(args.data_root, "train", image_size=(1024, 1024),
                                   train_subset=0.1797752808988764, seed=44, load_uav=True)
    val_ds = UEDualStreamDataset(args.data_root, "val", image_size=(1024, 1024), load_uav=True)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2,
                              collate_fn=ue_dual_collate, drop_last=False)
    print(f"train={len(train_ds)} val={len(val_ds)} epochs={args.epochs}")

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

    def eval_epoch():
        """Full 259-val eval: COCO segm + per-scene boundary F1 (union@0.5)."""
        was_training = model.training
        model.eval()
        coco_dt, bf1s = [], []
        gt_imgs, gt_anns = [], []
        aid = 1
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
                    pmasks = (msk.to_ndarray() if isinstance(msk, BitmapMasks)
                              else msk.cpu().numpy())
                    scs = inst.scores.cpu().numpy()
                    keep = scs >= 0.5
                    pm = pmasks[keep].any(0) if keep.any() else np.zeros((1024, 1024), bool)
                    for m, s in zip(maskUtils_encode_all(pmasks), scs):
                        coco_dt.append(dict(image_id=i, category_id=1, segmentation=m,
                                            score=float(s)))
                else:
                    pm = np.zeros((1024, 1024), bool)
                gtm = item["gt_masks"]
                gtocc = np.zeros((1024, 1024), np.uint8)
                for m in gtm:
                    gtocc = np.maximum(gtocc, m)
                bf1s.append(boundary_f1(pm, gtocc > 0))
                gt_imgs.append(dict(id=i, file_name=item["img_metas"]["scene_id"],
                                    height=1024, width=1024))
                for m in gtm:
                    rle = maskUtils.encode(np.asfortranarray(m))
                    rle["counts"] = rle["counts"].decode()
                    ys, xs = np.where(m)
                    gt_anns.append(dict(id=aid, image_id=i, category_id=1, segmentation=rle,
                                        bbox=[float(xs.min()), float(ys.min()),
                                              float(xs.max()-xs.min()+1), float(ys.max()-ys.min()+1)],
                                        area=int(m.sum()), iscrowd=0))
                    aid += 1
        model.train(was_training)
        gt_json = dict(images=gt_imgs, annotations=gt_anns,
                       categories=[dict(id=1, name="building")])
        gp = out / "gt_val259.json"
        if not gp.exists():
            json.dump(gt_json, gp.open("w"))
        coco_gt = COCO(str(gp))
        ap = ap75 = ap50 = float("nan")
        if coco_dt:
            coco_dt_l = coco_gt.loadRes(coco_dt)
            ev = COCOeval(coco_gt, coco_dt_l, "segm")
            ev.params.imgIds = list(range(len(val_ds)))
            ev.evaluate(); ev.accumulate()
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                ev.summarize()
            ap, ap50, ap75 = float(ev.stats[0]), float(ev.stats[1]), float(ev.stats[2])
        return dict(boundary_f1=float(np.mean(bf1s)), AP=ap, AP50=ap50, AP75=ap75,
                    per_scene_bf1=[float(x) for x in bf1s])

    # ================= epoch-0 baseline =================
    print("===== epoch-0 baseline eval (exact single-stream model) =====", flush=True)
    history = [dict(epoch=0, **{k: v for k, v in eval_epoch().items()
                                if k != "per_scene_bf1"})]
    print("epoch0:", history[0], flush=True)

    # ================= screening training =================
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        for bi, batch in enumerate(train_loader):
            inputs, data_samples = make_batch(batch, device)
            with torch.amp.autocast("cuda"):
                loss_dict = model.loss(inputs, data_samples)
                loss_dict = _materialize_p2_boundary_refiner_loss(loss_dict, model, False)
            parts = [v for k, v in loss_dict.items()
                     if "loss" in k and isinstance(v, torch.Tensor)]
            loss = sum(parts) / GRAD_ACCUM
            scaler.scale(loss).backward()
            if (bi + 1) % GRAD_ACCUM == 0 or (bi + 1) == len(train_loader):
                step += 1
                total_steps = args.epochs * math.ceil(len(train_loader) / GRAD_ACCUM)
                warm = min(1.0, (step + 1) / (WARMUP_STEPS + 1))
                cosine = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(step / total_steps, 1.0)))
                for g in optimizer.param_groups:
                    g["lr"] = LR * warm * cosine
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        ev = eval_epoch()
        ev["epoch"] = epoch
        ev["train_seconds"] = round(time.time() - t0, 1)
        history.append({k: v for k, v in ev.items() if k != "per_scene_bf1"})
        json.dump(history, (out / "history.json").open("w"), indent=1)
        print(f"epoch {epoch}: BF1={ev['boundary_f1']:.4f} AP={ev['AP']:.4f} "
              f"AP75={ev['AP75']:.4f} ({ev['train_seconds']}s)", flush=True)

    json.dump(dict(decision_pending="bootstrap gate applied in result review",
                   history=history), (out / "summary_raw.json").open("w"), indent=1)
    print("screening complete; history at", out / "history.json")


def maskUtils_encode_all(pmasks):
    out = []
    for m in pmasks:
        rle = maskUtils.encode(np.asfortranarray(m.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode()
        out.append(rle)
    return out


if __name__ == "__main__":
    main()
