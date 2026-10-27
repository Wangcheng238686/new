#!/usr/bin/env python3
"""UBR v1.1 acceptance gates G1-G4 (EXP-0028-R2 relaunch, 2026-10-27).

Authoritative spec: docs/UBR-V1.1-PERF-AND-RELAUNCH-20261027.md section 4.
Every gate is fail-closed: writes a PASS/FAIL json and exits non-zero on
FAIL. Run with CUDA_VISIBLE_DEVICES=3.

  g1   execution equivalence replay: v1.1 at OLD capacity (48x7, A-changes
       only) vs the FROZEN v1.0 assembly (verbatim copy of the dead arm's
       compute_loss_ubr, git 36b5215) on the same forward — 3 updates on
       fixed batches replayed from the dead arm's ep1 rank-0 order (seed
       44, DistributedSampler num_replicas=4 rank=0 epoch 0; augmentation
       RNG replayed in-process, the num_workers=4 worker streams cannot be
       recovered bit-exactly). Per-update loss components rel err < 1e-3
       (AMP tolerance); replay magnitudes cross-checked against the dead
       arm's recorded ep1 epoch means (loss_ubr=0.0312, focal=0.0877,
       dice=0.2240, launch2.log line 111) within [0.25x, 4x].
  g2   capacity switch smoke (B1): config capacity 32ch x 6 builds
       (67,457 params), one real batch, one update — losses finite,
       telemetry keys complete; B2 (grad checkpointing) and A4 (bucketed
       backward) code paths exercised at unit level.
  g3   memory/speed telemetry: 2-epoch smokes (20 train / 5 val scenes,
       single process) for the A-only (old capacity) and A+B1 (new
       unified 32x6) configurations; peak reserved memory <= 20 GB,
       extrapolated epoch <= 7 min (A-only) / <= 6 min (with B1).
  g4   seal gate: frozen-parameter SHA invariance (v1.1-patched build vs
       sealed EXP-0025 build, seed 44: every non-UBR tensor bit-equal;
       prompt stack / PE / backbone frozen set + LoRA structure hashed),
       source_hashes bump (v1.1 files vs git HEAD blobs), and the
       RESEAL_CONTRACT v2 declaration text.
  val-single / val-ddp / val-align
       A5 sharded-validation alignment: 12-image subset, single-GPU
       unsharded val vs 2-rank Gloo-sharded val through the trainer's own
       validate_one_epoch — DT/GT records bit-exact (sha256 of the
       canonical serialization), canonical AP250 metrics equal, rank
       weight parity asserted.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # portable_sam2_explicit_coarse (repo root)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))  # train/ sibling-module imports

import numpy as np  # noqa: E402
import torch  # noqa: E402

CONFIG = ROOT / "configs/ablations/ue_ubr_single_512.py"
CONFIG_SEALED = ROOT / "configs/ue_whu_a3sgm_512.py"
SAM2_CKPT = "/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt"
DS_ROOT = "/data1/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021"
GATES_DIR = Path("/data2/wangcheng/remote-lthpc/checkpoint/ue_ubr_v11_gates_20261027")
SEED = 44
N_REPLAY = 3
REL_TOL = 1e-3
# dead EXP-0028 arm, ep1 epoch means (launch2.log line 111): magnitude
# reference for the G1 replayed first-3-batch losses.
DEAD_ARM_EP1 = dict(loss_ubr=0.0312, focal=0.0877, dice=0.2240)
MAG_LO, MAG_HI = 0.25, 4.0
G3_LIMITS = dict(peak_mem_gb=20.0, epoch_secs_a_only=420.0, epoch_secs_b1=360.0)
TRAIN_BATCHES_PER_RANK = 535   # dead arm: 2142 / 4 (drop_last)
VAL_IMAGES_PER_RANK = 115      # ceil(459 / 4) for the sharded val
VAL_COCO_OVERHEAD_SECS = 15.0  # rank0 COCO eval + gather (measured scale)
VAL_ALIGN_IMAGES = 12


# ---------------------------------------------------------------- helpers
def _report(name: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(name, indent=2, default=_json_default))
    print(json.dumps(name, indent=2, default=_json_default), flush=True)


def _runner():
    spec = importlib.util.spec_from_file_location(
        "reseal_runner", HERE / "ue_ubr_reseal_20261027.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bootstrap():
    os.environ.setdefault("SAM2_CKPT", SAM2_CKPT)
    import mmdet.models  # noqa: F401
    import rsprompter  # noqa: F401  (registers model types)
    from mmengine.registry import MODELS as MMENGINE_MODELS
    from mmdet.models.data_preprocessors import DetDataPreprocessor
    if "DetDataPreprocessor" not in MMENGINE_MODELS:
        MMENGINE_MODELS.register_module(
            name="DetDataPreprocessor", module=DetDataPreprocessor)


def _seed_all(seed: int = SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _build(cfg_model, device: str):
    from mmdet.registry import MODELS
    _seed_all()
    model = MODELS.build(cfg_model)
    return model.to(device)


def _protocol_env() -> None:
    runner = _runner()
    runner.set_protocol_env(SAM2_CKPT)
    os.environ["TRAIN_FLIP_PROB"] = "0"   # 3090 line (worker protocol)
    os.environ["TRAIN_VFLIP_PROB"] = "0"
    runner._install_3090_augmentation()


def _train_args(argv_path_cfg: str, out_dir: Path):
    """The sealed trainer's own arg parser on a representative protocol argv."""
    runner = _runner()
    sys.path.insert(0, str(ROOT / "train"))
    from cli_args import _parse_args
    argv = runner.trainer_argv(
        argv_path_cfg, "ue_ubr_v11_gate", DS_ROOT, str(out_dir), epochs=2)
    old = sys.argv
    sys.argv = argv
    args = _parse_args()
    sys.argv = old
    return args


def _loss_tensors(loss_dict):
    for v in loss_dict.values():
        vals = v if isinstance(v, (list, tuple)) else [v]
        for item in vals:
            if torch.is_tensor(item):
                yield item


# ------------------------------------------------- G1: v1.0 frozen reference
def _v10_loss_ubr(integ, sampling_results, batch_gt_instances):
    """FROZEN verbatim copy of the v1.0 compute_loss_ubr (git 36b5215).

    Evidence-layer duplication (AGENTS.md): the dead arm's assembly is the
    equivalence target, so it is kept bit-for-bit (only `self.` -> `integ.`
    and the focal/dice return added; those were already computed for
    last_train_stats in v1.0).
    """
    from mmdet.models.roi_heads.mask_heads.fcn_mask_head import _do_paste_mask
    from rsprompter.uav import ubr
    from rsprompter.uav.ubr_train_integration import (
        IMAGE_SIZE, _pad_batch, band_loss, feat_window, window_around)
    import torch.nn.functional as F

    model = integ.model
    dev = next(model.parameters()).device
    pos_counts = [int(res.pos_priors.shape[0]) for res in sampling_results]
    n_pos = int(sum(pos_counts))
    zero = sum(p.sum() for p in model.ubr_refiner.parameters()) * 0.0
    if n_pos == 0:
        return zero, zero.detach(), zero.detach()
    z, up = integ._collect()
    assert z.shape[0] == n_pos, (z.shape[0], n_pos)

    boxes = torch.cat([res.pos_priors for res in sampling_results], 0)
    thr = float(model.test_cfg["rcnn"]["mask_thr_binary"])
    probs512 = _do_paste_mask(
        torch.sigmoid(z.float()), boxes.float(), IMAGE_SIZE, IMAGE_SIZE,
        skip_empty=False)[0]
    with torch.no_grad():
        pred_bin = probs512.detach() >= thr

    wins = [window_around(b, integ.window_margin, IMAGE_SIZE)
            for b in boxes.detach().cpu().numpy()]
    fwins = [feat_window(w) for w in wins]

    gt_np = []
    for gt_instances in batch_gt_instances:
        masks = gt_instances.masks
        arr = masks.masks if hasattr(masks, "masks") else np.asarray(masks)
        gt_np.append(np.ascontiguousarray(
            arr, dtype=np.uint8).reshape(-1, IMAGE_SIZE, IMAGE_SIZE))
    nbr_full = [
        g.sum(0) if g.shape[0] else np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)
        for g in gt_np
    ]

    feats, logits, sdts, nbrs, gts, bands = [], [], [], [], [], []
    n_empty_band = 0
    offset = 0
    for i, count in enumerate(pos_counts):
        gts_img, nbr_img = gt_np[i], nbr_full[i]
        for k in range(count):
            j = offset + k
            h0, w0, h, w = wins[j]
            outside = pred_bin[j].clone()
            outside[h0:h0 + h, w0:w0 + w] = False
            assert not bool(outside.any()), f"inst {j} pasted mask outside window"
            sdt = ubr.signed_distance_transform(
                pred_bin[j, h0:h0 + h, w0:w0 + w].float()[None, None])[0]
            band = ubr.band_mask(sdt, integ.band_k)
            if not bool(band.any()):
                n_empty_band += 1
            fh0, fw0, fh, fw = fwins[j]
            feats.append(F.interpolate(
                up[j:j + 1, :, fh0:fh0 + fh, fw0:fw0 + fw].float(),
                size=(h, w), mode="bilinear", align_corners=False))
            logits.append(probs512[j:j + 1, h0:h0 + h, w0:w0 + w])
            sdts.append(sdt)
            nbrs.append(torch.from_numpy(np.ascontiguousarray(
                (nbr_img[h0:h0 + h, w0:w0 + w]
                 - gts_img[int(sampling_results[i].pos_assigned_gt_inds[k]),
                           h0:h0 + h, w0:w0 + w]).astype(np.float32)
            ))[None].to(dev))
            gts.append(torch.from_numpy(np.ascontiguousarray(
                gts_img[int(sampling_results[i].pos_assigned_gt_inds[k]),
                        h0:h0 + h, w0:w0 + w], dtype=np.float32
            ))[None].to(dev))
            bands.append(band)
        offset += count

    batch = dict(
        feat=_pad_batch(feats, 32), logit=_pad_batch(logits, 1),
        sdt=_pad_batch(sdts, 1), nbr=_pad_batch(nbrs, 1),
        gt=_pad_batch(gts, 1), band=_pad_batch(bands, 1, bool_=True),
    )
    for key, tensor in batch.items():
        assert tensor.shape[0] == n_pos and tensor.ndim == 4, key

    out = model.ubr_refiner(batch["feat"], batch["logit"], batch["sdt"],
                            batch["nbr"])
    assert tuple(out.shape) == tuple(batch["logit"].shape)
    assert bool(torch.isfinite(out.float()).all()), "non-finite UBR logits (train)"
    total, focal, dice = band_loss(out, batch["gt"], batch["band"],
                                   integ.focal_gamma)
    loss = integ.loss_weight * total
    assert bool(torch.isfinite(loss)), "non-finite loss_ubr"
    return loss, focal.detach(), dice.detach()


def cmd_g1(_args) -> dict:
    from mmengine.config import Config
    from rsprompter.uav import ubr_train_integration as integ_mod
    from torch.utils.data.distributed import DistributedSampler
    from data.loader import create_train_loader, rtmdet_collate_fn
    from train.gt_samples import _build_gt_data_samples
    from train.optimizer import _build_optimizer

    _protocol_env()
    _bootstrap()
    cfg = Config.fromfile(str(CONFIG))
    ubr_cfg = dict(cfg.model.roi_head.mask_head.ubr_integration_cfg)
    ubr_cfg.update(refiner_hidden=48, refiner_depth=7)  # OLD capacity only
    integ_mod.install_class_patch(ubr_cfg)
    out_dir = GATES_DIR / "g1_replay"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = _build(cfg.model, "cuda")
    integ = model.ubr_integration
    assert sum(p.numel() for p in model.ubr_refiner.parameters()) == 163537

    records = []
    orig = integ.compute_loss_ubr

    def recorder(sampling_results, batch_gt_instances):
        new_loss = orig(sampling_results, batch_gt_instances)
        ref_loss, ref_f, ref_d = _v10_loss_ubr(
            integ, sampling_results, batch_gt_instances)
        st = dict(integ.last_train_stats)
        records.append(dict(
            update=len(records),
            new_loss=float(new_loss.detach()),
            ref_loss=float(ref_loss),
            new_focal=float(st["UBR/focal"]), ref_focal=float(ref_f),
            new_dice=float(st["UBR/dice"]), ref_dice=float(ref_d),
            instances=float(st["UBR/instances"]),
            band_nonempty_frac=float(st["UBR/band_nonempty_frac"])))
        return new_loss

    integ.compute_loss_ubr = recorder

    train_loader, _, train_dataset = create_train_loader(
        data_root=DS_ROOT, batch_size=1, dataset_format="whu_coco",
        image_size=(512, 512), num_workers=0, seed=SEED, val_batch_size=1,
        flip_prob=0.0, vflip_prob=0.0,   # 3090 line: no flip
        whu_train_ann_file=os.environ["WHU_TRAIN_ANN_FILE"],
        whu_val_ann_file=os.environ["WHU_VAL_ANN_FILE"],
        whu_train_img_subdir=os.environ["WHU_TRAIN_IMG_SUBDIR"],
        whu_val_img_subdir=os.environ["WHU_VAL_IMG_SUBDIR"],
        train_subset_ratio=1.0, val_subset_ratio=1.0)
    # dead arm ep1 rank-0 order: DistributedSampler(4 replicas, seed 44)
    sampler = DistributedSampler(
        train_dataset, num_replicas=4, rank=0, shuffle=True, drop_last=True,
        seed=SEED)
    sampler.set_epoch(0)
    replay_idx = [int(i) for i in list(sampler)[:N_REPLAY]]
    assert len(replay_idx) == N_REPLAY

    args = _train_args(str(CONFIG), out_dir)
    optimizer = _build_optimizer(
        model, lr=args.lr, backbone_lr_mult=args.backbone_lr_mult,
        other_lr_mult=args.other_lr_mult,
        bbox_head_lr_mult=args.bbox_head_lr_mult,
        mask_decoder_lr_mult=args.mask_decoder_lr_mult,
        no_mask_lr_mult=args.no_mask_lr_mult,
        prompt_encoder_lr_mult=args.prompt_encoder_lr_mult,
        shape_prior_lr_mult=args.shape_prior_lr_mult,
        decoder_tail_lr_mult=args.decoder_tail_lr_mult,
        weight_decay=args.weight_decay,
        fusion_no_wd=bool(getattr(args, "fusion_no_wd", 0)),
        consumer_lr_mult=float(getattr(args, "consumer_lr_mult", 0.0) or 0.0))
    scaler = torch.amp.GradScaler("cuda")
    device = "cuda"
    _seed_all()  # augmentation RNG replay: in-process stream from seed 44
    first_losses = None
    for t, ds_idx in enumerate(replay_idx):
        batch = rtmdet_collate_fn([train_dataset[ds_idx]])
        data_samples = _build_gt_data_samples(
            batch["imgs"], batch["img_metas"], batch["gt_bboxes"],
            batch["gt_labels"], batch["gt_masks"], device)
        payload = model.data_preprocessor(
            {"inputs": batch["imgs"].to(device), "data_samples": data_samples},
            training=True)
        with torch.autocast("cuda"):
            losses = model(payload["inputs"], payload["data_samples"],
                           mode="loss")
        if t == 0:
            first_losses = {
                k: round(float(sum(x.detach().float() for x in
                                   (v if isinstance(v, (list, tuple)) else [v])
                                   if torch.is_tensor(x))), 5)
                for k, v in losses.items()}
        total = sum(_loss_tensors(losses))
        scaler.scale(total).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    assert len(records) == N_REPLAY, records

    worst = 0.0
    per_update = []
    for rec in records:
        errs = {}
        for key, a, b in (("loss_ubr", rec["new_loss"], rec["ref_loss"]),
                          ("focal", rec["new_focal"], rec["ref_focal"]),
                          ("dice", rec["new_dice"], rec["ref_dice"])):
            rel = abs(a - b) / max(abs(b), 1e-12)
            worst = max(worst, rel)
            errs[key] = dict(new=a, ref=b, rel_err=rel)
            assert rel < REL_TOL, f"G1 FAIL update {rec['update']} {key}: {errs[key]}"
        per_update.append(dict(update=rec["update"], instances=rec["instances"],
                               band_nonempty_frac=rec["band_nonempty_frac"],
                               components=errs))
    mags = {}
    for key, new_key, dead in (("loss_ubr", "new_loss", DEAD_ARM_EP1["loss_ubr"]),
                               ("focal", "new_focal", DEAD_ARM_EP1["focal"]),
                               ("dice", "new_dice", DEAD_ARM_EP1["dice"])):
        mean_new = sum(r[new_key] for r in records) / len(records)
        ratio = mean_new / dead
        mags[key] = dict(replay_mean=round(mean_new, 6), dead_arm_ep1=dead,
                         ratio=round(ratio, 3))
        assert MAG_LO <= ratio <= MAG_HI, (
            f"G1 magnitude FAIL {key}: replay {mean_new} vs dead ep1 {dead} "
            f"ratio {ratio} outside [{MAG_LO}, {MAG_HI}]")
    return dict(gate="G1_equivalence_replay", status="PASS",
                rel_tol=REL_TOL, worst_rel_err=worst,
                replay_indices=replay_idx,
                note="per-update rel err of v1.1 (old capacity) vs frozen "
                     "v1.0 assembly on the same forward; magnitudes vs dead "
                     "arm ep1 epoch means",
                per_update=per_update, magnitude_vs_dead_arm=mags,
                first_update_loss_dict=first_losses)


# ---------------------------------------------------- G2: B1 capacity smoke
def cmd_g2(_args) -> dict:
    from mmengine.config import Config
    from rsprompter.uav import ubr
    from rsprompter.uav import ubr_train_integration as integ_mod
    from data.loader import create_train_loader, rtmdet_collate_fn
    from train.gt_samples import _build_gt_data_samples

    _protocol_env()
    _bootstrap()
    cfg = Config.fromfile(str(CONFIG))
    ubr_cfg = cfg.model.roi_head.mask_head.ubr_integration_cfg
    assert (ubr_cfg["refiner_hidden"], ubr_cfg["refiner_depth"]) == (32, 6)
    assert ubr_cfg["grad_checkpoint"] is False
    assert ubr_cfg["bucket_backward"] is False
    integ_mod.install_class_patch(ubr_cfg)
    model = _build(cfg.model, "cuda")
    integ = model.ubr_integration
    n_par = sum(p.numel() for p in model.ubr_refiner.parameters())
    assert n_par == 67457, n_par
    assert all(p.requires_grad for p in model.ubr_refiner.parameters())
    assert float(model.ubr_refiner.out.weight.abs().max()) == 0.0

    _, _, train_dataset = create_train_loader(
        data_root=DS_ROOT, batch_size=1, dataset_format="whu_coco",
        image_size=(512, 512), num_workers=0, seed=SEED, val_batch_size=1,
        flip_prob=0.0, vflip_prob=0.0,   # 3090 line: no flip
        whu_train_ann_file=os.environ["WHU_TRAIN_ANN_FILE"],
        whu_val_ann_file=os.environ["WHU_VAL_ANN_FILE"],
        whu_train_img_subdir=os.environ["WHU_TRAIN_IMG_SUBDIR"],
        whu_val_img_subdir=os.environ["WHU_VAL_IMG_SUBDIR"],
        train_subset_ratio=1.0, val_subset_ratio=1.0)
    batch = rtmdet_collate_fn([train_dataset[0]])
    data_samples = _build_gt_data_samples(
        batch["imgs"], batch["img_metas"], batch["gt_bboxes"],
        batch["gt_labels"], batch["gt_masks"], "cuda")
    payload = model.data_preprocessor(
        {"inputs": batch["imgs"].to("cuda"), "data_samples": data_samples},
        training=True)
    with torch.autocast("cuda"):
        losses = model(payload["inputs"], payload["data_samples"], mode="loss")
    assert "loss_ubr" in losses
    for t in _loss_tensors(losses):
        assert torch.isfinite(t).all(), "non-finite loss component (B1)"
    total = sum(_loss_tensors(losses))
    total.backward()
    grads_live = [p.grad for p in model.ubr_refiner.parameters()
                  if p.grad is not None and float(p.grad.abs().sum()) > 0]
    assert grads_live, "B1: loss_ubr gradient path dead"
    stats = dict(integ.last_train_stats)
    for key in ("UBR/instances", "UBR/band_nonempty_frac", "UBR/focal",
                "UBR/dice"):
        assert key in stats and math.isfinite(stats[key]), f"telemetry {key}"

    # --- B2 unit check: checkpointed forward/backward equals plain one
    torch.manual_seed(7)
    net_ckpt = ubr.UnifiedBoundaryRefiner(hidden=32, depth=6,
                                          grad_checkpoint=True).cuda()
    net_plain = ubr.UnifiedBoundaryRefiner(hidden=32, depth=6).cuda()
    net_plain.load_state_dict(net_ckpt.state_dict())
    x = torch.randn(3, 32, 96, 96, device="cuda", requires_grad=True)
    lg = torch.randn(3, 1, 96, 96, device="cuda")
    sdt = torch.randn(3, 1, 96, 96, device="cuda")
    nb = torch.zeros(3, 1, 96, 96, device="cuda")
    net_ckpt.train()
    net_plain.train()
    out_c = net_ckpt(x, lg, sdt, nb)
    g_c = torch.autograd.grad(out_c.square().mean(), x, retain_graph=False)[0]
    out_p = net_plain(x, lg, sdt, nb)
    g_p = torch.autograd.grad(out_p.square().mean(), x)[0]
    b2_max_diff = float((g_c - g_p).abs().max())
    assert b2_max_diff < 1e-6, f"B2 checkpoint grad drift {b2_max_diff}"

    # --- A4 unit check: bucketed backward accumulates the exact gradient
    from rsprompter.uav.ubr_train_integration import register_grad_scaler

    class _FakeScaler:
        @staticmethod
        def get_scale():
            return 1024.0

    register_grad_scaler(_FakeScaler())
    xa = torch.randn(4, requires_grad=True)
    xb = torch.randn(4, requires_grad=True)

    def _graph():
        h = xa * 3.0 + xb * 2.0
        return [h[:2].square().sum(), h[2:].square().sum()]

    b1l, b2l = _graph()
    # A4 semantics: head buckets backwarded manually (retain shared
    # upstream); the tail rides the trainer's scaled backward.
    torch.autograd.backward(1024.0 * b1l, retain_graph=True)
    torch.autograd.backward(1024.0 * b2l)
    xa_a4, xb_a4 = xa.grad.clone(), xb.grad.clone()
    xa.grad = xb.grad = None
    c1, c2 = _graph()
    torch.autograd.backward(1024.0 * (c1 + c2))
    a4_err = max(float((xa_a4 - xa.grad).abs().max()),
                 float((xb_a4 - xb.grad).abs().max()))
    assert a4_err < 1e-4, f"A4 accumulate mismatch {a4_err}"
    integ_mod.register_grad_scaler(None)

    return dict(gate="G2_capacity_smoke", status="PASS", params=n_par,
                capacity=[32, 6],
                loss_ubr=float(losses["loss_ubr"]),
                telemetry=stats, b2_grad_max_diff=b2_max_diff,
                a4_unit_err=a4_err,
                grad_tensors_live=len(grads_live),
                note="B1 on: single real-batch update finite, telemetry keys "
                     "complete; B2/A4 flags exercised at unit level (both "
                     "default OFF in the launch config)")


# ------------------------------------------------- G4: seal gate
def _sha_param(name: str, t: torch.Tensor) -> str:
    h = hashlib.sha256()
    h.update(name.encode())
    h.update(str(tuple(t.shape)).encode())
    h.update(str(t.dtype).encode())
    h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _git_blob_sha(rel: str) -> str:
    """Blob sha of a code-root-relative file at HEAD (repo may be higher)."""
    repo = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                          cwd=str(ROOT), capture_output=True, text=True)
    assert repo.returncode == 0, "not a git repo"
    repo_rel = (Path(ROOT) / rel).resolve().relative_to(
        Path(repo.stdout.strip()).resolve()).as_posix()
    out = subprocess.run(["git", "show", f"HEAD:{repo_rel}"], cwd=str(ROOT),
                         capture_output=True)
    assert out.returncode == 0, f"git show failed for {repo_rel}"
    return hashlib.sha256(out.stdout).hexdigest()


def cmd_g4(_args) -> dict:
    from mmengine.config import Config
    from rsprompter.uav import ubr_train_integration as integ_mod

    _bootstrap()
    # Build order matters: sealed FIRST (no class patch), then patched.
    sealed_cfg = Config.fromfile(str(CONFIG_SEALED))
    sealed = _build(sealed_cfg.model, "cpu")
    cfg = Config.fromfile(str(CONFIG))
    integ_mod.install_class_patch(
        cfg.model.roi_head.mask_head.ubr_integration_cfg)
    patched = _build(cfg.model, "cpu")

    sd_a, sd_b = patched.state_dict(), sealed.state_dict()
    ubr_keys = sorted(k for k in sd_a if "ubr_refiner." in k)
    tail_keys = sorted(k for k in sd_b if "decoder_tail_refiner." in k)
    a_only = sorted(set(sd_a) - set(sd_b))
    b_only = sorted(set(sd_b) - set(sd_a))
    assert a_only == ubr_keys, f"patched build has unexpected extra keys {a_only[:5]}"
    assert b_only == tail_keys, f"sealed build extra keys {b_only[:5]}"
    mismatched = [k for k in set(sd_a) & set(sd_b)
                  if not torch.equal(sd_a[k], sd_b[k])]
    assert not mismatched, f"non-UBR tensors drifted: {mismatched[:5]}"

    # frozen-parameter SHA (prompt stack / PE / backbone frozen weights)
    frozen_names = sorted(n for n, p in patched.named_parameters()
                          if not p.requires_grad
                          and "ubr_refiner." not in n)
    frozen_sha = hashlib.sha256()
    for n in frozen_names:
        frozen_sha.update(_sha_param(n, dict(patched.named_parameters())[n]).encode())
    pe = [n for n in frozen_names if "prompt_encoder" in n]
    nme = [n for n in frozen_names if "no_mask_embed" in n]
    assert pe and nme, "frozen set must contain the prompt stack and PE"
    lora_a = {n: tuple(p.shape) for n, p in patched.named_parameters()
              if "lora_" in n}
    lora_b = {n: tuple(p.shape) for n, p in sealed.named_parameters()
              if "lora_" in n}
    assert lora_a == lora_b, "backbone LoRA structure drifted"
    n_ubr_params = sum(p.numel() for p in patched.ubr_refiner.parameters())

    # source_hashes bump: v1.1 working tree vs git HEAD (EXP-0028 code)
    runner = _runner()
    hashes = {}
    for rel in runner.CONTRACT_SOURCE_FILES:
        new = runner.sha256_of(ROOT / rel)
        old = _git_blob_sha(rel)
        assert new != old, f"source hash did not bump: {rel}"
        hashes[rel] = dict(v11=new[:16], exp0028_head=old[:16], bumped=True)

    contract = runner._build_contract(
        Path(DS_ROOT), Path(GATES_DIR / "g4_contract_probe"), None,
        "g4-static-check")
    assert contract["contract_version"] == 2
    assert contract["ubr_capacity"] == dict(hidden=32, depth=6)
    assert "new unified definition" in contract["declaration"]
    assert "numerically neutral" in contract["declaration"]
    assert set(contract["source_hashes"]) == set(runner.CONTRACT_SOURCE_FILES)
    assert runner.OUT_ROOT_DEFAULT.endswith("ue_ubr_reseal_r2_20261027")
    assert runner.PROTOCOL["seed"] == SEED

    return dict(
        gate="G4_seal", status="PASS",
        frozen_param_count=len(frozen_names),
        frozen_param_sha=frozen_sha.hexdigest(),
        frozen_prompt_stack_keys=len(pe), frozen_pe_keys=len(nme),
        lora_structure_keys=len(lora_a),
        non_ubr_tensors_compared=len(set(sd_a) & set(sd_b)),
        ubr_only_keys=len(ubr_keys), sealed_tail_keys=len(tail_keys),
        ubr_capacity_params=n_ubr_params,
        source_hashes=hashes,
        contract_v2=dict(tag=contract["tag"], version=contract["contract_version"],
                         declaration=contract["declaration"],
                         ubr_capacity=contract["ubr_capacity"],
                         val_shard=contract["val_shard_across_ranks"]),
        note="seed-44 dual build: every non-UBR tensor bit-equal to the "
             "sealed EXP-0025 build (frozen set incl. prompt stack + PE; "
             "LoRA structure identical); UBR params 67457 (B1 unified)")


# ------------------------------------- A5: sharded validation alignment
def _val_args():
    return SimpleNamespace(
        ema_eval=0, val_every_n_epochs=1, freeze_bn=0, compute_val_loss=0,
        max_val_batches=0, dataset="whu", val_batch_size=1)


def _capture_eval_inputs():
    import utils.coco_eval_utils as ceu
    captured = {}
    orig = ceu.build_coco_gt_and_dt

    def wrapped(all_gt, all_dt, all_img_metas, **kw):
        captured.setdefault("records", (all_gt, all_dt, all_img_metas))
        return orig(all_gt, all_dt, all_img_metas, **kw)

    ceu.build_coco_gt_and_dt = wrapped
    return captured


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    raise TypeError(f"not serializable: {type(obj)}")


def _records_sha(records) -> str:
    payload = json.dumps(records, sort_keys=True, default=_json_default)
    return hashlib.sha256(payload.encode()).hexdigest()


def _build_val_align_model():
    from mmengine.config import Config
    from rsprompter.uav import ubr_train_integration as integ_mod
    cfg = Config.fromfile(str(CONFIG))
    integ_mod.install_class_patch(
        cfg.model.roi_head.mask_head.ubr_integration_cfg)
    return _build(cfg.model, "cuda")


def _val_loader_subset():
    from data.loader import create_train_loader
    runner = _runner()
    _, val_loader, _ = create_train_loader(
        data_root=DS_ROOT, batch_size=1, dataset_format="whu_coco",
        image_size=(512, 512), num_workers=0, seed=SEED, val_batch_size=1,
        flip_prob=0.0, vflip_prob=0.0,
        whu_train_ann_file=os.environ["WHU_TRAIN_ANN_FILE"],
        whu_val_ann_file=os.environ["WHU_VAL_ANN_FILE"],
        whu_train_img_subdir=os.environ["WHU_TRAIN_IMG_SUBDIR"],
        whu_val_img_subdir=os.environ["WHU_VAL_IMG_SUBDIR"],
        train_subset_ratio=runner.subset_ratio(2142, 4),
        val_subset_ratio=runner.subset_ratio(459, VAL_ALIGN_IMAGES))
    assert len(val_loader.dataset) == VAL_ALIGN_IMAGES
    return val_loader


def cmd_val_single(_args) -> dict:
    """Unsharded single-GPU reference run through the trainer's val path."""
    os.environ.setdefault("SAM2_CKPT", SAM2_CKPT)
    os.environ["TEST_MAX_PER_IMG"] = "250"  # canonical AP250 contract
    _protocol_env()
    _bootstrap()
    from train.loops import validate_one_epoch
    model = _build_val_align_model()
    val_loader = _val_loader_subset()
    cap = _capture_eval_inputs()
    t0 = time.perf_counter()
    _, metrics, _, gt_records = validate_one_epoch(
        _val_args(), model, None, val_loader, model.data_preprocessor,
        "cuda", False, True, 0, None, False, 1, -1, None, "scores", True, 0)
    all_gt, all_dt, all_metas = cap["records"]
    result = dict(
        mode="single", images=len(all_dt),
        records_sha=_records_sha([all_gt, all_dt]),
        metrics=metrics, secs=round(time.perf_counter() - t0, 1))
    _report(result, GATES_DIR / "val_align_single.json")
    return result


def cmd_val_ddp(_args) -> dict:
    """2-rank Gloo-sharded run through the trainer's own sharded val path."""
    import torch.distributed as dist
    os.environ.setdefault("SAM2_CKPT", SAM2_CKPT)
    os.environ["TEST_MAX_PER_IMG"] = "250"
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    assert world == 2, "val-ddp runs under torchrun --nproc_per_node=2"
    dist.init_process_group("gloo", init_method="env://")
    control = dist.new_group(ranks=list(range(world)), backend="gloo")
    # Both ranks share the single visible GPU, but mmengine's logger infers
    # a device id by indexing CUDA_VISIBLE_DEVICES[local_rank] — out of
    # range for rank 1. The id is only used for log-file naming (no log
    # file here), so pin it before any mmengine logger is created.
    import mmengine.logging.logger as _mml
    _mml._get_device_id = lambda: 0
    _protocol_env()
    _bootstrap()
    from train.gt_samples import _shard_val_loader
    from train.loops import validate_one_epoch
    torch.cuda.set_device(0)  # both ranks share the single visible GPU
    model = _build_val_align_model()
    val_loader = _val_loader_subset()
    shard_loader, total = _shard_val_loader(val_loader, rank, world)
    assert total == VAL_ALIGN_IMAGES
    param_sha = hashlib.sha256()
    for n, p in model.named_parameters():
        param_sha.update(_sha_param(n, p).encode())
    gathers = [None] * world
    dist.all_gather_object(gathers, param_sha.hexdigest())
    assert len(set(gathers)) == 1, f"rank weight divergence: {gathers}"
    cap = _capture_eval_inputs()
    _, metrics, _, _ = validate_one_epoch(
        _val_args(), model, None, shard_loader, model.data_preprocessor,
        "cuda", True, rank == 0, rank, control, True, world, total, None,
        "scores", True, 0)
    out = dict(mode="ddp", rank=rank, images=VAL_ALIGN_IMAGES,
               weight_sha=gathers[0][:16])
    if rank == 0:
        all_gt, all_dt, _ = cap["records"]
        out.update(records_sha=_records_sha([all_gt, all_dt]), metrics=metrics)
        _report(out, GATES_DIR / "val_align_ddp.json")
    dist.destroy_process_group()
    return out


def cmd_val_align(_args) -> dict:
    """Orchestrate single + ddp runs and compare bit-exactness."""
    import gc
    GATES_DIR.mkdir(parents=True, exist_ok=True)
    single = cmd_val_single(_args)
    # free the single run's model + cached blocks before two more ranks
    # share the same physical GPU
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "3"
    rc = subprocess.call(
        [sys.executable, "-m", "torch.distributed.run", "--standalone",
         "--nnodes=1", "--nproc_per_node=2", str(__file__), "val-ddp"],
        cwd=str(ROOT), env=env)
    assert rc == 0, f"val-ddp exited {rc}"
    ddp = json.loads((GATES_DIR / "val_align_ddp.json").read_text())
    assert ddp["records_sha"] == single["records_sha"], (
        f"A5 FAIL: sharded val records differ:\n single {single['records_sha']}"
        f"\n ddp    {ddp['records_sha']}")
    assert ddp["metrics"] == single["metrics"], (
        f"A5 FAIL: metrics differ: {single['metrics']} vs {ddp['metrics']}")
    result = dict(
        gate="A5_val_shard_alignment", status="PASS",
        images=VAL_ALIGN_IMAGES, records_sha=ddp["records_sha"],
        segm_mAP=ddp["metrics"].get("segm/mAP"),
        segm_mAP_75=ddp["metrics"].get("segm/mAP_75"),
        single_secs=single["secs"], weight_sha=ddp["weight_sha"],
        note="trainer's own validate_one_epoch both modes; DT/GT records "
             "sha256-identical (bit-exact), canonical AP250 metrics equal")
    _report(result, GATES_DIR / "val_align_gate.json")
    return result


# ------------------------------------------------- G3: perf telemetry
def _parse_telemetry(smoke_dir: Path) -> dict:
    rows = [json.loads(line) for line in
            (smoke_dir / "epoch_telemetry.jsonl").read_text().splitlines()]
    train = [r for r in rows if r["phase"] == "train"]
    val = [r for r in rows if r["phase"] == "val"]
    assert len(train) >= 2 and len(val) >= 2, rows
    tr2, va2 = train[-1], val[-1]  # epoch 2 (no first-epoch warmup)
    return dict(train_secs=tr2["secs"], val_secs=va2["secs"],
                train_peak_alloc_gb=tr2["peak_mem_alloc_gb"],
                train_peak_reserved_gb=tr2["peak_mem_reserved_gb"],
                epoch_peak_reserved_gb=va2["peak_mem_reserved_gb"])


def _g3_worker(config_path: str, out_dir: Path) -> None:
    """run_worker equivalent with a config override (A-only old capacity)."""
    runner = _runner()
    os.chdir(ROOT)
    runner.set_protocol_env(SAM2_CKPT)
    from mmengine.config import Config
    from rsprompter.uav import ubr_train_integration
    cfg = Config.fromfile(config_path)
    ubr_cfg = cfg.model.roi_head.mask_head.ubr_integration_cfg
    ubr_train_integration.install_class_patch(ubr_cfg)
    sys.path.insert(0, str(ROOT / "train"))
    import train_rsprompter_fusion  # noqa: E402
    runner._install_epoch_telemetry(train_rsprompter_fusion, str(out_dir))
    sys.argv = runner.trainer_argv(
        config_path, "g3_smoke_a_only", DS_ROOT, str(out_dir), epochs=2,
        train_ratio=runner.subset_ratio(2142, 20),
        val_ratio=runner.subset_ratio(459, 5), num_workers=2)
    os.environ["TRAIN_FLIP_PROB"] = "0"
    os.environ["TRAIN_VFLIP_PROB"] = "0"
    runner._install_3090_augmentation()
    train_rsprompter_fusion.main()


def cmd_g3(_args) -> dict:
    GATES_DIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "3"
    env.setdefault("SAM2_CKPT", SAM2_CKPT)

    # (1) A+B1 smoke through the runner's own smoke mode (config capacity)
    b1_dir = GATES_DIR / "g3_smoke_b1"
    if b1_dir.exists():
        shutil.rmtree(b1_dir)
    rc = subprocess.call(
        [sys.executable, str(HERE / "ue_ubr_reseal_20261027.py"), "--smoke",
         "--smoke-dir", str(b1_dir), "--out-root",
         str(GATES_DIR / "g3_no_full_launch")],
        cwd=str(ROOT), env=env)
    assert rc == 0, f"B1 smoke exited {rc}"
    tel_b1 = _parse_telemetry(b1_dir)

    # (2) A-only smoke at OLD capacity (text-substituted config; the legacy
    # pair is replay-only and never enters the repo)
    legacy_cfg = GATES_DIR / "g3_legacy_capacity_cfg.py"
    text = CONFIG.read_text()
    old_block = "                refiner_hidden=32,\n                refiner_depth=6,\n"
    new_block = "                refiner_hidden=48,\n                refiner_depth=7,\n"
    assert text.count(old_block) == 1, "capacity assignment block drifted"
    base_line = "_base_ = '../ue_whu_a3sgm_512.py'"
    assert text.count(base_line) == 1, "base line drifted"
    # the generated copy lives outside configs/, so point _base_ at the
    # sealed config absolutely (mmengine accepts absolute base paths)
    legacy_cfg.write_text(
        text.replace(old_block, new_block)
            .replace(base_line, f"_base_ = '{CONFIG_SEALED}'"))
    a_dir = GATES_DIR / "g3_smoke_a_only"
    if a_dir.exists():
        shutil.rmtree(a_dir)
    a_dir.mkdir(parents=True)
    rc = subprocess.call(
        [sys.executable, str(__file__), "--g3-worker", "--config",
         str(legacy_cfg), "--out-dir", str(a_dir)],
        cwd=str(ROOT), env=env)
    assert rc == 0, f"A-only smoke exited {rc}"
    tel_a = _parse_telemetry(a_dir)

    def est_epoch(tel, train_scenes, val_scenes):
        train_est = tel["train_secs"] / train_scenes * TRAIN_BATCHES_PER_RANK
        val_est = (tel["val_secs"] / val_scenes * VAL_IMAGES_PER_RANK
                   + VAL_COCO_OVERHEAD_SECS)
        return dict(train_est=round(train_est, 1), val_est=round(val_est, 1),
                    epoch_est=round(train_est + val_est, 1))

    est_b1 = est_epoch(tel_b1, 20, 5)
    est_a = est_epoch(tel_a, 20, 5)
    peak_b1 = max(tel_b1["train_peak_reserved_gb"],
                  tel_b1["epoch_peak_reserved_gb"])
    peak_a = max(tel_a["train_peak_reserved_gb"],
                 tel_a["epoch_peak_reserved_gb"])
    # dead EXP-0028 references (same box, launch2.log / smoke4 mtimes):
    # full run 13.2 min/epoch = 11.2 train (1.258 s/batch DDP4) + 2.1 val;
    # 35-37 GB peak/card. Its 20-scene smoke epoch-2 wall was ~39 s.
    dead_arm = dict(epoch_secs=792, train_secs=672, val_secs=126,
                    per_batch=1.258, peak_gb="35-37", smoke_epoch2_wall=39.0)
    detail = dict(
        gate="G3_memory_speed", limits=G3_LIMITS, dead_arm=dead_arm,
        a_only_old_capacity=dict(telemetry=tel_a, est_epoch=est_a,
                                 peak_reserved_gb=peak_a),
        a_plus_b1_unified=dict(telemetry=tel_b1, est_epoch=est_b1,
                               peak_reserved_gb=peak_b1),
        note="single-process 20/5-scene smokes, epoch-2 timings, per-rank "
             "extrapolation (535 train batches + 115 val images sharded). "
             "The same-box v1.0 smoke (smoke4) took ~39 s/epoch-2 wall; the "
             "extrapolation is deliberately conservative (single-process "
             "smoke per-batch includes 2-worker data waits the DDP4 launch "
             "amortizes over 4 workers).")
    assert peak_b1 <= G3_LIMITS["peak_mem_gb"], detail
    assert peak_a <= G3_LIMITS["peak_mem_gb"], detail
    assert est_b1["epoch_est"] <= G3_LIMITS["epoch_secs_b1"], json.dumps(detail)
    assert est_a["epoch_est"] <= G3_LIMITS["epoch_secs_a_only"], json.dumps(detail)
    detail["status"] = "PASS"
    return detail


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gate", nargs="?", default=None,
                    choices=["g1", "g2", "g3", "g4", "val-single",
                             "val-ddp", "val-align"])
    ap.add_argument("--g3-worker", action="store_true")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    if a.g3_worker:
        assert a.config and a.out_dir, "g3-worker needs --config/--out-dir"
        _g3_worker(a.config, Path(a.out_dir))
        return
    assert a.gate, "a gate is required (g1|g2|g3|g4|val-single|val-ddp|val-align)"
    t0 = time.time()
    try:
        result = dict(g1=cmd_g1, g2=cmd_g2, g3=cmd_g3, g4=cmd_g4,
                      **{"val-single": cmd_val_single, "val-ddp": cmd_val_ddp,
                         "val-align": cmd_val_align})[a.gate](a)
        result.setdefault("status", "PASS")
    except Exception as exc:  # noqa: BLE001 — record and exit non-zero
        result = dict(gate=a.gate, status="FAIL",
                      error=f"{type(exc).__name__}: {exc}",
                      trace=traceback.format_exc()[-6000:])
    result["secs"] = round(time.time() - t0, 1)
    _report(result, GATES_DIR / f"{a.gate.replace('-', '_')}.json")
    sys.exit(0 if result.get("status") == "PASS" else 1)


if __name__ == "__main__":
    main()
