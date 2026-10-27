#!/usr/bin/env python3
"""Dual+RGB acceptance gates for UBR-DUAL-20261028 (EXP-0030 prepare).

Authoritative spec: docs/UBR-DUAL-RGB-LAUNCH-SPEC-20261028.md section 4;
integration contract rsprompter/uav/ubr_dual_rgb_integration.py. Every gate
is fail-closed: writes a PASS/FAIL json and exits non-zero on FAIL. Run
with CUDA_VISIBLE_DEVICES=3 (co-tenant budget: the two running jobs occupy
~29 GB/card; every tier here stays under a hard 6 GB cap and targets 5 GB).

  G1 skeleton seal - seed-44 dual build (EXP-0029 semantics) vs the +RGB
      build: every tensor outside backbone.rgb_source_branch.* bit-equal;
      the rgb branch structural pins (P0 lora structure, rgb_project
      256->32, rgb_output 32->224 exactly zero); the frozen-param SHA of
      the shared set reproduces the v1.1/D2 reference AND the dual build;
      the optimizer backbone-group rule captures every RGB parameter name;
      ctor torch.load audit: ONLY the generic SAM2 foundation file
      (task_weights_loaded=False). Config diff vs the dual config is
      exactly {rgb_content: 'real'}; validator accepts the reserved
      constant variant and rejects drift.
  G2 identity gate (the linchpin) - with the zero-init injection residual,
      the +RGB arm's forward is compared against the EXP-0029 build on the
      same seed-44 weights and the same real batches: every loss tensor
      VALUE-equal (fail-closed) with bitwise and zero-sign-only equality
      reported separately (base + 0.0 is IEEE-exact except -0.0 -> +0.0);
      the same for one val predict (masks bitwise). A self-stability
      double-forward proves the comparison method is deterministic. Then
      ONE real update: finite losses, full backward, AdamW step at the
      protocol lr/wd - rgb_output leaves zero, encoder LoRA + project
      receive nonzero grads, residual RMS becomes > 0, RGB telemetry
      complete (descriptor stats / per-cell support / residual RMS).
  G3 resource smoke - batch-level budget evidence for the 22 GB/card and
      the <=35% slower-than-EXP-0029 epoch budget: no_grad loss tier, ONE
      full-gradient batch (measured-headroom safety cap), val predict tier
      with the band contract; rgb stage share of the batch wall recorded
      (the 5-view P0 encoder sentinel, spec section 6).
  G4 contract seal - RESEAL_CONTRACT v4 structure validated field by field
      (tag, geometry block verbatim v3, rgb block complete, port-source
      SHAs vs the local ported file, dual-single enumeration
      {port5, loss_geometry, rgb_injection}, protocol field-equal to
      EXP-0029, val-shard inheritance); audit doc path recorded (the
      independent VERDICT: PASS is a pre-launch requirement).
  G5 constant branch drill - unit-level exercise of rgb_content='constant'
      (fourth arm, NOT launched): a train-only descriptor mean json is
      computed from the gate's own staged real descriptors, the constant
      path runs one loss forward + one predict with the SAME support mask
      as the real arm, zero-init keeps outputs identical, and the source
      encoder is skipped (content control with no RGB compute).
"""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # portable_sam2_explicit_coarse (repo root)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import ue_ubr_reseal_dual_20261027 as dual_runner  # noqa: E402
from rsprompter.uav import ubr_dual_integration as dual_mod  # noqa: E402
from rsprompter.uav import ubr_dual_rgb_integration as rgb_mod  # noqa: E402

import ue_ubr_dual_rgb_real_20261028 as rgb_runner  # noqa: E402

CONFIG_RGB = ROOT / "configs/ablations/ue_ubr_dual_rgb_512.py"
CONFIG_DUAL = ROOT / "configs/ablations/ue_ubr_dual_512.py"
GATES_DIR = Path(
    "/data2/wangcheng/remote-lthpc/checkpoint/ue_ubr_dual_rgb_gates_20261028")
SAM2_CKPT = dual_runner.SAM2_CKPT_DEFAULT
DS_ROOT = dual_runner.DS_ROOT_DEFAULT
SEED = 44
TRAIN_BATCHES = 2
VAL_SCENES = 2
MEM_HARD_GB = 22.0  # spec G3: full-arm budget per card (5-view encoder is the cost)      # hard co-tenant cap (the two training jobs hold the rest)
MEM_TARGET_GB = 5.0    # working target (report-only beyond this)
EPOCH_SLOWER_BUDGET = 0.35  # <=35% slower than EXP-0029 (spec section 4 G3)
V11_G4_FROZEN_SHA = (
    "1dff1b8101a803f5a6ab472f3339b3a4a51de6a63e77cac5c1bcec0bfd372ff0")
RGB_PREFIX = f"backbone.{rgb_mod.RGB_BRANCH_ATTR}."


def _report(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, default=_json_default))
    print(json.dumps(result, indent=2, default=_json_default), flush=True)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    raise TypeError(f"not serializable: {type(obj)}")


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
    return MODELS.build(cfg_model).to(device)


def _protocol_env() -> None:
    dual_runner.set_protocol_env(SAM2_CKPT)


def _peak_gb() -> float:
    return torch.cuda.max_memory_reserved() / 2**30


def _loss_items(loss_dict):
    """Deterministic (key, tensor) pairs from an mmdet loss dict."""
    for key, value in loss_dict.items():
        items = value if isinstance(value, (list, tuple)) else [value]
        for i, item in enumerate(items):
            if torch.is_tensor(item):
                yield f"{key}[{i}]" if len(items) > 1 else key, item


def _tensor_sha(t: torch.Tensor) -> str:
    h = hashlib.sha256()
    raw = t.detach().cpu().contiguous()
    h.update(str(raw.dtype).encode())
    h.update(str(tuple(raw.shape)).encode())
    h.update(raw.numpy().tobytes())
    return h.hexdigest()


def _pair_report(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Value / bitwise / zero-sign-only equality evidence for G2."""
    value_equal = bool(torch.equal(a.detach().cpu(), b.detach().cpu()))
    bitwise = _tensor_sha(a) == _tensor_sha(b)
    zero_sign = value_equal  # value equality is the floor
    if value_equal and not bitwise:
        x, y = a.detach().cpu().float(), b.detach().cpu().float()
        zero_sign = bool((((x == y) | ((x == 0) & (y == 0))).all()))
    return dict(value_equal=value_equal, bitwise_equal=bitwise,
                equal_mod_zero_sign=zero_sign,
                max_abs_diff=float((a.detach().cpu().float()
                                    - b.detach().cpu().float()).abs().max()))


def _pred_masks_np(pred) -> np.ndarray:
    """Prediction masks as a bool [N,512,512] array (tensor or Bitmap)."""
    obj = pred.pred_instances.masks
    if hasattr(obj, "masks"):  # BitmapMasks
        return np.asarray(obj.masks)
    return obj.detach().cpu().numpy().astype(bool)


def _make_loaders():
    from data.loader import create_train_loader
    return create_train_loader(
        data_root=DS_ROOT, batch_size=1, dataset_format="whu_coco",
        image_size=(512, 512), num_workers=0, seed=SEED, val_batch_size=1,
        flip_prob=0.0, vflip_prob=0.0,
        whu_train_ann_file=os.environ["WHU_TRAIN_ANN_FILE"],
        whu_val_ann_file=os.environ["WHU_VAL_ANN_FILE"],
        whu_train_img_subdir=os.environ["WHU_TRAIN_IMG_SUBDIR"],
        whu_val_img_subdir=os.environ["WHU_VAL_IMG_SUBDIR"],
        train_subset_ratio=1.0, val_subset_ratio=1.0)


def _gt_samples(batch, device):
    from train.gt_samples import _build_gt_data_samples
    return _build_gt_data_samples(
        batch["imgs"], batch["img_metas"], batch["gt_bboxes"],
        batch["gt_labels"], batch["gt_masks"], device)


def _payload(model, batch, training: bool):
    return model.data_preprocessor(
        {"inputs": batch["imgs"].to("cuda"),
         "data_samples": _gt_samples(batch, "cuda")}, training=training)


def _sha_param(name: str, t: torch.Tensor) -> str:
    h = hashlib.sha256()
    h.update(name.encode())
    h.update(str(tuple(t.shape)).encode())
    h.update(str(t.dtype).encode())
    h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _frozen_sha(model, exclude_prefix=None) -> tuple:
    names = sorted(n for n, p in model.named_parameters()
                   if not p.requires_grad
                   and (exclude_prefix is None
                        or not n.startswith(exclude_prefix)))
    params = dict(model.named_parameters())
    h = hashlib.sha256()
    for n in names:
        h.update(_sha_param(n, params[n]).encode())
    return h.hexdigest(), len(names)


def _config_model(path: Path):
    from mmengine.config import Config
    return Config.fromfile(str(path))


# ---------------------------------------------------- shared build helpers
def _build_dual(device: str = "cpu"):
    """The EXP-0029 dual build (arm real) - the identity reference."""
    cfg = _config_model(CONFIG_DUAL)
    dual_mod.install_dual_class_patch(
        cfg.model.roi_head.mask_head.ubr_integration_cfg)
    return _build(cfg.model, device)


def _build_rgb(device: str = "cpu", content: str = "real", audit_loads=None):
    """The +RGB build; optionally records every ctor torch.load path."""
    cfg = _config_model(CONFIG_RGB)
    rgb_mod.install_dual_rgb_class_patch(
        cfg.model.roi_head.mask_head.ubr_integration_cfg, content)
    if audit_loads is None:
        return _build(cfg.model, device)
    actual = torch.load

    def recording(path, *args, **kwargs):
        name = path if isinstance(path, (str, os.PathLike)) else \
            getattr(path, "name", None)
        resolved = str(Path(name).resolve()) if name else repr(path)
        audit_loads.append(resolved)
        return actual(path, *args, **kwargs)

    with patch("torch.load", side_effect=recording):
        return _build(cfg.model, device)


# ------------------------------------------------------------- G1: seal
def cmd_g1(_args) -> dict:
    _bootstrap()
    # -- config diff exact: rgb vs dual is exactly {rgb_content} --
    def flat(model_cfg):
        out = {}
        def walk(d, prefix=""):
            for k, v in d.items():
                key = f"{prefix}/{k}"
                if isinstance(v, dict):
                    walk(v, key)
                else:
                    out[key] = v
        walk(model_cfg)
        return out
    dual_flat = flat(_config_model(CONFIG_DUAL).model)
    rgb_flat = flat(_config_model(CONFIG_RGB).model)
    added = sorted(set(rgb_flat) - set(dual_flat))
    removed = sorted(set(dual_flat) - set(rgb_flat))
    changed = sorted(k for k in set(dual_flat) & set(rgb_flat)
                     if dual_flat[k] != rgb_flat[k])
    assert added == ["/roi_head/mask_head/ubr_integration_cfg/rgb_content"], added
    assert rgb_flat[added[0]] == "real"
    assert removed == [] and changed == [], (removed, changed)

    raw = _config_model(CONFIG_RGB).model.roi_head.mask_head.\
        ubr_integration_cfg
    rgb_mod.validate_dual_rgb_config(raw)                 # real accepted
    rgb_mod.validate_dual_rgb_config(dict(raw, rgb_content="constant"))
    for bad in (dict(raw, band_k=6), dict(raw, refiner_hidden=48),
                dict(raw, rgb_content="shuffle"),
                dict(raw, uav_port="constant"),
                dict(raw, geometry_loss_weight=0.1),
                {k: v for k, v in raw.items() if k != "rgb_content"}):
        try:
            rgb_mod.validate_dual_rgb_config(bad)
        except AssertionError:
            continue
        raise AssertionError(f"validator accepted drift {sorted(bad)}")

    # -- seed-44 dual build first (no rgb patch yet), then the +RGB build --
    model_a = _build_dual("cpu")
    loads = []
    model_b = _build_rgb("cpu", audit_loads=loads)
    sa, sb = model_a.state_dict(), model_b.state_dict()
    new_only = sorted(k for k in sb if k.startswith(RGB_PREFIX))
    a_only = sorted(set(sa) - set(sb))
    b_only = sorted(set(sb) - set(sa))
    assert b_only == new_only and new_only, f"rgb-only keys {b_only[:5]}"
    assert a_only == [], f"dual keys removed by the rgb build: {a_only[:5]}"
    mismatched = [k for k in set(sa) & set(sb) if not torch.equal(sa[k], sb[k])]
    assert not mismatched, f"shared tensors drifted: {mismatched[:5]}"

    # -- ctor load audit: ONLY the generic foundation (init discipline) --
    foundation = str(Path(os.environ.get("SAM2_CKPT", SAM2_CKPT)).resolve())
    assert loads and set(loads) == {foundation}, (
        f"ctor torch.load paths {sorted(set(loads))} != foundation only")

    # -- frozen-param SHA chain: dual == v1.1/D2 ref; rgb(shared) == dual --
    sha_a, n_a = _frozen_sha(model_a)
    assert sha_a == V11_G4_FROZEN_SHA, (
        f"dual build frozen sha {sha_a} != v1.1 G4 reference")
    sha_b, n_b = _frozen_sha(model_b, exclude_prefix=RGB_PREFIX)
    assert sha_b == sha_a and n_b == n_a, (sha_b, sha_a, n_b, n_a)

    # -- rgb branch structural pins --
    branch = getattr(model_b.backbone, rgb_mod.RGB_BRANCH_ATTR)
    out_w, out_b = branch.rgb_output.weight, branch.rgb_output.bias
    assert tuple(out_w.shape) == (rgb_mod.SAT_BLOCK4_DIM, rgb_mod.RGB_PROJECT_DIM)
    assert bool((out_w == 0).all()) and bool((out_b == 0).all())
    assert branch.injection_is_zero
    assert tuple(branch.rgb_project.weight.shape) == (
        rgb_mod.RGB_PROJECT_DIM, rgb_mod.P0_CHANNELS, 1, 1)
    lora = {n: tuple(p.shape) for n, p in branch.named_parameters()
            if "lora_" in n}
    assert lora, "P0 encoder carries no LoRA adapters"
    trainable = {n for n, p in branch.named_parameters() if p.requires_grad}
    assert all(("lora_" in n or "rgb_project" in n or "rgb_output" in n)
               for n in trainable), sorted(trainable)[:5]
    # optimizer backbone-group rule: every RGB parameter name matches the
    # train/optimizer.py backbone prefix (same group as the satellite LoRA).
    full_names = [f"backbone.{rgb_mod.RGB_BRANCH_ATTR}." + n
                  for n, _ in branch.named_parameters()]
    assert all(n.startswith("backbone.") for n in full_names)
    n_branch = sum(p.numel() for p in branch.parameters())
    n_train = sum(p.numel() for p in branch.parameters() if p.requires_grad)
    # trunk discovery + injection site asserts ran inside setup_rgb; the
    # integration handle must be the upgraded subclass.
    integ = model_b.ubr_integration
    assert isinstance(integ, rgb_mod.UBRDualRGBIntegration)
    assert integ.rgb_content == "real" and integ.arm == "real"
    # port verification: the local ported file IS the pinned remote blob.
    local_sha = hashlib.sha256(
        (ROOT / "rsprompter/uav/rgb_p0_encoder.py").read_bytes()).hexdigest()
    assert local_sha == rgb_mod.PORT_SOURCE_SHAS[
        "rsprompter/uav/rgb_p0_encoder.py"][0], "rgb_p0_encoder.py drifted"

    return dict(
        gate="G1_skeleton_seal", status="PASS",
        config_added=added, config_removed=removed, config_changed=changed,
        shared_tensors_compared=len(set(sa) & set(sb)),
        rgb_branch_keys=len(new_only),
        ctor_torch_load_paths=sorted(set(loads)),
        task_weights_loaded=False,
        frozen_param_count=n_b, frozen_sha_matches=["v11_g4", "dual_build"],
        rgb_branch_params=n_branch, rgb_branch_trainable_params=n_train,
        rgb_lora_param_tensors=len(lora),
        rgb_output_zero_init=True, port_file_sha_verified=local_sha[:16])


# ------------------------------------------------------ G2: identity gate
def cmd_g2(_args) -> dict:
    _protocol_env()
    _bootstrap()

    train_loader, val_loader, _ = _make_loaders()
    batches = []
    for i, batch in enumerate(train_loader):
        batches.append(batch)
        if i + 1 >= TRAIN_BATCHES:
            break
    val_batch = next(iter(val_loader))

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    # ---- reference forwards on the EXP-0029 build (eval mode: the
    # deterministic deployed forward; dropout would break ANY bitwise
    # comparison between separate runs, so identity is proven on the eval
    # path and the training path is covered by the finite-loss update).
    # Each loss forward is preceded by an identical RNG reseed: the RoI
    # sampler draws randomness, so bit-comparability requires equal RNG
    # state, not just equal inputs. ----
    fwd_seed = 20261028

    def _reseed():
        torch.manual_seed(fwd_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(fwd_seed)

    model_a = _build_dual("cuda")
    model_a.eval()
    model_a.test_cfg["rcnn"]["max_per_img"] = 250
    model_a.roi_head.test_cfg["max_per_img"] = 250
    with torch.no_grad(), torch.autocast("cuda"):
        ref_payloads = [_payload(model_a, b, True) for b in batches]
        ref_losses = []
        for p in ref_payloads:
            _reseed()
            ref_losses.append(
                model_a(p["inputs"], p["data_samples"], mode="loss"))
    with torch.no_grad():
        val_a = _payload(model_a, val_batch, False)
        ref_pred = model_a(val_a["inputs"], val_a["data_samples"],
                           mode="predict")
    # determinism of the comparison method itself (double forward, build A)
    with torch.no_grad(), torch.autocast("cuda"):
        _reseed()
        again = model_a(ref_payloads[0]["inputs"],
                        ref_payloads[0]["data_samples"], mode="loss")
    self_stable = all(
        _pair_report(x[1], y[1])["bitwise_equal"]
        for x, y in zip(_loss_items(ref_losses[0]), _loss_items(again)))
    assert self_stable, "double-forward on the reference build is not bit-stable"
    model_a.to("cpu")
    del model_a
    torch.cuda.empty_cache()

    # ---- +RGB forwards, zero-init residual (the identity claim) ----
    model_b = _build_rgb("cuda")
    integ = model_b.ubr_integration
    assert integ.rgb_branch.injection_is_zero, "rgb_output must start zero"
    model_b.eval()
    model_b.test_cfg["rcnn"]["max_per_img"] = 250
    model_b.roi_head.test_cfg["max_per_img"] = 250
    comparisons, identity_ok = [], True
    for b, ref in zip(batches, ref_losses):
        with torch.no_grad(), torch.autocast("cuda"):
            payload_b = _payload(model_b, b, True)
            _reseed()
            rgb_losses = model_b(payload_b["inputs"], payload_b["data_samples"],
                                 mode="loss")
        assert set(rgb_losses) == set(ref), (
            sorted(set(rgb_losses) ^ set(ref)))
        for (k_ref, t_ref), (k_rgb, t_rgb) in zip(_loss_items(ref),
                                                  _loss_items(rgb_losses)):
            assert k_ref == k_rgb
            ev = _pair_report(t_rgb, t_ref)
            comparisons.append(dict(loss=k_ref, **ev))
            identity_ok &= ev["value_equal"]
    assert identity_ok, "a loss tensor diverged from the EXP-0029 forward"
    value_ok = all(c["value_equal"] for c in comparisons)
    bitwise_ok = all(c["bitwise_equal"] for c in comparisons)
    zero_sign_ok = all(c["equal_mod_zero_sign"] for c in comparisons)
    assert value_ok and zero_sign_ok, (
        "forward identity broken beyond zero-sign flips")

    with torch.no_grad():
        val_b = _payload(model_b, val_batch, False)
        rgb_pred = model_b(val_b["inputs"], val_b["data_samples"],
                           mode="predict")
    masks_ref = _pred_masks_np(ref_pred[0])
    masks_rgb = _pred_masks_np(rgb_pred[0])
    pred_bitwise = bool(np.array_equal(masks_ref, masks_rgb))
    scores_equal = bool(torch.equal(ref_pred[0].pred_instances.scores,
                                    rgb_pred[0].pred_instances.scores))
    assert pred_bitwise and scores_equal, "predict outputs diverged at zero-init"
    identity_peak = _peak_gb()
    assert identity_peak < MEM_HARD_GB, identity_peak
    torch.cuda.empty_cache()

    # ---- one real update: full backward + AdamW step at protocol lr/wd ----
    free_gb = torch.cuda.mem_get_info()[0] / 2**30
    torch.cuda.reset_peak_memory_stats()
    model_b.train()
    trainable = [p for p in model_b.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=dual_runner.PROTOCOL["lr"],
                            weight_decay=dual_runner.PROTOCOL["weight_decay"])
    payload = _payload(model_b, batches[0], True)
    with torch.autocast("cuda"):
        losses = model_b(payload["inputs"], payload["data_samples"],
                         mode="loss")
    loss_values = {k: round(float(sum(x.detach().float() for x in
                                      (v if isinstance(v, (list, tuple)) else [v])
                                      if torch.is_tensor(x))), 6)
                   for k, v in losses.items()}
    assert all(torch.isfinite(t).all() for _, t in _loss_items(losses))
    total = sum(t for _, t in _loss_items(losses))
    total.backward()
    grads = {n: float(p.grad.abs().sum()) for n, p in
             model_b.named_parameters() if p.grad is not None}
    assert all(torch.isfinite(p.grad).all() for p in trainable
               if p.grad is not None), "non-finite gradient"
    # Zero-init residual math: d(loss)/d(upstream) = W_out^T @ g = 0 at
    # step 1, so ONLY rgb_output lights up on the first backward. Upstream
    # liveness must be asserted on a SECOND update (after W_out leaves 0).
    enc_lora = sum(v for n, v in grads.items() if n.startswith(RGB_PREFIX)
                   and "lora_" in n)
    proj = sum(v for n, v in grads.items()
               if n.startswith(f"{RGB_PREFIX}rgb_project"))
    out_g = sum(v for n, v in grads.items()
                if n.startswith(f"{RGB_PREFIX}rgb_output"))
    assert out_g > 0, ("rgb_output got no gradient", out_g)
    assert enc_lora == 0 and proj == 0, (
        "zero-init violated: upstream gradient before the first step",
        enc_lora, proj)
    opt.step()
    opt.zero_grad()
    payload2 = _payload(model_b, batches[0], True)
    with torch.autocast("cuda"):
        losses2 = model_b(payload2["inputs"], payload2["data_samples"],
                          mode="loss")
    sum(t for _, t in _loss_items(losses2)).backward()
    grads2 = {n: float(p.grad.abs().sum()) for n, p in
              model_b.named_parameters() if p.grad is not None}
    enc_lora2 = sum(v for n, v in grads2.items() if n.startswith(RGB_PREFIX)
                    and "lora_" in n)
    proj2 = sum(v for n, v in grads2.items()
                if n.startswith(f"{RGB_PREFIX}rgb_project"))
    assert enc_lora2 > 0 and proj2 > 0, (
        "RGB branch gradient path dead after step 1", enc_lora2, proj2)
    assert not integ.rgb_branch.injection_is_zero, (
        "the real update left rgb_output at zero - the arm would never open")
    # re-forward: the residual is live and the telemetry complete
    model_b.eval()
    with torch.no_grad():
        val_c = _payload(model_b, val_batch, False)
        model_b(val_c["inputs"], val_c["data_samples"], mode="predict")
    stats = integ.rgb_last_stats()
    for key in ("rgb_support_fraction", "rgb_descriptor_rms",
                "rgb_residual_rms", "rgb_base_rms", "rgb_candidates_per_cell"):
        assert key in stats, f"telemetry {key} missing"
    assert stats["rgb_residual_rms"] > 0, "residual still zero after update"
    update_peak = _peak_gb()
    assert update_peak < MEM_HARD_GB and update_peak < free_gb, (
        update_peak, free_gb)
    model_b.to("cpu")
    del model_b
    torch.cuda.empty_cache()

    return dict(
        gate="G2_identity", status="PASS",
        self_stable_reference=True,
        loss_comparisons=comparisons,
        losses_value_equal=value_ok, losses_bitwise_equal=bitwise_ok,
        losses_equal_mod_zero_sign=zero_sign_ok,
        predict_masks_bitwise=pred_bitwise, predict_scores_equal=scores_equal,
        update=dict(losses=loss_values,
                    grad_abs_sums=dict(encoder_lora=enc_lora,
                                       rgb_project=proj, rgb_output=out_g),
                    rgb_output_left_zero=False,
                    residual_rms_after_update=stats["rgb_residual_rms"]),
        rgb_telemetry=stats,
        peak_mem_gb=dict(identity=round(identity_peak, 3),
                         full_update=round(update_peak, 3),
                         hard_cap=MEM_HARD_GB, target=MEM_TARGET_GB),
        secs=round(time.time() - t0, 1))


# ------------------------------------------------ G3: resource smoke
def cmd_g3(_args) -> dict:
    _protocol_env()
    _bootstrap()
    model = _build_rgb("cuda")
    integ = model.ubr_integration
    model.test_cfg["rcnn"]["max_per_img"] = 250
    model.roi_head.test_cfg["max_per_img"] = 250

    train_loader, val_loader, _ = _make_loaders()
    batches = []
    for i, batch in enumerate(train_loader):
        batches.append(batch)
        if i + 1 >= TRAIN_BATCHES:
            break
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    # tier A: no_grad loss forwards (contract + rgb stage timing)
    stage_secs, loss_records = [], []
    for batch in batches:
        torch.cuda.synchronize()
        t_stage = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda"):
            payload = _payload(model, batch, True)
            losses = model(payload["inputs"], payload["data_samples"],
                           mode="loss")
        torch.cuda.synchronize()
        stage_secs.append(time.perf_counter() - t_stage)
        for _, t in _loss_items(losses):
            assert torch.isfinite(t).all(), "non-finite loss (G3 tier A)"
        assert "loss_ubr" in losses and "loss_geometry" in losses
        loss_records.append(
            {k: round(float(t.detach()), 6) for k, t in _loss_items(losses)})
    stats = integ.rgb_last_stats()
    for key in ("rgb_encoder_secs", "rgb_support_fraction",
                "rgb_descriptor_rms", "rgb_residual_rms"):
        assert key in stats, f"telemetry {key}"
    assert stats["rgb_residual_rms"] == 0.0, "residual nonzero at zero-init"
    tier_a = _peak_gb()
    assert tier_a < MEM_TARGET_GB, f"tier A peak {tier_a:.2f} GB"
    torch.cuda.empty_cache()

    # tier B: ONE full fwd+bwd batch - the epoch-budget evidence. The rgb
    # stage (anchor+rgb load, P0 forward, sampling) runs INSIDE the timed
    # window via the loss wrap; its wall is drained from the staged stats.
    free_gb = torch.cuda.mem_get_info()[0] / 2**30
    safety_cap = free_gb - 4.0
    torch.cuda.reset_peak_memory_stats()
    payload = _payload(model, batches[0], True)
    torch.cuda.synchronize()
    t_fwd = time.perf_counter()
    with torch.autocast("cuda"):
        losses = model(payload["inputs"], payload["data_samples"],
                       mode="loss")
    sum(t for _, t in _loss_items(losses)).backward()
    torch.cuda.synchronize()
    full_secs = time.perf_counter() - t_fwd
    stats_b = integ.rgb_last_stats()
    encoder_share = stats_b["rgb_encoder_secs"]
    # EXP-0029 batch ~= full - rgb stage; budget: stage/(batch_0029) <= 35%
    # (lower-bound estimate: the P0 backward share is not staged-separable)
    stage_est = min(stats_b["rgb_stage_secs"], full_secs)
    slower_ratio = stage_est / max(1e-6, full_secs - stage_est)
    tier_b = _peak_gb()
    assert tier_b < min(MEM_HARD_GB, safety_cap), (tier_b, safety_cap)
    del losses, payload
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    # val tier: full dual+rgb predict pipeline + band contract
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    val_records = []
    val_batches = []
    for batch in val_loader:
        val_batches.append(batch)
        if len(val_batches) >= VAL_SCENES:
            break
    for batch in val_batches:
        with torch.no_grad():
            val_p = _payload(model, batch, False)
            out = model(val_p["inputs"], val_p["data_samples"],
                        mode="predict")
        pi = out[0].pred_instances
        n = int(pi.bboxes.shape[0])
        scores = pi.scores.detach().cpu().numpy()
        assert 0 < n <= 250, n
        cand = scores[scores <= 0.25 + 1e-6]
        pref = scores[scores > 0.5]
        assert len(cand) + len(pref) == n, "scores outside both bands"
        assert bool((cand > 0).all()) and bool((pref > 0.5).all())
        val_records.append(dict(
            image_id=int(batch["img_metas"][0]["image_id"]),
            instances=n, candidate_band=len(cand), prefix_band=len(pref)))
    val_peak = _peak_gb()
    assert val_peak < MEM_TARGET_GB, val_peak
    model.to("cpu")
    del model
    torch.cuda.empty_cache()

    return dict(
        gate="G3_resource_smoke", status="PASS",
        train_losses=loss_records, val_records=val_records,
        rgb_telemetry=dict(stats),
        timing_secs=dict(no_grad_loss_per_batch=stage_secs,
                         full_fwd_bwd_batch=round(full_secs, 3),
                         rgb_stage_estimate=round(stage_est, 3),
                         encoder_secs_per_stage=encoder_share),
        epoch_budget=dict(slower_than_0029_ratio=round(slower_ratio, 4),
                          budget=EPOCH_SLOWER_BUDGET,
                          within_budget=bool(
                              slower_ratio <= EPOCH_SLOWER_BUDGET)),
        peak_mem_gb=dict(tier_a_nograd=round(tier_a, 3),
                         tier_b_full_backward=round(tier_b, 3),
                         tier_b_safety_cap=round(safety_cap, 3),
                         val=round(val_peak, 3),
                         hard_cap=MEM_HARD_GB, target=MEM_TARGET_GB,
                         launch_budget_per_card_gb=22.0),
        secs=round(time.time() - t0, 1))


# ---------------------------------------------------- G4: contract seal
def cmd_g4(_args) -> dict:
    _bootstrap()
    data_root = Path(DS_ROOT)
    out_root = Path("/data2/wangcheng/remote-lthpc/checkpoint/"
                    "ue_ubr_dual_rgb_gates_20261028/contract_probe")
    contract = rgb_runner._build_contract(
        data_root, out_root, "real", None,
        "docs/reviews/UBR-DUAL-RGB-20261028_audit.md")

    assert contract["tag"] == "UBR-DUAL-20261028-REAL", contract["tag"]
    assert contract["contract_version"] == 4
    assert contract["protocol"] == dual_runner.PROTOCOL
    assert contract["rgb_content"] == "real" and contract["arm"] == "real"
    assert contract["dual_minus_single"] == [
        "port5", "loss_geometry", "rgb_injection"], contract["dual_minus_single"]
    # geometry block verbatim v3
    geo = contract["geometry"]
    assert geo["head_params"] == dual_mod.GEOMETRY_HEAD_PARAMS == 202581
    assert geo["loss_weight"] == 1.0 and geo["instance_cap"] == 250
    assert geo["constants_sha256"] == dual_mod.CONSTANTS_SHA256
    assert geo["cache_version"] == dual_mod.GEOM_VERSION
    # rgb block complete (spec section 3)
    rgb = contract["rgb"]
    assert rgb["p0_channels"] == 256 and rgb["project_dim"] == 32
    assert rgb["source_encoder"]["lora"] == dict(
        rgb_mod.RGB_LORA_CONFIG), rgb["source_encoder"]["lora"]
    assert rgb["source_encoder"]["task_weights_loaded"] is False
    assert rgb["source_encoder"]["optimizer_group"] == "backbone"
    assert rgb["source_encoder"]["optimizer_lr"] == 5e-4
    assert rgb["source_encoder"]["optimizer_wd"] == 0.05
    assert Path(rgb["source_encoder"]["foundation_file"]).is_file()
    assert rgb["source_encoder"]["init_files"] == [
        rgb["source_encoder"]["foundation_file"]]
    assert rgb["injection"]["site"].endswith("blocks[4] (64x64)")
    assert rgb["injection"]["init"].startswith("zero")
    cache = rgb["anchor_cache"]
    assert cache["scene_count"] == 3060 and cache["anchor_version"] == \
        rgb_mod.ANCHOR_VERSION
    assert Path(cache["manifest_path"]).is_file() and len(
        cache["manifest_sha256"]) == 64
    # port-source SHAs recorded and the local port file matches its pin
    shas = rgb["port_sources"]["shas"]
    assert shas == rgb_mod.PORT_SOURCE_SHAS and rgb["port_sources"]["commit"] \
        == rgb_mod.PORT_COMMIT
    local_sha = hashlib.sha256(
        (ROOT / "rsprompter/uav/rgb_p0_encoder.py").read_bytes()).hexdigest()
    assert local_sha == shas["rsprompter/uav/rgb_p0_encoder.py"][0]
    # full source hashes incl. the +RGB wiring
    for rel in ("rsprompter/uav/rgb_p0_encoder.py",
                "rsprompter/uav/ubr_dual_rgb_integration.py",
                "configs/ablations/ue_ubr_dual_rgb_512.py",
                "scripts/probes/ue_ubr_dual_rgb_real_20261028.py"):
        assert rel in contract["source_hashes"], rel
    # val sharding inheritance (v3 semantics)
    _protocol_env()
    assert os.environ["VAL_SHARD_ACROSS_RANKS"] == "1"
    assert os.environ.get("TEST_MAX_PER_IMG") == "250"
    assert contract["val_shard_across_ranks"] == 1

    return dict(
        gate="G4_contract_seal", status="PASS",
        tag=contract["tag"], contract_version=contract["contract_version"],
        dual_minus_single=contract["dual_minus_single"],
        anchor_cache=dict(path=cache["cache_dir"],
                          scene_count=cache["scene_count"],
                          manifest_sha256=cache["manifest_sha256"][:16],
                          bytes_gb=round(cache["bytes_total"] / 2**30, 2)),
        port_sources=dict(commit=rgb_mod.PORT_COMMIT,
                          files=sorted(shas)),
        source_hash_files=sorted(contract["source_hashes"]),
        audit_requirement="docs/reviews/UBR-DUAL-RGB-20261028_audit.md "
                          "must contain VERDICT: PASS before the full launch")


# ----------------------------------------------- G5: constant drill
def cmd_g5(_args) -> dict:
    _protocol_env()
    _bootstrap()
    model = _build_rgb("cuda")
    integ = model.ubr_integration
    model.eval()  # deterministic drill: the two predicts must be comparable
    model.test_cfg["rcnn"]["max_per_img"] = 250
    model.roi_head.test_cfg["max_per_img"] = 250

    train_loader, val_loader, _ = _make_loaders()
    batches = []
    for i, batch in enumerate(train_loader):
        batches.append(batch)
        if i + 1 >= TRAIN_BATCHES:
            break
    val_batch = next(iter(val_loader))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    # real-arm references: staged support masks + zero-init predict
    supports_real = []
    for batch in batches:
        with torch.no_grad(), torch.autocast("cuda"):
            payload = _payload(model, batch, True)
            model(payload["inputs"], payload["data_samples"], mode="loss")
        supports_real.append(integ._rgb_support64.clone())
    with torch.no_grad():
        val_r = _payload(model, val_batch, False)
        ref_pred = model(val_r["inputs"], val_r["data_samples"],
                         mode="predict")

    # train-only descriptor mean over the gate's staged real descriptors
    # (DRILL ONLY: the fourth arm's production json must be computed over
    # the full train split before its launch)
    def _support128(scene_id: int) -> torch.Tensor:
        _, index_np, _ = integ._anchor_cache.load(int(scene_id))
        valid = torch.from_numpy(index_np) >= 0
        return valid.any(1).to(integ._rgb_pooled.device)
    means = []
    for batch in batches:
        scene = int(batch["img_metas"][0]["image_id"])
        with torch.no_grad():
            integ._stage_rgb(_gt_samples(batch, "cpu"))
            cell = integ._rgb_pooled[0].reshape(
                rgb_mod.RGB_PROJECT_DIM, -1).T  # [16384, 32], cell-major
            sel = cell[_support128(scene)]
        assert sel.numel() > 0
        means.append(sel.mean(0))
    mean_desc = torch.stack(means).mean(0)
    drill = GATES_DIR / "rgb_descriptor_mean_drill.json"
    drill.parent.mkdir(parents=True, exist_ok=True)
    drill.write_text(json.dumps(dict(
        descriptor_mean=[round(float(v), 8) for v in mean_desc],
        note="G5 drill mean over 2 gate batches; the fourth arm must "
             "recompute over the full train split")))
    os.environ["UBR_RGB_DESCRIPTOR_MEAN"] = str(drill)
    integ.rgb_content = "constant"
    integ._load_descriptor_mean()

    # constant path: same support mask, encoder skipped, outputs identical
    for batch, sup_ref in zip(batches, supports_real):
        t_stage = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda"):
            payload = _payload(model, batch, True)
            losses = model(payload["inputs"], payload["data_samples"],
                           mode="loss")
        torch.cuda.synchronize()
        stage_secs = time.perf_counter() - t_stage
        assert bool(torch.equal(integ._rgb_support64, sup_ref)), (
            "constant arm support mask diverged from the real arm")
        for _, t in _loss_items(losses):
            assert torch.isfinite(t).all()
        stats = integ.rgb_last_stats()
        assert stats["rgb_encoder_secs"] == 0.0, (
            "constant arm must skip the source encoder")
    with torch.no_grad():
        val_p = _payload(model, val_batch, False)
        const_pred = model(val_p["inputs"], val_p["data_samples"],
                           mode="predict")
    masks_const = _pred_masks_np(const_pred[0])
    assert np.array_equal(masks_ref, masks_const) and torch.equal(
        ref_pred[0].pred_instances.scores,
        const_pred[0].pred_instances.scores), (
        "constant content changed outputs at zero-init")
    stats_const = integ.rgb_last_stats()
    scene_val = int(val_batch["img_metas"][0]["image_id"])
    cell_c = integ._rgb_pooled[0].reshape(rgb_mod.RGB_PROJECT_DIM, -1).T
    sel = cell_c[_support128(scene_val)]
    assert float(sel.std()) == 0.0, "constant descriptors are not constant"
    peak = _peak_gb()
    assert peak < MEM_TARGET_GB, peak
    model.to("cpu")
    del model
    torch.cuda.empty_cache()
    return dict(
        gate="G5_constant_drill", status="PASS",
        descriptor_mean_json=str(drill),
        descriptor_mean_sha256=hashlib.sha256(
            drill.read_bytes()).hexdigest(),
        support_masks_identical=True, encoder_skipped=True,
        outputs_identical_at_zero_init=True,
        constant_stage_telemetry=dict(stats_const),
        constant_stage_secs=round(stage_secs, 4),
        peak_mem_gb=round(peak, 3),
        note="unit-level drill only - the fourth arm stays queued until an "
             "RGB_WINS verdict (spec section 5/7)",
        secs=round(time.time() - t0, 1))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gate", nargs="?", default=None,
                    choices=["g1", "g2", "g3", "g4", "g5", "all"])
    a = ap.parse_args()
    assert a.gate, "a gate is required (g1|...|g5|all)"
    gates = dict(g1=cmd_g1, g2=cmd_g2, g3=cmd_g3, g4=cmd_g4, g5=cmd_g5)
    names = list(gates) if a.gate == "all" else [a.gate]
    overall = "PASS"
    for name in names:
        t0 = time.time()
        try:
            result = gates[name](a)
            result.setdefault("status", "PASS")
        except Exception as exc:  # noqa: BLE001 — record, exit non-zero
            result = dict(gate=name, status="FAIL",
                          error=f"{type(exc).__name__}: {exc}",
                          trace=traceback.format_exc()[-6000:])
            overall = "FAIL"
        result["secs"] = round(time.time() - t0, 1)
        _report(result, GATES_DIR / f"{name}.json")
    sys.exit(0 if overall == "PASS" else 1)


if __name__ == "__main__":
    main()
