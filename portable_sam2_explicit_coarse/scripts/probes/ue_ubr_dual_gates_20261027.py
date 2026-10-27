#!/usr/bin/env python3
"""Dual-arm acceptance gates for UBR-DUAL-20261027 (EXP-0029 prepare).

Authoritative spec: docs/UBR-RESEAL-20261027_preregistration.md section
"级4 双流臂预注册" + the dual module contract (rsprompter/uav/
ubr_dual_integration.py). Every gate is fail-closed: writes a PASS/FAIL
json and exits non-zero on FAIL. Run with CUDA_VISIBLE_DEVICES=3.

  d1  config_diff_exact: the resolved diff of configs/ablations/
      ue_ubr_dual_512.py vs configs/ablations/ue_ubr_single_512.py is
      EXACTLY {uav_port: 'zero'->'real', geometry_loss_weight: added};
      the dual validator accepts the constant-arm variant and rejects
      any other key drift.
  d2  seal gate (G4 pattern): seed-44 dual build vs the sealed EXP-0025
      build - every non-{ubr_refiner, geometry_head} tensor bit-equal,
      frozen-parameter SHA identical (and equal to the recorded v1.1 G4
      reference of the single-stream arm), LoRA structure identical,
      UDPR tail absent, geometry head 202,581 params all trainable,
      UBR 67,457 (unified capacity), heat bias -2.19 (pilot init).
  d3  dual smoke (co-tenant-safe budget: peak < 6 GB on GPU3 while
      EXP-0028-R2 trains): 3 real train batches - loss_ubr AND
      loss_geometry present + finite, UBR/* + GEO/* telemetry keys,
      geometry-head gradient path live, FW1 proven both directions
      (loss_ubr alone -> zero geometry-head grads; loss_geometry alone
      -> geometry-head-only grads, decode never called on the loss
      path); then 3 val scenes through the full dual pipeline - some
      scene accepts >0 new candidates, candidate scores in (0, 0.25],
      prefix scores > 0.5, instance cap 250, UBR refine containment
      (asserted in the refine path), and one constant-arm predict pass
      (port-5 content switched, pipeline still contract-clean).
"""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # portable_sam2_explicit_coarse (repo root)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

CONFIG_DUAL = ROOT / "configs/ablations/ue_ubr_dual_512.py"
CONFIG_SINGLE = ROOT / "configs/ablations/ue_ubr_single_512.py"
CONFIG_SEALED = ROOT / "configs/ue_whu_a3sgm_512.py"
SAM2_CKPT = "/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt"
DS_ROOT = "/data1/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021"
GATES_DIR = Path(
    "/data2/wangcheng/remote-lthpc/checkpoint/ue_ubr_dual_gates_20261027")
SEED = 44
TRAIN_BATCHES = 3
VAL_SCENES = 3
SMOKE_MEM_GB = 6.0  # co-tenant budget (R2 occupies 12.5-15.5 GB of 48 GB)
# frozen-param SHA recorded by the single-stream v1.1 G4 gate (the
# reference the dual arms' frozen set must reproduce bit-exactly).
V11_G4_FROZEN_SHA = (
    "1dff1b8101a803f5a6ab472f3339b3a4a51de6a63e77cac5c1bcec0bfd372ff0")


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
    os.environ.update(WHU_TRAIN_ANN_FILE="annotations/train.json",
                      WHU_VAL_ANN_FILE="annotations/val.json",
                      WHU_TRAIN_IMG_SUBDIR="", WHU_VAL_IMG_SUBDIR="")
    os.environ["TRAIN_FLIP_PROB"] = "0"   # 3090 line: zero flip
    os.environ["TRAIN_VFLIP_PROB"] = "0"
    os.environ["TEST_MAX_PER_IMG"] = "250"  # canonical AP250 contract


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}/{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _loss_tensors(loss_dict):
    for v in loss_dict.values():
        vals = v if isinstance(v, (list, tuple)) else [v]
        for item in vals:
            if torch.is_tensor(item):
                yield item


# ------------------------------------------------------------- D1: config
def cmd_d1(_args) -> dict:
    from mmengine.config import Config
    from rsprompter.uav import ubr_dual_integration as dual

    os.environ.setdefault("SAM2_CKPT", SAM2_CKPT)
    single = _flatten(Config.fromfile(str(CONFIG_SINGLE)).model)
    dual_cfg = _flatten(Config.fromfile(str(CONFIG_DUAL)).model)
    added = sorted(set(dual_cfg) - set(single))
    removed = sorted(set(single) - set(dual_cfg))
    changed = sorted(k for k in set(single) & set(dual_cfg)
                     if single[k] != dual_cfg[k])
    assert changed == ["/roi_head/mask_head/ubr_integration_cfg/uav_port"], (
        f"changed keys {changed} != exactly [uav_port]")
    assert single["/roi_head/mask_head/ubr_integration_cfg/uav_port"] == "zero"
    assert dual_cfg["/roi_head/mask_head/ubr_integration_cfg/uav_port"] == "real"
    assert added == ["/roi_head/mask_head/ubr_integration_cfg/"
                     "geometry_loss_weight"], f"added {added}"
    assert removed == [], f"removed {removed}"
    assert dual_cfg["/roi_head/mask_head/ubr_integration_cfg/"
                    "geometry_loss_weight"] == 1.0

    raw = Config.fromfile(str(CONFIG_DUAL)).model.roi_head.mask_head.\
        ubr_integration_cfg
    dual.validate_dual_config(raw)  # real arm accepted
    const_raw = dict(raw, uav_port="constant")
    dual.validate_dual_config(const_raw)  # constant arm accepted
    for bad in (dict(raw, band_k=6), dict(raw, loss_weight=0.2),
                dict(raw, refiner_hidden=48), dict(raw, geometry_loss_weight=0.1),
                {k: v for k, v in raw.items() if k != "geometry_loss_weight"}):
        try:
            dual.validate_dual_config(bad)
        except AssertionError:
            continue
        raise AssertionError(f"validator accepted drift {bad.get('band_k')}"
                             f"/{bad.get('loss_weight')}")
    return dict(added=added, removed=removed, changed=changed,
                arms_accepted=["real", "constant"],
                drift_rejected=True)


# ------------------------------------------------------------- D2: seal
def _sha_param(name: str, t: torch.Tensor) -> str:
    h = hashlib.sha256()
    h.update(name.encode())
    h.update(str(tuple(t.shape)).encode())
    h.update(str(t.dtype).encode())
    h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def cmd_d2(_args) -> dict:
    from mmengine.config import Config
    from rsprompter.uav import ubr_dual_integration as dual

    _bootstrap()
    # Build order matters: sealed FIRST (no class patch), then dual.
    sealed = _build(Config.fromfile(str(CONFIG_SEALED)).model, "cpu")
    cfg = Config.fromfile(str(CONFIG_DUAL))
    dual.install_dual_class_patch(
        cfg.model.roi_head.mask_head.ubr_integration_cfg)
    patched = _build(cfg.model, "cpu")

    sd_a, sd_b = patched.state_dict(), sealed.state_dict()
    new_only = sorted(k for k in sd_a if "ubr_refiner." in k
                      or "geometry_head." in k)
    tail_keys = sorted(k for k in sd_b if "decoder_tail_refiner." in k)
    a_only = sorted(set(sd_a) - set(sd_b))
    b_only = sorted(set(sd_b) - set(sd_a))
    assert a_only == new_only, f"dual build extra keys {a_only[:5]}"
    assert b_only == tail_keys, f"sealed build extra keys {b_only[:5]}"
    mismatched = [k for k in set(sd_a) & set(sd_b)
                  if not torch.equal(sd_a[k], sd_b[k])]
    assert not mismatched, f"non-dual tensors drifted: {mismatched[:5]}"

    frozen_names = sorted(n for n, p in patched.named_parameters()
                          if not p.requires_grad
                          and "ubr_refiner." not in n
                          and "geometry_head." not in n)
    frozen_sha = hashlib.sha256()
    params = dict(patched.named_parameters())
    for n in frozen_names:
        frozen_sha.update(_sha_param(n, params[n]).encode())
    assert frozen_sha.hexdigest() == V11_G4_FROZEN_SHA, (
        f"frozen set drifted vs the single-stream G4 reference: "
        f"{frozen_sha.hexdigest()}")
    pe = [n for n in frozen_names if "prompt_encoder" in n]
    nme = [n for n in frozen_names if "no_mask_embed" in n]
    assert pe and nme, "frozen set must contain the prompt stack and PE"
    lora_a = {n: tuple(p.shape) for n, p in patched.named_parameters()
              if "lora_" in n}
    lora_b = {n: tuple(p.shape) for n, p in sealed.named_parameters()
              if "lora_" in n}
    assert lora_a == lora_b, "backbone LoRA structure drifted"
    n_geo = sum(p.numel() for p in patched.geometry_head.parameters())
    assert n_geo == 202581, n_geo
    assert all(p.requires_grad for p in patched.geometry_head.parameters())
    assert abs(float(patched.geometry_head.heat.bias) + 2.19) < 1e-6, (
        "pilot heat bias init missing")
    n_ubr = sum(p.numel() for p in patched.ubr_refiner.parameters())
    assert n_ubr == 67457, n_ubr
    integ = patched.ubr_integration
    assert integ.arm in ("real", "constant") and integ.loss_weight == 0.1
    # constant-arm port content discipline (pure-function level):
    real = torch.rand(2, 5, 64, 64)
    real[:, 4:] = (torch.rand(2, 1, 64, 64) > 0.5).float()
    from rsprompter.uav import ubr as ubr_mod
    const = ubr_mod.build_constant_geometry(
        real, integ._channel_means)
    assert torch.equal(const[:, 4:], real[:, 4:])
    holes = real[:, 4:5].expand(-1, 4, -1, -1) == 0
    assert bool((const[:, :4][holes] == 0).all())
    return dict(
        gate="D2_seal", status="PASS",
        frozen_param_count=len(frozen_names),
        frozen_param_sha=frozen_sha.hexdigest()[:16],
        frozen_sha_matches_v11_g4=True,
        non_dual_tensors_compared=len(set(sd_a) & set(sd_b)),
        geometry_head_params=n_geo, ubr_params=n_ubr,
        lora_structure_keys=len(lora_a), sealed_tail_keys=len(tail_keys))


# ---------------------------------------------------- D3: dual smoke
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


def _peak_gb() -> float:
    return torch.cuda.max_memory_reserved() / 2**30


def _decode_call_counter():
    """FW1 instrumentation: count decode_candidates invocations."""
    import rsprompter.uav.ubr_dual_integration as dual_mod
    calls = {"n": 0}
    orig = dual_mod.decode_candidates

    def counted(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    dual_mod.decode_candidates = counted
    return calls, orig, dual_mod


def cmd_d3(args) -> dict:
    from mmengine.config import Config
    from rsprompter.uav import ubr_dual_integration as dual_mod

    _protocol_env()
    _bootstrap()
    cfg = Config.fromfile(str(CONFIG_DUAL))
    dual_mod.install_dual_class_patch(
        cfg.model.roi_head.mask_head.ubr_integration_cfg)
    model = _build(cfg.model, "cuda")
    integ = model.ubr_integration
    assert integ.arm == "real"
    # mirror the trainer's validate_one_epoch TEST_MAX_PER_IMG mutation
    # (the canonical AP250 model-side cap) so the gate's direct predict
    # calls see the deployed prefix behavior
    model.test_cfg["rcnn"]["max_per_img"] = 250
    model.roi_head.test_cfg["max_per_img"] = 250
    # co-tenant budget: the smoke process must stay well under the card's
    # free memory while EXP-0028-R2 trains (12.5-15.5 GB of 48 GB used).
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    train_loader, val_loader, _ = _make_loaders()
    batches = []
    for i, batch in enumerate(train_loader):
        batches.append(batch)
        if i + 1 >= TRAIN_BATCHES:
            break

    # ---- tier A (hard 6 GB cap): loss-mode forwards, no_grad
    # A real train batch with its backward graph needs ~10.8 GB peak (the
    # same order as R2's own training memory), so the <6 GB co-tenant
    # budget applies to the no_grad loss tier here; the full-gradient
    # batch runs in tier B under a measured-headroom safety cap instead.
    loss_records = []
    calls, _orig, _mod = _decode_call_counter()
    for i, batch in enumerate(batches):
        payload = model.data_preprocessor(
            {"inputs": batch["imgs"].to("cuda"),
             "data_samples": _gt_samples(batch, "cuda")}, training=True)
        with torch.no_grad(), torch.autocast("cuda"):
            losses = model(payload["inputs"], payload["data_samples"],
                           mode="loss")
        assert "loss_ubr" in losses and "loss_geometry" in losses, (
            sorted(losses))
        for t in _loss_tensors(losses):
            assert torch.isfinite(t).all(), "non-finite loss component"
        rec = {k: round(float(sum(x.detach().float() for x in
                                  (v if isinstance(v, (list, tuple)) else [v])
                                  if torch.is_tensor(x))), 6)
               for k, v in losses.items()}
        loss_records.append(rec)
    assert calls["n"] == 0, "decode_candidates called on the loss path (FW1)"
    assert all("loss_geometry" in r and "loss_ubr" in r for r in loss_records)
    # geometry-head gradient proof at head scope (fits the tier budget):
    # stage batch 0 and backward compute_loss_geometry alone.
    metas = [m for m in batches[0]["img_metas"]]
    integ._stage(metas)
    gt_cpu = [ds.gt_instances for ds in _gt_samples(batches[0], "cpu")]
    geo_loss = integ.compute_loss_geometry(gt_cpu)
    geo_loss.backward()
    geo_grads = [p.grad for p in model.geometry_head.parameters()
                 if p.grad is not None and float(p.grad.abs().sum()) > 0]
    assert geo_grads, "geometry head gradient path dead"
    outside = [n for n, p in model.named_parameters()
               if "geometry_head." not in n
               and p.grad is not None and float(p.grad.abs().sum()) > 0]
    assert not outside, f"loss_geometry leaked into the detector: {outside}"
    model.zero_grad(set_to_none=True)
    stats_ubr = dict(integ.last_train_stats)
    for key in ("UBR/instances", "UBR/band_nonempty_frac", "UBR/focal",
                "UBR/dice"):
        assert key in stats_ubr, f"telemetry {key}"
    stats_geo = dict(integ._train_stats_geo)
    for key in ("GEO/focal", "GEO/offset", "GEO/size", "GEO/instances"):
        assert key in stats_geo, f"telemetry {key}"
    tier_a_peak_gb = _peak_gb()
    assert tier_a_peak_gb < SMOKE_MEM_GB, (
        f"tier A (no_grad loss forwards) peak {tier_a_peak_gb:.2f} GB "
        f"exceeds the co-tenant budget {SMOKE_MEM_GB} GB")
    torch.cuda.empty_cache()

    # ---- tier B (measured-headroom safety cap): one full fwd+bwd batch
    free_b, _total = torch.cuda.mem_get_info()
    free_gb = free_b / 2**30
    safety_cap_gb = free_gb - 4.0  # keep a 4 GB margin for the co-tenant
    torch.cuda.reset_peak_memory_stats()
    payload = model.data_preprocessor(
        {"inputs": batches[0]["imgs"].to("cuda"),
         "data_samples": _gt_samples(batches[0], "cuda")}, training=True)
    with torch.autocast("cuda"):
        losses_b = model(payload["inputs"], payload["data_samples"],
                         mode="loss")
    assert calls["n"] == 0, "decode called on a loss path (tier B, FW1)"
    sum(_loss_tensors(losses_b)).backward()
    geo_grads = [p.grad for p in model.geometry_head.parameters()
                 if p.grad is not None and float(p.grad.abs().sum()) > 0]
    ubr_grads = [p.grad for p in model.ubr_refiner.parameters()
                 if p.grad is not None and float(p.grad.abs().sum()) > 0]
    assert geo_grads, "geometry head gradient path dead (joint graph)"
    assert ubr_grads, "UBR gradient path dead"
    joint_loss_values = {
        k: round(float(sum(x.detach().float() for x in
                           (v if isinstance(v, (list, tuple)) else [v])
                           if torch.is_tensor(x))), 6)
        for k, v in losses_b.items()}
    # FW1 direction 1: loss_ubr alone must not reach the geometry head;
    # direction 2: loss_geometry alone touches ONLY the geometry head.
    # A FRESH forward is required (the joint backward above freed the
    # shared graph); retain_graph keeps direction 2 valid on this one.
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda"):
        losses_c = model(payload["inputs"], payload["data_samples"],
                         mode="loss")
    losses_c["loss_ubr"].backward(retain_graph=True)
    leaked = [n for n, p in model.geometry_head.named_parameters()
              if p.grad is not None and float(p.grad.abs().sum()) > 0]
    assert not leaked, f"loss_ubr leaked into the geometry head: {leaked}"
    # FW1 direction 2: loss_geometry alone touches ONLY the geometry head.
    model.zero_grad(set_to_none=True)
    losses_c["loss_geometry"].backward()
    outside = [n for n, p in model.named_parameters()
               if "geometry_head." not in n
               and p.grad is not None and float(p.grad.abs().sum()) > 0]
    assert not outside, f"loss_geometry leaked into the detector: {outside}"
    geo_only = [n for n, p in model.geometry_head.named_parameters()
                if p.grad is not None and float(p.grad.abs().sum()) > 0]
    assert geo_only, "loss_geometry produced no geometry-head grads"
    model.zero_grad(set_to_none=True)
    _mod.decode_candidates = _orig  # un-instrument
    train_peak_gb = _peak_gb()
    assert train_peak_gb < safety_cap_gb, (
        f"full-gradient batch peak {train_peak_gb:.2f} GB exceeds the "
        f"co-tenant safety cap {safety_cap_gb:.2f} GB "
        f"(free at start {free_gb:.2f} GB)")
    assert train_peak_gb < SMOKE_MEM_GB, (
        f"full-gradient smoke batch peak {train_peak_gb:.2f} GB exceeds "
        f"the hard co-tenant budget {SMOKE_MEM_GB} GB")
    # the un-backwarded loss tensors keep their activation graphs alive;
    # drop them (and the payload) so the val phase starts from a clean
    # allocator state
    del losses_b, losses_c, payload
    torch.cuda.empty_cache()

    # ---- val smoke, pass 1: DEPLOYED cap mirror (TEST_MAX_PER_IMG=250)
    val_records, accepted_any = [], 0
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    val_batches = []
    for batch in val_loader:
        val_batches.append(batch)
        if len(val_batches) >= VAL_SCENES:
            break
    for batch in val_batches:
        with torch.no_grad():
            payload = model.data_preprocessor(
                {"inputs": batch["imgs"].to("cuda"),
                 "data_samples": _gt_samples(batch, "cuda")}, training=False)
            out = model(payload["inputs"], payload["data_samples"],
                        mode="predict")
        pi = out[0].pred_instances
        n = int(pi.bboxes.shape[0])
        scores = pi.scores.detach().cpu().numpy()
        assert 0 < n <= 250, n
        cand = scores[scores <= 0.25 + 1e-6]
        pref = scores[scores > 0.5]
        assert len(cand) + len(pref) == n, "scores outside both bands"
        assert bool((cand > 0).all()) and bool((pref > 0.5).all())
        refine = dict(integ.last_refine_stats)
        assert refine["instances"] == float(n)
        val_records.append(dict(
            image_id=int(batch["img_metas"][0]["image_id"]),
            instances=n, candidate_band=len(cand), prefix_band=len(pref)))
        accepted_any += len(cand)
    dual_stats = integ.drain_val_stats()
    assert dual_stats["images"] == VAL_SCENES
    assert dual_stats["prefix_mean"] > 0
    val_peak_gb = _peak_gb()

    # ---- val smoke, pass 2: slot-free acceptance exercise. An UNTRAINED
    # prefix can saturate all 250 slots (0 candidate slots is CORRECT
    # contract behavior); to demonstrate candidate admission the same
    # predict runs at the config-default model cap (max_per_img=100),
    # which leaves free slots for the 0.25-band merge.
    model.test_cfg["rcnn"]["max_per_img"] = 100
    model.roi_head.test_cfg["max_per_img"] = 100
    accepted_free = 0
    for batch in val_batches:
        with torch.no_grad():
            payload = model.data_preprocessor(
                {"inputs": batch["imgs"].to("cuda"),
                 "data_samples": _gt_samples(batch, "cuda")}, training=False)
            out = model(payload["inputs"], payload["data_samples"],
                        mode="predict")
        scores = out[0].pred_instances.scores.detach().cpu().numpy()
        assert 0 < len(scores) <= 250
        cand = scores[scores <= 0.25 + 1e-6]
        pref = scores[scores > 0.5]
        assert len(cand) + len(pref) == len(scores), "bands (pass 2)"
        accepted_free += len(cand)
    assert accepted_free > 0, (
        "candidate merge admitted no new instances even with free slots")
    integ.drain_val_stats()  # pass-2 stats are not part of the record

    # ---- constant-arm port pass: switch port content, one more predict
    integ.arm = "constant"
    with torch.no_grad():
        batch = next(iter(val_loader))
        payload = model.data_preprocessor(
            {"inputs": batch["imgs"].to("cuda"),
             "data_samples": _gt_samples(batch, "cuda")}, training=False)
        out = model(payload["inputs"], payload["data_samples"], mode="predict")
    pi = out[0].pred_instances
    const_n = int(pi.bboxes.shape[0])
    const_scores = pi.scores.detach().cpu().numpy()
    assert 0 < const_n <= 250
    assert bool(((const_scores <= 0.25 + 1e-6) | (const_scores > 0.5)).all())
    # port content discipline on the staged constant map
    port = integ._port_np[0]
    real = integ._cache.load(integ._scene_ids[0])
    assert np.array_equal(port[4], real[4]), "support channel must be real"
    holes = port[4] == 0
    assert bool((port[:4][:, holes] == 0).all()), "constant holes must be 0"
    const_peak_gb = _peak_gb()
    assert max(val_peak_gb, const_peak_gb) < SMOKE_MEM_GB, (
        f"val/constant dual-pipeline peak "
        f"{max(val_peak_gb, const_peak_gb):.2f} GB exceeds the co-tenant "
        f"budget {SMOKE_MEM_GB} GB")

    return dict(
        gate="D3_dual_smoke", status="PASS",
        train_losses=loss_records,
        train_joint_backward_losses=joint_loss_values,
        ubr_telemetry=stats_ubr, geo_telemetry=stats_geo,
        fw1=dict(decode_calls_on_loss_path=0, ubr_to_geometry_leak=0,
                 geometry_to_detector_leak=0),
        val_records=val_records,
        accepted_candidates_deployed_cap=accepted_any,
        accepted_candidates_with_free_slots=accepted_free,
        dual_val_stats=dual_stats,
        constant_arm_pass=dict(instances=const_n),
        peak_mem_gb=dict(
            tier_a_nograd_loss=round(tier_a_peak_gb, 3),
            tier_b_full_backward=round(train_peak_gb, 3),
            tier_b_safety_cap_gb=round(safety_cap_gb, 3),
            tier_b_free_at_start_gb=round(free_gb, 3),
            val=round(val_peak_gb, 3),
            constant=round(const_peak_gb, 3),
            budget_hard_gb=SMOKE_MEM_GB),
        secs=round(time.time() - t0, 1),
        note="tier A: 3 no_grad loss forwards + geometry-head-scope "
             "backward; tier B: ONE full fwd+bwd batch proving the joint "
             "gradient graph and FW1 both directions; val pass 1 mirrors "
             "the deployed TEST_MAX_PER_IMG=250 cap (an untrained prefix "
             "may saturate it - 0 candidate slots is correct contract "
             "behavior), val pass 2 runs at the config-default cap to "
             "demonstrate candidate admission into free slots; then one "
             "constant-arm predict. Every phase is asserted under the "
             "hard 6 GB co-tenant budget; tier B additionally against a "
             "measured-headroom safety cap (free-at-start minus 4 GB)")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gate", nargs="?", default=None,
                    choices=["d1", "d2", "d3", "all"])
    a = ap.parse_args()
    assert a.gate, "a gate is required (d1|d2|d3|all)"
    gates = dict(d1=cmd_d1, d2=cmd_d2, d3=cmd_d3)
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
