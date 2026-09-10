#!/usr/bin/env python
"""Arm-level verification for NWPU P2-v2 arms, including R3 and R3+UDPR.

For each arm: capture the REAL environment produced by vhr10_p2v2_dev.sh
(DEV_DUMP_ENV), build the model from the actual config, and assert the
structural expectations of the ablation axis.  Then pairwise-diff the
resolved model_config dicts to prove each arm pair differs ONLY in the
intended knobs.  It performs no training, but SAM2 construction allocates its
configured CUDA position embedding, so run it only when one GPU is available.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[2]  # portable_sam2_explicit_coarse/
MAINLINE = ROOT.parent                      # repo root (repo/)
sys.path.insert(0, str(MAINLINE))
sys.path.insert(0, str(MAINLINE / "portable_sam2_explicit_coarse" / "inference"))

ARMS = ["p", "pb", "a0", "a1", "a2", "a2e", "a3", "a3r", "a4", "r3", "r3_udpr"]

# Every config input read by the VHR-10 inheritance chain.  ``build`` clears
# these before applying an arm dump, so one arm (or the caller's terminal)
# cannot contaminate the next arm's static audit.
CONFIG_ENV_KEYS = {
    "WHU1024_DATA_ROOT", "SAM2_CKPT", "SAM_IMAGE_EMBED_STRIDE", "SEGM_SCORE_MODE",
    "EXPLICIT_PROMPT_MODE", "P2_BOUNDARY_REFINER_ENABLED", "P2_BOUNDARY_REFINER_LOSS_MODE",
    "P2_BOUNDARY_REFINER_PROJECTED_CHANNELS", "P2_BOUNDARY_REFINER_MID_CHANNELS",
    "P2_BOUNDARY_REFINER_BETA", "P2_BOUNDARY_REFINER_DELTA_LOGIT_MAX",
    "P2_BOUNDARY_REFINER_LOSS_WEIGHT", "P2_BOUNDARY_REFINER_CORRECTION_MARGIN",
    "P2_BOUNDARY_REFINER_KEEP_LOSS_WEIGHT", "MAX_EPOCHS", "SHAPE_CONTEXT_FUSION",
    "SHAPE_POINT_ADAPTIVE_VALIDITY", "SHAPE_LOSS_SCHEDULE_MODE", "SHAPE_PRIOR_LOSS_WEIGHT",
    "SHAPE_LOSS_STAGE1_END", "SHAPE_LOSS_WEIGHT_STAGE1", "SHAPE_LOSS_WEIGHT_STAGE2",
    "SHAPE_DENSE_TRANSFORM", "SHAPE_DENSE_DETACH", "SHAPE_GAUSSIAN_FOREGROUND_THRESHOLD",
    "SHAPE_GAUSSIAN_OMEGA", "SHAPE_GAUSSIAN_GAMMA", "SHAPE_DENSE_ALPHA_INIT",
    "SHAPE_DENSE_TEMPERATURE", "SHAPE_DENSE_OUTSIDE_FILL", "COARSE_MASK_OUTPUT_SIZE",
    "FINAL_MASK_COORDINATE_MODE", "FINAL_MASK_LOSS_MODE", "FINAL_MASK_ROI_EXPAND_RATIO",
    "FINAL_MASK_ROI_BCE_WEIGHT", "FINAL_MASK_ROI_DICE_WEIGHT", "FINAL_MASK_OUTSIDE_BCE_WEIGHT",
    "ROI_SAM_ENABLED", "ROI_SAM_SAMPLING_RATIO", "POINT_WARMUP_ENABLED",
    "POINT_WARMUP_NO_POINT_EPOCHS", "POINT_WARMUP_ONE_PAIR_EPOCHS",
    "POINT_WARMUP_FULL_START_EPOCH", "POINT_NO_POINT_EPOCHS", "POINT_ONE_PAIR_EPOCHS",
    "POINT_FULL_START_EPOCH", "PROMPT_ENCODER_TRAIN_MASK_DOWNSCALING",
    "DECODER_TAIL_REFINER_ENABLED", "DECODER_TAIL_NUM_POINTS", "DECODER_TAIL_HIDDEN_DIM",
    "DECODER_TAIL_POINT_LOSS_WEIGHT", "DECODER_TAIL_DELTA_LOGIT_MAX",
    "DECODER_TAIL_MODE", "DECODER_TAIL_GATE_INIT_PROB", "DECODER_TAIL_GATE_LOSS_WEIGHT",
    "DECODER_TAIL_KEEP_LOSS_WEIGHT",
    "CANVAS_RENDERER_ENABLED", "CANVAS_RENDERER_LOSS_WEIGHT",
    "MASK_LOSS_RAMP_EPOCHS", "MASK_LOSS_RAMP_START",
}

# Values deliberately incompatible with D5-B.  A0 and A3 must resolve to
# exactly their clean contracts even when launched from this polluted shell.
POLLUTION = {
    "DECODER_TAIL_LR_MULT": "7.5", "ROI_SAM_ENABLED": "1",
    "SHAPE_CONTEXT_FUSION": "gated_spatial_film", "SHAPE_DENSE_TEMPERATURE": "7.0",
    "SHAPE_DENSE_OUTSIDE_FILL": "-3.0", "SHAPE_LOSS_SCHEDULE_MODE": "two_stage",
    "SHAPE_LOSS_STAGE1_END": "50", "SHAPE_LOSS_WEIGHT_STAGE1": "0.9",
    "POINT_WARMUP_ENABLED": "1", "POINT_WARMUP_NO_POINT_EPOCHS": "3",
    "POINT_ONE_PAIR_EPOCHS": "4", "FINAL_MASK_OUTSIDE_BCE_WEIGHT": "0.7",
    "FINAL_MASK_ROI_EXPAND_RATIO": "2.0", "P2_BOUNDARY_REFINER_ENABLED": "1",
    "DECODER_TAIL_TRAIN_ONLY": "1", "INIT_FROM": "/tmp/forbidden-init.pth",
    "DECODER_TAIL_MODE": "confidence_gated", "DECODER_TAIL_GATE_INIT_PROB": "0.9",
    "DECODER_TAIL_GATE_LOSS_WEIGHT": "7.0", "DECODER_TAIL_KEEP_LOSS_WEIGHT": "3.0",
}


def arm_env(arm: str, extra_env: Optional[Dict[str, str]] = None) -> dict:
    out = subprocess.run(
        ["bash", str(ROOT / "scripts/ablations/vhr10_p2v2_dev.sh"), arm],
        env={**os.environ, **(extra_env or {}), "DEV_DUMP_ENV": "1"},
        capture_output=True, text=True, check=True,
    ).stdout
    env = {}
    for line in out.splitlines():
        key, _, value = line.partition("=")
        env[key] = value
    assert env.get("EXPLICIT_PROMPT_MODE"), f"{arm}: env capture empty"
    return env


def build(env: dict):
    from infer_from_checkpoint import _register_and_build
    from mmengine.config import Config
    old_env = os.environ.copy()
    for key in CONFIG_ENV_KEYS:
        os.environ.pop(key, None)
    os.environ.update(env)
    try:
        cfg = Config.fromfile(str(ROOT / "configs/vhr10_baseplus_explicit_coarse.py"))
        model = _register_and_build(cfg.model.to_dict())
        return cfg.model.to_dict(), model
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def config_only(env: dict) -> dict:
    """CPU-only static config replay used for pollution regression."""
    from mmengine.config import Config
    old_env = os.environ.copy()
    for key in CONFIG_ENV_KEYS:
        os.environ.pop(key, None)
    os.environ.update(env)
    try:
        return Config.fromfile(str(ROOT / "configs/vhr10_baseplus_explicit_coarse.py")).model.to_dict()
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def dry_run(arm: str, extra_env: Optional[Dict[str, str]] = None) -> str:
    return subprocess.run(
        ["bash", str(ROOT / "scripts/ablations/vhr10_p2v2_dev.sh"), arm],
        env={**os.environ, **(extra_env or {}), "DRY_RUN": "1"},
        capture_output=True, text=True, check=True,
    ).stdout


def preflight_errors() -> list[str]:
    errors = []
    for arm in ("a0", "a3", "a4", "r3", "r3_udpr"):
        clean_env, polluted_env = arm_env(arm), arm_env(arm, POLLUTION)
        clean_cfg, polluted_cfg = config_only(clean_env), config_only(polluted_env)
        if clean_cfg != polluted_cfg:
            errors.append(f"{arm}: polluted shell changed resolved model_config")
        if arm in {"a3", "a4", "r3", "r3_udpr"}:
            expected = {
                "DECODER_TAIL_LR_MULT": "1.0", "ROI_SAM_ENABLED": "0",
                "SHAPE_CONTEXT_FUSION": "roi_only", "SHAPE_DENSE_TEMPERATURE": "1.0",
                "SHAPE_DENSE_OUTSIDE_FILL": "0.0", "SHAPE_LOSS_SCHEDULE_MODE": "fixed",
                "POINT_WARMUP_ENABLED": "0", "FINAL_MASK_OUTSIDE_BCE_WEIGHT": "0.05",
            }
            for key, value in expected.items():
                if polluted_env.get(key) != value:
                    errors.append(f"{arm}: {key}={polluted_env.get(key)!r}, expected {value!r}")
            text = dry_run(arm, POLLUTION)
            for required in ("init=none", "resume=none", "tail_only=0/lr_mult=1.0"):
                if required not in text:
                    errors.append(f"{arm}: dry-run lacks {required!r}: {text}")
    return errors


def trainable(module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def check_arm(arm: str, model, cfg) -> list:
    head = model.roi_head.mask_head
    errs = []

    def want(cond, msg):
        if not cond:
            errs.append(f"{arm}: {msg}")

    # Common contract for every arm
    want(head.final_mask_coordinate_mode == "full_image", "coord != full_image")
    want(head.final_mask_loss_cfg["mode"] == "roi_balanced_dice", "loss != roi_balanced_dice")
    want(head.prompt_sparse_mode == "shape_point", "sparse mode != shape_point")
    want(head.prompt_encoder is not None, "no PromptEncoder")
    want(head.shape_injector is not None, "no shape injector (mining source)")
    want(model.sam_image_embedding_stride == 16, "stride != 16")

    pe_train = trainable(head.prompt_encoder)
    md_train = trainable(head.prompt_encoder.mask_downscaling)

    if arm == "p":
        want(head.explicit_prompt_mode == "points", "mode != points")
        want(head.explicit_use_point_prompt and not head.explicit_use_box_prompt
             and not head.explicit_use_dense_prompt, "prompt flags wrong")
        want(not head.use_shape_dense, "use_shape_dense should be False")
        want(head.shape_dense_alpha_raw is None, "dense gate param should not exist")
        want(head.p2_boundary_refiner is None, "refiner should be absent")
        want(md_train == 0 and pe_train == 0, f"PE trainable {pe_train} != 0 (DDP-unused!)")
    elif arm == "pb":
        want(head.explicit_prompt_mode == "points_box", "mode != points_box")
        want(head.explicit_use_point_prompt and head.explicit_use_box_prompt
             and not head.explicit_use_dense_prompt, "prompt flags wrong")
        want(head.shape_dense_alpha_raw is None, "dense gate param should not exist")
        want(head.p2_boundary_refiner is None, "refiner should be absent")
        want(md_train == 0 and pe_train == 0, f"PE trainable {pe_train} != 0 (DDP-unused!)")
    else:
        want(head.explicit_prompt_mode == "points_box_dense", "mode != points_box_dense")
        want(head.explicit_use_dense_prompt, "dense flag off")
        import torch
        want(isinstance(head.shape_dense_alpha_raw, torch.nn.Parameter),
             "dense gate param missing")
        alpha = torch.sigmoid(head.shape_dense_alpha_raw.detach()).item()
        want(abs(alpha - 0.5) < 0.01, f"gate init {alpha:.3f} != 0.5")
        want(md_train == 4684, f"mask_downscaling trainable {md_train} != 4684")
        want(pe_train == 4684, f"PE trainable {pe_train} != 4684")
        refiner = head.p2_boundary_refiner
        is_r3 = arm in {"r3", "r3_udpr"}
        renderer = getattr(head, "canvas_renderer", None)
        if is_r3:
            want(renderer is not None, "R3 renderer missing")
            want(abs(float(head.canvas_renderer_loss_weight) - 0.05) < 1e-9,
                 "R3 loss weight != 0.05")
        else:
            want(renderer is None, "legacy arm unexpectedly has R3 renderer")
        if arm in {"a0", "r3"}:
            want(refiner is None, "refiner should be absent")
            want(not head.decoder_tail_enabled, "UDPR should be disabled")
        elif arm in {"a3", "a3r", "a4", "r3_udpr"}:
            want(refiner is None, "P2 refiner should be absent")
            want(head.decoder_tail_enabled, "UDPR should be enabled")
            tcfg = head._decoder_tail_cfg
            want(int(tcfg.get("num_points", -1)) == 64, "UDPR K != 64")
            want(int(tcfg.get("hidden_dim", -1)) == 128, "UDPR hidden != 128")
            want(abs(float(tcfg.get("point_loss_weight", -1)) - 1.0) < 1e-9,
                 "UDPR point loss weight != 1")
            want(abs(float(tcfg.get("delta_logit_max", -1)) - 2.0) < 1e-9,
                 "UDPR delta cap != 2")
            if arm in {"a3", "a3r", "r3_udpr"}:
                want("mode" not in tcfg, "v1 A3 must not materialize a mode key")
            else:
                want(tcfg.get("mode") == "confidence_gated", "DCR mode != confidence_gated")
                want(abs(float(tcfg.get("gate_init_prob", -1)) - 0.1) < 1e-9,
                     "DCR gate init != 0.1")
                want(abs(float(tcfg.get("gate_loss_weight", -1)) - 1.0) < 1e-9,
                     "DCR gate loss != 1")
                want(abs(float(tcfg.get("keep_loss_weight", -1)) - 0.05) < 1e-9,
                     "DCR keep loss != 0.05")
        else:
            want(refiner is not None, "refiner missing")
            rcfg = head.p2_boundary_refiner_cfg
            if arm == "a1":
                want(rcfg.get("loss_mode", "boundary") == "boundary", "loss_mode != boundary")
                want(abs(float(rcfg.get("beta", 0.2)) - 0.10) < 1e-9, "beta != 0.10")
                want(abs(float(rcfg.get("delta_logit_max", 2.0)) - 0.50) < 1e-9, "delta != 0.50")
            else:
                want(rcfg.get("loss_mode") == "correction_keep", "loss_mode != correction_keep")
                want(abs(float(rcfg.get("correction_margin", 1.0)) - 1.0) < 1e-9, "margin != 1.0")
                want(abs(float(rcfg.get("keep_loss_weight", 0.1)) - 0.1) < 1e-9, "keep != 0.1")
                if arm == "a2":
                    want(abs(float(rcfg["beta"]) - 0.10) < 1e-9, "beta != 0.10")
                    want(abs(float(rcfg["delta_logit_max"]) - 0.50) < 1e-9, "delta != 0.50")
                if arm == "a2e":
                    want(abs(float(rcfg["beta"]) - 0.30) < 1e-9, "beta != 0.30")
                    want(abs(float(rcfg["delta_logit_max"]) - 1.00) < 1e-9, "delta != 1.00")
    return errs


def diff_configs(a: dict, b: dict, path: str = "") -> list:
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            out += diff_configs(a.get(k, "<MISSING>"), b.get(k, "<MISSING>"), f"{path}.{k}")
    elif a != b:
        out.append((path, str(a)[:50], str(b)[:50]))
    return out


def main() -> int:
    configs, errors = {}, {}
    preflight = preflight_errors()
    for error in preflight:
        print("[preflight] !!", error)
    print(f"[preflight] {'PASS' if not preflight else 'FAIL'}: pollution-isolated config + A3 CLI")
    for arm in ARMS:
        env = arm_env(arm)
        cfg, model = build(env)
        errs = check_arm(arm, model, cfg)
        errors[arm] = errs
        configs[arm] = cfg
        from rsprompter.architecture_contract import architecture_contract
        fp = architecture_contract(cfg)["model_fingerprint"][:12]
        print(f"[{arm:4s}] fingerprint={fp}  checks={'PASS' if not errs else 'FAIL'}")
        for e in errs:
            print("   !!", e)
        del model

    print("\n===== pairwise model_config diffs (ablation feasibility) =====")
    pairs = [("p", "pb"), ("pb", "a0"), ("a0", "a1"), ("a1", "a2"), ("a2", "a2e"), ("a0", "a3"), ("a3", "a3r"), ("a3", "a4"), ("pb", "r3"), ("r3", "r3_udpr")]
    expected = {
        ("p", "pb"): {"explicit_prompt_mode"},
        ("pb", "a0"): {"explicit_prompt_mode", "train_mask_downscaling",
                       "use_shape_dense"},  # use_shape_dense derives from mode
        ("a0", "a1"): {"enabled"},
        ("a1", "a2"): {"loss_mode", "correction_margin", "keep_loss_weight"},
        ("a2", "a2e"): {"beta", "delta_logit_max"},
        ("a0", "a3"): {"decoder_tail_refiner_cfg"},
        ("a3", "a3r"): set(),  # trainer-side knob only: model identical
        ("a3", "a4"): {"gate_init_prob", "gate_loss_weight", "keep_loss_weight", "mode"},
        ("pb", "r3"): {"explicit_prompt_mode", "train_mask_downscaling", "use_shape_dense", "canvas_renderer_cfg"},
        ("r3", "r3_udpr"): {"decoder_tail_refiner_cfg"},
    }
    ok = True
    for x, y in pairs:
        diffs = diff_configs(configs[x], configs[y])
        keys = {d[0].split(".")[-1] for d in diffs}
        # allowed = expected keys (values may differ); plus nothing else
        extra = keys - expected[(x, y)]
        status = "OK" if not extra else "UNEXPECTED"
        if extra:
            ok = False
        print(f"{x:4s} vs {y:4s}: diff keys={sorted(keys)} -> {status}")
        for path, va, vb in diffs:
            print(f"    {path}: {x}={va} | {y}={vb}")

    all_pass = not preflight and all(not e for e in errors.values()) and ok
    print(f"\nVERDICT: {'ALL PASS' if all_pass else 'ISSUES FOUND'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
