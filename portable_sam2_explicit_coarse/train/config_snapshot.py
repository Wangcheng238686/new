"""Self-describing config snapshot: git state capture and the snapshot
assembly (architecture contract, sam2 identity, resolved telemetry knobs)."""

import os
from typing import Dict

from cli_args import _env_flag
from distributed import _resolve_ddp_control_timeout_seconds, _resolve_ddp_timeout_seconds


def _capture_git_state() -> Dict[str, str]:
    """Fingerprint the executed code: HEAD, dirty flag, tracked-diff sha.

    Complements the runner's resolved_hyperparameters git_* lines so both the
    training log and every checkpoint pin the exact working-tree state, not
    just the commit (a dirty tree silently diverging from HEAD is what made
    the resumed paper run hard to trace).
    """
    import subprocess

    def _git(*argv: str) -> str:
        try:
            return subprocess.run(
                ["git", *argv],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        except Exception:
            return ""

    head = _git("rev-parse", "HEAD") or "unknown"
    status = _git("status", "--porcelain")
    dirty = "1" if status else "0"
    code_files = ",".join(
        line.split(maxsplit=1)[1].strip()
        for line in status.splitlines()
        if line.split(maxsplit=1)[-1].strip().endswith((".py", ".sh", ".md"))
    )[:400]
    try:
        diff_sha = subprocess.run(
            ["git", "diff", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        import hashlib

        diff_sha16 = (
            hashlib.sha256(diff_sha.stdout.encode("utf-8")).hexdigest()[:16]
            if diff_sha.stdout
            else "none"
        )
    except Exception:
        diff_sha16 = "unknown"
    return {
        "git_commit": head,
        "git_dirty": dirty,
        "git_dirty_code_files": code_files or "none",
        "git_diff_sha16": diff_sha16,
    }


def _build_config_snapshot(args, cfg, resolved_architecture, sam2_identity,
                              dist_info, control_group, train_flip_prob,
                              train_vflip_prob, train_ms_resize_prob,
                              train_ms_mode, train_ms_img_scale,
                              train_subset_ratio, val_subset_ratio,
                              warmup_iters, optimizer_steps_per_epoch,
                              total_optimizer_steps):
    """Assemble the self-describing snapshot stored in every checkpoint.

    Also returns the five env-derived selection knobs that main()
    consumes afterwards (eval/bbox-best/composite/early-stop specs).
    """
    # === 构建训练 config 快照 (保存到 checkpoint, 推理时可直接加载) ===
    eval_bbox = _env_flag("EVAL_BBOX", "1")
    # Historical WHU1024 baseline reports/selects bbox/mAP as its primary bbox
    # metric.  Keep mAP_75 in the detailed metrics, but do not silently use it
    # as the headline or best-bbox checkpoint criterion.
    save_bbox_best_metric = os.environ.get("SAVE_BBOX_BEST_METRIC", "bbox/mAP")
    save_composite_best = _env_flag("SAVE_COMPOSITE_BEST", "0")
    composite_metric_spec = os.environ.get(
        "SAVE_COMPOSITE_WEIGHTS", "0.5*bbox/mAP_75+0.5*segm/mAP_75"
    )
    early_stopping_metric_spec = os.environ.get(
        "EARLY_STOPPING_METRIC", "segm/mAP"
    )
    resolved_mask_head_cfg = cfg.model.roi_head.mask_head
    resolved_point_warmup = dict(
        resolved_mask_head_cfg.get("point_warmup_cfg", {})
    )
    resolved_shape_point_miner = dict(
        resolved_mask_head_cfg.get("shape_point_miner_cfg", {})
    )
    resolved_coarse_loss = dict(
        resolved_mask_head_cfg.get("coarse_mask_loss_cfg", {})
    )
    resolved_shape_prior = dict(
        resolved_mask_head_cfg.get("shape_prior_cfg", {})
    )
    config_snapshot = {
        "checkpoint_schema_version": 2,
        "architecture_contract": dict(resolved_architecture),
        "config_path": str(args.config),
        "training_args": dict(vars(args)),
        "train_augmentation": {
            "flip_prob": train_flip_prob,
            "vflip_prob": train_vflip_prob,
            "multi_scale_resize_prob": train_ms_resize_prob,
            "multi_scale_mode": train_ms_mode,
            "multi_scale_img_scale": train_ms_img_scale,
        },
        # HEAD alone cannot identify the executed code when the working tree
        # is dirty; carry the dirty flag + diff fingerprint so a checkpoint
        # uniquely pins its code state (same fields as the runner's
        # resolved_hyperparameters git_* lines).
        "git_state": _capture_git_state(),
        "data_config": {
            "dataset_format": ("vhr10_coco" if (args.dataset == "vhr10") else "whu_coco"),
            "data_root": str(args.data_root),
            "image_size": list(args.image_size),
            "single_class": not (args.dataset == "vhr10"),
            "num_classes": 10 if (args.dataset == "vhr10") else 1,
            "validation": (
                {
                    "ann_file": os.environ.get(
                        "VHR10_VAL_ANN_FILE", "coco_split/instances_val.json"
                    ),
                    "image_subdir": os.environ.get(
                        "VHR10_VAL_IMG_SUBDIR", "positive image set"
                    ),
                }
                if (args.dataset == "vhr10")
                else {
                    "ann_file": "2.4 annotation/annotation/validation.json",
                    "image_subdir": "2.3 valid/validation",
                }
            ),
            "test": (
                {
                    "ann_file": os.environ.get(
                        "VHR10_VAL_ANN_FILE", "coco_split/instances_val.json"
                    ),
                    "image_subdir": os.environ.get(
                        "VHR10_VAL_IMG_SUBDIR", "positive image set"
                    ),
                }
                if (args.dataset == "vhr10")
                else {
                    "ann_file": "2.4 annotation/annotation/test.json",
                    "image_subdir": "2.2 test/test",
                }
            ),
        },
        "runtime_config": {
            "sam2_repo": os.environ.get("SAM2_REPO", ""),
            "sam2_checkpoint": os.environ.get("SAM2_CKPT", ""),
            "sam2_checkpoint_sha256": sam2_identity[1],
            "sam2_model_size": os.environ.get("SAM2_MODEL_SIZE", ""),
            "ddp_timeout_seconds": _resolve_ddp_timeout_seconds(),
            "ddp_control_backend": "gloo" if control_group is not None else "none",
            "ddp_control_timeout_seconds": _resolve_ddp_control_timeout_seconds(),
        },
        "train_subset_ratio": float(train_subset_ratio),
        "val_subset_ratio": float(val_subset_ratio),
        # Display-only mirror of the prompt form (the whole env-mirror family
        # is gone: nothing re-reads these keys at checkpoint load time).
        "explicit_prompt_mode": os.environ.get("EXPLICIT_PROMPT_MODE", ""),
        "neck_type": os.environ.get("NECK_TYPE", ""),
        "eval_bbox": int(eval_bbox),
        "save_bbox_best_metric": save_bbox_best_metric,
        "save_composite_best": int(save_composite_best),
        "composite_metric_spec": composite_metric_spec,
        "early_stopping_metric_spec": early_stopping_metric_spec,
        "early_stopping_patience": int(args.early_stopping_patience),
        "early_stopping_start_epoch": int(args.early_stopping_start_epoch),
        "early_stopping_min_delta": float(args.early_stopping_min_delta),
        "early_stopping_smooth_window": int(
            args.early_stopping_smooth_window
        ),
        "point_warmup": {
            "enabled": bool(resolved_point_warmup.get("enabled", False)),
            "no_point_epochs": int(
                resolved_point_warmup.get("no_point_epochs", 0)
            ),
            "one_pair_epochs": int(
                resolved_point_warmup.get("one_pair_epochs", 0)
            ),
            "full_2p2n_start_epoch": int(
                resolved_point_warmup.get("full_2p2n_start_epoch", 1)
            ),
        },
        "shape_point_mining": {
            "adaptive_validity": bool(
                resolved_shape_point_miner.get("adaptive_validity", False)
            ),
        },
        "shape_loss_schedule": {
            "mode": resolved_coarse_loss.get("schedule_mode", "fixed"),
            "fixed_weight": float(resolved_coarse_loss.get("weight", 0.0)),
            "ranges": [
                dict(item)
                for item in resolved_coarse_loss.get("weight_schedule", [])
            ],
        },
        "shape_dense_gate": {
            "mode": resolved_shape_prior.get("shape_scale_mode", ""),
            "init": resolved_shape_prior.get("prompt_scale_init", ""),
        },
        "shape_prior_loss_weight": float(
            resolved_mask_head_cfg.get("shape_prior_loss_weight", 0.0)
        ),
        "p2_boundary_refiner": dict(
            resolved_mask_head_cfg.get("p2_boundary_refiner_cfg", {})
        ),
        "init_from": args.init_from or "",
        "init_exclude_prefixes": args.init_exclude_prefixes,
        "seed": args.seed,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "epochs": args.epochs,
        "warmup_iters": warmup_iters,
        # 问题 10：scheduler 真实进度依据（optimizer step 维度）
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "total_optimizer_steps": total_optimizer_steps,
        "effective_batch_size": args.batch_size * args.grad_accum_steps * max(1, int(os.environ.get("WORLD_SIZE", "1"))),
        "weight_decay": args.weight_decay,
        "lr_mults": {
            "backbone": args.backbone_lr_mult,
            "other": args.other_lr_mult,
            "bbox_head": args.bbox_head_lr_mult,
            "mask_decoder": args.mask_decoder_lr_mult,
            "no_mask": args.no_mask_lr_mult,
            "prompt_encoder": args.prompt_encoder_lr_mult,
            "shape_prior": args.shape_prior_lr_mult,
        },
        "image_size": list(args.image_size),
        "mask_head_config": dict(cfg.model.roi_head.mask_head)
            if hasattr(cfg.model.roi_head, "mask_head") else {},
        "neck_config": dict(cfg.model.get("neck", {})),
        # 完整解析后的 cfg.model (含 type/backbone/roi_head/...): 推理时可据此直接重建模型,
        # 无需再传 .py config 或重设环境变量。新字段, 旧 ckpt 没有时推理脚本自动 fallback。
        "model_config": cfg.model.to_dict(),
    }

    knobs = (
        eval_bbox, save_bbox_best_metric, save_composite_best,
        composite_metric_spec, early_stopping_metric_spec,
    )
    return config_snapshot, knobs
