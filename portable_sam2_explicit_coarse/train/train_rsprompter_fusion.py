#!/usr/bin/env python

import gc
import logging
import math
import time
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

import torch.distributed as dist
import torch.optim as optim
from torch import Tensor
from torch.utils.data.distributed import DistributedSampler


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("portable_sam_fusion")








# Pure-structure split (2026-10): implementations live in sibling modules.
from cli_args import (
    _env_flag,
    _parse_args,
)
from loops import _finalize_epoch, train_one_epoch, validate_one_epoch
from telemetry import (
    _DENSE_RESIDUAL_MONITOR_KEYS,
    _PROMPT_TELEMETRY_PREFIXES,
    _accumulate_dense_residual_monitor,
    _log_forward_telemetry,
    _log_trainable_state,
    _metric_expression_value,
    _metric_value,
    _reduce_dense_residual_monitor,
    _summarize_mask_density,
    _verify_ddp_parameter_sync_after_first_update,
    _weighted_metric_value,
)
from gt_samples import (
    _build_gt_data_samples,
    _extract_instances_numpy,
    _shard_val_loader,
)
from distributed import (
    _broadcast_stop_training,
    _get_model_core,
    _init_distributed,
    _is_main_process,
    _resolve_ddp_control_timeout_seconds,
    _resolve_ddp_timeout_seconds,
)
from optimizer import (
    _assert_no_mask_embedding_contract,
    _build_optimizer,
    _partition_aux_clip_params,
    _set_norm_eval,
)
from ema import (
    ExponentialMovingAverage,
)
from checkpointing import (
    _load_checkpoint,
    _load_init_checkpoint,
    _save_checkpoint,
    _save_verified_best_checkpoint,
    _set_model_current_epoch,
)
from config_snapshot import (
    _build_config_snapshot,
    _capture_git_state,
)


def main():
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    import ue.dual  # registers the dual-arm model/preprocessor types

    # Canonical multi-class category tables live in utils.coco_eval_utils so
    # the trainer and checkpoint inference share one definition.
    # Canonical COCO-eval implementation lives in utils.coco_eval_utils
    # (single source for trainer, checkpoint inference and probes).
    from utils.coco_eval_utils import (
        VHR10_CATEGORIES,
        _masks_to_rles,
        build_coco_gt_and_dt,
        run_coco_eval,
    )

    import mmdet.models
    from mmengine.config import Config
    from mmengine.registry import MODELS as MMENGINE_MODELS
    from mmdet.registry import MODELS
    from mmdet.models.data_preprocessors import DetDataPreprocessor

    if "DetDataPreprocessor" not in MMENGINE_MODELS:
        MMENGINE_MODELS.register_module(
            name="DetDataPreprocessor", module=DetDataPreprocessor
        )

    import rsprompter

    # === 固定随机种子 (可复现) ===
    import random as _random
    _random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # CUDNN_BENCHMARK=1 trades bit-level reproducibility for conv autotuning
    # on fixed-size inputs (public runners keep the deterministic default).
    cudnn_benchmark = os.environ.get("CUDNN_BENCHMARK", "0") == "1"
    torch.backends.cudnn.deterministic = not cudnn_benchmark
    torch.backends.cudnn.benchmark = cudnn_benchmark
    if int(os.environ.get("RANK", "0")) == 0:
        print(
            f"[seed] Random seed fixed to {args.seed} "
            f"| cudnn.deterministic={not cudnn_benchmark} "
            f"| cudnn.benchmark={cudnn_benchmark}"
        )

    dist_info = _init_distributed()
    distributed = bool(dist_info.get("distributed", 0))
    rank = int(dist_info["rank"])
    control_group = dist_info.get("control_group")

    if torch.cuda.is_available():
        device = f"cuda:{dist_info['local_rank']}" if distributed else "cuda:0"
    else:
        device = "cpu"

    is_main = _is_main_process(rank)
    checkpoint_dir = Path(args.checkpoint_dir)
    if is_main:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "DDP process-group timeout: %d seconds",
            _resolve_ddp_timeout_seconds(),
        )
        if control_group is not None:
            logger.info(
                "DDP validation control: backend=gloo timeout=%d seconds",
                _resolve_ddp_control_timeout_seconds(),
            )

    cfg = Config.fromfile(args.config)
    from rsprompter.architecture_contract import architecture_contract

    resolved_architecture = architecture_contract(cfg.model.to_dict())
    sam2_checkpoint_path = os.environ.get("SAM2_CKPT", "")
    sam2_identity = [sam2_checkpoint_path, ""]
    if is_main and sam2_checkpoint_path:
        from rsprompter.architecture_contract import file_sha256

        sam2_identity[1] = file_sha256(sam2_checkpoint_path)
    if distributed and dist.is_initialized():
        dist.broadcast_object_list(
            sam2_identity, src=0, group=control_group
        )
    if is_main:
        logger.info(
            "Architecture contract: id=%s schema=%d fingerprint=%s",
            resolved_architecture["architecture_id"],
            resolved_architecture["schema_version"],
            resolved_architecture["model_fingerprint"],
        )

    if args.dataset == "ue_raw_surface":
        from ue.contract import assert_whu_model, raw_data_contract
        assert_whu_model(cfg.model.to_dict())
        raw_data_contract(args.data_root, args.raw_split_manifest)
    # Keep the pre-registry config for UE snapshots; MMEngine can consume keys.
    import copy
    ue_model_config = copy.deepcopy(cfg.model.to_dict()) if args.dataset == "ue_raw_surface" else None
    model = MODELS.build(cfg.model)
    model.to(device)

    model_for_preproc = model.module if hasattr(model, "module") else model
    data_preprocessor = model_for_preproc.data_preprocessor
    evaluation_mask_head = model_for_preproc.roi_head.mask_head
    segm_score_mode = getattr(evaluation_mask_head, "segm_score_mode", "detector")
    segm_score_key = getattr(evaluation_mask_head, "segm_score_key", "scores")
    if segm_score_mode not in {"detector", "mask_quality"}:
        raise RuntimeError(f"Invalid segmentation score mode: {segm_score_mode!r}")
    if segm_score_key not in {"scores", "mask_scores"}:
        raise RuntimeError(f"Invalid segmentation score key: {segm_score_key!r}")
    if is_main:
        logger.info(
            "Evaluation score contract: bbox=scores segm=%s mode=%s",
            segm_score_key,
            segm_score_mode,
        )

    if args.freeze_bn:
        _set_norm_eval(model)

    # DUAL-WARMSTART-SCREEN-20261017 (--freeze-base): warm-start screening -
    # every parameter outside the allow-prefixes (the whole satellite stack,
    # including its LoRA adapters) is frozen and pinned to eval mode so the
    # UAV-off forward is bit-identical to the EXP-0009 baseline by
    # construction; only fusion/UAV modules train. Applied BEFORE the DDP
    # wrap so bucket construction sees the final requires_grad state.
    if args.freeze_base:
        if not args.init_from:
            raise SystemExit("--freeze-base requires --init-from (warm-start protocol)")
        prefixes = tuple(p for p in args.freeze_allow_prefixes.split(",") if p)
        if not prefixes:
            raise SystemExit("--freeze-base requires a non-empty --freeze-allow-prefixes")
        # Audit finding (DUAL-WARMSTART-SCREEN implementation review): a
        # blanket requires_grad_(True) over the allow-prefixes would re-enable
        # the peft-LoRA-frozen base of the UAV encoder (70.6M params) and
        # break the "EXP-0015 recipe verbatim" anchor. Trainable = prefix
        # match AND already trainable at build time (LoRA adapters + the
        # modules_to_save neck keep 0015's exact trainable set).
        pre_trainable = {name for name, p in model.named_parameters() if p.requires_grad}
        frozen_params, trainable_names = 0, []
        frozen_modules = []
        for name, module in model.named_modules():
            if name == "":
                continue
            allowed_module = any(
                name == p.rstrip(".") or name.startswith(p) for p in prefixes
            )
            if not allowed_module:
                frozen_modules.append(module)
        for name, param in model.named_parameters():
            if (any(name == p.rstrip(".") or name.startswith(p) for p in prefixes)
                    and name in pre_trainable):
                trainable_names.append(name)
            else:
                param.requires_grad_(False)
                frozen_params += 1
        # pin frozen modules to eval mode; loops.py re-applies after every
        # model.train() call (frozen LoRA dropout must never fire).
        model._warm_frozen_modules = frozen_modules
        if not trainable_names:
            raise SystemExit("--freeze-base froze everything: empty allow-prefixes match")
        if is_main:
            logger.info(
                "WARM-START freeze: frozen_params=%d trainable_tensors=%d prefixes=%s "
                "first=%s last=%s",
                frozen_params, len(trainable_names), list(prefixes),
                trainable_names[0], trainable_names[-1],
            )

    if distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP

        # find_unused_parameters=True：PE freeze_all=True 后有参数不收梯度，
        # 且 mask_head 的 ROI 数量随 batch 动态变化，必须开启避免 DDP hang。
        model = DDP(
            model,
            device_ids=[int(device.split(":")[1])]
            if device.startswith("cuda")
            else None,
            find_unused_parameters=True,
        )

    from data import create_train_loader

    train_subset_ratio = float(args.train_subset_ratio)
    val_subset_ratio = float(args.val_subset_ratio)
    for name, value in (
        ("train_subset_ratio", train_subset_ratio),
        ("val_subset_ratio", val_subset_ratio),
    ):
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be in (0, 1], got {value}")

    # --- train-time augmentation knobs (env-driven; defaults preserve the
    # historical hflip-only protocol, individual wrappers opt in) ---
    train_flip_prob = float(os.environ.get("TRAIN_FLIP_PROB", "0.5"))
    train_vflip_prob = float(os.environ.get("TRAIN_VFLIP_PROB", "0.0"))
    train_ms_resize_prob = float(
        os.environ.get("TRAIN_MULTI_SCALE_RESIZE_PROB", "0.0")
    )
    train_ms_mode = os.environ.get("TRAIN_MULTI_SCALE_MODE", "value")
    train_ms_img_scale_spec = os.environ.get(
        "TRAIN_MULTI_SCALE_IMG_SCALE",
        "896:896,960:960,1024:1024,1088:1088,1152:1152",
    )
    train_ms_img_scale = None
    if train_ms_img_scale_spec.strip():
        train_ms_img_scale = []
        for part in train_ms_img_scale_spec.split(","):
            w_text, h_text = part.strip().split(":", 1)
            train_ms_img_scale.append((int(w_text), int(h_text)))
    for name, value in (
        ("TRAIN_FLIP_PROB", train_flip_prob),
        ("TRAIN_VFLIP_PROB", train_vflip_prob),
        ("TRAIN_MULTI_SCALE_RESIZE_PROB", train_ms_resize_prob),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0,1], got {value}")
    if train_ms_mode not in ("value", "range"):
        raise ValueError(
            f"TRAIN_MULTI_SCALE_MODE must be value or range, got {train_ms_mode}"
        )

    if args.dataset == "ue_raw_surface":
        train_flip_prob = train_vflip_prob = train_ms_resize_prob = 0.0

    train_loader, val_loader, train_dataset = (
        create_train_loader(
            data_root=args.data_root,
            batch_size=args.batch_size,
            # Horizontal flip is safe for dual-stream after fix:
            # only intrinsics cx is adjusted (cx -> W-cx); extrinsics are unchanged.
            # Vertical flip treats nadir imagery as orientation-free; img,
            # bbox and mask are flipped in sync inside the dataset.
            flip_prob=train_flip_prob,
            vflip_prob=train_vflip_prob,
            gaussian_noise_prob=0.0,
            gaussian_noise_std=0.02,
            random_erasing_prob=0.0,
            random_erasing_scale=(0.02, 0.15),
            random_erasing_ratio=(0.3, 3.3),
            # Multi-scale jitter resizes BEFORE the fixed resize back to
            # image_size: the network (and frozen SAM2) always receives its
            # native 1024x1024 input; only object scale varies. Kept off by
            # default — random_erasing stays disabled because it erases image
            # pixels without updating GT masks (label corruption for
            # instance segmentation).
            multi_scale_resize_prob=train_ms_resize_prob,
            multi_scale_mode=train_ms_mode,
            multi_scale_img_scale=train_ms_img_scale,
            image_size=tuple(args.image_size),
            distributed=distributed,
            rank=rank,
            world_size=int(dist_info.get("world_size", 1)),
            val_batch_size=args.val_batch_size,
            dataset_format=("ue_raw_surface" if args.dataset == "ue_raw_surface" else "vhr10_coco" if args.dataset == "vhr10" else "whu_coco"),
            raw_split_manifest=args.raw_split_manifest,
            num_workers=args.num_workers,
            vhr10_train_ann_file=os.environ.get(
                "VHR10_TRAIN_ANN_FILE", "coco_split/instances_train.json"
            ),
            vhr10_val_ann_file=os.environ.get(
                "VHR10_VAL_ANN_FILE", "coco_split/instances_val.json"
            ),
            vhr10_train_img_subdir=os.environ.get(
                "VHR10_TRAIN_IMG_SUBDIR", "positive image set"
            ),
            vhr10_val_img_subdir=os.environ.get(
                "VHR10_VAL_IMG_SUBDIR", "positive image set"
            ),
            whu_train_ann_file=os.environ.get(
                "WHU_TRAIN_ANN_FILE", "2.4 annotation/annotation/train.json"
            ),
            whu_val_ann_file=os.environ.get(
                "WHU_VAL_ANN_FILE", "2.4 annotation/annotation/validation.json"
            ),
            whu_train_img_subdir=os.environ.get(
                "WHU_TRAIN_IMG_SUBDIR", "2.1 train/train"
            ),
            whu_val_img_subdir=os.environ.get(
                "WHU_VAL_IMG_SUBDIR", "2.3 valid/validation"
            ),
            train_subset_ratio=train_subset_ratio,
            val_subset_ratio=val_subset_ratio,
            seed=int(args.seed),
        )
    )
    # Preserve the proven WHU1024 baseline validation contract: rank 0 runs
    # the complete validation loader while the other DDP ranks wait at the
    # next training all-reduce.  Iterating the unsharded loader on every rank
    # duplicates full-resolution mask accumulation and can exhaust host
    # memory.  VAL_SHARD_ACROSS_RANKS=1 opts into sharded validation instead:
    # every rank owns loader batches rank, rank+W, ... (identical batch
    # composition because the val pipeline is deterministic), predictions are
    # gathered to rank 0 over the gloo control group and reassembled in
    # global batch order, so COCO inputs match the rank0-only path exactly.
    val_shard_world = int(dist_info.get("world_size", 1)) if distributed else 1
    val_shard_enabled = (
        os.environ.get("VAL_SHARD_ACROSS_RANKS", "0") == "1"
        and distributed
        and val_shard_world > 1
    )
    val_shard_total = -1
    if val_shard_enabled:
        if args.compute_val_loss:
            raise SystemExit(
                "VAL_SHARD_ACROSS_RANKS=1 requires --compute-val-loss 0: a "
                "shard-mean validation loss would silently change the metric."
            )
        if args.max_val_batches > 0:
            # The gather count check would otherwise fire only on rank 0 and
            # leave the other ranks blocked on the gloo control group until
            # its 86400s timeout.
            raise SystemExit(
                "VAL_SHARD_ACROSS_RANKS=1 is incompatible with "
                "--max-val-batches (truncated shards cannot be reassembled)."
            )
        val_loader, val_shard_total = _shard_val_loader(
            val_loader, rank, val_shard_world
        )
    elif distributed and not is_main:
        val_loader = None
    if is_main:
        logger.info(
            "Validation policy: rank0_only=%d sharded=%d batch_size=%d every_n_epochs=%d compute_loss=%d",
            int(distributed and not val_shard_enabled),
            int(val_shard_enabled),
            int(args.val_batch_size),
            int(args.val_every_n_epochs),
            int(args.compute_val_loss),
        )

    num_batches_per_epoch = len(train_loader)
    if args.max_train_batches > 0:
        num_batches_per_epoch = min(num_batches_per_epoch, args.max_train_batches)

    # 问题 10：scheduler 必须按 optimizer step 计数，而不是 mini-batch 数。
    # 之前 cosine_t_max / warmup_iters 都按 mini-batch 算，但 scheduler.step() 只在
    # optimizer.step() 后调用一次（grad_accum 组才一次），导致 grad_accum>1 时
    # cosine 永远走不到 eta_min、warmup 需要 accum 倍 epoch 才结束。
    grad_accum_for_sched = max(1, int(args.grad_accum_steps))
    optimizer_steps_per_epoch = math.ceil(num_batches_per_epoch / grad_accum_for_sched)
    total_optimizer_steps = max(1, args.epochs * optimizer_steps_per_epoch)

    if is_main:
        logger.info(
            "Dataset size: %d | mini-batch/epoch=%d | grad_accum=%d | "
            "optimizer_steps/epoch=%d | total_optimizer_steps=%d",
            len(train_dataset), num_batches_per_epoch, grad_accum_for_sched,
            optimizer_steps_per_epoch, total_optimizer_steps,
        )

    optimizer = _build_optimizer(
        model.module if hasattr(model, "module") else model,
        lr=args.lr,
        backbone_lr_mult=args.backbone_lr_mult,
        other_lr_mult=args.other_lr_mult,
        bbox_head_lr_mult=args.bbox_head_lr_mult,
        mask_decoder_lr_mult=args.mask_decoder_lr_mult,
        no_mask_lr_mult=args.no_mask_lr_mult,
        prompt_encoder_lr_mult=args.prompt_encoder_lr_mult,
        shape_prior_lr_mult=args.shape_prior_lr_mult,
        decoder_tail_lr_mult=args.decoder_tail_lr_mult,
        weight_decay=args.weight_decay,
        fusion_no_wd=bool(getattr(args, "fusion_no_wd", 0)),
    )
    if is_main:
        _log_trainable_state(model)

    # 问题 10：warmup / cosine 全部按 optimizer step 计数（与 scheduler.step() 调用频率一致）。
    # --warmup-iters 现在表示 optimizer step 数（而非 mini-batch），语义更准确；
    # grad_accum==1 时 optimizer step == mini-batch，行为与旧版完全一致。
    if args.warmup_iters > 0:
        warmup_optimizer_steps = args.warmup_iters
        if is_main:
            logger.info(f"Using user-specified warmup: {warmup_optimizer_steps} optimizer steps")
    elif args.warmup_epochs > 0:
        warmup_optimizer_steps = args.warmup_epochs * optimizer_steps_per_epoch
        if is_main:
            logger.info(f"Using epoch-based warmup: {args.warmup_epochs} epochs "
                        f"({warmup_optimizer_steps} optimizer steps)")
    else:
        warmup_optimizer_steps = min(50, optimizer_steps_per_epoch)
        if is_main:
            logger.info(f"Using default warmup: {warmup_optimizer_steps} optimizer steps")

    cosine_t_max = max(1, total_optimizer_steps - warmup_optimizer_steps)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_t_max, eta_min=args.lr * 0.001
    )
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.001, total_iters=warmup_optimizer_steps
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_optimizer_steps],
    )
    # 保留旧变量名供 config_snapshot 引用（向后兼容）
    warmup_iters = warmup_optimizer_steps

    # AMP scaler (--amp 0 关闭混合精度，用 fp32 训练)
    amp_enabled = bool(args.amp) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    config_snapshot, (
        eval_bbox, save_bbox_best_metric, save_composite_best,
        composite_metric_spec, early_stopping_metric_spec,
    ) = _build_config_snapshot(
        args, cfg, resolved_architecture, sam2_identity, dist_info,
        control_group, train_flip_prob, train_vflip_prob,
        train_ms_resize_prob, train_ms_mode, train_ms_img_scale,
        train_subset_ratio, val_subset_ratio, warmup_iters,
        optimizer_steps_per_epoch, total_optimizer_steps,
    )
    if args.dataset == "ue_raw_surface":
        from ue.contract import baseline_metadata, raw_data_contract
        config_snapshot["model_config"] = ue_model_config
        config_snapshot["ue_baseline"] = baseline_metadata(
            ue_model_config, validation_only=bool(args.max_train_batches or args.max_val_batches
                                                 or train_subset_ratio < 1 or val_subset_ratio < 1))
        config_snapshot["data_config"] = raw_data_contract(
            args.data_root, args.raw_split_manifest, image_size=tuple(args.image_size))

    start_epoch = 0
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    best_segm_map = 0.0  # Track best segm mAP
    best_bbox_score = 0.0
    best_composite_score = 0.0
    best_early_stopping_score = float("-inf")
    early_counter = 0
    early_stopping_metric_history = []
    history = {"epochs": [], "train_losses": [], "val_losses": [], "val_metrics": [], "learning_rates": []}

    # EMA shadow weights (eval/save-best only, never for backprop)
    ema = None
    ema_update_steps = 0
    ddp_sync_audited = False
    ema_enabled = bool(args.ema_enabled) and (0.0 < float(args.ema_decay) < 1.0)
    if ema_enabled:
        ema = ExponentialMovingAverage(model, decay=float(args.ema_decay))
        if is_main:
            logger.info(
                "EMA enabled: decay=%.4f update_every=%d eval=%d save_best=%d eval_start_epoch=%d",
                float(args.ema_decay), int(args.ema_update_every),
                int(args.ema_eval), int(args.ema_eval_start_epoch),
            )

    if args.resume_from and args.init_from:
        raise ValueError("--resume-from and --init-from are mutually exclusive")

    if args.init_from and args.dataset == "ue_raw_surface" and not args.allow_task_init:
        raise ValueError("UE baseline starts from SAM2; task-weight initialization needs a separate protocol (preregistered exception: --allow-task-init)")
    if args.init_from:
        exclude_prefixes = tuple(
            prefix.strip()
            for prefix in args.init_exclude_prefixes.split(",")
            if prefix.strip()
        )
        # warm-start screening: missing keys must be exactly the new dual
        # modules (governance missing-keys check); empty keeps legacy lenient
        # behaviour for the non-UE init paths.
        allowed_missing_prefixes = tuple(
            prefix.strip()
            for prefix in args.init_allowed_missing_prefixes.split(",")
            if prefix.strip()
        )
        init_stats = _load_init_checkpoint(
            Path(args.init_from),
            model,
            exclude_prefixes=exclude_prefixes,
            expected_architecture=resolved_architecture,
            allow_cross_arch=args.allow_cross_arch_init,
            allowed_missing_prefixes=allowed_missing_prefixes,
        )
        if is_main:
            logger.info(
                "Initialized model from %s loaded=%d excluded=%d shape_mismatch=%d unexpected=%d missing=%d migrated=%d prefixes=%s",
                args.init_from,
                init_stats["loaded"],
                init_stats["excluded"],
                init_stats["shape_mismatch"],
                init_stats["unexpected"],
                init_stats["missing"],
                init_stats["migrated"],
                exclude_prefixes or "<none>",
            )
            if init_stats.get("migration_pairs"):
                logger.info(
                    "Checkpoint key migration (INIT_FROM): %s",
                    init_stats["migration_pairs"],
                )
    if args.resume_from:
        if args.dataset == "ue_raw_surface":
            from ue.contract import validate_snapshot
            saved = torch.load(args.resume_from, map_location="cpu", weights_only=False)
            validate_snapshot(saved.get("config_snapshot", {}))
            if saved["config_snapshot"]["ue_baseline"].get("validation_only"):
                raise RuntimeError("validation-only checkpoint cannot resume a formal baseline")
            del saved
        ckpt = _load_checkpoint(
            Path(args.resume_from),
            model,
            optimizer,
            scheduler,
            scaler,
            expected_architecture=resolved_architecture,
        )
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best = ckpt.get("best_metrics", {}) or {}
        best_val_loss = float(best.get("best_val_loss", best_val_loss))
        best_train_loss = float(best.get("best_train_loss", best_train_loss))
        best_segm_map = float(best.get("best_segm_map", best_segm_map))
        best_bbox_score = float(best.get("best_bbox_score", best_bbox_score))
        best_composite_score = float(best.get("best_composite_score", best_composite_score))
        best_early_stopping_score = float(
            best.get("best_early_stopping_score", best_early_stopping_score)
        )
        early_counter = int(best.get("early_counter", early_counter))
        early_stopping_metric_history = list(
            best.get(
                "early_stopping_metric_history",
                early_stopping_metric_history,
            )
        )
        history = ckpt.get("history", history) or history
        if ema is not None and "ema_state" in ckpt:
            ema.load_state_dict(ckpt["ema_state"])
            if is_main:
                logger.info("Restored EMA shadow weights from checkpoint")
        # 恢复 config 快照 (从 checkpoint 加载, 保证续训一致)
        if "config_snapshot" in ckpt:
            saved_snapshot = ckpt["config_snapshot"]
            if isinstance(saved_snapshot, dict):
                # Preserve the resumed run's recorded values while retaining
                # new schema fields for checkpoints produced by newer code.
                config_snapshot = {**config_snapshot, **saved_snapshot}
                config_snapshot["checkpoint_schema_version"] = 2
                config_snapshot["architecture_contract"] = dict(
                    resolved_architecture
                )
        if is_main:
            logger.info(
                "Resumed from %s (start_epoch=%d, best_segm_map=%.4f, best_bbox_score=%.4f, best_composite_score=%.4f)",
                args.resume_from,
                start_epoch,
                best_segm_map,
                best_bbox_score,
                best_composite_score,
            )

    nonfinite_batches = 0
    # Validation order is deterministic within a run. Cache compact GT RLE
    # records after the first validation instead of rebuilding them every epoch.
    cached_val_gt_records: Optional[List[dict]] = None
    decoupled_clip_partition = None
    if args.decoupled_aux_clip:
        decoupled_clip_partition = _partition_aux_clip_params(model)
        if is_main:
            logger.info(
                "Clip decomposition ON: base %d tensors at max_norm 0.5 + aux groups %s",
                len(decoupled_clip_partition[0]),
                [len(group) for group in decoupled_clip_partition[1]],
            )

    for epoch in range(start_epoch, args.epochs):
        epoch_number = epoch + 1
        _set_model_current_epoch(model, epoch_number)
        if distributed:
            sampler = getattr(train_loader, "sampler", None)
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)

        avg_loss, lr_groups, ema_update_steps, nonfinite_batches, ddp_sync_audited = (
            train_one_epoch(
                args, model, train_loader, optimizer, scheduler, scaler, ema,
                data_preprocessor, device, decoupled_clip_partition, amp_enabled,
                distributed, is_main, epoch, epoch_number,
                total_optimizer_steps, ema_update_steps, nonfinite_batches,
                ddp_sync_audited,
            )
        )

        if is_main:
            _model_raw = model.module if hasattr(model, "module") else model
            if hasattr(_model_raw, "drain_telemetry"):
                _tel = _model_raw.drain_telemetry()
                if _tel.get("gate_steps"):
                    logger.info("Epoch %d DUAL telemetry: %s", epoch_number,
                                " ".join(f"{k}={v:.6g}" for k, v in _tel.items()))

        val_loss, val_metrics, ema_applied_for_eval, cached_val_gt_records = (
            validate_one_epoch(
                args, model, ema, val_loader, data_preprocessor, device,
                distributed, is_main, rank, control_group,
                val_shard_enabled, val_shard_world, val_shard_total,
                cached_val_gt_records, segm_score_key, eval_bbox, epoch,
            )
        )

        # Match the baseline's post-validation cleanup and reduce allocator
        # fragmentation across long runs.  Non-main ranks reach this point
        # early and then wait at the existing epoch-end DDP synchronization.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        (
            stop_training, best_val_loss, best_train_loss, best_segm_map,
            best_bbox_score, best_composite_score, best_early_stopping_score,
            early_counter, early_stopping_metric_history, history,
        ) = _finalize_epoch(
            args, is_main, epoch, epoch_number, avg_loss, val_loss,
            val_metrics, lr_groups, history, best_val_loss, best_train_loss,
            best_segm_map, best_bbox_score, best_composite_score,
            best_early_stopping_score, early_counter,
            early_stopping_metric_history, checkpoint_dir, config_snapshot,
            ema, model, optimizer, scheduler, scaler,
            save_bbox_best_metric, save_composite_best,
            composite_metric_spec, early_stopping_metric_spec,
        )

        # Restore training weights after EMA-based eval/save.  This must run
        # on EVERY rank: with VAL_SHARD_ACROSS_RANKS=1 all ranks apply EMA
        # inside the validation block, and a rank that keeps the shadow
        # weights would silently desync DDP training.  It stays after rank
        # 0's checkpoint saves and epoch logging (unchanged rank-0 ordering)
        # and before the stop broadcast below.
        if ema_applied_for_eval:
            ema.restore(model)
            ema_applied_for_eval = False

        # Rank 0 decides whether to stop after its rank-0-only validation. The
        # CPU/Gloo broadcast doubles as the epoch boundary: non-main ranks wait
        # here without keeping an NCCL collective alive during long COCO eval.
        if distributed and dist.is_initialized() and control_group is not None:
            stop_training = _broadcast_stop_training(
                stop_training, control_group
            )
        if stop_training:
            break

    if distributed and dist.is_initialized():
        if control_group is not None:
            dist.destroy_process_group(control_group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
