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




_CANONICAL_DETECTION_LOSS_KEYS = {
    "loss_rpn_cls",
    "loss_rpn_bbox",
    "loss_cls",
    "loss_bbox",
    "loss_bbox_cls",
    "loss_bbox_reg",
}





# Pure-structure split (2026-10): implementations live in sibling modules.
from cli_args import (
    _env_flag,
    _parse_args,
)
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
            dataset_format=("vhr10_coco" if (args.dataset == "vhr10") else "whu_coco"),
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

    if args.init_from:
        exclude_prefixes = tuple(
            prefix.strip()
            for prefix in args.init_exclude_prefixes.split(",")
            if prefix.strip()
        )
        allowed_missing_prefixes = ()
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

        model.train()
        if args.freeze_bn:
            _set_norm_eval(model)

        total_loss = 0.0
        loss_meter = {}
        dense_monitor_sums = {key: 0.0 for key in _DENSE_RESIDUAL_MONITOR_KEYS}
        dense_monitor_counts = {key: 0 for key in _DENSE_RESIDUAL_MONITOR_KEYS}
        num_batches = 0
        grad_accum = args.grad_accum_steps
        train_batches_limit = len(train_loader)
        if args.max_train_batches > 0:
            train_batches_limit = min(train_batches_limit, args.max_train_batches)
        # SEGMENT_TIMING=1: per-micro-batch phase timers (prep/fwd/item/bwd),
        # averaged and logged every 50 steps.  Zero overhead when disabled.
        _seg_timing = os.environ.get("SEGMENT_TIMING", "0") == "1"
        _seg_sum = {"prep": 0.0, "fwd": 0.0, "item": 0.0, "bwd": 0.0, "total": 0.0}
        _seg_n = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(train_loader):
            if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
                break

            _seg_t0 = time.perf_counter() if _seg_timing else 0.0
            imgs = batch["imgs"].to(device)
            img_metas = batch["img_metas"]
            gt_bboxes = [b.to(device) for b in batch["gt_bboxes"]]
            gt_labels = [l.to(device) for l in batch["gt_labels"]]
            gt_masks = batch["gt_masks"]

            data_samples = _build_gt_data_samples(
                imgs, img_metas, gt_bboxes, gt_labels, gt_masks, device
            )

            # Match the historical WHU1024 baseline: empty-GT training batches
            # still enter the detector loss. A rank-local early continue here
            # would make DDP ranks execute different collective sequences when
            # their DistributedSampler shards contain different empty images.

            # Synchronize data preprocessing for both images and ground truth
            processed = data_preprocessor(
                {"inputs": imgs, "data_samples": data_samples}, training=True
            )
            imgs = processed["inputs"]
            data_samples = processed["data_samples"]

            _seg_t1 = time.perf_counter() if _seg_timing else 0.0
            model_inputs = imgs

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                # Invoke the DDP wrapper itself. Calling ``model.module.loss``
                # bypasses DDP's reducer and makes each rank update independently.
                loss_dict = model(model_inputs, data_samples, mode="loss")
                # The loss channel contains ONLY backward-able losses by
                # construction, so every tensor joins the sum — no key-name
                # filtering to drift out of sync with the model side.  mmdet
                # RPN emits per-image lists; extend preserves the historical
                # addition order bit-for-bit.
                loss_parts = []
                for value in loss_dict.values():
                    if isinstance(value, torch.Tensor):
                        loss_parts.append(value)
                    elif isinstance(value, (list, tuple)):
                        loss_parts.extend(
                            item for item in value if isinstance(item, torch.Tensor)
                        )
                loss = sum(loss_parts) if loss_parts else torch.tensor(0.0, device=device)
                # 审查 3.6 / 问题 10：loss 归一化按当前 batch 所属 accumulation group 的
                # 真实长度，而非固定 grad_accum。之前只在最后一个 batch 修正分母，导致
                # 末组前面的 batch（如 batch_idx=4,5，末组真实只有 3 个）仍除以 grad_accum，
                # 该组总梯度被低估。
                group_start = (batch_idx // grad_accum) * grad_accum
                group_end = min(group_start + grad_accum, train_batches_limit)
                current_group_size = max(1, group_end - group_start)
                loss_for_backward = loss / float(current_group_size)

            _seg_t2 = time.perf_counter() if _seg_timing else 0.0
            if _seg_timing:
                _seg_sum["prep"] += _seg_t1 - _seg_t0
                _seg_sum["fwd"] += _seg_t2 - _seg_t1
            # Record losses plus this iteration's loss-path telemetry.
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    loss_meter[k] = loss_meter.get(k, 0.0) + v.detach().item()
                elif isinstance(v, (list, tuple)):
                    loss_meter[k] = loss_meter.get(k, 0.0) + sum(
                        item.detach().item() for item in v if isinstance(item, torch.Tensor)
                    )
            core = model.module if hasattr(model, "module") else model
            telemetry_stash = getattr(core.roi_head, "_loss_telemetry", None) or {}
            for k, v in telemetry_stash.items():
                if isinstance(v, torch.Tensor):
                    loss_meter[k] = loss_meter.get(k, 0.0) + v.detach().item()
            _accumulate_dense_residual_monitor(
                model, dense_monitor_sums, dense_monitor_counts
            )

            finite_flag = torch.isfinite(loss).to(dtype=torch.int32)
            if distributed and dist.is_initialized():
                dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
            if finite_flag.item() == 0:
                nonfinite_batches += 1
                if is_main:
                    nonfinite_parts = [
                        name
                        for name, value in loss_dict.items()
                        if isinstance(value, torch.Tensor)
                        and not torch.isfinite(value).all()
                    ]
                    logger.error(
                        "Non-finite loss detected (epoch=%d count=%d parts=%s).",
                        epoch + 1,
                        nonfinite_batches,
                        nonfinite_parts or "remote-rank",
                    )
                optimizer.zero_grad(set_to_none=True)
                if (
                    args.max_nonfinite_batches > 0
                    and nonfinite_batches >= args.max_nonfinite_batches
                ):
                    raise FloatingPointError(
                        "Non-finite batch limit reached: "
                        f"{nonfinite_batches}/{args.max_nonfinite_batches}"
                    )
                continue

            global_step = epoch * max(1, train_batches_limit) + batch_idx + 1
            prompt_stats_due = (
                args.prompt_debug_stats
                and args.prompt_debug_stats_interval > 0
                and global_step % args.prompt_debug_stats_interval == 0
            )
            if is_main and prompt_stats_due:
                _log_forward_telemetry(model, epoch, global_step, _PROMPT_TELEMETRY_PREFIXES)
            _seg_t3 = time.perf_counter() if _seg_timing else 0.0
            if _seg_timing:
                _seg_sum["item"] += _seg_t3 - _seg_t2
            scaler.scale(loss_for_backward).backward()
            if _seg_timing:
                _seg_t3b = time.perf_counter()
                _seg_sum["bwd"] += _seg_t3b - _seg_t3
                _seg_sum["total"] += _seg_t3b - _seg_t0
                _seg_n += 1
                if is_main and _seg_n % 100 == 0:
                    logger.info(
                        "TRAIN/TIME (avg %d micro-steps) prep=%.3fs fwd=%.3fs "
                        "item=%.3fs bwd=%.3fs total=%.3fs",
                        _seg_n,
                        _seg_sum["prep"] / _seg_n,
                        _seg_sum["fwd"] / _seg_n,
                        _seg_sum["item"] / _seg_n,
                        _seg_sum["bwd"] / _seg_n,
                        _seg_sum["total"] / _seg_n,
                    )
                    _seg_sum = {k: 0.0 for k in _seg_sum}
                    _seg_n = 0

            # Step optimizer every grad_accum batches or at the last batch
            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == train_batches_limit:
                scaler.unscale_(optimizer)
                if decoupled_clip_partition is not None:
                    base_clip_params, aux_clip_groups = decoupled_clip_partition
                    torch.nn.utils.clip_grad_norm_(base_clip_params, 0.5)
                    for aux_clip_group in aux_clip_groups:
                        torch.nn.utils.clip_grad_norm_(aux_clip_group, 0.5)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
                ema_update_steps += 1
                if (
                    not ddp_sync_audited
                    and any(float(group["lr"]) > 0.0 for group in optimizer.param_groups)
                ):
                    ddp_max_parameter_deviation = _verify_ddp_parameter_sync_after_first_update(
                        model, device, distributed
                    )
                    if ddp_max_parameter_deviation is not None:
                        ddp_sync_audited = True
                    if is_main and ddp_max_parameter_deviation is not None:
                        logger.info(
                            "DDP parameter-sync audit after first nonzero-LR update: "
                            "max_abs_deviation=%.3e",
                            ddp_max_parameter_deviation,
                        )
                if ema is not None and (ema_update_steps % max(1, int(args.ema_update_every)) == 0):
                    ema.update(model)
                optimizer.zero_grad(set_to_none=True)
                # Step scheduler after optimizer step (iteration-level)
                scheduler.step()
                # 问题 10：每 50 个 optimizer step 打印进度，便于确认 cosine/warmup 走对
                if is_main and ema_update_steps % 50 == 0:
                    _cur_lr = optimizer.param_groups[0]["lr"]
                    logger.info(
                        "epoch %d | mini-batch %d/%d | optimizer_step %d/%d | lr=%.3e",
                        epoch + 1, batch_idx + 1, train_batches_limit,
                        ema_update_steps, total_optimizer_steps, _cur_lr,
                    )

            total_loss += float(loss.detach().item())
            num_batches += 1
        avg_loss = total_loss / max(1, num_batches)
        # Collect LRs for all groups
        lr_groups = {}
        for group in optimizer.param_groups:
            name = group.get("name", "unknown")
            lr_groups[name] = group["lr"]

        dense_monitor_epoch = _reduce_dense_residual_monitor(
            dense_monitor_sums,
            dense_monitor_counts,
            device=device,
            distributed=distributed,
        )
        if is_main:
            loss_str = []
            for k, v in loss_meter.items():
                avg_k = v / max(1, num_batches)
                loss_str.append(f"{k}={avg_k:.4f}")
            logger.info(f"Epoch {epoch + 1} detailed losses: {', '.join(loss_str)}")
            if dense_monitor_epoch:
                logger.info(
                    "Epoch %d dense residual monitor: alpha=%.6f "
                    "source_delta_norm=%.4f applied_delta_norm=%.4f "
                    "source_delta_ratio=%.6f applied_delta_ratio=%.6f",
                    epoch_number,
                    dense_monitor_epoch.get("DENSE/residual_alpha", float("nan")),
                    dense_monitor_epoch.get("DENSE/source_delta_norm", float("nan")),
                    dense_monitor_epoch.get("DENSE/applied_delta_norm", float("nan")),
                    dense_monitor_epoch.get("DENSE/source_delta_ratio", float("nan")),
                    dense_monitor_epoch.get("DENSE/applied_delta_ratio", float("nan")),
                )
            averaged_components = {
                key: value / max(1, num_batches) for key, value in loss_meter.items()
            }
            det_component = sum(
                value for key, value in averaged_components.items()
                if key in _CANONICAL_DETECTION_LOSS_KEYS or key.startswith("loss_rpn_")
            )
            final_mask_component = float(averaged_components.get("loss_mask", 0.0))
            shape_component = float(averaged_components.get("loss_shape_prior", 0.0))
            component_total = det_component + final_mask_component + shape_component
            if component_total > 0:
                logger.info(
                    "Epoch %d primary loss shares: det=%.2f%% final_mask=%.2f%% "
                    "shape=%.2f%%",
                    epoch_number,
                    100.0 * det_component / component_total,
                    100.0 * final_mask_component / component_total,
                    100.0 * shape_component / component_total,
                )
            logger.info(
                "Epoch %d schedule: early_stop_start=%d min_delta=%.4g",
                epoch_number,
                int(args.early_stopping_start_epoch),
                float(args.early_stopping_min_delta),
            )
            # Debug: log current learning rates
            lr_str = ", ".join([f"{k}={v:.2e}" for k, v in lr_groups.items()])
            logger.info(f"Epoch {epoch + 1} learning rates: {lr_str}")

        val_loss = None
        val_metrics = None
        ema_applied_for_eval = False
        use_ema_this_epoch = (
            ema is not None
            and bool(args.ema_eval)
            and (epoch + 1) >= max(1, int(args.ema_eval_start_epoch))
        )
        if (
            val_loader is not None
            and (epoch + 1) % int(args.val_every_n_epochs) == 0
        ):
            _val_wall_t0 = time.monotonic()
            if use_ema_this_epoch:
                ema.apply_to(model)
                ema_applied_for_eval = True
            model.eval()
            if args.freeze_bn:
                _set_norm_eval(model)

            # Apply the TEST_MAX_PER_IMG model-side cap BEFORE the inference
            # loop and on EVERY rank (sharded validation runs predict() on all
            # ranks, and the cap doubles as the decoder memory-spike guard).
            # Previously this mutation ran inside the rank0-only eval section
            # after inference, so it only took effect from the second val.
            _val_max_dets = None
            _val_md_env = int(os.environ.get("TEST_MAX_PER_IMG", "0") or 0)
            if _val_md_env > 0:
                _val_max_dets = [1, 10, _val_md_env]
                _val_model_for_eval = (
                    model.module if hasattr(model, "module") else model
                )
                _val_test_cfg = getattr(_val_model_for_eval, "test_cfg", None)
                if _val_test_cfg is not None and "rcnn" in _val_test_cfg:
                    _val_test_cfg["rcnn"]["max_per_img"] = _val_md_env
                _val_roi_cfg = getattr(
                    getattr(_val_model_for_eval, "roi_head", None),
                    "test_cfg",
                    None,
                )
                if _val_roi_cfg is not None and "max_per_img" in _val_roi_cfg:
                    _val_roi_cfg["max_per_img"] = _val_md_env

            collect_gt_records = cached_val_gt_records is None
            all_gt: List[dict] = (
                [] if collect_gt_records else cached_val_gt_records
            )
            all_dt: List[dict] = []
            all_img_metas: List[dict] = []
            total_val_loss = 0.0
            val_batches = 0
            skipped_no_gt_val_loss_batches = 0
            seen_val_batches = 0
            # (global_batch_idx, images_in_batch) for the sharded gather; the
            # global index is rank + local_idx * world_size because the shard
            # loader yields batches rank, rank+W, ... in order.
            shard_batch_spans: List[Tuple[int, int]] = []

            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if args.max_val_batches > 0 and batch_idx >= args.max_val_batches:
                        break
                    seen_val_batches += 1

                    imgs = batch["imgs"].to(device)
                    img_metas = batch["img_metas"]
                    gt_bboxes = [b.to(device) for b in batch["gt_bboxes"]]
                    gt_labels = [l.to(device) for l in batch["gt_labels"]]
                    gt_masks = batch["gt_masks"]

                    data_samples = _build_gt_data_samples(
                        imgs, img_metas, gt_bboxes, gt_labels, gt_masks, device
                    )

                    # Optional validation loss is a second model forward. Public
                    # ablation runners disable it for fast point screening; all
                    # images still participate in prediction and COCO evaluation.
                    val_batch_has_valid_gt = False
                    if args.compute_val_loss:
                        for ds in data_samples:
                            labels = getattr(
                                ds.gt_instances, "labels", torch.tensor([])
                            )
                            if labels.numel() > 0 and (labels >= 0).any():
                                val_batch_has_valid_gt = True
                                break
                        if not val_batch_has_valid_gt:
                            skipped_no_gt_val_loss_batches += 1

                    # Synchronize data preprocessing for both images and ground truth
                    processed = data_preprocessor(
                        {"inputs": imgs, "data_samples": data_samples}, training=False
                    )
                    imgs_processed = processed["inputs"]
                    data_samples = processed["data_samples"]

                    model_inputs = imgs_processed

                    model_for_eval = model.module if hasattr(model, "module") else model

                    # ---- Compute loss ----
                    if args.compute_val_loss and val_batch_has_valid_gt:
                        loss_dict = model_for_eval.loss(model_inputs, data_samples)
                        loss_parts = []
                        for value in loss_dict.values():
                            if isinstance(value, torch.Tensor):
                                loss_parts.append(value)
                            elif isinstance(value, (list, tuple)):
                                loss_parts.extend(
                                    item for item in value if isinstance(item, torch.Tensor)
                                )
                        loss = sum(loss_parts) if loss_parts else torch.tensor(0.0, device=device)
                        total_val_loss += float(loss.detach().item())
                        val_batches += 1

                    # ---- Run prediction for COCO evaluation ----
                    outputs = model_for_eval.predict(model_inputs, data_samples, rescale=False)

                    # Collect GT and predictions
                    for j, output in enumerate(outputs):
                        meta = img_metas[j]
                        img_shape = meta["img_shape"]

                        # GT - from data_samples
                        if collect_gt_records:
                            gt_inst = data_samples[j].gt_instances
                            gt_np = _extract_instances_numpy(
                                gt_inst, img_shape, encode_masks=True
                            )
                            all_gt.append(gt_np)

                        # Predictions - from output.pred_instances (InstanceData)
                        pred_inst = output.pred_instances if hasattr(output, "pred_instances") else output
                        pred_np = _extract_instances_numpy(
                            pred_inst, img_shape, encode_masks=True
                        )
                        all_dt.append(pred_np)

                        all_img_metas.append(meta)

                    if val_shard_enabled:
                        shard_batch_spans.append(
                            (
                                rank + batch_idx * val_shard_world,
                                len(outputs),
                            )
                        )

            if val_shard_enabled:
                _payload = {
                    "spans": shard_batch_spans,
                    "dt": all_dt,
                    "metas": all_img_metas,
                    "gt": all_gt if collect_gt_records else None,
                }
                _gathered = [None] * val_shard_world if is_main else None
                dist.gather_object(_payload, _gathered, dst=0, group=control_group)
                if is_main:
                    _merged: Dict[int, Dict[str, Any]] = {}
                    for _pl in _gathered:
                        if _pl is None:
                            continue
                        _pos = 0
                        for _gb, _cnt in _pl["spans"]:
                            _merged[_gb] = {
                                "dt": _pl["dt"][_pos : _pos + _cnt],
                                "metas": _pl["metas"][_pos : _pos + _cnt],
                                "gt": (
                                    _pl["gt"][_pos : _pos + _cnt]
                                    if _pl["gt"] is not None
                                    else None
                                ),
                            }
                            _pos += _cnt
                    _ordered = sorted(_merged)
                    _total_imgs = sum(len(_merged[g]["dt"]) for g in _ordered)
                    if _total_imgs != val_shard_total:
                        raise RuntimeError(
                            "Sharded validation gather assembled "
                            f"{_total_imgs} images, expected {val_shard_total}"
                        )
                    all_dt = [rec for g in _ordered for rec in _merged[g]["dt"]]
                    all_img_metas = [
                        m for g in _ordered for m in _merged[g]["metas"]
                    ]
                    if collect_gt_records:
                        all_gt = [rec for g in _ordered for rec in _merged[g]["gt"]]
                    logger.info(
                        "Validation shard gather: ranks=%d batches=%d images=%d "
                        "wall=%.1fs",
                        val_shard_world,
                        len(_ordered),
                        _total_imgs,
                        time.monotonic() - _val_wall_t0,
                    )
                else:
                    # Non-main ranks must not feed shard-local records into
                    # the COCO section below; empty lists skip it naturally
                    # (total_gt == 0), matching the unsharded contract.
                    all_gt, all_dt, all_img_metas = [], [], []

            if args.compute_val_loss and skipped_no_gt_val_loss_batches > 0:
                logger.warning(
                    (
                        "Epoch %d skipped val loss on batches with no valid GT: "
                        "%d/%d (%.2f%%); these batches were still included in "
                        "COCO evaluation"
                    ),
                    epoch + 1,
                    skipped_no_gt_val_loss_batches,
                    seen_val_batches,
                    100.0
                    * skipped_no_gt_val_loss_batches
                    / max(1, seen_val_batches),
                )

            val_loss = (
                total_val_loss / max(1, val_batches)
                if args.compute_val_loss
                else None
            )

            if is_main and collect_gt_records:
                cached_val_gt_records = all_gt
                logger.info(
                    "Cached validation GT as RLE records: images=%d",
                    len(cached_val_gt_records),
                )
            elif is_main and len(all_gt) != len(all_dt):
                raise RuntimeError(
                    "Cached validation GT count no longer matches predictions: "
                    f"GT={len(all_gt)} DT={len(all_dt)}"
                )

            # ---- COCO-style evaluation ----
            val_metrics = {}
            total_gt = sum(len(g["labels"]) for g in all_gt)
            total_dt = sum(len(d["scores"]) for d in all_dt if "scores" in d)
            avg_mask_fill, avg_masks_per_image = _summarize_mask_density(all_dt)

            if is_main:
                logger.info(
                    (
                        "Validation samples: GT=%d, DT=%d, "
                        "avg_masks_per_image=%.2f, avg_mask_fill=%.4f"
                    ),
                    total_gt,
                    total_dt,
                    avg_masks_per_image,
                    avg_mask_fill,
                )

            if total_gt > 0:
                # Honor TEST_MAX_PER_IMG during per-epoch validation so that
                # best-model selection runs at the same maxDets contract as
                # the final --test-only report (previously this env only
                # affected the test-only path, silently leaving best selection
                # at the COCO default maxDets=100). The model-side cap mirrors
                # the test-only override: dense tiles would otherwise batch
                # hundreds of instances through the SAM2 decoder (~10 GB
                # spike) only for the eval to keep the top 150 anyway. Ranking
                # is score-based, so metrics are identical either way.
                # (The model-side cap itself is now applied at validation
                # entry on every rank — see the _val_max_dets block above the
                # inference loop; only the eval maxDets list is consumed here.)
                # The segmentation score contract is stored in cfg.model and
                # consumed identically by validation and checkpoint inference.
                # Bbox candidate selection/ranking always remains detector-score based.
                eval_categories = VHR10_CATEGORIES if (args.dataset == "vhr10") else None
                coco_gt, coco_bbox_dt = build_coco_gt_and_dt(
                    all_gt, all_dt, all_img_metas, score_key="scores",
                    categories=eval_categories,
                )
                if eval_bbox:
                    bbox_metrics = run_coco_eval(
                        coco_gt, coco_bbox_dt, iou_type="bbox",
                        max_dets=_val_max_dets,
                    )
                    val_metrics.update(bbox_metrics)
                    logger.info(
                        "Validation bbox/mAP: %.4f bbox/mAP_75: %.4f",
                        bbox_metrics.get("bbox/mAP", 0.0),
                        bbox_metrics.get("bbox/mAP_75", 0.0),
                    )
                if segm_score_key == "scores":
                    coco_segm_dt = coco_bbox_dt
                else:
                    _, coco_segm_dt = build_coco_gt_and_dt(
                        all_gt,
                        all_dt,
                        all_img_metas,
                        score_key=segm_score_key,
                        categories=eval_categories,
                    )
                segm_metrics = run_coco_eval(
                    coco_gt, coco_segm_dt, iou_type="segm",
                    max_dets=_val_max_dets,
                )
                val_metrics.update(segm_metrics)
                logger.info(
                    "Validation segm/mAP: %.4f",
                    segm_metrics.get("segm/mAP", 0.0),
                )

                # Optional compatibility metric at a second maxDets setting.
                # Used by accelerated runs whose primary eval runs at a
                # non-default maxDets (TEST_MAX_PER_IMG) but that still need a
                # curve directly comparable with historical maxDets=100 logs.
                # Stored under suffixed keys so best-model/early-stop lookups
                # on the plain "segm/mAP" key are unaffected.
                compat_max_dets = int(os.environ.get("VAL_COMPAT_MAX_DETS", "0") or 0)
                if compat_max_dets > 0:
                    for _iou_type, _dt in (
                        ("bbox", coco_bbox_dt),
                        ("segm", coco_segm_dt),
                    ):
                        _compat = run_coco_eval(
                            coco_gt,
                            _dt,
                            iou_type=_iou_type,
                            max_dets=[1, 10, compat_max_dets],
                        )
                        val_metrics.update(
                            {
                                f"{k}@md{compat_max_dets}": v
                                for k, v in _compat.items()
                            }
                        )
                        logger.info(
                            "Validation %s/mAP@maxDet%d: %.4f",
                            _iou_type,
                            compat_max_dets,
                            _compat.get(f"{_iou_type}/mAP", 0.0),
                        )

        # Match the baseline's post-validation cleanup and reduce allocator
        # fragmentation across long runs.  Non-main ranks reach this point
        # early and then wait at the existing epoch-end DDP synchronization.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        stop_training = False
        if is_main:
            history["epochs"].append(epoch)
            history["train_losses"].append(avg_loss)
            history["val_losses"].append(val_loss)
            history["val_metrics"].append(val_metrics)
            history["learning_rates"].append(lr_groups)

            # Update best metrics tracking
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss

            if avg_loss < best_train_loss:
                best_train_loss = avg_loss

            # Check if this is the best model based on segm/mAP
            is_best = False
            is_bbox_best = False
            is_composite_best = False
            current_segm_map = 0.0
            current_bbox_score = None
            current_composite_score = None
            if val_metrics is not None and "segm/mAP" in val_metrics:
                current_segm_map = val_metrics["segm/mAP"]
                if current_segm_map > best_segm_map:
                    best_segm_map = current_segm_map
                    is_best = True
                    if is_main:
                        logger.info(f"New best segm/mAP: {best_segm_map:.4f}")

            if val_metrics is not None and save_bbox_best_metric:
                current_bbox_score = _metric_value(val_metrics, save_bbox_best_metric)
                if current_bbox_score is not None and current_bbox_score > best_bbox_score:
                    best_bbox_score = current_bbox_score
                    is_bbox_best = True
                    logger.info("New best %s: %.4f", save_bbox_best_metric, best_bbox_score)

            if val_metrics is not None and save_composite_best:
                current_composite_score = _weighted_metric_value(val_metrics, composite_metric_spec)
                if (
                    current_composite_score is not None
                    and current_composite_score > best_composite_score
                ):
                    best_composite_score = current_composite_score
                    is_composite_best = True
                    logger.info(
                        "New best composite (%s): %.4f",
                        composite_metric_spec,
                        best_composite_score,
                    )

            current_early_stopping_score = None
            if val_metrics is not None:
                current_early_stopping_score = _metric_expression_value(
                    val_metrics, early_stopping_metric_spec
                )
                if current_early_stopping_score is None:
                    raise RuntimeError(
                        "Early-stopping metric could not be resolved: "
                        f"{early_stopping_metric_spec!r}. Available metrics: "
                        f"{sorted(val_metrics.keys())}"
                    )
                if current_early_stopping_score is not None:
                    early_stopping_metric_history.append(
                        float(current_early_stopping_score)
                    )
                    smooth_window = max(
                        1, int(args.early_stopping_smooth_window)
                    )
                    smoothed_early_stopping_score = float(
                        np.mean(
                            early_stopping_metric_history[-smooth_window:]
                        )
                    )
                    improved = (
                        smoothed_early_stopping_score
                        > best_early_stopping_score
                        + float(args.early_stopping_min_delta)
                    )
                    if improved:
                        best_early_stopping_score = (
                            smoothed_early_stopping_score
                        )
                        early_counter = 0
                        logger.info(
                            "New best early-stop metric (%s): raw=%.4f "
                            "smooth(%d)=%.4f",
                            early_stopping_metric_spec,
                            current_early_stopping_score,
                            smooth_window,
                            best_early_stopping_score,
                        )
                    elif epoch_number >= int(args.early_stopping_start_epoch):
                        early_counter += 1
                        logger.info(
                            "Early-stop metric (%s): raw=%.4f smooth(%d)=%.4f "
                            "best=%.4f counter=%d",
                            early_stopping_metric_spec,
                            current_early_stopping_score,
                            smooth_window,
                            smoothed_early_stopping_score,
                            best_early_stopping_score,
                            early_counter,
                        )
                    else:
                        logger.info(
                            "Early-stop metric (%s): raw=%.4f smooth(%d)=%.4f; "
                            "counter disabled before epoch %d",
                            early_stopping_metric_spec,
                            current_early_stopping_score,
                            smooth_window,
                            smoothed_early_stopping_score,
                            int(args.early_stopping_start_epoch),
                        )

            best_metrics = {
                "best_train_loss": best_train_loss,
                "best_val_loss": best_val_loss,
                "best_segm_map": best_segm_map,
                "best_bbox_score": best_bbox_score,
                "best_bbox_metric": save_bbox_best_metric,
                "best_composite_score": best_composite_score,
                "best_composite_metric": composite_metric_spec,
                "best_early_stopping_score": best_early_stopping_score,
                "early_stopping_metric": early_stopping_metric_spec,
                "early_counter": int(early_counter),
                "early_stopping_start_epoch": int(args.early_stopping_start_epoch),
                "early_stopping_min_delta": float(args.early_stopping_min_delta),
                "early_stopping_smooth_window": int(
                    args.early_stopping_smooth_window
                ),
                "early_stopping_metric_history": list(
                    early_stopping_metric_history
                ),
            }

            # Save best model: write to new file first, verify, then delete old.
            # Only rank 0 does checkpoint I/O; all ranks synchronize after this
            # block below. Placing a barrier inside `if is_main` deadlocks DDP.
            ema_state_to_save = ema.state_dict() if ema is not None else None
            if is_best:
                _save_verified_best_checkpoint(
                    checkpoint_dir,
                    "best_model",
                    "segm/mAP",
                    best_segm_map,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    best_metrics,
                    history,
                    config_snapshot,
                    scaler,
                    ema_state=ema_state_to_save,
                )


            if is_bbox_best and current_bbox_score is not None:
                _save_verified_best_checkpoint(
                    checkpoint_dir,
                    "best_bbox_model",
                    save_bbox_best_metric,
                    current_bbox_score,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    best_metrics,
                    history,
                    config_snapshot,
                    scaler,
                    ema_state=ema_state_to_save,
                )

            if is_composite_best and current_composite_score is not None:
                _save_verified_best_checkpoint(
                    checkpoint_dir,
                    "best_composite_model",
                    composite_metric_spec,
                    current_composite_score,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    best_metrics,
                    history,
                    config_snapshot,
                    scaler,
                    ema_state=ema_state_to_save,
                )

            if epoch == args.epochs - 1 and _env_flag("SAVE_LAST_MODEL", "0"):
                last_path = checkpoint_dir / f"last_model_epoch{epoch + 1}.pth"
                _save_checkpoint(
                    last_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_metrics,
                    history,
                    config_snapshot,
                    scaler,
                )
                logger.info("Saved final checkpoint: %s", last_path)

            # Periodic last_checkpoint.pth: overwrite every epoch so a crash can
            # be resumed from the most recent epoch without losing progress.
            # Controlled by SAVE_LAST_CHECKPOINT (default on). Unlike best_model,
            # this is not gated on metric improvement.
            if _env_flag("SAVE_LAST_CHECKPOINT", "1") and epoch < args.epochs - 1:
                _last_ckpt = checkpoint_dir / "last_checkpoint.pth"
                _save_checkpoint(
                    _last_ckpt,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_metrics,
                    history,
                    config_snapshot,
                    scaler,
                    ema_state=ema_state_to_save,
                )

            # Logging: metric evaluation and early stopping do not depend on
            # whether the optional validation-loss forward is enabled.
            lr_str = " ".join([f"{k}={v:.2e}" for k, v in lr_groups.items()])
            if val_metrics is not None:
                segm_info = f" segm/mAP={current_segm_map:.4f}" if current_segm_map > 0 else ""
                bbox_info = (
                    f" {save_bbox_best_metric}={current_bbox_score:.4f}"
                    if current_bbox_score is not None
                    else ""
                )
                composite_info = (
                    f" composite={current_composite_score:.4f}"
                    if current_composite_score is not None
                    else ""
                )
                val_info = (
                    f"{val_loss:.6f}" if val_loss is not None else "disabled"
                )
                logger.info(
                    "Epoch %d/%d | train=%.6f val=%s%s%s%s | best_segm=%.4f best_bbox=%.4f best_comp=%.4f | %s",
                    epoch + 1,
                    args.epochs,
                    avg_loss,
                    val_info,
                    segm_info,
                    bbox_info,
                    composite_info,
                    best_segm_map,
                    best_bbox_score,
                    best_composite_score,
                    lr_str,
                )
            else:
                logger.info(
                    "Epoch %d/%d | train=%.6f | %s",
                    epoch + 1,
                    args.epochs,
                    avg_loss,
                    lr_str,
                )

            if (
                current_early_stopping_score is not None
                and epoch_number >= int(args.early_stopping_start_epoch)
                and args.early_stopping_patience > 0
                and early_counter >= int(args.early_stopping_patience)
            ):
                logger.info(
                    "Early stopping triggered (patience=%d)",
                    int(args.early_stopping_patience),
                )
                stop_training = True

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
