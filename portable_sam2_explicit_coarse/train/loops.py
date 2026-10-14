"""The three epoch-phase bodies of the trainer main loop, moved verbatim
(dedented one level) from main() by the 2026-10 structure round.

main() keeps the DDP-sequencing skeleton: epoch head, the calls below,
EMA restore and the stop broadcast. Cross-epoch mutable state
(ema_update_steps / nonfinite_batches / ddp_sync_audited / best-* metrics /
early-stop history) is threaded explicitly through parameters and returns.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

from cli_args import _env_flag
from checkpointing import _save_checkpoint, _save_verified_best_checkpoint
from gt_samples import _build_gt_data_samples, _extract_instances_numpy
from optimizer import _set_norm_eval
from telemetry import (
    _DENSE_RESIDUAL_MONITOR_KEYS,
    _PROMPT_TELEMETRY_PREFIXES,
    _accumulate_dense_residual_monitor,
    _log_forward_telemetry,
    _metric_expression_value,
    _metric_value,
    _reduce_dense_residual_monitor,
    _summarize_mask_density,
    _verify_ddp_parameter_sync_after_first_update,
    _weighted_metric_value,
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


def train_one_epoch(
    args, model, train_loader, optimizer, scheduler, scaler, ema,
    data_preprocessor, device, decoupled_clip_partition, amp_enabled,
    distributed, is_main, epoch, epoch_number, total_optimizer_steps,
    ema_update_steps, nonfinite_batches, ddp_sync_audited,
):
    """One training epoch; returns aggregates and threaded counters."""


    model.train()
    if args.freeze_bn:
        _set_norm_eval(model)

    total_loss = 0.0
    loss_meter = {}
    dense_monitor_sums = {key: 0.0 for key in _DENSE_RESIDUAL_MONITOR_KEYS}
    dense_monitor_counts = {key: 0 for key in _DENSE_RESIDUAL_MONITOR_KEYS}
    ue_monitor = None
    if args.dataset == "ue":
        from ue_telemetry import UEEpochTelemetry
        ue_monitor = UEEpochTelemetry(device)
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
        if "uav_payloads" in batch:
            from data.ue_payload import attach_uav_payload
            attach_uav_payload(data_samples, batch)

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

        if ue_monitor is not None:
            ue_monitor.accumulate(model)
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
            if args.dataset == 'ue' and is_main and prompt_stats_due:
                from ue_telemetry import log_ue_gradients
                log_ue_gradients(model, optimizer, logger)
            if decoupled_clip_partition is not None:
                base_clip_params, aux_clip_groups = decoupled_clip_partition
                torch.nn.utils.clip_grad_norm_(base_clip_params, 0.5)
                for aux_clip_group in aux_clip_groups:
                    torch.nn.utils.clip_grad_norm_(aux_clip_group, 0.5)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scale_before = scaler.get_scale() if ue_monitor is not None else None
            scaler.step(optimizer)
            scaler.update()
            if ue_monitor is not None:
                ue_monitor.record_step(scale_before, scaler.get_scale())
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
    if ue_monitor is not None:
        ue_monitor.finish(logger, epoch_number, distributed, is_main, scaler)
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

    return avg_loss, lr_groups, ema_update_steps, nonfinite_batches, ddp_sync_audited



def validate_one_epoch(
    args, model, ema, val_loader, data_preprocessor, device,
    distributed, is_main, rank, control_group,
    val_shard_enabled, val_shard_world, val_shard_total,
    cached_val_gt_records, segm_score_key, eval_bbox, epoch,
):
    """One validation pass (rank0-only or sharded); returns val results and
    the updated GT-record cache."""

    # utils/ is importable only after the entry's sys.path.insert ran; keep
    # this import lazy (module cache makes repeated calls free).
    from utils.coco_eval_utils import (
        VHR10_CATEGORIES,
        build_coco_gt_and_dt,
        run_coco_eval,
    )

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
        if args.dataset == "ue":
            _val_max_dets = [1,10,100]
            _val_md_env = 0
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
                if "uav_payloads" in batch:
                    from data.ue_payload import attach_uav_payload
                    attach_uav_payload(data_samples, batch)

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
            if args.dataset == "ue":
                from utils.ue_evaluation import dense_diagnostics
                for kind,records in (("bbox",coco_bbox_dt),("segm",coco_segm_dt)):
                    val_metrics.update(dense_diagnostics(coco_gt,records,kind))
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

    return val_loss, val_metrics, ema_applied_for_eval, cached_val_gt_records



def _finalize_epoch(
    args, is_main, epoch, epoch_number, avg_loss, val_loss, val_metrics,
    lr_groups, history, best_val_loss, best_train_loss, best_segm_map,
    best_bbox_score, best_composite_score, best_early_stopping_score,
    early_counter, early_stopping_metric_history,
    checkpoint_dir, config_snapshot, ema, model, optimizer, scheduler,
    scaler, save_bbox_best_metric, save_composite_best,
    composite_metric_spec, early_stopping_metric_spec,
):
    """Rank-0 epoch bookkeeping: history, best tracking, early stopping and
    checkpoint saves. Returns updated cross-epoch state + stop_training."""

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
                ema_state=ema_state_to_save if args.dataset == 'ue' else None,
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

    return (
        stop_training, best_val_loss, best_train_loss, best_segm_map,
        best_bbox_score, best_composite_score, best_early_stopping_score,
        early_counter, early_stopping_metric_history, history,
    )
