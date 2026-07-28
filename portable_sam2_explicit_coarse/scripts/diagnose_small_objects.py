#!/usr/bin/env python3
"""Diagnose where small objects are lost in the RSPrompter detection chain.

The script is read-only: it loads a trained checkpoint, runs the validation
loader, and reports size-wise RPN proposal recall, RCNN sampler statistics, and
RoI FPN level assignment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.structures import InstanceData
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import bbox2roi, bbox_overlaps
from mmdet.structures.mask import BitmapMasks


SIZE_BINS = {
    "small": (0.0, 32.0**2),
    "medium": (32.0**2, 96.0**2),
    "large": (96.0**2, float("inf")),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_imports() -> None:
    root = _repo_root()
    sys.path.insert(0, str(root))
    import mmdet.models  # noqa: F401
    import models  # noqa: F401
    import rsprompter  # noqa: F401

    if "DetDataPreprocessor" not in MMENGINE_MODELS:
        MMENGINE_MODELS.register_module(
            name="DetDataPreprocessor", module=DetDataPreprocessor
        )


def _build_val_loader(args):
    from data import create_train_loader

    _, val_loader, _ = create_train_loader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        flip_prob=0.0,
        vflip_prob=0.0,
        gaussian_noise_prob=0.0,
        random_erasing_prob=0.0,
        image_size=tuple(args.image_size),
        use_drone=False,
        distributed=False,
        val_ratio=1.0,
        val_batch_size=args.batch_size,
        random_sample=False,
        seed=args.seed,
        dataset_format="whu_coco",
        whu_train_ann_file=args.whu_train_ann_file,
        whu_val_ann_file=args.whu_val_ann_file,
        whu_train_img_subdir=args.whu_train_img_subdir,
        whu_val_img_subdir=args.whu_val_img_subdir,
        whu_single_class=True,
        whu_enable_category_mapping=False,
        num_workers=args.num_workers,
    )
    if val_loader is None:
        raise RuntimeError("Validation loader was not created; check WHU val args.")
    return val_loader


def _apply_minimal_overrides(cfg: Config, args) -> None:
    if args.sam2_model_size:
        cfg.model.backbone.model_size = args.sam2_model_size
        cfg.model.neck.in_channels = f"sam2_hiera_{args.sam2_model_size}"
    if args.fpn_type:
        cfg.model.neck.fpn_type = args.fpn_type
    if args.dense_prompt_mode:
        cfg.model.roi_head.mask_head.dense_prompt_mode = args.dense_prompt_mode
    if args.rpn_train_max_per_img > 0:
        cfg.model.train_cfg.rpn_proposal.max_per_img = int(args.rpn_train_max_per_img)
    if args.rpn_test_max_per_img > 0:
        cfg.model.test_cfg.rpn.max_per_img = int(args.rpn_test_max_per_img)
    if args.rcnn_max_per_img > 0:
        cfg.model.test_cfg.rcnn.max_per_img = int(args.rcnn_max_per_img)
    if 0.0 <= args.rcnn_pos_fraction <= 1.0:
        cfg.model.train_cfg.rcnn.sampler.pos_fraction = float(args.rcnn_pos_fraction)
    if args.iimr_enabled in (0, 1):
        cfg.model.roi_head.iimr.enabled = bool(args.iimr_enabled)
    if args.iimr_num_iterations > 0:
        cfg.model.roi_head.iimr.num_iterations = int(args.iimr_num_iterations)
    if args.iimr_memory_residual_weight >= 0:
        cfg.model.roi_head.iimr.memory_residual_weight = float(
            args.iimr_memory_residual_weight
        )
    if args.iimr_use_learnable_residual in (0, 1):
        cfg.model.roi_head.iimr.use_learnable_residual = bool(
            args.iimr_use_learnable_residual
        )


def _select_state_dict(checkpoint: dict, source: str):
    raw_state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    ema_state = checkpoint.get("ema_state")
    if isinstance(ema_state, dict) and "ema_state" in ema_state:
        ema_state = ema_state["ema_state"]
    if source == "ema":
        if ema_state is None:
            raise KeyError("Checkpoint has no ema_state")
        return "ema", ema_state
    if source == "raw":
        return "raw", raw_state
    if ema_state is not None:
        return "ema", ema_state
    return "raw", raw_state


def _load_checkpoint(model: torch.nn.Module, path: str, source: str) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    selected, state_dict = _select_state_dict(checkpoint, source)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        f"[checkpoint] loaded={path} source={selected} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    if missing[:5]:
        print(f"[checkpoint] missing sample: {missing[:5]}")
    if unexpected[:5]:
        print(f"[checkpoint] unexpected sample: {unexpected[:5]}")


def _make_data_samples(batch: dict, device: torch.device) -> List[DetDataSample]:
    imgs = batch["imgs"]
    img_metas = batch["img_metas"]
    gt_bboxes = [b.to(device) for b in batch["gt_bboxes"]]
    gt_labels = [l.to(device) for l in batch["gt_labels"]]
    gt_masks = batch["gt_masks"]
    data_samples: List[DetDataSample] = []

    for i in range(len(imgs)):
        ds = DetDataSample()
        ds.set_metainfo(img_metas[i])
        gt_instances = InstanceData()
        num_inst = len(gt_labels[i])
        if num_inst > 0:
            valid_mask = gt_labels[i] >= 0
            valid_indices = valid_mask.nonzero().squeeze(-1).cpu()
            if valid_indices.numel() > 0:
                gt_instances.bboxes = gt_bboxes[i][valid_indices]
                gt_instances.labels = gt_labels[i][valid_indices.to(gt_labels[i].device)]
                masks = gt_masks[i][valid_indices.numpy()]
                gt_instances.masks = BitmapMasks(masks, *img_metas[i]["img_shape"])
            else:
                gt_instances.bboxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
                gt_instances.labels = torch.zeros((0,), dtype=torch.int64, device=device)
                gt_instances.masks = BitmapMasks(
                    np.zeros((0, *img_metas[i]["img_shape"]), dtype=np.uint8),
                    *img_metas[i]["img_shape"],
                )
        else:
            gt_instances.bboxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
            gt_instances.labels = torch.zeros((0,), dtype=torch.int64, device=device)
            gt_instances.masks = BitmapMasks(
                np.zeros((0, *img_metas[i]["img_shape"]), dtype=np.uint8),
                *img_metas[i]["img_shape"],
            )
        ds.gt_instances = gt_instances
        data_samples.append(ds)
    return data_samples


def _area_bins(bboxes: torch.Tensor) -> Dict[str, torch.Tensor]:
    wh = (bboxes[:, 2:4] - bboxes[:, 0:2]).clamp(min=0)
    areas = wh[:, 0] * wh[:, 1]
    out = {}
    for name, (lo, hi) in SIZE_BINS.items():
        out[name] = (areas >= lo) & (areas < hi)
    return out


def _init_counter(iou_thresholds: Iterable[float], topks: Iterable[int]):
    return {
        "gt": 0,
        "recall": {
            str(iou): {str(k): 0 for k in topks}
            for iou in iou_thresholds
        },
        "max_iou_sum": 0.0,
    }


def _update_rpn_recall(
    counters: dict,
    gt_bboxes: torch.Tensor,
    proposal_bboxes: torch.Tensor,
    iou_thresholds: List[float],
    topks: List[int],
) -> None:
    bins = _area_bins(gt_bboxes)
    if gt_bboxes.numel() == 0:
        return
    max_topk = max(topks)
    proposals = proposal_bboxes[:max_topk]
    if proposals.numel() == 0:
        max_iou_by_gt = gt_bboxes.new_zeros((gt_bboxes.size(0),))
        overlaps_by_topk = {k: max_iou_by_gt for k in topks}
    else:
        overlaps_by_topk = {}
        for k in topks:
            top_props = proposals[:k]
            if top_props.numel() == 0:
                overlaps_by_topk[k] = gt_bboxes.new_zeros((gt_bboxes.size(0),))
            else:
                overlaps_by_topk[k] = bbox_overlaps(gt_bboxes, top_props).max(dim=1).values

    for name, mask in bins.items():
        n = int(mask.sum().item())
        if n == 0:
            continue
        counters[name]["gt"] += n
        counters[name]["max_iou_sum"] += float(overlaps_by_topk[max(topks)][mask].sum().item())
        for iou in iou_thresholds:
            for k in topks:
                counters[name]["recall"][str(iou)][str(k)] += int(
                    (overlaps_by_topk[k][mask] >= iou).sum().item()
                )


def _update_sampler_stats(counters: dict, sampling_results: list, data_samples: list) -> None:
    for res, ds in zip(sampling_results, data_samples):
        gt_bboxes = ds.gt_instances.bboxes
        if gt_bboxes.numel() == 0:
            continue
        gt_bins = _area_bins(gt_bboxes)
        for name, mask in gt_bins.items():
            counters[name]["gt_seen"] += int(mask.sum().item())

        pos_gt_inds = res.pos_assigned_gt_inds
        counters["all"]["pos"] += int(pos_gt_inds.numel())
        counters["all"]["neg"] += int(res.neg_priors.size(0))
        if pos_gt_inds.numel() == 0:
            continue

        assigned_gt = res.pos_gt_bboxes
        pos_bins = _area_bins(assigned_gt)
        for name, mask in pos_bins.items():
            counters[name]["pos"] += int(mask.sum().item())
        unique_gt = torch.unique(pos_gt_inds.long())
        for name, gt_mask in gt_bins.items():
            if gt_mask.any():
                counters[name]["matched_gt"] += int(gt_mask[unique_gt].sum().item())


def _update_roi_level_stats(counters: dict, roi_extractor, sampling_results: list) -> None:
    pos_priors = [res.pos_priors for res in sampling_results]
    if not pos_priors or sum(p.size(0) for p in pos_priors) == 0:
        return
    rois = bbox2roi(pos_priors)
    levels = roi_extractor.map_roi_levels(rois, roi_extractor.num_inputs)
    offset = 0
    for res in sampling_results:
        n = res.pos_priors.size(0)
        if n == 0:
            continue
        assigned_gt = res.pos_gt_bboxes
        bins = _area_bins(assigned_gt)
        img_levels = levels[offset : offset + n]
        for name, mask in bins.items():
            for lvl in range(roi_extractor.num_inputs):
                counters[name][f"P{lvl + 2}"] += int(((img_levels == lvl) & mask).sum().item())
        offset += n


def _format_pct(num: int, den: int) -> str:
    return "n/a" if den == 0 else f"{100.0 * num / den:.2f}%"


def _print_summary(rpn, sampler, levels, iou_thresholds, topks) -> None:
    print("\n=== RPN proposal recall by GT size ===")
    for name in SIZE_BINS:
        gt = rpn[name]["gt"]
        mean_iou = rpn[name]["max_iou_sum"] / gt if gt else 0.0
        print(f"[{name}] gt={gt} mean_best_iou@top{max(topks)}={mean_iou:.4f}")
        for iou in iou_thresholds:
            vals = []
            for k in topks:
                hit = rpn[name]["recall"][str(iou)][str(k)]
                vals.append(f"R@{k}={hit}/{gt} ({_format_pct(hit, gt)})")
            print(f"  IoU>={iou}: " + ", ".join(vals))

    print("\n=== RCNN sampler positives by assigned GT size ===")
    total_pos = sampler["all"]["pos"]
    total_neg = sampler["all"]["neg"]
    print(f"[all] pos={total_pos} neg={total_neg} pos_fraction={_format_pct(total_pos, total_pos + total_neg)}")
    for name in SIZE_BINS:
        gt_seen = sampler[name]["gt_seen"]
        matched = sampler[name]["matched_gt"]
        pos = sampler[name]["pos"]
        avg_pos = pos / matched if matched else 0.0
        print(
            f"[{name}] gt_seen={gt_seen} matched_gt={matched} "
            f"matched_gt_ratio={_format_pct(matched, gt_seen)} pos_rois={pos} "
            f"pos_share={_format_pct(pos, total_pos)} avg_pos_per_matched_gt={avg_pos:.2f}"
        )

    print("\n=== Positive RoI FPN level distribution ===")
    for name in SIZE_BINS:
        total = sum(levels[name].values())
        parts = [f"{lvl}={cnt} ({_format_pct(cnt, total)})" for lvl, cnt in sorted(levels[name].items())]
        print(f"[{name}] total={total} " + ", ".join(parts))

    print("\n=== Raw JSON ===")
    print(json.dumps({"rpn": rpn, "sampler": sampler, "levels": levels}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Small-object RPN/ROI diagnostic")
    parser.add_argument("--config", default="configs/rsprompter_anchor_satS_v11_sam2_large_full.py")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ckpt-weight-source", choices=["auto", "raw", "ema"], default="auto")
    parser.add_argument(
        "--sam2-ckpt",
        default="",
        help="Optional SAM2 checkpoint path; sets SAM2_CKPT before reading the config.",
    )
    parser.add_argument(
        "--sam2-model-size",
        default="",
        choices=["", "tiny", "small", "base_plus", "large"],
        help="Override cfg.model.backbone.model_size and matching neck in_channels.",
    )
    parser.add_argument("--data-root", default="/data/wangcheng/dataset/WHU-512")
    parser.add_argument("--whu-train-ann-file", default="annotations/train.json")
    parser.add_argument("--whu-val-ann-file", default="annotations/val.json")
    parser.add_argument("--whu-train-img-subdir", default="train/image")
    parser.add_argument("--whu-val-img-subdir", default="val/image")
    parser.add_argument("--image-size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--topk", type=int, nargs="+", default=[100, 300, 1000])
    parser.add_argument("--iou-thrs", type=float, nargs="+", default=[0.5, 0.75])
    parser.add_argument("--fpn-type", default="")
    parser.add_argument("--dense-prompt-mode", default="")
    parser.add_argument("--rpn-train-max-per-img", type=int, default=-1)
    parser.add_argument("--rpn-test-max-per-img", type=int, default=-1)
    parser.add_argument("--rcnn-max-per-img", type=int, default=-1)
    parser.add_argument("--rcnn-pos-fraction", type=float, default=-1.0)
    parser.add_argument("--iimr-enabled", type=int, default=-1)
    parser.add_argument("--iimr-num-iterations", type=int, default=-1)
    parser.add_argument("--iimr-memory-residual-weight", type=float, default=-1.0)
    parser.add_argument("--iimr-use-learnable-residual", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_imports()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.sam2_ckpt:
        os.environ["SAM2_CKPT"] = args.sam2_ckpt
    cfg = Config.fromfile(args.config)
    _apply_minimal_overrides(cfg, args)

    model = MODELS.build(cfg.model)
    _load_checkpoint(model, args.checkpoint, args.ckpt_weight_source)
    model.to(device)
    model.eval()

    val_loader = _build_val_loader(args)
    rpn_counters = {
        name: _init_counter(args.iou_thrs, args.topk) for name in SIZE_BINS
    }
    sampler_counters = defaultdict(lambda: defaultdict(int))
    level_counters = defaultdict(lambda: defaultdict(int))

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            imgs = batch["imgs"].to(device)
            data_samples = _make_data_samples(batch, device)

            x, _, _, _ = model.extract_feat(imgs)
            rpn_results = model.rpn_head.predict(x, data_samples, rescale=False)

            for ds, proposals in zip(data_samples, rpn_results):
                proposal_bboxes = proposals.bboxes
                if hasattr(proposals, "scores"):
                    order = proposals.scores.argsort(descending=True)
                    proposal_bboxes = proposal_bboxes[order]
                _update_rpn_recall(
                    rpn_counters,
                    ds.gt_instances.bboxes,
                    proposal_bboxes,
                    args.iou_thrs,
                    args.topk,
                )

            roi_head = model.roi_head
            sampling_results = []
            for i, proposals in enumerate(rpn_results):
                if not hasattr(proposals, "priors"):
                    proposals.priors = proposals.bboxes
                assign_result = roi_head.bbox_assigner.assign(
                    proposals,
                    data_samples[i].gt_instances,
                    data_samples[i].ignored_instances
                    if hasattr(data_samples[i], "ignored_instances")
                    else None,
                )
                sampling_result = roi_head.bbox_sampler.sample(
                    assign_result,
                    proposals,
                    data_samples[i].gt_instances,
                    feats=[lvl_feat[i][None] for lvl_feat in x],
                )
                sampling_results.append(sampling_result)

            _update_sampler_stats(sampler_counters, sampling_results, data_samples)
            _update_roi_level_stats(
                level_counters, roi_head.mask_roi_extractor, sampling_results
            )

            if (batch_idx + 1) % 20 == 0:
                print(f"[progress] processed {batch_idx + 1} batches")

    _print_summary(rpn_counters, sampler_counters, level_counters, args.iou_thrs, args.topk)


if __name__ == "__main__":
    main()
