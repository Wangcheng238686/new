#!/usr/bin/env python
"""RSPrompter Fusion inference script with COCO-style evaluation metrics.

Metric computation follows migrated_project/tools/test.py which uses
mmengine Runner + CocoMetric (pycocotools.COCOeval).
"""

import argparse
import json
import logging
import os
import os.path as osp
import sys
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from mmdet.structures import DetDataSample
from mmdet.structures.mask import BitmapMasks
from mmengine.config import Config, DictAction
from mmengine.structures import InstanceData

# Ensure project root is importable when launched as a file path.
project_root = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.coco_eval_utils import build_coco_gt_and_dt, run_coco_eval

# Register components (same as train script)
import mmdet.models
from mmdet.registry import MODELS
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmdet.models.data_preprocessors import DetDataPreprocessor

if "DetDataPreprocessor" not in MMENGINE_MODELS:
    MMENGINE_MODELS.register_module(name="DetDataPreprocessor", module=DetDataPreprocessor)

import rsprompter
import models

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("inference")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def visualize_comparison(img_path, pred_instances, gt_instances, output_path, score_thr=0.3, target_size=None):
    """Visualize original image, ground truth, and predictions side-by-side."""
    img_orig = cv2.imread(img_path)
    if img_orig is None:
        return
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    
    target_h, target_w = img_orig.shape[:2]
    
    if target_size is not None:
        # target_size is (H, W)
        target_h, target_w = target_size
    # Try to infer target size from instances
    elif gt_instances is not None and hasattr(gt_instances, "metainfo"):
        if "img_shape" in gt_instances.metainfo:
            target_h, target_w = gt_instances.metainfo["img_shape"][:2]
    elif pred_instances is not None and hasattr(pred_instances, "metainfo"):
        if "img_shape" in pred_instances.metainfo:
            target_h, target_w = pred_instances.metainfo["img_shape"][:2]
    # Fallback: check if instances have masks and use their shape
    elif gt_instances is not None and hasattr(gt_instances, "masks"):
        masks = gt_instances.masks
        if hasattr(masks, "masks"):
            target_h, target_w = masks.masks.shape[1:]
        elif isinstance(masks, torch.Tensor):
            target_h, target_w = masks.shape[1:]
        elif isinstance(masks, np.ndarray):
            target_h, target_w = masks.shape[1:]

    # Resize original image to match instance coordinates
    img = cv2.resize(img_orig, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    axes[0].imshow(img)
    axes[0].set_title(f"Original Image ({target_w}x{target_h})")
    axes[0].axis("off")

    def draw_instances(ax, instances, title, color):
        vis_img = img.copy()
        if instances is None or len(instances) == 0:
            ax.imshow(vis_img)
            ax.set_title(f"{title} (0 items)")
            ax.axis("off")
            return

        bboxes = instances.bboxes
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.cpu().numpy()

        scores = None
        if hasattr(instances, "scores"):
            scores = instances.scores
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()

        masks = None
        if hasattr(instances, "masks"):
            masks = instances.masks
            if hasattr(masks, "masks"):
                masks = masks.masks
            elif hasattr(masks, "to_ndarray"):
                masks = masks.to_ndarray()
            elif isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()

        if scores is not None:
            valid = scores > score_thr
            if not valid.any():
                ax.imshow(vis_img)
                ax.set_title(f"{title} (0 items)")
                ax.axis("off")
                return
            bboxes = bboxes[valid]
            if masks is not None:
                masks = masks[valid]
            scores = scores[valid]

        count = len(bboxes)
        for i in range(count):
            x1, y1, x2, y2 = bboxes[i].astype(int)
            if masks is not None:
                mask = masks[i]
                if mask.shape[:2] != vis_img.shape[:2]:
                    mask = cv2.resize(mask.astype(np.uint8), (vis_img.shape[1], vis_img.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
                mask_colored = np.zeros_like(vis_img)
                mask_colored[mask > 0] = color
                vis_img = cv2.addWeighted(vis_img, 1, mask_colored, 0.5, 0)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            if scores is not None:
                cv2.putText(vis_img, f"{scores[i]:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        ax.imshow(vis_img)
        ax.set_title(f"{title} ({count} items)")
        ax.axis("off")

    draw_instances(axes[1], gt_instances, "Ground Truth", (0, 255, 0))
    draw_instances(axes[2], pred_instances, "Prediction", (255, 0, 0))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def _prepare_data_samples(batch: Dict, device: str) -> List[DetDataSample]:
    """Prepare DetDataSample list from batch, mirroring the training script."""
    imgs = batch["imgs"]
    img_metas = batch["img_metas"]
    gt_bboxes_list = [b.to(device) for b in batch["gt_bboxes"]]
    gt_labels_list = [l.to(device) for l in batch["gt_labels"]]
    gt_masks_list = batch["gt_masks"]

    data_samples: List[DetDataSample] = []
    for i in range(len(imgs)):
        ds = DetDataSample()
        ds.set_metainfo(img_metas[i])
        gt_instances = InstanceData()
        num_inst = len(gt_labels_list[i])
        if num_inst > 0:
            valid_mask = gt_labels_list[i] >= 0
            num_valid = int(valid_mask.sum().item())
            if num_valid > 0:
                valid_indices = valid_mask.nonzero().squeeze(-1)[:num_valid].cpu()
                gt_instances.bboxes = gt_bboxes_list[i][valid_indices]
                gt_instances.labels = gt_labels_list[i][valid_indices]
                masks = gt_masks_list[i][valid_indices.cpu().numpy()]
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
        data_samples.append(ds.to(device))
    return data_samples


def _extract_instances_numpy(
    instances, img_shape: Tuple[int, int]
) -> dict:
    """Extract bboxes, labels, scores, masks from InstanceData to numpy."""
    bboxes = instances.bboxes
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.detach().cpu().numpy()
    labels = instances.labels
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    scores = None
    if hasattr(instances, "scores"):
        scores = instances.scores
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()

    masks = None
    if hasattr(instances, "masks"):
        masks = instances.masks
        if hasattr(masks, "masks"):
            masks = masks.masks  # BitmapMasks → ndarray
        elif hasattr(masks, "to_ndarray"):
            masks = masks.to_ndarray()
        elif isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()

    # Ensure masks shape matches img_shape
    h, w = img_shape[:2]
    if masks is not None and len(masks) > 0 and masks.shape[1:] != (h, w):
        resized = []
        for m in masks:
            resized.append(
                cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            )
        masks = np.stack(resized, axis=0)

    if masks is None or len(masks) == 0:
        masks = np.zeros((0, h, w), dtype=np.uint8)

    out = {"bboxes": bboxes, "labels": labels, "masks": masks}
    if scores is not None:
        out["scores"] = scores
    return out


def _resolve_use_drone(args, enable_drone_branch: bool) -> bool:
    """Resolve whether to enable drone guidance based on args and model capability."""
    if args.no_drone:
        logger.info("Drone guidance force disabled via --no-drone flag")
        return False

    if args.use_drone and not enable_drone_branch:
        logger.warning(
            "--use-drone specified but model has enable_drone_branch=False. "
            "Will run in satellite-only mode."
        )
        return False

    use_drone = args.use_drone or enable_drone_branch
    if use_drone:
        logger.info("Drone guidance enabled")
    else:
        logger.info("Drone guidance disabled")
    return use_drone


def _log_inference_summary(
    test_dataset,
    args,
    sat_data_root: str,
    use_drone: bool,
    drone_data_root: str,
    metrics: Dict[str, float],
    visualized_count: int,
):
    logger.info("Inference samples: %d", len(test_dataset))
    logger.info("Config: %s", args.config)
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Satellite data root: %s", sat_data_root)
    if use_drone:
        logger.info("Drone data root: %s", drone_data_root)
        logger.info("Num views: %d", args.num_views)

    logger.info("Evaluation metrics:")
    for k, v in metrics.items():
        logger.info("  %s: %.4f", k, v)

    if args.show_dir:
        logger.info("Visualization output: %s (%d images)", args.show_dir, visualized_count)
    if args.work_dir:
        logger.info("Evaluation metrics saved: %s", osp.join(args.work_dir, "eval_results.json"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="RSPrompter Inference Script")
    parser.add_argument("config", help="config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--work-dir", help="directory to save evaluation metrics")
    parser.add_argument("--show-dir", help="directory to save visualizations")
    parser.add_argument("--show-score-thr", type=float, default=0.3,
                        help="score threshold for visualization")
    parser.add_argument("--max-num", type=int, default=100, help="max images to visualize")
    parser.add_argument("--gpu-id", type=int, default=0, help="gpu id")
    parser.add_argument("--data-root", help="satellite test data root")
    parser.add_argument("--drone-data-root", help="drone test data root")
    parser.add_argument("--use-drone", action="store_true", default=False,
                        help="use drone guidance (auto-detected from model config if not specified)")
    parser.add_argument("--no-drone", action="store_true",
                        help="force disable drone guidance even if model supports it")
    parser.add_argument("--random-sample", action="store_true",
                        help="randomly sample drone images")
    parser.add_argument("--num-views", type=int, default=4, help="number of drone views")
    parser.add_argument("--image-size", type=int, nargs=2, default=[1024, 1024])
    parser.add_argument("--drone-image-size", type=int, nargs=2, default=[512, 512])
    
    # WHU dataset arguments
    parser.add_argument("--dataset-format", type=str, default="labelme", choices=["labelme", "whu_coco"])
    parser.add_argument("--whu-val-ann-file", type=str, default="annotations/val.json")
    parser.add_argument("--whu-val-img-subdir", type=str, default="val/image")
    parser.add_argument("--whu-single-class", type=int, default=1)
    parser.add_argument("--whu-enable-category-mapping", type=int, default=0)
    parser.add_argument(
        "--ckpt-weight-source",
        type=str,
        default="raw",
        choices=["raw", "ema", "auto"],
        help="Choose which weights to load from checkpoint: raw=model, ema=ema_state, auto=prefer ema_state when available",
    )
    parser.add_argument(
        "--ema-fallback-to-raw",
        type=int,
        default=1,
        help="When loading EMA fails or mismatches, fallback to raw model weights (1=yes, 0=no)",
    )
    parser.add_argument(
        "--train-args-priority",
        type=str,
        default="env",
        choices=["checkpoint", "env"],
        help="checkpoint: train_args restored from checkpoint has highest priority and cfg-options will not override; env: use cfg-options as final override",
    )

    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Restore training config from saved args
# ---------------------------------------------------------------------------

# IIMR args that affect model structure or inference behavior.
# Maps (arg_name, cfg_path) where cfg_path is relative to cfg.model.
_IIMR_ARGS = [
    # (arg_name, cfg_dot_path, value_type)
    ("iimr_enabled", "roi_head.iimr.enabled", bool),
    ("iimr_num_iterations", "roi_head.iimr.num_iterations", int),
    ("iimr_use_roi_pos_encoding", "roi_head.iimr.use_roi_pos_encoding", bool),
    ("iimr_use_geometry_aware_kv", "roi_head.iimr.use_geometry_aware_kv", bool),
    ("iimr_use_topology_memory", "roi_head.iimr.use_topology_memory", bool),
    ("iimr_topology_to_decoder", "roi_head.iimr.topology_to_decoder", bool),
    ("iimr_memory_residual_weight", "roi_head.iimr.memory_residual_weight", float),
    ("iimr_intermediate_loss_weight", "roi_head.iimr.intermediate_loss_weight", float),
    ("iimr_use_learnable_residual", "roi_head.iimr.use_learnable_residual", bool),
    ("iimr_detach_mask_path", "roi_head.iimr.detach_mask_path", bool),
    ("iimr_use_dynamic_prompting", "roi_head.iimr.use_dynamic_prompting", bool),
    ("iimr_num_uncertainty_points", "roi_head.iimr.num_uncertainty_points", int),
    ("iimr_dynamic_balance_ratio", "roi_head.iimr.dynamic_balance_ratio", float),
    ("iimr_dynamic_min_point_distance", "roi_head.iimr.dynamic_min_point_distance", int),
    ("iimr_dynamic_start_iteration", "roi_head.iimr.dynamic_start_iteration", int),
    ("iimr_dynamic_point_mix", "roi_head.iimr.dynamic_point_mix", float),
    ("iimr_dynamic_fusion_mode", "roi_head.iimr.dynamic_fusion_mode", str),
    ("iimr_use_boundary_uncertainty", "roi_head.iimr.use_boundary_uncertainty", bool),
    ("iimr_boundary_kernel_size", "roi_head.iimr.boundary_kernel_size", int),
    ("iimr_boundary_mix_weight", "roi_head.iimr.boundary_mix_weight", float),
    ("iimr_adaptive_balance", "roi_head.iimr.adaptive_balance", bool),
    ("iimr_use_gt_topology_tokens", "roi_head.iimr.use_gt_topology_tokens", bool),
    ("iimr_gt_topology_only_first_iter", "roi_head.iimr.gt_topology_only_first_iter", bool),
]

# Structural args that affect model architecture (must match training config).
_STRUCTURAL_ARGS = [
    ("dense_prompt_mode", "roi_head.mask_head.dense_prompt_mode", str),
    ("fpn_type", "neck.fpn_type", str),
    ("topdown_downstream_head", "roi_head.topdown_downstream_head", str),
]

# Test-time args that affect evaluation metrics (must match training validation).
_TEST_CFG_ARGS = [
    ("rcnn_max_per_img", "test_cfg.rcnn.max_per_img", int),
    ("rpn_test_max_per_img", "test_cfg.rpn.max_per_img", int),
    ("rcnn_score_thr", "test_cfg.rcnn.score_thr", float),
    ("mask_thr_binary", "test_cfg.rcnn.mask_thr_binary", float),
]

# Sentinel values that mean "don't override" for each type
_SKIP_VALUES = {bool: None, int: -1, float: -1.0, str: ""}


def _apply_train_args_to_cfg(cfg, train_args: dict):
    """Override cfg.model IIMR, structural and test_cfg settings from saved training arguments."""
    for arg_name, cfg_path, vtype in _IIMR_ARGS + _STRUCTURAL_ARGS + _TEST_CFG_ARGS:
        val = train_args.get(arg_name, _SKIP_VALUES[vtype])
        skip = _SKIP_VALUES[vtype]
        if val == skip:
            continue

        # Navigate to parent key and set final key
        parts = cfg_path.split(".")
        # cfg_path is defined relative to cfg.model
        obj = cfg.model
        for p in parts[:-1]:
            if not hasattr(obj, p):
                logger.warning(
                    "Skip restoring %s -> %s: missing cfg path segment '%s'",
                    arg_name,
                    cfg_path,
                    p,
                )
                obj = None
                break
            obj = getattr(obj, p)
        if obj is None:
            continue
        setattr(obj, parts[-1], vtype(val))

    # Log what was restored
    n_total = len(_IIMR_ARGS) + len(_STRUCTURAL_ARGS) + len(_TEST_CFG_ARGS)
    logger.info("Restored %d params (IIMR + structural + test_cfg) from checkpoint train_args", n_total)


def _select_checkpoint_weights(checkpoint: dict, weight_source: str):
    raw_state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    ema_state = checkpoint.get("ema_state", None)
    # Compatible with wrapped EMA formats, e.g. {"decay": ..., "ema_state": {...}}
    if isinstance(ema_state, dict) and "ema_state" in ema_state and isinstance(ema_state.get("ema_state"), dict):
        ema_state = ema_state["ema_state"]
    has_ema_state = isinstance(ema_state, dict) and len(ema_state) > 0

    if weight_source == "raw":
        return "raw", raw_state, has_ema_state
    if weight_source == "ema":
        if not has_ema_state:
            raise KeyError("ema_state not found in checkpoint")
        return "ema", ema_state, has_ema_state

    # auto
    if has_ema_state:
        return "ema", ema_state, has_ema_state
    return "raw", raw_state, has_ema_state


def _apply_inference_resolution_overrides(cfg, image_size: Tuple[int, int]):
    """Override resolution-coupled config fields for inference-time resizing.

    This keeps inference stable when users switch between 512/1024 via CLI
    without needing a dedicated config per size.
    """
    img_h, img_w = int(image_size[0]), int(image_size[1])
    if img_h <= 0 or img_w <= 0:
        return

    # shared_image_embedding in this codebase uses a single square size.
    if img_h == img_w:
        try:
            if hasattr(cfg.model, "shared_image_embedding"):
                cfg.model.shared_image_embedding.image_size = img_h
        except Exception:
            pass

    # Keep data preprocessor pad size aligned when the field exists.
    try:
        data_preprocessor = getattr(cfg.model, "data_preprocessor", None)
        if data_preprocessor is not None and hasattr(data_preprocessor, "batch_augments"):
            for aug in data_preprocessor.batch_augments:
                if getattr(aug, "type", "") == "BatchFixedSizePad":
                    aug.size = (img_h, img_w)
    except Exception:
        pass

    # train_cfg/mask_size is mostly training-time, but keep it aligned for consistency.
    try:
        train_cfg = getattr(cfg.model, "train_cfg", None)
        if train_cfg is not None and hasattr(train_cfg, "rcnn"):
            train_cfg.rcnn.mask_size = (img_h, img_w)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # ---- Load checkpoint early to extract train_args ----
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint.get("train_args", None)

    checkpoint_cfg_locked = False
    if train_args is not None and args.train_args_priority == "checkpoint":
        # Lock checkpoint train_args as highest priority.
        _apply_train_args_to_cfg(cfg, train_args)
        checkpoint_cfg_locked = True
        logger.info("Restored training config from checkpoint (priority=checkpoint, cfg-options overrides disabled)")
    elif train_args is not None:
        logger.info("Checkpoint train_args found but priority=env; cfg-options can override config")
    else:
        logger.warning(
            "Checkpoint has no train_args. Using config file defaults + --cfg-options. "
            "Results may be incorrect if training overrides were used."
        )

    if args.cfg_options is not None:
        if checkpoint_cfg_locked:
            logger.info("Ignoring --cfg-options because train-args-priority=checkpoint and train_args exists")
        else:
            cfg.merge_from_dict(args.cfg_options)

    _apply_inference_resolution_overrides(cfg, tuple(args.image_size))
    logger.info("Inference resolution override: image_size=%sx%s", args.image_size[0], args.image_size[1])

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"

    # ---- Build model ----
    model = MODELS.build(cfg.model)
    model.to(device)
    model_base = model.module if hasattr(model, "module") else model
    data_preprocessor = model_base.data_preprocessor

    # ---- Load weights ----
    # raw: strict=True only when train_args exist; legacy checkpoints use strict=False.
    # ema: always strict=False to tolerate slight key drifts; can fallback to raw if requested.
    strict_load_raw = train_args is not None
    fallback_to_raw = bool(args.ema_fallback_to_raw)

    selected_source, state_dict, has_ema_state = _select_checkpoint_weights(
        checkpoint, args.ckpt_weight_source
    )
    strict_load = strict_load_raw if selected_source == "raw" else False

    try:
        mismatch = model_base.load_state_dict(state_dict, strict=strict_load)
    except Exception as exc:
        if selected_source == "ema" and fallback_to_raw:
            logger.warning(
                "Failed to load EMA weights (%s). Falling back to raw model weights.",
                exc,
            )
            selected_source = "raw"
            strict_load = strict_load_raw
            raw_state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
            mismatch = model_base.load_state_dict(raw_state, strict=strict_load)
        else:
            raise

    # EMA loaded with strict=False: fallback to raw when mismatch is significant and allowed.
    if selected_source == "ema" and (mismatch.missing_keys or mismatch.unexpected_keys):
        if fallback_to_raw:
            logger.warning(
                "EMA load mismatch: missing=%d unexpected=%d. Falling back to raw model weights.",
                len(mismatch.missing_keys), len(mismatch.unexpected_keys),
            )
            selected_source = "raw"
            strict_load = strict_load_raw
            raw_state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
            mismatch = model_base.load_state_dict(raw_state, strict=strict_load)
        else:
            logger.warning(
                "EMA load mismatch kept: missing=%d unexpected=%d",
                len(mismatch.missing_keys), len(mismatch.unexpected_keys),
            )

    if selected_source == "raw" and not strict_load:
        if mismatch.missing_keys or mismatch.unexpected_keys:
            logger.warning(
                "Non-strict raw load (legacy checkpoint). missing=%d unexpected=%d. "
                "Results may be incorrect. Re-train to save train_args.",
                len(mismatch.missing_keys), len(mismatch.unexpected_keys),
            )

    logger.info(
        "Loaded checkpoint from %s (source=%s strict=%s has_ema_state=%s)",
        args.checkpoint,
        selected_source,
        strict_load,
        has_ema_state,
    )
    model.eval()

    # ---- Create dataloader ----
    from data.satellite_drone_dataset import (
        SatelliteDroneDataset, rtmdet_drone_collate_fn,
    )
    from data.satellite_dataset import SatelliteInstanceDataset
    from data.loader import rtmdet_collate_fn
    from torch.utils.data import DataLoader

    sat_data_root = (args.data_root or
                     "/home/wangcheng/data/unversity-big-after-without-negative/university-s-test")
    drone_data_root = (args.drone_data_root or
                       "/home/wangcheng/data/unversity-big-after-without-negative/university-d-test-selected-5")
    image_size = tuple(args.image_size)
    drone_image_size = tuple(args.drone_image_size)

    enable_drone_branch = getattr(model_base, "enable_drone_branch", False)
    use_drone = _resolve_use_drone(args, enable_drone_branch)

    if use_drone:
        test_dataset = SatelliteDroneDataset(
            satellite_data_root=sat_data_root,
            drone_data_root=drone_data_root,
            scene_ids=None,
            image_size=image_size,
            drone_image_size=drone_image_size,
            num_sample_images=args.num_views,
            random_sample=args.random_sample,
            normalize_drone=True,
        )
        collate_fn = rtmdet_drone_collate_fn
    else:
        if args.dataset_format == "whu_coco":
            from data.whu_instance_dataset import WHUCocoInstanceDataset
            test_dataset = WHUCocoInstanceDataset(
                data_root=sat_data_root,
                ann_file=args.whu_val_ann_file,
                image_subdir=args.whu_val_img_subdir,
                image_size=image_size,
                single_class=bool(args.whu_single_class),
                enable_category_mapping=bool(args.whu_enable_category_mapping),
                flip_prob=0.0,
                vflip_prob=0.0,
            )
        else:
            test_dataset = SatelliteInstanceDataset(
                satellite_data_root=sat_data_root,
                scene_ids=None,
                image_size=image_size,
                val_ratio=0.0,
                is_val=False,
            )
        collate_fn = rtmdet_collate_fn

    dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=collate_fn, pin_memory=True, drop_last=False,
    )
    logger.info("Test dataset: %d samples", len(test_dataset))

    # ---- Prepare output dirs ----
    if args.show_dir:
        os.makedirs(args.show_dir, exist_ok=True)
    if args.work_dir:
        os.makedirs(args.work_dir, exist_ok=True)

    # ---- Inference loop ----
    all_gt: List[dict] = []
    all_dt: List[dict] = []
    all_img_metas: List[dict] = []
    visualized_count = 0
    # Loss meters
    total_loss = 0.0
    loss_meter: Dict[str, float] = {}
    num_loss_batches = 0

    logger.info("Start inference")

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Inference")):
        with torch.no_grad():
            imgs = batch["imgs"].to(device)
            img_metas = batch["img_metas"]

            # ---- Prepare data_samples ----
            data_samples = _prepare_data_samples(batch, device)

            batch_has_valid_gt = any(
                ds.gt_instances.labels.numel() > 0 and (ds.gt_instances.labels >= 0).any()
                for ds in data_samples
            )

            # ---- Data preprocessing ----
            processed = data_preprocessor(
                {"inputs": imgs, "data_samples": data_samples}, training=False
            )
            imgs_processed = processed["inputs"]
            data_samples = processed["data_samples"]

            # ---- Build model inputs ----
            if use_drone and "drone_images" in batch:
                model_inputs = {
                    "sat": imgs_processed,
                    "drone_images": batch["drone_images"].to(device),
                    "intrinsics": batch["intrinsics"].to(device),
                    "extrinsics": batch["extrinsics"].to(device),
                }
            else:
                model_inputs = imgs_processed

            # ---- Run prediction (same as runner.test → model.predict) ----
            outputs = model_base.predict(model_inputs, data_samples, rescale=False)

            # ---- Collect GT and DT for COCO evaluation ----
            for j, output in enumerate(outputs):
                meta = img_metas[j]
                img_shape = meta["img_shape"]

                # GT - from data_samples (DetDataSample), not from output (InstanceData)
                gt_inst = data_samples[j].gt_instances
                gt_np = _extract_instances_numpy(gt_inst, img_shape)
                all_gt.append(gt_np)

                # Predictions - from output (InstanceData)
                pred_inst = output
                pred_np = _extract_instances_numpy(pred_inst, img_shape)
                all_dt.append(pred_np)

                all_img_metas.append(meta)

            # ---- Compute loss (additional metric) ----
            if batch_has_valid_gt:
                loss_dict = model_base.loss(model_inputs, data_samples)
                loss_parts = []
                for k, v in loss_dict.items():
                    if "loss" not in k.lower():
                        continue
                    if isinstance(v, torch.Tensor):
                        loss_parts.append(v)
                    elif isinstance(v, (list, tuple)):
                        loss_parts.extend(x for x in v if isinstance(x, torch.Tensor))
                loss = sum(loss_parts) if loss_parts else torch.tensor(0.0)
                total_loss += float(loss.detach().item())
                num_loss_batches += 1
                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        loss_meter[k] = loss_meter.get(k, 0.0) + float(v.detach().item())
                    elif isinstance(v, (list, tuple)):
                        val = sum(float(x.detach().item()) for x in v if isinstance(x, torch.Tensor))
                        loss_meter[k] = loss_meter.get(k, 0.0) + val

            # ---- Visualization ----
            if args.show_dir and visualized_count < args.max_num:
                for j, output in enumerate(outputs):
                    if visualized_count >= args.max_num:
                        break
                    img_path = img_metas[j].get("img_path", img_metas[j].get("filename", ""))
                    if not img_path:
                        continue
                    # GT for visualization (from raw batch, un-preprocessed)
                    gt_vis = None
                    if "gt_bboxes" in batch and j < len(batch["gt_bboxes"]):
                        gt_vis = InstanceData()
                        curr_labels = batch["gt_labels"][j]
                        valid = curr_labels >= 0 if isinstance(curr_labels, torch.Tensor) else np.array(curr_labels) >= 0
                        gt_vis.bboxes = batch["gt_bboxes"][j][valid]
                        gt_vis.labels = batch["gt_labels"][j][valid]
                        if "gt_masks" in batch:
                            idx = valid.cpu().numpy() if isinstance(valid, torch.Tensor) else valid
                            gt_vis.masks = batch["gt_masks"][j][idx]
                    out_file = osp.join(args.show_dir, osp.basename(img_path))
                    visualize_comparison(
                        img_path, output, gt_vis, out_file,
                        score_thr=args.show_score_thr,
                        target_size=img_metas[j].get("img_shape", None)[:2]
                    )
                    visualized_count += 1

    # ---- COCO-style evaluation ----
    logger.info("Computing COCO metrics")

    metrics: Dict[str, float] = {}
    total_gt = sum(len(g["labels"]) for g in all_gt)
    total_dt = sum(len(d["scores"]) for d in all_dt if "scores" in d)
    logger.info("Total GT instances: %d", total_gt)
    logger.info("Total predicted instances: %d", total_dt)

    if total_gt > 0:
        coco_gt, coco_dt = build_coco_gt_and_dt(all_gt, all_dt, all_img_metas)

        bbox_metrics = run_coco_eval(coco_gt, coco_dt, iou_type="bbox")
        metrics.update(bbox_metrics)

        segm_metrics = run_coco_eval(coco_gt, coco_dt, iou_type="segm")
        metrics.update(segm_metrics)

        # F-0: maxDets diagnostic
        segm_metrics_300 = run_coco_eval(coco_gt, coco_dt, iou_type="segm", max_dets=[1, 10, 100, 300])
        ar_small_100 = segm_metrics.get("segm/AR_s", 0.0)
        ar_small_300 = segm_metrics_300.get("segm/AR_s", 0.0)
        logger.info(
            "[maxDets diagnostic] segm AR_small: maxDets=100 → %.4f, maxDets=300 → %.4f, Δ=%.4f",
            ar_small_100, ar_small_300, ar_small_300 - ar_small_100,
        )
    else:
        logger.warning("No GT instances found. Skipping COCO evaluation.")

    # ---- Loss metrics ----
    if num_loss_batches > 0:
        avg_loss = total_loss / num_loss_batches
        metrics["loss/avg_total"] = avg_loss
        logger.info("Average total loss: %.6f (%d batches)", avg_loss, num_loss_batches)
        for k, v in loss_meter.items():
            avg_k = v / num_loss_batches
            metrics[f"loss/{k}"] = avg_k
    else:
        logger.warning("No batches with valid GT for loss computation.")

    if args.work_dir:
        with open(osp.join(args.work_dir, "eval_results.json"), "w") as f:
            json.dump(metrics, f, indent=4)

    _log_inference_summary(
        test_dataset,
        args,
        sat_data_root,
        use_drone,
        drone_data_root,
        metrics,
        visualized_count,
    )


if __name__ == "__main__":
    main()
