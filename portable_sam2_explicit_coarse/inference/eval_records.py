"""Dataset contract resolution, loader construction, batch-to-sample
conversion, prediction records and the checkpoint summary report."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import cv2
import numpy as np
import torch
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
from mmdet.structures.mask import BitmapMasks
from torch.utils.data import DataLoader


logger = logging.getLogger("checkpoint_inference")


def _resolve_dataset_contract(
    args: argparse.Namespace,
    snapshot: Mapping[str, Any],
) -> Dict[str, Any]:
    data_config = snapshot.get("data_config", {})
    if not isinstance(data_config, Mapping):
        data_config = {}
    if data_config.get("dataset_format") == "ue_raw_surface":
        from ue.contract import validate_snapshot, raw_data_contract
        validate_snapshot(snapshot)
        if args.split not in {"validation", "test"}:
            raise ValueError("raw UE inference requires validation or test split")
        resolved = raw_data_contract(args.data_root or data_config["data_root"], data_config["split_manifest"])
        if args.image_size and list(args.image_size) != [1024, 1024]:
            raise ValueError("raw UE inference requires 1024 inputs")
        resolved.update(split="val" if args.split == "validation" else "test",
                        batch_size=args.batch_size or snapshot.get("training_args", {}).get("val_batch_size", 1))
        return resolved
    training_args = snapshot.get("training_args", {})
    if not isinstance(training_args, Mapping):
        training_args = {}

    defaults = {
        "validation": (
            "2.4 annotation/annotation/validation.json",
            "2.3 valid/validation",
        ),
        "test": (
            "2.4 annotation/annotation/test.json",
            "2.2 test/test",
        ),
    }
    saved_split = data_config.get(args.split, {})
    if not isinstance(saved_split, Mapping):
        saved_split = {}
    fallback_ann, fallback_images = defaults.get(args.split, ("", ""))

    data_root = (
        args.data_root
        or data_config.get("data_root")
        or training_args.get("data_root")
        or "/data/wangcheng/dataset/WHU"
    )
    ann_file = args.ann_file or saved_split.get("ann_file") or fallback_ann
    image_subdir = (
        args.image_subdir or saved_split.get("image_subdir") or fallback_images
    )
    if args.split == "custom" and (not ann_file or not image_subdir):
        raise ValueError("custom split requires --ann-file and --image-subdir")

    image_size = (
        args.image_size
        or data_config.get("image_size")
        or snapshot.get("image_size")
        or [1024, 1024]
    )
    if len(image_size) != 2:
        raise ValueError(f"image_size must contain two integers, got {image_size!r}")
    batch_size = args.batch_size or training_args.get("val_batch_size") or 1
    return {
        "data_root": str(data_root),
        "ann_file": str(ann_file),
        "image_subdir": str(image_subdir),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "batch_size": int(batch_size),
        "single_class": bool(data_config.get("single_class", True)),
        "num_classes": int(data_config.get("num_classes", 0) or 0),
    }


def _build_loader(contract: Mapping[str, Any], num_workers: int) -> DataLoader:
    from data.loader import rtmdet_collate_fn
    from data.whu_instance_dataset import WHUCocoInstanceDataset

    if contract.get("dataset_format") == "ue_raw_surface":
        from data.ue_raw_pair_dataset import UERawPairDataset
        dataset = UERawPairDataset(contract["data_root"], contract["split"],
                                  split_manifest=contract["split_manifest"])
        return DataLoader(dataset, batch_size=int(contract["batch_size"]), shuffle=False,
                          num_workers=int(num_workers), collate_fn=rtmdet_collate_fn,
                          pin_memory=torch.cuda.is_available(), persistent_workers=int(num_workers) > 0)
    single_class = bool(contract["single_class"])
    # Multi-class runs train with canonical ids 1..N -> labels 0..N-1; the
    # loader must apply the same mapping so GT labels match the head.
    num_classes = int(contract.get("num_classes", 0) or 0)
    if single_class:
        category_mapping = None
    elif num_classes > 0:
        category_mapping = {i: i - 1 for i in range(1, num_classes + 1)}
    else:
        raise RuntimeError(
            "Multi-class checkpoint without a stored num_classes cannot be "
            "resolved (the legacy iSAID 15-class fallback was removed); use a "
            "schema-v2 checkpoint that embeds data_config.num_classes."
        )
    dataset = WHUCocoInstanceDataset(
        data_root=contract["data_root"],
        ann_file=contract["ann_file"],
        image_subdir=contract["image_subdir"],
        image_size=tuple(contract["image_size"]),
        single_class=single_class,
        enable_category_mapping=not single_class,
        category_mapping=category_mapping,
        flip_prob=0.0,
        vflip_prob=0.0,
        gaussian_noise_prob=0.0,
        random_erasing_prob=0.0,
        multi_scale_resize_prob=0.0,
    )
    return DataLoader(
        dataset,
        batch_size=int(contract["batch_size"]),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=rtmdet_collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=int(num_workers) > 0,
    )


def _build_data_samples(
    batch: Mapping[str, Any],
    device: torch.device,
) -> Tuple[torch.Tensor, List[DetDataSample]]:
    imgs = batch["imgs"].to(device)
    img_metas = batch["img_metas"]
    gt_bboxes = [value.to(device) for value in batch["gt_bboxes"]]
    gt_labels = [value.to(device) for value in batch["gt_labels"]]
    gt_masks = batch["gt_masks"]
    data_samples: List[DetDataSample] = []

    for index in range(len(imgs)):
        sample = DetDataSample()
        sample.set_metainfo(img_metas[index])
        instances = InstanceData()
        valid = gt_labels[index] >= 0
        valid_indices = valid.nonzero().squeeze(-1).cpu()
        height, width = img_metas[index]["img_shape"][:2]
        if valid_indices.numel() > 0:
            instances.bboxes = gt_bboxes[index][valid_indices]
            instances.labels = gt_labels[index][valid_indices]
            masks = gt_masks[index][valid_indices.numpy()]
            instances.masks = BitmapMasks(masks, height, width)
        else:
            instances.bboxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
            instances.labels = torch.zeros((0,), dtype=torch.int64, device=device)
            instances.masks = BitmapMasks(
                np.zeros((0, height, width), dtype=np.uint8),
                height,
                width,
            )
        sample.gt_instances = instances
        data_samples.append(sample.to(device))
    return imgs, data_samples


def _extract_instances_numpy(instances: InstanceData, img_shape: Tuple[int, int]) -> dict:
    height, width = img_shape[:2]

    def as_numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return value

    bboxes = as_numpy(instances.bboxes)
    labels = as_numpy(instances.labels)
    scores = as_numpy(instances.scores) if hasattr(instances, "scores") else None
    mask_scores = (
        as_numpy(instances.mask_scores) if hasattr(instances, "mask_scores") else None
    )
    masks = getattr(instances, "masks", None)
    if hasattr(masks, "masks"):
        masks = masks.masks
    elif hasattr(masks, "to_ndarray"):
        masks = masks.to_ndarray()
    else:
        masks = as_numpy(masks)
    if masks is None or len(masks) == 0:
        masks = np.zeros((0, height, width), dtype=np.uint8)
    elif masks.shape[1:] != (height, width):
        masks = np.stack(
            [
                cv2.resize(
                    mask.astype(np.uint8),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                for mask in masks
            ],
            axis=0,
        )
    output = {"bboxes": bboxes, "labels": labels, "masks": masks}
    if scores is not None:
        output["scores"] = scores
    if mask_scores is not None:
        output["mask_scores"] = mask_scores
    return output


def _filter_predictions(predictions: dict, threshold: float) -> dict:
    if threshold <= 0 or "scores" not in predictions:
        return predictions
    keep = np.asarray(predictions["scores"]) >= float(threshold)
    return {
        key: value[keep] if isinstance(value, np.ndarray) else value
        for key, value in predictions.items()
    }


def _mask_to_rle(mask: np.ndarray) -> dict:
    from pycocotools import mask as mask_utils

    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    if isinstance(rle.get("counts"), bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    return rle


def _prediction_records(
    predictions: Mapping[str, Any],
    meta: Mapping[str, Any],
) -> Iterable[dict]:
    bboxes = predictions["bboxes"]
    labels = predictions["labels"]
    scores = predictions.get("scores", np.ones(len(bboxes), dtype=np.float32))
    mask_scores = predictions.get("mask_scores", scores)
    masks = predictions["masks"]
    for index in range(len(bboxes)):
        x1, y1, x2, y2 = [float(value) for value in bboxes[index]]
        yield {
            "image_id": int(meta.get("image_id", meta.get("scene_id", 0))),
            "category_id": int(labels[index]) + 1,
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "score": float(scores[index]),
            "mask_score": float(mask_scores[index]),
            "segmentation": _mask_to_rle(masks[index]),
            "file_name": str(meta.get("filename", "")),
        }


def _checkpoint_report(
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_schema_version": int(
            checkpoint.get(
                "checkpoint_schema_version",
                snapshot.get("checkpoint_schema_version", 0),
            )
        ),
        "epoch": checkpoint.get("epoch"),
        "best_metrics": checkpoint.get("best_metrics", {}),
        "has_model_config": bool(snapshot.get("model_config")),
        "has_training_args": bool(snapshot.get("training_args")),
        "has_data_config": bool(snapshot.get("data_config")),
        "has_ema_state": bool(checkpoint.get("ema_state")),
        "config_path": snapshot.get("config_path"),
        "neck_type": snapshot.get("neck_type"),
        "explicit_prompt_mode": snapshot.get("explicit_prompt_mode"),
        "p2_boundary_refiner": snapshot.get("p2_boundary_refiner", {}),
        "architecture_contract": snapshot.get("architecture_contract", {}),
        "image_size": snapshot.get("image_size"),
    }

