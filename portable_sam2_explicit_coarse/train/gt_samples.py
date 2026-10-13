"""GT construction shared by train/val: instance extraction to numpy
and DetDataSample building, plus validation-loader sharding."""

import math
from typing import Any, List, Tuple

import cv2
import numpy as np
import torch
from mmdet.structures import DetDataSample
from mmdet.structures.mask import BitmapMasks
from mmengine.structures import InstanceData

def _extract_instances_numpy(
    instances, img_shape: Tuple[int, int], encode_masks: bool = False
) -> dict:
    """Extract InstanceData, optionally replacing dense masks with COCO RLE."""
    # Lazy import: the project-root sys.path bootstrap runs inside main(); this
    # module-level helper must not bind project packages at import time.
    from utils.coco_eval_utils import _masks_to_rles

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
            masks = masks.masks
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

    out = {"bboxes": bboxes, "labels": labels}
    if encode_masks:
        mask_count = len(masks)
        out["rles"] = _masks_to_rles(masks)
        out["mask_count"] = mask_count
        out["mask_fill_sum"] = (
            float(np.count_nonzero(masks)) / float(max(1, h * w))
            if mask_count > 0
            else 0.0
        )
    else:
        out["masks"] = masks
    if scores is not None:
        out["scores"] = scores
    if hasattr(instances, "mask_scores"):
        mask_scores = instances.mask_scores
        if isinstance(mask_scores, torch.Tensor):
            mask_scores = mask_scores.detach().cpu().numpy()
        out["mask_scores"] = mask_scores
    return out



def _build_gt_data_samples(
    imgs: torch.Tensor,
    img_metas: List[dict],
    gt_bboxes: List[torch.Tensor],
    gt_labels: List[torch.Tensor],
    gt_masks,
    device: torch.device,
) -> List[DetDataSample]:
    """Wrap a dataset batch's GT into DetDataSamples on ``device``.

    Shared by the training and validation loops (previously two
    semantically-identical inline blocks).  Padding labels are dropped;
    empty images get empty GT.
    """
    data_samples: List[DetDataSample] = []
    for i in range(len(imgs)):
        ds = DetDataSample()
        ds.set_metainfo(img_metas[i])
        gt_instances = InstanceData()
        num_inst = len(gt_labels[i])
        if num_inst > 0:
            valid_mask = gt_labels[i] >= 0
            num_valid = int(valid_mask.sum().item())
            if num_valid > 0:
                valid_indices = (
                    valid_mask.nonzero().squeeze(-1)[:num_valid].cpu()
                )
                gt_instances.bboxes = gt_bboxes[i][valid_indices]
                gt_instances.labels = gt_labels[i][valid_indices]
                masks = gt_masks[i][valid_indices.cpu().numpy()]
                gt_instances.masks = BitmapMasks(
                    masks, *img_metas[i]["img_shape"]
                )
            else:
                gt_instances.bboxes = torch.zeros(
                    (0, 4), dtype=torch.float32, device=device
                )
                gt_instances.labels = torch.zeros(
                    (0,), dtype=torch.int64, device=device
                )
                gt_instances.masks = BitmapMasks(
                    np.zeros((0, *img_metas[i]["img_shape"]), dtype=np.uint8),
                    *img_metas[i]["img_shape"],
                )
        else:
            gt_instances.bboxes = torch.zeros(
                (0, 4), dtype=torch.float32, device=device
            )
            gt_instances.labels = torch.zeros(
                (0,), dtype=torch.int64, device=device
            )
            gt_instances.masks = BitmapMasks(
                np.zeros((0, *img_metas[i]["img_shape"]), dtype=np.uint8),
                *img_metas[i]["img_shape"],
            )
        ds.gt_instances = gt_instances
        data_samples.append(ds.to(device))
    return data_samples

def _shard_val_loader(val_loader, rank: int, world_size: int) -> Tuple[Any, int]:
    """Rebuild the validation loader so each rank owns batches rank, rank+W, ...

    The val dataset pipeline is deterministic (no flip/noise/erasing/multi-scale)
    and the loader never shuffles, so batching the same Subset indices with the
    same batch_size/collate reproduces the exact per-batch composition of the
    unsharded loader: per-image tensors and padding are unchanged and only the
    ownership of batches differs.  Returns (shard_loader, total_images).
    """
    dataset = val_loader.dataset
    batch_size = val_loader.batch_size or 1
    total = len(dataset)
    num_batches = math.ceil(total / batch_size)
    owned_indices: List[int] = []
    for b in range(rank, num_batches, world_size):
        owned_indices.extend(range(b * batch_size, min((b + 1) * batch_size, total)))
    from torch.utils.data import DataLoader, Subset

    shard = DataLoader(
        Subset(dataset, owned_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=val_loader.num_workers,
        collate_fn=val_loader.collate_fn,
        pin_memory=True,
        drop_last=False,
        persistent_workers=val_loader.num_workers > 0 and len(owned_indices) > 0,
    )
    return shard, total
