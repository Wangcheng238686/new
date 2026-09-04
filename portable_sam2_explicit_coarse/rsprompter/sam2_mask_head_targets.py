"""GT targets and coarse/final-mask losses of the SAM2 mask head, as a plain mixin (split from models_sam2; no super(), no attributes)."""

"""
SAM2 Adapter for Portable SAM Fusion - Simplified Version
直接使用SAM2的build_sam2函数加载模型
"""

import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine import ConfigDict
from mmengine.dist import is_main_process
from mmengine.model import BaseModule
from torch import Tensor

from mmcv.ops import RoIAlign
from mmdet.models import MaskRCNN, StandardRoIHead
from mmdet.models.roi_heads.mask_heads import FCNMaskHead
from mmdet.models.task_modules import SamplingResult
from mmdet.models.utils import empty_instances, unpack_gt_instances
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample, SampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.structures.mask.mask_target import mask_target
from mmdet.utils import ConfigType, InstanceList, OptConfigType

from .ckpt_utils import load_module_state_dict_strict



class _MaskHeadTargetsMixin:
    def get_targets(
        self,
        sampling_results: List[SamplingResult],
        batch_gt_instances: InstanceList,
        rcnn_train_cfg: ConfigDict,
        pos_priors_override: Optional[List[Tensor]] = None,
    ) -> Tensor:
        """Build ROI-local final-mask targets for historical B0/B1 reproduction.

        This legacy adapter deliberately interprets each decoder output as an
        ROI-local map, crops the assigned GT by the proposal and uses standard
        MMDetection bbox paste at prediction time.
        """
        pos_proposals = (
            pos_priors_override
            if pos_priors_override is not None
            else [res.pos_priors for res in sampling_results]
        )
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]
        gt_masks = [res.masks for res in batch_gt_instances]
        return mask_target(
            pos_proposals,
            pos_assigned_gt_inds,
            gt_masks,
            rcnn_train_cfg,
        )

    def get_full_image_targets(
        self,
        sampling_results: List[SamplingResult],
        batch_gt_instances: InstanceList,
        target_size: Tuple[int, int],
        device: torch.device,
    ) -> Tensor:
        """Build one native full-image target for every positive proposal."""
        targets = []
        target_h, target_w = int(target_size[0]), int(target_size[1])
        for result, gt_instances in zip(sampling_results, batch_gt_instances):
            assigned = result.pos_assigned_gt_inds
            if assigned.numel() == 0:
                continue
            gt_masks = gt_instances.masks
            unique_ids, inverse = torch.unique(
                assigned.detach().cpu(), return_inverse=True
            )
            unique_masks = gt_masks[unique_ids].to_tensor(
                dtype=torch.float32, device=device
            )
            selected = unique_masks[inverse.to(device=unique_masks.device)]
            selected = F.interpolate(
                selected[:, None],
                size=(target_h, target_w),
                mode="nearest",
            )[:, 0]
            targets.append(selected)
        if not targets:
            return torch.zeros(
                (0, target_h, target_w), device=device, dtype=torch.float32
            )
        return torch.cat(targets, dim=0)

    def get_final_mask_roi_support(
        self,
        sampling_results: List[SamplingResult],
        batch_gt_instances: InstanceList,
        target_size: Tuple[int, int],
        device: torch.device,
    ) -> Tensor:
        """Rasterize expanded positive proposal boxes on the final-mask grid.

        C2-L retains SAM2's full-image coordinate contract.  These proposal
        supports affect only loss weighting; they never crop or paste decoder
        predictions and are not needed by validation/inference.
        """
        target_h, target_w = int(target_size[0]), int(target_size[1])
        expand_ratio = float(self.final_mask_loss_cfg.roi_expand_ratio)
        supports = []
        for result, gt_instances in zip(sampling_results, batch_gt_instances):
            priors = getattr(result.pos_priors, "tensor", result.pos_priors)
            if priors.numel() == 0:
                continue
            full_h = int(gt_instances.masks.height)
            full_w = int(gt_instances.masks.width)
            scale_x = float(target_w) / float(full_w)
            scale_y = float(target_h) / float(full_h)
            for prior in priors.detach():
                x1, y1, x2, y2 = [float(value) for value in prior[:4].cpu()]
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                half_w = max(0.5, 0.5 * (x2 - x1) * expand_ratio)
                half_h = max(0.5, 0.5 * (y2 - y1) * expand_ratio)
                gx1 = max(0, min(target_w - 1, math.floor((cx - half_w) * scale_x)))
                gy1 = max(0, min(target_h - 1, math.floor((cy - half_h) * scale_y)))
                gx2 = max(gx1 + 1, min(target_w, math.ceil((cx + half_w) * scale_x)))
                gy2 = max(gy1 + 1, min(target_h, math.ceil((cy + half_h) * scale_y)))
                support = torch.zeros(
                    (target_h, target_w), device=device, dtype=torch.bool
                )
                support[gy1:gy2, gx1:gx2] = True
                supports.append(support)
        if not supports:
            return torch.zeros(
                (0, target_h, target_w), device=device, dtype=torch.bool
            )
        return torch.stack(supports, dim=0)

    @staticmethod
    def roi_balanced_final_mask_loss(
        logits: Tensor,
        targets: Tensor,
        roi_support: Tensor,
        cfg: ConfigDict,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Balanced ROI BCE + ROI Dice + weak outside-background BCE."""
        if logits.shape != targets.shape or targets.shape != roi_support.shape:
            raise ValueError(
                "ROI-focused final-mask tensors must have identical [N,H,W] shapes: "
                f"logits={tuple(logits.shape)} targets={tuple(targets.shape)} "
                f"support={tuple(roi_support.shape)}"
            )
        eps = float(cfg.eps)
        target = targets.float().clamp(0, 1)
        inside = roi_support.to(dtype=logits.dtype)
        outside = 1.0 - inside
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        reduce_dims = (1, 2)

        positive = inside * target
        negative = inside * (1.0 - target)
        positive_count = positive.sum(dim=reduce_dims)
        negative_count = negative.sum(dim=reduce_dims)
        positive_bce = (bce * positive).sum(dim=reduce_dims) / positive_count.clamp_min(1.0)
        negative_bce = (bce * negative).sum(dim=reduce_dims) / negative_count.clamp_min(1.0)
        has_positive = positive_count > 0
        has_negative = negative_count > 0
        both = has_positive & has_negative
        roi_bce_per_roi = torch.where(
            both,
            0.5 * (positive_bce + negative_bce),
            torch.where(has_positive, positive_bce, negative_bce),
        )
        roi_bce = roi_bce_per_roi.mean()

        probability = logits.sigmoid() * inside
        target_inside = target * inside
        intersection = (probability * target_inside).sum(dim=reduce_dims)
        denominator = probability.sum(dim=reduce_dims) + target_inside.sum(dim=reduce_dims)
        roi_dice = (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()

        outside_count = outside.sum(dim=reduce_dims)
        outside_bce_per_roi = (bce * outside).sum(dim=reduce_dims) / outside_count.clamp_min(1.0)
        outside_bce_per_roi = torch.where(
            outside_count > 0,
            outside_bce_per_roi,
            torch.zeros_like(outside_bce_per_roi),
        )
        outside_bce = outside_bce_per_roi.mean()
        total = (
            float(cfg.roi_bce_weight) * roi_bce
            + float(cfg.roi_dice_weight) * roi_dice
            + float(cfg.outside_bce_weight) * outside_bce
        )
        stats = {
            "debug_final_mask_roi_bce": roi_bce.detach(),
            "debug_final_mask_roi_dice": roi_dice.detach(),
            "debug_final_mask_outside_bce": outside_bce.detach(),
            "debug_final_mask_roi_coverage": inside.detach().mean(),
            "debug_final_mask_roi_target_fill": (
                target_inside.sum() / inside.sum().clamp_min(1.0)
            ).detach(),
        }
        return total, stats

    def get_coarse_targets(
        self,
        sampling_results: List[SamplingResult],
        batch_gt_instances: InstanceList,
        mask_size: Optional[Tuple[int, int]] = None,
        prompt_pos_priors: Optional[List[Tensor]] = None,
    ) -> Tensor:
        """coarse mask（ShapePriorInjector 输出）的 ROI-local 监督 target。

        coarse mask 在 ROI 局部坐标系生成与监督。本方法按 positive proposal box
        crop GT mask 并 resize 到 [M,H,W]（pred 原生分辨率），供
        CoarseMaskLoss 使用。旧 B0/B1 的最终 SAM2 mask 仍由 MMDetection
        ``mask_target`` 独立生成；普通显式 coarse 路线使用 full-image target；
        C2-R 则复用本方法，在 MaskDecoder 原生 ROI 网格上生成 final-mask target。

        性能优化：先对去重后的 GT mask 调一次 to_tensor（避免重复渲染整图 mask），
        再逐 ROI crop+resize（crop/resize 本身轻量，瓶颈在 to_tensor 的整图渲染）。
        """
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]
        if prompt_pos_priors is not None:
            pos_priors_list = prompt_pos_priors
        else:
            pos_priors_list = [res.pos_priors for res in sampling_results]
        gt_masks = [res.masks for res in batch_gt_instances]
        device = pos_priors_list[0].device if pos_priors_list else torch.device("cpu")
        if mask_size is None:
            roi_h = roi_w = self.roi_mask_size
        else:
            roi_h, roi_w = int(mask_size[0]), int(mask_size[1])
        coarse_targets_list = []
        for pos_priors, pos_gt_inds, gt_mask in zip(
            pos_priors_list, pos_assigned_gt_inds, gt_masks
        ):
            if len(pos_gt_inds) == 0:
                coarse_targets_list.append(
                    torch.zeros((0, roi_h, roi_w), device=device, dtype=torch.float32)
                )
                continue
            # 性能优化：去重 GT 索引，只渲染唯一的整图 mask 一次（避免重复 to_tensor）
            pos_gt_inds_cpu = pos_gt_inds.cpu()
            unique_gt_inds, inverse_idx = torch.unique(
                pos_gt_inds_cpu, return_inverse=True
            )
            # 只对唯一 GT 调一次 to_tensor（整图渲染是大头开销）
            gt_mask_tensor_unique = gt_mask[unique_gt_inds].to_tensor(
                dtype=torch.float32, device=device
            )  # [num_unique_gt, H_full, W_full]
            # 按 inverse_idx 映射回每个 positive proposal（索引复用，无重复渲染）
            gt_mask_tensor = gt_mask_tensor_unique[inverse_idx]  # [num_pos, H, W]
            boxes = pos_priors
            cropped = []
            for i in range(gt_mask_tensor.shape[0]):
                # 审查第七节：用 floor(x1/y1)/ceil(x2/y2) 量化，避免 to(long) 截断丢边界。
                import math as _math
                _bx = boxes[i].cpu()
                x1f, y1f, x2f, y2f = _bx[0].item(), _bx[1].item(), _bx[2].item(), _bx[3].item()
                x1 = _math.floor(x1f)
                y1 = _math.floor(y1f)
                x2 = _math.ceil(x2f)
                y2 = _math.ceil(y2f)
                x1c = max(0, x1)
                y1c = max(0, y1)
                x2c = min(gt_mask_tensor.shape[-1], max(x1c + 1, x2))
                y2c = min(gt_mask_tensor.shape[-2], max(y1c + 1, y2))
                patch = gt_mask_tensor[i : i + 1, y1c:y2c, x1c:x2c]
                if patch.numel() == 0:
                    patch = torch.zeros(1, 1, device=device, dtype=torch.float32)
                patch = patch.unsqueeze(0)
                # Binary instance targets are resized with nearest-neighbor so
                # the coarse loss is supervised by a crisp mask at its native size.
                patch = F.interpolate(
                    patch, size=(roi_h, roi_w), mode="nearest"
                )
                patch = patch.squeeze(0).squeeze(0).clamp(0, 1)  # [H,W]
                cropped.append(patch)
            coarse_targets_list.append(torch.stack(cropped, dim=0))
        coarse_targets = torch.cat(coarse_targets_list)
        assert coarse_targets.dim() == 3, (
            f"coarse_targets must be [M,H,W], got {tuple(coarse_targets.shape)}"
        )
        return coarse_targets

    def coarse_mask_loss(
        self,
        shape_prior_mask_logits: Tensor,
        sampling_results: List[SamplingResult],
        batch_gt_instances: InstanceList,
        prompt_pos_priors: Optional[List[Tensor]] = None,
    ) -> Tuple[Tensor, Dict]:
        """计算 coarse mask 独立 loss（替换旧 _shape_prior_aux_loss）。

        在预测原生分辨率上计算（target 用 nearest 对齐到 pred 尺寸）。
        prompt_pos_priors 用于 box jitter 后的 target crop（见审查 2.7）。
        """
        target_size = shape_prior_mask_logits.shape[-2:]
        coarse_targets = self.get_coarse_targets(
            sampling_results, batch_gt_instances,
            mask_size=target_size, prompt_pos_priors=prompt_pos_priors,
        )
        gt_point_stats = self._shape_point_gt_stats(coarse_targets)
        if not hasattr(self, "_last_uav_debug_stats") or self._last_uav_debug_stats is None:
            self._last_uav_debug_stats = {}
        self._last_uav_debug_stats.update(gt_point_stats)
        return self.coarse_mask_loss_module(shape_prior_mask_logits, coarse_targets)

    def loss_and_target(
        self,
        mask_preds: Tensor,
        sampling_results: List[SamplingResult],
        batch_gt_instances: InstanceList,
        rcnn_train_cfg: ConfigDict,
        final_mask_pos_priors: Optional[List[Tensor]] = None,
    ) -> dict:
        if self.final_mask_coordinate_mode == "full_image":
            mask_targets = self.get_full_image_targets(
                sampling_results,
                batch_gt_instances,
                target_size=mask_preds.shape[-2:],
                device=mask_preds.device,
            )
        elif self.roi_sam_enabled:
            # ROI-SAM keeps the target on the decoder's native ROI grid instead
            # of collapsing the 128x128 prediction to the historical 28x28
            # Mask R-CNN target used by B0/B1.
            mask_targets = self.get_coarse_targets(
                sampling_results,
                batch_gt_instances,
                mask_size=mask_preds.shape[-2:],
                prompt_pos_priors=final_mask_pos_priors,
            )
        else:
            mask_targets = self.get_targets(
                sampling_results=sampling_results,
                batch_gt_instances=batch_gt_instances,
                rcnn_train_cfg=rcnn_train_cfg,
                pos_priors_override=final_mask_pos_priors,
            )
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        if mask_preds.shape[-2:] != mask_targets.shape[-2:]:
            mask_preds = F.interpolate(
                mask_preds,
                size=mask_targets.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        loss = dict()
        debug_stats = dict()
        if mask_preds.size(0) == 0:
            loss_mask = mask_preds.sum()
        else:
            if self.final_mask_loss_cfg.mode == "roi_balanced_dice":
                if self.class_agnostic:
                    selected_logits = mask_preds[:, 0]
                else:
                    row = torch.arange(mask_preds.shape[0], device=mask_preds.device)
                    selected_logits = mask_preds[row, pos_labels]
                roi_support = self.get_final_mask_roi_support(
                    sampling_results,
                    batch_gt_instances,
                    target_size=mask_targets.shape[-2:],
                    device=mask_preds.device,
                )
                loss_mask, focused_stats = self.roi_balanced_final_mask_loss(
                    selected_logits,
                    mask_targets,
                    roi_support,
                    self.final_mask_loss_cfg,
                )
                debug_stats.update(focused_stats)
            elif self.class_agnostic:
                loss_mask = self.loss_mask(mask_preds, mask_targets, torch.zeros_like(pos_labels))
            else:
                loss_mask = self.loss_mask(mask_preds, mask_targets, pos_labels)
            mask_probs = mask_preds.sigmoid().detach()
            debug_stats.update(
                debug_mask_pos_rois=mask_preds.new_tensor(float(mask_preds.size(0))),
                debug_mask_logit_mean=mask_preds.detach().mean(),
                debug_mask_logit_std=mask_preds.detach().std(unbiased=False),
                debug_mask_prob_mean=mask_probs.mean(),
                debug_mask_prob_ge05=(mask_probs >= 0.5).float().mean(),
                debug_mask_target_fill=mask_targets.detach().float().mean(),
            )
        loss["loss_mask"] = loss_mask
        return dict(
            loss_mask=loss,
            mask_targets=mask_targets,
            debug_stats=debug_stats,
        )

