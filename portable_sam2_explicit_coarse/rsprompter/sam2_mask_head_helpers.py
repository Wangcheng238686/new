"""Prompt-canvas / ROI-SAM / debug helper methods of the SAM2 mask head, as a plain mixin (split from models_sam2; no super(), no attributes)."""

"""
SAM2 Adapter for Portable SAM Fusion - Simplified Version
直接使用SAM2的build_sam2函数加载模型
"""

import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from mmcv.ops import RoIAlign




class _MaskHeadPromptHelpers:
    def _shape_prior_to_prompt_mask(
        self,
        roi_mask_logits: Tensor,
        boxes: Tensor,
    ):
        """批次2 子任务3：ROI-local coarse logits → full-image PE mask grid。

        处理顺序：ROI-local transform → resize 到 bbox 对应 prompt-grid → paste 到
        full-image canvas → bbox 外 outside_fill_logit → 最终 clamp（若非 None）。
        """
        from .dense_prompt_utils import (
            gaussian_prompt_from_roi_coarse,
            paste_roi_to_full_canvas,
            transform_coarse_prompt,
        )
        mask_h, mask_w = self._pe_mask_input_size()
        transform = str(self.dense_prompt_cfg.get("transform", "raw_logits"))
        if transform == "gaussian_edt":
            return gaussian_prompt_from_roi_coarse(
                roi_mask_logits,
                boxes,
                mask_h=mask_h,
                mask_w=mask_w,
                image_size=(
                    self.prompt_encoder_image_size,
                    self.prompt_encoder_image_size,
                ),
                foreground_threshold=float(
                    self.dense_prompt_cfg.get("foreground_threshold", 0.5)
                ),
                omega=float(self.dense_prompt_cfg.get("gaussian_omega", 15.0)),
                gamma=float(self.dense_prompt_cfg.get("gaussian_gamma", 4.0)),
            )
        # 1. ROI-local transform（在 resize/paste 之前完成）
        transformed = transform_coarse_prompt(
            roi_mask_logits,
            transform=transform,
            gamma=self.dense_prompt_cfg.get("gamma", 1.0),
            strength=self.dense_prompt_cfg.get("strength", 1.0),
            clamp_range=self.dense_prompt_cfg.get("clamp_range", None),
            temperature=self.dense_prompt_cfg.get("temperature", 1.0),
            detach_input=bool(self.dense_prompt_cfg.get("detach_input", False)),
        )
        # 2. paste 到 full-image canvas（PE mask 尺寸从 PE 属性读，不硬编码）
        prompt = paste_roi_to_full_canvas(
            transformed, boxes, mask_h, mask_w,
            image_size=(self.prompt_encoder_image_size, self.prompt_encoder_image_size),
            outside_fill_logit=float(self.dense_prompt_cfg.get("outside_fill_logit", 0.0)),
        )
        valid = torch.ones(
            roi_mask_logits.shape[0],
            device=roi_mask_logits.device,
            dtype=torch.bool,
        )
        return prompt, valid, {}

    def _select_explicit_prompt_inputs(
        self,
        coords: Tensor,
        labels: Tensor,
        boxes: Tensor,
        masks: Optional[Tensor],
    ):
        """Select SAM2 PromptEncoder inputs for the explicit-mask ablation.

        The detector boxes remain the internal geometry for ROI extraction and
        coordinate mapping in every mode.  This method controls only which
        prompt modalities are exposed to the frozen PromptEncoder.
        """
        if self.prompt_sparse_mode != "shape_point":
            raise RuntimeError(
                "_select_explicit_prompt_inputs is only valid for shape_point"
            )
        # A warm-up/all-invalid batch must be a true no-point prompt. Passing
        # four label=-1 entries would create four learned not-a-point tokens.
        points = (
            None
            if not self.explicit_use_point_prompt or bool((labels < 0).all())
            else (coords, labels)
        )
        boxes_for_pe = boxes if self.explicit_use_box_prompt else None
        masks_for_pe = masks if self.explicit_use_dense_prompt else None
        return points, boxes_for_pe, masks_for_pe

    @staticmethod
    def _neutralize_invalid_point_tokens(
        sparse_embeddings: Tensor, labels: Tensor
    ) -> Tensor:
        """Zero invalid point slots while preserving any following box tokens."""
        point_count = int(labels.shape[1])
        point_tokens = sparse_embeddings[:, :point_count]
        valid = (labels >= 0).unsqueeze(-1).to(dtype=point_tokens.dtype)
        point_tokens = point_tokens * valid
        return torch.cat(
            [point_tokens, sparse_embeddings[:, point_count:]], dim=1
        )

    def _effective_shape_dense_alpha(self):
        """批次2 子任务4：返回 shape dense gate 的有效值（三模式统一）。

        - legacy_unbounded：self.shape_prompt_scale（可训练无界，旧实现）
        - fixed：self.shape_dense_alpha（buffer，不训练）
        - global_sigmoid：sigmoid(self.shape_dense_alpha_raw) ∈ (0,1)
        - use_shape_dense=False：返回 None（forward 里据此跳过 dense 混合）
        """
        if not self.use_shape_dense:
            return None
        if self.shape_scale_mode == "legacy_unbounded":
            return self.shape_prompt_scale
        elif self.shape_scale_mode == "fixed":
            return self.shape_dense_alpha
        elif self.shape_scale_mode == "global_sigmoid":
            return torch.sigmoid(self.shape_dense_alpha_raw)
        return None

    def _dense_embedding_box_support(
        self, boxes: Tensor, spatial_size: Tuple[int, int], dtype: torch.dtype
    ) -> Tensor:
        """Restrict dense-prompt embedding changes to the proposal box."""
        height, width = int(spatial_size[0]), int(spatial_size[1])
        yy = (
            torch.arange(height, device=boxes.device, dtype=boxes.dtype) + 0.5
        ) * (float(self.prompt_encoder_image_size) / float(height))
        xx = (
            torch.arange(width, device=boxes.device, dtype=boxes.dtype) + 0.5
        ) * (float(self.prompt_encoder_image_size) / float(width))
        x1, y1, x2, y2 = boxes.detach().unbind(dim=1)
        support = (
            (xx[None, None, None, :] >= x1[:, None, None, None])
            & (xx[None, None, None, :] < x2[:, None, None, None])
            & (yy[None, None, :, None] >= y1[:, None, None, None])
            & (yy[None, None, :, None] < y2[:, None, None, None])
        )
        return support.to(dtype=dtype)

    def _pe_mask_input_size(self) -> Tuple[int, int]:
        """批次2 子任务3：读取 PromptEncoder 的 mask 输入尺寸（不硬编码）。

        优先 self.prompt_encoder.mask_input_size（SAM2 PE 属性）；
        若 PE 不存在则 fallback 到 prompt_encoder_embed_size*4（旧逻辑）。
        不假设正方形（返回 (H,W)）。
        """
        pe = getattr(self, "prompt_encoder", None)
        if pe is not None and hasattr(pe, "mask_input_size"):
            mis = pe.mask_input_size
            return (int(mis[0]), int(mis[1]))
        es = self.prompt_encoder_embed_size
        return (es * 4, es * 4)

    @staticmethod
    def _debug_delta_ratio(after: Optional[Tensor], before: Optional[Tensor]) -> Optional[float]:
        if after is None or before is None:
            return None
        with torch.no_grad():
            denom = before.detach().float().norm().clamp_min(1e-6)
            ratio = (after.detach().float() - before.detach().float()).norm() / denom
            return float(ratio.item())

    def set_current_epoch(self, epoch_number: int) -> None:
        """Synchronize epoch-dependent point/loss schedules (1-based)."""
        self.current_epoch = max(1, int(epoch_number))
        if self.shape_point_miner is not None:
            self.shape_point_miner.set_current_epoch(self.current_epoch)
        if self.coarse_mask_loss_module is not None and hasattr(
            self.coarse_mask_loss_module, "set_current_epoch"
        ):
            self.coarse_mask_loss_module.set_current_epoch(self.current_epoch)

    def _shape_point_gt_stats(self, coarse_targets: Tensor) -> Dict[str, Tensor]:
        local_yx = self._last_shape_point_local_yx
        labels = self._last_shape_point_labels
        if local_yx is None or labels is None or coarse_targets.shape[0] == 0:
            zero = coarse_targets.new_zeros(())
            return {
                "SP/p1_in_gt_fg_count": zero, "SP/p1_gt_valid_count": zero,
                "SP/p2_in_gt_fg_count": zero, "SP/p2_gt_valid_count": zero,
                "SP/n1_in_gt_bg_count": zero, "SP/n1_gt_valid_count": zero,
                "SP/n2_in_gt_bg_count": zero, "SP/n2_gt_valid_count": zero,
            }
        if local_yx.shape[0] != coarse_targets.shape[0]:
            raise RuntimeError(
                "Shape-point/target ROI count mismatch: "
                f"{local_yx.shape[0]} vs {coarse_targets.shape[0]}"
            )
        h, w = coarse_targets.shape[-2:]
        y = local_yx[..., 0].clamp(0, h - 1)
        x = local_yx[..., 1].clamp(0, w - 1)
        batch = torch.arange(coarse_targets.shape[0], device=coarse_targets.device)[:, None]
        sampled = coarse_targets[batch, y, x] >= 0.5
        out: Dict[str, Tensor] = {}
        for slot, name, positive in (
            (0, "p1", True), (1, "p2", True),
            (2, "n1", False), (3, "n2", False),
        ):
            valid = labels[:, slot] >= 0
            correct = sampled[:, slot] if positive else ~sampled[:, slot]
            suffix = "in_gt_fg" if positive else "in_gt_bg"
            out[f"SP/{name}_{suffix}_count"] = (correct & valid).sum().detach()
            out[f"SP/{name}_gt_valid_count"] = valid.sum().detach()
        return out

    def get_uav_debug_stats(self) -> Dict[str, float]:
        # UAV branch removed (doc §1). This stub is kept for backward
        # compatibility with the legacy trainer logger call sites; it now
        # surfaces the GT-point diagnostics previously written here.
        return dict(getattr(self, "_last_uav_debug_stats", {}) or {})

    def _roi_sam_local_boxes(self, boxes: Tensor) -> Tensor:
        """Return canonical full-canvas boxes for ROI-local PromptEncoder input."""
        local_boxes = torch.zeros_like(boxes)
        local_boxes[:, 2:] = float(self.prompt_encoder_image_size)
        return local_boxes

    def _roi_sam_crop_feature(
        self,
        feature: Tensor,
        boxes: Tensor,
        roi_img_ids: Tensor,
    ) -> Tensor:
        """Normalize each proposal crop back to one native SAM2 feature canvas."""
        if feature.dim() != 4:
            raise ValueError(f"ROI-SAM feature must be BCHW, got {tuple(feature.shape)}")
        if boxes.shape[0] != roi_img_ids.shape[0]:
            raise ValueError("ROI-SAM boxes and roi_img_ids must have equal length")
        if boxes.numel() == 0:
            return feature.new_empty((0, feature.shape[1], *feature.shape[-2:]))
        if int(roi_img_ids.min()) < 0 or int(roi_img_ids.max()) >= feature.shape[0]:
            raise ValueError("ROI-SAM roi_img_ids contain an invalid image index")
        height, width = feature.shape[-2:]
        scale_x = float(width) / float(self.prompt_encoder_image_size)
        scale_y = float(height) / float(self.prompt_encoder_image_size)
        if not math.isclose(scale_x, scale_y, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(
                "ROI-SAM currently requires square images/features with one spatial scale"
            )
        rois = torch.cat(
            [roi_img_ids.to(device=boxes.device, dtype=boxes.dtype)[:, None], boxes],
            dim=1,
        )
        align = RoIAlign(
            output_size=(height, width),
            spatial_scale=scale_x,
            sampling_ratio=int(self.roi_sam_cfg.sampling_ratio),
            pool_mode="avg",
            aligned=bool(self.roi_sam_cfg.aligned),
        )
        return align(feature, rois)

