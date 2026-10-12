"""Prompt-canvas and point-token helper methods of the SAM2 mask head.

Plain mixin (no super(), no module attributes): everything here operates on
state owned by RSPrompterAnchorMaskHeadSAM2.
"""
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from .dense_prompt_utils import paste_roi_to_full_canvas, transform_coarse_prompt


class _MaskHeadPromptHelpers:
    def _shape_prior_to_prompt_mask(
        self,
        roi_mask_logits: Tensor,
        boxes: Tensor,
    ):
        """ROI-local coarse logits -> full-image PE mask canvas (raw route).

        Order: ROI-local transform -> resize to the box's prompt-grid cells ->
        paste onto the canvas -> outside fill (-> clamp if configured). Under
        the sg contract the caller passes detach_input=True via config, which
        cuts the dense channel away from the coarse head's training signal.
        """
        mask_h, mask_w = self._pe_mask_input_size()
        transform = str(self.dense_prompt_cfg.get("transform", "raw_logits"))
        transformed = transform_coarse_prompt(
            roi_mask_logits,
            transform=transform,
            gamma=self.dense_prompt_cfg.get("gamma", 1.0),
            strength=self.dense_prompt_cfg.get("strength", 1.0),
            clamp_range=self.dense_prompt_cfg.get("clamp_range", None),
            temperature=self.dense_prompt_cfg.get("temperature", 1.0),
            detach_input=bool(self.dense_prompt_cfg.get("detach_input", False)),
        )
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
        """Pick which prompt modalities reach the frozen PromptEncoder.

        The detector boxes remain the internal geometry for ROI extraction and
        coordinate mapping in every mode; this controls only the modalities
        exposed to the PE (the paper's P/B/P+B/P+B+M ablation axis).
        """
        if self.prompt_sparse_mode != "shape_point":
            raise RuntimeError(
                "_select_explicit_prompt_inputs is only valid for shape_point"
            )
        # An all-invalid batch must be a true no-point prompt: passing four
        # label=-1 entries would create four learned not-a-point tokens.
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
        """The bounded shape-dense gate value, or None for sparse-only runs.

        global_sigmoid is the only surviving mode: sigmoid(alpha_raw) in (0,1)
        interpolates between the pretrained no-mask base and the shape dense
        embedding.
        """
        if not self.use_shape_dense:
            return None
        return torch.sigmoid(self.shape_dense_alpha_raw)

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
        """Read the PromptEncoder's mask-input size instead of hardcoding it.

        Prefers the SAM2 PE attribute; falls back to embed_size*4 when no PE
        exists. Does not assume square (returns (H, W)).
        """
        pe = getattr(self, "prompt_encoder", None)
        if pe is not None and hasattr(pe, "mask_input_size"):
            mis = pe.mask_input_size
            return (int(mis[0]), int(mis[1]))
        es = self.prompt_encoder_embed_size
        return (es * 4, es * 4)

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
        """GT-side point telemetry: do the mined points land in the true fg/bg?"""
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

    def get_forward_telemetry(self) -> Dict[str, float]:
        # Historical name: the UAV branch is gone, and this now surfaces the
        # forward telemetry bus (SP/DENSE/GATE/TAIL/PROMPT stats).
        return dict(getattr(self, "_forward_telemetry", {}) or {})
