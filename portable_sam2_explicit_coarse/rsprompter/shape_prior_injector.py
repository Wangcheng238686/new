"""Coarse-mask prompt head: RoI feature -> SmallMaskDecoder -> 64x64 logits.

The a3sgm route is ``roi_only``: the RoI feature goes straight to the coarse
decoder with no context/FiLM modulation, so the coarse head is trained purely
by its auxiliary loss (points are mined under no_grad and the dense canvas is
detached at the paste site).
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SmallMaskDecoder(nn.Module):
    """Lightweight decoder: ``[N,C,Hroi,Wroi] -> [N,1,Hc,Wc]``."""

    def __init__(self, in_ch=512, mid_ch=128, out_ch=1, out_size=64):
        super().__init__()
        self.out_size = int(out_size)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1),
            nn.GroupNorm(32, mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1),
            nn.GroupNorm(32, mid_ch),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1),
            nn.GroupNorm(32, mid_ch),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(mid_ch, mid_ch // 2, 3, padding=1),
            nn.GroupNorm(16, mid_ch // 2),
            nn.GELU(),
            nn.Conv2d(mid_ch // 2, out_ch, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.net(x)
        if out.shape[-2:] != (self.out_size, self.out_size):
            out = F.interpolate(
                out,
                size=(self.out_size, self.out_size),
                mode="bilinear",
                align_corners=False,
            )
        return out


class ShapePriorInjector(nn.Module):
    """Generate an explicitly supervised ROI-local coarse-mask prompt.

    Only the ``roi_only`` fusion route survives: no context/box/FiLM
    parameters are constructed, so none leak into the optimizer or
    checkpoint. The remaining constructor parameters stay accepted because
    historical config dicts pass them.
    """

    def __init__(
        self,
        context_source="visual",
        d_inner=512,
        num_heads=8,
        num_context_tokens=64,
        roi_feat_channels=512,
        img_size=512,
        visual_context_dim=256,
        gate_init="zero",
        spatial_attn=True,
        coarse_mask_output_size=64,
        fusion_type="roi_only",
        context_gate_init=0.10,
        gamma_activation="tanh",
        beta_activation="tanh",
        zero_init_gamma=True,
        zero_init_beta=True,
    ):
        super().__init__()
        del context_source, gate_init
        self.context_source = "visual"
        self.num_context_tokens = int(num_context_tokens)
        self.d_inner = int(d_inner)
        if isinstance(img_size, (tuple, list)):
            self.img_size = (int(img_size[0]), int(img_size[1]))
        else:
            self.img_size = (int(img_size), int(img_size))
        self.spatial_attn = bool(spatial_attn)
        self.coarse_mask_output_size = int(coarse_mask_output_size)
        self.fusion_type = str(fusion_type)
        if self.fusion_type != "roi_only":
            raise ValueError(
                "fusion_type must be 'roi_only' (context/FiLM fusion arms were "
                f"removed), got {self.fusion_type!r}"
            )
        self._last_debug_stats: Dict[str, Tensor] = {}

        self.mask_decoder = SmallMaskDecoder(
            in_ch=roi_feat_channels,
            mid_ch=128,
            out_ch=1,
            out_size=self.coarse_mask_output_size,
        )

    def forward_context_visual(self, visual_context: Tensor) -> None:
        """roi_only route has no context pathway; kept for the caller contract."""
        del visual_context
        return None

    def forward_roi(
        self,
        ctx_tokens: Optional[Tensor],
        roi_feats: Tensor,
        roi_boxes: Tensor,
        roi_img_ids: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        del kwargs, ctx_tokens, roi_boxes, roi_img_ids
        self._last_debug_stats = {
            "CONTEXT/enabled": roi_feats.new_zeros(()),
        }
        mask_logits = self.mask_decoder(roi_feats)
        return mask_logits, None, None, None

    def get_debug_stats(self) -> Dict[str, Tensor]:
        return dict(self._last_debug_stats)
