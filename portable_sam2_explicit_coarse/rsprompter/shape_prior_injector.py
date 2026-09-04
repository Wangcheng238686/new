"""Explicit ROI-local coarse-mask prompt: context fusion + SmallMaskDecoder.

Split out of the original shape_prior.py (facade kept for import compat)."""

"""Explicit ROI-local coarse-mask prompt generation and point mining.

Canonical shape-point route:
- ROI feature -> lightweight coarse-mask decoder
- optional global SAM2 context + box-conditioned Spatial FiLM ablation
- ROI-local coarse mask
- adaptive hard 2P2N point mining (stop-gradient)

Only PyTorch is required.
"""
import logging
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


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


class RoIBoxEncoding(nn.Module):
    """Encode normalized ROI geometry with Fourier features."""

    def __init__(self, out_dim=512, n_freq=16):
        super().__init__()
        self.n_freq = int(n_freq)
        self.in_dim = 6
        self.register_buffer("freqs", torch.pow(2.0, torch.arange(n_freq).float()))
        self.proj = nn.Sequential(
            nn.Linear(self.in_dim * (1 + 2 * n_freq), out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, boxes: Tensor, img_size: Tuple[int, int]) -> Tensor:
        h_img, w_img = float(img_size[0]), float(img_size[1])
        x1, y1, x2, y2 = boxes.unbind(-1)
        cx = ((x1 + x2) * 0.5) / w_img
        cy = ((y1 + y2) * 0.5) / h_img
        width = (x2 - x1).clamp(min=1.0) / w_img
        height = (y2 - y1).clamp(min=1.0) / h_img
        area_ratio = (width * height).clamp(min=1e-6)
        aspect = (width / height.clamp(min=1e-6)).clamp(min=1e-6)
        features = torch.stack(
            [
                cx,
                cy,
                torch.log(width + 1e-6),
                torch.log(height + 1e-6),
                torch.log(aspect),
                torch.log(area_ratio),
            ],
            dim=-1,
        )
        expanded = features.unsqueeze(-1)
        encoded = torch.cat(
            [
                features,
                torch.sin(expanded * self.freqs).flatten(1),
                torch.cos(expanded * self.freqs).flatten(1),
            ],
            dim=-1,
        )
        return self.proj(encoded)


def _logit(value: float) -> float:
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


class ShapePriorInjector(nn.Module):
    """Generate an explicitly supervised ROI-local coarse-mask prompt.

    The default ``roi_only`` route sends the local RoI feature directly to the
    coarse decoder and does not construct any context/box/FiLM parameters.
    ``gated_spatial_film`` is retained as an explicit ablation::

        gamma = tanh(gamma_head(context))
        beta  = tanh(beta_head(context))
        gate  = sigmoid(context_gate_raw)
        x'    = x * (1 + gate * gamma) + gate * beta

    Gamma/beta heads are zero-initialized, so the initial forward is exactly the
    local ROI feature path.  ``legacy_multiplicative`` remains for ablations.
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
        self.gamma_activation = str(gamma_activation)
        self.beta_activation = str(beta_activation)
        self._last_debug_stats: Dict[str, Tensor] = {}

        if self.fusion_type not in {
            "roi_only",
            "legacy_multiplicative",
            "gated_spatial_film",
        }:
            raise ValueError(
                "fusion_type must be roi_only, legacy_multiplicative or "
                f"gated_spatial_film, got {self.fusion_type!r}"
            )
        if self.fusion_type == "gated_spatial_film" and not self.spatial_attn:
            raise ValueError("gated_spatial_film requires spatial_attn=True")

        if self.fusion_type == "roi_only":
            # A true no-FiLM route: do not leave inactive context parameters in
            # the optimizer/checkpoint and do not spend compute producing them.
            self.visual_context_proj = None
            self.context_grid = 0
            self.token_pool = None
            self.box_pe = None
            self.roi_proj = None
            self.visual_cross_attn = None
            self.cond_proj = None
            self.spatial_roi_proj = None
            self.spatial_cross_attn = None
            self.spatial_cond_proj = None
            self.gamma_head = None
            self.beta_head = None
            self.register_parameter("context_gate_raw", None)
        else:
            self.visual_context_proj = nn.Linear(int(visual_context_dim), d_inner)
            context_grid = int(round(math.sqrt(self.num_context_tokens)))
            if context_grid * context_grid != self.num_context_tokens:
                raise ValueError(
                    "num_context_tokens must be a perfect square for 2D pooling, "
                    f"got {self.num_context_tokens}"
                )
            self.context_grid = context_grid
            self.token_pool = nn.AdaptiveAvgPool2d(
                (self.context_grid, self.context_grid)
            )
            self.box_pe = RoIBoxEncoding(out_dim=d_inner)

        if self.fusion_type != "roi_only" and self.spatial_attn:
            self.roi_proj = None
            self.visual_cross_attn = None
            self.cond_proj = None
            self.spatial_roi_proj = nn.Conv2d(roi_feat_channels, d_inner, 1)
            self.spatial_cross_attn = nn.MultiheadAttention(
                d_inner, num_heads, batch_first=True
            )
            if self.fusion_type == "gated_spatial_film":
                self.spatial_cond_proj = None
                self.gamma_head = nn.Sequential(
                    nn.Conv2d(d_inner, d_inner // 2, 3, padding=1),
                    nn.GroupNorm(min(32, d_inner // 2), d_inner // 2),
                    nn.GELU(),
                    nn.Conv2d(d_inner // 2, roi_feat_channels, 1),
                )
                self.beta_head = nn.Sequential(
                    nn.Conv2d(d_inner, d_inner // 2, 3, padding=1),
                    nn.GroupNorm(min(32, d_inner // 2), d_inner // 2),
                    nn.GELU(),
                    nn.Conv2d(d_inner // 2, roi_feat_channels, 1),
                )
                if bool(zero_init_gamma):
                    nn.init.zeros_(self.gamma_head[-1].weight)
                    nn.init.zeros_(self.gamma_head[-1].bias)
                if bool(zero_init_beta):
                    nn.init.zeros_(self.beta_head[-1].weight)
                    nn.init.zeros_(self.beta_head[-1].bias)
                self.context_gate_raw = nn.Parameter(
                    torch.tensor(_logit(float(context_gate_init)), dtype=torch.float32)
                )
            else:
                self.gamma_head = None
                self.beta_head = None
                self.register_parameter("context_gate_raw", None)
                self.spatial_cond_proj = nn.Sequential(
                    nn.Conv2d(d_inner, d_inner // 2, 3, padding=1),
                    nn.GroupNorm(min(32, d_inner // 2), d_inner // 2),
                    nn.GELU(),
                    nn.Conv2d(d_inner // 2, roi_feat_channels, 1),
                )
                nn.init.zeros_(self.spatial_cond_proj[-1].weight)
                nn.init.zeros_(self.spatial_cond_proj[-1].bias)
        elif self.fusion_type != "roi_only":
            self.roi_proj = nn.Linear(roi_feat_channels, d_inner)
            self.visual_cross_attn = nn.MultiheadAttention(
                d_inner, num_heads, batch_first=True
            )
            self.cond_proj = nn.Linear(d_inner, roi_feat_channels)
            self.spatial_roi_proj = None
            self.spatial_cross_attn = None
            self.spatial_cond_proj = None
            self.gamma_head = None
            self.beta_head = None
            self.register_parameter("context_gate_raw", None)

        self.mask_decoder = SmallMaskDecoder(
            in_ch=roi_feat_channels,
            mid_ch=128,
            out_ch=1,
            out_size=self.coarse_mask_output_size,
        )

    def forward_context_visual(self, visual_context: Tensor) -> Optional[Tensor]:
        if self.fusion_type == "roi_only":
            return None
        pooled = self.token_pool(visual_context)
        tokens = pooled.flatten(2).transpose(1, 2)
        return self._project_context(tokens)

    def _project_context(self, tokens: Tensor) -> Tensor:
        return self.visual_context_proj(tokens)

    @staticmethod
    def _activate(value: Tensor, name: str) -> Tensor:
        if name == "tanh":
            return torch.tanh(value)
        if name in {"identity", "none"}:
            return value
        raise ValueError(f"Unsupported FiLM activation: {name!r}")

    def forward_roi(
        self,
        ctx_tokens: Optional[Tensor],
        roi_feats: Tensor,
        roi_boxes: Tensor,
        roi_img_ids: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        del kwargs
        debug: Dict[str, Tensor] = {}

        if self.fusion_type == "roi_only":
            modulated = roi_feats
            attn = None
            debug["CONTEXT/enabled"] = roi_feats.new_zeros(())
        elif self.spatial_attn:
            if ctx_tokens is None:
                raise ValueError("context fusion requires visual context tokens")
            kv = ctx_tokens[roi_img_ids.long()]
            h_roi, w_roi = roi_feats.shape[-2:]
            roi_spatial = self.spatial_roi_proj(roi_feats)
            roi_tokens = roi_spatial.flatten(2).transpose(1, 2)
            roi_tokens = roi_tokens + self.box_pe(roi_boxes, self.img_size)[:, None, :]
            enriched, attn = self.spatial_cross_attn(
                roi_tokens, kv, kv, need_weights=False
            )
            enriched = enriched.transpose(1, 2).reshape(
                -1, self.d_inner, h_roi, w_roi
            )
            if self.fusion_type == "gated_spatial_film":
                gamma = self._activate(
                    self.gamma_head(enriched), self.gamma_activation
                )
                beta = self._activate(self.beta_head(enriched), self.beta_activation)
                gate = torch.sigmoid(self.context_gate_raw)
                modulated = roi_feats * (1.0 + gate * gamma) + gate * beta
                debug["CONTEXT/gate"] = gate.detach()
                debug["CONTEXT/gamma_abs_mean"] = gamma.detach().abs().mean()
                debug["CONTEXT/beta_abs_mean"] = beta.detach().abs().mean()
            else:
                cond = self.spatial_cond_proj(enriched)
                modulated = roi_feats * (1.0 + cond)
                debug["CONTEXT/legacy_cond_abs_mean"] = cond.detach().abs().mean()
        else:
            if ctx_tokens is None:
                raise ValueError("context fusion requires visual context tokens")
            kv = ctx_tokens[roi_img_ids.long()]
            roi_global = roi_feats.mean(dim=(2, 3))
            q = self.roi_proj(roi_global) + self.box_pe(roi_boxes, self.img_size)
            delta, attn = self.visual_cross_attn(
                q[:, None], kv, kv, need_weights=False
            )
            cond = self.cond_proj(delta[:, 0])
            modulated = roi_feats * (1.0 + cond[:, :, None, None])
            debug["CONTEXT/legacy_cond_abs_mean"] = cond.detach().abs().mean()

        self._last_debug_stats = debug
        mask_logits = self.mask_decoder(modulated)
        return mask_logits, None, attn, None

    def get_debug_stats(self) -> Dict[str, Tensor]:
        return dict(self._last_debug_stats)


