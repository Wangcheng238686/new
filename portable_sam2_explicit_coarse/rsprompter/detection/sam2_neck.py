"""Detection-oriented FPN/PAN neck for SAM2 backbone_fpn features."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS

@MODELS.register_module()
class RSSAM2PAFPN(BaseModule):
    """Detection-oriented FPN/PAN neck for SAM2 backbone_fpn features.

    SAM2 provides four 256-channel maps at strides 4/8/16/32. This neck keeps
    those levels separate, sends semantics down with an FPN path, then sends
    localization back up with a PAN path. The output contract matches the
    existing detector: five 512-channel maps at strides 4/8/16/32/64.

    Channel design (mid/out decoupling):
      - ``mid_channels`` controls the internal FPN/PAN computation width. All
        lateral/top-down/downsample/bottom-up/p64 convs operate at this width.
        Default 256 matches the SAM2 backbone output, so no information is lost
        while the N×N conv cost drops from out² to mid² (4× at mid=256).
      - ``out_channels`` (512) is the detector interface contract: RPN/SABL/
        RoIExtractor all expect 512. A per-level 1×1 projection (``output_proj``)
        lifts mid→out at the very end, keeping downstream configs untouched.
      - To reproduce the legacy all-out behavior set ``mid_channels=out_channels``
        (the output_proj then becomes an identity-ish 1×1).
    """

    in_channels_dict = {
        "sam2_hiera_tiny": [256, 256, 256, 256],
        "sam2_hiera_small": [256, 256, 256, 256],
        "sam2_hiera_base": [256, 256, 256, 256],
        "sam2_hiera_base_plus": [256, 256, 256, 256],
        "sam2_hiera_large": [256, 256, 256, 256],
    }

    def __init__(
        self,
        in_channels: str = "sam2_hiera_large",
        out_channels: int = 512,
        mid_channels: int = 256,
        num_levels: int = 4,
        norm_groups: int = 32,
        init_cfg: Optional[Dict] = None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.out_channels = out_channels
        self.mid_channels = int(mid_channels)
        self.num_levels = num_levels
        # 审查 3.2：禁止 neck key 静默 fallback。未知 key 直接报错。
        if in_channels not in self.in_channels_dict:
            raise KeyError(
                f"Unsupported SAM2 neck key: {in_channels!r}. "
                f"Expected one of {sorted(self.in_channels_dict.keys())}."
            )
        self.in_channels = self.in_channels_dict[in_channels]
        if len(self.in_channels) < num_levels:
            self.in_channels = self.in_channels + [self.in_channels[-1]] * (num_levels - len(self.in_channels))

        # Internal FPN/PAN computation lives at mid_channels.
        self.lateral_convs = nn.ModuleList([
            self._conv_block(ch, self.mid_channels, kernel_size=1, padding=0, norm_groups=norm_groups)
            for ch in self.in_channels[:num_levels]
        ])
        self.top_down_refine = nn.ModuleList([
            self._conv_block(self.mid_channels, self.mid_channels, kernel_size=3, padding=1, norm_groups=norm_groups)
            for _ in range(num_levels)
        ])
        self.downsample_convs = nn.ModuleList([
            self._conv_block(self.mid_channels, self.mid_channels, kernel_size=3, stride=2, padding=1, norm_groups=norm_groups)
            for _ in range(num_levels - 1)
        ])
        self.bottom_up_refine = nn.ModuleList([
            self._conv_block(self.mid_channels, self.mid_channels, kernel_size=3, padding=1, norm_groups=norm_groups)
            for _ in range(num_levels)
        ])
        self.p64_conv = self._conv_block(self.mid_channels, self.mid_channels, kernel_size=3, stride=2, padding=1, norm_groups=norm_groups)
        # Per-level 1×1 projection lifts mid→out to satisfy the detector's
        # in_channels=512 contract without paying out² across all N×N convs.
        num_outputs = num_levels + 1  # + P64
        self.output_proj = nn.ModuleList([
            self._conv_block(self.mid_channels, out_channels, kernel_size=1, padding=0, norm_groups=norm_groups)
            for _ in range(num_outputs)
        ])
        self.level_scales = nn.Parameter(torch.ones(num_outputs))
        self._init_weights()

    @staticmethod
    def _conv_block(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        norm_groups: int,
        stride: int = 1,
    ) -> nn.Sequential:
        groups = min(norm_groups, out_channels)
        while out_channels % groups != 0:
            groups -= 1
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    @staticmethod
    def _parse_inputs(inputs) -> Tuple[Tensor, ...]:
        if isinstance(inputs, tuple) and len(inputs) == 2 and isinstance(inputs[1], tuple):
            inputs = inputs[1]
        if not isinstance(inputs, (tuple, list)) or len(inputs) == 0:
            raise ValueError(f"Unexpected inputs to RSSAM2PAFPN: {type(inputs)}")
        return tuple(inputs)

    def forward(self, inputs) -> Tuple[Tensor, ...]:
        inputs = self._parse_inputs(inputs)
        while len(inputs) < self.num_levels:
            inputs = inputs + (inputs[-1],)
        inputs = inputs[:self.num_levels]

        laterals = [conv(x) for conv, x in zip(self.lateral_convs, inputs)]

        top_down = [None] * self.num_levels
        top_down[-1] = self.top_down_refine[-1](laterals[-1])
        for i in range(self.num_levels - 2, -1, -1):
            up = F.interpolate(top_down[i + 1], size=laterals[i].shape[-2:], mode="nearest")
            top_down[i] = self.top_down_refine[i](laterals[i] + up)

        outputs = [None] * self.num_levels
        outputs[0] = self.bottom_up_refine[0](top_down[0])
        for i in range(1, self.num_levels):
            down = self.downsample_convs[i - 1](outputs[i - 1])
            if down.shape[-2:] != top_down[i].shape[-2:]:
                down = F.interpolate(down, size=top_down[i].shape[-2:], mode="nearest")
            outputs[i] = self.bottom_up_refine[i](top_down[i] + down)

        outputs.append(self.p64_conv(outputs[-1]))
        # Lift mid→out per level to satisfy the detector's in_channels contract.
        outputs = [proj(out) for proj, out in zip(self.output_proj, outputs)]
        outputs = [out * scale for out, scale in zip(outputs, self.level_scales)]
        return tuple(outputs)
