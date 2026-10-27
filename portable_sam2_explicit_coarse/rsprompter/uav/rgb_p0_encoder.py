"""The exact early Hiera path consumed by the current UAV P0 reader.

This opt-in module is not wired into any frozen probe. It removes source stages
whose outputs do not contribute to P0; it does not add or approximate features.
"""
import copy

import torch
from torch import nn
from torch.nn import functional as F


class RGBP0Encoder(nn.Module):
    def __init__(self, source_encoder):
        super().__init__()
        image = source_encoder.encoder
        if hasattr(image, 'get_base_model'):
            image = image.get_base_model()
        trunk, neck = image.trunk, image.neck
        assert list(trunk.stage_ends) == [1, 4, 20, 23]
        assert trunk.return_interm_layers
        assert 0 not in neck.fpn_top_down_levels, 'P0 must be lateral-only'
        assert len(neck.convs) == 4
        assert trunk.blocks[1].dim_out == 112
        self.patch_embed = copy.deepcopy(trunk.patch_embed)
        self.pos_embed = nn.Parameter(trunk.pos_embed.detach().clone(),
                                      requires_grad=trunk.pos_embed.requires_grad)
        self.pos_embed_window = nn.Parameter(trunk.pos_embed_window.detach().clone(),
                                             requires_grad=trunk.pos_embed_window.requires_grad)
        self.blocks = nn.ModuleList(copy.deepcopy(list(trunk.blocks[:2])))
        self.lateral = copy.deepcopy(neck.convs[3])
        self.train(False)

    def train(self, mode=True):
        # Match the probe: fixed original encoder state, LoRA dropout may train.
        super().train(False)
        for name, module in self.named_modules():
            if 'lora_dropout' in name:
                module.train(mode)
        return self

    def forward(self, image):
        x = self.patch_embed(image)
        h, w = x.shape[1:3]
        position = F.interpolate(self.pos_embed, size=(h, w), mode='bicubic')
        tile = [a // b for a, b in zip(position.shape, self.pos_embed_window.shape)]
        position = position + self.pos_embed_window.tile(tile)
        x = x + position.permute(0, 2, 3, 1)
        for block in self.blocks:
            x = block(x)
        return self.lateral(x.permute(0, 3, 1, 2))

    @staticmethod
    def source_key(name):
        if name.startswith('lateral.'):
            return 'neck.convs.3.' + name[len('lateral.'):]
        return 'trunk.' + name

    def audit_source_state(self, source_encoder):
        image = source_encoder.encoder
        if hasattr(image, 'get_base_model'):
            image = image.get_base_model()
        state = image.state_dict()
        for name, value in self.state_dict().items():
            assert torch.equal(value, state[self.source_key(name)]), name
        return len(self.state_dict())
