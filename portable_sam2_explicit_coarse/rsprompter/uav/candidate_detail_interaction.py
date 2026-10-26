"""Two isolated interventions: select before pooling; consume geometry at stride4.

The existing single-surface CNN and stride8 interaction are retained as the
matched reference path. This module bypasses those losses for retained samples;
it does not reconstruct points removed by the existing eight-candidate cache.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .attention_ladder import masked_weights, sample_source_continuous
from .compact_joint import CompactInteraction


class CandidateDetailInteraction(CompactInteraction):
    def configure(self, variant, satellite):
        assert variant in ('query', 'detail', 'both')
        self.variant = variant
        self.candidate_geometry = None
        self.read_context = None
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2060)
            self.read_query = nn.Linear(112, 32)
            self.read_key = nn.Linear(32, 32, bias=False)
            self.read_geometry_key = nn.Linear(7, 32, bias=False)
            # Q starts as the original mean, including missing views.
            nn.init.zeros_(self.read_key.weight)
            nn.init.zeros_(self.read_geometry_key.weight)
            self.detail_query = nn.Linear(112, 32)
            self.detail_value = nn.Sequential(nn.Linear(7, 32), nn.GELU(), nn.Linear(32, 32))
            self.detail_key = nn.Linear(32, 32, bias=False)
            self.detail_output = nn.Linear(32, 112)
            nn.init.zeros_(self.detail_key.weight)
            nn.init.zeros_(self.detail_output.weight)
            nn.init.zeros_(self.detail_output.bias)
        for module in (self.read_query, self.read_key, self.read_geometry_key):
            module.requires_grad_(variant in ('query', 'both'))
        for module in (self.detail_query, self.detail_value, self.detail_key, self.detail_output):
            module.requires_grad_(variant in ('detail', 'both'))
        trunks = [m for m in satellite.backbone.modules()
                  if hasattr(m, 'stage_ends') and hasattr(m, 'blocks')]
        assert len(trunks) == 1 and trunks[0].blocks[1].dim_out == 112
        self.handles.append(trunks[0].blocks[1].register_forward_hook(self._early_inject))

    def encode(self, geometry):
        assert geometry.shape == (1, 5, 512, 512)
        assert self.candidate_geometry is not None
        b = self.b(self.a(geometry))
        c = self.c(b)
        d = self.d(c)
        u = self.up(torch.cat((b, F.interpolate(c, (128, 128), mode='bilinear', align_corners=False),
                               F.interpolate(d, (128, 128), mode='bilinear', align_corners=False)), 1))
        rgb, index, anchor = self.rgb_context
        maps = []
        for image in rgb.split(1):
            _, pyramid = self.rgb_encoder((image-self.rgb_mean)/self.rgb_std)
            maps.append(self.rgb_project(pyramid[0]))
        features = torch.cat(maps).permute(0, 2, 3, 1).reshape(-1, 32)
        features = F.layer_norm(features, (32,))
        selected = sample_source_continuous(features, index, anchor)
        support = F.max_pool2d(geometry[:, 4:5], 4, 4) > 0
        self.read_context = (u, selected, index >= 0, self.candidate_geometry, support)
        self.context = (u, support)
        self.outputs = None

    def reduce_rgb(self, satellite, selected, valid, geometry):
        if self.variant == 'detail':
            return (selected*valid[..., None]).sum(1)/valid.sum(1, keepdim=True).clamp_min(1)
        query = self.read_query(F.layer_norm(satellite, (112,))).float()
        key = self.read_key(selected).float() + self.read_geometry_key(geometry.to(selected.dtype)).float()
        logits = (query[:, None]*key).sum(-1)/math.sqrt(32)
        weight = masked_weights(logits, valid)
        return (weight[..., None]*selected.float()).sum(1).to(selected.dtype)

    def geometry_delta(self, satellite, geometry, valid):
        if self.variant == 'query':
            return torch.zeros_like(satellite)
        query = self.detail_query(F.layer_norm(satellite, (112,))).float()
        value = self.detail_value(geometry.to(satellite.dtype))
        key = self.detail_key(value).float()
        logits = (query[:, None]*key).sum(-1)/math.sqrt(32)
        weight = masked_weights(logits, valid)
        evidence = (weight[..., None]*value.float()).sum(1)
        delta = self.detail_output(F.layer_norm(evidence, (32,)).to(satellite.dtype))
        return delta*valid.any(1, keepdim=True)

    def _early_inject(self, module, inputs, base):
        if not self.enabled or self.read_context is None:
            return base
        assert base.shape == (1, 128, 128, 112)
        u, selected, valid, geometry, support = self.read_context
        satellite = base.reshape(16384, 112)
        aligned = self.reduce_rgb(satellite, selected, valid, geometry)
        aligned = aligned.T.reshape(1, 32, 128, 128)
        self.context = (self.rgb_fusion(torch.cat((u, aligned), 1)), support)
        delta = self.geometry_delta(satellite, geometry, valid)
        delta = delta.reshape_as(base)*support.permute(0, 2, 3, 1)
        return base+delta.to(base.dtype)

    def clear(self):
        super().clear()
        self.read_context = None
        self.candidate_geometry = None


def install(adapter, variant, satellite):
    assert isinstance(adapter, CompactInteraction)
    adapter.__class__ = CandidateDetailInteraction
    adapter.configure(variant, satellite)
    adapter.cuda()
    return adapter
