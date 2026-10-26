"""Depth-constrained cross-view attention on a fixed metric BEV grid."""

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .contract import geometry_config, geometry_signature
from .diagnostics import sampled_attention_stats


class CrossViewLayer(nn.Module):
    def __init__(self, channels, heads, ffn_dim, geometry):
        super().__init__()
        if channels % heads:
            raise ValueError("Attention channels must divide evenly into heads")
        self.heads = heads
        self.channels = channels
        self.geometry = geometry
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.position = nn.Sequential(
            nn.Linear(3, channels), nn.SiLU(), nn.Linear(channels, channels)
        )
        self.view = nn.Embedding(5, channels)
        self.relative = nn.Sequential(nn.Linear(4, 32), nn.SiLU(), nn.Linear(32, heads))
        self.output = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, channels),
        )

    def attend(
        self,
        query,
        keys,
        values,
        position,
        reference,
        anchor_xyz,
        anchor_token,
        indices,
    ):
        valid = indices >= 0
        has_candidate = valid.any(dim=1)
        safe = indices.clamp_min(0)
        token = anchor_token[safe]
        key = keys[token] + position[safe]
        value = values[token]
        relative = anchor_xyz[safe] - reference[:, None]
        scale = relative.new_tensor(
            [self.geometry["radius_m"], self.geometry["radius_m"], 10.0]
        )
        relative = (relative / scale).clamp(-10, 10)
        distance = relative[..., :2].square().sum(dim=-1, keepdim=True)
        bias = self.relative(torch.cat((relative, distance), dim=-1))
        bias = bias - 0.5 * distance
        bias = bias.permute(0, 2, 1).unsqueeze(2)
        bias = bias.masked_fill(~valid[:, None, None], float("-inf"))
        # A dummy key prevents NaN without pretending empty queries are observed.
        bias[~has_candidate, :, :, 0] = 0
        n, count = indices.shape
        width = self.channels // self.heads
        q = query.reshape(n, self.heads, 1, width)
        k = key.reshape(n, count, self.heads, width).transpose(1, 2)
        v = value.reshape(n, count, self.heads, width).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        return out.reshape(n, self.channels) * has_candidate[:, None]

    def forward(self, query, tokens, payload, chunk, use_checkpoint):
        anchor_xyz, anchor_token = payload.anchor_xyz, payload.anchor_token
        if not len(anchor_xyz):
            anchor_xyz = tokens.new_zeros(1, 3)
            anchor_token = torch.zeros(1, dtype=torch.long, device=tokens.device)
        scale = anchor_xyz.new_tensor(
            [self.geometry["side_m"] / 2, self.geometry["side_m"] / 2, 100.0]
        )
        position = self.position(anchor_xyz / scale)
        token_grid = (
            self.geometry["image_size"] // self.geometry["source_stride"]
        ) ** 2
        position = position + self.view(anchor_token // token_grid)
        q = self.query(self.norm(query))
        keys, values = self.key(tokens), self.value(tokens)
        self.diagnostics = sampled_attention_stats(
            self,
            q,
            keys,
            position,
            payload.reference_xyz,
            anchor_xyz,
            anchor_token,
            payload.candidate_index,
        )
        output = []
        for start in range(0, len(query), chunk):
            args = (
                q[start : start + chunk],
                keys,
                values,
                position,
                payload.reference_xyz[start : start + chunk],
                anchor_xyz,
                anchor_token,
                payload.candidate_index[start : start + chunk],
            )
            if use_checkpoint and self.training:
                result = checkpoint(self.attend, *args, use_reentrant=False)
            else:
                result = self.attend(*args)
            output.append(result)
        query = query + self.output(torch.cat(output))
        return query + self.ffn(query)


class DepthCVT(nn.Module):
    def __init__(
        self,
        geometry,
        channels=256,
        layers=2,
        heads=8,
        ffn_dim=512,
        query_chunk=1024,
        checkpoint_attention=True,
    ):
        super().__init__()
        self.geometry = geometry_config(geometry)
        if layers < 1 or query_chunk < 1 or heads < 1:
            raise ValueError("CVT requires positive layers, query chunk and heads")
        self.signature = geometry_signature(geometry)
        self.channels = channels
        self.query_chunk = int(query_chunk)
        self.checkpoint_attention = checkpoint_attention
        n = self.geometry["grid_size"]
        self.query = nn.Parameter(torch.empty(n * n, channels))
        nn.init.normal_(self.query, std=0.02)
        self.query_position = nn.Sequential(
            nn.Linear(3, channels), nn.SiLU(), nn.Linear(channels, channels)
        )
        self.source = nn.Sequential(
            nn.Conv2d(256, channels, 1), nn.GroupNorm(32, channels), nn.SiLU()
        )
        self.layers = nn.ModuleList(
            [
                CrossViewLayer(channels, heads, ffn_dim, self.geometry)
                for _ in range(layers)
            ]
        )

    def forward(self, features, payloads):
        n = self.geometry["grid_size"]
        source_hw = self.geometry["image_size"] // self.geometry["source_stride"]
        if features.shape[1:] != (5, 256, source_hw, source_hw):
            raise ValueError(f"Unexpected UAV feature shape: {features.shape}")
        projected = self.source(features.flatten(0, 1)).reshape(
            len(payloads), 5, self.channels, source_hw, source_hw
        )
        result = []
        diagnostics = []
        for source, payload in zip(projected, payloads):
            if payload.geometry_signature != self.signature:
                raise ValueError("Dataset and model geometry contracts differ")
            tokens = source.permute(0, 2, 3, 1).reshape(-1, self.channels)
            reference = payload.reference_xyz
            scale = reference.new_tensor(
                [self.geometry["side_m"] / 2, self.geometry["side_m"] / 2, 100.0]
            )
            query = self.query + self.query_position(reference / scale)
            for layer in self.layers:
                query = layer(
                    query, tokens, payload, self.query_chunk, self.checkpoint_attention
                )
            # Query bias is not evidence where there is no candidate.
            diagnostics.append(self.layers[-1].diagnostics)
            has_candidate = (payload.candidate_index >= 0).any(dim=1)
            query = query * has_candidate[:, None]
            result.append(query.T.reshape(self.channels, n, n))
        self.diagnostics = {
            key: torch.stack([d[key] for d in diagnostics]).mean()
            for key in diagnostics[0]
        }
        return torch.stack(result)
