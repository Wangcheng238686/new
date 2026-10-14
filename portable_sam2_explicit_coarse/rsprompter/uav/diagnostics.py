"""Bounded, detached attention diagnostics; never used to gate model output."""

import math

import torch


@torch.no_grad()
def sampled_attention_stats(
    layer, query, keys, position, reference, xyz, token, indices
):
    n = layer.geometry["grid_size"]
    axis = (
        (torch.arange(min(n, 16), device=query.device) + 0.5) * n / min(n, 16)
    ).long()
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    rows = (y * n + x).flatten()
    rows = rows[(indices[rows] >= 0).any(dim=1)]
    if not len(rows):
        return {
            name: query.new_zeros(())
            for name in (
                "local_mass",
                "height_compatible_mass",
                "xy_bias_m",
                "height_bias_m",
                "sampled_queries",
            )
        }
    selected = indices[rows]
    valid = selected >= 0
    safe = selected.clamp_min(0)
    delta = xyz[safe] - reference[rows, None]
    scale = delta.new_tensor(
        [layer.geometry["radius_m"], layer.geometry["radius_m"], 10.0]
    )
    relative = (delta / scale).clamp(-10, 10)
    distance = relative[..., :2].square().sum(-1, keepdim=True)
    # A no-grad autocast of these shared weights can cache detached half
    # tensors, poisoning the later training/checkpointed attention call.
    # Evaluate diagnostics without autocast or its weight cache.
    with torch.autocast(
        device_type=query.device.type, enabled=False, cache_enabled=False
    ):
        bias = layer.relative(
            torch.cat((relative, distance), -1).to(layer.relative[0].weight.dtype)
        )
    bias = bias.float() - 0.5 * distance
    key = keys[token[safe]] + position[safe]
    width = layer.channels // layer.heads
    q = query[rows].reshape(len(rows), layer.heads, width).float()
    k = (
        key.reshape(len(rows), selected.shape[1], layer.heads, width)
        .permute(0, 2, 1, 3)
        .float()
    )
    logits = (q[:, :, None] * k).sum(-1) / math.sqrt(width) + bias.permute(
        0, 2, 1
    ).float()
    weights = logits.masked_fill(~valid[:, None], float("-inf")).softmax(-1).mean(1)
    cell = layer.geometry["side_m"] / layer.geometry["grid_size"]
    local = (delta[..., :2].abs() <= cell / 2).all(-1)
    compatible = delta[..., 2].abs() <= layer.geometry["height_bin_m"]
    offset = (weights[..., None] * delta).sum(1)
    return dict(
        local_mass=(weights * local).sum(1).mean(),
        height_compatible_mass=(weights * compatible).sum(1).mean(),
        xy_bias_m=offset[:, :2].norm(dim=1).mean(),
        height_bias_m=offset[:, 2].abs().mean(),
        sampled_queries=query.new_tensor(float(len(rows))),
    )
