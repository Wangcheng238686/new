"""Detached gradient diagnostics for the new UE branches, after AMP unscale."""

import math

import torch


def log_ue_gradients(optimizer, logger):
    for group in optimizer.param_groups:
        name = group.get("name", "")
        if name not in ("uav", "cvt", "fusion"):
            continue
        gradients = [
            p.grad.detach().float() for p in group["params"] if p.grad is not None
        ]
        if not gradients:
            logger.info("UAV/grad/%s: used=0 norm=0", name)
            continue
        squares = torch.stack([g.square().sum() for g in gradients]).sum()
        finite = all(bool(torch.isfinite(g).all()) for g in gradients)
        norm = math.sqrt(float(squares))
        logger.info(
            "UAV/grad/%s: used=%d nonzero=%d finite=%d norm=%.6g clip_scale=%.6g",
            name,
            len(gradients),
            sum(bool(g.count_nonzero()) for g in gradients),
            finite,
            norm,
            min(1.0, 0.5 / (norm + 1e-6)),
        )
