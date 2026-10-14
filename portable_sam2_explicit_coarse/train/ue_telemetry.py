"""Detached UE telemetry; rank-local snapshots and cross-rank epoch means."""

import json

import torch
import torch.distributed as dist


@torch.no_grad()
def log_ue_gradients(model, optimizer, logger):
    """Match actual decoupled clip partitions, before clipping, after unscale."""
    core = model.module if hasattr(model, "module") else model
    prefixes = ("uav_backbone.", "uav_bev.", "fpn_fusion.", "sam_fusion.")
    lrs = {g.get("name"): g["lr"] for g in optimizer.param_groups}
    for prefix, lr_name in zip(prefixes, ("uav", "cvt", "fusion", "fusion")):
        params = [
            p
            for n, p in core.named_parameters()
            if n.startswith(prefix) and p.requires_grad
        ]
        if not params:
            continue
        grads = [p.grad.detach().float() for p in params if p.grad is not None]
        if not grads:
            logger.info("UAV/grad/%s: used=0 norm=0 lr=%.6g", prefix[:-1], lrs[lr_name])
            continue
        norm = torch.stack([g.square().sum() for g in grads]).sum().sqrt()
        finite = all(bool(torch.isfinite(g).all()) for g in grads)
        clip_scale = (0.5 / (norm + 1e-6)).clamp(max=1)
        logger.info(
            "UAV/grad/%s: used=%d nonzero=%d finite=%d norm=%.6g clip_scale=%.6g lr=%.6g",
            prefix[:-1],
            len(grads),
            sum(bool(g.count_nonzero()) for g in grads),
            finite,
            float(norm),
            float(clip_scale),
            lrs[lr_name],
        )


class UEEpochTelemetry:
    """Per-key finite-forward means; no per-forward host transfer or collectives."""

    def __init__(self, device):
        self.device = device
        self.totals = {}
        self.attempts = 0
        self.skips = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    @torch.no_grad()
    def accumulate(self, model):
        core = model.module if hasattr(model, "module") else model
        for key, value in (core.get_telemetry() or {}).items():
            if not key.startswith("UAV/"):
                continue
            value = torch.as_tensor(value, device=self.device).detach().float()
            if value.numel() != 1:
                continue
            value = value.reshape(())
            finite = torch.isfinite(value)
            pair = torch.stack(
                (torch.where(finite, value, 0).double(), finite.double())
            )
            self.totals[key] = self.totals.get(key, 0) + pair

    def record_step(self, scale_before, scale_after):
        self.attempts += 1
        self.skips += int(scale_after < scale_before)

    @torch.no_grad()
    def finish(self, logger, epoch, distributed, is_main, scaler):
        keys = sorted(self.totals)
        if distributed and dist.is_initialized():
            all_keys = [None] * dist.get_world_size()
            dist.all_gather_object(all_keys, keys)
            keys = sorted(set().union(*map(set, all_keys)))
        packed = (
            torch.stack(
                [
                    self.totals.get(
                        k, torch.zeros(2, dtype=torch.float64, device=self.device)
                    )
                    for k in keys
                ]
            )
            if keys
            else torch.zeros((0, 2), dtype=torch.float64, device=self.device)
        )
        counters = torch.tensor(
            [self.attempts, self.skips], dtype=torch.float64, device=self.device
        )
        peak = torch.tensor(
            [
                (
                    torch.cuda.max_memory_allocated(self.device) / 2**20
                    if self.device.type == "cuda"
                    else 0
                ),
                (
                    torch.cuda.max_memory_reserved(self.device) / 2**20
                    if self.device.type == "cuda"
                    else 0
                ),
            ],
            dtype=torch.float64,
            device=self.device,
        )
        if distributed and dist.is_initialized():
            if keys:
                dist.all_reduce(packed)
            dist.all_reduce(counters)
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        report = {
            k: {
                "mean": float(packed[i, 0] / packed[i, 1]),
                "finite_forwards": int(packed[i, 1]),
            }
            for i, k in enumerate(keys)
            if packed[i, 1] > 0
        }
        report["runtime"] = dict(
            rank_step_attempts=int(counters[0]),
            rank_amp_skips=int(counters[1]),
            amp_skip_fraction=float(counters[1] / counters[0].clamp_min(1)),
            amp_scale_rank0=float(scaler.get_scale()),
            max_rank_allocated_mib=float(peak[0]),
            max_rank_reserved_mib=float(peak[1]),
        )
        if is_main:
            logger.info(
                "[UE-EPOCH] epoch=%d %s",
                epoch,
                json.dumps(report, sort_keys=True, allow_nan=False),
            )
        return report
