"""Geometry-faithful UAV -> BEV fusion used by the UE dual-stream experiment.

The metadata contract is deliberately small: ``drone_images`` is ``[B,V,3,H,W]``
in RGB/[0,1], ``intrinsics`` is ``[B,V,3,3]``, ``extrinsics`` is W2C in UE cm,
and ``scene_centers`` is ``[B,2]`` in UE cm.  No learned camera alignment or
scene embedding is used here.
"""
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class GeometrySparseBEV(nn.Module):
    """Project a fixed 360 m ground square into five UAV feature pyramids.

    Queries are processed in chunks, so local tokens are never materialised for
    the whole 128x128 BEV simultaneously.  The fusion output is zero at init.
    """
    def __init__(self, channels=512, bev_dim=128, heads=4, query_chunk=256,
                 target_meters=360.0, windows=(7, 5, 3, 3)):
        super().__init__()
        if bev_dim % heads:
            raise ValueError("bev_dim must be divisible by heads")
        self.channels, self.bev_dim, self.heads = channels, bev_dim, heads
        self.query_chunk, self.target_meters = int(query_chunk), float(target_meters)
        self.windows = tuple(int(x) for x in windows)
        self.last_stats: dict = {}
        self.query = nn.Conv2d(channels, bev_dim, 1, bias=False)
        self.key = nn.ModuleList(nn.Conv2d(256, bev_dim, 1, bias=False) for _ in windows)
        self.value = nn.ModuleList(nn.Conv2d(256, bev_dim, 1, bias=False) for _ in windows)

    @staticmethod
    def _ground_points(height: int, width: int, centers: Tensor, device, dtype) -> Tensor:
        # target image: u is horizontal, v is vertical; local X/Y then UE cm.
        u = (torch.arange(width, device=device, dtype=dtype) + .5) * (512.0 / width)
        v = (torch.arange(height, device=device, dtype=dtype) + .5) * (512.0 / height)
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        x = -(vv.reshape(-1) / 512.0 - .5) * 360.0
        y = (uu.reshape(-1) / 512.0 - .5) * 360.0
        ones = torch.ones_like(x)
        local = torch.stack((x * 100.0, y * 100.0, torch.zeros_like(x), ones), -1)
        points = local[None].repeat(centers.shape[0], 1, 1)
        points[..., 0] += centers[:, None, 0]
        points[..., 1] += centers[:, None, 1]
        return points

    @staticmethod
    def _sample_tokens(feature: Tensor, uv: Tensor, valid: Tensor, window: int) -> Tensor:
        # feature [B,V,C,H,W], uv [B,V,Q,2] in original 512-pixel image coords.
        b, views, channels, h, w = feature.shape
        radius = window // 2
        offsets_y, offsets_x = torch.meshgrid(
            torch.arange(-radius, radius + 1, device=feature.device, dtype=uv.dtype),
            torch.arange(-radius, radius + 1, device=feature.device, dtype=uv.dtype), indexing="ij")
        offsets = torch.stack((offsets_x.reshape(-1), offsets_y.reshape(-1)), -1)
        # uv / 512 maps naturally to every UAV FPN resolution.  uv stays fp32
        # through the grid math: raw pixel coords near 512 lose sub-pixel
        # precision in fp16, while normalized grid values are safe to cast.
        grid = (uv[..., None, :] / 512.0) * 2.0 - 1.0
        grid = grid + offsets[None, None, None] * torch.tensor(
            (2.0 / w, 2.0 / h), device=feature.device, dtype=uv.dtype)
        grid = grid.to(feature.dtype)
        sampled = F.grid_sample(feature.reshape(b * views, channels, h, w),
                                grid.reshape(b * views, uv.shape[2], -1, 2),
                                mode="bilinear", padding_mode="zeros", align_corners=False)
        sampled = sampled.reshape(b, views, channels, uv.shape[2], -1).permute(0, 3, 1, 4, 2)
        token_valid = valid[..., None].expand(-1, -1, -1, window * window).permute(0, 2, 1, 3)
        return sampled, token_valid.reshape(b, uv.shape[2], -1)

    def forward(self, satellite_p3: Tensor, uav_features: Sequence[Tensor], intrinsics: Tensor,
                extrinsics_w2c: Tensor, scene_centers: Tensor) -> Tensor:
        if len(uav_features) != 4:
            raise ValueError("SAM2 UAV encoder must provide four FPN levels")
        b, _, h, w = satellite_p3.shape
        views = intrinsics.shape[1]
        if views != 5:
            raise ValueError(f"expected exactly 5 UAV views, got {views}")
        # UE world coordinates are centimetres (|scene_center| ~ 2e5, camera
        # translations ~ 1e5), far beyond the fp16 range.  Under AMP the
        # satellite pyramid is fp16, and einsum is autocast-aware so even
        # fp32 inputs get recast to fp16 inside the region: the projection
        # chain must run with autocast disabled, or every ground point
        # overflows to -inf and the BEV collapses to exact zeros.
        with torch.autocast(device_type=satellite_p3.device.type, enabled=False):
            points = self._ground_points(h, w, scene_centers.float(), satellite_p3.device, torch.float32)
            camera = torch.einsum("bvij,bqj->bvqi", extrinsics_w2c.float(), points)
            z = camera[..., 2]
            pix = torch.einsum("bvij,bvqj->bvqi", intrinsics.float(), camera[..., :3])
            uv = pix[..., :2] / pix[..., 2:3].clamp_min(1e-6)
        valid = (z > 1e-4) & (uv[..., 0] >= 0) & (uv[..., 0] < 512) & (uv[..., 1] >= 0) & (uv[..., 1] < 512)
        with torch.no_grad():
            self.last_stats = {
                "valid_ratio": valid.float().mean().detach(),
                "point_valid_ratio": valid.any(1).float().mean().detach(),
            }
        query = self.query(satellite_p3).flatten(2).transpose(1, 2)
        out = query.new_zeros((b, h * w, self.bev_dim))
        d = self.bev_dim // self.heads
        pyramids = [x.reshape(b, views, x.shape[1], x.shape[2], x.shape[3]) for x in uav_features]
        projected = [layer(x) for layer, x in zip(self.key, [p.reshape(b * views, *p.shape[2:]) for p in pyramids])]
        values = [layer(x) for layer, x in zip(self.value, [p.reshape(b * views, *p.shape[2:]) for p in pyramids])]
        projected = [x.reshape(b, views, self.bev_dim, x.shape[-2], x.shape[-1]) for x in projected]
        values = [x.reshape(b, views, self.bev_dim, x.shape[-2], x.shape[-1]) for x in values]
        for start in range(0, h * w, self.query_chunk):
            end = min(start + self.query_chunk, h * w)
            keys, vals, masks = [], [], []
            for kmap, vmap, window in zip(projected, values, self.windows):
                k, mask = self._sample_tokens(kmap, uv[:, :, start:end], valid[:, :, start:end], window)
                v, _ = self._sample_tokens(vmap, uv[:, :, start:end], valid[:, :, start:end], window)
                keys.append(k.reshape(b, end - start, -1, self.bev_dim))
                vals.append(v.reshape(b, end - start, -1, self.bev_dim))
                masks.append(mask)
            k = torch.cat(keys, 2).reshape(b, end - start, -1, self.heads, d).permute(0, 1, 3, 2, 4)
            v = torch.cat(vals, 2).reshape(b, end - start, -1, self.heads, d).permute(0, 1, 3, 2, 4)
            mask = torch.cat(masks, 2)[:, :, None, :]
            q = query[:, start:end].reshape(b, end - start, self.heads, d)
            logits = (q.unsqueeze(-2) * k).sum(-1) * (d ** -0.5)
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
            # Every BEV point has >=1 valid view in the audited dataset.
            out[:, start:end] = (torch.softmax(logits, -1).unsqueeze(-1) * v).sum(-2).reshape(b, end-start, self.bev_dim)
        bev = out.transpose(1, 2).reshape(b, self.bev_dim, h, w)
        with torch.no_grad():
            self.last_stats.update({
                "bev_abs_mean": bev.detach().float().abs().mean(),
                "bev_abs_max": bev.detach().float().abs().max(),
            })
        # The residual gates live at P2/P3 injection.  Do not add a second
        # zero gate here: two serial zero gates would block all fusion gradients.
        return bev


class P2P3ZeroGate(nn.Module):
    """Inject BEV residual into P2/P3 while exactly preserving the baseline at init."""
    def __init__(self, channels=512, bev_dim=128):
        super().__init__()
        self.last_stats: dict = {}
        self.p3 = nn.Conv2d(bev_dim, channels, 1, bias=False)
        self.p2 = nn.Conv2d(bev_dim, channels, 1, bias=False)
        self.gamma_p2 = nn.Parameter(torch.zeros(()))
        self.gamma_p3 = nn.Parameter(torch.zeros(()))

    def forward(self, pyramids: Sequence[Tensor], bev: Tensor):
        out = list(pyramids)
        delta3 = self.gamma_p3 * self.p3(bev)
        delta2 = self.gamma_p2 * self.p2(F.interpolate(bev, size=out[0].shape[-2:], mode="bilinear", align_corners=False))
        with torch.no_grad():
            self.last_stats = {
                "gamma_p2": self.gamma_p2.detach(),
                "gamma_p3": self.gamma_p3.detach(),
                "injected_ratio_p2": delta2.detach().float().norm() / out[0].detach().float().norm().clamp_min(1e-6),
                "injected_ratio_p3": delta3.detach().float().norm() / out[1].detach().float().norm().clamp_min(1e-6),
            }
        out[1] = out[1] + delta3
        out[0] = out[0] + delta2
        return tuple(out)


class A1P2GeometryResidual(nn.Module):
    """A1: frozen-UAV, single-height z=0 residual for satellite P2 only.

    This is intentionally simpler than the legacy P2/P3 BEV attention arm:
    high-resolution UAV FPN tokens are bilinearly sampled at each calibrated
    P2 ground query, averaged over valid views, hard-masked below two views,
    then learnedly projected into P2.  ``gamma`` starts at zero so a warm
    start from the single-stream checkpoint is exactly the anchor at step 0.
    """
    def __init__(self, channels=512, uav_channels=256, query_chunk=2048,
                 target_meters=360.0):
        super().__init__()
        self.channels = int(channels)
        self.uav_channels = int(uav_channels)
        self.query_chunk = int(query_chunk)
        self.target_meters = float(target_meters)
        self.proj = nn.Conv2d(self.uav_channels, self.channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(()))
        self.last_stats: dict = {}

    def forward(self, p2: Tensor, uav_fine: Tensor, intrinsics: Tensor,
                extrinsics_w2c: Tensor, scene_centers: Tensor) -> Tensor:
        b, channels, h, w = p2.shape
        if channels != self.channels:
            raise ValueError(f"P2 channels {channels} != configured {self.channels}")
        views = intrinsics.shape[1]
        if views != 5 or uav_fine.shape[:2] != (b, views):
            raise ValueError("A1 requires Bx5 calibrated UAV fine features")
        with torch.autocast(device_type=p2.device.type, enabled=False):
            points = GeometrySparseBEV._ground_points(h, w, scene_centers.float(), p2.device, torch.float32)
            camera = torch.einsum("bvij,bqj->bvqi", extrinsics_w2c.float(), points)
            z = camera[..., 2]
            pixel = torch.einsum("bvij,bvqj->bvqi", intrinsics.float(), camera[..., :3])
            uv = pixel[..., :2] / pixel[..., 2:3].clamp_min(1e-6)
        valid = ((z > 1e-4) & (uv[..., 0] >= 0) & (uv[..., 0] < 512) &
                 (uv[..., 1] >= 0) & (uv[..., 1] < 512))
        support = valid.sum(1) >= 2
        fmap = uav_fine.reshape(b * views, self.uav_channels, *uav_fine.shape[-2:])
        chunks = []
        for start in range(0, h * w, self.query_chunk):
            end = min(start + self.query_chunk, h * w)
            grid = ((uv[:, :, start:end] / 512.0) * 2.0 - 1.0).to(fmap.dtype)
            sampled = F.grid_sample(fmap, grid.reshape(b * views, 1, end - start, 2),
                                    mode="bilinear", padding_mode="zeros", align_corners=False)
            sampled = sampled.reshape(b, views, self.uav_channels, end - start)
            counts = valid[:, :, start:end].sum(1).clamp_min(1).to(sampled.dtype)
            fused = (sampled * valid[:, :, None, start:end].to(sampled.dtype)).sum(1) / counts[:, None]
            fused = fused * support[:, None, start:end].to(fused.dtype)
            chunks.append(self.proj(fused.unsqueeze(2)).squeeze(2))
        residual = torch.cat(chunks, -1).reshape(b, self.channels, h, w)
        delta = self.gamma * residual
        with torch.no_grad():
            self.last_stats = {
                "candidate_valid_ratio": valid.float().mean().detach(),
                "candidate_two_view_ratio": support.float().mean().detach(),
                "candidate_hard_mask_zero": (
                    residual.masked_select((~support.reshape(b, 1, h, w)).expand_as(residual))
                    .abs().max() == 0
                ).detach(),
                "gamma_p2": self.gamma.detach(),
                "residual_abs_mean": residual.detach().float().abs().mean(),
                "residual_abs_max": residual.detach().float().abs().max(),
                "injected_ratio_p2": delta.detach().float().norm() / p2.detach().float().norm().clamp_min(1e-6),
            }
        return p2 + delta


class EmbeddingZeroGate(nn.Module):
    """Inject the BEV residual into the SAM2 stride-16 image embedding.

    The embedding canvas shares the satellite tile's ground coordinate frame
    (one cell = 16 sat pixels), so the BEV can be injected with the same
    zero-gated residual contract as the P2/P3 pyramid.  This puts UAV
    information on the only feature the mask decoder attends to directly.
    """

    def __init__(self, bev_dim=128, embed_channels=256):
        super().__init__()
        self.last_stats: dict = {}
        self.proj = nn.Conv2d(bev_dim, embed_channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(()))

    def forward(self, image_embeddings: Tensor, bev: Tensor) -> Tensor:
        delta = self.gamma * self.proj(F.interpolate(
            bev, size=image_embeddings.shape[-2:], mode="bilinear", align_corners=False))
        with torch.no_grad():
            self.last_stats = {
                "gamma_emb": self.gamma.detach(),
                "injected_ratio_emb": delta.detach().float().norm()
                / image_embeddings.detach().float().norm().clamp_min(1e-6),
            }
        return image_embeddings + delta
