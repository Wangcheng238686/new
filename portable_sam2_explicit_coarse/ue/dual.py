"""Perimeter dual-stream adapter for the sealed a3sgm baseline.

EXP-0008 arm: frozen UAV encoder + world-geometry projection of UAV features
onto the satellite P3 grid (single surface bin via the SurfaceBEV rasterizer)
+ zero-initialised per-channel gate. At initialisation the model is exactly
the single-stream architecture (gate=0 identity); telemetry records whether
training opens the UAV tap. The sealed rsprompter packages are untouched.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.config import ConfigDict
from rsprompter.detection.anchor_detector import RSPrompterAnchor
from rsprompter.detection.anchor_roi_head import RSPrompterAnchorRoIPromptHead
from rsprompter.surfacebev import rasterize_surfacebev
from mmdet.models.data_preprocessors import DetDataPreprocessor

_UAV_MEAN = (0.485, 0.456, 0.406)
_UAV_STD = (0.229, 0.224, 0.225)


class BEVDescAttention(nn.Module):
    """CVT-style attention aggregation of per-view descriptor cells into the
    BEV (PIXEL-CLS probe 2026-10-17: the building signal EXISTS per-pixel and
    is washed out by the rasterizer's cross-view averaging).

    Queries: BEV cells anchored in world coordinates (satellite-camera
    ray/ground-plane intersection, z=0). Keys: per-view descriptor cells with
    geometric embedding from the exact metric cameras (world ray direction +
    camera centre). Values: projected per-view descriptors. The metric
    geometry anchors WHICH cells can talk; the softmax routes the content.
    """

    def __init__(self, desc_dim=256, out_dim=128, geom_dim=64, heads=4,
                 sat_dim=0):
        super().__init__()
        self.heads = heads
        self.sat_dim = int(sat_dim)
        self.q_geom = nn.Sequential(nn.Linear(2, geom_dim), nn.ReLU(),
                                    nn.Linear(geom_dim, geom_dim))
        self.k_dir = nn.Sequential(nn.Linear(3, geom_dim), nn.ReLU(),
                                   nn.Linear(geom_dim, geom_dim))
        self.k_center = nn.Sequential(nn.Linear(3, geom_dim), nn.ReLU(),
                                      nn.Linear(geom_dim, geom_dim))
        d = geom_dim + desc_dim
        # WARM-R1Q: query contentization - the query concatenates the frozen
        # satellite P2 feature at the same BEV cell (projected, no gradient
        # backflow into the frozen base) so keys can be retrieved by CONTENT
        # match, not only by geometry. sat_dim=0 keeps the R1C query verbatim.
        self.sat_proj = (nn.Linear(int(sat_dim), geom_dim)
                         if int(sat_dim) > 0 else None)
        self.q_proj = nn.Linear(geom_dim * (2 if int(sat_dim) > 0 else 1),
                                out_dim)
        self.k_proj = nn.Linear(d, out_dim, bias=False)
        self.v_proj = nn.Linear(desc_dim, out_dim, bias=False)
        self.out_norm = nn.GroupNorm(8, out_dim)
        self.scale = (out_dim // heads) ** -0.5

    def forward(self, desc, uav_k, uav_c2w, sat_k, sat_w2c, out_hw, sat_feat=None):
        """desc [V,C,h,w]; uav_k [V,3,3]; uav_c2w [V,4,4]; sat on BEV grid.
        Attention runs on the 64x64 probe grid (PIXEL-CLS granularity) and
        the aggregate is bilinearly upsampled to the injection resolution."""
        V, C, h, w = desc.shape
        dev = desc.device
        # BEV cell world anchors: sat pixel grid -> camera ray -> z=0 ground
        oh, ow = 64, 64
        vs, us = torch.meshgrid(torch.arange(oh, device=dev, dtype=torch.float32),
                                torch.arange(ow, device=dev, dtype=torch.float32),
                                indexing='ij')
        # audit fixes 2026-10-17 (implementation review FAIL->fix):
        # (i) rescale grid centres into the 512-px intrinsics frame;
        # (ii) apply R_c2w (row form: d_cam @ R_w2c), NOT R_w2c;
        # (iii) sign-preserving denominator - downward rays have denom<0,
        #       clamp_min flipped them to +1e-9 and pushed s to -3e13.
        pix = torch.stack(((us + .5) * 512.0 / ow, (vs + .5) * 512.0 / oh,
                           torch.ones_like(us)), -1).reshape(-1, 3)      # [HW,3]
        sat_c2w = torch.linalg.inv(sat_w2c)
        dirs_cam = torch.linalg.solve(sat_k, pix.T).T                    # [HW,3] rays in cam
        dirs_w = dirs_cam @ sat_w2c[:3, :3]                              # world dirs (R_c2w row form)
        org = sat_c2w[:3, 3]                                             # cam centre world
        denom = dirs_w[:, 2]
        denom_safe = torch.where(denom.abs() > 1e-9, denom,
                                 torch.full_like(denom, -1e-9))
        s = -org[2] / denom_safe
        gnd = org + s[:, None] * dirs_w                                  # [HW,3] on z=0
        q_g = self.q_geom(gnd[:, :2] / 20000.0)                          # tile half-extent norm
        # WARM-R1Q query contentization: concatenate the frozen satellite
        # feature at the same BEV cell (row-major [HW] order identical to
        # the vs/us meshgrid; .float() for the fp16-under-AMP backbone out;
        # bilinear-downsampled only when the grid differs from 64x64).
        if self.sat_proj is not None and sat_feat is not None:
            assert sat_feat.dim() in (3, 4) and sat_feat.shape[-3] == self.sat_dim
            sf = sat_feat.detach().float()
            if sf.dim() == 4:            # batched neck output [1,C,H,W]
                sf = sf[0]
            if tuple(sf.shape[-2:]) != (oh, ow):
                sf = F.interpolate(sf[None], size=(oh, ow), mode='bilinear',
                                   align_corners=False)[0]
            q_in = torch.cat((q_g, self.sat_proj(sf.flatten(1).T)), dim=1)
        else:
            q_in = q_g
        q = self.q_proj(q_in)                                            # [HW,d]

        # keys/values: per-view cells, world ray dirs + camera centres
        # (cell centres rescaled into the 512-px intrinsics frame - audit fix)
        ys, xs = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32),
                                torch.arange(w, device=dev, dtype=torch.float32),
                                indexing='ij')
        cell = torch.stack(((xs + .5) * 512.0 / w, (ys + .5) * 512.0 / h,
                            torch.ones_like(xs)), -1).reshape(-1, 3)     # [N,3]
        ks, vs_, centers = [], [], []
        for v in range(V):
            d_cam = torch.linalg.solve(uav_k[v], cell.T).T
            d_w = d_cam @ uav_c2w[v, :3, :3].T
            d_w = d_w / d_w.norm(dim=1, keepdim=True).clamp_min(1e-9)
            ks.append(self.k_dir(d_w) + self.k_center(
                uav_c2w[v, :3, 3][None].expand(d_w.shape[0], 3) / 40000.0))
            vs_.append(desc[v].flatten(1).T)                            # [N,C]
        k = self.k_proj(torch.cat([torch.cat([g, f], 1)
                                   for g, f in zip(ks, vs_)], 0))        # [V*N,d]
        vfeat = self.v_proj(torch.cat(vs_, 0))                          # [V*N,d]
        d = q.shape[-1]
        Hh = self.heads
        kc = k.reshape(-1, Hh, d // Hh).transpose(0, 1).contiguous()     # [Hh,VN,dh]
        vc = vfeat.reshape(-1, Hh, d // Hh).transpose(0, 1).contiguous()
        outs = []
        CH = 1024  # chunked SDPA: caps the materialised attn per chunk even
        # when the math backend is selected (autocast off in the dual path)
        for i in range(0, q.shape[0], CH):
            qc = q[i:i + CH].reshape(-1, Hh, d // Hh).transpose(0, 1).contiguous()
            o = F.scaled_dot_product_attention(qc, kc, vc, scale=self.scale)
            outs.append(o.transpose(0, 1).reshape(-1, d))
        out = torch.cat(outs, 0).T.reshape(-1, oh, ow)                  # [d,64,64]
        if (oh, ow) != tuple(out_hw):
            out = F.interpolate(out[None], size=tuple(out_hw), mode='bilinear',
                                align_corners=False)[0]
        # normalise across channels (BN-free, batch=1 safe)
        out = self.out_norm(out[None])[0]
        return out                                                      # [d,oh,ow]


@MODELS.register_module()
class RoIUAVSampler(nn.Module):
    """WARM-R1R: per-box UAV descriptor sampling (PI spec 2026-10-19).

    For each RoI box: a 7x7 grid of points in the box -> satellite pixels ->
    height z bilinearly read from the zmean surface -> world points via the
    satellite ray/height-plane intersection -> projected into the 5 UAV views
    -> per-pixel descriptors bilinearly sampled with a validity mask (depth>0,
    in-bounds) -> masked-mean pooled to a 256-d per-box feature. Exact metric
    cameras; the zmean surface comes from the rasterizer's stats channel.
    """

    def __init__(self, desc_dim=256, out_dim=256, grid=7, mode='desc'):
        super().__init__()
        self.grid = int(grid)
        self.mode = str(mode)
        if self.mode == 'desc':
            self.proj = nn.Linear(desc_dim, out_dim)
        elif self.mode != 'evi':
            raise ValueError(f"unknown RoIUAVSampler mode {mode}")

    def forward(self, rois, desc, uav_k, uav_c2w, sat_k, sat_w2c, zmean_map):
        """rois [N,5] (batch_idx + x1y1x2y2 in sat pixels); desc [V,C,h,w]
        (mode='desc': 256-d descriptors through the proj; mode='evi': the 2-ch
        decoded sigmoid maps, flattened 7x7x2 per box, no proj);
        uav_k [V,3,3]; uav_c2w [V,4,4]; sat_k [3,3]; sat_w2c [4,4];
        zmean_map [H,W] heights in the sat-pixel frame."""
        N = rois.shape[0]
        dev = rois.device
        if N == 0:
            if self.mode == 'desc':
                return rois.new_zeros(0, self.proj.out_features)
            return rois.new_zeros(0, self.grid * self.grid * 2)
        g = self.grid
        # 7x7 grid in normalised box coords -> sat pixels [N, 49, 2]
        t = torch.linspace(0.0, 1.0, g, device=dev)
        gy, gx = torch.meshgrid(t, t, indexing="ij")
        gx = gx.reshape(-1); gy = gy.reshape(-1)
        x1, y1, x2, y2 = rois[:, 1], rois[:, 2], rois[:, 3], rois[:, 4]
        u = x1[:, None] + gx[None] * (x2 - x1)[:, None]
        v = y1[:, None] + gy[None] * (y2 - y1)[:, None]
        # z from the zmean surface (bilinear, normalised coords)
        H, W = zmean_map.shape[-2:]
        zgrid = torch.stack((u / W * 2 - 1, v / H * 2 - 1), -1)[..., None, :] \
            .reshape(1, -1, 1, 2)                                     # [1,N*49,1,2]
        zg = F.grid_sample(zmean_map[None, None].float(), zgrid,
                           mode='bilinear', align_corners=False)[0, 0, :, 0]  # [N*49]
        # sat pixel -> world ray -> height-plane intersection
        pix = torch.stack((u.reshape(-1) + .5, v.reshape(-1) + .5,
                           torch.ones_like(u.reshape(-1))), -1)        # [N*49,3]
        sat_c2w = torch.linalg.inv(sat_w2c)
        d_cam = torch.linalg.solve(sat_k, pix.T).T
        d_w = d_cam @ sat_w2c[:3, :3]
        org = sat_c2w[:3, 3]
        denom = d_w[:, 2]
        dsafe = torch.where(denom.abs() > 1e-9, denom,
                            torch.full_like(denom, -1e-9))
        sc = (zg.reshape(-1) - org[2]) / dsafe
        world = org + sc[:, None] * d_w                                # [N*49,3]
        # project into each view, masked descriptor sampling
        V, C, h, w = desc.shape
        uv_scale = torch.tensor([w / 512.0, h / 512.0],
                                device=desc.device, dtype=desc.dtype)
        acc = desc.new_zeros(N * g * g, C)
        cnt = desc.new_zeros(N * g * g)
        for vi in range(V):
            cam = world @ uav_c2w[vi, :3, :3].T + uav_c2w[vi, :3, 3]
            zc = cam[:, 2]
            # project with the 512-frame intrinsics, then rescale into the
            # descriptor grid frame (fpn h x w) - the frame mismatch had
            # invalidated every point in the first smoke (cnt0_frac=1)
            uv = torch.stack((uav_k[vi, 0, 0] * cam[:, 0] / zc.clamp_min(1e-6)
                              + uav_k[vi, 0, 2],
                              uav_k[vi, 1, 1] * cam[:, 1] / zc.clamp_min(1e-6)
                              + uav_k[vi, 1, 2]), -1) * uv_scale
            inb = (uv[:, 0] >= 0) & (uv[:, 0] <= w - 1) \
                  & (uv[:, 1] >= 0) & (uv[:, 1] <= h - 1)
            valid = (zc > 0) & inb
            gnm = (torch.where(inb, uv[:, 0] / (w - 1) * 2 - 1,
                               torch.zeros_like(uv[:, 0])),
                   torch.where(inb, uv[:, 1] / (h - 1) * 2 - 1,
                               torch.zeros_like(uv[:, 1])))
            vgrid = torch.stack(gnm, -1)[..., None, :].reshape(1, -1, 1, 2)
            sampled = F.grid_sample(desc[vi][None], vgrid,
                                    mode='bilinear', align_corners=True,
                                    padding_mode='zeros')[0, :, :, 0].T  # [N*49,C]
            if os.environ.get('R1R_DEBUG'):
                print(f'[r1r-view{vi}] zc_pos_frac={float((zc > 0).float().mean()):.3g} '
                      f'inb_frac={float(inb.float().mean()):.3g}', flush=True)
            acc = acc + sampled * valid[:, None].float()
            cnt = cnt + valid.float()
        per_point = acc / cnt.clamp_min(1.0)[:, None]                   # [N*49,C]
        if self.mode == 'evi':
            # decoded [evidence, boundary] values: keep the 7x7 spatial layout
            return per_point.reshape(N, g * g * C)
        per_box = per_point.reshape(N, g * g, C).mean(1)                # [N,C]
        if os.environ.get('R1R_DEBUG'):
            def _f(t):
                return bool(torch.isfinite(t).all()) if torch.isfinite(t).numel() else True
            print(f'[r1r-smp] rois_f={_f(rois)} u_f={_f(u)} v_f={_f(v)} zg_f={_f(zg)} '
                  f'zg_absmax={float(zg.abs().max()):.3g} world_f={_f(world)} '
                  f'cnt0_frac={float((cnt==0).float().mean()):.3g} acc_f={_f(acc)} '
                  f'per_box_f={_f(per_box)} per_box_absmax={float(per_box.abs().max()):.3g}', flush=True)
        return self.proj(per_box)


class EviDecoder(nn.Module):
    """WARM-R1V supply side (arm E prereg 2026-10-19): per-view conv decoder
    mapping the frozen-encoder descriptor grid to [building-evidence,
    footprint-boundary] logits. Pretrained offline by probe_evi_decoder.py
    (BCE+Dice on PIXEL-CLS geometry labels) and loaded FROZEN at runtime -
    the arm consumes the decoded answers, never the raw descriptors. The
    architecture must match scripts/ue/probe_evi_decoder.py::EviDecoder
    verbatim (state_dict load is strict)."""

    def __init__(self, in_ch=256, hidden=(128, 64), out_ch=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden[0], 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden[0], hidden[1], 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden[1], out_ch, 1),
        )

    def forward(self, x):
        return self.net(x)


def _view_label_grid(uav, hw, dev):
    """Per-view satellite-pixel lookup grid at descriptor-cell resolution
    (PIXEL-CLS geometry): returns su, sv [V,h,w] long indices into the 512
    sat frame and valid [V,h,w] (depth>0, in front of the sat camera, sane
    height). Consumed by the WARM-R1V decode-consistency aux at loss time
    (GT union is only reachable there)."""
    from rsprompter.surfacebev import devicez_to_camera_z
    depth = uav['devicez_rgb'][0]
    if not torch.is_tensor(depth):
        depth = torch.as_tensor(np.asarray(depth))
    depth = depth.to(dev)
    fh, fw = hw
    V = depth.shape[0]
    yy, xx = torch.meshgrid(torch.arange(fh, device=dev),
                            torch.arange(fw, device=dev), indexing='ij')
    px = ((xx.reshape(-1).float() + .5) * 512.0 / fw).long().clamp(max=511)
    py = ((yy.reshape(-1).float() + .5) * 512.0 / fh).long().clamp(max=511)
    z = devicez_to_camera_z(depth)[:, py, px]                          # [V,hw]
    u = ((xx.reshape(-1).float() + .5) * 512.0 / fw)[None].expand(V, -1)
    w = ((yy.reshape(-1).float() + .5) * 512.0 / fh)[None].expand(V, -1)
    sk, s_w2c, _ = _k_and_w2c(uav['satellite_camera'][0])
    sk, s_w2c = sk.to(dev), s_w2c.to(dev)
    su = torch.zeros(V, fh * fw, dtype=torch.long, device=dev)
    sv = torch.zeros(V, fh * fw, dtype=torch.long, device=dev)
    valid = torch.zeros(V, fh * fw, dtype=torch.bool, device=dev)
    for vi in range(V):
        k, _, c2w = _k_and_w2c(uav['uav_cameras'][0][vi])
        k, c2w = k.to(dev), c2w.to(dev)
        zv = z[vi]
        pc = torch.stack(((u[vi] - k[0, 2]) * zv / k[0, 0],
                          (w[vi] - k[1, 2]) * zv / k[1, 1], zv), -1)
        world = pc @ c2w[:3, :3].T + c2w[:3, 3]
        cam = world @ s_w2c[:3, :3].T + s_w2c[:3, 3]
        su[vi] = (sk[0, 0] * cam[:, 0] / cam[:, 2].clamp_min(1e-6)
                  + sk[0, 2]).long().clamp(0, 511)
        sv[vi] = (sk[1, 1] * cam[:, 1] / cam[:, 2].clamp_min(1e-6)
                  + sk[1, 2]).long().clamp(0, 511)
        valid[vi] = (zv > 0) & (cam[:, 2] > 0) & (world[:, 2] < 50000)
    m = (fh, fw)
    return su.reshape(V, *m), sv.reshape(V, *m), valid.reshape(V, *m)


class UEDualPreprocessor(DetDataPreprocessor):
    """DetDataPreprocessor with a UAV side-channel.

    The trainer passes {'inputs','data_samples','uav_batch'}; the UAV payload
    is moved to the compute device and stashed for the dual model, which
    fetches (and clears) it at the start of loss/predict. Single writer,
    single reader, immediately adjacent calls - asserted by fetch-and-clear.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_uav = None

    def forward(self, data, training=True):
        uav = None
        if isinstance(data, dict) and 'uav_batch' in data:
            uav = data.pop('uav_batch')
        out = super().forward(data, training)
        if uav is not None:
            device = out['inputs'].device
            moved = {}
            for k, v in uav.items():
                moved[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            self._last_uav = moved
        else:
            self._last_uav = None
        return out


# BaseModel.__init__ resolves nested data_preprocessor via the mmengine
# registry, so the preprocessor must live in both scopes.
MMENGINE_MODELS.register_module(module=UEDualPreprocessor)


def _k_and_w2c(pair):
    intr, extr = pair
    k = torch.tensor([[intr["fx"], 0., intr["cx"]], [0., intr["fy"], intr["cy"]], [0., 0., 1.]],
                     dtype=torch.float64)
    w2c64 = torch.tensor(extr["world_to_camera_cv"], dtype=torch.float64)
    c2w64 = torch.tensor(extr["camera_to_world_cv"], dtype=torch.float64)
    if not torch.allclose(w2c64 @ c2w64, torch.eye(4, dtype=torch.float64), atol=1e-6, rtol=0):
        raise RuntimeError("UAV/satellite camera C2W-W2C contract failed")
    return k.float(), w2c64.float(), c2w64.float()


@MODELS.register_module()
class RSPrompterAnchorRoIUAVHead(RSPrompterAnchorRoIPromptHead):
    """WARM-R1R: RoI head that adds zero-init per-box UAV descriptor deltas.

    The detector stashes a per-scene context (desc/views/cameras/zmean) on
    `.uav_owner_ctx` before loss/predict; _bbox_forward/_mask_forward sample
    per-box features via the detector's roi_uav_sampler and broadcast-add the
    zero-initialised adapter outputs to the RoI features. No ctx -> the
    standard forward, bit-identically (UAV-off parity).
    """

    def _bbox_forward(self, x, rois):
        # ERRATUM (arm E prereg 2026-10-19): R1R added the delta to the
        # informational bbox_feats dict entry AFTER super() had already
        # computed cls_score/bbox_pred from unmodified features - the bbox
        # adapter gradient was structurally zero (R1R verdict's bbox side
        # void; checkpoint evidence: sampler proj + mask adapter trained,
        # bbox adapter bitwise zero). The delta must land on the extractor
        # output BEFORE the head call.
        owner = getattr(self, "uav_owner", None)
        if os.environ.get('R1R_BYPASS') == '1':
            owner = None
        ctx = getattr(owner, "_uav_roi_ctx", None) if owner is not None else None
        if ctx is not None and rois.shape[0] > 0:
            feats = owner.roi_uav_sampler(rois, ctx["desc"], ctx["uav_k"],
                                          ctx["uav_c2w"], ctx["sat_k"],
                                          ctx["sat_w2c"], ctx["zmean"])
            delta = owner.roi_bbox_adapter(feats)
            bbox_feats = self.bbox_roi_extractor(
                x[: self.bbox_roi_extractor.num_inputs], rois)
            base_norm = bbox_feats.detach().float().norm().clamp_min(1e-6)
            bbox_feats = bbox_feats + delta[:, :, None, None]
            cls_score, bbox_pred = self.bbox_head(bbox_feats)
            if owner.training:
                owner.telemetry.setdefault('roi_bbox_delta_ratio', []).append(
                    float(delta.detach().norm() / base_norm))
            if os.environ.get('R1R_DEBUG'):
                print(f'[r1r-dbg] bbox rois={tuple(rois.shape)} feats finite={bool(torch.isfinite(feats).all())} '
                      f'feats_absmax={float(feats.abs().max())}', flush=True)
            return dict(cls_score=cls_score, bbox_pred=bbox_pred,
                        bbox_feats=bbox_feats)
        return super()._bbox_forward(x, rois)

    def _mask_forward(self, x, rois, image_embeddings=None,
                      image_positional_embeddings=None, high_res_features=None):
        owner = getattr(self, "uav_owner", None)
        if os.environ.get('R1R_BYPASS') == '1':
            owner = None
        ctx = getattr(owner, "_uav_roi_ctx", None) if owner is not None else None
        if ctx is not None and rois.shape[0] > 0:
            feats = owner.roi_uav_sampler(rois, ctx["desc"], ctx["uav_k"],
                                          ctx["uav_c2w"], ctx["sat_k"],
                                          ctx["sat_w2c"], ctx["zmean"])
            delta = owner.roi_mask_adapter(feats)
            mask_roi_feats = self.mask_roi_extractor(
                x[: self.mask_roi_extractor.num_inputs], rois)
            if os.environ.get('R1R_DEBUG'):
                print(f'[r1r-dbg] mask feats={tuple(feats.shape)} finite={bool(torch.isfinite(feats).all())} '
                      f'delta_absmax={float(delta.abs().max()) if delta.numel() else 0.0} '
                      f'mrf_finite={bool(torch.isfinite(mask_roi_feats).all())}', flush=True)
            mask_base_norm = mask_roi_feats.detach().float().norm().clamp_min(1e-6)
            mask_roi_feats = mask_roi_feats + delta[:, :, None, None]
            if owner.training:
                owner.telemetry.setdefault('roi_mask_delta_ratio', []).append(
                    float(delta.detach().norm() / mask_base_norm))
            head_out = self.mask_head(
                mask_roi_feats,
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings,
                roi_img_ids=rois[:, 0],
                high_res_features=high_res_features,
                boxes=rois[:, 1:],
                p2_feature=x[0],
                prompt_rois=rois,
                renderer_features=x,
            )
            return dict(mask_preds=head_out.mask_logits,
                        coarse_outputs=head_out.coarse_outputs,
                        tail_outputs=head_out.tail_outputs)
        return super()._mask_forward(
            x, rois,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            high_res_features=high_res_features)


@MODELS.register_module()
class RSPrompterAnchorDual(RSPrompterAnchor):
    """Sealed a3sgm + frozen UAV branch + zero-init gated P3 fusion.

    fusion: P3' = P3 + gate ⊙ conv1x1(surface_projected_uav_features)
    gate and conv are zero-initialised, so step 0 equals the single-stream
    forward bit-for-bit; any divergence is chosen by training itself.
    """

    def __init__(self, uav_encoder=None, bev_channels=262, sat_channels=512,
                 target_level=1, target_levels=None, uav_eval_off=False,
                 bev_stats=None, embed_fusion=False, embed_channels=256,
                 gate_init=0.0, bev_head_hidden=0, bev_head_stats=None,
                 bev_aux_weight=0.05, bev_desc_attention=False,
                 bev_attn_dim=128, bev_attn_geom_dim=64, bev_attn_heads=4,
                 height_edges_cfg=None, bev_attn_sat_query=False,
                 bev_head_source='bev', gate_trainable=True,
                 bev_stats_only=False, roi_uav_only=False,
                 roi_uav_bridge=False, bridge_decoder_ckpt=None,
                 bridge_aux_weight=0.05,
                 **kwargs):
        super().__init__(**kwargs)
        if uav_encoder is None:
            raise ValueError("dual arm requires a uav_encoder config")
        enc = dict(uav_encoder)
        self.uav_encoder_name = enc.pop('type')
        self.uav_encoder = MODELS.build(ConfigDict({'type': self.uav_encoder_name, **enc}))
        # with use_lora the encoder self-freezes the base and leaves LoRA
        # params trainable; a blanket freeze here would silently kill them
        self.uav_encoder_lora = bool(enc.get('use_lora', False))
        if not self.uav_encoder_lora:
            self.uav_encoder.requires_grad_(False)
        self.uav_encoder.eval()
        # gate=0 with DEFAULT proj init: step 0 equals the single-stream
        # forward (0 x finite = 0) while dL/dgate = upstream . proj(bev) != 0
        # (a doubly-zero init is an unescapable saddle - launch-review finding).
        # target_levels=[] (explicit empty list) means no pyramid fusion at
        # all - the EXP-0016 embed-only arm.
        levels = [int(t) for t in target_levels] if target_levels is not None else [int(target_level)]
        self.target_levels = levels
        # WARM-R1G (DUAL-WARMSTART-SCREEN-20261017): non-zero initial gate
        # participation. Default 0.0 keeps every earlier arm bit-identical;
        # 0.01 sits at ~2x the EXP-0015 final gate std (0.0055) so step-0
        # injection stays inside the k=2 gate-sweep envelope.
        self.gate_init = float(gate_init)
        if len(levels) == 1:
            # keep EXP-0010's attribute names so its checkpoints strict-load
            self.proj = nn.Conv2d(bev_channels, sat_channels, 1)
            self.gate = nn.Parameter(torch.full((sat_channels,), self.gate_init))
        elif len(levels) > 1:
            self.proj = nn.ModuleDict({str(t): nn.Conv2d(bev_channels, sat_channels, 1) for t in levels})
            self.gate = nn.ParameterDict({str(t): nn.Parameter(torch.full((sat_channels,), self.gate_init)) for t in levels})
        # EXP-0016 injection site: image_embeddings (the SAM2 mask-decoder
        # dense condition field, stride-16) instead of the PAFN pyramid.
        # Mutually exclusive with pyramid levels by prereg contract.
        self.embed_fusion = bool(embed_fusion)
        if self.embed_fusion:
            if levels:
                raise ValueError("embed_fusion arm must not also fuse pyramid levels (prereg single-site)")
            self.proj_e = nn.Conv2d(bev_channels, embed_channels, 1)
            self.gate_e = nn.Parameter(torch.full((embed_channels,), self.gate_init))
        # WARM-R1B (DUAL-WARMSTART-SCREEN-20261017): buildingness spatial gate
        # + auxiliary supervision. The head consumes the SAME BEV tensor as
        # the injection path but through its own frozen per-channel rescale
        # (0014-lineage harvest JSON, s_c = 1/max(rms_c, 0.05)); the proj path
        # stays untouched so the EXP-0015 recipe remains the single base.
        # Default bev_head_hidden=0 creates no attribute and no state_dict
        # key - every earlier arm is bit-identical (same pattern as bev_stats).
        self.bev_head_source = str(bev_head_source)
        if bev_head_hidden > 0:
            if len(levels) != 1 or self.embed_fusion:
                raise ValueError("bev_head arm is single-level pyramid fusion only (prereg)")
            if self.bev_head_source == 'bev':
                # R1B lineage: head consumes the raw BEV through the frozen
                # 0014-lineage per-channel rescale (behaviour unchanged)
                if bev_head_stats is None:
                    raise ValueError("bev_head_source='bev' requires bev_head_stats (0014 harvest JSON)")
                payload = json.loads(Path(bev_head_stats).read_text())
                scale = torch.tensor(payload['scale'], dtype=torch.float32)
                if tuple(scale.shape) != (bev_channels,):
                    raise ValueError(f"bev_head_stats scale has {scale.numel()} channels, expected {bev_channels}")
                self.register_buffer('bev_head_scale', scale.view(1, -1, 1, 1))
                head_in = bev_channels
            elif self.bev_head_source == 'attn':
                # WARM-R1Q: head input LOCKED to the attention output map -
                # supervision cannot leak through the stats channels, so the
                # aux objective forces scene-specific content into the
                # attention aggregate (GroupNorm'd, no rescale needed)
                if not bev_desc_attention:
                    raise ValueError("bev_head_source='attn' requires bev_desc_attention")
                head_in = int(bev_attn_dim)
            else:
                raise ValueError(f"unknown bev_head_source {bev_head_source}")
            self.bev_head = nn.Sequential(
                nn.Conv2d(head_in, int(bev_head_hidden), 3, padding=1, bias=False),
                nn.GroupNorm(8, int(bev_head_hidden)),
                nn.ReLU(inplace=True),
                nn.Conv2d(int(bev_head_hidden), int(bev_head_hidden), 3, padding=1, bias=False),
                nn.GroupNorm(8, int(bev_head_hidden)),
                nn.ReLU(inplace=True),
                nn.Conv2d(int(bev_head_hidden), 1, 1),
            )
            self.bev_aux_weight = float(bev_aux_weight)
            self._bev_aux_logits = None
        # WARM-R1C (PIXEL-CLS 2026-10-17 A-PASS): descriptors travel through a
        # CVT-style attention pathway instead of the rasterizer's cross-view
        # averaging; the 6 stats channels keep the exact backprojection. The
        # constructor bev_channels then counts attn_dim + 6 (proj input); the
        # rasterizer still runs at its native 262 for the stats slice. Default
        # False creates no attribute - earlier arms bit-identical.
        # bin count derived early (the attention channel check needs it)
        _edges = list(height_edges_cfg) if height_edges_cfg is not None else [-1e9, 1e9]
        self._n_height_bins = len(_edges) - 1
        if self._n_height_bins < 1:
            raise ValueError("height_edges_cfg needs >= 2 ascending edges")
        self.bev_stats_only = bool(bev_stats_only)
        if self.bev_stats_only and bev_desc_attention:
            raise ValueError('bev_stats_only excludes bev_desc_attention (terminal clause)')
        self.bev_desc_attention = bool(bev_desc_attention)
        if self.bev_desc_attention:
            self.bev_attn = BEVDescAttention(
                desc_dim=256, out_dim=int(bev_attn_dim),
                geom_dim=int(bev_attn_geom_dim), heads=int(bev_attn_heads),
                sat_dim=512 if bev_attn_sat_query else 0)
            self.bev_attn_sat_query = bool(bev_attn_sat_query)
            self._last_attn_out = None
            expected = int(bev_attn_dim) + 6 * self._n_height_bins
            if bev_channels != expected:
                raise ValueError(
                    f"bev_desc_attention expects bev_channels={expected} "
                    f"(attn_dim + 6 stats x {self._n_height_bins} bins), "
                    f"got {bev_channels}")
        # WARM-R1F: fixed (non-learnable) transport gate - the gate is pinned
        # at gate_init and the optimizer never sees it (proj still
        # self-calibrates its output norm). Default True keeps learnability.
        if not gate_trainable:
            self.gate.requires_grad_(False)
            if getattr(self, 'gate_e', None) is not None:
                self.gate_e.requires_grad_(False)
        # WARM-R1R: RoI-level conditioning only - the pyramid injection is off,
        # the BEV pipeline merely supplies the zmean surface; per-box features
        # enter through zero-init adapters in RSPrompterAnchorRoIUAVHead.
        self.roi_uav_only = bool(roi_uav_only)
        # WARM-R1V (arm E, D-v2 supervised bridge): payload = the offline
        # pretrained decoder's sigmoid maps, sampled by the same geometry
        # and consumed through zero-init adapters. The decoder is FROZEN -
        # trainable set = adapters + UAV LoRA; a decode-consistency aux
        # (same objective as pretraining, weight bridge_aux_weight) pins the
        # LoRA supply against drift away from the verified decode quality.
        self.roi_uav_bridge = bool(roi_uav_bridge)
        self.bridge_aux_weight = float(bridge_aux_weight)
        if self.roi_uav_only:
            if self.roi_uav_bridge:
                if not bridge_decoder_ckpt:
                    raise ValueError("roi_uav_bridge requires bridge_decoder_ckpt")
                self.roi_bridge_decoder = EviDecoder()
                _dec = torch.load(bridge_decoder_ckpt, map_location='cpu',
                                  weights_only=False)
                self.roi_bridge_decoder.load_state_dict(_dec['state_dict'])
                self.roi_bridge_decoder.requires_grad_(False)
                self.roi_bridge_decoder.eval()
                self._bridge_aux = None
                self.roi_uav_sampler = RoIUAVSampler(mode='evi')
                self.roi_bbox_adapter = nn.Linear(7 * 7 * 2, sat_channels)
                nn.init.zeros_(self.roi_bbox_adapter.weight)
                nn.init.zeros_(self.roi_bbox_adapter.bias)
                mask_ch = 256
                self.roi_mask_adapter = nn.Linear(7 * 7 * 2, mask_ch)
                nn.init.zeros_(self.roi_mask_adapter.weight)
                nn.init.zeros_(self.roi_mask_adapter.bias)
            else:
                self.roi_uav_sampler = RoIUAVSampler(desc_dim=256, out_dim=256, grid=7)
                self.roi_bbox_adapter = nn.Linear(256, sat_channels)
                nn.init.zeros_(self.roi_bbox_adapter.weight)
                nn.init.zeros_(self.roi_bbox_adapter.bias)
                # mask extractor channels resolved lazily at first wiring below
                mask_ch = 256
                self.roi_mask_adapter = nn.Linear(256, mask_ch)
                nn.init.zeros_(self.roi_mask_adapter.weight)
                nn.init.zeros_(self.roi_mask_adapter.bias)
            self._uav_roi_ctx = None
        self.uav_eval_off = bool(uav_eval_off) or os.environ.get('UE_UAV_EVAL_OFF') == '1'
        # EXP-0014: optional frozen per-channel rescaling of the BEV before the
        # projection (identity: ue-whu-a3sgm-512-dualnorm-v1). DUAL-DELTA-DECOMP
        # showed the raw cm-scale channels (zmean/zres) dominate ||proj(bev)||
        # by 5 orders of magnitude and the 256 descriptor channels never
        # participate numerically; the scale vector is harvested offline from
        # the full train split (scripts/ue/harvest_bev_stats_exp0014.py,
        # s_c = 1/max(rms_c, 0.05)) and registered as a buffer so it never
        # trains. None (all pre-EXP-0014 arms) keeps this path bit-identical
        # (no attribute, no state_dict key); the use site probes with getattr.
        if bev_stats is not None:
            payload = json.loads(Path(bev_stats).read_text())
            scale = torch.tensor(payload['scale'], dtype=torch.float32)
            if tuple(scale.shape) != (bev_channels,):
                raise ValueError(f"bev_stats scale has {scale.numel()} channels, expected {bev_channels}")
            self.register_buffer('bev_scale', scale.view(1, -1, 1, 1))
        # WARM-R1CH: multi-height-bin stats widening. Default None keeps the
        # legacy single-surface bin [-1e9, 1e9] bit-identically; a config list
        # (cm, ascending) yields K = len-1 bins whose 6 stats channels each are
        # flattened (bin-major) into the BEV channels (requires the attention
        # descriptor pathway - single-variable stack on the leading recipe).
        if self._n_height_bins > 1 and not self.bev_desc_attention:
            raise ValueError("multi-bin stats requires bev_desc_attention (R1CH arm scope)")
        self.register_buffer('height_edges', torch.tensor(_edges, dtype=torch.float32))
        if self.roi_uav_only:
            # object.__setattr__ bypasses nn.Module child registration - a
            # registered back-reference would recurse infinitely in .to()/.cuda()
            object.__setattr__(self.roi_head, 'uav_owner', self)
            real_mask_ch = self.mask_roi_extractor.out_channels \
                if hasattr(self, 'mask_roi_extractor') else None
            # the detector-level mask extractor lives on the roi head
            real_mask_ch = self.roi_head.mask_roi_extractor.out_channels
            if real_mask_ch != self.roi_mask_adapter.out_features:
                in_feat = 7 * 7 * 2 if self.roi_uav_bridge else 256
                del self.roi_mask_adapter
                self.roi_mask_adapter = nn.Linear(in_feat, real_mask_ch)
                nn.init.zeros_(self.roi_mask_adapter.weight)
                nn.init.zeros_(self.roi_mask_adapter.bias)
        self.telemetry = {'gate_mean': [], 'gate_std': [], 'gate_absmax': [],
                          'delta_p3_ratio': [], 'bev_coverage': []}
        self._uav_pending = None

    def train(self, mode=True):
        super().train(mode)
        if self.uav_encoder_lora:
            self.uav_encoder.train(mode)  # LoRA dropout active in training
        else:
            self.uav_encoder.eval()
        if getattr(self, 'roi_bridge_decoder', None) is not None:
            self.roi_bridge_decoder.eval()   # frozen supply: no dropout/BN drift
        return self

    def _fetch_uav(self):
        dp = self.data_preprocessor
        if hasattr(dp, '_last_uav') and getattr(dp, '_last_uav', None) is not None:
            self._uav_pending = dp._last_uav
            dp._last_uav = None

    def _stash_roi_ctx(self, uav, hw):
        rgb = uav['drone_images']
        mean = rgb.new_tensor(_UAV_MEAN)[:, None, None]
        std = rgb.new_tensor(_UAV_STD)[:, None, None]
        if self.uav_encoder_lora:
            _, fpn = self.uav_encoder((rgb[0] - mean) / std)
            desc = max(tuple(fpn[-4:]), key=lambda f: f.shape[-2] * f.shape[-1]).float()
        else:
            with torch.no_grad():
                _, fpn = self.uav_encoder((rgb[0] - mean) / std)
                desc = max(tuple(fpn[-4:]), key=lambda f: f.shape[-2] * f.shape[-1]).float()
        cams = uav['uav_cameras'][0]
        dev = desc.device
        uk, _, uc = zip(*(_k_and_w2c(p) for p in cams))
        sk, swc, _ = _k_and_w2c(uav['satellite_camera'][0])
        uk = torch.stack([k.to(dev) for k in uk])
        uc = torch.stack([c.to(dev) for c in uc])
        sk, swc = sk.to(dev), swc.to(dev)
        bev, _ = rasterize_surfacebev(
            desc.detach(), uav['devicez_rgb'][0], uk, uc, sk, swc,
            self.height_edges.to(dev), hw, with_grad=False)
        # zmean is stats channel 256 on the single surface bin; heights in cm
        zmean = bev[0, 256]
        if self.roi_uav_bridge:
            # supply = decoded answers, never raw descriptors. Grad flows
            # through the FROZEN decoder into the LoRA supply (pinned by the
            # decode-consistency aux); no-grad encoder keeps it detached.
            evi_logits = self.roi_bridge_decoder(desc)
            payload = torch.sigmoid(evi_logits)
            if self.training:
                su, sv, vvalid = _view_label_grid(uav, tuple(desc.shape[-2:]), dev)
                self._bridge_aux = dict(logits=evi_logits, su=su, sv=sv,
                                        valid=vvalid)
            self._uav_roi_ctx = dict(desc=payload, uav_k=uk, uav_c2w=uc,
                                     sat_k=sk, sat_w2c=swc, zmean=zmean)
            return
        if os.environ.get('R1R_DEBUG'):
            print(f'[r1r-dbg] ctx desc_finite={bool(torch.isfinite(desc).all())} '
                  f'zmean_finite={bool(torch.isfinite(zmean).all())} '
                  f'zmean_absmax={float(zmean.abs().max())} '
                  f'uk_finite={bool(torch.isfinite(uk).all())} '
                  f'uc_finite={bool(torch.isfinite(uc).all())} '
                  f'sk_finite={bool(torch.isfinite(sk).all())} '
                  f'swc_finite={bool(torch.isfinite(swc).all())}', flush=True)
        self._uav_roi_ctx = dict(desc=desc, uav_k=uk, uav_c2w=uc,
                                 sat_k=sk, sat_w2c=swc, zmean=zmean)

    def _uav_bev(self, uav, hw, sat_feat=None):
        rgb = uav['drone_images']
        if rgb.dim() != 5 or rgb.shape[0] != 1:
            raise RuntimeError(f"dual arm expects per-GPU batch 1, got {tuple(rgb.shape)}")
        mean = rgb.new_tensor(_UAV_MEAN)[:, None, None]
        std = rgb.new_tensor(_UAV_STD)[:, None, None]
        if self.uav_encoder_lora:
            _, fpn = self.uav_encoder((rgb[0] - mean) / std)
            desc = max(tuple(fpn[-4:]), key=lambda f: f.shape[-2] * f.shape[-1]).float()
        else:
            with torch.no_grad():
                _, fpn = self.uav_encoder((rgb[0] - mean) / std)
                desc = max(tuple(fpn[-4:]), key=lambda f: f.shape[-2] * f.shape[-1]).float()
        cams = uav['uav_cameras'][0]  # collate keeps a per-sample batch layer
        dev = desc.device
        uk, _, uc = zip(*(_k_and_w2c(p) for p in cams))
        sk, swc, _ = _k_and_w2c(uav['satellite_camera'][0])
        uk = tuple(k.to(dev) for k in uk); uc = tuple(c.to(dev) for c in uc)
        sk, swc = sk.to(dev), swc.to(dev)
        # EXP-0023 terminal clause (handoff 2026-10-18): hard-cut the descriptor
        # pathway - the BEV carries ONLY the 6 exact-backprojection stats
        # channels (zmean/zstd/vcount/zres/flags). Every content-rescue of the
        # descriptors failed its gate; the method converges to "UAV depth
        # geometry prior injection".
        if self.bev_stats_only:
            bev, stats = rasterize_surfacebev(
                desc.detach(), uav['devicez_rgb'][0], torch.stack(uk), torch.stack(uc),
                sk, swc, self.height_edges.to(desc.device), hw, with_grad=False)
            coverage = float(stats.get('covered_cells', 0)) / float(hw[0] * hw[1])
            return bev[0, 256:262], coverage
        bev, stats = rasterize_surfacebev(
            desc, uav['devicez_rgb'][0], torch.stack(uk), torch.stack(uc),
            sk, swc, self.height_edges.to(desc.device), hw,
            with_grad=self.uav_encoder_lora)
        if self.bev_desc_attention:
            # PIXEL-CLS A-PASS: replace the rasterizer's averaged descriptor
            # block with attention-aggregated descriptors; keep the exact
            # backprojected stats channels (slice [256:262] of the raster).
            # WARM-R1CH: with K height bins the per-bin stats blocks are
            # flattened bin-major into 6K channels.
            attn_desc = self.bev_attn(desc, torch.stack(uk), torch.stack(uc),
                                      sk, swc, hw, sat_feat=sat_feat)
            self._last_attn_out = attn_desc
            stats_blk = bev[:, 256:262]          # must NOT shadow the raster
            if bev.shape[0] > 1:                 # stats dict used 3 lines below
                stats_blk = stats_blk.reshape(1, -1, *stats_blk.shape[-2:])
            bev = torch.cat((attn_desc[None], stats_blk), dim=1)
        if getattr(self, 'bev_scale', None) is not None:
            bev = bev * self.bev_scale.to(bev.device)
        coverage = float(stats.get('covered_cells', 0)) / float(hw[0] * hw[1])
        return bev[0], coverage

    def extract_feat(self, batch_inputs: torch.Tensor):
        out = super().extract_feat(batch_inputs)
        x = list(out[0])
        image_embeddings = out[1]
        uav = self._uav_pending
        self._uav_pending = None
        if uav is not None and not (not self.training and self.uav_eval_off) \
                and self.roi_uav_only:
            # WARM-R1R: no pyramid injection; build the per-scene RoI context
            # (descriptors + exact cameras + the zmean surface at 128x128)
            with torch.autocast('cuda', enabled=False):
                self._stash_roi_ctx(uav, tuple(x[0].shape[-2:]))
            return tuple(x), image_embeddings, *out[2:]
        if uav is not None and not (not self.training and self.uav_eval_off):
            with torch.autocast('cuda', enabled=False):
                for t in self.target_levels:
                    base = x[t]
                    single = len(self.target_levels) == 1
                    proj_t = self.proj if single else self.proj[str(t)]
                    gate_t = self.gate if single else self.gate[str(t)]
                    _sat = base.detach() if getattr(self, 'bev_attn_sat_query', False) else None
                    bev, coverage = self._uav_bev(uav, tuple(base.shape[-2:]), sat_feat=_sat)
                    delta = proj_t(bev.unsqueeze(0).float())
                    delta = (gate_t.view(1, -1, 1, 1) * delta)
                    if getattr(self, 'bev_head', None) is not None:
                        if self.bev_head_source == 'attn':
                            b_in = self._last_attn_out.unsqueeze(0).float()
                        else:
                            b_in = bev.unsqueeze(0).float() * self.bev_head_scale.to(bev.device)
                        b_logits = self.bev_head(b_in)
                        delta = delta * (1.0 + torch.sigmoid(b_logits))
                        if self.training:
                            # kept graph-attached: the aux loss (loss()) trains
                            # bev_head AND feeds the UAV LoRA supply-side
                            self._bev_aux_logits = b_logits
                    delta = delta.to(base.dtype)
                    x[t] = base + delta
                    if self.training:
                        g = gate_t.detach()
                        tag = '' if single else f'_L{t}'
                        self.telemetry.setdefault('gate_mean' + tag, []).append(float(g.mean()))
                        self.telemetry.setdefault('gate_std' + tag, []).append(float(g.std()))
                        self.telemetry.setdefault('gate_absmax' + tag, []).append(float(g.abs().max()))
                        self.telemetry.setdefault('delta_p3_ratio' + tag, []).append(
                            float(delta.detach().float().norm()
                                  / base.detach().float().norm().clamp_min(1e-6)))
                        self.telemetry.setdefault('bev_coverage' + tag, []).append(coverage)
                if self.embed_fusion:
                    base_e = image_embeddings
                    bev, coverage = self._uav_bev(uav, tuple(base_e.shape[-2:]))
                    delta = self.proj_e(bev.unsqueeze(0).float())
                    delta = (self.gate_e.view(1, -1, 1, 1) * delta).to(base_e.dtype)
                    image_embeddings = base_e + delta
                    if self.training:
                        g = self.gate_e.detach()
                        self.telemetry.setdefault('gate_mean_E', []).append(float(g.mean()))
                        self.telemetry.setdefault('gate_std_E', []).append(float(g.std()))
                        self.telemetry.setdefault('gate_absmax_E', []).append(float(g.abs().max()))
                        self.telemetry.setdefault('delta_p3_ratio_E', []).append(
                            float(delta.detach().float().norm()
                                  / base_e.detach().float().norm().clamp_min(1e-6)))
                        self.telemetry.setdefault('bev_coverage_E', []).append(coverage)
        return tuple(x), image_embeddings, *out[2:]

    def loss(self, batch_inputs, batch_data_samples, **kwargs):
        self._fetch_uav()
        if self.roi_uav_only:
            self._uav_roi_ctx = None
        if self.roi_uav_bridge:
            self._bridge_aux = None
        if getattr(self, 'bev_head', None) is not None:
            self._bev_aux_logits = None
        losses = super().loss(batch_inputs, batch_data_samples, **kwargs)
        # WARM-R1B auxiliary supervision: BCE(buildingness logits, GT union
        # footprint downsampled to the BEV grid). Per-GPU batch is asserted 1
        # upstream, so a single scene's union is the target.
        logits = getattr(self, '_bev_aux_logits', None)
        if getattr(self, 'bev_head', None) is not None and logits is not None:
            self._bev_aux_logits = None
            target = torch.zeros(logits.shape[-2:], device=logits.device, dtype=torch.float32)
            for sample in batch_data_samples:
                masks = getattr(getattr(sample, 'gt_instances', None), 'masks', None)
                if masks is None or len(masks) == 0:
                    continue
                union = torch.from_numpy(masks.to_ndarray().astype(bool).any(0)).float()
                union = union[None, None].to(logits.device)
                target = torch.maximum(
                    target,
                    torch.nn.functional.interpolate(union, size=logits.shape[-2:], mode='area')[0, 0],
                )
            bce = F.binary_cross_entropy_with_logits(logits[0, 0], target)
            if self.bev_head_source == 'attn':
                p = torch.sigmoid(logits[0, 0])
                inter = (p * target).sum()
                dice = 1.0 - (2.0 * inter + 1.0) / (p.sum() + target.sum() + 1.0)
                losses['loss_bev_aux'] = self.bev_aux_weight * (bce + dice)
            else:
                losses['loss_bev_aux'] = self.bev_aux_weight * bce
        # WARM-R1V decode-consistency aux (arm E prereg): pin the LoRA supply
        # to the verified decode quality - SAME objective as the offline
        # pretraining (per-view BCE+Dice on both channels, valid cells only),
        # labels from the stashed PIXEL-CLS geometry grid + this batch's GT.
        aux = getattr(self, '_bridge_aux', None)
        if self.roi_uav_bridge and aux is not None:
            self._bridge_aux = None
            dev_aux = aux['logits'].device
            gt_union = torch.zeros((512, 512), dtype=torch.bool, device=dev_aux)
            for sample in batch_data_samples:
                masks = getattr(getattr(sample, 'gt_instances', None), 'masks', None)
                if masks is None or len(masks) == 0:
                    continue
                gt_union |= torch.from_numpy(
                    masks.to_ndarray().astype(bool).any(0)).to(dev_aux)
            evi_lab = gt_union[aux['sv'], aux['su']] & aux['valid']     # [V,h,w]
            e = evi_lab[:, None].float()
            dil = F.max_pool2d(e, 3, stride=1, padding=1)[:, 0] > 0.5
            ero = -F.max_pool2d(-e, 3, stride=1, padding=1)[:, 0] > 0.5
            bnd_lab = dil & ~ero & aux['valid']
            total = 0.0
            for ch, lab in ((0, evi_lab), (1, bnd_lab)):
                m = aux['valid']
                lg = aux['logits'][:, ch][m]
                yy = lab[m].float()
                bce = F.binary_cross_entropy_with_logits(lg, yy)
                p = torch.sigmoid(lg)
                dice = 1.0 - (2.0 * (p * yy).sum() + 1.0) / (p.sum() + yy.sum() + 1.0)
                total = total + bce + dice
            losses['loss_bridge_aux'] = self.bridge_aux_weight * total
        # adapter-liveness telemetry (arm E erratum clause): a bitwise-zero
        # bbox adapter at screen end is INERT-WIRING-CHECK, not a verdict
        if self.roi_uav_only and self.training:
            self.telemetry.setdefault('roi_bbox_adapter_absmax', []).append(
                float(self.roi_bbox_adapter.weight.abs().max()))
            self.telemetry.setdefault('roi_mask_adapter_absmax', []).append(
                float(self.roi_mask_adapter.weight.abs().max()))
        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale=True, **kwargs):
        self._fetch_uav()
        if self.roi_uav_only:
            self._uav_roi_ctx = None
        if self.roi_uav_bridge:
            self._bridge_aux = None
        if getattr(self, 'bev_head', None) is not None:
            self._bev_aux_logits = None
        return super().predict(batch_inputs, batch_data_samples, rescale=rescale, **kwargs)

    def drain_telemetry(self):
        out = {k: (float(sum(v) / len(v)) if v else 0.0)
               for k, v in self.telemetry.items() if v}
        # count steps from whatever per-level gate key actually recorded
        out['gate_steps'] = sum(len(v) for k, v in self.telemetry.items()
                                if k.startswith('gate_absmax'))
        out['roi_steps'] = sum(len(v) for k, v in self.telemetry.items()
                               if k.startswith('roi_'))
        for k in self.telemetry:
            self.telemetry[k] = []
        return out
