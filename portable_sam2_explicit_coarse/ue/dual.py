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
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.config import ConfigDict
from rsprompter.detection.anchor_detector import RSPrompterAnchor
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
                 bev_stats_only=False,
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
        self.telemetry = {'gate_mean': [], 'gate_std': [], 'gate_absmax': [],
                          'delta_p3_ratio': [], 'bev_coverage': []}
        self._uav_pending = None

    def train(self, mode=True):
        super().train(mode)
        if self.uav_encoder_lora:
            self.uav_encoder.train(mode)  # LoRA dropout active in training
        else:
            self.uav_encoder.eval()
        return self

    def _fetch_uav(self):
        dp = self.data_preprocessor
        if hasattr(dp, '_last_uav') and getattr(dp, '_last_uav', None) is not None:
            self._uav_pending = dp._last_uav
            dp._last_uav = None

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
        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale=True, **kwargs):
        self._fetch_uav()
        if getattr(self, 'bev_head', None) is not None:
            self._bev_aux_logits = None
        return super().predict(batch_inputs, batch_data_samples, rescale=rescale, **kwargs)

    def drain_telemetry(self):
        out = {k: (float(sum(v) / len(v)) if v else 0.0)
               for k, v in self.telemetry.items() if v}
        # count steps from whatever per-level gate key actually recorded
        out['gate_steps'] = sum(len(v) for k, v in self.telemetry.items()
                                if k.startswith('gate_absmax'))
        for k in self.telemetry:
            self.telemetry[k] = []
        return out
