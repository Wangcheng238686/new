"""Perimeter dual-stream adapter for the sealed a3sgm baseline.

EXP-0008 arm: frozen UAV encoder + world-geometry projection of UAV features
onto the satellite P3 grid (single surface bin via the SurfaceBEV rasterizer)
+ zero-initialised per-channel gate. At initialisation the model is exactly
the single-stream architecture (gate=0 identity); telemetry records whether
training opens the UAV tap. The sealed rsprompter packages are untouched.
"""
from __future__ import annotations
import os
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
                 target_level=1, uav_eval_off=False, **kwargs):
        super().__init__(**kwargs)
        if uav_encoder is None:
            raise ValueError("dual arm requires a uav_encoder config")
        enc = dict(uav_encoder)
        self.uav_encoder_name = enc.pop('type')
        self.uav_encoder = MODELS.build(ConfigDict({'type': self.uav_encoder_name, **enc}))
        self.uav_encoder.eval()
        self.uav_encoder.requires_grad_(False)
        self.proj = nn.Conv2d(bev_channels, sat_channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.gate = nn.Parameter(torch.zeros(sat_channels))
        self.target_level = int(target_level)
        self.uav_eval_off = bool(uav_eval_off) or os.environ.get('UE_UAV_EVAL_OFF') == '1'
        self.register_buffer('height_edges', torch.tensor([-1e9, 1e9]))
        self.telemetry = {'gate_mean': [], 'gate_std': [], 'gate_absmax': [],
                          'delta_p3_ratio': [], 'bev_coverage': []}
        self._uav_pending = None

    def train(self, mode=True):
        super().train(mode)
        self.uav_encoder.eval()
        return self

    def _fetch_uav(self):
        dp = self.data_preprocessor
        if hasattr(dp, '_last_uav') and getattr(dp, '_last_uav', None) is not None:
            self._uav_pending = dp._last_uav
            dp._last_uav = None

    def _uav_bev(self, uav, hw):
        rgb = uav['drone_images']
        if rgb.dim() != 5 or rgb.shape[0] != 1:
            raise RuntimeError(f"dual arm expects per-GPU batch 1, got {tuple(rgb.shape)}")
        mean = rgb.new_tensor(_UAV_MEAN)[:, None, None]
        std = rgb.new_tensor(_UAV_STD)[:, None, None]
        with torch.no_grad():
            _, fpn = self.uav_encoder((rgb[0] - mean) / std)
            desc = max(tuple(fpn[-4:]), key=lambda f: f.shape[-2] * f.shape[-1]).float()
        cams = uav['uav_cameras'][0]  # collate keeps a per-sample batch layer
        dev = desc.device
        uk, _, uc = zip(*(_k_and_w2c(p) for p in cams))
        sk, swc, _ = _k_and_w2c(uav['satellite_camera'][0])
        uk = tuple(k.to(dev) for k in uk); uc = tuple(c.to(dev) for c in uc)
        sk, swc = sk.to(dev), swc.to(dev)
        bev, stats = rasterize_surfacebev(
            desc, uav['devicez_rgb'][0], torch.stack(uk), torch.stack(uc),
            sk, swc, self.height_edges.to(desc.device), hw)
        coverage = float(stats.get('covered_cells', 0)) / float(hw[0] * hw[1])
        return bev[0], coverage

    def extract_feat(self, batch_inputs: torch.Tensor):
        out = super().extract_feat(batch_inputs)
        x = list(out[0])
        uav = self._uav_pending
        self._uav_pending = None
        if uav is not None and not (not self.training and self.uav_eval_off):
            bev, coverage = self._uav_bev(uav, tuple(x[self.target_level].shape[-2:]))
            delta = self.proj(bev.unsqueeze(0))
            base = x[self.target_level]
            x[self.target_level] = base + self.gate.view(1, -1, 1, 1) * delta
            if self.training:
                g = self.gate.detach()
                self.telemetry['gate_mean'].append(float(g.mean()))
                self.telemetry['gate_std'].append(float(g.std()))
                self.telemetry['gate_absmax'].append(float(g.abs().max()))
                self.telemetry['delta_p3_ratio'].append(
                    float((self.gate.view(1, -1, 1, 1) * delta).detach().norm()
                          / base.detach().norm().clamp_min(1e-6)))
                self.telemetry['bev_coverage'].append(coverage)
        return tuple(x), *out[1:]

    def loss(self, batch_inputs, batch_data_samples, **kwargs):
        self._fetch_uav()
        return super().loss(batch_inputs, batch_data_samples, **kwargs)

    def predict(self, batch_inputs, batch_data_samples, rescale=True, **kwargs):
        self._fetch_uav()
        return super().predict(batch_inputs, batch_data_samples, rescale=rescale, **kwargs)

    def drain_telemetry(self):
        out = {k: (float(sum(v) / len(v)) if v else 0.0) for k, v in self.telemetry.items()}
        out['gate_steps'] = len(self.telemetry['gate_mean'])
        for k in self.telemetry:
            self.telemetry[k] = []
        return out
