"""Canonical source slots with explicit camera subsets; legacy sources unchanged."""
import numpy as np
import torch
from torch.nn import functional as F
from .local_adapter_probe import YAWS
from .geometry_inputs import candidate_arrays, native_arrays
from .attention_ladder import sample_source_continuous
from .ubr_native_integration import NativeEvidenceG

PRESETS = {1: (0,), 3: (0, 144, 288), 5: YAWS}


def canonical_views(view_ids):
    values = tuple(view_ids)
    if not values or any(type(v) is not int or v not in YAWS for v in values) or len(set(values)) != len(values):
        raise ValueError('Specify unique integer UAV yaw IDs from '+str(YAWS))
    return tuple(v for v in YAWS if v in values)


def subset_key(view_ids):
    return 'views_'+'_'.join('%03d' % v for v in canonical_views(view_ids))


def subset_arrays(cameras, depths, view_ids):
    """Recompute BEFORE top8/nearest-surface selection; never filter old caches.

    depths is canonical [5,512,512] camera-Z metres. Removed slots are replaced
    with NaNs before ANY geometric operation. Counts retain fixed /5 scaling.
    """
    views = canonical_views(view_ids)
    if depths.shape != (5,512,512):
        raise ValueError('Canonical five depth slots required')
    selected = np.full((5,512,512), np.nan, np.float64)
    for yaw in views:
        slot = YAWS.index(yaw)
        selected[slot] = depths[slot]
    index, geom, anchor = candidate_arrays(cameras, selected)
    fields, support = native_arrays(cameras, selected)
    return dict(index=index, geom=geom, anchor=anchor, fields=fields, support=support)


def check_index_views(index, view_ids):
    views = canonical_views(view_ids)
    valid = index >= 0
    slots = index[valid] // 16384
    allowed = torch.tensor([YAWS.index(v) for v in views], device=index.device, dtype=slots.dtype)
    if (index < -1).any() or (index >= 5*16384).any() or (slots.numel() and not (slots[:,None] == allowed).any(1).all()):
        raise ValueError('Candidate refers to a removed or invalid source slot')


def source_tokens(adapter, rgb, view_ids):
    """Encode selected views only, preserving five canonical token slot IDs.

    Absent maps are literal zeros and are never passed through the encoder.
    Five-view order/math equals the established G/detail reader.
    """
    views = canonical_views(view_ids)
    if rgb.shape != (5,3,512,512):
        raise ValueError('RGB must use canonical five slots')
    maps = {}
    for yaw in views:
        slot = YAWS.index(yaw)
        image = rgb[slot:slot+1]
        _, pyramid = adapter.rgb_encoder((image-adapter.rgb_mean)/adapter.rgb_std)
        maps[slot] = adapter.rgb_project(pyramid[0])
    reference = next(iter(maps.values()))
    features = torch.cat([maps[i] if i in maps else torch.zeros_like(reference) for i in range(5)])
    return F.layer_norm(features.permute(0,2,3,1).reshape(-1,32), (32,))


class SubsetNativeEvidenceG(NativeEvidenceG):
    """Existing G architecture/hooks/losses; only source-reader views differ."""
    def encode(self, geometry):
        self.native_geometry = geometry
        self.native_aligned_rgb = self.native_rgb_valid = None
        if geometry.shape != (1,5,512,512) or self.candidate_geometry is None:
            raise ValueError('Require rebuilt geometry5 and candidate geometry')
        b = self.b(self.a(geometry)); c = self.c(b); d = self.d(c)
        u = self.up(torch.cat((b, F.interpolate(c,(128,128),mode='bilinear',align_corners=False),
                              F.interpolate(d,(128,128),mode='bilinear',align_corners=False)),1))
        rgb, index, anchor = self.rgb_context
        check_index_views(index, self.view_ids)
        selected = sample_source_continuous(source_tokens(self,rgb,self.view_ids),index,anchor)
        support = F.max_pool2d(geometry[:,4:5],4,4) > 0
        self.read_context = (u, selected, index >= 0, self.candidate_geometry, support)
        self.context = (u,support)
        self.outputs = None


def configure_upstream_views(adapter, view_ids):
    views = canonical_views(view_ids)
    if not isinstance(adapter, NativeEvidenceG) or adapter.variant != 'detail':
        raise ValueError('Subset wiring requires established native G/detail')
    # No new parameters or hooks; original forward loss/candidate consumers stay.
    adapter.__class__ = SubsetNativeEvidenceG
    adapter.view_ids = views
    return adapter
