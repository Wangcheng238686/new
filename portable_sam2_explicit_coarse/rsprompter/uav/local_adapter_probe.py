"""Experimental frozen-base RoI-local UAV adapter; no production registration.

Geometry is measured camera-Z at sampled source pixels, not averaged depth or BEV.
The adapter is attached only by scripts/probes/ue_v1_adapter.py.
"""
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

YAWS = (0, 72, 144, 216, 288)
GEOMETRY_VERSION = 'sat128_source128_depth2_k8_v1'


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def camera_arrays(c):
    a = c['intrinsics']
    k = np.array([[a['fx'], 0, a['cx']], [0, a['fy'], a['cy']], [0, 0, 1.]])
    e, t = (np.array(c[key], dtype=np.float64) for key in ('world_to_camera_cv', 'camera_to_world_cv'))
    e[:3, 3] /= 100
    t[:3, 3] /= 100
    assert a['width'] == a['height'] == 512
    assert np.allclose(e @ t, np.eye(4), atol=1e-6)
    return k, e, t


def build_candidates(folder):
    c = json.loads((folder/'cameras.json').read_text())
    assert c['world_units'] == 'cm' and c['camera_axes'] == 'x_right_y_down_z_forward'
    sk, se, _ = camera_arrays(c['satellite'])
    yy, xx = np.mgrid[1:512:2, 1:512:2]
    xx, yy = xx.ravel(), yy.ravel()
    cells, ids, depths, metas, views = [], [], [], [], []
    hashes = {'cameras.json': file_sha(folder/'cameras.json')}
    max_roundtrip = 0.
    for view, yaw in enumerate(YAWS):
        name = f'uav/yaw_{yaw:03d}/depth_camera_z.exr'
        hashes[name] = file_sha(folder/name)
        raw = cv2.imread(str(folder/name), cv2.IMREAD_UNCHANGED)
        assert raw is not None and raw.shape[:2] == (512, 512)
        z = raw[yy, xx, 2].astype(np.float64)/100.  # EXR R / camera-Z / cm -> m
        good = np.isfinite(z) & (z > 0) & (z < 10000)
        u, v, z = xx[good], yy[good], z[good]
        k, e, t = camera_arrays(c['uav'][f'yaw_{yaw:03d}'])
        cam = np.stack(((u-k[0,2])*z/k[0,0], (v-k[1,2])*z/k[1,1], z), 1)
        world = cam @ t[:3,:3].T + t[:3,3]
        replay = world @ e[:3,:3].T + e[:3,3]
        replay_uv = replay @ k.T
        max_roundtrip = max(max_roundtrip, float(np.abs(replay_uv[:,:2]/replay_uv[:,2:] - np.stack((u,v),1)).max()))
        sat = world @ se[:3,:3].T + se[:3,3]
        pix = sat @ sk.T
        uv = pix[:,:2]/pix[:,2:]
        ok = (sat[:,2] > 0) & np.isfinite(uv).all(1) & (uv >= 0).all(1) & (uv < 512).all(1)
        uv, world, z, sat, u, v = uv[ok], world[ok], z[ok], sat[ok], u[ok], v[ok]
        cell = (uv[:,1]//4).astype(np.int64)*128+(uv[:,0]//4).astype(np.int64)
        token = view*128*128+(v//4)*128+u//4
        # Retain unique source tokens in each target cell; nearest observed satellite depth wins duplicates.
        order = np.lexsort((sat[:,2], token, cell))
        cell, token = cell[order], token[order]
        keep = np.ones(len(order), bool)
        keep[1:] = (cell[1:] != cell[:-1]) | (token[1:] != token[:-1])
        order = order[keep]
        cell, token = cell[keep], token[keep]
        uv, world, z, sat = uv[order], world[order], z[order], sat[order]
        # At most two distinct tokens per view/cell, ordered by observed satellite depth.
        order = np.lexsort((sat[:,2], cell))
        cell, token, uv, world, z, sat = (a[order] for a in (cell, token, uv, world, z, sat))
        starts = np.maximum.accumulate(np.where(np.r_[True, cell[1:] != cell[:-1]], np.arange(len(cell)), 0))
        rank = np.arange(len(cell))-starts
        keep = rank < 2
        cell, token, uv, world, z, sat, rank = (a[keep] for a in (cell, token, uv, world, z, sat, rank))
        # dx/dy relative to cell center; height relative to sampling origin, not absolute city coordinates.
        meta = np.stack(((uv[:,0]-(cell%128*4+2))/4,
                         (uv[:,1]-(cell//128*4+2))/4,
                         (world[:,2]-c['center_ue_cm'][2]/100)/100,
                         z/1000, np.full(len(cell), np.cos(np.deg2rad(yaw))),
                         np.full(len(cell), np.sin(np.deg2rad(yaw)))),1)
        cells.append(cell); ids.append(token); depths.append(rank*5+view)
        metas.append(meta); views.append(np.full(len(cell), view))
    cell, token, priority, meta = map(np.concatenate, (cells, ids, depths, metas))
    order = np.lexsort((priority, cell))
    cell, token, meta = cell[order], token[order], meta[order]
    starts = np.maximum.accumulate(np.where(np.r_[True, cell[1:] != cell[:-1]], np.arange(len(cell)), 0))
    rank = np.arange(len(cell))-starts
    keep = rank < 8
    index = np.full((128*128, 8), -1, np.int32)
    geom = np.zeros((128*128, 8, 6), np.float16)
    index[cell[keep], rank[keep]] = token[keep]
    geom[cell[keep], rank[keep]] = meta[keep]
    assert np.isfinite(geom).all() and max_roundtrip < 1e-5
    return index, geom, dict(source_hashes=hashes, roundtrip_max_px=max_roundtrip,
        support_cells=int((index>=0).any(1).sum()), candidates=int((index>=0).sum()))


def prepare_point(folder, cache):
    path = Path(cache)/f'{folder.name}.npz'
    if path.exists():
        with np.load(path) as a:
            assert str(a['version']) == GEOMETRY_VERSION
        return str(path)
    index, geom, report = build_candidates(folder)
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, index=index, geom=geom, version=GEOMETRY_VERSION,
                            report=json.dumps(report))
    tmp.replace(path)
    return str(path)


class LocalUAVAdapter(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        from rsprompter.backbone.sam2_vision import RSSAM2VisionEncoder
        self.encoder = RSSAM2VisionEncoder(checkpoint_path=pretrained, model_size='base_plus',
            use_gradient_checkpointing=True, use_lora=True, freeze_backbone=False,
            lora_config=dict(r=16,lora_alpha=32,target_modules=['qkv','proj'],
                             lora_dropout=.05,bias='none',modules_to_save=[]))
        self.source = nn.Conv2d(256, 32, 1)
        self.query = nn.Conv2d(256, 32, 1)
        self.key = nn.Linear(32, 32)
        self.value = nn.Linear(32, 32)
        self.geom_key = nn.Linear(6, 32, bias=False)
        self.geom_value = nn.Linear(6, 32, bias=False)
        self.output = nn.Linear(32, 32)
        self.alpha = nn.Parameter(torch.zeros(()))
        self.register_buffer('mean', torch.tensor([123.675,116.28,103.53]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([58.395,57.12,57.375]).view(1,3,1,1))
        self.context = None
        self.boxes = None
        self.enabled = True
        self.stats = {}
        self.handles = []

    def attach(self, model):
        head = model.roi_head.mask_head
        self.handles.append(head.register_forward_pre_hook(self._boxes_hook, with_kwargs=True))
        self.handles.append(head.mask_decoder.decoder.conv_s0.register_forward_hook(self._inject))

    def _boxes_hook(self, module, args, kwargs):
        self.boxes = kwargs['boxes']
        if self.context is not None:
            assert kwargs['image_embeddings'].shape[0] == 1, 'Probe uses image batch1'

    def encode(self, rgb, index, geom, satellite_high):
        features = []
        for image in rgb.split(1):
            _, pyramid = self.encoder((image-self.mean)/self.std)
            assert pyramid[0].shape[-2:] == (128,128)
            features.append(self.source(pyramid[0]))
        f = torch.cat(features).permute(0,2,3,1).reshape(-1,32)
        self.context = (f, index, geom, satellite_high)
        self.stats = {}

    def _chunk(self, qimage, keys, values, index, boxes, size):
        lin = (torch.arange(size, device=boxes.device, dtype=torch.float32)+.5)/size
        yy, xx = torch.meshgrid(lin,lin,indexing='ij')
        uv = boxes[:,None,None,:2] + torch.stack((xx,yy),-1)[None]*(boxes[:,None,None,2:]-boxes[:,None,None,:2])
        grid = uv/256-1  # align_corners=False: image edges 0/512, pixel centers .5
        q = F.grid_sample(qimage.expand(len(boxes),-1,-1,-1),grid.to(qimage.dtype),align_corners=False).permute(0,2,3,1)
        cell = ((uv[...,1]//4).long().clamp(0,127)*128 + (uv[...,0]//4).long().clamp(0,127))
        ids = index[cell]
        valid = (ids>=0) & ((uv>=0)&(uv<512)).all(-1)[...,None]
        safe = ids.clamp_min(0)
        k, v = keys[cell], values[cell]
        logits = (F.normalize(q.float(),dim=-1)[...,None,:]*F.normalize(k.float(),dim=-1)).sum(-1)*4
        logits = logits.masked_fill(~valid,-1e4)
        weight = logits.softmax(-1)*valid
        weight = weight/weight.sum(-1,keepdim=True).clamp_min(1e-6)
        pooled = (weight[...,None]*v).sum(-2)
        delta = self.output(pooled.to(q.dtype))*valid.any(-1)[...,None]
        return delta.permute(0,3,1,2)

    def _inject(self, module, args, base):
        if not self.enabled or self.context is None:
            return base
        f, index, geom, high = self.context
        # Candidate projections are computed once per image, not once per RoI.
        ids = index.clamp_min(0)
        k = self.key(f)[ids] + self.geom_key(geom)
        v = self.value(f)[ids] + self.geom_value(geom)
        qimage = self.query(high)
        chunks = []
        for boxes in self.boxes.split(4):
            fn = lambda qi, kk, vv, ix, bb: self._chunk(qi,kk,vv,ix,bb,base.shape[-1])
            if self.training and torch.is_grad_enabled():
                delta = checkpoint(fn,qimage,k,v,index,boxes,use_reentrant=False)
            else:
                delta = fn(qimage,k,v,index,boxes)
            chunks.append(delta)
        delta = torch.cat(chunks)
        applied = self.alpha*delta
        with torch.no_grad():
            self.stats = dict(alpha=float(self.alpha), support_fraction=float((index>=0).any(-1).float().mean()),
                source_delta_rms=float(delta.float().square().mean().sqrt()),
                applied_delta_rms=float(applied.float().square().mean().sqrt()),
                base_rms=float(base.float().square().mean().sqrt()), rois=len(base))
        return base+applied
