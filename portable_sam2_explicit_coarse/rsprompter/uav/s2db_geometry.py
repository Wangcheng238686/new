"""S2DB-inspired study cache; preserves v1 candidate selection exactly.

anchor fields: original source u,v [RGB pixels], camera-Z [m], projected
satellite u,v [pixels], satellite camera-Z [m]. No shifted 3D claims.
"""
import json
import os
from pathlib import Path
import cv2
import numpy as np
from .local_adapter_probe import YAWS, camera_arrays, file_sha, build_candidates as legacy_build

VERSION = 's2db_anchor_metadata_v2_fp64'
LEGACY_CACHE = Path('/data/wangcheng/cache/ue_v1_local_adapter_geometry')

def build_candidates(folder):
    c = json.loads((folder/'cameras.json').read_text())
    assert c['world_units'] == 'cm' and c['camera_axes'] == 'x_right_y_down_z_forward'
    sk, se, _ = camera_arrays(c['satellite'])
    yy, xx = np.mgrid[1:512:2, 1:512:2]
    xx, yy = xx.ravel(), yy.ravel()
    cells, ids, depths, metas, extras = [], [], [], [], []
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
        uv, world, z, sat, u, v = (a[order] for a in (uv, world, z, sat, u, v))
        # At most two distinct tokens per view/cell, ordered by observed satellite depth.
        order = np.lexsort((sat[:,2], cell))
        cell, token, uv, world, z, sat, u, v = (a[order] for a in (cell, token, uv, world, z, sat, u, v))
        starts = np.maximum.accumulate(np.where(np.r_[True, cell[1:] != cell[:-1]], np.arange(len(cell)), 0))
        rank = np.arange(len(cell))-starts
        keep = rank < 2
        cell, token, uv, world, z, sat, u, v, rank = (a[keep] for a in (cell, token, uv, world, z, sat, u, v, rank))
        # dx/dy relative to cell center; height relative to sampling origin, not absolute city coordinates.
        meta = np.stack(((uv[:,0]-(cell%128*4+2))/4,
                         (uv[:,1]-(cell//128*4+2))/4,
                         (world[:,2]-c['center_ue_cm'][2]/100)/100,
                         z/1000, np.full(len(cell), np.cos(np.deg2rad(yaw))),
                         np.full(len(cell), np.sin(np.deg2rad(yaw)))),1)
        cells.append(cell); ids.append(token); depths.append(rank*5+view)
        metas.append(meta); extras.append(np.stack((u,v,z,uv[:,0],uv[:,1],sat[:,2]),1))
    cell, token, priority, meta, extra = map(np.concatenate, (cells, ids, depths, metas, extras))
    order = np.lexsort((priority, cell))
    cell, token, meta, extra = (a[order] for a in (cell, token, meta, extra))
    starts = np.maximum.accumulate(np.where(np.r_[True, cell[1:] != cell[:-1]], np.arange(len(cell)), 0))
    rank = np.arange(len(cell))-starts
    keep = rank < 8
    index = np.full((128*128, 8), -1, np.int32)
    geom = np.zeros((128*128, 8, 6), np.float16)
    anchor = np.zeros((128*128,8,6),np.float64)
    anchor[cell[keep],rank[keep]] = extra[keep]
    index[cell[keep], rank[keep]] = token[keep]
    geom[cell[keep], rank[keep]] = meta[keep]
    assert np.isfinite(geom).all() and max_roundtrip < 1e-5
    return index, geom, anchor, dict(source_hashes=hashes, roundtrip_max_px=max_roundtrip,
        support_cells=int((index>=0).any(1).sum()), candidates=int((index>=0).sum()))


def prepare_point(folder, cache):
    folder, cache = Path(folder), Path(cache)
    cache.mkdir(parents=True,exist_ok=True)
    path=cache/f'{folder.name}.npz'
    if path.exists():
        with np.load(path) as a:
            assert str(a['version'])==VERSION
        return path
    index,geom,anchor,report=build_candidates(folder)
    old=LEGACY_CACHE/path.name
    if old.exists():
        with np.load(old) as a: oi,og=a['index'],a['geom']
    else:
        oi,og,_=legacy_build(folder)
    assert np.array_equal(index,oi) and np.array_equal(geom,og), 'Legacy cache changed'
    valid=index>=0
    assert np.isfinite(anchor).all()
    expected=index[valid]%16384
    assert np.array_equal((anchor[...,1][valid].astype(int)//4)*128+anchor[...,0][valid].astype(int)//4,expected)
    assert np.all(anchor[...,2][valid]>0) and np.all(anchor[...,5][valid]>0)
    uv=anchor[...,3:5][valid]
    cells=np.broadcast_to(np.arange(16384)[:,None],index.shape)[valid]
    assert np.array_equal((uv[:,1]//4).astype(int)*128+(uv[:,0]//4).astype(int),cells)
    report.update(legacy_index_geom_exact=True,anchor_fields=['source_u_px','source_v_px','source_z_m',
        'satellite_u_px','satellite_v_px','satellite_z_m'],version=VERSION,
        builder_sha256=file_sha(__file__))
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f,index=index,geom=geom,anchor=anchor,version=VERSION,report=json.dumps(report))
    tmp.replace(path)
    return path


def load_depth(folder):
    out=[]
    for yaw in YAWS:
        raw=cv2.imread(str(Path(folder)/f'uav/yaw_{yaw:03d}/depth_camera_z.exr'),cv2.IMREAD_UNCHANGED)
        assert raw is not None and raw.shape==(512,512,3)
        out.append(raw[:,:,2].astype(np.float32)/100)
    return np.stack(out)


def observed_support(index,anchor):
    mask=np.zeros((512,512),bool)
    uv=anchor[...,3:5][index>=0]
    mask[uv[:,1].astype(int),uv[:,0].astype(int)]=True
    return mask
