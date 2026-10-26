"""Explicit measured/reference geometry inputs for the compact joint model."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from rsprompter.uav.local_adapter_probe import YAWS, file_sha


class JointUAVInputs:
    def __init__(self, data_root, native_cache, anchor_cache, reference_cache):
        self.data_root = Path(data_root)
        self.native_cache = Path(native_cache)
        self.anchor_cache = Path(anchor_cache)
        self.reference_cache = Path(reference_cache)

    def load(self, image_id, *, geometry_mode='measured', rgb_mode='correct', device='cuda'):
        if geometry_mode not in ('measured', 'reference_plane') or rgb_mode not in ('correct', 'constant'):
            raise ValueError((geometry_mode, rgb_mode))
        folder = self.data_root / 'points' / f'{int(image_id):06d}'
        camera_sha = file_sha(folder / 'cameras.json')
        if geometry_mode == 'measured':
            with np.load(self.native_cache / f'{int(image_id):06d}.npz') as z:
                assert str(z['version']) == 'native512_height_dx_dy_viewcount_v1'
                fields, support = z['fields'].astype(np.float32), z['support'].astype(np.float32)
                native = json.loads(str(z['report']))
            with np.load(self.anchor_cache / f'{int(image_id):06d}.npz') as z:
                assert str(z['version']) == 's2db_anchor_metadata_v2_fp64'
                index, anchor = z['index'].astype(np.int64), z['anchor'].copy()
                anchored = json.loads(str(z['report']))
            assert native['source_hashes'] == anchored['source_hashes']
            assert native['source_hashes']['cameras.json'] == camera_sha
            assert len([name for name in native['source_hashes'] if name.endswith('.exr')]) == 5
            provenance = dict(measured_depth=True, camera_sha256=camera_sha,
                              measured_source_hashes=native['source_hashes'])
        else:
            # No fallback to measured caches when the reference cache is absent.
            with np.load(self.reference_cache / f'{int(image_id):06d}.npz') as z:
                assert str(z['version']) == 'reference_plane_sampling_origin_v1'
                fields, support = z['fields'].astype(np.float32), z['support'].astype(np.float32)
                index, anchor = z['index'].astype(np.int64), z['anchor'].copy()
                report = json.loads(str(z['report']))
            assert report['camera_sha256'] == camera_sha and report['measured_depth_read'] is False
            assert report['relative_reference_z_m'] == 0.
            provenance = dict(measured_depth=False, camera_sha256=camera_sha,
                              reference='sampling-origin horizontal plane', source_hashes=report['source_hashes'])
        geometry = np.concatenate((fields, support[None]))[None]
        assert geometry.shape == (1, 5, 512, 512) and np.isfinite(geometry).all()
        assert anchor.dtype == np.float64 and np.isfinite(anchor).all()
        assert index.shape == (16384, 8) and anchor.shape == (16384, 8, 6)
        rgb = []
        for yaw in YAWS:
            image = cv2.imread(str(folder / f'uav/yaw_{yaw:03d}/rgb.png'))
            assert image is not None and image.shape == (512, 512, 3)
            rgb.append(image[:, :, ::-1].transpose(2, 0, 1).copy())
        rgb = torch.from_numpy(np.stack(rgb)).to(device=device, dtype=torch.float32)
        if rgb_mode == 'constant':
            rgb[:] = torch.tensor([123.675, 116.28, 103.53], device=device).view(1, 3, 1, 1)
        return dict(geometry=torch.from_numpy(geometry).to(device), rgb=rgb,
                    index=torch.from_numpy(index).to(device), anchor=torch.from_numpy(anchor).to(device),
                    provenance=dict(image_id=int(image_id), geometry_mode=geometry_mode, rgb_mode=rgb_mode, **provenance))
