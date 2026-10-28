"""Isolated measured-depth subset caches; no fallback to five-view products."""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from rsprompter.uav.local_adapter_probe import YAWS, file_sha
from rsprompter.uav.view_subset import canonical_views, subset_key, subset_arrays

VERSION = 'measured_view_subset_canonical_slots_v1'
CODE_FILES = ('data/ue_view_subset_inputs.py','rsprompter/uav/view_subset.py',
              'rsprompter/uav/geometry_inputs.py','rsprompter/uav/local_adapter_probe.py')
WORK = Path(__file__).resolve().parents[1]


def code_hashes():
    return {name:file_sha(WORK/name) for name in CODE_FILES}


def build_image(data_root, image_id, view_ids):
    views = canonical_views(view_ids)
    folder = Path(data_root)/'points'/('%06d' % int(image_id))
    camera_file = folder/'cameras.json'
    cameras = json.loads(camera_file.read_text())
    if cameras['world_units'] != 'cm' or cameras['camera_axes'] != 'x_right_y_down_z_forward':
        raise ValueError('Unexpected raw depth/camera units')
    sources = {'cameras.json':file_sha(camera_file)}
    depths = np.full((5,512,512), np.nan, np.float64)
    for yaw in views:
        name = 'uav/yaw_%03d/depth_camera_z.exr' % yaw
        raw = cv2.imread(str(folder/name), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.shape != (512,512,3):
            raise ValueError('Expected BGR EXR depth at '+str(folder/name))
        sources[name] = file_sha(folder/name)
        depths[YAWS.index(yaw)] = raw[:,:,2].astype(np.float64)/100.
    arrays = subset_arrays(cameras,depths,views)
    report = dict(image_id=int(image_id),view_ids=list(views),source_hashes=sources,
        source_code_sha256=code_hashes(),depth_units='camera-Z metres from EXR R centimetres',
        view_count_denominator=5,canonical_slots=list(YAWS),candidate_cap=8,per_view_cap=2,
        candidates=int((arrays['index']>=0).sum()),supported_pixels=int(arrays['support'].sum()))
    return arrays, report


class SubsetUAVInputs:
    def __init__(self, data_root, cache_root, view_ids, *, require_full_dataset=True):
        self.data_root = Path(data_root).resolve()
        self.view_ids = canonical_views(view_ids)
        self.root = Path(cache_root)/subset_key(self.view_ids)
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        self.complete = json.loads((self.root/'BUILD_COMPLETE.json').read_text())
        expected = self.manifest
        if (expected['version'] != VERSION or expected['view_ids'] != list(self.view_ids)
                or expected['data_root'] != str(self.data_root) or expected['source_code_sha256'] != code_hashes()
                or self.complete['complete'] is not True
                or self.complete['manifest_sha256'] != file_sha(self.root/'manifest.json')):
            raise ValueError('Subset cache contract mismatch')
        for split, sha in expected['split_sha256'].items():
            if file_sha(self.data_root/'splits'/(split+'.json')) != sha:
                raise ValueError('Dataset split identity changed')
        ids = expected['image_ids']
        self.rows = {int(row['image_id']):row for row in self.complete['rows']}
        if len(self.rows) != len(ids) or set(self.rows) != set(ids):
            raise ValueError('Incomplete or duplicate cache receipts')
        if require_full_dataset:
            all_ids = [int(i) for split in ('train','val','test') for i in json.loads((self.data_root/'splits'/(split+'.json')).read_text())]
            if len(set(all_ids)) != len(all_ids) or set(ids) != set(all_ids):
                raise ValueError('Formal runtime requires all train/val/test subset products')

    def load(self, image_id, *, geometry_mode='measured', rgb_mode='correct', device='cuda'):
        if geometry_mode != 'measured' or rgb_mode not in ('correct','constant'):
            raise ValueError('Subset input supports measured geometry and correct/constant RGB')
        ident = int(image_id)
        path = self.root/('%06d.npz' % ident)
        row = self.rows[ident]
        if file_sha(path) != row['cache_sha256']:
            raise ValueError('Subset cache bytes changed')
        with np.load(path,allow_pickle=False) as z:
            report = json.loads(str(z['report']))
            if str(z['version']) != VERSION or report != row['report'] or report['view_ids'] != list(self.view_ids):
                raise ValueError('Subset cache image identity mismatch')
            index = z['index'].astype(np.int64)
            anchor = z['anchor'].copy()
            geometry = np.concatenate((z['fields'].astype(np.float32), z['support'][None].astype(np.float32)))[None]
        folder = self.data_root/'points'/('%06d' % ident)
        if file_sha(folder/'cameras.json') != report['source_hashes']['cameras.json']:
            raise ValueError('Camera source changed since subset build')
        # Five fixed slots retain original token identities. Absent slots are
        # never read from disk and never encoded by the subset model reader.
        rgb = np.zeros((5,3,512,512), np.float32)
        for yaw in self.view_ids:
            image = cv2.imread(str(folder/('uav/yaw_%03d/rgb.png' % yaw)))
            if image is None or image.shape != (512,512,3):
                raise ValueError('Missing selected UAV RGB')
            rgb[YAWS.index(yaw)] = image[:,:,::-1].transpose(2,0,1)
        rgb = torch.from_numpy(rgb).to(device)
        if rgb_mode == 'constant':
            mean = rgb.new_tensor([123.675,116.28,103.53]).view(3,1,1)
            for yaw in self.view_ids:
                rgb[YAWS.index(yaw)] = mean
        if geometry.shape != (1,5,512,512) or anchor.dtype != np.float64 or not np.isfinite(geometry).all() or not np.isfinite(anchor).all():
            raise ValueError('Invalid subset tensor shape/units')
        return dict(geometry=torch.from_numpy(geometry).to(device),rgb=rgb,
            index=torch.from_numpy(index).to(device),anchor=torch.from_numpy(anchor).to(device),
            provenance=dict(measured_depth=True,view_ids=list(self.view_ids),view_subset_key=subset_key(self.view_ids),
                            cache_manifest_sha256=self.complete['manifest_sha256'],source_hashes=report['source_hashes']))
