#!/usr/bin/env python3
"""Paired Sat-UAV loader for the finalized CitySample archive
(CitySample_SatUAV_TrainMinimal_v1_20261021, label revision
review99_v3_user_approved_20261021). Produces the probe-side `it` dict:
satellite rgb tensor, GT instance masks (COCO uncompressed RLE), 5-view UAV
rgb + EXR camera-Z depth (cm, float - NOT the old lossy packed PNG), and
camera pairs in the exact (intrinsics, extrinsics) dict shape consumed by
ue.dual._k_and_w2c. Geometry verified empirically 2026-10-21: UAV center
pixel backprojects to satellite uv (256,259); 300 m ring contract intact.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2

ROOT_ARCHIVE = '/data1/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021'


def decode_rle(rle):
    runs = np.asarray(rle['counts'], dtype=np.int64)
    h, w = rle['size']
    if runs.ndim != 1 or (runs < 0).any() or int(runs.sum()) != h * w:
        raise ValueError('Invalid COCO RLE')
    return np.repeat(np.arange(len(runs)) % 2, runs).reshape((h, w), order='F').astype(bool)


def read_camera_z_cm(path):
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise RuntimeError(f'EXR unavailable: {path}')
    if d.ndim == 3:
        d = d[:, :, 2]
    return d.astype(np.float32)          # cm, camera-space Z


class CitySamplePairDataset:
    def __init__(self, root=ROOT_ARCHIVE, split='val', limit=0):
        self.root = Path(root)
        ann = json.load(open(self.root / f'annotations/{split}.json'))
        self.by_img = {}
        for a in ann['annotations']:
            self.by_img.setdefault(a['image_id'], []).append(a)
        self.imgs = ann['images']
        if limit:
            self.imgs = self.imgs[:int(limit)]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        e = self.imgs[i]
        point_dir = self.root / Path(e['file_name']).parents[1]
        sat_rgb = np.asarray(Image.open(self.root / e['file_name']).convert('RGB'))
        cams = json.load(open(point_dir / 'cameras.json'))
        yaws = [f'yaw_{y:03d}' for y in cams['uav_yaw_order']]
        drone = np.stack([np.asarray(Image.open(point_dir / 'uav' / y / 'rgb.png').convert('RGB'))
                          for y in yaws])                        # [5,512,512,3] uint8
        depth = np.stack([read_camera_z_cm(point_dir / 'uav' / y / 'depth_camera_z.exr')
                          for y in yaws])                        # [5,512,512] cm
        uav_pairs = [(cams['uav'][y]['intrinsics'], cams['uav'][y]) for y in yaws]
        sat_pair = (cams['satellite']['intrinsics'], cams['satellite'])
        gt_masks = [decode_rle(a['segmentation']) for a in self.by_img.get(e['id'], [])]
        scene_id = Path(e['file_name']).parts[1]
        return {
            'scene_id': scene_id,
            'img': torch.from_numpy(sat_rgb.copy()).permute(2, 0, 1).float(),
            'gt_masks': gt_masks,
            'drone_images': torch.from_numpy(drone.copy()).permute(0, 3, 1, 2).float(),
            'depth_cm': torch.from_numpy(depth.copy()),
            'uav_cameras': uav_pairs,
            'satellite_camera': sat_pair,
            'img_metas': {'scene_id': scene_id, 'ori_shape': (512, 512)},
        }


if __name__ == '__main__':
    ds = CitySamplePairDataset(split='val', limit=3)
    it = ds[0]
    for k, v in it.items():
        if isinstance(v, torch.Tensor):
            print(k, tuple(v.shape), float(v.min()), float(v.max()))
        else:
            print(k, type(v).__name__, len(v))
    print('scene', it['scene_id'], 'gt', len(it['gt_masks']))
