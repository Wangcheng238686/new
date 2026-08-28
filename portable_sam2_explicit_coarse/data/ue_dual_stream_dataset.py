"""Official-split UE satellite/UAV dataset for geometry-driven dual stream."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


MISSING_SCENE_CENTERS = {"003496", "003497", "003498", "003500"}


class UEDualStreamDataset(Dataset):
    def __init__(self, root, split, image_size=(1024, 1024), train_subset=1.0, seed=42,
                 load_uav=True):
        self.root, self.split, self.image_size = Path(root), str(split), tuple(image_size)
        self.load_uav = bool(load_uav)
        centers_path = self.root / "scene_centers.json"
        with centers_path.open() as f:
            self.centers = json.load(f)
        satellite = self.root / self.split / "satellite"
        ids = []
        for path in sorted(satellite.glob("*.png")):
            sid = path.stem
            camera = self.root / self.split / "drone" / sid / "ue_render_metadata" / "camera_params.json"
            if (satellite / f"{sid}.json").is_file() and camera.is_file() and sid in self.centers:
                ids.append(sid)
        if self.split == "test":
            ids = [sid for sid in ids if sid not in MISSING_SCENE_CENTERS]
        if self.split == "train" and train_subset < 1.0:
            n = max(1, int(len(ids) * float(train_subset)))
            ids = [ids[i] for i in sorted(np.random.RandomState(seed).permutation(len(ids))[:n])]
        self.ids = ids

    def __len__(self): return len(self.ids)

    @staticmethod
    def _instances(annotation, height, width):
        groups = {}
        for shape in annotation.get("shapes", []):
            if shape.get("shape_type") == "polygon" and len(shape.get("points", [])) >= 3:
                groups.setdefault(shape.get("group_id", len(groups)), []).append(np.asarray(shape["points"], np.int32))
        masks, boxes = [], []
        for polygons in groups.values():
            mask = np.zeros((height, width), np.uint8)
            cv2.fillPoly(mask, polygons, 1)
            ys, xs = np.where(mask)
            if len(xs):
                masks.append(mask); boxes.append([xs.min(), ys.min(), xs.max()+1, ys.max()+1])
        return np.stack(masks) if masks else np.zeros((0, height, width), np.uint8), boxes

    def __getitem__(self, index):
        sid = self.ids[index]; sat_dir = self.root / self.split / "satellite"
        image = cv2.imread(str(sat_dir / f"{sid}.png")); h0, w0 = image.shape[:2]
        with (sat_dir / f"{sid}.json").open() as f: annotation = json.load(f)
        masks, boxes = self._instances(annotation, h0, w0)
        sx, sy = self.image_size[0] / w0, self.image_size[1] / h0
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        masks = np.stack([cv2.resize(m, self.image_size, interpolation=cv2.INTER_NEAREST) for m in masks]) if len(masks) else np.zeros((0, self.image_size[1], self.image_size[0]), np.uint8)
        boxes = torch.tensor([[x1*sx, y1*sy, x2*sx, y2*sy] for x1,y1,x2,y2 in boxes], dtype=torch.float32).reshape(-1,4)
        item = dict(img=torch.from_numpy(image).permute(2,0,1).float(),
                    img_metas=dict(img_shape=(self.image_size[1],self.image_size[0]), ori_shape=(h0,w0), pad_shape=(self.image_size[1],self.image_size[0]), batch_input_shape=(self.image_size[1],self.image_size[0]), scale_factor=(sx,sy), filename=str(sat_dir/f"{sid}.png"), scene_id=sid),
                    gt_bboxes=boxes, gt_labels=torch.zeros(len(boxes), dtype=torch.long), gt_masks=masks)
        if not self.load_uav:
            return item
        drone_dir = self.root / self.split / "drone" / sid
        with (drone_dir / "ue_render_metadata" / "camera_params.json").open() as f: camera = json.load(f)
        drone = []
        for name in camera["image_names"]:
            path = drone_dir / name
            if not path.is_file(): path = drone_dir / "ue_render_metadata" / name
            arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
            drone.append(torch.from_numpy(arr).permute(2,0,1))
        item.update(drone_images=torch.stack(drone), intrinsics=torch.tensor(camera["intrinsics"], dtype=torch.float32),
                    extrinsics=torch.tensor(camera["extrinsics"], dtype=torch.float32), scene_center=torch.tensor(self.centers[sid], dtype=torch.float32))
        return item


def ue_dual_collate(batch):
    from .loader import rtmdet_collate_fn
    base = rtmdet_collate_fn(batch)
    if "drone_images" in batch[0]:
        base.update(drone_images=torch.stack([x["drone_images"] for x in batch]), intrinsics=torch.stack([x["intrinsics"] for x in batch]), extrinsics=torch.stack([x["extrinsics"] for x in batch]), scene_centers=torch.stack([x["scene_center"] for x in batch]))
    return base
