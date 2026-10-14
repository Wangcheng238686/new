"""Paired UE visible instances; local IDs become per-image satellite masks."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from rsprompter.uav.contract import PAYLOAD_SCHEMA, geometry_config, geometry_signature
from .loader import rtmdet_collate_fn, _seed_worker
from .ue_geometry_cache import YAWS, geometry_for_sample
from .ue_payload import tensor_payload


class UERGBDInstanceDataset(Dataset):
    def __init__(
        self,
        manifest,
        split,
        geometry,
        load_uav=True,
        cache_root=None,
        satellite_root=None,
        uav_root=None,
        subset_ratio=1.0,
    ):
        self.manifest_path = Path(manifest).resolve()
        data = json.loads(self.manifest_path.read_text())
        if data.get("schema") != PAYLOAD_SCHEMA:
            raise ValueError("Unsupported UE manifest schema")
        if split not in data["splits"]:
            raise ValueError(
                f"Manifest has no {split} split; do not silently reuse validation"
            )
        self.manifest_sha256 = hashlib.sha256(
            self.manifest_path.read_bytes()
        ).hexdigest()
        self.geometry = geometry_config(geometry)
        self.signature = geometry_signature(geometry)
        self.satellite_root = Path(satellite_root or data["satellite_root"])
        self.uav_root = Path(uav_root or data["uav_root"])
        self.load_uav = load_uav
        self.cache_root = cache_root
        self.floor_m = data["split_audit"]["floor_m"]
        by_id = {row["index"]: row for row in data["points"]}
        ids = data["splits"][split]
        if not 0 < subset_ratio <= 1:
            raise ValueError("Subset ratio must lie in (0,1]")
        ids = ids[: max(1, int(len(ids) * subset_ratio))]
        self.samples = [by_id[index] for index in ids]
        if not self.samples:
            raise ValueError(f"Empty UE split: {split}")
        self.data_config = dict(
            dataset_format="ue_rgbd",
            manifest=str(self.manifest_path),
            manifest_sha256=self.manifest_sha256,
            geometry=self.geometry,
            geometry_signature=self.signature,
            split_floor_m=self.floor_m,
            satellite_root=str(self.satellite_root),
            uav_root=str(self.uav_root),
            single_class=True,
            num_classes=1,
            image_size=[self.geometry["image_size"]] * 2,
            load_uav=load_uav,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]
        satellite = self.satellite_root / row["satellite_dir"]
        uav = self.uav_root / row["uav_dir"]
        size = self.geometry["image_size"]
        image = cv2.imread(str(satellite / "satellite_rgb.0000.png"))
        ids = cv2.imread(
            str(satellite / "satellite_instance_id_clean.png"), cv2.IMREAD_UNCHANGED
        )
        if (
            image is None
            or ids is None
            or image.shape[:2] != (size, size)
            or ids.shape != (size, size)
        ):
            raise ValueError(f"Invalid satellite RGB/instance map: {satellite}")
        instances = json.loads((satellite / "instances.json").read_text())["instances"]
        mapping = {
            int(item["local_instance_id"]): int(item["global_instance_id"])
            for item in instances
        }
        if len(mapping) != len(instances) or any(key <= 0 for key in mapping):
            raise ValueError(f"Invalid local/global instance mapping: {satellite}")
        visible_ids = set(map(int, np.unique(ids))) - {0}
        if visible_ids - mapping.keys():
            raise ValueError(f"Visible instance IDs have no mapping: {satellite}")
        masks, boxes = [], []
        for local_id in np.unique(ids):
            if local_id == 0:
                continue
            mask = ids == local_id
            y, x = np.where(mask)
            masks.append(mask)
            boxes.append([x.min(), y.min(), x.max() + 1, y.max() + 1])
        item = dict(
            img=torch.from_numpy(image.copy()).permute(2, 0, 1).float(),
            img_metas=dict(
                img_shape=(size, size),
                ori_shape=(size, size),
                pad_shape=(size, size),
                batch_input_shape=(size, size),
                scale_factor=(1.0, 1.0),
                filename=str(satellite / "satellite_rgb.0000.png"),
                image_id=row["index"],
                scene_id=str(row["index"]),
            ),
            gt_bboxes=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            gt_labels=torch.zeros(len(boxes), dtype=torch.long),
            gt_masks=np.array(masks, dtype=np.uint8).reshape(-1, size, size),
        )
        if self.load_uav:
            rgb = []
            for yaw in YAWS:
                path = uav / "oblique" / f"view_yaw_{yaw:03d}" / "oblique_rgb.0000.png"
                raw = cv2.imread(str(path))
                if raw is None or raw.shape[:2] != (size, size):
                    raise ValueError(f"Invalid UAV RGB: {path}")
                rgb.append(raw[..., ::-1].transpose(2, 0, 1))
            arrays = geometry_for_sample(
                satellite,
                uav,
                np.asarray(row["origin_m"]),
                self.geometry,
                self.cache_root,
                self.floor_m,
            )
            item["uav_payload"] = tensor_payload(arrays, np.stack(rgb), self.signature)
        return item


def ue_collate(batch):
    result = rtmdet_collate_fn(batch)
    if "uav_payload" in batch[0]:
        if any("uav_payload" not in item for item in batch):
            raise ValueError("Mixed single/dual payloads in one batch")
        result["uav_payloads"] = [item["uav_payload"] for item in batch]
    return result


def ue_loader(
    dataset,
    batch_size,
    num_workers=4,
    train=False,
    distributed=False,
    rank=0,
    world_size=1,
    seed=42,
):
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=train,
            seed=seed,
            drop_last=train,
        )
        if distributed
        else None
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=ue_collate,
        drop_last=train,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
        generator=torch.Generator().manual_seed(seed),
        persistent_workers=num_workers > 0,
    )
