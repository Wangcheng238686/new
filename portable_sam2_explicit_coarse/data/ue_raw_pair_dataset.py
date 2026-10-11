"""Raw strict HERO-UAV pairset reader with a manifest-authoritative B0 split."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

PAIR_SHA256 = "f15cd8f5b8d10a69d721546d6cfa1f2cce58af2861d57b1357c5e9ff1e1fe62a"


class UERawPairDataset(Dataset):
    """512px strict visible-roof instances; UAV payload is opt-in only."""
    def __init__(self, root, split, load_uav=False, split_manifest=None):
        self.root = Path(root)
        self.split = str(split)
        self.load_uav = bool(load_uav)
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"invalid split {split!r}")
        accepted = self.root / "accepted_pairs.jsonl"
        if hashlib.sha256(accepted.read_bytes()).hexdigest() != PAIR_SHA256:
            raise RuntimeError("raw pair manifest hash is not the registered SURFBEV contract")
        split_manifest = Path(split_manifest or self.root / "surfbev_b0_split.json")
        data = json.loads(split_manifest.read_text())
        if data.get("accepted_manifest_sha256") != PAIR_SHA256:
            raise RuntimeError("split does not belong to the registered pair manifest")
        self.records = [x for x in data["scenes"] if x["split"] == self.split]
        expected = {"train": 949, "val": 100, "test": 151}[self.split]
        if len(self.records) != expected:
            raise RuntimeError(f"{self.split} expected {expected} scenes, got {len(self.records)}")

    def __len__(self):
        return len(self.records)

    @staticmethod
    def _ids(path):
        a = np.asarray(Image.open(path))
        if a.ndim == 3:
            if not (np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 0], a[..., 2])):
                raise ValueError(f"non-grayscale strict ID raster: {path}")
            a = a[..., 0]
        return a.astype(np.int32)

    @staticmethod
    def _instances(ids):
        labels = np.unique(ids[ids > 0])
        masks, boxes = [], []
        for lid in labels:
            mask = (ids == lid).astype(np.uint8)
            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue
            masks.append(mask); boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        return (np.stack(masks) if masks else np.zeros((0, *ids.shape), np.uint8), boxes)

    def __getitem__(self, index):
        rec = self.records[index]
        scene = self.root / "scenes" / rec["scene_dir"]
        sat = scene / "satellite"
        image = cv2.imread(str(sat / "satellite_rgb.0000.png"), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (512, 512):
            raise RuntimeError(f"invalid satellite RGB for {rec['scene_dir']}")
        ids = self._ids(sat / "satellite_instance_id_clean.png")
        if ids.shape != (512, 512):
            raise RuntimeError(f"invalid strict ID shape for {rec['scene_dir']}")
        masks, boxes = self._instances(ids)
        item = dict(img=torch.from_numpy(image).permute(2, 0, 1).float(),
                    img_metas=dict(img_shape=(512, 512), ori_shape=(512, 512), pad_shape=(512, 512),
                                   batch_input_shape=(512, 512), scale_factor=(1., 1.),
                                   filename=str(sat / "satellite_rgb.0000.png"), scene_id=rec["scene_dir"],
                                   gx=rec["gx"], gy=rec["gy"]),
                    gt_bboxes=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                    gt_labels=torch.zeros(len(boxes), dtype=torch.long), gt_masks=masks)
        if not self.load_uav:
            return item
        # This branch is intentionally unreachable during B0.  P1 may opt in.
        uav, cams = scene / "uav", []
        drone, depth = [], []
        for view in sorted(uav.glob("view_yaw_*")):
            drone.append(torch.from_numpy(np.asarray(Image.open(view / "oblique_rgb.0000.png").convert("RGB"), np.float32) / 255.).permute(2, 0, 1))
            depth.append(torch.from_numpy(np.asarray(Image.open(view / "depth_devicez_packed.0000.png").convert("RGB"), np.uint8)))
            cams.append((json.loads((view / "camera_intrinsics.json").read_text()), json.loads((view / "camera_extrinsics.json").read_text())))
        item.update(drone_images=torch.stack(drone), devicez_rgb=torch.stack(depth), uav_cameras=cams,
                    satellite_camera=(json.loads((sat / "camera_intrinsics.json").read_text()), json.loads((sat / "camera_extrinsics.json").read_text())))
        return item
