"""Raw strict HERO-UAV pairset reader with a manifest-authoritative B0 split."""
import hashlib
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

PAIR_SHA256 = "f15cd8f5b8d10a69d721546d6cfa1f2cce58af2861d57b1357c5e9ff1e1fe62a"
SPLIT_SHA256 = {
    "surfbev_b0_split.json": "894df4561b166d37d06d033a6fc326223062384b402f730591b4c76d41f926d5",
    "surfbev_b0_split_v2.json": "17f49430b15037c8da25c63a1bf7fe4d510cf5ff153263d2feba62c4393441d5",
}
GRID = re.compile(r"_gx_(-?\d+)_gy_(-?\d+)_")


class UERawPairDataset(Dataset):
    """Strict visible-roof instances resized for the satellite model."""
    def __init__(self, root, split, load_uav=False, split_manifest=None, image_size=(1024, 1024),
                 load_targets=True):
        self.root = Path(root)
        self.split = str(split)
        self.load_uav = bool(load_uav)
        self.load_targets = bool(load_targets)
        self.image_size = tuple(image_size)
        if self.image_size not in {(1024, 1024), (512, 512)}:
            raise ValueError("satellite input must be the registered 1024 upsample or the native 512 raster")
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"invalid split {split!r}")
        accepted = self.root / "accepted_pairs.jsonl"
        if hashlib.sha256(accepted.read_bytes()).hexdigest() != PAIR_SHA256:
            raise RuntimeError("raw pair manifest hash is not the registered SURFBEV contract")
        split_manifest = Path(split_manifest or "surfbev_b0_split_v2.json")
        if split_manifest.name not in SPLIT_SHA256:
            raise ValueError(f"unregistered raw SurfaceBEV split manifest: {split_manifest}")
        if not split_manifest.is_absolute():
            split_manifest = self.root / split_manifest
        if hashlib.sha256(split_manifest.read_bytes()).hexdigest() != SPLIT_SHA256[split_manifest.name]:
            raise RuntimeError("raw SurfaceBEV split manifest hash is not registered")
        data = json.loads(split_manifest.read_text())
        if data.get("accepted_manifest_sha256") != PAIR_SHA256:
            raise RuntimeError("split does not belong to the registered pair manifest")
        v2 = split_manifest.name == "surfbev_b0_split_v2.json"
        expected_schema = "surfacebev_b0_spatial_object_disjoint_v2" if v2 else "surfacebev_b0_spatial_split_v1"
        if data.get("schema_version") != expected_schema:
            raise RuntimeError("raw SurfaceBEV split schema does not match manifest name")
        accepted_rows = [json.loads(x) for x in accepted.read_text().splitlines() if x]
        accepted_by_scene = {r["scene_dir"]: r for r in accepted_rows}
        if len(accepted_by_scene) != len(accepted_rows):
            raise RuntimeError("accepted manifest contains duplicate scene names")
        seen, expected_records = set(), []
        for record in data["scenes"]:
            scene = record.get("scene_dir")
            m = GRID.search(str(scene))
            if scene in seen or scene not in accepted_by_scene or not m:
                raise RuntimeError("split contains duplicate, foreign, or malformed scene")
            seen.add(scene)
            gx, gy = int(m.group(1)), int(m.group(2))
            rule = "test" if 2 <= gx <= 4 and -7 <= gy <= -5 else "val" if -7 <= gx <= -5 and -3 <= gy <= -1 else "train"
            actual = record.get("split")
            valid = actual == rule if rule in {"test", "val"} or not v2 else actual in {"train", "excluded"}
            if (record.get("gx"), record.get("gy")) != (gx, gy) or not valid:
                raise RuntimeError("split record violates registered spatial rule")
            expected_records.append(record)
        if seen != set(accepted_by_scene):
            raise RuntimeError("split is not a complete one-to-one view of accepted_pairs.jsonl")
        self.records = [x for x in expected_records if x["split"] == self.split]
        expected = ({"train": 694, "val": 100, "test": 151} if v2 else
                    {"train": 949, "val": 100, "test": 151})[self.split]
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
        native = self.image_size == (512, 512)
        if not native:
            image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        scale = self.image_size[0] / 512.
        item = dict(img=torch.from_numpy(image).permute(2, 0, 1).float(),
                    img_metas=dict(img_shape=self.image_size, ori_shape=(512, 512), pad_shape=self.image_size,
                                   batch_input_shape=self.image_size, scale_factor=(scale, scale),
                                   filename=str(sat / "satellite_rgb.0000.png"), scene_id=rec["scene_dir"],
                                   image_id=int(hashlib.sha256(rec["scene_dir"].encode()).hexdigest()[:12], 16),
                                   gx=rec["gx"], gy=rec["gy"]))
        if self.load_targets:
            ids = self._ids(sat / "satellite_instance_id_clean.png")
            if ids.shape != (512, 512):
                raise RuntimeError(f"invalid strict ID shape for {rec['scene_dir']}")
            if not native:
                ids = cv2.resize(ids, self.image_size, interpolation=cv2.INTER_NEAREST)
            masks, boxes = self._instances(ids)
            item.update(gt_bboxes=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
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
