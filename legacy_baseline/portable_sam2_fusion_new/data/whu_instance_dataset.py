import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


class WHUCocoInstanceDataset(Dataset):
    """WHU instance dataset from COCO-style json files.

    Current default behavior is single-class training (all instances -> label 0).
    A category mapping path is reserved for future multi-class expansion.
    """

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        image_subdir: str,
        image_size: Tuple[int, int] = (512, 512),
        single_class: bool = True,
        enable_category_mapping: bool = False,
        category_mapping: Optional[Dict[int, int]] = None,
        flip_prob: float = 0.0,
        vflip_prob: float = 0.0,
        gaussian_noise_prob: float = 0.0,
        gaussian_noise_std: float = 0.02,
        random_erasing_prob: float = 0.0,
        random_erasing_scale: Tuple[float, float] = (0.02, 0.2),
        random_erasing_ratio: Tuple[float, float] = (0.3, 3.3),
        multi_scale_resize_prob: float = 0.0,
        multi_scale_mode: str = "range",
        multi_scale_img_scale: Optional[List[Tuple[int, int]]] = None,
    ):
        self.data_root = Path(data_root)
        self.ann_path = self.data_root / ann_file
        self.image_root = self.data_root / image_subdir
        self.image_size = tuple(image_size)

        self.multi_scale_resize_prob = float(multi_scale_resize_prob)
        self.multi_scale_mode = str(multi_scale_mode)
        if multi_scale_img_scale is None:
            self.multi_scale_img_scale = [(1344, 819), (1344, 1344)]
        else:
            self.multi_scale_img_scale = [
                (int(s[0]), int(s[1])) for s in multi_scale_img_scale
            ]

        # Pre-compute long/short edge ranges for 'range' mode (mmdet semantics).
        if self.multi_scale_img_scale:
            _longs = [max(s[0], s[1]) for s in self.multi_scale_img_scale]
            _shorts = [min(s[0], s[1]) for s in self.multi_scale_img_scale]
            self._ms_long_range = (min(_longs), max(_longs))
            self._ms_short_range = (min(_shorts), max(_shorts))
        else:
            self._ms_long_range = (0, 0)
            self._ms_short_range = (0, 0)
            self.multi_scale_resize_prob = 0.0  # safety: disable if empty

        self.single_class = bool(single_class)
        self.enable_category_mapping = bool(enable_category_mapping)
        self.category_mapping = category_mapping or {}

        self.flip_prob = float(flip_prob)
        self.vflip_prob = float(vflip_prob)
        self.gaussian_noise_prob = float(gaussian_noise_prob)
        self.gaussian_noise_std = float(gaussian_noise_std)
        self.random_erasing_prob = float(random_erasing_prob)
        self.random_erasing_scale = tuple(random_erasing_scale)
        self.random_erasing_ratio = tuple(random_erasing_ratio)

        if not self.ann_path.exists():
            raise FileNotFoundError(f"WHU annotation file not found: {self.ann_path}")
        if not self.image_root.exists():
            raise FileNotFoundError(f"WHU image directory not found: {self.image_root}")

        self.samples = self._load_samples()
        logger.info("加载 WHU 样本: %d (%s)", len(self.samples), self.ann_path.name)

    def _load_samples(self) -> List[Dict]:
        with open(self.ann_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        images = coco.get("images", [])
        annotations = coco.get("annotations", [])

        anns_by_img: Dict[int, List[Dict]] = {}
        for ann in annotations:
            image_id = int(ann.get("image_id", -1))
            if image_id < 0:
                continue
            anns_by_img.setdefault(image_id, []).append(ann)

        samples: List[Dict] = []
        for img in images:
            image_id = int(img.get("id", -1))
            file_name = str(img.get("file_name", ""))
            if image_id < 0 or not file_name:
                continue

            image_path = self.image_root / file_name
            if not image_path.exists():
                # Some COCO exports may keep relative path in file_name.
                candidate = self.data_root / file_name
                if candidate.exists():
                    image_path = candidate
                else:
                    continue

            img_anns = anns_by_img.get(image_id, [])
            samples.append(
                {
                    "image_id": image_id,
                    "file_name": file_name,
                    "image_path": image_path,
                    "width": int(img.get("width", 0)),
                    "height": int(img.get("height", 0)),
                    "annotations": img_anns,
                }
            )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _to_train_label(self, ann: Dict) -> Optional[int]:
        if self.single_class:
            return 0

        category_id = int(ann.get("category_id", -1))
        if self.enable_category_mapping:
            if category_id not in self.category_mapping:
                return None
            return int(self.category_mapping[category_id])
        return category_id

    def _decode_mask(self, ann: Dict, h: int, w: int) -> np.ndarray:
        segmentation = ann.get("segmentation", None)
        mask = np.zeros((h, w), dtype=np.uint8)

        if segmentation is None:
            return mask

        if isinstance(segmentation, list):
            # Polygon format: list of flattened coordinate arrays.
            for poly in segmentation:
                if not isinstance(poly, list) or len(poly) < 6:
                    continue
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                pts = np.round(pts).astype(np.int32)
                cv2.fillPoly(mask, [pts], 1)
            return mask

        if isinstance(segmentation, dict) and "counts" in segmentation and "size" in segmentation:
            try:
                from pycocotools import mask as mask_utils
            except Exception as exc:
                raise RuntimeError(
                    "RLE segmentation found but pycocotools is not available"
                ) from exc
            decoded = mask_utils.decode(segmentation)
            if decoded.ndim == 3:
                decoded = decoded[..., 0]
            return decoded.astype(np.uint8)

        return mask

    def _parse_instances(self, sample: Dict, img_h: int, img_w: int) -> List[Dict]:
        instances: List[Dict] = []
        for ann in sample["annotations"]:
            if int(ann.get("iscrowd", 0)) == 1:
                continue

            label = self._to_train_label(ann)
            if label is None:
                continue

            mask = self._decode_mask(ann, img_h, img_w)
            if mask.sum() == 0:
                continue

            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if not rows.any() or not cols.any():
                continue

            y_indices = np.where(rows)[0]
            x_indices = np.where(cols)[0]
            y1, y2 = y_indices[0], y_indices[-1] + 1
            x1, x2 = x_indices[0], x_indices[-1] + 1

            y1 = max(0, min(int(y1), img_h - 1))
            y2 = max(1, min(int(y2), img_h))
            x1 = max(0, min(int(x1), img_w - 1))
            x2 = max(1, min(int(x2), img_w))
            if y2 <= y1 or x2 <= x1:
                continue

            instances.append(
                {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "label": int(label),
                    "mask": mask,
                }
            )

        return instances

    def _apply_random_erasing(
        self,
        img: np.ndarray,
        instances: List[Dict],
    ) -> Tuple[np.ndarray, List[Dict]]:
        h, w = img.shape[:2]
        img_area = h * w

        for _ in range(10):
            target_area = img_area * np.random.uniform(self.random_erasing_scale[0], self.random_erasing_scale[1])
            aspect_ratio = np.random.uniform(self.random_erasing_ratio[0], self.random_erasing_ratio[1])

            erase_h = int(round(np.sqrt(target_area * aspect_ratio)))
            erase_w = int(round(np.sqrt(target_area / aspect_ratio)))
            if erase_h >= h or erase_w >= w:
                continue

            y1 = np.random.randint(0, h - erase_h)
            x1 = np.random.randint(0, w - erase_w)
            y2 = y1 + erase_h
            x2 = x1 + erase_w

            erase_mask = np.zeros((h, w), dtype=np.uint8)
            erase_mask[y1:y2, x1:x2] = 1

            overlap_ratio = 0.0
            for inst in instances:
                inst_mask = inst["mask"]
                inter = np.logical_and(inst_mask > 0, erase_mask > 0).sum()
                area = max(int((inst_mask > 0).sum()), 1)
                overlap_ratio = max(overlap_ratio, inter / area)

            if overlap_ratio > 0.5:
                continue

            fill = np.random.randint(0, 256, size=(erase_h, erase_w, 3), dtype=np.uint8)
            img[y1:y2, x1:x2] = fill
            break

        return img, instances

    def _multi_scale_resize(self, img, instances):
        """mmdet-style multi-scale resize with keep_ratio.

        Operates on numpy (H,W,C) BGR uint8 image. Mutates and returns img
        and the instances list (bbox + mask fields updated in place).

        For 'range' mode: long_edge ~ U[long_min, long_max],
        short_edge ~ U[short_min, short_max], scale_factor keeps ratio.
        For 'value' mode: pick one (W,H) tuple uniformly from img_scale.
        """
        if self.multi_scale_mode == "value":
            idx = np.random.randint(0, len(self.multi_scale_img_scale))
            target_w, target_h = self.multi_scale_img_scale[idx]
            orig_h, orig_w = img.shape[:2]
            scale_factor = min(target_w / orig_w, target_h / orig_h)
        else:
            # 'range' mode (default)
            long_edge = np.random.randint(
                self._ms_long_range[0], self._ms_long_range[1] + 1
            )
            short_edge = np.random.randint(
                self._ms_short_range[0], self._ms_short_range[1] + 1
            )
            orig_h, orig_w = img.shape[:2]
            orig_long = max(orig_w, orig_h)
            orig_short = min(orig_w, orig_h)
            scale_factor = min(
                long_edge / orig_long, short_edge / orig_short
            )

        new_w = int(round(orig_w * scale_factor))
        new_h = int(round(orig_h * scale_factor))

        # Avoid degenerate resize (and skip if no-op).
        if (new_w, new_h) == (orig_w, orig_h):
            return img, instances
        # Guard against zero-size outputs.
        new_w = max(new_w, 1)
        new_h = max(new_h, 1)

        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        for inst in instances:
            x1, y1, x2, y2 = inst["bbox"]
            inst["bbox"] = [
                x1 * scale_factor, y1 * scale_factor,
                x2 * scale_factor, y2 * scale_factor,
            ]
            mask = inst.get("mask")
            if mask is not None and getattr(mask, "size", 0) > 0:
                inst["mask"] = cv2.resize(
                    mask.astype(np.uint8), (new_w, new_h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.uint8)
        return img, instances

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        image_path = sample["image_path"]

        img = cv2.imread(str(image_path))
        if img is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        orig_h, orig_w = img.shape[:2]

        instances = self._parse_instances(sample, orig_h, orig_w)

        # Multi-scale random resize (mmdet range mode). Runs on the numpy
        # image before the fixed resize-to-image_size. The fixed resize below
        # then brings the result back to image_size for the network input.
        if self.multi_scale_resize_prob > 0 and np.random.rand() < self.multi_scale_resize_prob:
            img, instances = self._multi_scale_resize(img, instances)
            orig_h, orig_w = img.shape[:2]

        scale_w, scale_h = 1.0, 1.0
        if (orig_w, orig_h) != self.image_size:
            scale_h = self.image_size[1] / orig_h
            scale_w = self.image_size[0] / orig_w
            img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LINEAR)

            for inst in instances:
                x1, y1, x2, y2 = inst["bbox"]
                inst["bbox"] = [x1 * scale_w, y1 * scale_h, x2 * scale_w, y2 * scale_h]
                inst["mask"] = cv2.resize(inst["mask"].astype(np.uint8), self.image_size, interpolation=cv2.INTER_NEAREST).astype(np.uint8)

        if self.flip_prob > 0 and np.random.rand() < self.flip_prob:
            img = img[:, ::-1].copy()
            w = self.image_size[0]
            for inst in instances:
                x1, y1, x2, y2 = inst["bbox"]
                inst["bbox"] = [w - x2, y1, w - x1, y2]
                inst["mask"] = inst["mask"][:, ::-1].copy()

        if self.vflip_prob > 0 and np.random.rand() < self.vflip_prob:
            img = img[::-1, :].copy()
            h = self.image_size[1]
            for inst in instances:
                x1, y1, x2, y2 = inst["bbox"]
                inst["bbox"] = [x1, h - y2, x2, h - y1]
                inst["mask"] = inst["mask"][::-1, :].copy()

        if self.gaussian_noise_prob > 0 and np.random.rand() < self.gaussian_noise_prob:
            noise = np.random.randn(*img.shape) * (self.gaussian_noise_std * 255)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        if self.random_erasing_prob > 0 and np.random.rand() < self.random_erasing_prob:
            img, instances = self._apply_random_erasing(img, instances)

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()

        if len(instances) == 0:
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = np.zeros((0, *self.image_size), dtype=np.uint8)
        else:
            bboxes = torch.tensor([inst["bbox"] for inst in instances], dtype=torch.float32)
            labels = torch.tensor([inst["label"] for inst in instances], dtype=torch.int64)
            masks = np.stack([inst["mask"] for inst in instances], axis=0)

        img_metas = {
            "img_shape": self.image_size,
            "ori_shape": (orig_h, orig_w),
            "pad_shape": self.image_size,
            "batch_input_shape": self.image_size,
            "scale_factor": (scale_w, scale_h),
            "filename": str(image_path),
            "scene_id": str(sample["image_id"]),
            "image_id": int(sample["image_id"]),
        }

        return {
            "img": img_tensor,
            "img_metas": img_metas,
            "gt_bboxes": bboxes,
            "gt_labels": labels,
            "gt_masks": masks,
            "scene_id": str(sample["image_id"]),
        }
