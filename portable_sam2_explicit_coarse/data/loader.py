import logging
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .satellite_dataset import SatelliteInstanceDataset
from .whu_instance_dataset import WHUCocoInstanceDataset
from .satellite_drone_dataset import (
    SatelliteDroneDataset,
    rtmdet_drone_collate_fn,
    build_scene_id_mapping,
)


logger = logging.getLogger(__name__)


def _validate_subset_ratio(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be in (0, 1], got {value}")
    return value


def _subset_whu_dataset(dataset, ratio: float, seed: int, split: str):
    """Apply a deterministic image-level subset while preserving sample order."""
    ratio = _validate_subset_ratio(ratio, f"{split}_subset_ratio")
    full_size = len(dataset)
    if ratio >= 1.0 or full_size == 0:
        dataset.subset_ratio = ratio
        dataset.full_size = full_size
        dataset.subset_indices = list(range(full_size))
        return dataset

    subset_size = max(1, int(full_size * ratio))
    rng = np.random.RandomState(int(seed))
    indices = sorted(rng.permutation(full_size)[:subset_size].tolist())
    dataset.samples = [dataset.samples[index] for index in indices]
    dataset.subset_ratio = ratio
    dataset.full_size = full_size
    dataset.subset_indices = indices
    logger.info(
        "WHU %s subset: %d/%d images (ratio=%.6f, seed=%d)",
        split,
        len(dataset),
        full_size,
        ratio,
        int(seed),
    )
    return dataset


def _seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)


def rtmdet_collate_fn(batch):
    import numpy as np
    import torch
    import torch.nn.functional as F

    imgs = []
    img_metas = []
    gt_bboxes = []
    gt_labels = []
    gt_masks = []

    max_instances = max(len(item["gt_labels"]) for item in batch)

    for item in batch:
        imgs.append(item["img"])
        img_metas.append(item["img_metas"])

        bboxes = item["gt_bboxes"]
        labels = item["gt_labels"]
        masks = item["gt_masks"]

        num_inst = len(labels)
        if num_inst < max_instances:
            pad_size = max_instances - num_inst
            bboxes = F.pad(bboxes, (0, 0, 0, pad_size))
            labels = F.pad(labels, (0, pad_size), value=-1)
            masks = np.concatenate(
                [
                    masks,
                    np.zeros(
                        (pad_size, *item["img_metas"]["img_shape"]), dtype=np.uint8
                    ),
                ],
                axis=0,
            )

        gt_bboxes.append(bboxes)
        gt_labels.append(labels)
        gt_masks.append(masks)

    return {
        "imgs": torch.stack(imgs),
        "img_metas": img_metas,
        "gt_bboxes": gt_bboxes,
        "gt_labels": gt_labels,
        "gt_masks": gt_masks,
    }


def create_train_loader(
    data_root: str,
    batch_size: int,
    flip_prob: float = 0.0,
    vflip_prob: float = 0.0,
    gaussian_noise_prob: float = 0.0,
    gaussian_noise_std: float = 0.02,
    random_erasing_prob: float = 0.0,
    random_erasing_scale: Tuple[float, float] = (0.02, 0.2),
    random_erasing_ratio: Tuple[float, float] = (0.3, 3.3),
    image_size: Tuple[int, int] = (512, 512),
    multi_scale_resize_prob: float = 0.0,
    multi_scale_mode: str = "range",
    multi_scale_img_scale=None,
    use_drone: bool = False,
    drone_data_root: str = "",
    num_views: int = 15,
    drone_image_size: Tuple[int, int] = (512, 512),
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    val_ratio: float = 0.0,
    val_batch_size: Optional[int] = None,
    num_workers: int = 4,
    normalize_drone: bool = True,
    random_sample: bool = True,
    seed: int = 42,
    dataset_format: str = "labelme",
    whu_train_ann_file: str = "annotations/train.json",
    whu_val_ann_file: str = "annotations/val.json",
    whu_train_img_subdir: str = "train/image",
    whu_val_img_subdir: str = "val/image",
    whu_single_class: bool = True,
    whu_enable_category_mapping: bool = False,
    whu_category_mapping: Optional[dict] = None,
    train_subset_ratio: float = 1.0,
    val_subset_ratio: float = 1.0,
):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    if use_drone:
        scene_id_to_index = build_scene_id_mapping(data_root, drone_data_root)
        logger.info("Built scene ID mapping: %d scenes", len(scene_id_to_index))
        
        train_dataset = SatelliteDroneDataset(
            satellite_data_root=data_root,
            drone_data_root=drone_data_root,
            scene_ids=None,
            image_size=image_size,
            drone_image_size=drone_image_size,
            num_sample_images=num_views,
            random_sample=random_sample,
            normalize_drone=normalize_drone,
            val_ratio=val_ratio,
            is_val=False,
            flip_prob=flip_prob,
            scene_id_to_index=scene_id_to_index,
            gaussian_noise_prob=gaussian_noise_prob,
            gaussian_noise_std=gaussian_noise_std,
            random_erasing_prob=random_erasing_prob,
            random_erasing_scale=random_erasing_scale,
            random_erasing_ratio=random_erasing_ratio,
            seed=seed,
        )
        collate_fn = rtmdet_drone_collate_fn
        if val_ratio > 0:
            val_dataset = SatelliteDroneDataset(
                satellite_data_root=data_root,
                drone_data_root=drone_data_root,
                scene_ids=None,
                image_size=image_size,
                drone_image_size=drone_image_size,
                num_sample_images=num_views,
                random_sample=False,
                normalize_drone=normalize_drone,
                val_ratio=val_ratio,
                is_val=True,
                scene_id_to_index=scene_id_to_index,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=val_batch_size if val_batch_size else batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
                worker_init_fn=_seed_worker,
                generator=generator,
            )
        else:
            val_loader = None
    elif dataset_format == "whu_coco":
        train_dataset = WHUCocoInstanceDataset(
            data_root=data_root,
            ann_file=whu_train_ann_file,
            image_subdir=whu_train_img_subdir,
            image_size=image_size,
            single_class=whu_single_class,
            enable_category_mapping=whu_enable_category_mapping,
            category_mapping=whu_category_mapping,
            flip_prob=flip_prob,
            vflip_prob=vflip_prob,
            gaussian_noise_prob=gaussian_noise_prob,
            gaussian_noise_std=gaussian_noise_std,
            random_erasing_prob=random_erasing_prob,
            random_erasing_scale=random_erasing_scale,
            random_erasing_ratio=random_erasing_ratio,
            multi_scale_resize_prob=multi_scale_resize_prob,
            multi_scale_mode=multi_scale_mode,
            multi_scale_img_scale=multi_scale_img_scale,
        )
        train_dataset = _subset_whu_dataset(
            train_dataset,
            ratio=train_subset_ratio,
            seed=int(seed),
            split="train",
        )
        collate_fn = rtmdet_collate_fn

        val_loader = None
        if whu_val_ann_file:
            val_dataset = WHUCocoInstanceDataset(
                data_root=data_root,
                ann_file=whu_val_ann_file,
                image_subdir=whu_val_img_subdir,
                image_size=image_size,
                single_class=whu_single_class,
                enable_category_mapping=whu_enable_category_mapping,
                category_mapping=whu_category_mapping,
                flip_prob=0.0,
                vflip_prob=0.0,
                gaussian_noise_prob=0.0,
                random_erasing_prob=0.0,
            )
            val_dataset = _subset_whu_dataset(
                val_dataset,
                ratio=val_subset_ratio,
                seed=int(seed) + 10000,
                split="val",
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=val_batch_size or batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
                drop_last=False,
                persistent_workers=num_workers > 0,
                worker_init_fn=_seed_worker,
                generator=generator,
            )
            logger.info("创建 WHU 验证集加载器: %d 样本", len(val_dataset))
    else:
        train_dataset = SatelliteInstanceDataset(
            satellite_data_root=data_root,
            scene_ids=None,
            image_size=image_size,
            val_ratio=val_ratio,
            is_val=False,
            flip_prob=flip_prob,
            vflip_prob=vflip_prob,
            gaussian_noise_prob=gaussian_noise_prob,
            gaussian_noise_std=gaussian_noise_std,
            random_erasing_prob=random_erasing_prob,
            random_erasing_scale=random_erasing_scale,
            random_erasing_ratio=random_erasing_ratio,
            seed=seed,
        )
        collate_fn = rtmdet_collate_fn

        val_loader = None
        if val_ratio > 0:
            val_dataset = SatelliteInstanceDataset(
                satellite_data_root=data_root,
                scene_ids=None,
                image_size=image_size,
                val_ratio=val_ratio,
                is_val=True,
                flip_prob=0.0,
                vflip_prob=0.0,
                gaussian_noise_prob=0.0,
                random_erasing_prob=0.0,
                seed=seed,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=val_batch_size or batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
                drop_last=False,
                persistent_workers=num_workers > 0,
                worker_init_fn=_seed_worker,
                generator=generator,
            )
            logger.info("创建验证集加载器: %d 样本", len(val_dataset))

    sampler = None
    shuffle = True
    if distributed and world_size > 1:
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
            seed=int(seed),
        )
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=_seed_worker,
        generator=generator,
    )

    return train_loader, val_loader, train_dataset


def create_test_loader(
    data_root: str,
    ann_file: str,
    image_subdir: str,
    image_size: Tuple[int, int] = (1024, 1024),
    batch_size: int = 1,
    num_workers: int = 4,
    seed: int = 42,
):
    """Build a WHU-COCO test loader for standalone test-set evaluation.

    Mirrors the validation loader built by ``create_train_loader`` (whu_coco
    branch) but points at the held-out test split: no augmentation, no subset
    sampling, deterministic order.  The collate function matches the val/train
    loaders so existing predict/COCO-eval code is reused unchanged.
    """
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    test_dataset = WHUCocoInstanceDataset(
        data_root=data_root,
        ann_file=ann_file,
        image_subdir=image_subdir,
        image_size=image_size,
        single_class=True,
        enable_category_mapping=False,
        category_mapping=None,
        flip_prob=0.0,
        vflip_prob=0.0,
        gaussian_noise_prob=0.0,
        random_erasing_prob=0.0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=rtmdet_collate_fn,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
        worker_init_fn=_seed_worker,
        generator=generator,
    )
    logger.info("创建 WHU 测试集加载器: %d 样本", len(test_dataset))
    return test_loader
