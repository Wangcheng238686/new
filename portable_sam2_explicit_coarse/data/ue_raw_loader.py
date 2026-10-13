"""Deterministic satellite-only raw-pair loaders, no augmentation."""
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .ue_raw_pair_dataset import UERawPairDataset


def create_raw_loaders(root, batch_size, val_batch_size, image_size, split_manifest,
                       train_ratio, val_ratio, seed, num_workers, distributed,
                       rank, world_size):
    from .loader import rtmdet_collate_fn, _seed_worker
    from ue.contract import raw_data_contract

    raw_data_contract(root, split_manifest)
    datasets = []
    for split, ratio in (("train", train_ratio), ("val", val_ratio)):
        if not 0 < ratio <= 1:
            raise ValueError("subset ratios must be in (0,1]")
        ds = UERawPairDataset(root, split, load_uav=False,
                              split_manifest=split_manifest, image_size=image_size)
        ds.full_size = len(ds)
        indices = sorted(np.random.RandomState(seed).permutation(len(ds))[:max(1, int(len(ds)*ratio))].tolist())
        ds.records = [ds.records[i] for i in indices]
        ds.subset_indices, ds.subset_ratio = indices, ratio
        datasets.append(ds)
    sampler = DistributedSampler(datasets[0], num_replicas=world_size, rank=rank,
                                 shuffle=True, seed=seed) if distributed else None
    common = dict(num_workers=num_workers, collate_fn=rtmdet_collate_fn,
                  pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0,
                  worker_init_fn=_seed_worker)
    train = DataLoader(datasets[0], batch_size=batch_size, sampler=sampler,
                       shuffle=sampler is None, generator=torch.Generator().manual_seed(seed), **common)
    val = DataLoader(datasets[1], batch_size=val_batch_size or batch_size,
                     shuffle=False, **common)
    return train, val, datasets[0]
