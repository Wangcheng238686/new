"""Satellite-only adapter for the fixed minimal archive; COCO RLE is authoritative."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mask_utils
from torchvision.transforms import ColorJitter, RandomApply
from torch.utils.data import DataLoader, DistributedSampler

from .whu_instance_dataset import WHUCocoInstanceDataset
from .loader import rtmdet_collate_fn, _seed_worker


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archive_identity(root):
    root = Path(root)
    # Test annotations are hashed for identity only; they are never decoded/evaluated here.
    files = ['dataset.json', 'SHA256SUMS.txt'] + [f'{kind}/{s}.json'
        for kind in ('splits', 'annotations') for s in ('train', 'val', 'test')]
    return {name: sha(root / name) for name in files}


class MinimalSatelliteDataset(WHUCocoInstanceDataset):
    def __init__(self, root, split, augment=False):
        if split not in ('train', 'val', 'test'):
            raise ValueError(split)
        self.split = split
        self.color = RandomApply([ColorJitter(brightness=.15, contrast=.15,
                                  saturation=.10, hue=.02)], p=.5) if augment else None
        super().__init__(root, f'annotations/{split}.json', '.', image_size=(512,512),
                         gaussian_noise_prob=.5 if augment else 0., gaussian_noise_std=.02)
        ids = json.loads((Path(root)/'splits'/f'{split}.json').read_text())
        if [x['image_id'] for x in self.samples] != ids:
            raise ValueError('COCO/split image order or membership mismatch')
        if any((s['width'],s['height']) != (512,512) for s in self.samples):
            raise ValueError('Minimal protocol requires native512')

    def _decode_mask(self, ann, h, w):
        rle = ann['segmentation']
        if not isinstance(rle, dict) or rle['size'] != [h,w]:
            raise ValueError('Expected independent COCO RLE')
        if isinstance(rle['counts'], list):
            rle = mask_utils.frPyObjects(rle,h,w)
        mask = mask_utils.decode(rle)
        if mask.shape != (h,w) or int(mask.sum()) != int(ann['area']):
            raise ValueError('RLE shape/area mismatch')
        return mask.astype(np.uint8)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        if self.color is not None:
            # Base loader is BGR 0..255; torchvision hue expects RGB uint8.
            rgb = item['img'][[2,1,0]].to(torch.uint8)
            item['img'] = self.color(rgb)[[2,1,0]].float()
        return item


def create_minimal_train_loader(*,data_root,batch_size,val_batch_size,distributed,rank,
                                world_size,seed,train_subset_ratio,val_subset_ratio,
                                image_size,**unused):
    if train_subset_ratio != 1 or val_subset_ratio != 1 or tuple(image_size)!=(512,512):
        raise ValueError('Minimal full protocol forbids subsets/resizing')
    train = MinimalSatelliteDataset(data_root,'train',augment=True)
    val = MinimalSatelliteDataset(data_root,'val')
    identity = archive_identity(data_root)
    train.data_config = dict(dataset_format='citysample_minimal',data_root=str(Path(data_root).resolve()),
        manifest=str(Path(data_root).resolve()/'dataset.json'),manifest_sha256=identity['dataset.json'],
        archive_hashes=identity,image_size=[512,512],single_class=True,num_classes=1,
        validation=dict(ann_file='annotations/val.json',image_subdir='.'),
        test=dict(ann_file='annotations/test.json',image_subdir='.'))
    sampler = DistributedSampler(train,num_replicas=world_size,rank=rank,shuffle=True,
                                 seed=seed,drop_last=True) if distributed else None
    def loader(ds,bs,training):
        gen=torch.Generator().manual_seed(seed if training else seed+10000)
        return DataLoader(ds,batch_size=bs,shuffle=training and sampler is None,
            sampler=sampler if training else None,num_workers=4,drop_last=training,
            pin_memory=True,collate_fn=rtmdet_collate_fn,worker_init_fn=_seed_worker,
            generator=gen,persistent_workers=True)
    return loader(train,batch_size,True),loader(val,val_batch_size,False),train


def resolve_minimal_data(args,snapshot):
    data=copy.deepcopy(snapshot['data_config'])
    root=Path(args.data_root or data['data_root'])
    if archive_identity(root)!=data['archive_hashes']:
        raise ValueError('Frozen minimal archive identity changed')
    if args.split not in ('validation','test'):
        raise ValueError('Minimal evaluation requires validation or test')
    data.update(data_root=str(root),split='val' if args.split=='validation' else 'test',
                batch_size=args.batch_size or 1)
    if args.ann_file or args.image_subdir or args.image_size:
        raise ValueError('Minimal archive controls annotations/images/native size')
    return data


def build_minimal_eval_loader(contract,num_workers):
    ds=MinimalSatelliteDataset(contract['data_root'],contract['split'])
    return DataLoader(ds,batch_size=contract['batch_size'],shuffle=False,num_workers=num_workers,
                      collate_fn=rtmdet_collate_fn,pin_memory=True,drop_last=False)


def validate_minimal_resume(checkpoint,current):
    saved=checkpoint.get('config_snapshot',{})
    for key in ('data_config','evaluation_contract','train_augmentation','training_args','model_config','runtime_config','effective_batch_size','optimizer_steps_per_epoch','total_optimizer_steps'):
        a=copy.deepcopy(saved.get(key,{})); b=copy.deepcopy(current[key])
        if key=='training_args':
            for k in ('resume_from','checkpoint_dir'):
                a.pop(k,None); b.pop(k,None)
        if a!=b:
            raise ValueError(f'Minimal resume identity changed: {key}')
