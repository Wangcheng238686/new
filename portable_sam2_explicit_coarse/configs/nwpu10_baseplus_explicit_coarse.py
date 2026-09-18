"""Explicit coarse-mask prompt route on the NWPU VHR-10 Base+ baseline.

Full PromptMiner-SAM2 (Ours) config for NWPU VHR-10.  Inherits the complete
WHU-1024 explicit-coarse method config (all environment-variable switches —
EXPLICIT_PROMPT_MODE / SHAPE_* / P2_* / FINAL_MASK_* — keep working) and only
switches:

- the detection head to the 10 NWPU categories (bbox_head.num_classes=10; the
  SAM2 mask head stays class-agnostic via the prompt-conditioned decoder), and
- the dataset wiring to the NWPU COCO split (images in "positive image set",
  shared by every split; the 130-image val split doubles as the test split).

Training-side dataset knobs (ann files / image subdir / multi-class mapping)
are consumed from the environment by train_rsprompter_fusion.py; see
scripts/run_nwpu10_explicit_coarse.sh.
"""

import os

_base_ = ["./whu1024_baseplus_explicit_coarse.py"]

data_root = os.environ.get(
    "VHR10_DATA_ROOT", "/data1/wangcheng/dataset/NWPU VHR-10 dataset"
)
train_ann_file = "coco_split/NWPU_instances_train.json"
val_ann_file = "coco_split/NWPU_instances_val.json"
image_subdir = "positive image set"

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=10),
    ),
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True, poly2mask=False),
    dict(type="Resize", scale=(1024, 1024), keep_ratio=False),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1024, 1024), keep_ratio=False),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True, poly2mask=False),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]
train_dataloader = dict(
    dataset=dict(
        type="WHUCocoInstanceDataset",
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=image_subdir),
        pipeline=train_pipeline,
    )
)
val_dataloader = dict(
    dataset=dict(
        type="WHUCocoInstanceDataset",
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=image_subdir),
        pipeline=test_pipeline,
        test_mode=True,
    )
)
test_dataloader = dict(
    dataset=dict(
        type="WHUCocoInstanceDataset",
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=image_subdir),
        pipeline=test_pipeline,
        test_mode=True,
    )
)
val_evaluator = dict(
    ann_file=os.path.join(data_root, val_ann_file)
)
test_evaluator = dict(
    ann_file=os.path.join(data_root, val_ann_file)
)
