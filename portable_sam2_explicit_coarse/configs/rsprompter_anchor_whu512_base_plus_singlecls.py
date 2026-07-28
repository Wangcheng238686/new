"""
WHU-512 single-class config for RSPrompter + SAM2 base_plus.
Adapted for the 512x512 pre-cropped WHU dataset with COCO annotations
generated from binary masks via tools/mask_to_coco_whu512.py.
"""

import os

_base_ = ["./rsprompter_anchor_satS_v11_sam2_large_full.py"]

work_dir = os.environ.get(
    "WORK_DIR",
    "./work_dirs/portable_sam_fusion/rsprompter_anchor_whu512_base_plus_singlecls",
)

data_root = os.environ.get(
    "WHU512_DATA_ROOT",
    "/data/wangcheng/dataset/WHU-512",
)

sam2_ckpt_path = os.environ.get(
    "SAM2_CKPT",
    "/data/wangcheng/pretrained-models/sam2/sam2_hiera_base_plus.pt",
)

num_classes = 1

model = dict(
    backbone=dict(
        checkpoint_path=sam2_ckpt_path,
        init_cfg=dict(type="Pretrained", checkpoint=sam2_ckpt_path),
        model_size="base_plus",
    ),
    neck=dict(
        in_channels="sam2_hiera_base",
        fpn_type="fused",
    ),
    roi_head=dict(
        topdown_downstream_head="none",
        bbox_head=dict(num_classes=num_classes),
        mask_head=dict(
            sam2_mask_decoder=dict(checkpoint_path=sam2_ckpt_path),
            dense_prompt_mode="none",
        ),
    ),
)

dataset_type = "WHUCocoSingleClass"
train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotations/train.json",
        data_prefix=dict(img="train/image"),
    ),
)

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotations/val.json",
        data_prefix=dict(img="val/image"),
    ),
)

test_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotations/test.json",
        data_prefix=dict(img="test/image"),
    ),
)

val_evaluator = dict(
    ann_file=os.path.join(data_root, "annotations/val.json"),
)

test_evaluator = dict(
    ann_file=os.path.join(data_root, "annotations/test.json"),
)
