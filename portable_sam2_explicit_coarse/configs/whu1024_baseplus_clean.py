"""Clean WHU-1024 / SAM2 Base+ detector baseline.

This is the migration base.  It intentionally contains no IIMR, SABL, UAV or
multi-view branch.  The neck is selectable between the retained legacy
aggregator and the migrated PAFPN through ``NECK_TYPE``.
"""

import os

_base_ = ["./rsprompter_anchor_satS_v11_sam2_large_full.py"]

data_root = os.environ.get("WHU1024_DATA_ROOT", "/data/wangcheng/dataset/WHU")
sam2_ckpt_path = os.environ.get(
    "SAM2_CKPT",
    "/data/wangcheng/pretrained-models/sam2/sam2_hiera_base_plus.pt",
)

model = dict(
    type="RSPrompterAnchor",
    data_preprocessor=dict(
        batch_augments=[
            dict(
                type="BatchFixedSizePad",
                size=(1024, 1024),
                img_pad_value=0,
                pad_mask=True,
                mask_pad_value=0,
                pad_seg=False,
            )
        ]
    ),
    shared_image_embedding=dict(image_size=1024),
    backbone=dict(
        checkpoint_path=sam2_ckpt_path,
        init_cfg=dict(type="Pretrained", checkpoint=sam2_ckpt_path),
        model_size="base_plus",
    ),
    neck=dict(
        in_channels="sam2_hiera_base_plus",
    ),
    train_cfg=dict(
        rcnn=dict(mask_size=(1024, 1024)),
    ),
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(
            sam2_mask_decoder=dict(checkpoint_path=sam2_ckpt_path),
            prompt_encoder_image_size=1024,
            prompt_encoder_embed_size=32,
            prompt_sparse_mode="point",
            prompt_encoder_enabled=False,
            freeze_mask_decoder=False,
            freeze_no_mask_embed=True,
            shape_prior_cfg=dict(enabled=False),
            densebr_cfg=dict(enabled=False),
        ),
    ),
)

dataset_type = "WHUCocoSingleClass"
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
        type=dataset_type,
        data_root=data_root,
        ann_file="2.4 annotation/annotation/train.json",
        data_prefix=dict(img="2.1 train/train"),
        pipeline=train_pipeline,
    )
)
val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="2.4 annotation/annotation/validation.json",
        data_prefix=dict(img="2.3 valid/validation"),
        pipeline=test_pipeline,
        test_mode=True,
    )
)
test_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="2.4 annotation/annotation/test.json",
        data_prefix=dict(img="2.2 test/test"),
        pipeline=test_pipeline,
        test_mode=True,
    )
)
val_evaluator = dict(
    ann_file=os.path.join(data_root, "2.4 annotation/annotation/validation.json")
)
test_evaluator = dict(
    ann_file=os.path.join(data_root, "2.4 annotation/annotation/test.json")
)
