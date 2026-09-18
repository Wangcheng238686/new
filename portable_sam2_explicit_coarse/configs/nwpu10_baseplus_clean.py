"""Clean NWPU VHR-10 / SAM2 Base+ detector baseline (10 classes).

Ours base config for the NWPU VHR-10 benchmark.  Structurally identical to
``whu1024_baseplus_clean.py`` except:

- ``bbox_head.num_classes=10`` (airplane, ship, storage_tank, baseball_diamond,
  tennis_court, basketball_court, ground_track_field, harbor, bridge, vehicle).
- The SAM2 mask head stays class-agnostic (prompt-conditioned SAM2 decoder);
  only the detection head carries the class vocabulary.
- Data wiring points at the NWPU COCO split; the 130-image val split doubles
  as the test split (protocol: no separate test json).
- ``VHR10_DATA_ROOT`` replaces ``WHU1024_DATA_ROOT`` (the NWPU root contains a
  space: "/data1/wangcheng/dataset/NWPU VHR-10 dataset").

Training-side dataset knobs (ann files / image subdir / multi-class mapping)
are consumed from the environment by train_rsprompter_fusion.py; see
scripts/run_nwpu10_explicit_coarse.sh.
"""

import os

_base_ = ["./rsprompter_anchor_satS_v11_sam2_large_full.py"]

data_root = os.environ.get(
    "VHR10_DATA_ROOT", "/data1/wangcheng/dataset/NWPU VHR-10 dataset"
)
sam2_ckpt_path = os.environ.get(
    "SAM2_CKPT",
    "/data/wangcheng/pretrained-models/sam2/sam2_hiera_base_plus.pt",
)
sam_image_embedding_stride = int(
    os.environ.get("SAM_IMAGE_EMBED_STRIDE", "32")
)
if sam_image_embedding_stride not in (16, 32):
    raise ValueError(
        "SAM_IMAGE_EMBED_STRIDE must be 16 (official 64x64) or "
        f"32 (legacy 32x32), got {sam_image_embedding_stride}"
    )
sam_image_embedding_size = 1024 // sam_image_embedding_stride
final_mask_coordinate_mode = os.environ.get(
    "FINAL_MASK_COORDINATE_MODE", "roi_local"
).strip().lower()
if final_mask_coordinate_mode not in {"roi_local", "full_image"}:
    raise ValueError(
        "FINAL_MASK_COORDINATE_MODE must be roi_local or full_image, "
        f"got {final_mask_coordinate_mode!r}"
    )
segm_score_mode = os.environ.get("SEGM_SCORE_MODE", "detector").strip().lower()
if segm_score_mode not in {"detector", "mask_quality"}:
    raise ValueError(
        "SEGM_SCORE_MODE must be detector or mask_quality, "
        f"got {segm_score_mode!r}"
    )

model = dict(
    type="RSPrompterAnchor",
    sam_image_embedding_stride=sam_image_embedding_stride,
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
    shared_image_embedding=dict(
        image_size=1024,
        embedding_stride=sam_image_embedding_stride,
    ),
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
        bbox_head=dict(num_classes=10),
        mask_head=dict(
            sam2_mask_decoder=dict(checkpoint_path=sam2_ckpt_path),
            prompt_encoder_image_size=1024,
            prompt_encoder_embed_size=sam_image_embedding_size,
            prompt_sparse_mode="point",
            prompt_encoder_enabled=False,
            freeze_mask_decoder=False,
            freeze_no_mask_embed=False,
            load_no_mask_pretrained=False,
            final_mask_coordinate_mode=final_mask_coordinate_mode,
            segm_score_mode=segm_score_mode,
            shape_prior_cfg=dict(enabled=False),
            p2_boundary_refiner_cfg=dict(enabled=False),
        ),
    ),
)

dataset_type = "WHUCocoInstanceDataset"
# NWPU VHR-10 protocol: images live in "positive image set" (shared by every
# split); train/val COCO jsons live in coco_split/; val doubles as test.
train_ann_file = "coco_split/NWPU_instances_train.json"
val_ann_file = "coco_split/NWPU_instances_val.json"
image_subdir = "positive image set"
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
        ann_file=train_ann_file,
        data_prefix=dict(img=image_subdir),
        pipeline=train_pipeline,
    )
)
val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=image_subdir),
        pipeline=test_pipeline,
        test_mode=True,
    )
)
test_dataloader = dict(
    dataset=dict(
        type=dataset_type,
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
