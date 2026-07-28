"""
RSPrompter with SAM2-Hiera-Base+ for satellite single-stream setting.
Keep v11-full training recipe unchanged and only switch SAM2 scale.
"""

import os

_base_ = ["./rsprompter_anchor_satS_v11_sam2_large_full.py"]

work_dir = os.environ.get(
    "WORK_DIR",
    "./work_dirs/portable_sam_fusion/rsprompter_anchor_satS_v11_sam2_base_plus_full",
)

sam2_ckpt_path = os.environ.get(
    "SAM2_CKPT",
    "/data/wangcheng/pretrained-models/sam2/sam2_hiera_base_plus.pt",
)

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
        mask_head=dict(
            sam2_mask_decoder=dict(checkpoint_path=sam2_ckpt_path),
        ),
    ),
)
