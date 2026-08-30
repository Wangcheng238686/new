"""NWPU VHR-10 variant with the SAM2 Hiera LARGE encoder (224M).

Same recipe as vhr10_baseplus_explicit_coarse (10 classes, RSPrompter 520/130
split); only the frozen SAM2 backbone is upgraded base_plus(80M) -> large
(224M) to close the mask-boundary-sharpness gap against the ViT-H-based
RSPrompter baseline.  All SAM2-side weights (vision encoder, mask decoder,
prompt encoder) load from the SAME large checkpoint so no cross-variant
weight mixing can occur (the mask head enforces this at load time).
"""

import os

_base_ = ["./vhr10_baseplus_explicit_coarse.py"]

sam2_large_ckpt = os.environ.get(
    "SAM2_CKPT",
    "/data/wangcheng/pretrained-models/sam2/sam2_hiera_large.pt",
)

model = dict(
    backbone=dict(
        checkpoint_path=sam2_large_ckpt,
        init_cfg=dict(type="Pretrained", checkpoint=sam2_large_ckpt),
        model_size="large",
    ),
    neck=dict(in_channels="sam2_hiera_large"),
    roi_head=dict(
        mask_head=dict(
            sam2_mask_decoder=dict(checkpoint_path=sam2_large_ckpt),
        ),
    ),
)
