"""Exact sealed WHU model, with explicit machine-local SAM2 identity."""
import os

_base_ = "./a3sgm_whu1024.py"

# Runtime paths do not change the normalized architecture fingerprint.
sam2_checkpoint = os.environ["SAM2_CKPT"]
model = dict(
    backbone=dict(checkpoint_path=sam2_checkpoint,
                  init_cfg=dict(checkpoint=sam2_checkpoint)),
    roi_head=dict(mask_head=dict(
        sam2_mask_decoder=dict(checkpoint_path=sam2_checkpoint))),
)
