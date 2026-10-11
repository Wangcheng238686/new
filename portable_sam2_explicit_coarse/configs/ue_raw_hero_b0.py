"""SURFBEV-B0: satellite-only anchor for the strict raw HERO pairset."""
import os

_base_ = ["./whu1024_baseplus_explicit_coarse.py"]

# The raw contract is native 512x512.  Keep the model preprocessor explicit so
# a future base-config change cannot silently resize B0 to a different tile.
model = dict(
    data_preprocessor=dict(batch_augments=[dict(
    type="BatchFixedSizePad", size=(512, 512), img_pad_value=0,
    pad_mask=True, mask_pad_value=0, pad_seg=False,
    )]),
    shared_image_embedding=dict(image_size=512, embedding_stride=16),
    train_cfg=dict(rcnn=dict(mask_size=(512, 512))),
    roi_head=dict(mask_head=dict(prompt_encoder_image_size=512,
        prompt_encoder_embed_size=32, shape_prior_cfg=dict(img_size=512))),
)

train_cfg = dict(max_epochs=int(os.environ.get("MAX_EPOCHS", "100")))
