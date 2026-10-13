"""SURFBEV-B0: satellite-only anchor for the strict raw HERO pairset."""
import os

_base_ = ["./whu1024_baseplus_explicit_coarse.py"]

# Keep the proven 1024px model contract explicit.  The dataset upsamples the
# original 512px satellite RGB/ID raster before it reaches the model.
model = dict(
    data_preprocessor=dict(batch_augments=[dict(
    type="BatchFixedSizePad", size=(1024, 1024), img_pad_value=0,
    pad_mask=True, mask_pad_value=0, pad_seg=False,
    )]),
    shared_image_embedding=dict(image_size=1024, embedding_stride=16),
    train_cfg=dict(rcnn=dict(mask_size=(1024, 1024))),
    roi_head=dict(mask_head=dict(prompt_encoder_image_size=1024,
        prompt_encoder_embed_size=64, shape_prior_cfg=dict(img_size=1024))),
)

train_cfg = dict(max_epochs=int(os.environ.get("MAX_EPOCHS", "100")))
