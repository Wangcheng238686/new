"""C4 dense-gate strength and PromptEncoder mask-path ablations."""

import os

_base_ = ["./whu1024_baseplus_explicit_coarse.py"]

_train_mask_downscaling_raw = os.environ.get(
    "PROMPT_ENCODER_TRAIN_MASK_DOWNSCALING", "0"
)
if _train_mask_downscaling_raw not in {"0", "1"}:
    raise ValueError(
        "PROMPT_ENCODER_TRAIN_MASK_DOWNSCALING must be 0 or 1, got "
        f"{_train_mask_downscaling_raw!r}"
    )

_mask_head_override = dict(
    shape_prior_cfg=dict(
        shape_scale_mode="fixed",
        prompt_scale_init=float(os.environ.get("SHAPE_DENSE_ALPHA_INIT", "0.5")),
    )
)
if _train_mask_downscaling_raw == "1":
    _mask_head_override["prompt_encoder_cfg"] = dict(
        train_mask_downscaling=True,
    )

model = dict(roi_head=dict(mask_head=_mask_head_override))
