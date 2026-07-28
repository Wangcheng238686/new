"""Explicit coarse-mask prompt route on the clean WHU-1024 Base+ baseline."""

import os

_base_ = ["./whu1024_baseplus_clean.py"]

_mode = os.environ.get("EXPLICIT_PROMPT_MODE", "points_box_dense").strip().lower()
if _mode not in {"points", "points_box", "points_box_dense"}:
    raise ValueError(f"Unsupported EXPLICIT_PROMPT_MODE={_mode!r}")

_use_dense = _mode == "points_box_dense"
_densebr = os.environ.get("DENSEBR_ENABLED", "0") == "1"
_shape_context_fusion = os.environ.get(
    "SHAPE_CONTEXT_FUSION", "roi_only"
).strip().lower()
if _shape_context_fusion not in {
    "roi_only",
    "gated_spatial_film",
    "legacy_multiplicative",
}:
    raise ValueError(
        "SHAPE_CONTEXT_FUSION must be roi_only, gated_spatial_film or "
        f"legacy_multiplicative, got {_shape_context_fusion!r}"
    )
_densebr_detach_raw = os.environ.get("DENSEBR_DETACH_PROMPT_CUES", "1")
if _densebr_detach_raw not in {"0", "1"}:
    raise ValueError(
        "DENSEBR_DETACH_PROMPT_CUES must be 0 or 1, "
        f"got {_densebr_detach_raw!r}"
    )
_epochs = int(os.environ.get("MAX_EPOCHS", "80"))
_adaptive_validity_raw = os.environ.get("SHAPE_POINT_ADAPTIVE_VALIDITY", "1")
if _adaptive_validity_raw not in {"0", "1"}:
    raise ValueError(
        "SHAPE_POINT_ADAPTIVE_VALIDITY must be 0 or 1, "
        f"got {_adaptive_validity_raw!r}"
    )
_adaptive_validity = _adaptive_validity_raw == "1"
_shape_loss_mode = os.environ.get(
    "SHAPE_LOSS_SCHEDULE_MODE", "fixed"
).strip().lower()
if _shape_loss_mode not in {"fixed", "two_stage"}:
    raise ValueError(
        "SHAPE_LOSS_SCHEDULE_MODE must be fixed or two_stage, "
        f"got {_shape_loss_mode!r}"
    )
_shape_loss_weight = float(os.environ.get("SHAPE_PRIOR_LOSS_WEIGHT", "0.10"))
_shape_stage1_end = int(os.environ.get("SHAPE_LOSS_STAGE1_END", "5"))
_shape_stage1_weight = float(
    os.environ.get("SHAPE_LOSS_WEIGHT_STAGE1", "0.20")
)
_shape_stage2_weight = float(
    os.environ.get("SHAPE_LOSS_WEIGHT_STAGE2", "0.10")
)
if _shape_loss_weight < 0 or _shape_stage1_weight < 0 or _shape_stage2_weight < 0:
    raise ValueError("coarse loss weights must be non-negative")
if _shape_loss_mode == "two_stage" and not (1 <= _shape_stage1_end < _epochs):
    raise ValueError(
        "two_stage coarse loss requires 1 <= SHAPE_LOSS_STAGE1_END < MAX_EPOCHS"
    )

_point_no_point_epochs = int(
    os.environ.get(
        "POINT_WARMUP_NO_POINT_EPOCHS",
        os.environ.get("POINT_NO_POINT_EPOCHS", "0"),
    )
)
_point_one_pair_epochs = int(
    os.environ.get(
        "POINT_WARMUP_ONE_PAIR_EPOCHS",
        os.environ.get("POINT_ONE_PAIR_EPOCHS", "0"),
    )
)
_point_full_start_epoch = int(
    os.environ.get(
        "POINT_WARMUP_FULL_START_EPOCH",
        os.environ.get(
            "POINT_FULL_START_EPOCH",
            str(_point_no_point_epochs + _point_one_pair_epochs + 1),
        ),
    )
)

model = dict(
    roi_head=dict(
        mask_head=dict(
            prompt_sparse_mode="shape_point",
            explicit_prompt_mode=_mode,
            prompt_encoder_enabled=True,
            prompt_encoder_cfg=dict(
                require_pretrained=True,
                freeze_all=True,
            ),
            checkpoint_load_cfg=dict(strict_reproduction=True),
            freeze_mask_decoder=False,
            freeze_no_mask_embed=True,
            load_no_mask_pretrained=True,
            final_mask_coordinate_mode="full_image",
            shape_base_dense_mode="no_mask",
            shape_prior_loss_weight=_shape_loss_weight,
            shape_prior_cfg=dict(
                enabled=True,
                img_size=1024,
                visual_context_dim=256,
                roi_feat_channels=512,
                coarse_mask_output_size=int(
                    os.environ.get("COARSE_MASK_OUTPUT_SIZE", "64")
                ),
                fusion_type=_shape_context_fusion,
                context_gate_init=0.10,
                zero_init_gamma=True,
                zero_init_beta=True,
                use_shape_dense=_use_dense,
                shape_scale_mode="global_sigmoid",
                prompt_scale_init=float(
                    os.environ.get("SHAPE_DENSE_ALPHA_INIT", "0.25")
                ),
            ),
            shape_point_miner_cfg=dict(
                foreground_thr=0.50,
                max_positive_points=2,
                max_negative_points=2,
                adaptive_validity=_adaptive_validity,
                positive_conf_thr=0.60,
                empty_positive_conf_thr=0.35,
                negative_conf_thr=0.30,
                min_inside_distance_ratio=0.05,
                min_positive_distance_ratio=0.15,
                min_negative_distance_ratio=0.15,
                outer_ring_radius=1,
                safe_background_radius_ratio=0.10,
            ),
            point_warmup_cfg=dict(
                enabled=os.environ.get("POINT_WARMUP_ENABLED", "0") == "1",
                no_point_epochs=_point_no_point_epochs,
                one_pair_epochs=_point_one_pair_epochs,
                full_2p2n_start_epoch=_point_full_start_epoch,
            ),
            dense_prompt_cfg=dict(
                transform=os.environ.get("SHAPE_DENSE_TRANSFORM", "raw_logits"),
                temperature=float(
                    os.environ.get("SHAPE_DENSE_TEMPERATURE", "1.0")
                ),
                outside_fill_logit=float(
                    os.environ.get("SHAPE_DENSE_OUTSIDE_FILL", "0.0")
                ),
                clamp_range=(-8.0, 8.0),
            ),
            restrict_dense_prompt_to_box=True,
            coarse_mask_loss_cfg=dict(
                weight=_shape_loss_weight,
                schedule_mode=_shape_loss_mode,
                weight_schedule=[
                    dict(
                        start_epoch=1,
                        end_epoch=_shape_stage1_end,
                        weight=_shape_stage1_weight,
                    ),
                    dict(
                        start_epoch=_shape_stage1_end + 1,
                        end_epoch=_epochs,
                        weight=_shape_stage2_weight,
                    ),
                ] if _shape_loss_mode == "two_stage" else [],
                bce_weight=1.0,
                dice_weight=1.0,
                boundary_weight=0.0,
                distance_weight=0.0,
            ),
            # Shared2FC has no SABL decoded-box tuple.  Proposal boxes are the
            # canonical train prompt source; final detected boxes are used at test.
            mask_prompt_box_cfg=dict(
                train_box_source="proposal",
                test_box_source="proposal",
                fallback_to_proposal=True,
                min_box_size=2.0,
                clamp_to_image=True,
            ),
            box_prompt_cfg=dict(jitter_prob=0.0),
            densebr_cfg=dict(
                enabled=_densebr,
                roi_channels=int(os.environ.get("DENSEBR_ROI_CHANNELS", "64")),
                cue_channels=int(os.environ.get("DENSEBR_CUE_CHANNELS", "32")),
                mid_channels=int(os.environ.get("DENSEBR_MID_CHANNELS", "64")),
                beta_init=float(os.environ.get("DENSEBR_BETA_INIT", "0.05")),
                beta_max=float(os.environ.get("DENSEBR_BETA_MAX", "0.20")),
                delta_logit_max=float(os.environ.get("DENSEBR_DELTA_LOGIT_MAX", "2.0")),
                quality_gate_init=float(os.environ.get("DENSEBR_QUALITY_GATE_INIT", "0.50")),
                image_size=1024,
                foreground_thr=0.50,
                detach_prompt_cues=_densebr_detach_raw == "1",
                zero_init_residual=True,
            ),
            prune_unused_prompt_components=False,
        )
    )
)

train_cfg = dict(max_epochs=_epochs, val_interval=1)
