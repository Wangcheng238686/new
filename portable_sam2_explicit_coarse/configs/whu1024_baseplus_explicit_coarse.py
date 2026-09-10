"""Explicit coarse-mask prompt route on the clean WHU-1024 Base+ baseline."""

import os

_base_ = ["./whu1024_baseplus_clean.py"]

_mode = os.environ.get("EXPLICIT_PROMPT_MODE", "points_box_dense").strip().lower()
if _mode not in {"points", "box", "mask", "points_box", "points_box_dense"}:
    raise ValueError(f"Unsupported EXPLICIT_PROMPT_MODE={_mode!r}")

_use_dense = _mode in {"mask", "points_box_dense"}
_canvas_renderer_raw = os.environ.get("CANVAS_RENDERER_ENABLED", "0")
if _canvas_renderer_raw not in {"0", "1"}:
    raise ValueError("CANVAS_RENDERER_ENABLED must be 0 or 1")
_canvas_renderer_enabled = _canvas_renderer_raw == "1"
if _canvas_renderer_enabled and not _use_dense:
    raise ValueError("R3 requires EXPLICIT_PROMPT_MODE=points_box_dense")
_canvas_renderer_loss_weight = float(
    os.environ.get("CANVAS_RENDERER_LOSS_WEIGHT", "0.05")
)
if _canvas_renderer_loss_weight < 0:
    raise ValueError("CANVAS_RENDERER_LOSS_WEIGHT must be non-negative")
_canvas_renderer_override = {}
if _canvas_renderer_enabled:
    # Kept absent for legacy arms, preserving their resolved config/fingerprint.
    _canvas_renderer_override["canvas_renderer_cfg"] = dict(
        enabled=True,
        feature_indices=(1, 2),
        featmap_strides=(8, 16),
        in_channels=512,
        branch_channels=128,
        p3_roi_size=64,
        p4_roi_size=32,
        output_size=128,
        sampling_ratio=0,
        aligned=True,
        loss_weight=_canvas_renderer_loss_weight,
    )
_p2_boundary_refiner = (
    os.environ.get("P2_BOUNDARY_REFINER_ENABLED", "0") == "1"
)
_p2_loss_mode = os.environ.get(
    "P2_BOUNDARY_REFINER_LOSS_MODE", "boundary"
).strip().lower()
if _p2_loss_mode not in {"boundary", "correction_keep"}:
    raise ValueError(
        "P2_BOUNDARY_REFINER_LOSS_MODE must be boundary or correction_keep, "
        f"got {_p2_loss_mode!r}"
    )
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

_dense_transform = os.environ.get(
    "SHAPE_DENSE_TRANSFORM", "raw_logits"
).strip().lower()
if _dense_transform not in {"raw_logits", "confidence_signed", "gaussian_edt"}:
    raise ValueError(f"Unsupported SHAPE_DENSE_TRANSFORM={_dense_transform!r}")
_dense_detach_raw = os.environ.get("SHAPE_DENSE_DETACH", "0")
if _dense_detach_raw not in {"0", "1"}:
    raise ValueError("SHAPE_DENSE_DETACH must be 0 or 1")
_dense_detach = _dense_detach_raw == "1"
_gaussian_foreground_threshold = float(
    os.environ.get("SHAPE_GAUSSIAN_FOREGROUND_THRESHOLD", "0.5")
)
_gaussian_omega = float(os.environ.get("SHAPE_GAUSSIAN_OMEGA", "15.0"))
_gaussian_gamma = float(os.environ.get("SHAPE_GAUSSIAN_GAMMA", "4.0"))
if not 0.0 < _gaussian_foreground_threshold < 1.0:
    raise ValueError("SHAPE_GAUSSIAN_FOREGROUND_THRESHOLD must be in (0,1)")
if _gaussian_omega <= 0.0 or _gaussian_gamma <= 0.0:
    raise ValueError("SHAPE_GAUSSIAN_OMEGA and SHAPE_GAUSSIAN_GAMMA must be positive")
if _dense_transform == "gaussian_edt" and not _dense_detach:
    raise ValueError("gaussian_edt requires SHAPE_DENSE_DETACH=1")

_final_mask_loss_mode = os.environ.get(
    "FINAL_MASK_LOSS_MODE", "standard"
).strip().lower()
if _final_mask_loss_mode not in {"standard", "roi_balanced_dice"}:
    raise ValueError(
        "FINAL_MASK_LOSS_MODE must be standard or roi_balanced_dice, "
        f"got {_final_mask_loss_mode!r}"
    )
_final_mask_loss_override = {}
if _final_mask_loss_mode == "roi_balanced_dice":
    _final_mask_loss_override["final_mask_loss_cfg"] = dict(
        mode=_final_mask_loss_mode,
        roi_expand_ratio=float(os.environ.get("FINAL_MASK_ROI_EXPAND_RATIO", "1.20")),
        roi_bce_weight=float(os.environ.get("FINAL_MASK_ROI_BCE_WEIGHT", "1.0")),
        roi_dice_weight=float(os.environ.get("FINAL_MASK_ROI_DICE_WEIGHT", "1.0")),
        outside_bce_weight=float(
            os.environ.get("FINAL_MASK_OUTSIDE_BCE_WEIGHT", "0.05")
        ),
        eps=1e-6,
    )

_roi_sam_raw = os.environ.get("ROI_SAM_ENABLED", "0")
if _roi_sam_raw not in {"0", "1"}:
    raise ValueError(f"ROI_SAM_ENABLED must be 0 or 1, got {_roi_sam_raw!r}")
_roi_sam_enabled = _roi_sam_raw == "1"
if _roi_sam_enabled and _mode != "points":
    raise ValueError("ROI-SAM is a strict C2 points-only experiment")
if _roi_sam_enabled and _final_mask_loss_mode != "standard":
    raise ValueError("ROI-SAM uses standard ROI-local final-mask supervision")
_roi_sam_override = {}
if _roi_sam_enabled:
    _roi_sam_override["roi_sam_cfg"] = dict(
        enabled=True,
        sampling_ratio=int(os.environ.get("ROI_SAM_SAMPLING_RATIO", "2")),
        aligned=True,
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

_pe_train_mask_downscaling_raw = os.environ.get(
    "PROMPT_ENCODER_TRAIN_MASK_DOWNSCALING", "0"
)
if _pe_train_mask_downscaling_raw not in {"0", "1"}:
    raise ValueError(
        "PROMPT_ENCODER_TRAIN_MASK_DOWNSCALING must be 0 or 1, got "
        f"{_pe_train_mask_downscaling_raw!r}"
    )

_decoder_tail_enabled_raw = os.environ.get("DECODER_TAIL_REFINER_ENABLED", "0")
if _decoder_tail_enabled_raw not in {"0", "1"}:
    raise ValueError("DECODER_TAIL_REFINER_ENABLED must be 0 or 1")
_decoder_tail_enabled = _decoder_tail_enabled_raw == "1"
_decoder_tail_cfg = dict(enabled=_decoder_tail_enabled)
if _decoder_tail_enabled:
    _decoder_tail_cfg.update(
        num_points=int(os.environ.get("DECODER_TAIL_NUM_POINTS", "64")),
        hidden_dim=int(os.environ.get("DECODER_TAIL_HIDDEN_DIM", "128")),
        point_loss_weight=float(os.environ.get("DECODER_TAIL_POINT_LOSS_WEIGHT", "1.0")),
        delta_logit_max=float(os.environ.get("DECODER_TAIL_DELTA_LOGIT_MAX", "2.0")),
    )
    # Keep absence of these keys as the v1 checkpoint contract.  Only the
    # explicit A4/DCR mode materializes new architecture-affecting fields.
    _decoder_tail_mode = os.environ.get("DECODER_TAIL_MODE", "residual_v1")
    if _decoder_tail_mode == "confidence_gated":
        _decoder_tail_gate_init_prob = float(os.environ.get("DECODER_TAIL_GATE_INIT_PROB", "0.1"))
        _decoder_tail_gate_loss_weight = float(os.environ.get("DECODER_TAIL_GATE_LOSS_WEIGHT", "1.0"))
        _decoder_tail_keep_loss_weight = float(os.environ.get("DECODER_TAIL_KEEP_LOSS_WEIGHT", "0.05"))
        if not 0.0 < _decoder_tail_gate_init_prob < 1.0:
            raise ValueError("DECODER_TAIL_GATE_INIT_PROB must lie in (0, 1)")
        if _decoder_tail_gate_loss_weight < 0 or _decoder_tail_keep_loss_weight < 0:
            raise ValueError("DECODER_TAIL gate/keep loss weights must be non-negative")
        _decoder_tail_cfg.update(
            mode="confidence_gated",
            gate_init_prob=_decoder_tail_gate_init_prob,
            gate_loss_weight=_decoder_tail_gate_loss_weight,
            keep_loss_weight=_decoder_tail_keep_loss_weight,
        )
    elif _decoder_tail_mode != "residual_v1":
        raise ValueError(
            "DECODER_TAIL_MODE must be residual_v1 or confidence_gated, got "
            f"{_decoder_tail_mode!r}"
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
                # D5-B dense adaptation: unfreeze ONLY the mask-input
                # convolution stack (4,684 params).  Default 0 keeps the
                # historical fully-frozen PE; point/box embeddings stay
                # frozen either way (validated by the ablation contract).
                train_mask_downscaling=_pe_train_mask_downscaling_raw == "1",
            ),
            checkpoint_load_cfg=dict(strict_reproduction=True),
            freeze_mask_decoder=False,
            freeze_no_mask_embed=True,
            load_no_mask_pretrained=True,
            final_mask_coordinate_mode=(
                "roi_local"
                if _roi_sam_enabled
                else os.environ.get(
                    "FINAL_MASK_COORDINATE_MODE", "roi_local"
                ).strip().lower()
            ),
            **_final_mask_loss_override,
            **_roi_sam_override,
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
                transform=_dense_transform,
                detach_input=_dense_detach,
                temperature=float(
                    os.environ.get("SHAPE_DENSE_TEMPERATURE", "1.0")
                ),
                outside_fill_logit=float(
                    os.environ.get("SHAPE_DENSE_OUTSIDE_FILL", "0.0")
                ),
                clamp_range=(
                    None if _dense_transform == "gaussian_edt" else (-8.0, 8.0)
                ),
                foreground_threshold=_gaussian_foreground_threshold,
                gaussian_omega=_gaussian_omega,
                gaussian_gamma=_gaussian_gamma,
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
                compute_metrics=True,
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
            p2_boundary_refiner_cfg=dict(
                enabled=_p2_boundary_refiner,
                p2_in_channels=512,
                projected_channels=int(os.environ.get(
                    "P2_BOUNDARY_REFINER_PROJECTED_CHANNELS", "64"
                )),
                mid_channels=int(os.environ.get(
                    "P2_BOUNDARY_REFINER_MID_CHANNELS", "64"
                )),
                roi_output_size=32,
                coarse_size=64,
                spatial_scale=0.25,
                sampling_ratio=0,
                aligned=True,
                beta=float(os.environ.get(
                    "P2_BOUNDARY_REFINER_BETA", "0.20"
                )),
                delta_logit_max=float(os.environ.get(
                    "P2_BOUNDARY_REFINER_DELTA_LOGIT_MAX", "2.0"
                )),
                foreground_thr=0.50,
                boundary_band_radius=2,
                search_band_radius=4,
                max_search_coverage=0.50,
                boundary_loss_weight=float(os.environ.get(
                    "P2_BOUNDARY_REFINER_LOSS_WEIGHT", "0.05"
                )),
                zero_init_residual=True,
                # Do not materialize R1-only keys for legacy/boundary runs:
                # config_snapshot feeds the architecture fingerprint.
                **(
                    dict(
                        loss_mode="correction_keep",
                        correction_margin=float(os.environ.get(
                            "P2_BOUNDARY_REFINER_CORRECTION_MARGIN", "1.0"
                        )),
                        keep_loss_weight=float(os.environ.get(
                            "P2_BOUNDARY_REFINER_KEEP_LOSS_WEIGHT", "0.1"
                        )),
                    )
                    if _p2_boundary_refiner and _p2_loss_mode == "correction_keep"
                    else {}
                ),
            ),
            # Do not materialize an inactive key: historical A0/P2 config
            # fingerprints must remain reconstructible byte-for-byte.
            **({"decoder_tail_refiner_cfg": _decoder_tail_cfg}
               if _decoder_tail_enabled else {}),
            **_canvas_renderer_override,
            prune_unused_prompt_components=False,
        )
    )
)

# Training-time RCNN proposal sampler size (positive-RoI cap = num *
# pos_fraction 0.25 = 64 at the historical default 256).  RCNN_SAMPLER_NUM is
# a train-only screening knob (num=128 -> <=32 positives per image) for
# mask-head cost reduction studies; RPN sampling and inference are untouched.
# The default reproduces the historical merged config value exactly.  NOTE:
# any non-default value changes model.train_cfg and therefore the
# architecture model_fingerprint — such checkpoints need allow_cross_arch to
# load against a canonical-env contract (2026-09-08 screening: num=128 lost
# ~-0.02..-0.03 segm/mAP vs 256 over a 30ep compressed schedule; keep 256).
_rcnn_sampler_num = int(os.environ.get("RCNN_SAMPLER_NUM", "256"))
if _rcnn_sampler_num <= 0:
    raise ValueError(f"RCNN_SAMPLER_NUM must be > 0, got {_rcnn_sampler_num}")
model.update(train_cfg=dict(rcnn=dict(sampler=dict(num=_rcnn_sampler_num))))

train_cfg = dict(max_epochs=_epochs, val_interval=1)
