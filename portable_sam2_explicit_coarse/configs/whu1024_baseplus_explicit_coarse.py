"""Explicit coarse-mask prompt route on the clean WHU-1024 Base+ baseline."""

import os

_base_ = ["./whu1024_baseplus_clean.py"]

_mode = os.environ.get("EXPLICIT_PROMPT_MODE", "points_box_dense").strip().lower()
if _mode not in {"points", "points_box", "points_box_dense"}:
    raise ValueError(f"Unsupported EXPLICIT_PROMPT_MODE={_mode!r}")

_use_dense = _mode == "points_box_dense"
_densebr = os.environ.get("DENSEBR_ENABLED", "0") == "1"
_epochs = int(os.environ.get("MAX_EPOCHS", "80"))

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
            shape_base_dense_mode="no_mask",
            shape_prior_loss_weight=float(
                os.environ.get("SHAPE_PRIOR_LOSS_WEIGHT", "0.10")
            ),
            shape_prior_cfg=dict(
                enabled=True,
                img_size=1024,
                visual_context_dim=256,
                roi_feat_channels=512,
                coarse_mask_output_size=int(
                    os.environ.get("COARSE_MASK_OUTPUT_SIZE", "64")
                ),
                fusion_type="gated_spatial_film",
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
                adaptive_validity=True,
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
                no_point_epochs=int(os.environ.get("POINT_NO_POINT_EPOCHS", "0")),
                one_pair_epochs=int(os.environ.get("POINT_ONE_PAIR_EPOCHS", "0")),
                full_2p2n_start_epoch=int(
                    os.environ.get("POINT_FULL_START_EPOCH", "1")
                ),
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
            coarse_mask_loss_cfg=dict(
                weight=0.10,
                weight_schedule=[
                    dict(start_epoch=1, end_epoch=min(5, _epochs), weight=0.20),
                    dict(start_epoch=6, end_epoch=_epochs, weight=0.10),
                ] if _epochs > 5 else [
                    dict(start_epoch=1, end_epoch=_epochs, weight=0.20)
                ],
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
                roi_channels=64,
                cue_channels=32,
                mid_channels=64,
                beta_init=0.05,
                beta_max=0.20,
                delta_logit_max=2.0,
                quality_gate_init=0.50,
                image_size=1024,
                foreground_thr=0.50,
                detach_prompt_cues=True,
                zero_init_residual=True,
            ),
            prune_unused_prompt_components=False,
        )
    )
)

train_cfg = dict(max_epochs=_epochs, val_interval=1)

