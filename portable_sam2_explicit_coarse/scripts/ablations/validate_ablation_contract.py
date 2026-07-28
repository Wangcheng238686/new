#!/usr/bin/env python
import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ablation-id", required=True)
    parser.add_argument("--expected-neck", choices=("aggregator", "pafpn"), required=True)
    parser.add_argument("--prompt-route", choices=("mlp", "coarse"), required=True)
    parser.add_argument("--explicit-prompt-mode", required=True)
    parser.add_argument("--densebr-enabled", type=int, choices=(0, 1), required=True)
    parser.add_argument(
        "--expected-final-mask-mode",
        choices=("roi_local", "full_image"),
        required=True,
    )
    parser.add_argument(
        "--expected-image-embed-stride",
        type=int,
        choices=(16, 32),
        required=True,
    )
    parser.add_argument(
        "--expected-segm-score-mode",
        choices=("detector", "mask_quality"),
        required=True,
    )
    parser.add_argument("--train-subset-ratio", type=float, default=1.0)
    parser.add_argument("--val-subset-ratio", type=float, default=1.0)
    parser.add_argument("--subset-seed", type=int, default=44)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--build-model", action="store_true")
    return parser.parse_args()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def expected_subset_size(full_size, ratio):
    return full_size if ratio >= 1.0 else max(1, int(full_size * ratio))


def main():
    args = parse_args()
    for name, value in (
        ("train_subset_ratio", args.train_subset_ratio),
        ("val_subset_ratio", args.val_subset_ratio),
    ):
        require(0.0 < value <= 1.0, f"{name} must be in (0,1], got {value}")

    from mmengine import Config

    cfg = Config.fromfile(str(PROJECT_ROOT / args.config))
    neck = cfg.model.neck
    head = cfg.model.roi_head.mask_head
    expected_neck_type = (
        "RSSAM2PAFPN" if args.expected_neck == "pafpn"
        else "RSFeatureAggregatorSAM2"
    )
    require(neck.type == expected_neck_type, (neck.type, expected_neck_type))
    require(neck.in_channels == "sam2_hiera_base_plus", neck.in_channels)
    expected_embed_size = 1024 // args.expected_image_embed_stride
    require(
        int(cfg.model.sam_image_embedding_stride)
        == args.expected_image_embed_stride,
        "top-level SAM image embedding stride mismatch",
    )
    require(
        int(cfg.model.shared_image_embedding.embedding_stride)
        == args.expected_image_embed_stride,
        "positional embedding stride mismatch",
    )
    require(
        int(head.prompt_encoder_embed_size) == expected_embed_size,
        "PromptEncoder/image embedding spatial size mismatch",
    )
    require(
        head.final_mask_coordinate_mode == args.expected_final_mask_mode,
        "final-mask coordinate mode mismatch",
    )
    require(
        head.segm_score_mode == args.expected_segm_score_mode,
        "segmentation score mode mismatch",
    )
    quality_head_enabled = bool(
        head.get("quality_head_cfg", {}).get("enabled", False)
    )
    if head.segm_score_mode == "mask_quality":
        require(
            quality_head_enabled,
            "mask-quality segmentation ranking requires an enabled quality head",
        )
    require(
        ("mask_scores" if head.segm_score_mode == "mask_quality" else "scores")
        == ("mask_scores" if args.expected_segm_score_mode == "mask_quality" else "scores"),
        "segmentation score key mismatch",
    )

    if args.prompt_route == "mlp":
        require(head.prompt_sparse_mode == "point", head.prompt_sparse_mode)
        require(not bool(head.prompt_encoder_enabled), "MLP route must disable PromptEncoder")
        require(not bool(head.shape_prior_cfg.enabled), "MLP route must disable coarse head")
        require(not bool(head.densebr_cfg.enabled), "MLP route must disable DenseBR")
        require(
            not bool(head.load_no_mask_pretrained),
            "MLP baseline must use the legacy zero-initialized no_mask_embed",
        )
        require(
            not bool(head.freeze_no_mask_embed),
            "MLP baseline no_mask_embed must remain trainable",
        )
    else:
        require(head.prompt_sparse_mode == "shape_point", head.prompt_sparse_mode)
        require(bool(head.prompt_encoder_enabled), "coarse route must enable PromptEncoder")
        require(bool(head.shape_prior_cfg.enabled), "coarse route must enable shape prior")
        expected_context_fusion = os.environ.get(
            "SHAPE_CONTEXT_FUSION", "roi_only"
        ).strip().lower()
        require(
            head.shape_prior_cfg.fusion_type == expected_context_fusion,
            "shape context/FiLM fusion mismatch",
        )
        require(
            head.explicit_prompt_mode == args.explicit_prompt_mode,
            (head.explicit_prompt_mode, args.explicit_prompt_mode),
        )
        expect_dense = args.explicit_prompt_mode == "points_box_dense"
        require(
            bool(head.shape_prior_cfg.use_shape_dense) == expect_dense,
            "dense prompt config mismatch",
        )
        require(
            bool(head.restrict_dense_prompt_to_box),
            "dense prompt embedding delta must be restricted to the proposal box",
        )
        require(
            bool(head.densebr_cfg.enabled) == bool(args.densebr_enabled),
            "DenseBR config mismatch",
        )
        densebr_expected = {
            "beta_init": float(os.environ.get("DENSEBR_BETA_INIT", "0.05")),
            "beta_max": float(os.environ.get("DENSEBR_BETA_MAX", "0.20")),
            "delta_logit_max": float(
                os.environ.get("DENSEBR_DELTA_LOGIT_MAX", "2.0")
            ),
            "quality_gate_init": float(
                os.environ.get("DENSEBR_QUALITY_GATE_INIT", "0.50")
            ),
            "roi_channels": int(os.environ.get("DENSEBR_ROI_CHANNELS", "64")),
            "cue_channels": int(os.environ.get("DENSEBR_CUE_CHANNELS", "32")),
            "mid_channels": int(os.environ.get("DENSEBR_MID_CHANNELS", "64")),
            "detach_prompt_cues": os.environ.get(
                "DENSEBR_DETACH_PROMPT_CUES", "1"
            ) == "1",
        }
        for key, expected in densebr_expected.items():
            actual = head.densebr_cfg[key]
            if isinstance(expected, float):
                require(math.isclose(float(actual), expected), (key, actual, expected))
            else:
                require(actual == expected, (key, actual, expected))
        require(
            bool(head.load_no_mask_pretrained),
            "coarse route must retain the pretrained SAM2 no-mask embedding",
        )
        require(
            bool(head.freeze_no_mask_embed),
            "coarse route must freeze the pretrained no-mask embedding",
        )
        expected_adaptive = os.environ.get(
            "SHAPE_POINT_ADAPTIVE_VALIDITY", "1"
        ) == "1"
        require(
            bool(head.shape_point_miner_cfg.adaptive_validity)
            == expected_adaptive,
            "shape-point adaptive-validity mismatch",
        )
        expected_shape_weight = float(
            os.environ.get("SHAPE_PRIOR_LOSS_WEIGHT", "0.10")
        )
        expected_schedule_mode = os.environ.get(
            "SHAPE_LOSS_SCHEDULE_MODE", "fixed"
        )
        require(
            head.coarse_mask_loss_cfg.schedule_mode == expected_schedule_mode,
            "coarse loss schedule mode mismatch",
        )
        require(
            math.isclose(
                float(head.coarse_mask_loss_cfg.weight),
                expected_shape_weight,
            ),
            "coarse loss fixed/fallback weight mismatch",
        )
        schedule = list(head.coarse_mask_loss_cfg.weight_schedule)
        if expected_schedule_mode == "fixed":
            require(not schedule, "fixed coarse loss must have an empty schedule")
        else:
            require(len(schedule) == 2, "two-stage coarse loss requires two ranges")
            require(
                int(schedule[0]["end_epoch"])
                == int(os.environ.get("SHAPE_LOSS_STAGE1_END", "5")),
                "coarse loss stage-1 end mismatch",
            )
            require(
                math.isclose(
                    float(schedule[0]["weight"]),
                    float(os.environ.get("SHAPE_LOSS_WEIGHT_STAGE1", "0.20")),
                ),
                "coarse loss stage-1 weight mismatch",
            )
            require(
                math.isclose(
                    float(schedule[1]["weight"]),
                    float(os.environ.get("SHAPE_LOSS_WEIGHT_STAGE2", "0.10")),
                ),
                "coarse loss stage-2 weight mismatch",
            )

    report = {
        "ablation_id": args.ablation_id,
        "config": args.config,
        "neck": neck.type,
        "prompt_route": args.prompt_route,
        "prompt_sparse_mode": head.prompt_sparse_mode,
        "explicit_prompt_mode": (
            None if args.prompt_route == "mlp" else head.explicit_prompt_mode
        ),
        "prompt_encoder_enabled": bool(head.prompt_encoder_enabled),
        "shape_prior_enabled": bool(head.shape_prior_cfg.enabled),
        "shape_context_fusion": (
            None
            if args.prompt_route == "mlp"
            else head.shape_prior_cfg.fusion_type
        ),
        "dense_prompt_enabled": bool(head.shape_prior_cfg.get("use_shape_dense", False)),
        "densebr_enabled": bool(head.densebr_cfg.enabled),
        "final_mask_coordinate_mode": head.final_mask_coordinate_mode,
        "segm_score_mode": head.segm_score_mode,
        "segm_score_key": (
            "mask_scores" if head.segm_score_mode == "mask_quality" else "scores"
        ),
        "quality_head_enabled": quality_head_enabled,
        "densebr_config": dict(head.densebr_cfg),
        "sam_image_embedding_stride": int(cfg.model.sam_image_embedding_stride),
        "sam_image_embedding_size": expected_embed_size,
        "shape_point_adaptive_validity": (
            None
            if args.prompt_route == "mlp"
            else bool(head.shape_point_miner_cfg.adaptive_validity)
        ),
        "shape_loss_schedule_mode": (
            None
            if args.prompt_route == "mlp"
            else head.coarse_mask_loss_cfg.schedule_mode
        ),
        "shape_prior_loss_weight": (
            None
            if args.prompt_route == "mlp"
            else float(head.coarse_mask_loss_cfg.weight)
        ),
        "train_subset_ratio": args.train_subset_ratio,
        "val_subset_ratio": args.val_subset_ratio,
        "subset_seed": args.subset_seed,
    }

    if args.check_data:
        from data import create_train_loader

        train_loader, val_loader, train_dataset = create_train_loader(
            data_root=args.data_root,
            batch_size=1,
            image_size=(1024, 1024),
            dataset_format="whu_coco",
            whu_train_ann_file="2.4 annotation/annotation/train.json",
            whu_val_ann_file="2.4 annotation/annotation/validation.json",
            whu_train_img_subdir="2.1 train/train",
            whu_val_img_subdir="2.3 valid/validation",
            train_subset_ratio=args.train_subset_ratio,
            val_subset_ratio=args.val_subset_ratio,
            num_workers=0,
            seed=args.subset_seed,
        )
        val_dataset = val_loader.dataset
        require(
            len(train_dataset)
            == expected_subset_size(train_dataset.full_size, args.train_subset_ratio),
            "training subset size mismatch",
        )
        require(
            len(val_dataset)
            == expected_subset_size(val_dataset.full_size, args.val_subset_ratio),
            "validation subset size mismatch",
        )
        report["data"] = {
            "train": [len(train_dataset), train_dataset.full_size],
            "val": [len(val_dataset), val_dataset.full_size],
        }

    if args.build_model:
        from mmengine.registry import init_default_scope

        init_default_scope("mmdet")
        import data  # noqa: F401
        import rsprompter  # noqa: F401
        from mmdet.registry import MODELS

        model = MODELS.build(cfg.model)
        mask_head = model.roi_head.mask_head
        require(
            mask_head.segm_score_mode == args.expected_segm_score_mode,
            "built model segmentation score mode mismatch",
        )
        require(
            mask_head.segm_score_key
            == ("mask_scores" if args.expected_segm_score_mode == "mask_quality" else "scores"),
            "built model segmentation score key mismatch",
        )
        require(
            int(model.sam_image_embedding_stride)
            == args.expected_image_embed_stride,
            "built model SAM image embedding stride mismatch",
        )
        require(type(model.neck).__name__ == expected_neck_type, type(model.neck).__name__)
        if args.prompt_route == "mlp":
            require(mask_head.point_emb is not None, "MLP point_emb was not constructed")
            require(mask_head.prompt_encoder is None, "MLP unexpectedly has PromptEncoder")
            require(mask_head.shape_injector is None, "MLP unexpectedly has coarse head")
            require(
                bool(mask_head.no_mask_embed.requires_grad),
                "MLP no_mask_embed is not trainable",
            )
            require(
                not bool(mask_head.load_no_mask_pretrained),
                "MLP no_mask_embed unexpectedly loaded SAM2 weights",
            )
            raw_pe = model.shared_image_embedding.get_dense_pe()
            require(
                tuple(raw_pe.shape[-2:]) == (64, 64),
                "MLP legacy positional parameter must retain its 64x64 shape",
            )
            require(
                bool(model.shared_image_embedding.pos_embed.requires_grad),
                "MLP legacy positional embedding is not trainable",
            )
            require(
                int(torch.count_nonzero(raw_pe).item()) == 0,
                "MLP legacy positional embedding is not zero-initialized",
            )
            effective_pe = model.get_image_wide_positional_embeddings(
                size=expected_embed_size
            )
            require(
                tuple(effective_pe.shape[-2:])
                == (expected_embed_size, expected_embed_size),
                "MLP effective positional embedding spatial size mismatch",
            )
        else:
            require(mask_head.point_emb is None, "coarse route constructed legacy MLP")
            require(mask_head.prompt_encoder is not None, "coarse route missing PromptEncoder")
            require(mask_head.shape_injector is not None, "coarse route missing shape head")
            expected_context_fusion = os.environ.get(
                "SHAPE_CONTEXT_FUSION", "roi_only"
            ).strip().lower()
            require(
                mask_head.shape_injector.fusion_type == expected_context_fusion,
                "built model shape context/FiLM fusion mismatch",
            )
            if expected_context_fusion == "roi_only":
                require(
                    mask_head.shape_injector.visual_context_proj is None
                    and mask_head.shape_injector.spatial_cross_attn is None
                    and mask_head.shape_injector.gamma_head is None
                    and mask_head.shape_injector.beta_head is None,
                    "roi_only must not construct inactive FiLM/context parameters",
                )
            require(
                bool(mask_head.no_mask_pretrained_loaded),
                "coarse route silently missed pretrained no_mask_embed",
            )
            require(
                mask_head.no_mask_embed is not None
                and not bool(mask_head.no_mask_embed.requires_grad),
                "coarse route pretrained no_mask_embed must exist and be frozen",
            )
            mask_head.assert_no_mask_embedding_contract(
                f"{args.ablation_id} preflight"
            )
            require(
                tuple(mask_head.prompt_encoder.image_embedding_size)
                == (expected_embed_size, expected_embed_size),
                "built PromptEncoder image embedding size mismatch",
            )
            require(
                tuple(mask_head.prompt_encoder.mask_input_size)
                == (expected_embed_size * 4, expected_embed_size * 4),
                "built PromptEncoder dense-mask input size mismatch",
            )
            require(
                (mask_head.densebr is not None) == bool(args.densebr_enabled),
                "DenseBR model/config mismatch",
            )

        # Regression guard shared by validation and checkpoint inference.
        image_size = 64
        bbox = torch.tensor([[16.0, 16.0, 48.0, 48.0]])
        probe_logits = torch.full((1, 1, 8, 8), 20.0)
        if args.expected_final_mask_mode == "full_image":
            probe_logits.fill_(-20.0)
            probe_logits[:, :, :4, :4] = 20.0
        pasted = mask_head._predict_by_feat_single(
            mask_preds=probe_logits,
            bboxes=bbox,
            labels=torch.zeros(1, dtype=torch.long),
            img_meta={
                "scale_factor": (1.0, 1.0),
                "ori_shape": (image_size, image_size),
                "img_shape": (image_size, image_size),
            },
            rcnn_test_cfg=cfg.model.test_cfg.rcnn,
            rescale=False,
            activate_map=False,
        )
        if args.expected_final_mask_mode == "roi_local":
            require(bool(pasted[0, 24:40, 24:40].any()), "pasted mask is empty")
            outside = pasted.clone()
            outside[:, 15:49, 15:49] = False
            require(
                not bool(outside.any()),
                "ROI-local mask leaked outside bbox",
            )
            postprocess = "roi_local_bbox_paste"
        else:
            require(bool(pasted[0, 8, 8]), "full-image top-left signal was lost")
            require(
                not bool(pasted[0, 56, 56]),
                "full-image mask was incorrectly pasted through the bbox",
            )
            postprocess = "full_image_resize"
        report["model_build"] = "ok"
        report["mask_postprocess"] = postprocess
        report["no_mask_pretrained_loaded"] = bool(
            mask_head.no_mask_pretrained_loaded
        )
        report["no_mask_abs_mean"] = float(
            mask_head.no_mask_embed.detach().abs().mean().item()
        )
        report["prompt_dense_mask_input_size"] = (
            None
            if mask_head.prompt_encoder is None
            else list(mask_head.prompt_encoder.mask_input_size)
        )

    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    print("ablation contract: OK")


if __name__ == "__main__":
    main()
