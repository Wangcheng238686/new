#!/usr/bin/env python
import argparse
import json
import math
import os
import sys
from pathlib import Path


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

    if args.prompt_route == "mlp":
        require(head.prompt_sparse_mode == "point", head.prompt_sparse_mode)
        require(not bool(head.prompt_encoder_enabled), "MLP route must disable PromptEncoder")
        require(not bool(head.shape_prior_cfg.enabled), "MLP route must disable coarse head")
        require(not bool(head.densebr_cfg.enabled), "MLP route must disable DenseBR")
    else:
        require(head.prompt_sparse_mode == "shape_point", head.prompt_sparse_mode)
        require(bool(head.prompt_encoder_enabled), "coarse route must enable PromptEncoder")
        require(bool(head.shape_prior_cfg.enabled), "coarse route must enable shape prior")
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
            bool(head.densebr_cfg.enabled) == bool(args.densebr_enabled),
            "DenseBR config mismatch",
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
        "dense_prompt_enabled": bool(head.shape_prior_cfg.get("use_shape_dense", False)),
        "densebr_enabled": bool(head.densebr_cfg.enabled),
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
        require(type(model.neck).__name__ == expected_neck_type, type(model.neck).__name__)
        if args.prompt_route == "mlp":
            require(mask_head.point_emb is not None, "MLP point_emb was not constructed")
            require(mask_head.prompt_encoder is None, "MLP unexpectedly has PromptEncoder")
            require(mask_head.shape_injector is None, "MLP unexpectedly has coarse head")
        else:
            require(mask_head.point_emb is None, "coarse route constructed legacy MLP")
            require(mask_head.prompt_encoder is not None, "coarse route missing PromptEncoder")
            require(mask_head.shape_injector is not None, "coarse route missing shape head")
            require(
                (mask_head.densebr is not None) == bool(args.densebr_enabled),
                "DenseBR model/config mismatch",
            )
        report["model_build"] = "ok"

    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    print("ablation contract: OK")


if __name__ == "__main__":
    main()
