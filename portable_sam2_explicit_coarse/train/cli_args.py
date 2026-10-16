"""CLI surface: env-flag helper and the full argparse tree with validation."""

import argparse
import os

def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _parse_args() -> argparse.Namespace:
    """Command line; validation errors fail fast before imports."""
    parser = argparse.ArgumentParser(
        description="portable_sam_fusion: RSPrompter(SAM) + Drone semantic guidance"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--raw-split-manifest", default="surfbev_b0_split_v2.json")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--checkpoint-dir", type=str, default="/tmp/portable_sam_fusion_ckpts"
    )
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--init-from", type=str, default=None)
    parser.add_argument(
        "--allow-cross-arch-init",
        action="store_true",
        help="Explicitly allow INIT_FROM from a different or legacy architecture.",
    )
    parser.add_argument(
        "--init-exclude-prefixes",
        type=str,
        default="",
        help="Comma-separated model state prefixes excluded from --init-from",
    )

    parser.add_argument(
        "--dataset",
        choices=("whu", "vhr10", "ue_raw_surface"),
        default="whu",
        help="Dataset protocol: WHU-1024 (1 class) or NWPU VHR-10 (10 classes).",
    )
    parser.add_argument(
        "--train-subset-ratio",
        type=float,
        default=1.0,
        help="Deterministic fraction of WHU training images to load; default 1.0.",
    )
    parser.add_argument(
        "--val-subset-ratio",
        type=float,
        default=1.0,
        help="Deterministic fraction of WHU validation images to load; default 1.0.",
    )


    parser.add_argument("--freeze-bn", action="store_true")

    parser.add_argument("--image-size", type=int, nargs=2, default=[512, 512])
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=1,
        help="Validation batch size (legacy WHU1024 baseline default: 1)",
    )
    parser.add_argument("--val-every-n-epochs", type=int, default=1)
    parser.add_argument(
        "--compute-val-loss",
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            "Whether validation also runs a second loss forward on positive-GT "
            "batches. COCO prediction/evaluation is unaffected."
        ),
    )
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-start-epoch", type=int, default=20)
    parser.add_argument("--early-stopping-min-delta", type=float, default=5e-4)
    parser.add_argument(
        "--early-stopping-smooth-window",
        type=int,
        default=5,
        help="Moving-average window for the resolved early-stopping metric.",
    )
    parser.add_argument(
        "--max-nonfinite-batches",
        type=int,
        default=0,
        help="Abort after this many synchronized non-finite batches; 0 disables abort.",
    )

    parser.add_argument("--backbone-lr-mult", type=float, default=1.0)
    parser.add_argument("--other-lr-mult", type=float, default=1.0)
    parser.add_argument(
        "--bbox-head-lr-mult",
        type=float,
        default=0.0,
        help="Use >0 to place bbox_head in a separate optimizer group",
    )
    parser.add_argument("--mask-decoder-lr-mult", type=float, default=1.0)
    parser.add_argument("--no-mask-lr-mult", type=float, default=1.0)
    parser.add_argument("--prompt-encoder-lr-mult", type=float, default=0.1)
    parser.add_argument("--shape-prior-lr-mult", type=float, default=1.0)
    parser.add_argument("--decoder-tail-lr-mult", type=float, default=1.0)
    parser.add_argument("--decoupled-aux-clip", action="store_true",
                        help="clip base params and each aux head separately at "
                             "max_norm 0.5 instead of one global clip; keeps the "
                             "base's clip semantics bit-identical to aux-free arms")
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--warmup-epochs", type=int, default=0, help="Number of warmup epochs for learning rate scheduling")
    parser.add_argument("--warmup-iters", type=int, default=100, help="Number of warmup optimizer steps (overrides warmup-epochs if > 0)")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay for optimizer")
    parser.add_argument("--fusion-no-wd", type=int, default=0,
                        help="Exempt the dual fusion modules (gate/proj[_e]) from weight decay (EXP-0015/0016)")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # --- AMP / EMA (训练基础设施) ---
    parser.add_argument("--amp", type=int, default=0, help="Enable AMP mixed precision when set to 1")
    parser.add_argument("--ema-enabled", type=int, default=1, help="Enable EMA shadow weights")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay in (0,1)")
    parser.add_argument("--ema-update-every", type=int, default=1, help="Update EMA every N optimizer steps")
    parser.add_argument("--ema-eval", type=int, default=1, help="Use EMA shadow weights for validation")
    parser.add_argument("--ema-eval-start-epoch", type=int, default=5, help="Start using EMA for eval after this epoch")
    parser.add_argument("--prompt-debug-stats", type=int, choices=[0, 1], default=0,
                        help="Print shape-point/coarse-mask prompt diagnostics")
    parser.add_argument("--prompt-debug-stats-interval", type=int, default=50,
                        help="Training iteration interval for prompt diagnostics")
    args = parser.parse_args()

    if int(args.early_stopping_start_epoch) < 1:
        raise ValueError("early_stopping_start_epoch must be >= 1")
    if int(args.early_stopping_patience) < 0:
        raise ValueError("early_stopping_patience must be >= 0")
    if float(args.early_stopping_min_delta) < 0:
        raise ValueError("early_stopping_min_delta must be >= 0")
    if int(args.early_stopping_smooth_window) < 1:
        raise ValueError("early_stopping_smooth_window must be >= 1")
    if int(args.val_batch_size) < 1:
        raise ValueError("val_batch_size must be >= 1")
    if int(args.val_every_n_epochs) < 1:
        raise ValueError("val_every_n_epochs must be >= 1")

    return args
