#!/usr/bin/env bash
# NWPU P2-v2 development protocol.  This is deliberately one
# parameterized thin entry, so A-series arms cannot drift through copied runners.
# It never starts in the background: one arm consumes all four GPUs.
#
# Augmentation ruling (round 1, 2026-09-07, audit P2-1): rot90 is NOT
# enabled — the first A0/A1/A2 round keeps the legacy VHR-10 augmentation
# (hflip/vflip 0.5 + multi-scale jitter) uniformly across all three arms.
# Revisit only with the full NWPU matrix protocol (nwpu_fi_matrix_plan D8).
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM="${1:?usage: vhr10_p2v2_dev.sh <p|pb|a0|a1|a2|a2e|a3|udpr64>}"
shift || true

# Protocol entries share the same architecture/arm block below, so the
# development screen, the historical 600-epoch anchor, and the budgeted
# matrix protocol cannot drift through copied model exports.
P2V2_PROTOCOL="${P2V2_PROTOCOL:-dev100}"
case "${P2V2_PROTOCOL}" in
  dev100)
    P2V2_TAG_PREFIX="vhr10_p2v2_dev100"
    export MAX_EPOCHS=100
    export VAL_EVERY_N_EPOCHS=1
    export EARLY_STOPPING_PATIENCE=9999
    export EARLY_STOPPING_START_EPOCH=9999
    export EARLY_STOPPING_MIN_DELTA=5e-4
    export EARLY_STOPPING_SMOOTH_WINDOW=5
    export SAVE_BBOX_BEST_METRIC=""
    export SAVE_COMPOSITE_BEST=0
    export SAVE_COMPOSITE_WEIGHTS="0.5*bbox/mAP_75+0.5*segm/mAP_75"
    ;;
  full600)
    P2V2_TAG_PREFIX="vhr10_p2v2_full600"
    export MAX_EPOCHS=600
    export VAL_EVERY_N_EPOCHS=5
    # Formal runs must exhaust the entire cosine schedule.  Do not make
    # validation cadence implicitly alter the training horizon.
    export EARLY_STOPPING_PATIENCE=0
    export EARLY_STOPPING_START_EPOCH=9999
    export EARLY_STOPPING_MIN_DELTA=5e-4
    export EARLY_STOPPING_SMOOTH_WINDOW=5
    # Three independent best aliases: segmentation, detection, and the
    # pre-registered equal-weight balance of their headline mAP values.
    export SAVE_BBOX_BEST_METRIC="bbox/mAP"
    export SAVE_COMPOSITE_BEST=1
    export SAVE_COMPOSITE_WEIGHTS="0.5*bbox/mAP+0.5*segm/mAP"
    ;;
  matrix300)
    # Deadline-budgeted NWPU main-matrix protocol.  Every row must start from
    # scratch under this exact schedule; the completed A3/full600 run is a
    # historical long-horizon anchor, never a row to mix into this matrix.
    P2V2_TAG_PREFIX="vhr10_p2v2_matrix300"
    export MAX_EPOCHS=300
    export VAL_EVERY_N_EPOCHS=5
    export EARLY_STOPPING_PATIENCE=0
    export EARLY_STOPPING_START_EPOCH=9999
    export EARLY_STOPPING_MIN_DELTA=5e-4
    export EARLY_STOPPING_SMOOTH_WINDOW=5
    export SAVE_BBOX_BEST_METRIC="bbox/mAP"
    export SAVE_COMPOSITE_BEST=1
    export SAVE_COMPOSITE_WEIGHTS="0.5*bbox/mAP+0.5*segm/mAP"
    ;;
  *)
    echo "Unknown P2V2_PROTOCOL=${P2V2_PROTOCOL}; expected dev100, full600, or matrix300" >&2
    exit 2
    ;;
esac

# A-series arms are fresh, isolated comparisons, not continuations.  Prevent a
# caller's shell state from changing initialization or output ownership.
unset RESUME_FROM INIT_FROM CHECKPOINT_DIR RUN_TAG

# This protocol is explicitly four-card.  Do not inherit the machine's
# two-card WHU default, otherwise torchrun ranks 2/3 have no visible device.
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4
export RUN_IN_BACKGROUND=0
export SAVE_LAST_MODEL=1
export EXPLICIT_PROMPT_MODE=points_box_dense
# Also owned by the runner, but exported here so the arm environment is
# self-contained for DEV_DUMP_ENV audits (stride-16 is part of the arm
# contract; missing it would silently build the stride-32 variant).
export SAM_IMAGE_EMBED_STRIDE=16
export SEGM_SCORE_MODE=detector
export FINAL_MASK_COORDINATE_MODE=full_image
export FINAL_MASK_LOSS_MODE=roi_balanced_dice
export SHAPE_DENSE_TRANSFORM=raw_logits
export SHAPE_DENSE_DETACH=0
export SHAPE_DENSE_ALPHA_INIT=0.5
# Lock the complete D5-B/config surface.  These exports are intentionally
# verbose: a series arm must be immune to an interactive shell left over from
# a diagnostic, not merely work in a clean shell by accident.
export SHAPE_CONTEXT_FUSION=roi_only
export SHAPE_POINT_ADAPTIVE_VALIDITY=1
export SHAPE_LOSS_SCHEDULE_MODE=fixed
export SHAPE_LOSS_STAGE1_END=5
export SHAPE_LOSS_WEIGHT_STAGE1=0.20
export SHAPE_LOSS_WEIGHT_STAGE2=0.10
export SHAPE_GAUSSIAN_FOREGROUND_THRESHOLD=0.5
export SHAPE_GAUSSIAN_OMEGA=15.0
export SHAPE_GAUSSIAN_GAMMA=4.0
export SHAPE_DENSE_TEMPERATURE=1.0
export SHAPE_DENSE_OUTSIDE_FILL=0.0
export COARSE_MASK_OUTPUT_SIZE=64
export ROI_SAM_ENABLED=0
export ROI_SAM_SAMPLING_RATIO=2
export FINAL_MASK_ROI_EXPAND_RATIO=1.20
export FINAL_MASK_ROI_BCE_WEIGHT=1.0
export FINAL_MASK_ROI_DICE_WEIGHT=1.0
export FINAL_MASK_OUTSIDE_BCE_WEIGHT=0.05
export POINT_WARMUP_ENABLED=0
export POINT_WARMUP_NO_POINT_EPOCHS=0
export POINT_WARMUP_ONE_PAIR_EPOCHS=0
export POINT_WARMUP_FULL_START_EPOCH=1
unset POINT_NO_POINT_EPOCHS POINT_ONE_PAIR_EPOCHS POINT_FULL_START_EPOCH
export PROMPT_ENCODER_TRAIN_MASK_DOWNSCALING=1
export PROMPT_ENCODER_LR_MULT=0.1
export P2_BOUNDARY_REFINER_BETA=0.10
export P2_BOUNDARY_REFINER_DELTA_LOGIT_MAX=0.50
export P2_BOUNDARY_REFINER_LOSS_WEIGHT=0.05
export P2_BOUNDARY_REFINER_PROJECTED_CHANNELS=64
export P2_BOUNDARY_REFINER_MID_CHANNELS=64
export SHAPE_PRIOR_LOSS_WEIGHT=0.10
# A caller's old R1 environment must not silently leak into A0/A1.
unset P2_BOUNDARY_REFINER_CORRECTION_MARGIN P2_BOUNDARY_REFINER_KEEP_LOSS_WEIGHT
export DECODER_TAIL_REFINER_ENABLED=0
export DECODER_TAIL_LR_MULT=1.0
# Enabling the module and restricting optimization to it are independent
# choices.  A3 is a normal single-stage A0+UDPR arm; udpr64 remains the
# frozen-A0 mechanism-screening heat start.
unset DECODER_TAIL_TRAIN_ONLY

case "${ARM}" in
  p)
    # Matrix bottom anchor: point-only (2P2N mined points).  PE adaptation
    # is structurally OFF: points mode feeds no mask input, so unfrozen
    # mask_downscaling parameters would be DDP-unused.
    export EXPLICIT_PROMPT_MODE=points
    export P2_BOUNDARY_REFINER_ENABLED=0
    export P2_BOUNDARY_REFINER_LOSS_MODE=boundary
    export PROMPT_ENCODER_TRAIN_MASK_DOWNSCALING=0
    export PROMPT_ENCODER_LR_MULT=0.0
    export RUN_TAG="${RUN_TAG:-${P2V2_TAG_PREFIX}_p_point}"
    ;;
  pb)
    # Point+Box baseline: isolates the box-token gain (pb - p) and the
    # dense package gain (a0 - pb).  md=0 for the same DDP reason as p.
    export EXPLICIT_PROMPT_MODE=points_box
    export P2_BOUNDARY_REFINER_ENABLED=0
    export P2_BOUNDARY_REFINER_LOSS_MODE=boundary
    export PROMPT_ENCODER_TRAIN_MASK_DOWNSCALING=0
    export PROMPT_ENCODER_LR_MULT=0.0
    export RUN_TAG="${RUN_TAG:-${P2V2_TAG_PREFIX}_pb}"
    ;;
  a0)
    export P2_BOUNDARY_REFINER_ENABLED=0
    export P2_BOUNDARY_REFINER_LOSS_MODE=boundary
    export RUN_TAG="${RUN_TAG:-${P2V2_TAG_PREFIX}_a0_pbm_d5b}"
    ;;
  a1)
    export P2_BOUNDARY_REFINER_ENABLED=1
    export P2_BOUNDARY_REFINER_LOSS_MODE=boundary
    export RUN_TAG="${RUN_TAG:-${P2V2_TAG_PREFIX}_a1_p2v1}"
    ;;
  a2)
    export P2_BOUNDARY_REFINER_ENABLED=1
    export P2_BOUNDARY_REFINER_LOSS_MODE=correction_keep
    export P2_BOUNDARY_REFINER_CORRECTION_MARGIN=1.0
    export P2_BOUNDARY_REFINER_KEEP_LOSS_WEIGHT=0.1
    export RUN_TAG="${RUN_TAG:-${P2V2_TAG_PREFIX}_a2_r1}"
    ;;
  a2e)
    # Envelope-escalated R1 (2026-09-07, user-directed): live telemetry on
    # the default a2 showed active-ROI mean |delta| pinned at the 0.0495
    # saturation cap (beta 0.10 x delta_max 0.50) with delta_saturated_sum
    # climbing — before judging R1, test whether the budget binds.  Only the
    # two envelope knobs change vs a2: beta 0.10->0.30, delta_logit_max
    # 0.50->1.00 (cap +-0.05 -> +-0.30, sat_thr 0.297).  Supervision stays
    # correction_keep (margin 1.0, lambda 0.1, w_aux 0.05).  The default a2
    # runs on the other machine; do NOT relaunch it here.
    export P2_BOUNDARY_REFINER_ENABLED=1
    export P2_BOUNDARY_REFINER_LOSS_MODE=correction_keep
    export P2_BOUNDARY_REFINER_CORRECTION_MARGIN=1.0
    export P2_BOUNDARY_REFINER_KEEP_LOSS_WEIGHT=0.1
    export P2_BOUNDARY_REFINER_BETA=0.30
    export P2_BOUNDARY_REFINER_DELTA_LOGIT_MAX=1.00
    export RUN_TAG="${RUN_TAG:-${P2V2_TAG_PREFIX}_a2e_r1_esc030}"
    ;;
  a3)
    # A-series UDPR arm: relative to PBM/A0, add only the decoder-tail
    # module.  This is deliberately NOT A0 heat-start / tail-only training:
    # all A0 parameters and the zero-initialized tail share the same one-stage
    # 100ep schedule, so its result is comparable to A0/A1/A2 under one
    # protocol and usable as the formal-training feasibility gate.
    export P2_BOUNDARY_REFINER_ENABLED=0
    export P2_BOUNDARY_REFINER_LOSS_MODE=boundary
    export DECODER_TAIL_REFINER_ENABLED=1
    export DECODER_TAIL_NUM_POINTS=64
    export DECODER_TAIL_HIDDEN_DIM=128
    export DECODER_TAIL_POINT_LOSS_WEIGHT=1.0
    export DECODER_TAIL_DELTA_LOGIT_MAX=2.0
    export RUN_TAG="${RUN_TAG:-${P2V2_TAG_PREFIX}_a3_pbm_udprk64}"
    ;;
  udpr64)
    # Independent decoder-tail arm.  It starts from the frozen A0 winner;
    # only the newly introduced UDPR parameters are optimized.
    export P2_BOUNDARY_REFINER_ENABLED=0
    export P2_BOUNDARY_REFINER_LOSS_MODE=boundary
    export DECODER_TAIL_REFINER_ENABLED=1
    export DECODER_TAIL_NUM_POINTS=64
    export DECODER_TAIL_HIDDEN_DIM=128
    export DECODER_TAIL_POINT_LOSS_WEIGHT=1.0
    export DECODER_TAIL_DELTA_LOGIT_MAX=2.0
    export DECODER_TAIL_TRAIN_ONLY=1
    export INIT_FROM="${UDPR_INIT_FROM:-/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/vhr10_p2v2_dev100_a0_pbm_d5b_tr1.0_va1.0/best_model_epoch36.pth}"
    export RUN_TAG="${RUN_TAG:-${P2V2_TAG_PREFIX}_udprk64_a0init}"
    ;;
  *)
    echo "Unknown arm ${ARM}; expected p, pb, a0, a1, a2, a2e, a3, or udpr64" >&2
    exit 2
    ;;
esac

source "${SCRIPT_DIR}/vhr10_fi_overlay.sh"

# Debug hook for audits: dump the fully-resolved arm environment (common
# block + arm case + overlay) instead of launching.  Consumed by
# scripts/smoke/verify_p2v2_arms.py so model-construction checks replay the
# REAL arm env rather than a hand-copied one.
if [ "${DEV_DUMP_ENV:-0}" = "1" ]; then
  env | grep -E '^(NECK_TYPE|PROMPT_ROUTE|EXPLICIT_PROMPT_MODE|P2_BOUNDARY_REFINER_[A-Z_]+|DECODER_TAIL_[A-Z_]+|INIT_FROM|SAM_IMAGE_EMBED_STRIDE|SEGM_SCORE_MODE|ROI_SAM_[A-Z_]+|COARSE_MASK_OUTPUT_SIZE|POINT_(WARMUP_[A-Z_]+|NO_POINT_EPOCHS|ONE_PAIR_EPOCHS|FULL_START_EPOCH)|SHAPE_[A-Z_]+|PROMPT_ENCODER_[A-Z_]+|FINAL_MASK_[A-Z_]+|MAX_EPOCHS|BATCH_SIZE|GRAD_ACCUM_STEPS|NPROC_PER_NODE|VAL_EVERY_N_EPOCHS|EARLY_STOPPING_[A-Z_]+|SAVE_LAST_MODEL|RUN_TAG|CUDA_VISIBLE_DEVICES)=' | sort
  exit 0
fi

exec bash "${SCRIPT_DIR}/../_run_vhr10.sh" fast400 "$@"
