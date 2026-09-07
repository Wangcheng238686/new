#!/usr/bin/env bash
# NWPU P2-v2 100-epoch development protocol.  This is deliberately one
# parameterized thin entry, so A0/A1/A2 cannot drift through copied runners.
# It never starts in the background: one arm consumes all four GPUs.
#
# Augmentation ruling (round 1, 2026-09-07, audit P2-1): rot90 is NOT
# enabled — the first A0/A1/A2 round keeps the legacy VHR-10 augmentation
# (hflip/vflip 0.5 + multi-scale jitter) uniformly across all three arms.
# Revisit only with the full NWPU matrix protocol (nwpu_fi_matrix_plan D8).
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM="${1:?usage: vhr10_p2v2_dev.sh <a0|a1|a2>}"
shift || true

# A0/A1/A2 are fresh, isolated comparisons, not continuations.  Prevent a
# caller's shell state from changing initialization or output ownership.
unset RESUME_FROM INIT_FROM CHECKPOINT_DIR RUN_TAG

# This protocol is explicitly four-card.  Do not inherit the machine's
# two-card WHU default, otherwise torchrun ranks 2/3 have no visible device.
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4
export MAX_EPOCHS=100
export RUN_IN_BACKGROUND=0
export SAVE_LAST_MODEL=1
export EXPLICIT_PROMPT_MODE=points_box_dense
export FINAL_MASK_COORDINATE_MODE=full_image
export FINAL_MASK_LOSS_MODE=roi_balanced_dice
export SHAPE_DENSE_TRANSFORM=raw_logits
export SHAPE_DENSE_DETACH=0
export SHAPE_DENSE_ALPHA_INIT=0.5
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

case "${ARM}" in
  a0)
    export P2_BOUNDARY_REFINER_ENABLED=0
    export P2_BOUNDARY_REFINER_LOSS_MODE=boundary
    export RUN_TAG="${RUN_TAG:-vhr10_p2v2_dev100_a0_pbm_d5b}"
    ;;
  a1)
    export P2_BOUNDARY_REFINER_ENABLED=1
    export P2_BOUNDARY_REFINER_LOSS_MODE=boundary
    export RUN_TAG="${RUN_TAG:-vhr10_p2v2_dev100_a1_p2v1}"
    ;;
  a2)
    export P2_BOUNDARY_REFINER_ENABLED=1
    export P2_BOUNDARY_REFINER_LOSS_MODE=correction_keep
    export P2_BOUNDARY_REFINER_CORRECTION_MARGIN=1.0
    export P2_BOUNDARY_REFINER_KEEP_LOSS_WEIGHT=0.1
    export RUN_TAG="${RUN_TAG:-vhr10_p2v2_dev100_a2_r1}"
    ;;
  *)
    echo "Unknown arm ${ARM}; expected a0, a1, or a2" >&2
    exit 2
    ;;
esac

source "${SCRIPT_DIR}/vhr10_fi_overlay.sh"
exec bash "${SCRIPT_DIR}/../_run_vhr10.sh" fast400 "$@"
