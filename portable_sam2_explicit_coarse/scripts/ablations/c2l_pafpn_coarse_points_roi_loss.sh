#!/usr/bin/env bash
set -Eeuo pipefail
export ABLATION_ID="c2l_pafpn_coarse_points_roi_loss"
export RUN_TAG="${RUN_TAG:-c2l_pafpn_coarse_points_roi_loss_${SHAPE_CONTEXT_FUSION:-roi_only}}"
export NECK_TYPE="pafpn"
export PROMPT_ROUTE="coarse"
export EXPLICIT_PROMPT_MODE="points"
export P2_BOUNDARY_REFINER_ENABLED=0
export FINAL_MASK_LOSS_MODE="roi_balanced_dice"
export FINAL_MASK_ROI_EXPAND_RATIO="${FINAL_MASK_ROI_EXPAND_RATIO:-1.20}"
export FINAL_MASK_ROI_BCE_WEIGHT="${FINAL_MASK_ROI_BCE_WEIGHT:-1.0}"
export FINAL_MASK_ROI_DICE_WEIGHT="${FINAL_MASK_ROI_DICE_WEIGHT:-1.0}"
export FINAL_MASK_OUTSIDE_BCE_WEIGHT="${FINAL_MASK_OUTSIDE_BCE_WEIGHT:-0.05}"
exec bash "$(cd "$(dirname "$0")" && pwd)/_run_ablation.sh" "$@"
