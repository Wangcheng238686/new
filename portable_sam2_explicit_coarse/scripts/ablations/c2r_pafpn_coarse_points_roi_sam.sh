#!/usr/bin/env bash
set -Eeuo pipefail
export ABLATION_ID="c2r_pafpn_coarse_points_roi_sam"
export RUN_TAG="${RUN_TAG:-c2r_pafpn_coarse_points_roi_sam_${SHAPE_CONTEXT_FUSION:-roi_only}}"
export NECK_TYPE="pafpn"
export PROMPT_ROUTE="coarse"
export EXPLICIT_PROMPT_MODE="points"
export P2_BOUNDARY_REFINER_ENABLED=0
export ROI_SAM_ENABLED=1
export ROI_SAM_SAMPLING_RATIO="${ROI_SAM_SAMPLING_RATIO:-2}"
export FINAL_MASK_LOSS_MODE="standard"
exec bash "$(cd "$(dirname "$0")" && pwd)/_run_ablation.sh" "$@"
