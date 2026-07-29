#!/usr/bin/env bash
set -Eeuo pipefail

export ABLATION_ID="s2_c2_adaptive_two_stage_w020_w010"
export RUN_TAG="${RUN_TAG:-s2_c2_adaptive_two_stage_w020_w010_semanticfix_${SHAPE_CONTEXT_FUSION:-roi_only}}"
export NECK_TYPE="pafpn"
export PROMPT_ROUTE="coarse"
export EXPLICIT_PROMPT_MODE="points"
export P2_BOUNDARY_REFINER_ENABLED=0
export EXPECTED_ARCHITECTURE_ID="c2_pafpn_coarse_points"
export SHAPE_POINT_ADAPTIVE_VALIDITY=1
export SHAPE_LOSS_SCHEDULE_MODE="two_stage"
export SHAPE_PRIOR_LOSS_WEIGHT="0.10"
export SHAPE_LOSS_STAGE1_END="5"
export SHAPE_LOSS_WEIGHT_STAGE1="0.20"
export SHAPE_LOSS_WEIGHT_STAGE2="0.10"

exec bash "$(cd "$(dirname "$0")/.." && pwd)/_run_ablation.sh" "$@"
