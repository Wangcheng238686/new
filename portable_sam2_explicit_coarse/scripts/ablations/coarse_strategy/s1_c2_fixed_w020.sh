#!/usr/bin/env bash
set -Eeuo pipefail

export ABLATION_ID="s1_c2_adaptive_fixed_w020"
export RUN_TAG="${RUN_TAG:-s1_c2_adaptive_fixed_w020_semanticfix_${SHAPE_CONTEXT_FUSION:-roi_only}}"
export NECK_TYPE="pafpn"
export PROMPT_ROUTE="coarse"
export EXPLICIT_PROMPT_MODE="points"
export P2_BOUNDARY_REFINER_ENABLED=0
export EXPECTED_ARCHITECTURE_ID="c2_pafpn_coarse_points"
export SHAPE_POINT_ADAPTIVE_VALIDITY=1
export SHAPE_LOSS_SCHEDULE_MODE="fixed"
export SHAPE_PRIOR_LOSS_WEIGHT="0.20"

exec bash "$(cd "$(dirname "$0")/.." && pwd)/_run_ablation.sh" "$@"
