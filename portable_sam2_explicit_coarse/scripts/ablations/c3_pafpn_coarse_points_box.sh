#!/usr/bin/env bash
set -Eeuo pipefail
export ABLATION_ID="c3_pafpn_coarse_points_box"
export RUN_TAG="${RUN_TAG:-c3_pafpn_coarse_points_box_semanticfix_${SHAPE_CONTEXT_FUSION:-roi_only}}"
export NECK_TYPE="pafpn"
export PROMPT_ROUTE="coarse"
export EXPLICIT_PROMPT_MODE="points_box"
export P2_BOUNDARY_REFINER_ENABLED=0
exec bash "$(cd "$(dirname "$0")" && pwd)/_run_ablation.sh" "$@"
