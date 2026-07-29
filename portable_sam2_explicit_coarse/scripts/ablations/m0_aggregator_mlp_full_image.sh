#!/usr/bin/env bash
set -Eeuo pipefail
export ABLATION_ID="m0_aggregator_mlp_full_image"
export RUN_TAG="${RUN_TAG:-m0_aggregator_mlp_full_image}"
export NECK_TYPE="aggregator"
export PROMPT_ROUTE="mlp"
export P2_BOUNDARY_REFINER_ENABLED=0
export FINAL_MASK_COORDINATE_MODE="full_image"
exec bash "$(cd "$(dirname "$0")" && pwd)/_run_ablation.sh" "$@"
