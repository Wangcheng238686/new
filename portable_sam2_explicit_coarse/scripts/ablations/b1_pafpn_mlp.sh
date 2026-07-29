#!/usr/bin/env bash
set -Eeuo pipefail
export ABLATION_ID="b1_pafpn_mlp"
export NECK_TYPE="pafpn"
export PROMPT_ROUTE="mlp"
export P2_BOUNDARY_REFINER_ENABLED=0
export FINAL_MASK_COORDINATE_MODE="roi_local"
exec bash "$(cd "$(dirname "$0")" && pwd)/_run_ablation.sh" "$@"
