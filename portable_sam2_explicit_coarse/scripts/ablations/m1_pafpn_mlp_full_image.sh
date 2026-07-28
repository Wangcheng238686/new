#!/usr/bin/env bash
set -Eeuo pipefail
export ABLATION_ID="m1_pafpn_mlp_full_image"
export RUN_TAG="${RUN_TAG:-m1_pafpn_mlp_full_image}"
export NECK_TYPE="pafpn"
export PROMPT_ROUTE="mlp"
export DENSEBR_ENABLED=0
export FINAL_MASK_COORDINATE_MODE="full_image"
exec bash "$(cd "$(dirname "$0")" && pwd)/_run_ablation.sh" "$@"
