#!/usr/bin/env bash
set -Eeuo pipefail
export ABLATION_ID="r1_c3_pafpn_coarse_points_box_emb64"
export RUN_TAG="${RUN_TAG:-r1_c3_pafpn_coarse_points_box_emb64}"
export NECK_TYPE="pafpn"
export PROMPT_ROUTE="coarse"
export EXPLICIT_PROMPT_MODE="points_box"
export P2_BOUNDARY_REFINER_ENABLED=0
export SAM_IMAGE_EMBED_STRIDE=16
exec bash "$(cd "$(dirname "$0")" && pwd)/_run_ablation.sh" "$@"
