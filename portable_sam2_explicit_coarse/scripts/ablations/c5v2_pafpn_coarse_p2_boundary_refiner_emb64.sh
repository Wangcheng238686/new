#!/usr/bin/env bash
set -Eeuo pipefail
export ABLATION_ID="c5v2_pafpn_coarse_p2_boundary_refiner_emb64"
export RUN_TAG="${RUN_TAG:-c5v2_pafpn_coarse_p2_boundary_refiner_emb64}"
export NECK_TYPE="pafpn"
export PROMPT_ROUTE="coarse"
export EXPLICIT_PROMPT_MODE="points_box_dense"
export P2_BOUNDARY_REFINER_ENABLED=1
export SAM_IMAGE_EMBED_STRIDE=16
exec bash "$(cd "$(dirname "$0")" && pwd)/_run_ablation.sh" "$@"
