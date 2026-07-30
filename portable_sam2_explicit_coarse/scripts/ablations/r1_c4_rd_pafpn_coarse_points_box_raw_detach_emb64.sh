#!/usr/bin/env bash
set -Eeuo pipefail
export ABLATION_ID="r1_c4_rd_pafpn_coarse_points_box_raw_detach_emb64"
export RUN_TAG="${RUN_TAG:-r1_c4_rd_pafpn_coarse_points_box_raw_detach_emb64}"
export NECK_TYPE="pafpn"
export PROMPT_ROUTE="coarse"
export EXPLICIT_PROMPT_MODE="points_box_dense"
export P2_BOUNDARY_REFINER_ENABLED=0
export SAM_IMAGE_EMBED_STRIDE=16
export SHAPE_DENSE_TRANSFORM="raw_logits"
export SHAPE_DENSE_DETACH=1
exec bash "$(cd "$(dirname "$0")" && pwd)/_run_ablation.sh" "$@"
