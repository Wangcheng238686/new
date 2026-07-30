#!/usr/bin/env bash
set -Eeuo pipefail
export ABLATION_ID="r1_c4_g_pafpn_coarse_points_box_gaussian_emb64"
export RUN_TAG="${RUN_TAG:-r1_c4_g_pafpn_coarse_points_box_gaussian_emb64}"
export NECK_TYPE="pafpn"
export PROMPT_ROUTE="coarse"
export EXPLICIT_PROMPT_MODE="points_box_dense"
export P2_BOUNDARY_REFINER_ENABLED=0
export SAM_IMAGE_EMBED_STRIDE=16
export SHAPE_DENSE_TRANSFORM="gaussian_edt"
export SHAPE_DENSE_DETACH=1
export SHAPE_GAUSSIAN_FOREGROUND_THRESHOLD=0.5
export SHAPE_GAUSSIAN_OMEGA=15.0
export SHAPE_GAUSSIAN_GAMMA=4.0
exec bash "$(cd "$(dirname "$0")" && pwd)/_run_ablation.sh" "$@"
