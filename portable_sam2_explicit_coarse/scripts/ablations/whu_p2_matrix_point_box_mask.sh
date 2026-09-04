#!/usr/bin/env bash
set -Eeuo pipefail

export ABLATION_ID="r1_c4_pafpn_coarse_points_box_dense_emb64"
export RUN_TAG="${RUN_TAG:-whu_p2_matrix_point_box_mask}"
export EXPLICIT_PROMPT_MODE="points_box_dense"
export P2_BOUNDARY_REFINER_ENABLED=0
source "$(cd "$(dirname "$0")" && pwd)/whu_p2_matrix_common.sh"
exec bash "$(dirname "$0")/../_run_ablation.sh" "$@"
