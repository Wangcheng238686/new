#!/usr/bin/env bash
set -Eeuo pipefail

export ABLATION_ID="r1_c3_pafpn_coarse_points_box_emb64"
export RUN_TAG="${RUN_TAG:-whu_p2_matrix_point_box}"
export EXPLICIT_PROMPT_MODE="points_box"
export P2_BOUNDARY_REFINER_ENABLED=0
source "$(cd "$(dirname "$0")" && pwd)/whu_p2_matrix_common.sh"
exec bash "$(cd "$(dirname "$0")" && pwd)/_run_ablation.sh" "$@"
