#!/usr/bin/env bash
set -Eeuo pipefail

export ABLATION_ID="c5v2_pafpn_coarse_p2_boundary_refiner_emb64"
export RUN_TAG="${RUN_TAG:-whu_p2_matrix_full}"
export EXPLICIT_PROMPT_MODE="points_box_dense"
export P2_BOUNDARY_REFINER_ENABLED=1
source "$(cd "$(dirname "$0")" && pwd)/whu_p2_matrix_common.sh"
exec bash "$(cd "$(dirname "$0")" && pwd)/_run_ablation.sh" "$@"
