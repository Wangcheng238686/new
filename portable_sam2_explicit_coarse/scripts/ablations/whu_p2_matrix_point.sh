#!/usr/bin/env bash
set -Eeuo pipefail

export ABLATION_ID="${ABLATION_ID:-pafpn_coarse_points_emb64_p2br0}"
export RUN_TAG="${RUN_TAG:-whu_p2_matrix_point}"
export EXPLICIT_PROMPT_MODE="points"
export P2_BOUNDARY_REFINER_ENABLED=0
source "$(cd "$(dirname "$0")" && pwd)/whu_p2_matrix_common.sh"
exec bash "$(dirname "$0")/../_run_ablation.sh" "$@"
