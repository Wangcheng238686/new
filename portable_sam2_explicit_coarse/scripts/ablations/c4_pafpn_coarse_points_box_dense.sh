#!/usr/bin/env bash
set -Eeuo pipefail
export ABLATION_ID="c4_pafpn_coarse_points_box_dense"
export NECK_TYPE="pafpn"
export PROMPT_ROUTE="coarse"
export EXPLICIT_PROMPT_MODE="points_box_dense"
export DENSEBR_ENABLED=0
exec bash "$(cd "$(dirname "$0")" && pwd)/_run_ablation.sh" "$@"
