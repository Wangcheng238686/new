#!/usr/bin/env bash
set -Eeuo pipefail
export ABLATION_ID="c2_pafpn_coarse_points"
export NECK_TYPE="pafpn"
export PROMPT_ROUTE="coarse"
export EXPLICIT_PROMPT_MODE="points"
export DENSEBR_ENABLED=0
exec bash "$(cd "$(dirname "$0")" && pwd)/_run_ablation.sh" "$@"
