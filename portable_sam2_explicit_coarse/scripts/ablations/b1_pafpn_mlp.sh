#!/usr/bin/env bash
set -Eeuo pipefail
export ABLATION_ID="b1_pafpn_mlp"
export NECK_TYPE="pafpn"
export PROMPT_ROUTE="mlp"
export DENSEBR_ENABLED=0
exec bash "$(cd "$(dirname "$0")" && pwd)/_run_ablation.sh" "$@"
