#!/usr/bin/env bash
set -Eeuo pipefail
export ABLATION_ID="r1_c5_pafpn_coarse_densebr_emb64"
export RUN_TAG="${RUN_TAG:-r1_c5_pafpn_coarse_densebr_emb64_semanticfix_${SHAPE_CONTEXT_FUSION:-roi_only}}"
export NECK_TYPE="pafpn"
export PROMPT_ROUTE="coarse"
export EXPLICIT_PROMPT_MODE="points_box_dense"
export DENSEBR_ENABLED=1
export SAM_IMAGE_EMBED_STRIDE=16
exec bash "$(cd "$(dirname "$0")" && pwd)/_run_ablation.sh" "$@"
