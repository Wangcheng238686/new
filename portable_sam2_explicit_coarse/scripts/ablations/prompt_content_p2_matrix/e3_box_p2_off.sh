#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LEARNING_RATE="${LEARNING_RATE:-5e-5}"
export MAX_EPOCHS="${MAX_EPOCHS:-20}"
export EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-5}"
export EARLY_STOPPING_START_EPOCH="${EARLY_STOPPING_START_EPOCH:-1}"
export RUN_IN_BACKGROUND=0
exec bash "${SCRIPT_DIR}/../prompt_content_p2_from_r1_best.sh" box 0 "$@"
