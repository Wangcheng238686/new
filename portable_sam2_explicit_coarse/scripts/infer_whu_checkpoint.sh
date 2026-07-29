#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
# shellcheck source=load_environment.sh
source "${PROJECT_ROOT}/scripts/load_environment.sh"

if [ "$#" -lt 1 ]; then
  echo "usage: bash scripts/infer_whu_checkpoint.sh CHECKPOINT [extra inference args]" >&2
  exit 2
fi

CHECKPOINT="$1"
shift

exec "${PYTHON}" inference/infer_from_checkpoint.py \
  --checkpoint "${CHECKPOINT}" \
  "$@"
