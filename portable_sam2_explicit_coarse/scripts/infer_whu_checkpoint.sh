#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [ "$#" -lt 1 ]; then
  echo "usage: bash scripts/infer_whu_checkpoint.sh CHECKPOINT [extra inference args]" >&2
  exit 2
fi

CHECKPOINT="$1"
shift

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/portable_sam2_explicit_coarse_mpl}"
export SAM2_REPO="${SAM2_REPO:-$(cd "${PROJECT_ROOT}/../sam2" && pwd)}"

PYTHON="${PYTHON:-/data/wangcheng/envs/cvt2/bin/python}"
exec "${PYTHON}" inference/infer_from_checkpoint.py \
  --checkpoint "${CHECKPOINT}" \
  "$@"
