#!/usr/bin/env bash
# Shared loader. Source this file. The loader prefers a machine-local
# configs/environment.local.sh when present and otherwise falls back to
# configs/environment.sh. To force a specific file, set PORTABLE_SAM2_ENV_FILE.

_ENV_LOADER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_CONFIGS_DIR="$(cd "${_ENV_LOADER_DIR}/../configs" && pwd)"
# Prefer a machine-local environment.local.sh when present (per-host overrides),
# otherwise fall back to the committed environment.sh. Override explicitly with
# PORTABLE_SAM2_ENV_FILE.
if [[ -z "${PORTABLE_SAM2_ENV_FILE:-}" ]]; then
  if [[ -r "${_CONFIGS_DIR}/environment.local.sh" ]]; then
    PORTABLE_SAM2_ENV_FILE="${_CONFIGS_DIR}/environment.local.sh"
  else
    PORTABLE_SAM2_ENV_FILE="${_CONFIGS_DIR}/environment.sh"
  fi
fi

if [[ ! -r "${PORTABLE_SAM2_ENV_FILE}" ]]; then
  echo "Portable SAM2 environment config is not readable: ${PORTABLE_SAM2_ENV_FILE}" >&2
  return 2
fi

# shellcheck disable=SC1090
source "${PORTABLE_SAM2_ENV_FILE}"
PORTABLE_SAM2_ENV_FILE="$(cd "$(dirname "${PORTABLE_SAM2_ENV_FILE}")" && pwd)/$(basename "${PORTABLE_SAM2_ENV_FILE}")"
export PORTABLE_SAM2_ENV_FILE

_ENV_REQUIRED=(
  PYTHON SAM2_REPO SAM2_CKPT WHU1024_DATA_ROOT
  PORTABLE_SAM2_CHECKPOINT_ROOT PORTABLE_SAM2_LOG_ROOT PORTABLE_SAM2_TMP_ROOT
  CUDA_VISIBLE_DEVICES NPROC_PER_NODE BATCH_SIZE GRAD_ACCUM_STEPS AMP
)
for _ENV_NAME in "${_ENV_REQUIRED[@]}"; do
  if [[ -z "${!_ENV_NAME:-}" ]]; then
    echo "Required setting ${_ENV_NAME} is empty in ${PORTABLE_SAM2_ENV_FILE}" >&2
    return 2
  fi
done

unset _ENV_LOADER_DIR _CONFIGS_DIR _ENV_REQUIRED _ENV_NAME
