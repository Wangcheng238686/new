#!/usr/bin/env bash
# Shared loader. Source this file; edit configs/environment.sh, not this file.

_ENV_LOADER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_DEFAULT_ENV_FILE="$(cd "${_ENV_LOADER_DIR}/../configs" && pwd)/environment.sh"
PORTABLE_SAM2_ENV_FILE="${PORTABLE_SAM2_ENV_FILE:-${_DEFAULT_ENV_FILE}}"

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

unset _ENV_LOADER_DIR _DEFAULT_ENV_FILE _ENV_REQUIRED _ENV_NAME
