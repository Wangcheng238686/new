#!/usr/bin/env bash
# Machine-local defaults for every active shell entry in this project.
#
# When moving the project to another machine, edit this file only. Values
# already exported by the outer shell keep precedence, so one-off overrides
# remain possible without changing this file.

_ENV_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_MAINLINE_ROOT="$(cd "${_ENV_CONFIG_DIR}/.." && pwd)"
_REPOSITORY_ROOT="$(cd "${_MAINLINE_ROOT}/.." && pwd)"

# Project/runtime paths.
export PORTABLE_SAM2_MAINLINE_ROOT="${PORTABLE_SAM2_MAINLINE_ROOT:-${_MAINLINE_ROOT}}"
export PORTABLE_SAM2_REPOSITORY_ROOT="${PORTABLE_SAM2_REPOSITORY_ROOT:-${_REPOSITORY_ROOT}}"
export PYTHON="${PYTHON:-/data/wangcheng/envs/cvt2/bin/python}"
export SAM2_REPO="${SAM2_REPO:-${PORTABLE_SAM2_REPOSITORY_ROOT}/sam2}"
export SAM2_CKPT="${SAM2_CKPT:-/data/wangcheng/pretrained-models/sam2/sam2_hiera_base_plus.pt}"
export WHU1024_DATA_ROOT="${WHU1024_DATA_ROOT:-/data/wangcheng/dataset/WHU}"

# Writable output and temporary roots.
export PORTABLE_SAM2_CHECKPOINT_ROOT="${PORTABLE_SAM2_CHECKPOINT_ROOT:-/data/wangcheng/checkpoint/portable_sam2_explicit_coarse}"
export PORTABLE_SAM2_LOG_ROOT="${PORTABLE_SAM2_LOG_ROOT:-${PORTABLE_SAM2_MAINLINE_ROOT}/logs}"
export PORTABLE_SAM2_TMP_ROOT="${PORTABLE_SAM2_TMP_ROOT:-/tmp/portable_sam2_explicit_coarse}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${PORTABLE_SAM2_TMP_ROOT}/mpl}"

# Current machine execution defaults. The effective global batch remains 8.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
export AMP="${AMP:-0}"

unset _ENV_CONFIG_DIR _MAINLINE_ROOT _REPOSITORY_ROOT
