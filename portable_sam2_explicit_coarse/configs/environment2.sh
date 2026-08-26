#!/usr/bin/env bash
# HOST-SPECIFIC config for host "lthpc" (RTX 3090 x4). Do NOT activate on
# other machines. Machine-local defaults for machine2/fast-whu150 sessions.
#
# When porting this branch to another host: this file is inert unless a
# configs/environment.local.sh symlink points at it (the symlink itself is
# gitignored and never leaves this machine). On another host, either leave
# configs/environment.local.sh absent so scripts/load_environment.sh falls
# back to the shared configs/environment.sh, or create that host's own
# local symlink/file with its paths.
#
# Every entry below was verified against this machine on 2026-08-26:
#   - /data/wangcheng/envs/cvt2 (python 3.9.18, torch 2.8.0+cu128, 4x GPU)
#   - /data/wangcheng/dataset/WHU (2.1 train / 2.2 test / 2.3 valid /
#     2.4 annotation with train/validation/test COCO json)
#   - /data/wangcheng/pretrained-models/sam2/sam2_hiera_base_plus.pt
# Values already exported by the outer shell keep precedence, so one-off
# overrides remain possible without changing this file.
#
# Activation: configs/environment.local.sh symlinks to this file, and
# scripts/load_environment.sh prefers environment.local.sh over the shared
# configs/environment.sh. Delete the symlink to fall back to the shared file.

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

# This machine's execution defaults (all-acceleration-off; the fast wrapper
# opts in to AMP/batch/accum overrides itself). GPU note: wrappers that set
# their own CUDA_VISIBLE_DEVICES (the fast script defaults to 0,1) export it
# before the runner loads this file, so theirs wins; wrappers that do not set
# it fall back to the 1,2 below. All four GPUs are currently idle — adjust
# here or override in the launching shell if that changes.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
export AMP="${AMP:-0}"

unset _ENV_CONFIG_DIR _MAINLINE_ROOT _REPOSITORY_ROOT
