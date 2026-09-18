#!/usr/bin/env bash
# Main-regime K-sensitivity screen (dev100 protocol): three a3 arms that
# differ ONLY in DECODER_TAIL_NUM_POINTS, trained from scratch jointly (the
# a3 arm contract), one arm per GPU in the 1x1x8 single-card equivalent form.
#
# Complement of vhr10_p2v2_ksweep.sh (frozen-base margin tail, saturated at
# K=128): this screen answers the same K question in the MAIN regime that the
# matrix300 a3/a3r arms actually use (plain residual_v1 tail, joint training).
# Same seed + world_size=1 => identical sample order across arms => paired
# comparison (sync protocol §6, machine2 four-arm precedent).
#
# Default GPU map (GPU0 hosts an external job as of 2026-09-15):
#   GPU1 -> K=32   GPU2 -> K=64   GPU3 -> K=128
# Override: K_SCREEN_MAP="gpu:k gpu:k gpu:k"
#
# Usage:
#   bash scripts/ablations/vhr10_p2v2_dev100_k_screen.sh plan
#   nohup bash scripts/ablations/vhr10_p2v2_dev100_k_screen.sh run > /tmp/k_screen_dev100.log 2>&1 &
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PORTABLE_SAM2_CHECKPOINT_ROOT="${PORTABLE_SAM2_CHECKPOINT_ROOT:-/data1/wangcheng/checkpoint/portable_sam2_explicit_coarse}"

MAP="${K_SCREEN_MAP:-1:32 2:64 3:128}"
MODE="${1:-plan}"

echo "===== dev100 K-screen (${MODE}) ====="
for pair in ${MAP}; do
  gpu="${pair%%:*}"; k="${pair##*:}"
  TAG="vhr10_p2v2_dev100_a3_pbm_udprk${k}_tr1.0_va1.0"
  DIR="${PORTABLE_SAM2_CHECKPOINT_ROOT}/ablations/${TAG}"
  echo "--- GPU${gpu} K=${k} -> ${TAG}"
  if [ -n "$(ls -A "${DIR}" 2>/dev/null)" ]; then
    echo "    checkpoint dir non-empty, SKIPPING (delete to redo)"; continue
  fi
  if [ "${MODE}" != "run" ]; then continue; fi
  ( cd "${REPO_ROOT}" && \
    P2V2_PROTOCOL=dev100 A3_NUM_POINTS="${k}" \
    CUDA_VISIBLE_DEVICES="${gpu}" NPROC_PER_NODE=1 GRAD_ACCUM_STEPS=8 \
    nohup bash scripts/ablations/vhr10_p2v2_dev.sh a3 \
    > "/tmp/k_screen_dev100_k${k}.out" 2>&1 & )
done
[ "${MODE}" = "run" ] && echo "launched; per-arm stdout: /tmp/k_screen_dev100_k*.out"
echo "===== ${MODE} done ====="
