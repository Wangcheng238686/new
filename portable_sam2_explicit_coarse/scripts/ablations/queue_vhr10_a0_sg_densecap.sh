#!/usr/bin/env bash
# Durable, opt-in queue for the formal DenseCap comparison.  It never alters
# the frozen default matrix: both source-gradient-isolated rows are explicit.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

QUEUE_LOG="${QUEUE_LOG:-/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/vhr10_a0_sg_densecap_queue.log}"
POLL_SECONDS="${QUEUE_POLL_SECONDS:-60}"
mkdir -p "$(dirname "${QUEUE_LOG}")"
exec > >(tee -a "${QUEUE_LOG}") 2>&1

LOCK_FILE="${QUEUE_LOG}.lock"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "queue already active (lock=${LOCK_FILE}); refusing duplicate launch" >&2
  exit 1
fi

gpu_busy() {
  local gpu output
  for gpu in 0 1 2 3; do
    if ! output="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)"; then
      return 0
    fi
    if grep -Eq '^[[:space:]]*[0-9]+[[:space:]]*$' <<<"${output}"; then
      return 0
    fi
  done
  return 1
}

echo "============================================================"
echo "NWPU queue armed: a0_sg -> a0_sg_densecap64"
echo "Both are matrix300 fresh runs; only the second adds H_delta."
echo "watching GPUs 0,1,2,3; poll_seconds=${POLL_SECONDS}; pid=$$"
echo "queue_log=${QUEUE_LOG}"
echo "============================================================"
while gpu_busy; do
  echo "$(date '+%F %T') GPUs busy; waiting ${POLL_SECONDS}s"
  sleep "${POLL_SECONDS}"
done

echo "$(date '+%F %T') GPUs 0-3 idle; running no-launch contract checks"
DRY_RUN=1 bash "${SCRIPT_DIR}/vhr10_p2v2_matrix300.sh" a0_sg
DRY_RUN=1 bash "${SCRIPT_DIR}/vhr10_p2v2_matrix300.sh" a0_sg_densecap64

echo "$(date '+%F %T') preflight passed; launching foreground/serial matrix300 pair"
exec bash "${SCRIPT_DIR}/vhr10_p2v2_matrix300_series.sh" a0_sg a0_sg_densecap64
