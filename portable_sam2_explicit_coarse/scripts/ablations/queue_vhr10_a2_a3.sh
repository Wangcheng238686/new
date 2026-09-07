#!/usr/bin/env bash
# Durable local queue for the remaining NWPU A-series arms: A2 then A3.
# Both arms require all four local GPUs, so never run them concurrently.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

QUEUE_LOG="${QUEUE_LOG:-/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/vhr10_a2_a3_queue.log}"
POLL_SECONDS="${QUEUE_POLL_SECONDS:-60}"
mkdir -p "$(dirname "${QUEUE_LOG}")"
exec > >(tee -a "${QUEUE_LOG}") 2>&1

# A second accidental invocation must fail rather than later launch duplicate
# runs into the same checkpoint directories.
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
      # A failed state query is not evidence that a card is safe to use.
      return 0
    fi
    if grep -Eq '^[[:space:]]*[0-9]+[[:space:]]*$' <<<"${output}"; then
      return 0
    fi
  done
  return 1
}

echo "============================================================"
echo "NWPU queue armed: A2 (default R1, NOT A2e) -> A3 (PBM+UDPR-K64)"
echo "watching GPUs 0,1,2,3; poll_seconds=${POLL_SECONDS}; pid=$$"
echo "queue_log=${QUEUE_LOG}"
echo "============================================================"
while gpu_busy; do
  echo "$(date '+%F %T') GPUs busy; waiting ${POLL_SECONDS}s"
  sleep "${POLL_SECONDS}"
done

echo "$(date '+%F %T') GPUs 0-3 idle; running no-launch contract checks"
DRY_RUN=1 bash "${SCRIPT_DIR}/vhr10_p2v2_dev.sh" a2
DRY_RUN=1 bash "${SCRIPT_DIR}/vhr10_p2v2_dev.sh" a3

echo "$(date '+%F %T') preflight passed; launching A2 then A3 foreground/serial"
exec bash "${SCRIPT_DIR}/vhr10_p2v2_dev_series.sh" a2 a3
