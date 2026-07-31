#!/usr/bin/env bash
# Monitor that waits until all four GPUs are free (no train_rsprompter process),
# then launches B0 (stride32) on GPU 0,1 and R0 (stride16) on GPU 2,3 in
# parallel.  B0 and R0 are a clean single-variable pair (aggregator + MLP +
# roi_local, only SAM_IMAGE_EMBED_STRIDE differs) that adjudicates whether the
# stride-32 -> stride-16 resolution change alone improves segm/mAP.
#
# Run detached:
#   nohup setsid bash scripts/ablations/wait_then_launch_b0_r0.sh \
#     >/dev/null 2>&1 &
#
# Events are logged to logs/ablations/wait_then_launch_b0_r0_<ts>.log.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

MONITOR_TAG="wait_then_launch_b0_r0"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MONITOR_LOG="logs/ablations/${MONITOR_TAG}_${RUN_TIMESTAMP}.log"
mkdir -p "$(dirname "${MONITOR_LOG}")"
exec > >(tee -a "${MONITOR_LOG}") 2>&1

POLL_SECONDS="${POLL_SECONDS:-60}"

echo "============================================================"
echo "${MONITOR_TAG} started at $(date '+%Y-%m-%d %H:%M:%S')"
echo "monitor_log=${MONITOR_LOG}"
echo "poll_seconds=${POLL_SECONDS}"
echo "plan: when no train_rsprompter process is alive, launch"
echo "  B0  -> CUDA_VISIBLE_DEVICES=0,1 (stride32)"
echo "  R0  -> CUDA_VISIBLE_DEVICES=2,3 (stride16)"
echo "============================================================"

# Wait until the machine is idle.
while true; do
  running="$(ps aux | grep 'train_rsprompter' | grep -v grep | wc -l)"
  if [ "${running}" -eq 0 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - no train_rsprompter process; GPUs free. Launching B0 and R0."
    break
  fi
  echo "$(date '+%Y-%m-%d %H:%M:%S') - ${running} train_rsprompter process(es) still running; sleeping ${POLL_SECONDS}s."
  sleep "${POLL_SECONDS}"
done

# Small settle delay so a just-exited experiment releases GPU memory fully.
sleep 15

# Launch B0 on GPU 0,1 (stride32, the legacy baseline).
CUDA_VISIBLE_DEVICES=0,1 bash "${SCRIPT_DIR}/b0_aggregator_mlp.sh" --master-port 29701
echo "$(date '+%Y-%m-%d %H:%M:%S') - B0 launched on GPU 0,1."

# Launch R0 on GPU 2,3 (stride16, the resolution change).
CUDA_VISIBLE_DEVICES=2,3 bash "${SCRIPT_DIR}/r0_b0_aggregator_mlp_emb64.sh" --master-port 29702
echo "$(date '+%Y-%m-%d %H:%M:%S') - R0 launched on GPU 2,3."

echo "============================================================"
echo "$(date '+%Y-%m-%d %H:%M:%S') - ${MONITOR_TAG} done. Both experiments are running detached."
echo "Follow their logs:"
echo "  B0: tail -f logs/ablations/b0_aggregator_mlp_aligned_tr0.2_va1.0_*.log"
echo "  R0: tail -f logs/ablations/r0_b0_aggregator_mlp_emb64_tr0.2_va1.0_*.log"
echo "============================================================"
