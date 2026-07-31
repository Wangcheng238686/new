#!/usr/bin/env bash
# Monitor that waits until all four GPUs are free (no train_rsprompter process),
# then launches the ue_coco R1-C4-RD experiment on all 4 cards.
#
# Run detached:
#   nohup setsid bash scripts/ablations_uecoco/wait_then_launch_r1_c4_rd.sh \
#     >/dev/null 2>&1 &
#
# Events are logged to logs/ablations_uecoco/wait_then_launch_<ts>.log.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

MONITOR_TAG="wait_then_launch_uecoco_r1_c4_rd"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MONITOR_LOG="logs/ablations_uecoco/${MONITOR_TAG}_${RUN_TIMESTAMP}.log"
mkdir -p "$(dirname "${MONITOR_LOG}")"
exec > >(tee -a "${MONITOR_LOG}") 2>&1

POLL_SECONDS="${POLL_SECONDS:-60}"

echo "============================================================"
echo "${MONITOR_TAG} started at $(date '+%Y-%m-%d %H:%M:%S')"
echo "monitor_log=${MONITOR_LOG}"
echo "poll_seconds=${POLL_SECONDS}"
echo "plan: when no train_rsprompter process is alive, launch"
echo "  ue_coco R1-C4-RD on all 4 GPUs (CUDA_VISIBLE_DEVICES=0,1,2,3)"
echo "============================================================"

while true; do
  running="$(ps aux | grep 'train_rsprompter' | grep -v grep | wc -l)"
  if [ "${running}" -eq 0 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - no train_rsprompter process; GPUs free. Launching ue_coco R1-C4-RD."
    break
  fi
  echo "$(date '+%Y-%m-%d %H:%M:%S') - ${running} train_rsprompter process(es) still running; sleeping ${POLL_SECONDS}s."
  sleep "${POLL_SECONDS}"
done

# Small settle delay so a just-exited experiment releases GPU memory fully.
sleep 15

# Launch ue_coco R1-C4-RD on all 4 cards.
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash "${SCRIPT_DIR}/r1_c4_rd_pafpn_coarse_points_box_raw_detach_emb64.sh" \
  --master-port 29801
echo "$(date '+%Y-%m-%d %H:%M:%S') - ue_coco R1-C4-RD launched on GPU 0,1,2,3."

echo "============================================================"
echo "$(date '+%Y-%m-%d %H:%M:%S') - ${MONITOR_TAG} done. Experiment is running detached."
echo "Follow its log:"
echo "  tail -f logs/ablations_uecoco/uecoco_r1_c4_rd_*_tr1.0_va1.0_*.log"
echo "============================================================"
