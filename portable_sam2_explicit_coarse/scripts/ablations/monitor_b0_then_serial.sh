#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
# shellcheck source=../load_environment.sh
source "${PROJECT_ROOT}/scripts/load_environment.sh"

DEFAULT_B0_GLOB="${PORTABLE_SAM2_LOG_ROOT}/ablations/b0_aggregator_mlp_aligned_tr0.2_va1.0_*.log"
B0_LOG="${B0_LOG:-}"
WATCH_PID="${WATCH_PID:-}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MONITOR_TIMESTAMP="${MONITOR_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
MONITOR_LOG="${MONITOR_LOG:-${PORTABLE_SAM2_LOG_ROOT}/ablations/serial_after_b0_${MONITOR_TIMESTAMP}.log}"
MONITOR_LOCK="${MONITOR_LOCK:-${PORTABLE_SAM2_TMP_ROOT}/serial_after_b0.lock}"

DEFAULT_QUEUE=(
  b1_pafpn_mlp.sh
  m0_aggregator_mlp_full_image.sh
  m1_pafpn_mlp_full_image.sh
  c1_aggregator_coarse_points.sh
  c2_pafpn_coarse_points.sh
  c2l_pafpn_coarse_points_roi_loss.sh
  c2r_pafpn_coarse_points_roi_sam.sh
  c3_pafpn_coarse_points_box.sh
  c4_pafpn_coarse_points_box_dense.sh
  r1_c3_pafpn_coarse_points_box_emb64.sh
  r1_c4_pafpn_coarse_points_box_dense_emb64.sh
  r1_c4_rd_pafpn_coarse_points_box_raw_detach_emb64.sh
  r1_c4_g_pafpn_coarse_points_box_gaussian_emb64.sh
  r0_b0_aggregator_mlp_emb64.sh
  c5v2_pafpn_coarse_p2_boundary_refiner_emb64.sh
)

if [[ -n "${ABLATION_QUEUE:-}" ]]; then
  read -r -a QUEUE <<<"${ABLATION_QUEUE}"
else
  QUEUE=("${DEFAULT_QUEUE[@]}")
fi

usage() {
  printf '%s\n' \
    "Usage: B0_LOG=/path/to/b0.log bash scripts/ablations/monitor_b0_then_serial.sh" \
    "" \
    "Environment overrides:" \
    "  ABLATION_QUEUE='b1_pafpn_mlp.sh c1_aggregator_coarse_points.sh'" \
    "  WATCH_PID=12345 (for a foreground-launched torchrun)" \
    "  POLL_SECONDS=60" \
    "  MONITOR_LOG=/path/to/monitor.log" \
    "  --print-plan prints the resolved queue and exits."
}

if [[ "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "--print-plan" ]]; then
  printf 'poll_seconds=%s\n' "${POLL_SECONDS}"
  printf 'monitor_log=%s\n' "${MONITOR_LOG}"
  printf 'queue='
  printf ' %s' "${QUEUE[@]}"
  printf '\n'
  exit 0
fi

case "${POLL_SECONDS}" in
  ''|*[!0-9]*)
    printf 'POLL_SECONDS must be a positive integer, got %s\n' "${POLL_SECONDS}" >&2
    exit 2
    ;;
esac
if ((POLL_SECONDS < 1)); then
  printf 'POLL_SECONDS must be >= 1, got %s\n' "${POLL_SECONDS}" >&2
  exit 2
fi

mkdir -p "$(dirname "${MONITOR_LOG}")"
mkdir -p "$(dirname "${MONITOR_LOCK}")"

monitor_log() {
  printf '%s - serial-ablation-monitor - %s\n' \
    "$(date '+%F %T')" "$*" | tee -a "${MONITOR_LOG}"
}

if ! command -v flock >/dev/null 2>&1; then
  monitor_log "ERROR flock is unavailable; refusing to run without duplicate-monitor protection"
  exit 3
fi
exec 9>"${MONITOR_LOCK}"
if ! flock -n 9; then
  monitor_log "ERROR another serial monitor already owns ${MONITOR_LOCK}"
  exit 3
fi

if [[ -z "${B0_LOG}" ]]; then
  B0_LOG="$(find "${PORTABLE_SAM2_LOG_ROOT}/ablations" -maxdepth 1 -type f \
    -name 'b0_aggregator_mlp_aligned_tr0.2_va1.0_*.log' \
    -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR == 1 {sub(/^[^ ]+ /, ""); print}')"
fi
if [[ -z "${B0_LOG}" || ! -f "${B0_LOG}" ]]; then
  monitor_log "ERROR no B0 log found; searched ${DEFAULT_B0_GLOB}"
  exit 2
fi

B0_PID="${WATCH_PID}"
if [[ -z "${B0_PID}" ]]; then
  B0_PID="$(awk -F= '/^background_pid=/{pid=$2} END{print pid}' "${B0_LOG}")"
fi
if [[ -z "${B0_PID}" && "$(basename "${B0_LOG}")" =~ _pid([0-9]+)\.log$ ]]; then
  B0_PID="${BASH_REMATCH[1]}"
fi
if [[ ! "${B0_PID}" =~ ^[0-9]+$ ]]; then
  monitor_log "ERROR no valid WATCH_PID/background PID/log-name PID: ${B0_LOG}"
  exit 2
fi
B0_EPOCHS="$(awk -F= '/^epochs=/{epochs=$2} END{print epochs}' "${B0_LOG}")"
if [[ ! "${B0_EPOCHS}" =~ ^[0-9]+$ ]]; then
  B0_EPOCHS=80
fi

process_is_b0() {
  local state cmdline
  kill -0 "${B0_PID}" 2>/dev/null || return 1
  if [[ -r "/proc/${B0_PID}/stat" ]]; then
    state="$(awk '{print $3}' "/proc/${B0_PID}/stat" 2>/dev/null || true)"
    [[ "${state}" == "Z" ]] && return 1
  fi
  if [[ -r "/proc/${B0_PID}/cmdline" ]]; then
    cmdline="$(tr '\0' ' ' <"/proc/${B0_PID}/cmdline")"
    case "${cmdline}" in
      *torch.distributed.run*|*train_rsprompter_fusion.py*) return 0 ;;
      *) return 1 ;;
    esac
  fi
  return 0
}

monitor_log "START monitor_pid=$$ b0_pid=${B0_PID} b0_log=${B0_LOG}"
monitor_log "QUEUE ${QUEUE[*]}"

last_epoch=""
while process_is_b0; do
  current_epoch="$(sed -n 's/.*Epoch \([0-9][0-9]*\)\/[0-9][0-9]* |.*/\1/p' \
    "${B0_LOG}" | tail -n 1)"
  if [[ -n "${current_epoch}" && "${current_epoch}" != "${last_epoch}" ]]; then
    monitor_log "WAIT B0 completed_epoch=${current_epoch}/${B0_EPOCHS}"
    last_epoch="${current_epoch}"
  fi
  sleep "${POLL_SECONDS}"
done

if rg -q "Epoch ${B0_EPOCHS}/${B0_EPOCHS} \\||Early stopping triggered" "${B0_LOG}"; then
  monitor_log "B0_FINISHED status=success"
elif rg -q 'ChildFailedError|Traceback \(most recent call last\)|FAILED|SIGKILL|Signal 9' "${B0_LOG}"; then
  monitor_log "B0_FINISHED status=failed action=continue_queue"
else
  monitor_log "B0_FINISHED status=ended_without_completion_marker action=continue_queue"
fi

passed=0
failed=0
skipped=0
failed_tasks=()

for task in "${QUEUE[@]}"; do
  if [[ "${task}" = /* ]]; then
    task_path="${task}"
  else
    task_path="${SCRIPT_DIR}/${task}"
  fi
  if [[ ! -f "${task_path}" ]]; then
    monitor_log "TASK_SKIP task=${task} reason=missing_script"
    skipped=$((skipped + 1))
    failed_tasks+=("${task}:missing")
    continue
  fi

  monitor_log "TASK_START task=${task}"
  task_start="$(date +%s)"
  # The monitor owns fd 9 for flock; do not let torchrun/data-loader children
  # inherit it, otherwise a killed monitor leaves a stale lock until training ends.
  RUN_IN_BACKGROUND=0 bash "${task_path}" 9>&- >/dev/null 2>&1
  task_rc=$?
  task_end="$(date +%s)"
  task_seconds=$((task_end - task_start))
  if ((task_rc == 0)); then
    monitor_log "TASK_DONE task=${task} status=success elapsed_seconds=${task_seconds}"
    passed=$((passed + 1))
  else
    monitor_log "TASK_DONE task=${task} status=failed rc=${task_rc} elapsed_seconds=${task_seconds} action=continue_queue"
    failed=$((failed + 1))
    failed_tasks+=("${task}:rc${task_rc}")
  fi
done

monitor_log "QUEUE_DONE success=${passed} failed=${failed} skipped=${skipped}"
if ((${#failed_tasks[@]} > 0)); then
  monitor_log "FAILED_TASKS ${failed_tasks[*]}"
fi
exit 0
