#!/usr/bin/env bash
# Queue the 4-GPU WHU full A3sgm run behind the running NWPU a3sgm_b05 job.
#
# Arm: whu1024_a3sgm_roilocal_full_fast (NWPU 最优 A3sgm margin-tail × fast150
# 监督基底；P2BRR 槽位由 margin 尾顶替)。目的：A3sgm 配方在 WHU 全量上尽量
# 复现 fast150 轨迹与指标（锚点 E2 0.4521 / E150 0.7493 val，test 0.7342）。
#
# Form: 4 GPU × bs1 × accum2（有效 batch 8、368 步/epoch，与协议 §11 的
# 2×1×4 fp32 形态等价），AMP=0（本机 fp16 在 A3 形态发散，见 wrapper 头注）。
# `-- --decoupled-aux-clip` 经 _run_ablation.sh 的 TRAINER_EXTRA_ARGS 透传——
# WHU runner 不翻译 DECOUPLED_AUX_GRAD_CLIP env，缺此旗标会静默跑丢
# a3sgm 配方的解耦辅助裁剪半边（已用 DRY_RUN 验证落位）。
#
# Usage: nohup bash scripts/ablations/queue_whu_a3sgm_4gpu.sh &
#   WAIT_PID  被等待的 NWPU torchrun pid（默认 3532291）
#   POLL_SECS 轮询间隔（默认 300）
set -Eeuo pipefail

WAIT_PID="${WAIT_PID:-3532291}"
POLL_SECS="${POLL_SECS:-300}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

CKPT_DIR="/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/whu1024_a3sgm_roilocal_full_fast_tr1.0_va1.0"

log() { echo "[queue_whu_a3sgm $(date '+%F %T')] $*"; }

if [ -e "${CKPT_DIR}" ] && ls "${CKPT_DIR}"/best_model*.pth >/dev/null 2>&1; then
  log "ABORT: ${CKPT_DIR} 已存在 best_model——防止误重跑，人工确认后清理再排队。"
  exit 3
fi

if kill -0 "${WAIT_PID}" 2>/dev/null; then
  log "waiting for NWPU job pid=${WAIT_PID} (vhr10_p2v2_matrix300_a3sgm_b05) to finish; poll=${POLL_SECS}s"
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep "${POLL_SECS}"
  done
  log "pid=${WAIT_PID} exited."
else
  log "pid=${WAIT_PID} already gone — proceeding (GPUs assumed free)."
fi

# 给驱动/进程树留释放时间，再记录 GPU 状态（仅记录，不作硬门）。
sleep 120
log "nvidia-smi after release:" || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | while read -r line; do
  log "  ${line}"
done

log "launching WHU A3sgm 4-GPU fp32 (whu1024_a3sgm_roilocal_full_fast) ..."
exec env \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  NPROC_PER_NODE=4 \
  GRAD_ACCUM_STEPS=2 \
  AMP=0 \
  RUN_IN_BACKGROUND=0 \
  bash scripts/ablations/whu1024_a3sgm_roilocal_full_fast.sh -- --decoupled-aux-clip
