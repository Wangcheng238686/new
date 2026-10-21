#!/bin/bash
# =============================================================================
# RSPrompter (query) @ WHU-1024 对比实验（PromptMiner-SAM2 协议对齐版）
#
# 用法: bash run_rsprompter_whu1024.sh [mode] [gpu]
#   mode: smoke | train(300ep, AMP fp16, bs2x累积2) | resume | test
#   gpu : 默认 1
# 协议: train 2943 全量 / val 627 选 best(segm_mAP) / test 2220 终评 / maxDets 100
# 超参: 论文配方 AdamW 1e-4 wd0.05 + clip0.1, LSJ-1024+hflip0.5, 300ep cosine
# 注: 不用 DeepSpeed(官方注释的 AMP 备选路径); SAM ViT-B HF 权重在 work_dirs/sam_cache/
# =============================================================================
set -uo pipefail

MODE="${1:-smoke}"
GPU="${2:-1}"

PROJ_ROOT="/home/wangcheng/project/new/portable_sam2_explicit_coarse"
REPO="/home/wangcheng/project/RSPrompter-release/RSPrompter-release"
CFG="configs/rsprompter/rsprompter_anchor-whu-1024.py"
WORKDIR="/data1/wangcheng/checkpoint/whu1024_baselines/rsprompter/rsprompter_anchor_whu1024_bs1ai2_300ep"
RUNTAG="rsprompter_anchor_whu1024_bs1ai2_300ep"
LOGDIR="${PROJ_ROOT}/logs/baselines"
mkdir -p "${LOGDIR}"

# torch>=2.6 默认 weights_only=True 会拒载含 mmengine meta 的 ckpt(resume 需要)
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_VISIBLE_DEVICES="${GPU}"

TS=$(date +%Y%m%d_%H%M%S)
LOG="${LOGDIR}/${RUNTAG}_${MODE}_${TS}_pid$$.log"
GPULOG="${LOG%.log}.gpu.log"

activate_env() {
  source /home/wangcheng/miniconda3/etc/profile.d/conda.sh
  conda activate cvt2
}
start_gpu_monitor() {
  ( while true; do
      echo "[$(date '+%F %T')] $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i ${GPU} 2>/dev/null)"
      sleep 60
    done ) >> "${GPULOG}" 2>&1 &
  GPUPID=$!
}
stop_gpu_monitor() { [ -n "${GPUPID:-}" ] && kill "${GPUPID}" 2>/dev/null; }
trap stop_gpu_monitor EXIT

activate_env
cd "${REPO}"
export PYTHONPATH="${REPO}"

{
  echo "===== runtag: ${RUNTAG} mode=${MODE} ====="
  echo "date: $(date '+%F %T')  host: $(hostname)  gpu: ${GPU}"
  echo "repo: ${REPO}"
  echo "config: ${CFG}  work_dir: ${WORKDIR}"
  echo "data_root: /data1/wangcheng/dataset/WHU (train 2943 full / val 627 / test 2220)"
  echo "conda env: cvt2  python: $(which python)"
} | tee "${LOG}"

case "${MODE}" in
  smoke)
    echo "[smoke] 12min 超时, 1 epoch, 不验证不存 ckpt" | tee -a "${LOG}"
    start_gpu_monitor
    timeout 720 python tools/train.py "${CFG}" \
      --work-dir "${WORKDIR}/smoke" \
      --cfg-options \
        train_cfg.max_epochs=1 \
        train_cfg.val_interval=999 \
        default_hooks.checkpoint.interval=999 \
        default_hooks.logger.interval=5 \
      2>&1 | tee -a "${LOG}"
    echo "[smoke] exit=${PIPESTATUS[0]} (124=超时属预期)" | tee -a "${LOG}"
    ;;
  train)
    start_gpu_monitor
    python tools/train.py "${CFG}" --work-dir "${WORKDIR}" 2>&1 | tee -a "${LOG}"
    echo "[train] exit=${PIPESTATUS[0]}" | tee -a "${LOG}"
    ;;
  resume)
    start_gpu_monitor
    python tools/train.py "${CFG}" --work-dir "${WORKDIR}" --resume 2>&1 | tee -a "${LOG}"
    echo "[resume] exit=${PIPESTATUS[0]}" | tee -a "${LOG}"
    ;;
  *)
    echo "未知 mode: ${MODE}" | tee -a "${LOG}"; exit 1;;
esac

stop_gpu_monitor
echo "===== done $(date '+%F %T') =====" | tee -a "${LOG}"
