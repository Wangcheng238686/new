#!/bin/bash
# =============================================================================
# RS4D-box @ WHU-1024 实例分割对比实验（PromptMiner-SAM2 协议对齐版）
#
# 用法:
#   bash run_rs4d_whu1024.sh [mode] [gpu]
#     mode: smoke  – 冒烟测试(15min 超时, 1 epoch, 不存 ckpt 不验证)
#           train – 完整训练(800ep, val 选 best, bf16, bs8)
#           resume- 从最新 ckpt 续训
#           test  – 用 best ckpt 在 test.json(2220) 上出最终指标
#     gpu : 默认 0
#
# 协议要点(与作者 rs4d_bbox-whu.py 的差异, 详见 config 头注释):
#   - 数据: 本地 1024 COCO json (train 2943 全量不滤空图 / val 627 选模 / test 2220 终评)
#   - 蒸馏初始化: models_after_dis/20250227-021854 noise_20_20_500k (models.py:1157 已接本地路径)
#   - 超参保持论文值: AdamW lr 9.2224e-5 wd 0.05, cosine+50iter warmup, LSJ-1024, bf16
# =============================================================================
set -uo pipefail

MODE="${1:-smoke}"
GPU="${2:-0}"

PROJ_ROOT="/home/wangcheng/project/new/portable_sam2_explicit_coarse"
REPO="/home/wangcheng/project/RS4D"
CFG="configs/rs4d/rs4d_bbox-whu-1024.py"
WORKDIR="/data1/wangcheng/checkpoint/whu1024_baselines/rs4d/rs4d_bbox_whu1024_bs8_bf16_800ep"
RUNTAG="rs4d_bbox_whu1024_bs8_bf16_800ep"
LOGDIR="${PROJ_ROOT}/logs/baselines"
mkdir -p "${LOGDIR}"

TS=$(date +%Y%m%d_%H%M%S)
LOG="${LOGDIR}/${RUNTAG}_${MODE}_${TS}_pid$$.log"
GPULOG="${LOG%.log}.gpu.log"

# torch>=2.6 默认 weights_only=True 会拒载含 mmengine meta 的 ckpt(resume 需要)
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_VISIBLE_DEVICES="${GPU}"

env_note() {
  echo "===== runtag: ${RUNTAG} mode=${MODE} ====="
  echo "date: $(date '+%F %T')  host: $(hostname)  gpu: ${GPU} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
  echo "repo: ${REPO}  head: $(cd ${REPO} && git log --oneline -1 2>/dev/null)"
  echo "local patches: configs/rs4d/_base_/samseg-maskrcnn.py(custom_imports->mmdet.rs4d) mmdet/rs4d/models.py:1157(local ckpt path)"
  echo "config: ${CFG}"
  echo "work_dir: ${WORKDIR}"
  echo "data_root: /data1/wangcheng/dataset/WHU (train 2943 full / val 627 / test 2220)"
  echo "conda env: cvt2  python: $(which python 2>/dev/null)"
}

activate_env() {
  source /home/wangcheng/miniconda3/etc/profile.d/conda.sh
  conda activate cvt2
}

start_gpu_monitor() {
  (
    while true; do
      echo "[$(date '+%F %T')] $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i ${GPU} 2>/dev/null)"
      sleep 60
    done
  ) >> "${GPULOG}" 2>&1 &
  GPUPID=$!
}

stop_gpu_monitor() {
  [ -n "${GPUPID:-}" ] && kill "${GPUPID}" 2>/dev/null
}
trap stop_gpu_monitor EXIT

activate_env
cd "${REPO}"
export PYTHONPATH="${REPO}"

env_note | tee "${LOG}"

case "${MODE}" in
  smoke)
    echo "[smoke] 15min 超时, 1 epoch, 不验证不存 ckpt" | tee -a "${LOG}"
    start_gpu_monitor
    timeout 900 python tools/train.py "${CFG}" \
      --work-dir "${WORKDIR}/smoke" \
      --cfg-options \
        train_cfg.max_epochs=1 \
        train_cfg.val_interval=999 \
        default_hooks.checkpoint.interval=999 \
        default_hooks.logger.interval=5 \
      2>&1 | tee -a "${LOG}"
    rc=${PIPESTATUS[0]}
    echo "[smoke] exit=${rc} (124=超时截断属预期)" | tee -a "${LOG}"
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
  test)
    BEST=$(ls -t "${WORKDIR}"/best_coco_segm_mAP*.pth 2>/dev/null | head -1)
    if [ -z "${BEST}" ]; then echo "[test] 未找到 best ckpt" | tee -a "${LOG}"; exit 1; fi
    echo "[test] ckpt: ${BEST}" | tee -a "${LOG}"
    python tools/test.py "${CFG}" "${BEST}" --work-dir "${WORKDIR}/test_out" 2>&1 | tee -a "${LOG}"
    echo "[test] exit=${PIPESTATUS[0]}" | tee -a "${LOG}"
    ;;
  *)
    echo "未知 mode: ${MODE} (smoke|train|resume|test)" | tee -a "${LOG}"; exit 1;;
esac

stop_gpu_monitor
echo "===== done $(date '+%F %T') =====" | tee -a "${LOG}"
