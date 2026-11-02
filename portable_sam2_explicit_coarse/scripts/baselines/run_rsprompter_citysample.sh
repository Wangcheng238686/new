#!/bin/bash
# =============================================================================
# RSPrompter (anchor) @ CitySample-Sat v1 512-native + 默认增强(LSJ-512 锚定) 对比实验
#
# 用法: bash run_rsprompter_citysample.sh [mode] [gpu]
#   mode: smoke | train(300ep, AMP bf16, bs1x累积2) | resume | test
#   gpu : 默认 1
# 协议: train 2142 全量 / val 459 选 best(segm_mAP) / test 459 终评
#       maxDets=250 (P1/P2: test 单图实例最大 238) / 不上采样 / 无数据增强
#       (与 Ours 单流卫星无增强版公平对比; 无增强=去 RandomFlip+LSJ)
# 超参: 沿用 WHU v2 数值口径: AdamW 2e-4 wd0.05 + clip0.1, 300ep cosine,
#       bf16 (fp16 300ep 全线 NaN 报废的既定教训), bs1 x accum2
# test: citysample_sat_test.py (pycocotools maxDets=[1,10,250])
# 注: 不用 DeepSpeed(官方注释的 AMP 备选路径); SAM ViT-B HF 权重在
#       work_dirs/sam_cache/ (与 WHU 版共用)
# =============================================================================
set -uo pipefail

MODE="${1:-smoke}"
GPU="${2:-1}"

PROJ_ROOT="/home/wangcheng/project/new/portable_sam2_explicit_coarse"
SCRIPTS="${PROJ_ROOT}/scripts/baselines"
REPO="/home/wangcheng/project/RSPrompter-release/RSPrompter-release"
CFG="configs/rsprompter/rsprompter_anchor-citysample-v1-512defaug.py"
WORKDIR="/data1/wangcheng/checkpoint/citysample_sat_baselines/rsprompter/rsprompter_anchor_citysample_v1_512defaug_bs1ai2_300ep"
RUNTAG="rsprompter_anchor_citysample_v1_512defaug_bs1ai2_300ep"
LOGDIR="${PROJ_ROOT}/logs/baselines_citysample"
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
  echo "data: /data1/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021 (satellite-only, train 2142 / val 459 / test 459, no-aug, 512-native)"
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
  test)
    # val best ckpt 自动探测(save_best 命名 best_coco_segm_mAP_epoch_*.pth)
    BEST=$(ls -t "${WORKDIR}"/best_coco_segm_mAP_epoch_*.pth 2>/dev/null | head -1)
    if [ -z "${BEST}" ]; then
      echo "[test] 未找到 best ckpt 于 ${WORKDIR}" | tee -a "${LOG}"; exit 1
    fi
    echo "[test] ckpt: ${BEST}" | tee -a "${LOG}"
    python "${SCRIPTS}/citysample_sat_test.py" "${CFG}" "${BEST}" \
      --out-dir "${WORKDIR}/test_out" \
      --dump-prefix "${WORKDIR}/test_out/pred" 2>&1 | tee -a "${LOG}"
    echo "[test] exit=$?" | tee -a "${LOG}"
    ;;
  *) echo "未知 mode: ${MODE}" | tee -a "${LOG}"; exit 1;;
esac

stop_gpu_monitor
echo "===== done $(date '+%F %T') =====" | tee -a "${LOG}"
