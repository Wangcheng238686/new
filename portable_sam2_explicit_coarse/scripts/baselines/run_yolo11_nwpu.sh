#!/bin/bash
# =============================================================================
# YOLO11-seg (ultralytics 8.4, YOLO11s-seg) @ NWPU VHR-10 统一启动脚本
# 环境: cvt2 (ultralytics 以 --no-deps 安装, 不动 mm 栈)
#
# 用法: bash run_yolo11_nwpu.sh [mode] [gpu]
#   mode: convert(仅转标签) | smoke(默认) | train | resume | test
#   gpu : 默认 0
#
# 协议: train 520 / val 130 (NWPU 无独立 test, val 兼任 test 终评),
#       10 类, maxDets=100 / 不开 TTA
# 超参: ultralytics 官方默认配方 = 100ep, bs16, imgsz640, auto 优化器(lr0 0.01),
#       mosaic+mixup(阶段性关闭)+HSV+flip+尺度增强, seed=0 (全部默认, 不改)
# 初始化: yolo11s-seg.pt (官方 COCO 预训练, ultralytics 推荐做法)
# 许可: ultralytics 为 AGPL-3.0 (论文对比用途, 已知悉)
# =============================================================================
set -uo pipefail

MODE="${1:-smoke}"
GPU="${2:-0}"

PROJ_ROOT="/home/wangcheng/project/new/portable_sam2_explicit_coarse"
SCRIPTS="${PROJ_ROOT}/scripts/baselines"
DATA_YAML="/data1/wangcheng/dataset/NWPU VHR-10 dataset/yolo_seg/data.yaml"
RUNDIR="/data1/wangcheng/checkpoint/nwpu_vhr10_baselines/yolo11seg/yolo11s_seg_nwpu_default"
LOGDIR="${PROJ_ROOT}/logs/baselines"
WEIGHTS="${SCRIPTS}/yolo11s-seg.pt"
mkdir -p "${LOGDIR}" "${RUNDIR}"

RUNTAG="yolo11s_seg_nwpu_default"
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

{
  echo "===== runtag: ${RUNTAG} mode=${MODE} ====="
  echo "date: $(date '+%F %T')  host: $(hostname)  gpu: ${GPU}"
  echo "data: ${DATA_YAML}"
  echo "run dir: ${RUNDIR}"
  echo "conda env: cvt2  python: $(which python)"
  echo "ultralytics: $(python -c 'import ultralytics; print(ultralytics.__version__)')"
} | tee "${LOG}"

case "${MODE}" in
  convert)
    python "${SCRIPTS}/coco2yolo_seg_nwpu.py" 2>&1 | tee -a "${LOG}"
    ;;
  smoke)
    echo "[smoke] 1ep bs8, 10min 超时" | tee -a "${LOG}"
    start_gpu_monitor
    timeout 600 python - <<EOF 2>&1 | tee -a "${LOG}"
from ultralytics import YOLO
m = YOLO('${WEIGHTS}')
m.train(data='${DATA_YAML}', epochs=1, batch=8, imgsz=640, device=0,
        project='${RUNDIR}', name='smoke', seed=0, exist_ok=True, verbose=True)
EOF
    echo "[smoke] exit=$?" | tee -a "${LOG}"
    ;;
  train)
    start_gpu_monitor
    python - <<EOF 2>&1 | tee -a "${LOG}"
from ultralytics import YOLO
m = YOLO('${WEIGHTS}')
# 全部 ultralytics 默认配方: epochs=100 batch=16 imgsz=640, auto optimizer/scheduler/aug
m.train(data='${DATA_YAML}', device=0, project='${RUNDIR}',
        name='yolo11s_seg_nwpu_default', seed=0, exist_ok=True, verbose=True)
EOF
    echo "[train] exit=$?" | tee -a "${LOG}"
    ;;
  resume)
    start_gpu_monitor
    python - <<EOF 2>&1 | tee -a "${LOG}"
from ultralytics import YOLO
m = YOLO('${RUNDIR}/yolo11s_seg_nwpu_default/weights/last.pt')
m.train(resume=True, device=0)
EOF
    echo "[resume] exit=$?" | tee -a "${LOG}"
    ;;
  test)
    BEST="${RUNDIR}/yolo11s_seg_nwpu_default/weights/best.pt"
    [ -f "${BEST}" ] || { echo "[test] 未找到 best.pt" | tee -a "${LOG}"; exit 1; }
    echo "[test] ckpt: ${BEST}" | tee -a "${LOG}"
    python "${SCRIPTS}/eval_yolo_nwpu.py" "${BEST}" \
      --out-dir "${RUNDIR}/yolo11s_seg_nwpu_default/test_out" 2>&1 | tee -a "${LOG}"
    echo "[test] exit=$?" | tee -a "${LOG}"
    ;;
  *) echo "未知 mode: ${MODE}" | tee -a "${LOG}"; exit 1;;
esac

stop_gpu_monitor
echo "===== done $(date '+%F %T') =====" | tee -a "${LOG}"
