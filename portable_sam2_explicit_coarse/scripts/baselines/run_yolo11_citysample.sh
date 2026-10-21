#!/bin/bash
# =============================================================================
# YOLO11-seg (ultralytics 8.4, YOLO11s-seg) @ CitySample-Sat 512-native 无增强版
# 环境: cvt2 (ultralytics 以 --no-deps 安装, 不动 mm 栈)
#
# 用法: bash run_yolo11_citysample.sh [mode] [gpu]
#   mode: convert(仅转标签) | smoke(默认) | train | resume | test
#   gpu : 默认 2
#
# 协议: train 2142 全量 / val 459 选 best(ultralytics fitness) /
#       test 459 终评 maxDets=250 (P1/P2: test 单图实例最大 238) / 不开 TTA
# 无增强协议(与 Ours 单流卫星无增强版公平对比):
#   imgsz=512 原生不上采样(P4); 全部数据增强关闭——
#   mosaic/mixup/copy_paste/HSV/degrees/translate/scale/shear/perspective/
#   flipud/fliplr/bgr/multi_scale 全 0/False
# 超参: 除增强外保持 ultralytics 官方默认配方: 100ep, bs16, auto 优化器
#       (lr0 0.01), seed=0; 初始化 yolo11s-seg.pt (官方 COCO 预训练,
#       初始化≠增强, 与 WHU 协议一致保留)
# 许可: ultralytics 为 AGPL-3.0 (论文对比用途, 已知悉)
# =============================================================================
set -uo pipefail

MODE="${1:-smoke}"
GPU="${2:-2}"

PROJ_ROOT="/home/wangcheng/project/new/portable_sam2_explicit_coarse"
SCRIPTS="${PROJ_ROOT}/scripts/baselines"
DATA_YAML="/data1/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v3_20261021/yolo_seg/data.yaml"
RUNDIR="/data1/wangcheng/checkpoint/citysample_sat_baselines/yolo11seg/yolo11s_seg_citysample_v3_512noaug"
LOGDIR="${PROJ_ROOT}/logs/baselines_citysample"
mkdir -p "${LOGDIR}" "${RUNDIR}"

RUNTAG="yolo11s_seg_citysample_v3_512noaug"
RUNNAME="yolo11s_seg_citysample_v3_512noaug"
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
    python "${SCRIPTS}/coco2yolo_seg_citysample.py" 2>&1 | tee -a "${LOG}"
    ;;
  smoke)
    echo "[smoke] 1ep bs8, 10min 超时" | tee -a "${LOG}"
    start_gpu_monitor
    timeout 600 python - <<EOF 2>&1 | tee -a "${LOG}"
from ultralytics import YOLO
m = YOLO('/home/wangcheng/project/new/yolo11s-seg.pt')
m.train(data='${DATA_YAML}', epochs=1, batch=8, imgsz=512, device=0,
        project='${RUNDIR}', name='smoke', seed=0, exist_ok=True, verbose=True,
        mosaic=0.0, mixup=0.0, copy_paste=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.0, bgr=0.0, multi_scale=False)
EOF
    echo "[smoke] exit=$?" | tee -a "${LOG}"
    ;;
  train)
    start_gpu_monitor
    python - <<EOF 2>&1 | tee -a "${LOG}"
from ultralytics import YOLO
m = YOLO('/home/wangcheng/project/new/yolo11s-seg.pt')
# 无增强协议(公平对比): imgsz=512 原生 + 全部增强关闭;
# 其余(100ep/bs16/auto 优化器/seed=0)保持 ultralytics 官方默认
m.train(data='${DATA_YAML}', device=0, project='${RUNDIR}',
        name='${RUNNAME}', seed=0, exist_ok=True, verbose=True,
        imgsz=512,
        mosaic=0.0, mixup=0.0, copy_paste=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.0, bgr=0.0, multi_scale=False)
EOF
    echo "[train] exit=$?" | tee -a "${LOG}"
    ;;
  resume)
    start_gpu_monitor
    python - <<EOF 2>&1 | tee -a "${LOG}"
from ultralytics import YOLO
m = YOLO('${RUNDIR}/${RUNNAME}/weights/last.pt')
m.train(resume=True, device=0)
EOF
    echo "[resume] exit=$?" | tee -a "${LOG}"
    ;;
  test)
    BEST="${RUNDIR}/${RUNNAME}/weights/best.pt"
    [ -f "${BEST}" ] || { echo "[test] 未找到 best.pt" | tee -a "${LOG}"; exit 1; }
    echo "[test] ckpt: ${BEST}" | tee -a "${LOG}"
    python "${SCRIPTS}/eval_yolo_citysample.py" "${BEST}" \
      --out-dir "${RUNDIR}/${RUNNAME}/test_out" 2>&1 | tee -a "${LOG}"
    echo "[test] exit=$?" | tee -a "${LOG}"
    ;;
  *) echo "未知 mode: ${MODE}" | tee -a "${LOG}"; exit 1;;
esac

stop_gpu_monitor
echo "===== done $(date '+%F %T') =====" | tee -a "${LOG}"
