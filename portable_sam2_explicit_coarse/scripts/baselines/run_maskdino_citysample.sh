#!/bin/bash
# =============================================================================
# MaskDINO R50 @ CitySample-Sat v1 (512-native 基准 + 默认增强) 统一启动脚本
# 覆盖: maskdino   (环境: mdino, PYTHONPATH=MaskDINO 仓)
#
# 用法: bash run_maskdino_citysample.sh [mode] [gpu]
#   mode: smoke(默认) | train | resume | test
#   gpu : 默认 0
# 协议: train 2142 全量 / val 459 选模(日志解析 best) / test 459 终评
#       maxDets=250 + BoundaryAP(eval_maskdino_citysample.py)
# 300ep=321300 micro-iters, bs2 x accum8 = 有效16; 512 锚定(MIN_SIZE 512)
# =============================================================================
set -uo pipefail

MODE="${1:-smoke}"
GPU="${2:-0}"

PROJ_ROOT="/home/wangcheng/project/new/portable_sam2_explicit_coarse"
SCRIPTS="${PROJ_ROOT}/scripts/baselines"
REPO="/home/wangcheng/project/MaskDINO"
CFG="configs/citysample/maskdino_R50_citysample_v1_512defaug.yaml"
RUNTAG="maskdino_R50_citysample_v1_512defaug_bs2ai8_300ep"
WORKDIR="/data1/wangcheng/checkpoint/citysample_sat_baselines/maskdino/maskdino_R50_citysample_v1_512defaug_bs2ai8_300ep"
LOGDIR="${PROJ_ROOT}/logs/baselines_citysample"
MDINO_PY=/data2/wangcheng/envs/mdino/bin/python
mkdir -p "${LOGDIR}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_VISIBLE_DEVICES="${GPU}"

# ---- 协议B(minimal)变体: VARIANT=minimal 切 entry/config/runtag ----
TRAIN_ENTRY_PY="train_citysample.py"
if [ "${VARIANT:-}" = "minimal" ]; then
  CFG="configs/citysample/maskdino_R50_citysample_v1_minimal.yaml"
  RUNTAG="maskdino_R50_citysample_v1_minimal_bs2ai8_300ep"
  TRAIN_ENTRY_PY="train_citysample_minimal.py"
  WORKDIR="/data1/wangcheng/checkpoint/citysample_sat_baselines/maskdino/${RUNTAG}"
fi

TS=$(date +%Y%m%d_%H%M%S)
LOG="${LOGDIR}/${RUNTAG}_${MODE}_${TS}_pid$$.log"
GPULOG="${LOG%.log}.gpu.log"

activate_env() {
  source /home/wangcheng/miniconda3/etc/profile.d/conda.sh
  conda activate /data2/wangcheng/envs/mdino
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

best_ckpt() {  # metrics.json 解析 val segm AP 最高 ckpt
  "${MDINO_PY}" - "$1" << 'PYEOF'
import json, sys, os, glob
od = sys.argv[1]
best_iter, best_ap = None, -1
for mj in glob.glob(os.path.join(od, 'metrics.json')):
    for line in open(mj):
        try: d = json.loads(line)
        except Exception: continue
        ap = d.get('segm/AP')
        if ap is not None and 'iteration' in d and ap > best_ap:
            best_ap, best_iter = ap, d['iteration']
if best_iter is None:
    sys.exit(1)
cands = sorted(glob.glob(os.path.join(od, f'model_*{best_iter}.pth')))
print(cands[-1] if cands else '')
PYEOF
}

activate_env
cd "${REPO}"
export PYTHONPATH="${REPO}"

{
  echo "===== runtag: ${RUNTAG} mode=${MODE} ====="
  echo "date: $(date '+%F %T')  host: $(hostname)  gpu: ${GPU}"
  echo "repo: ${REPO}  head: $(git log --oneline -1 2>/dev/null)"
  echo "config: ${CFG}  work_dir: ${WORKDIR}"
  echo "data: /data1/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021 (train 2142 / val 459 / test 459, 512-native defaug)"
  echo "conda env: mdino  python: $(which python)"
} | tee "${LOG}"

case "${MODE}" in
  smoke)
    echo "[smoke] 10min 超时, 100 micro-iter, 不验证不存 ckpt" | tee -a "${LOG}"
    start_gpu_monitor
    timeout 600 "${MDINO_PY}" "${TRAIN_ENTRY_PY}" --config-file "${CFG}" --accum 8 \
      OUTPUT_DIR "${WORKDIR}/smoke" \
      SOLVER.MAX_ITER 100 SOLVER.IMS_PER_BATCH 2 SOLVER.CHECKPOINT_PERIOD 100000 \
      TEST.EVAL_PERIOD 0 DATALOADER.NUM_WORKERS 4 2>&1 | tee -a "${LOG}"
    echo "[smoke] exit=${PIPESTATUS[0]} (124=超时属预期)" | tee -a "${LOG}"
    ;;
  train)
    start_gpu_monitor
    "${MDINO_PY}" "${TRAIN_ENTRY_PY}" --config-file "${CFG}" --accum 8 2>&1 | tee -a "${LOG}"
    echo "[train] exit=${PIPESTATUS[0]}" | tee -a "${LOG}"
    ;;
  resume)
    start_gpu_monitor
    "${MDINO_PY}" "${TRAIN_ENTRY_PY}" --config-file "${CFG}" --accum 8 --resume 2>&1 | tee -a "${LOG}"
    echo "[resume] exit=${PIPESTATUS[0]}" | tee -a "${LOG}"
    ;;
  test)
    BEST=$(best_ckpt "${WORKDIR}")
    [ -z "${BEST}" ] && { echo "[test] 未找到 best ckpt" | tee -a "${LOG}"; exit 1; }
    echo "[test] ckpt: ${BEST}" | tee -a "${LOG}"
    "${MDINO_PY}" "${SCRIPTS}/eval_maskdino_citysample.py" "${CFG}" "${BEST}" \
      --out-dir "${WORKDIR}/test_out" \
      --dump-prefix "${WORKDIR}/test_out/pred" 2>&1 | tee -a "${LOG}"
    echo "[test] exit=$?" | tee -a "${LOG}"
    ;;
  *) echo "未知 mode: ${MODE}" | tee -a "${LOG}"; exit 1;;
esac

stop_gpu_monitor
echo "===== done $(date '+%F %T') =====" | tee -a "${LOG}"
