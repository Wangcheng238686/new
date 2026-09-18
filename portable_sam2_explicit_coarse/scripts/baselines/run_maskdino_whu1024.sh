#!/bin/bash
# =============================================================================
# MaskDINO R50 @ WHU-1024 统一启动脚本 (官方 IDEA-Research 实现, mdino 环境)
#   mdino = /data2/wangcheng/envs/mdino (torch1.13.1+cu117 + detectron2 v0.6
#           + MSDeformAttn 已编译)
#
# 用法: bash run_maskdino_whu1024.sh [mode] [gpu] [ckpt(test 模式用)]
#   mode: smoke(默认) | train | resume | test
#   gpu : 默认 3
#
# 协议: train 2943 全量不滤空图 / val 627 每10ep 评(日志选best) / test 2220 终评
#       / COCOEvaluator maxDets=[1,10,100] / 不开 TTA
# 超参: 官方 COCO 配方 bs16 AdamW 1e-4 LSJ-1024 AMP grad-clip;
#       单卡 bs4 x accum4 = 有效16; 300 WHU ep(对齐 M2F-v2 小数据集经验),
#       里程碑按 COCO 原比例
# =============================================================================
set -uo pipefail

MODE="${1:-smoke}"
GPU="${2:-3}"
CKPT="${3:-}"

PROJ_ROOT="/home/wangcheng/project/new/portable_sam2_explicit_coarse"
REPO="/home/wangcheng/project/MaskDINO"
CFG="configs/whu/maskdino_R50_whu1024.yaml"
RUNTAG="maskdino_R50_whu1024_bs2ai8_300ep"
LOGDIR="${PROJ_ROOT}/logs/baselines"
mkdir -p "${LOGDIR}"

export CUDA_VISIBLE_DEVICES="${GPU}"

TS=$(date +%Y%m%d_%H%M%S)
LOG="${LOGDIR}/${RUNTAG}_${MODE}_${TS}_pid$$.log"
GPULOG="${LOG%.log}.gpu.log"

WORKDIR=$(cd "${REPO}" && /data2/wangcheng/envs/mdino/bin/python -c "
from detectron2.config import get_cfg
from maskdino import add_maskdino_config
cfg = get_cfg(); add_maskdino_config(cfg); cfg.merge_from_file('${CFG}')
print(cfg.OUTPUT_DIR)" 2>/dev/null | tail -1)
[ -z "${WORKDIR}" ] && { echo "WORKDIR 解析失败"; exit 1; }

activate_env() {
  source /home/wangcheng/miniconda3/etc/profile.d/conda.sh
  conda activate mdino
  # 后台链路中 conda activate 偶发不生效, 统一走绝对路径
  export MDINO_PY=/data2/wangcheng/envs/mdino/bin/python
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

{
  echo "===== runtag: ${RUNTAG} mode=${MODE} ====="
  echo "date: $(date '+%F %T')  host: $(hostname)  gpu: ${GPU}"
  echo "repo: ${REPO}"
  echo "config: ${CFG}  output: ${WORKDIR}"
  echo "conda env: mdino  python: $(which python)"
} | tee "${LOG}"

case "${MODE}" in
  smoke)
    echo "[smoke] 40 iter, 10min 超时, 不验证" | tee -a "${LOG}"
    start_gpu_monitor
    timeout 600 "${MDINO_PY}" train_whu.py --config-file "${CFG}" --accum 8 \
      OUTPUT_DIR "${WORKDIR}/smoke" SOLVER.MAX_ITER 40 \
      TEST.EVAL_PERIOD 0 SOLVER.CHECKPOINT_PERIOD 100000 2>&1 | tee -a "${LOG}"
    echo "[smoke] exit=${PIPESTATUS[0]} (124=超时属预期)" | tee -a "${LOG}"
    ;;
  train)
    start_gpu_monitor
    "${MDINO_PY}" train_whu.py --config-file "${CFG}" --accum 8 2>&1 | tee -a "${LOG}"
    echo "[train] exit=${PIPESTATUS[0]}" | tee -a "${LOG}"
    ;;
  resume)
    start_gpu_monitor
    "${MDINO_PY}" train_whu.py --config-file "${CFG}" --accum 8 --resume 2>&1 | tee -a "${LOG}"
    echo "[resume] exit=${PIPESTATUS[0]}" | tee -a "${LOG}"
    ;;
  test)
    [ -z "${CKPT}" ] && { echo "[test] 用法: run_maskdino_whu1024.sh test <gpu> <ckpt.pth>" | tee -a "${LOG}"; exit 1; }
    echo "[test] ckpt: ${CKPT}" | tee -a "${LOG}"
    DUMPOPT=""
    [ -n "${DUMP_PREFIX:-}" ] && DUMPOPT="--dump-prefix ${DUMP_PREFIX}"
    "${MDINO_PY}" "${PROJ_ROOT}/scripts/baselines/eval_maskdino_whu.py" "${CKPT}" \
      --config "${CFG}" --out-dir "$(dirname "${CKPT}")/test_out" ${DUMPOPT} 2>&1 | tee -a "${LOG}"
    echo "[test] exit=${PIPESTATUS[0]}" | tee -a "${LOG}"
    ;;
  *) echo "未知 mode: ${MODE}" | tee -a "${LOG}"; exit 1;;
esac

stop_gpu_monitor
echo "===== done $(date '+%F %T') =====" | tee -a "${LOG}"
