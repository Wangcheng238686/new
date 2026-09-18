#!/bin/bash
# =============================================================================
# AdelaiDet 基线 @ WHU-1024 统一启动脚本 (环境: dt2)
# 覆盖: condinst | solov2   (官方 1x: 90k iter, steps(60k,80k), bs8 lr0.005 线性折算)
#
# 用法: bash run_adelai_whu1024.sh <condinst|solov2> [mode] [gpu]
#   mode: smoke(默认) | train | resume | test
#   gpu : 默认 1
# 协议: train 2943 全量(filter_images=False) / val 627 选模 / test 2220 终评
# =============================================================================
set -uo pipefail

MODEL="${1:?condinst|solov2}"
MODE="${2:-smoke}"
GPU="${3:-1}"

PROJ_ROOT="/home/wangcheng/project/new/portable_sam2_explicit_coarse"
REPO="/home/wangcheng/project/AdelaiDet"
LOGDIR="${PROJ_ROOT}/logs/baselines"
mkdir -p "${LOGDIR}"
# torch>=2.6 默认 weights_only=True 会拒载含 mmengine meta 的 ckpt(resume 需要)
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_VISIBLE_DEVICES="${GPU}"

case "${MODEL}" in
  condinst)
    CFG="configs/WHU/condinst_MS_R_50_1x_whu1024.yaml"
    RUNTAG="condinst_MS_R50_1x_whu1024_bs8lr05m";;
  solov2)
    CFG="configs/WHU/solov2_R50_1x_whu1024.yaml"
    RUNTAG="solov2_R50_1x_whu1024_bs8lr05m";;
  *) echo "未知 model: ${MODEL}"; exit 1;;
esac

WORKDIR=$(python3 -c "
import yaml,sys
d=yaml.safe_load(open('${REPO}/${CFG}').read().replace('_BASE_','_BASE_X'))
print(d.get('OUTPUT_DIR',''))" 2>/dev/null || echo "")
[ -z "${WORKDIR}" ] && WORKDIR="/data1/wangcheng/checkpoint/whu1024_baselines/${MODEL}"

TS=$(date +%Y%m%d_%H%M%S)
LOG="${LOGDIR}/${RUNTAG}_${MODE}_${TS}_pid$$.log"
GPULOG="${LOG%.log}.gpu.log"

activate_env() {
  source /home/wangcheng/miniconda3/etc/profile.d/conda.sh
  conda activate /data2/wangcheng/envs/dt2
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

best_ckpt() {  # 从 metrics.json 选 val segm AP 最高的 ckpt
  python3 - "$1" << 'PYEOF'
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

{
  echo "===== runtag: ${RUNTAG} mode=${MODE} ====="
  echo "date: $(date '+%F %T')  host: $(hostname)  gpu: ${GPU}"
  echo "repo: ${REPO}  head: $(git log --oneline -1 2>/dev/null) (+local patch: adet/data/whu_builtin.py)"
  echo "config: ${CFG}  work_dir: ${WORKDIR}"
  echo "data_root: /data1/wangcheng/dataset/WHU (train 2943 full / val 627 / test 2220)"
  echo "conda env: dt2  python: $(which python)"
} | tee "${LOG}"

case "${MODE}" in
  smoke)
    echo "[smoke] 10min 超时, 100 iter, 不验证不存 ckpt" | tee -a "${LOG}"
    start_gpu_monitor
    timeout 600 python tools/train_net.py --config-file "${CFG}" --num-gpus 1 \
      OUTPUT_DIR "${WORKDIR}/smoke" \
      SOLVER.MAX_ITER 100 SOLVER.IMS_PER_BATCH 2 SOLVER.CHECKPOINT_PERIOD 100000 \
      TEST.EVAL_PERIOD 0 DATALOADER.NUM_WORKERS 4 2>&1 | tee -a "${LOG}"
    echo "[smoke] exit=${PIPESTATUS[0]} (124=超时属预期)" | tee -a "${LOG}"
    ;;
  train)
    start_gpu_monitor
    python tools/train_net.py --config-file "${CFG}" --num-gpus 1 \
      OUTPUT_DIR "${WORKDIR}" 2>&1 | tee -a "${LOG}"
    echo "[train] exit=${PIPESTATUS[0]}" | tee -a "${LOG}"
    ;;
  resume)
    start_gpu_monitor
    RESUME_CKPT=$(ls -t "${WORKDIR}"/model_*.pth 2>/dev/null | head -1)
    echo "[resume] from ${RESUME_CKPT}" | tee -a "${LOG}"
    # 本版 detectron2 的 --resume 是 store_true flag(非 nargs), 置于 opts 前自动续 last ckpt
    python tools/train_net.py --config-file "${CFG}" --num-gpus 1 --resume \
      OUTPUT_DIR "${WORKDIR}" 2>&1 | tee -a "${LOG}"
    echo "[resume] exit=${PIPESTATUS[0]}" | tee -a "${LOG}"
    ;;
  test)
    BEST=$(best_ckpt "${WORKDIR}")
    [ -z "${BEST}" ] && { echo "[test] 未找到 best ckpt(metrics.json 解析失败)" | tee -a "${LOG}"; exit 1; }
    echo "[test] ckpt: ${BEST}" | tee -a "${LOG}"
    python tools/train_net.py --config-file "${CFG}" --num-gpus 1 --eval-only \
      OUTPUT_DIR "${WORKDIR}/test_out" \
      MODEL.WEIGHTS "${BEST}" \
      DATASETS.TEST '("whu1024_test",)' 2>&1 | tee -a "${LOG}"
    echo "[test] exit=${PIPESTATUS[0]}" | tee -a "${LOG}"
    ;;
  *) echo "未知 mode: ${MODE}" | tee -a "${LOG}"; exit 1;;
esac

stop_gpu_monitor
echo "===== done $(date '+%F %T') =====" | tee -a "${LOG}"
