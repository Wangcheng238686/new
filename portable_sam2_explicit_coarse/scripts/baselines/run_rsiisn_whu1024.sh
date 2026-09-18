#!/bin/bash
# =============================================================================
# RSIISN (ESWA'23 官方实现, Swin-T Cascade Mask R-CNN) @ WHU-1024
# 环境: mm2 (/data2/wangcheng/envs/mm2: torch1.13.1+cu117 + mmcv-full1.7.2
#        + mmdet2.23.0 + RSIISN 官方补丁文件已替换)
#
# 用法: bash run_rsiisn_whu1024.sh [mode] [gpu]
#   mode: smoke(默认) | train | resume | test
#
# 协议: train 2943 全量 / val 627 每1ep 评 best(segm_mAP) / test 2220 终评
#   maxDets=100 (rcnn max_per_img=100) / 不开 TTA(官方多尺度TTA->单尺度1200,800)
# 超参: 官方配方 AdamW 1.25e-3 12ep 多尺度训练; bs4 x accum2 = 官方总批 8
# =============================================================================
set -uo pipefail

MODE="${1:-smoke}"
GPU="${2:-0}"

PROJ_ROOT="/home/wangcheng/project/new/portable_sam2_explicit_coarse"
REPO="/home/wangcheng/project/RSIISN"
CFG="configs/whu1024/rsiisn_cascade_swinT_whu1024.py"
RUNTAG="rsiisn_cascade_swinT_whu1024_bs2ai4_12ep"
TRAIN_ENTRY="/data2/wangcheng/envs/mm2/lib/python3.9/site-packages/mmdet/.mim/tools/train.py"
TEST_ENTRY="/data2/wangcheng/envs/mm2/lib/python3.9/site-packages/mmdet/.mim/tools/test.py"
LOGDIR="${PROJ_ROOT}/logs/baselines"
WORKDIR="/data1/wangcheng/checkpoint/whu1024_baselines/rsiisn/${RUNTAG}"
mkdir -p "${LOGDIR}"

export CUDA_VISIBLE_DEVICES="${GPU}"

TS=$(date +%Y%m%d_%H%M%S)
LOG="${LOGDIR}/${RUNTAG}_${MODE}_${TS}_pid$$.log"
GPULOG="${LOG%.log}.gpu.log"

activate_env() {
  source /home/wangcheng/miniconda3/etc/profile.d/conda.sh
  conda activate mm2
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
  echo "repo: ${REPO}  config: ${CFG}"
  echo "work_dir: ${WORKDIR}"
  echo "conda env: mm2  python: $(which python)"
  /data2/wangcheng/envs/mm2/bin/python -c "import mmdet, mmcv, torch; print('mmdet', mmdet.__version__, '| mmcv', mmcv.__version__, '| torch', torch.__version__)"
} | tee "${LOG}"

case "${MODE}" in
  smoke)
    echo "[smoke] 10min 超时, 1ep 内截断, 不验证不存 ckpt" | tee -a "${LOG}"
    start_gpu_monitor
    timeout 600 /data2/wangcheng/envs/mm2/bin/python "${TRAIN_ENTRY}" "${CFG}" --work-dir "${WORKDIR}/smoke" \
      --cfg-options runner.max_epochs=1 evaluation.interval=999 \
      checkpoint_config.interval=999 log_config.interval=10 2>&1 | tee -a "${LOG}"
    echo "[smoke] exit=${PIPESTATUS[0]} (124=超时属预期)" | tee -a "${LOG}"
    ;;
  train)
    start_gpu_monitor
    /data2/wangcheng/envs/mm2/bin/python "${TRAIN_ENTRY}" "${CFG}" --work-dir "${WORKDIR}" 2>&1 | tee -a "${LOG}"
    echo "[train] exit=${PIPESTATUS[0]}" | tee -a "${LOG}"
    ;;
  resume)
    LATEST=$(ls -t "${WORKDIR}"/epoch_*.pth 2>/dev/null | grep -v best_ | head -1)
    [ -z "${LATEST}" ] && { echo "[resume] 无 ckpt" | tee -a "${LOG}"; exit 1; }
    echo "[resume] from ${LATEST}" | tee -a "${LOG}"
    start_gpu_monitor
    /data2/wangcheng/envs/mm2/bin/python "${TRAIN_ENTRY}" "${CFG}" --work-dir "${WORKDIR}" --resume-from "${LATEST}" 2>&1 | tee -a "${LOG}"
    echo "[resume] exit=${PIPESTATUS[0]}" | tee -a "${LOG}"
    ;;
  test)
    BEST=$(ls -t "${WORKDIR}"/best_segm_mAP*.pth "${WORKDIR}"/epoch_*.pth 2>/dev/null | head -1)
    [ -z "${BEST}" ] && { echo "[test] 无 ckpt" | tee -a "${LOG}"; exit 1; }
    echo "[test] ckpt: ${BEST}" | tee -a "${LOG}"
    start_gpu_monitor
    /data2/wangcheng/envs/mm2/bin/python "${TEST_ENTRY}" "${CFG}" "${BEST}" \
      --work-dir "${WORKDIR}/test_out" --out "${WORKDIR}/test_out/result.pkl" \
      --eval bbox segm 2>&1 | tee -a "${LOG}"
    # 从 COCO 标准输出解析 12 指标 -> test_metrics.json
    /data2/wangcheng/envs/mm2/bin/python - <<EOF | tee -a "${LOG}"
import json, re
log = open('${LOG}', errors='ignore').read()
# 依序: bbox 6 行, segm 6 行
pat = re.compile(r'maxDets=100 \] = ([0-9.]+)')
vals = pat.findall(log)
vals = [float(v) for v in vals][-12:]
assert len(vals) == 12, f'expect 12 metric lines, got {len(vals)}'
tasks = ['bbox', 'segm']
names = ['mAP', 'mAP50', 'mAP75', 'mAPs', 'mAPm', 'mAPl']
m = {}
for ti, t in enumerate(tasks):
    for ni, n in enumerate(names):
        m[f'{t}_{n}'] = round(vals[ti*6+ni], 4)
out = '${WORKDIR}/test_out/test_metrics.json'
json.dump(m, open(out, 'w'), indent=2)
print('saved', out, m)
EOF
    echo "[test] exit=0" | tee -a "${LOG}"
    ;;
  *) echo "未知 mode: ${MODE}" | tee -a "${LOG}"; exit 1;;
esac

stop_gpu_monitor
echo "===== done $(date '+%F %T') =====" | tee -a "${LOG}"
