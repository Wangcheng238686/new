#!/bin/bash
# =============================================================================
# mmdetection 家族基线 @ WHU-1024 统一启动脚本
# 覆盖: maskrcnn | msrcnn | htc | scnet | mask2former | rtmdet   (环境: cvt2)
#
# 用法: bash run_mmdet_whu1024.sh <model> [mode] [gpu]
#   mode: smoke(默认) | train | resume | test
#   gpu : 默认 0
# 协议: train 2943 全量不滤空图 / val 627 选 best(segm_mAP) / test 2220 终评 / maxDets 100
# 超参: 各官方 COCO 配方(1x 12ep SGD lr0.02@有效16 / mask2former AdamW 1e-4 50ep LSJ-1024)
# =============================================================================
set -uo pipefail

MODEL="${1:?maskrcnn|msrcnn|htc|scnet|mask2former}"
MODE="${2:-smoke}"
GPU="${3:-0}"

PROJ_ROOT="/home/wangcheng/project/new/portable_sam2_explicit_coarse"
REPO="/home/wangcheng/project/mmdetection"
TRAIN_ENTRY="/data2/wangcheng/envs/cvt2/lib/python3.9/site-packages/mmdet/.mim/tools/train.py"
TEST_ENTRY="/data2/wangcheng/envs/cvt2/lib/python3.9/site-packages/mmdet/.mim/tools/test.py"
LOGDIR="${PROJ_ROOT}/logs/baselines"
mkdir -p "${LOGDIR}"
# torch>=2.6 默认 weights_only=True 会拒载含 mmengine meta 的 ckpt(resume 需要)
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_VISIBLE_DEVICES="${GPU}"

case "${MODEL}" in
  maskrcnn)     CFG="configs/whu1024/mask-rcnn_r50_fpn_1x_whu1024.py";      RUNTAG="maskrcnn_r50_fpn_1x_whu1024_bs8ai2";    SMOKE_EP=1; EXTRA_OPTS="";;
  msrcnn)       CFG="configs/whu1024/ms-rcnn_r50-caffe_fpn_1x_whu1024.py";  RUNTAG="msrcnn_r50_caffe_fpn_1x_whu1024_bs4ai4"; SMOKE_EP=1; EXTRA_OPTS="model.backbone.init_cfg.checkpoint=/home/wangcheng/.cache/torch/hub/checkpoints/resnet50_msra-5891d200.pth";;
  htc)          CFG="configs/whu1024/htc-without-semantic_r50_fpn_1x_whu1024.py"; RUNTAG="htc_wosem_r50_fpn_1x_whu1024_bs4ai4"; SMOKE_EP=1; EXTRA_OPTS="";;
  scnet)        CFG="configs/whu1024/scnet_r50_fpn_1x_whu1024.py";          RUNTAG="scnet_r50_fpn_1x_whu1024_bs8ai2";       SMOKE_EP=1; EXTRA_OPTS="";;
  mask2former)  CFG="configs/whu1024/mask2former_r50_whu1024.py";           RUNTAG="mask2former_r50_whu1024_bs4ai4_300ep";        SMOKE_EP=0; EXTRA_OPTS="";;
  rtmdet)       CFG="configs/whu1024/rtmdet-ins_s_300e_whu1024.py";        RUNTAG="rtmdet_ins_s_whu1024_bs4ai64_300ep";         SMOKE_EP=1; EXTRA_OPTS="model.backbone.init_cfg.checkpoint=/home/wangcheng/.cache/torch/hub/checkpoints/cspnext-s_imagenet_600e.pth";;
  *) echo "未知 model: ${MODEL}"; exit 1;;
esac

WORKDIR=$(source ~/miniconda3/etc/profile.d/conda.sh; conda activate cvt2; python -c "
from mmengine.config import Config
import os; os.chdir('${REPO}')
print(Config.fromfile('${CFG}').work_dir)")

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

{
  echo "===== runtag: ${RUNTAG} mode=${MODE} ====="
  echo "date: $(date '+%F %T')  host: $(hostname)  gpu: ${GPU}"
  echo "repo: ${REPO}  head: $(git log --oneline -1 2>/dev/null)"
  echo "config: ${CFG}  work_dir: ${WORKDIR}"
  echo "data_root: /data1/wangcheng/dataset/WHU (train 2943 full / val 627 / test 2220)"
  echo "conda env: cvt2  python: $(which python)"
} | tee "${LOG}"

if [ "${MODEL}" = "mask2former" ]; then
  SMOKE_OPTS="train_cfg.max_iters=50 train_cfg.val_interval=10000 default_hooks.checkpoint.interval=10000 default_hooks.logger.interval=5"
else
  SMOKE_OPTS="train_cfg.max_epochs=1 train_cfg.val_interval=999 default_hooks.checkpoint.interval=999 default_hooks.logger.interval=5"
fi

case "${MODE}" in
  smoke)
    echo "[smoke] 10min 超时, 50 iter / 1 epoch, 不验证不存 ckpt" | tee -a "${LOG}"
    start_gpu_monitor
    timeout 600 python "${TRAIN_ENTRY}" "${CFG}" --work-dir "${WORKDIR}/smoke" \
      --cfg-options ${SMOKE_OPTS} ${EXTRA_OPTS} 2>&1 | tee -a "${LOG}"
    echo "[smoke] exit=${PIPESTATUS[0]} (124=超时属预期)" | tee -a "${LOG}"
    ;;
  train)
    start_gpu_monitor
    python "${TRAIN_ENTRY}" "${CFG}" --work-dir "${WORKDIR}" ${EXTRA_OPTS:+--cfg-options ${EXTRA_OPTS}} 2>&1 | tee -a "${LOG}"
    echo "[train] exit=${PIPESTATUS[0]}" | tee -a "${LOG}"
    ;;
  resume)
    start_gpu_monitor
    python "${TRAIN_ENTRY}" "${CFG}" --work-dir "${WORKDIR}" --resume ${EXTRA_OPTS:+--cfg-options ${EXTRA_OPTS}} 2>&1 | tee -a "${LOG}"
    echo "[resume] exit=${PIPESTATUS[0]}" | tee -a "${LOG}"
    ;;
  test)
    BEST=$(ls -t "${WORKDIR}"/best_coco_segm_mAP*.pth 2>/dev/null | head -1)
    [ -z "${BEST}" ] && { echo "[test] 未找到 best ckpt" | tee -a "${LOG}"; exit 1; }
    echo "[test] ckpt: ${BEST}" | tee -a "${LOG}"
    python "${TEST_ENTRY}" "${CFG}" "${BEST}" --work-dir "${WORKDIR}/test_out" 2>&1 | tee -a "${LOG}"
    echo "[test] exit=${PIPESTATUS[0]}" | tee -a "${LOG}"
    ;;
  *) echo "未知 mode: ${MODE}" | tee -a "${LOG}"; exit 1;;
esac

stop_gpu_monitor
echo "===== done $(date '+%F %T') =====" | tee -a "${LOG}"
