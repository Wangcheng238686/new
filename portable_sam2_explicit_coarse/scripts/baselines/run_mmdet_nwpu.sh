#!/bin/bash
# =============================================================================
# mmdetection 家族基线 @ NWPU VHR-10 统一启动脚本
# 覆盖: maskrcnn | msrcnn | htc | scnet | mask2former | rtmdet   (环境: cvt2)
#
# 用法: bash run_mmdet_nwpu.sh <model> [mode] [gpu]
#   mode: smoke(默认) | train | resume | test
#   gpu : 默认 0
# 协议: train 520 全量 / val 130 选 best(segm_mAP); NWPU 无独立 test 集,
#       test 模式同样在 val json 上终评 (val 兼任 test) / maxDets 100
# 超参: 各官方 COCO 配方(1x 12ep SGD lr0.02@有效16 / mask2former AdamW 1e-4
#       300ep LSJ-1024 / rtmdet 300ep AdamW), 与 WHU-1024 版一致, 仅类别数(10)与数据不同
# 注意: 数据根路径含空格 -> 任何引用必须整体加引号
# =============================================================================
set -uo pipefail

MODEL="${1:?maskrcnn|msrcnn|htc|scnet|mask2former|rtmdet}"
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
  maskrcnn)     CFG="configs/nwpu/mask-rcnn_r50_fpn_1x_nwpu.py";      RUNTAG="maskrcnn_r50_fpn_1x_nwpu_bs8ai2";    SMOKE_EP=1; EXTRA_OPTS="";;
  msrcnn)       CFG="configs/nwpu/ms-rcnn_r50-caffe_fpn_1x_nwpu.py";  RUNTAG="msrcnn_r50_caffe_fpn_1x_nwpu_bs4ai4"; SMOKE_EP=1; EXTRA_OPTS="model.backbone.init_cfg.checkpoint=/home/wangcheng/.cache/torch/hub/checkpoints/resnet50_msra-5891d200.pth";;
  htc)          CFG="configs/nwpu/htc-without-semantic_r50_fpn_1x_nwpu.py"; RUNTAG="htc_wosem_r50_fpn_1x_nwpu_bs4ai4"; SMOKE_EP=1; EXTRA_OPTS="";;
  scnet)        CFG="configs/nwpu/scnet_r50_fpn_1x_nwpu.py";          RUNTAG="scnet_r50_fpn_1x_nwpu_bs8ai2";       SMOKE_EP=1; EXTRA_OPTS="";;
  mask2former)  CFG="configs/nwpu/mask2former_r50_nwpu.py";           RUNTAG="mask2former_r50_nwpu_bs4ai4_300ep";        SMOKE_EP=0; EXTRA_OPTS="";;
  rtmdet)       CFG="configs/nwpu/rtmdet-ins_s_300e_nwpu.py";        RUNTAG="rtmdet_ins_s_nwpu_bs4ai64_300ep";         SMOKE_EP=1; EXTRA_OPTS="model.backbone.init_cfg.checkpoint=/home/wangcheng/.cache/torch/hub/checkpoints/cspnext-s_imagenet_600e.pth";;
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
  echo "data_root: /data1/wangcheng/dataset/NWPU VHR-10 dataset (train 520 full / val 130, val 兼任 test)"
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
    # 优先 best(segm_mAP); mask2former 为 IterBased 无 save_best -> 回退最新 iter_*.pth
    BEST=$(ls -t "${WORKDIR}"/best_coco_segm_mAP*.pth 2>/dev/null | head -1)
    if [ -z "${BEST}" ]; then
      BEST=$(ls -t "${WORKDIR}"/iter_*.pth 2>/dev/null | head -1)
    fi
    [ -z "${BEST}" ] && { echo "[test] 未找到 best/iter ckpt" | tee -a "${LOG}"; exit 1; }
    echo "[test] ckpt: ${BEST}  (NWPU 无独立 test 集, 在 val json 上终评)" | tee -a "${LOG}"
    python "${TEST_ENTRY}" "${CFG}" "${BEST}" --work-dir "${WORKDIR}/test_out" 2>&1 | tee -a "${LOG}"
    echo "[test] exit=${PIPESTATUS[0]}" | tee -a "${LOG}"
    ;;
  *) echo "未知 mode: ${MODE}" | tee -a "${LOG}"; exit 1;;
esac

stop_gpu_monitor
echo "===== done $(date '+%F %T') =====" | tee -a "${LOG}"
