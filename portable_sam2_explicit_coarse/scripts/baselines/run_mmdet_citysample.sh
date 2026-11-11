#!/bin/bash
# =============================================================================
# mmdetection 家族基线 @ CitySample-Sat v1 统一启动脚本
# 覆盖: *_v1d(默认增强 512 锚定, 当前主协议) | *_v1m(协议B minimal, 已归档)
#       环境: cvt2
#
# 用法: bash run_mmdet_citysample.sh <model> [mode] [gpu]
#   mode: smoke(默认) | train | resume | test
#   gpu : 默认 0
# 协议: train 2142 全量不滤空图 / val 459 选 best(segm_mAP) / test 459 终评
#       maxDets=250 (P1/P2: 全 split 单图实例最大 238) / 512 原生不上采样
# test: citysample_sat_test.py (pycocotools maxDets=[1,10,250] + BoundaryAP)
# 注: v3_noaug 旧入口已删(2026-11-11, v3 战役作废; git 历史可溯)
# =============================================================================
set -uo pipefail

MODEL="${1:?maskrcnn|msrcnn|rtmdet}"
MODE="${2:-smoke}"
GPU="${3:-0}"

PROJ_ROOT="/home/wangcheng/project/new/portable_sam2_explicit_coarse"
SCRIPTS="${PROJ_ROOT}/scripts/baselines"
REPO="/home/wangcheng/project/mmdetection"
TRAIN_ENTRY="/data2/wangcheng/envs/cvt2/lib/python3.9/site-packages/mmdet/.mim/tools/train.py"
LOGDIR="${PROJ_ROOT}/logs/baselines_citysample"
mkdir -p "${LOGDIR}"
# torch>=2.6 默认 weights_only=True 会拒载含 mmengine meta 的 ckpt(resume 需要)
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_VISIBLE_DEVICES="${GPU}"

case "${MODEL}" in
  # ---- v1 默认增强(P-R=B 512 锚定, 2026-11-02 立项 / 11-11 复活为主协议) ----
  maskrcnn_v1d)  CFG="configs/citysample/mask-rcnn_r50_fpn_1x_citysample_v1_512defaug.py";  RUNTAG="maskrcnn_r50_fpn_1x_citysample_v1_512defaug_bs8ai2"; EXTRA_OPTS="";;
  msrcnn_v1d)    CFG="configs/citysample/ms-rcnn_r50-caffe_fpn_1x_citysample_v1_512defaug.py"; RUNTAG="msrcnn_r50_caffe_fpn_1x_citysample_v1_512defaug_bs4ai4"; EXTRA_OPTS="model.backbone.init_cfg.checkpoint=/home/wangcheng/.cache/torch/hub/checkpoints/resnet50_msra-5891d200.pth";;
  htc_v1d)       CFG="configs/citysample/htc-wosem_r50_fpn_1x_citysample_v1_512defaug.py"; RUNTAG="htc_wosem_r50_fpn_1x_citysample_v1_512defaug_bs4ai4"; EXTRA_OPTS="";;
  rtmdet_v1d)    CFG="configs/citysample/rtmdet-ins_s_300e_citysample_v1_512defaug.py"; RUNTAG="rtmdet_ins_s_citysample_v1_512defaug_bs4ai64_300ep"; EXTRA_OPTS="model.backbone.init_cfg.checkpoint=/home/wangcheng/.cache/torch/hub/checkpoints/cspnext-s_imagenet_600e.pth";;
  mask2former_v1d) CFG="configs/citysample/mask2former_r50_citysample_v1_512defaug.py"; RUNTAG="mask2former_r50_citysample_v1_512defaug_bs4ai4_300ep"; EXTRA_OPTS="";;
  # ---- 协议B(匹配增强合同 minimal, 2026-11-10): cs_minimal_aug 需 PYTHONPATH=仓根 ----
  maskrcnn_v1m)  CFG="configs/citysample/mask-rcnn_r50_fpn_1x_citysample_v1_minimal.py";  RUNTAG="maskrcnn_r50_fpn_1x_citysample_v1_minimal_bs8ai2"; EXTRA_OPTS="";;
  msrcnn_v1m)    CFG="configs/citysample/ms-rcnn_r50-caffe_fpn_1x_citysample_v1_minimal.py"; RUNTAG="msrcnn_r50_caffe_fpn_1x_citysample_v1_minimal_bs4ai4"; EXTRA_OPTS="model.backbone.init_cfg.checkpoint=/home/wangcheng/.cache/torch/hub/checkpoints/resnet50_msra-5891d200.pth";;
  htc_v1m)       CFG="configs/citysample/htc-wosem_r50_fpn_1x_citysample_v1_minimal.py"; RUNTAG="htc_wosem_r50_fpn_1x_citysample_v1_minimal_bs4ai4"; EXTRA_OPTS="";;
  rtmdet_v1m)    CFG="configs/citysample/rtmdet-ins_s_300e_citysample_v1_minimal.py"; RUNTAG="rtmdet_ins_s_citysample_v1_minimal_bs4ai64_300ep"; EXTRA_OPTS="model.backbone.init_cfg.checkpoint=/home/wangcheng/.cache/torch/hub/checkpoints/cspnext-s_imagenet_600e.pth";;
  mask2former_v1m) CFG="configs/citysample/mask2former_r50_citysample_v1_minimal.py"; RUNTAG="mask2former_r50_citysample_v1_minimal_bs4ai4_300ep"; EXTRA_OPTS="";;
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
export PYTHONPATH="${REPO}"  # cs_minimal_aug(custom_imports)经仓根导入

{
  echo "===== runtag: ${RUNTAG} mode=${MODE} ====="
  echo "date: $(date '+%F %T')  host: $(hostname)  gpu: ${GPU}"
  echo "repo: ${REPO}  head: $(git log --oneline -1 2>/dev/null)"
  echo "config: ${CFG}  work_dir: ${WORKDIR}"
  echo "data: /data1/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021 (v1, train 2142 / val 459 / test 459, ${MODEL##*_})"
  echo "conda env: cvt2  python: $(which python)"
} | tee "${LOG}"

case "${MODEL}" in mask2former|mask2former_v1d|mask2former_v1m) M2F=1;; *) M2F=0;; esac
if [ "${M2F}" = "1" ]; then
  # M2F 为 IterBasedTrainLoop, 冒烟用 iter 计数
  SMOKE_OPTS="train_cfg.max_iters=60 train_cfg.val_interval=999999 default_hooks.checkpoint.interval=999999 default_hooks.logger.interval=5"
else
  SMOKE_OPTS="train_cfg.max_epochs=1 train_cfg.val_interval=999 default_hooks.checkpoint.interval=999 default_hooks.logger.interval=5"
fi

case "${MODE}" in
  smoke)
    echo "[smoke] 10min 超时, 1 epoch, 不验证不存 ckpt" | tee -a "${LOG}"
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
    python "${SCRIPTS}/citysample_sat_test.py" "${CFG}" "${BEST}" \
      --out-dir "${WORKDIR}/test_out" \
      --gt-json "/data1/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021/annotations/test.json" \
      --dump-prefix "${WORKDIR}/test_out/pred" 2>&1 | tee -a "${LOG}"
    echo "[test] exit=$?" | tee -a "${LOG}"
    ;;
  *) echo "未知 mode: ${MODE}" | tee -a "${LOG}"; exit 1;;
esac

stop_gpu_monitor
echo "===== done $(date '+%F %T') =====" | tee -a "${LOG}"
