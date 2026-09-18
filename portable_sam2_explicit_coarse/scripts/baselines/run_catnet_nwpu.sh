#!/bin/bash
# =============================================================================
# CAT Mask R-CNN (CATNet) @ NWPU VHR-10 实例分割对比实验（PromptMiner-SAM2 协议对齐版）
#
# 用法:
#   bash run_catnet_nwpu.sh [mode] [gpu]
#     mode: smoke  – 冒烟测试(15min 超时, 1 epoch, 不存 ckpt 不验证)
#           train – 完整训练(36ep 3x 配方, val 选 best, bs4x累积2=有效8)
#           resume- 从最新 ckpt 续训
#           test  – 用 best ckpt 在 val json(130, 兼任 test) 上出最终指标
#     gpu : 默认 1
#
# 协议要点(与 WHU-1024 版的差异, 详见 config 头注释):
#   - 数据: NWPU VHR-10 coco_split (10 类, category_id 1-10; train 520 图/3178
#     实例 / val 130 图/743 实例选模; val 兼任 test, 终评在同一 val json)
#   - 图片目录 train/val 共用 'positive image set'; data_root 路径含空格,
#     所有 shell 引用必须加引号
#   - 输入/增强/优化器保持 WHU-1024 版设定: 原生尺寸 Resize(1024,1024)、
#     三向翻转 0.75、SGD lr 0.01, 36ep, milestones[28,34]
#   - 单卡: bs4 x cumulative_iters=2 = 有效 batch 8, lr 不变
#   - checkpoint 输出根: /data1/wangcheng/checkpoint/nwpu_vhr10_baselines/catnet/<runtag>
# =============================================================================
set -uo pipefail

MODE="${1:-smoke}"
GPU="${2:-1}"

PROJ_ROOT="/home/wangcheng/project/new/portable_sam2_explicit_coarse"
REPO="/home/wangcheng/project/CATNet"
CFG="configs/nwpu/cat_mask_rcnn_r50_3x_nwpu.py"
TRAIN_ENTRY="/data2/wangcheng/envs/cvt2/lib/python3.9/site-packages/mmdet/.mim/tools/train.py"
TEST_ENTRY="/data2/wangcheng/envs/cvt2/lib/python3.9/site-packages/mmdet/.mim/tools/test.py"
WORKDIR="/data1/wangcheng/checkpoint/nwpu_vhr10_baselines/catnet/cat_mask_rcnn_r50_3x_nwpu_bs4ai2"
RUNTAG="catnet_nwpu_r50_3x_bs4ai2"
LOGDIR="${PROJ_ROOT}/logs/baselines"
mkdir -p "${LOGDIR}"

TS=$(date +%Y%m%d_%H%M%S)
LOG="${LOGDIR}/${RUNTAG}_${MODE}_${TS}_pid$$.log"
GPULOG="${LOG%.log}.gpu.log"

# torch>=2.6 默认 weights_only=True 会拒载含 mmengine meta 的 ckpt(resume 需要)
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_VISIBLE_DEVICES="${GPU}"

env_note() {
  echo "===== runtag: ${RUNTAG} mode=${MODE} ====="
  echo "date: $(date '+%F %T')  host: $(hostname)  gpu: ${GPU} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
  echo "repo: ${REPO}  head: $(cd ${REPO} && git log --oneline -1 2>/dev/null)"
  echo "entry: mmdet .mim tools (CATNet 为插件仓库, 无自带 train.py)"
  echo "config: ${CFG}"
  echo "work_dir: ${WORKDIR}"
  echo "data_root: /data1/wangcheng/dataset/NWPU VHR-10 dataset (10cls; train 520/3178 / val 130/743 = test)"
  echo "conda env: cvt2  python: $(which python 2>/dev/null)"
}

activate_env() {
  source /home/wangcheng/miniconda3/etc/profile.d/conda.sh
  conda activate cvt2
}

start_gpu_monitor() {
  (
    while true; do
      echo "[$(date '+%F %T')] $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i ${GPU} 2>/dev/null)"
      sleep 60
    done
  ) >> "${GPULOG}" 2>&1 &
  GPUPID=$!
}

stop_gpu_monitor() {
  [ -n "${GPUPID:-}" ] && kill "${GPUPID}" 2>/dev/null
}
trap stop_gpu_monitor EXIT

activate_env
cd "${REPO}"
export PYTHONPATH="${REPO}"

env_note | tee "${LOG}"

case "${MODE}" in
  smoke)
    echo "[smoke] 15min 超时, 1 epoch, 不验证不存 ckpt" | tee -a "${LOG}"
    start_gpu_monitor
    timeout 900 python "${TRAIN_ENTRY}" "${CFG}" \
      --work-dir "${WORKDIR}/smoke" \
      --cfg-options \
        train_cfg.max_epochs=1 \
        train_cfg.val_interval=999 \
        default_hooks.checkpoint.interval=999 \
        default_hooks.logger.interval=5 \
      2>&1 | tee -a "${LOG}"
    rc=${PIPESTATUS[0]}
    echo "[smoke] exit=${rc} (124=超时截断属预期)" | tee -a "${LOG}"
    ;;
  train)
    start_gpu_monitor
    python "${TRAIN_ENTRY}" "${CFG}" --work-dir "${WORKDIR}" 2>&1 | tee -a "${LOG}"
    echo "[train] exit=${PIPESTATUS[0]}" | tee -a "${LOG}"
    ;;
  resume)
    start_gpu_monitor
    python "${TRAIN_ENTRY}" "${CFG}" --work-dir "${WORKDIR}" --resume 2>&1 | tee -a "${LOG}"
    echo "[resume] exit=${PIPESTATUS[0]}" | tee -a "${LOG}"
    ;;
  test)
    BEST=$(ls -t "${WORKDIR}"/best_coco_segm_mAP*.pth 2>/dev/null | head -1)
    if [ -z "${BEST}" ]; then echo "[test] 未找到 best ckpt" | tee -a "${LOG}"; exit 1; fi
    echo "[test] ckpt: ${BEST}" | tee -a "${LOG}"
    echo "[test] NWPU 协议: val(130) 兼任 test, 终评在 NWPU_instances_val.json" | tee -a "${LOG}"
    python "${TEST_ENTRY}" "${CFG}" "${BEST}" --work-dir "${WORKDIR}/test_out" 2>&1 | tee -a "${LOG}"
    echo "[test] exit=${PIPESTATUS[0]}" | tee -a "${LOG}"
    ;;
  *)
    echo "未知 mode: ${MODE} (smoke|train|resume|test)" | tee -a "${LOG}"; exit 1;;
esac

stop_gpu_monitor
echo "===== done $(date '+%F %T') =====" | tee -a "${LOG}"
