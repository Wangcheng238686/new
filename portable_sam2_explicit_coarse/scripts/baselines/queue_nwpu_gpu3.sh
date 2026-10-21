#!/bin/bash
# =============================================================================
# NWPU VHR-10 对比实验 GPU3 顺序队列（与同卡 WHU 任务并行）
#
# 用法: nohup bash queue_nwpu_gpu3.sh [起始阶段序号] > /tmp/queue_nwpu_gpu3.log 2>&1 &
#   每阶段 = smoke(可选) + train + test；完成后写 .done 标记，重跑自动跳过已完成阶段
#   重显存阶段前会轮询等待 GPU3 空闲显存足够（与同卡 WHU MaskDINO 共存）
#   SCNet 未入队（WHU 上三次失败，NWPU 如需再人工启动）
# =============================================================================
set -u
GPU=3
SCRIPTS="/home/wangcheng/project/new/portable_sam2_explicit_coarse/scripts/baselines"
PROJ="/home/wangcheng/project/new/portable_sam2_explicit_coarse"
MARKERS="${PROJ}/logs/baselines/queue_nwpu_gpu3_markers"
mkdir -p "${MARKERS}"
START_FROM="${1:-1}"

free_mem() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i ${GPU} | tr -d ' '; }
wait_mem() { # $1=需要的MB
  while [ "$(free_mem)" -lt "$1" ]; do
    echo "[queue] $(date '+%F %T') GPU${GPU} free $(free_mem)MB < $1MB, 等待 120s ..."
    sleep 120
  done
}

stage() { # stage <序号> <名字> <需空显存MB> <命令...>
  local idx="$1" name="$2" need_mem="$3"; shift 3
  [ "${idx}" -lt "${START_FROM}" ] && return 0
  if [ -f "${MARKERS}/${name}.done" ]; then echo "[queue] #${idx} ${name} 已完成，跳过"; return 0; fi
  echo "[queue] ===== #${idx} ${name} 开始 $(date '+%F %T') ====="
  wait_mem "${need_mem}"
  "$@"; local rc=$?
  if [ ${rc} -eq 0 ]; then
    touch "${MARKERS}/${name}.done"
    echo "[queue] #${idx} ${name} 成功 (rc=0)"
  else
    echo "[queue] #${idx} ${name} 失败 (rc=${rc})，继续下一阶段（可重入队续跑）"
  fi
  echo "[queue] ===== #${idx} ${name} 结束 $(date '+%F %T') ====="
}

# ---- 轻量批（可与同卡 WHU 任务共存，需 ~10-12GB）----
stage 1  yolo11    10000 bash "${SCRIPTS}/run_yolo11_nwpu.sh" train ${GPU}
stage 2  yolo11test 8000 bash "${SCRIPTS}/run_yolo11_nwpu.sh" test ${GPU}
stage 3  maskrcnn  16000 bash "${SCRIPTS}/run_mmdet_nwpu.sh" maskrcnn smoke ${GPU}
stage 4  maskrcnnT 16000 bash "${SCRIPTS}/run_mmdet_nwpu.sh" maskrcnn train ${GPU}
stage 5  maskrcnnE 8000  bash "${SCRIPTS}/run_mmdet_nwpu.sh" maskrcnn test ${GPU}
stage 6  msrcnn    16000 bash "${SCRIPTS}/run_mmdet_nwpu.sh" msrcnn smoke ${GPU}
stage 7  msrcnnT   16000 bash "${SCRIPTS}/run_mmdet_nwpu.sh" msrcnn train ${GPU}
stage 8  msrcnnE   8000  bash "${SCRIPTS}/run_mmdet_nwpu.sh" msrcnn test ${GPU}
stage 9  htc       18000 bash "${SCRIPTS}/run_mmdet_nwpu.sh" htc smoke ${GPU}
stage 10 htcT      18000 bash "${SCRIPTS}/run_mmdet_nwpu.sh" htc train ${GPU}
stage 11 htcE      8000  bash "${SCRIPTS}/run_mmdet_nwpu.sh" htc test ${GPU}
stage 12 catnet    20000 bash "${SCRIPTS}/run_catnet_nwpu.sh" smoke ${GPU}
stage 13 catnetT   20000 bash "${SCRIPTS}/run_catnet_nwpu.sh" train ${GPU}
stage 14 catnetE   8000  bash "${SCRIPTS}/run_catnet_nwpu.sh" test ${GPU}
stage 15 rtmdet    16000 bash "${SCRIPTS}/run_mmdet_nwpu.sh" rtmdet smoke ${GPU}
stage 16 rtmdetT   16000 bash "${SCRIPTS}/run_mmdet_nwpu.sh" rtmdet train ${GPU}
stage 17 rtmdetE   8000  bash "${SCRIPTS}/run_mmdet_nwpu.sh" rtmdet test ${GPU}

# ---- 中量批（detectron2 系，~18-24GB）----
stage 18 condinst  18000 bash "${SCRIPTS}/run_adelai_nwpu.sh" condinst smoke ${GPU}
stage 19 condinstT 18000 bash "${SCRIPTS}/run_adelai_nwpu.sh" condinst train ${GPU}
stage 20 condinstE 8000  bash "${SCRIPTS}/run_adelai_nwpu.sh" condinst test ${GPU}
stage 21 solov2    18000 bash "${SCRIPTS}/run_adelai_nwpu.sh" solov2 smoke ${GPU}
stage 22 solov2T   18000 bash "${SCRIPTS}/run_adelai_nwpu.sh" solov2 train ${GPU}
stage 23 solov2E   8000  bash "${SCRIPTS}/run_adelai_nwpu.sh" solov2 test ${GPU}

# ---- 重量批（LSJ-1024 / SAM 系，独占倾向，等显存 ≥30GB；此时同卡 WHU 任务应已结束）----
stage 24 mask2former 30000 bash "${SCRIPTS}/run_mmdet_nwpu.sh" mask2former smoke ${GPU}
stage 25 mask2formerT 30000 bash "${SCRIPTS}/run_mmdet_nwpu.sh" mask2former train ${GPU}
stage 26 mask2formerE 8000  bash "${SCRIPTS}/run_mmdet_nwpu.sh" mask2former test ${GPU}
stage 27 maskdino   28000 bash "${SCRIPTS}/run_maskdino_nwpu.sh" smoke ${GPU}
stage 28 maskdinoT  28000 bash "${SCRIPTS}/run_maskdino_nwpu.sh" train ${GPU}
stage 29 maskdinoE  8000  bash "${SCRIPTS}/run_maskdino_nwpu.sh" test ${GPU}

# ---- Ours（单卡 GPU3，SAM2 base_plus ~20GB）----
stage 30 ours      22000 env CUDA_VISIBLE_DEVICES=${GPU} NPROC_PER_NODE=1 bash "${PROJ}/scripts/run_nwpu10_explicit_coarse.sh"
ours_infer() {
  local ckpt_root="/data1/wangcheng/checkpoint/portable_sam2_explicit_coarse/nwpu10"
  local best
  best=$(ls -t "${ckpt_root}"/points_box_dense_p2br0/best*.pth "${ckpt_root}"/points_box_dense_p2br0/*/best*.pth 2>/dev/null | head -1)
  if [ -z "${best}" ]; then echo "[queue][oursE] 未找到 best ckpt，跳过（训练产物命名可能不同，需人工终评）"; return 1; fi
  echo "[queue][oursE] ckpt: ${best}"
  env CUDA_VISIBLE_DEVICES=${GPU} VHR10_DATA_ROOT="/data1/wangcheng/dataset/NWPU VHR-10 dataset" \
    bash "${PROJ}/scripts/infer_nwpu_checkpoint.sh" "${best}"
}
stage 31 oursE     12000 ours_infer

# ---- 长尾批（慢方法放最后）----
stage 32 rs4d      24000 bash "${SCRIPTS}/run_rs4d_nwpu.sh" smoke ${GPU}
stage 33 rs4dT     24000 bash "${SCRIPTS}/run_rs4d_nwpu.sh" train ${GPU}
stage 34 rs4dE     8000  bash "${SCRIPTS}/run_rs4d_nwpu.sh" test ${GPU}
stage 35 rsprompter 24000 bash "${SCRIPTS}/run_rsprompter_nwpu.sh" smoke ${GPU}
stage 36 rsprompterT 24000 bash "${SCRIPTS}/run_rsprompter_nwpu.sh" train ${GPU}
stage 37 rsprompterE 8000  bash "${SCRIPTS}/run_rsprompter_nwpu.sh" test ${GPU}
stage 38 rsiisn    20000 bash "${SCRIPTS}/run_rsiisn_nwpu.sh" smoke ${GPU}
stage 39 rsiisnT   20000 bash "${SCRIPTS}/run_rsiisn_nwpu.sh" train ${GPU}
stage 40 rsiisnE   8000  bash "${SCRIPTS}/run_rsiisn_nwpu.sh" test ${GPU}

echo "[queue] ===== 全部队列处理完毕 $(date '+%F %T') ====="
ls "${MARKERS}" | sort
