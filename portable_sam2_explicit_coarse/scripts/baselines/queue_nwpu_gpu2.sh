#!/bin/bash
# =============================================================================
# NWPU VHR-10 GPU2 队列：CATNet + Mask2Former（大显存方法，从 GPU3 队列移出）
# GPU2 当前整卡空闲(49GB)；与 GPU3 队列共享完成标记目录（MARKERS），互不重复
# 用法: nohup bash queue_nwpu_gpu2.sh > /tmp/queue_nwpu_gpu2.log 2>&1 &
# =============================================================================
set -u
GPU=2
SCRIPTS="/home/wangcheng/project/new/portable_sam2_explicit_coarse/scripts/baselines"
LOGDIR="/home/wangcheng/project/new/portable_sam2_explicit_coarse/logs/baselines"
CKPT="/data1/wangcheng/checkpoint/nwpu_vhr10_baselines"
MARKERS="${LOGDIR}/queue_nwpu_gpu3_markers"
mkdir -p "${MARKERS}"

free_mem() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i ${GPU} | tr -d ' '; }
wait_mem() {
  while [ "$(free_mem)" -lt "$1" ]; do
    echo "[queue2] $(date '+%F %T') GPU${GPU} free $(free_mem)MB < $1MB, 等待 300s ..."
    sleep 300
  done
}
ok_ckpt() { local g; for g in "$@"; do compgen -G "$g" > /dev/null && return 0; done; return 1; }
ok_test() {
  local f; f=$(ls -t "${LOGDIR}"/${1}*_test_* 2>/dev/null | head -1)
  [ -n "$f" ] && grep -q "exit=0" "$f"
}
stage() { # stage <名> <需空MB> <成功校验命令...> -- <执行命令...>
  local name="$1" need="$2"; shift 2
  local check=() cmd=() seen=0
  for a in "$@"; do
    if [ "$a" = "--" ]; then seen=1; continue; fi
    if [ $seen -eq 0 ]; then check+=("$a"); else cmd+=("$a"); fi
  done
  if [ -f "${MARKERS}/${name}.done" ]; then echo "[queue2] ${name} 已完成，跳过"; return 0; fi
  echo "[queue2] ===== ${name} 开始 $(date '+%F %T') ====="
  wait_mem "${need}"
  "${cmd[@]}"; local rc=$?
  if [ ${rc} -eq 0 ] && "${check[@]}"; then
    touch "${MARKERS}/${name}.done"; echo "[queue2] ${name} 成功(产物校验通过)"
  else
    echo "[queue2] ${name} 失败(rc=${rc} 或产物缺失)"
  fi
  echo "[queue2] ===== ${name} 结束 $(date '+%F %T') ====="
}

# ---- CATNet（v1 在 GPU3 因共享显存 OOM；GPU2 整卡独占，峰值 ~29GB 充裕）----
stage catnetT 31000 ok_ckpt "${CKPT}/catnet/cat_mask_rcnn_r50_3x_nwpu_bs4ai2/best_coco_segm_mAP*.pth" "${CKPT}/catnet/cat_mask_rcnn_r50_3x_nwpu_bs4ai2/epoch_*.pth" -- \
  bash "${SCRIPTS}/run_catnet_nwpu.sh" train ${GPU}
stage catnetE 8000 ok_test "catnet_nwpu_r50_3x" -- \
  bash "${SCRIPTS}/run_catnet_nwpu.sh" test ${GPU}

# ---- Mask2Former（IterBased；best 无 save_best，用 iter_*.pth 兜底校验）----
stage m2fT 32000 ok_ckpt "${CKPT}/mask2former/*/iter_*.pth" "${CKPT}/mask2former/*/best_*.pth" -- \
  bash "${SCRIPTS}/run_mmdet_nwpu.sh" mask2former train ${GPU}
stage m2fE 8000 ok_test "mask2former_r50_nwpu" -- \
  bash "${SCRIPTS}/run_mmdet_nwpu.sh" mask2former test ${GPU}

echo "[queue2] ===== GPU2 队列处理完毕 $(date '+%F %T') ====="
