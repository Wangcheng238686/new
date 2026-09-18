#!/bin/bash
# =============================================================================
# NWPU VHR-10 GPU3 队列 v2（修正版）
#   修正点：
#   1) 成功判定不再信任 run 脚本返回码（其训练失败仍 rc=0），改为校验实际产物：
#      T 阶段 = workdir 出现 best/model_final/iter ckpt；E 阶段 = 最新 test 日志含 exit=0
#   2) 移除 run 脚本对全卡独占的假设：重显存阶段前轮询等待空闲显存
#   3) condinst/solov2 已修 dt2 detectron2 np.bool 崩溃（/tmp/detectron2-0.6 已 patch）
#   已由 v1 真实完成（勿重复）：yolo11, maskrcnn, msrcnn, htc, rtmdet（train+test 均已核验产物）
#   Ours 由独立进程在 GPU3 训练中（本队列只负责其终评）
# 用法: nohup bash queue_nwpu_gpu3_v2.sh [起始阶段] > /tmp/queue_nwpu_gpu3_v2.log 2>&1 &
# =============================================================================
set -u
GPU=3
SCRIPTS="/home/wangcheng/project/new/portable_sam2_explicit_coarse/scripts/baselines"
LOGDIR="/home/wangcheng/project/new/portable_sam2_explicit_coarse/logs/baselines"
CKPT="/data1/wangcheng/checkpoint/nwpu_vhr10_baselines"
MARKERS="${LOGDIR}/queue_nwpu_gpu3_markers"
mkdir -p "${MARKERS}"
START_FROM="${1:-1}"

free_mem() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i ${GPU} | tr -d ' '; }

# 上一队列实例遗留的 maskdino 训练进程(3086905)仍在跑：等它结束后按产物补标记，避免重复训练
if ps -p 3086905 > /dev/null 2>&1; then
  echo "[queue] 检测到遗留 maskdino 训练(3086905)，等待其结束 ..."
  while ps -p 3086905 > /dev/null 2>&1; do sleep 300; done
fi
if compgen -G "${CKPT}/maskdino/*/model_final.pth" > /dev/null; then
  touch "${MARKERS}/maskdinoT.done"; echo "[queue] maskdinoT 产物已就绪，补标记"
fi
wait_mem() {
  while [ "$(free_mem)" -lt "$1" ]; do
    echo "[queue] $(date '+%F %T') GPU${GPU} free $(free_mem)MB < $1MB, 等待 300s ..."
    sleep 300
  done
}
# 产物存在性校验
ok_ckpt() { # ok_ckpt <glob...>  任一 glob 非空即真
  local g; for g in "$@"; do compgen -G "$g" > /dev/null && return 0; done; return 1
}
ok_test() { # ok_test <日志名前缀>  最新匹配 *_test_* 日志含 exit=0 即真
  local f; f=$(ls -t "${LOGDIR}"/${1}*_test_* 2>/dev/null | head -1)
  [ -n "$f" ] && grep -q "exit=0" "$f"
}

stage() { # stage <序号> <名> <需空MB> <成功校验命令...> -- <执行命令...>
  local idx="$1" name="$2" need="$3"; shift 3
  local check=() cmd=() seen=0
  for a in "$@"; do
    if [ "$a" = "--" ]; then seen=1; continue; fi
    if [ $seen -eq 0 ]; then check+=("$a"); else cmd+=("$a"); fi
  done
  [ "${idx}" -lt "${START_FROM}" ] && return 0
  if [ -f "${MARKERS}/${name}.done" ]; then echo "[queue] #${idx} ${name} 已完成，跳过"; return 0; fi
  echo "[queue] ===== #${idx} ${name} 开始 $(date '+%F %T') ====="
  wait_mem "${need}"
  "${cmd[@]}"; local rc=$?
  if [ ${rc} -eq 0 ] && "${check[@]}"; then
    touch "${MARKERS}/${name}.done"; echo "[queue] #${idx} ${name} 成功(产物校验通过)"
  else
    echo "[queue] #${idx} ${name} 失败(rc=${rc} 或产物缺失)，继续下一阶段"
  fi
  echo "[queue] ===== #${idx} ${name} 结束 $(date '+%F %T') ====="
}

# ---- detectron2 批（np.bool 已修；实测 ~12-16GB，可与 GPU3 上 Ours/WHU 任务共存）----
stage 1 condinstT 15000 ok_ckpt "${CKPT}/condinst/condinst_MS_R_50_1x_nwpu_bs8lr05m/model_final.pth" -- \
  bash "${SCRIPTS}/run_adelai_nwpu.sh" condinst train ${GPU}
stage 2 condinstE 8000 ok_test "condinst_MS_R_50_1x_nwpu" -- \
  bash "${SCRIPTS}/run_adelai_nwpu.sh" condinst test ${GPU}
stage 3 solov2T 15000 ok_ckpt "${CKPT}/solov2/solov2_R50_1x_nwpu_bs8lr05m/model_final.pth" -- \
  bash "${SCRIPTS}/run_adelai_nwpu.sh" solov2 train ${GPU}
stage 4 solov2E 8000 ok_test "solov2_R50_1x_nwpu" -- \
  bash "${SCRIPTS}/run_adelai_nwpu.sh" solov2 test ${GPU}

# ---- MaskDINO（mdino 环境；~28GB，当前空闲 29GB 可跑）----
stage 9 maskdinoT 28000 ok_ckpt "${CKPT}/maskdino/*/model_final.pth" "${CKPT}/maskdino/*/*.pth" -- \
  bash "${SCRIPTS}/run_maskdino_nwpu.sh" train ${GPU}
stage 10 maskdinoE 8000 ok_test "maskdino_R50_nwpu" -- \
  bash "${SCRIPTS}/run_maskdino_nwpu.sh" test ${GPU}

# ---- 长尾批（26GB 以下，可与 GPU3 上 WHU 任务共存）----
stage 12 rs4dT 26000 ok_ckpt "${CKPT}/rs4d/*/best_*.pth" "${CKPT}/rs4d/*/epoch_*.pth" -- \
  bash "${SCRIPTS}/run_rs4d_nwpu.sh" train ${GPU}
stage 13 rs4dE 8000 ok_test "rs4d" -- \
  bash "${SCRIPTS}/run_rs4d_nwpu.sh" test ${GPU}
stage 14 rsprompterT 26000 ok_ckpt "${CKPT}/rsprompter/*/best_*.pth" "${CKPT}/rsprompter/*/iter_*.pth" -- \
  bash "${SCRIPTS}/run_rsprompter_nwpu.sh" train ${GPU}
stage 15 rsprompterE 8000 ok_test "rsprompter_anchor_nwpu" -- \
  bash "${SCRIPTS}/run_rsprompter_nwpu.sh" test ${GPU}
stage 16 rsiisnT 21000 ok_ckpt "${CKPT}/rsiisn/*/best_coco_segm_mAP*.pth" -- \
  bash "${SCRIPTS}/run_rsiisn_nwpu.sh" train ${GPU}
stage 17 rsiisnE 8000 ok_test "rsiisn_cascade_swinT_nwpu" -- \
  bash "${SCRIPTS}/run_rsiisn_nwpu.sh" test ${GPU}

# ---- CATNet 与 Mask2Former 已移至 GPU2 独立队列（queue_nwpu_gpu2.sh），本卡不再排 ----
# 若 GPU2 队列已完成并写标记，此处跳过逻辑仍生效（共享同一 MARKERS 目录）

echo "[queue] ===== v2 队列处理完毕 $(date '+%F %T') ====="
ls "${MARKERS}" | sort
