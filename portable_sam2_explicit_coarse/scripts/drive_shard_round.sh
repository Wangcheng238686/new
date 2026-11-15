#!/usr/bin/env bash
# 分片推理前台轮次驱动（kill-resume 安全）：
#   每轮对每个分片 k：统计已有连续 chunk 数 C_k，若未达总批数则以
#   --skip-batches C_k --max-batches <window> 推一个窗口（4 卡并行，wait 收口）。
#   前台运行（工具调用存活期间=已证安全模式）；被杀只损失在途一批，重跑即续。
# 用法: drive_shard_round.sh <ckpt> <split> <ngpu> <window_batches> [total_batches] [extra...]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAINLINE="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${MAINLINE}/scripts/load_environment.sh"

CKPT="$1"; SPLIT="$2"; NGPU="$3"; WINDOW="$4"; TOTAL="$5"; shift 5
OUT_DIR="$(dirname "${CKPT}")/inference_${SPLIT}"
mkdir -p "${OUT_DIR}"

contig() {  # 连续 chunk 计数: shardK_chunk0000.. 无空洞的前缀长度
  local k=$1 c=0
  while [ -f "${OUT_DIR}/shard${k}_chunk$(printf %04d "$c").pkl" ]; do c=$((c + 1)); done
  echo "$c"
}

PIDS=(); LAUNCHED=0
for k in $(seq 0 $((NGPU - 1))); do
  C=$(contig "$k")
  if [ "${TOTAL}" -gt 0 ] && [ "${C}" -ge "${TOTAL}" ]; then
    echo "shard${k}: ${C}/${TOTAL} 已完成，跳过"
    continue
  fi
  LAUNCHED=$((LAUNCHED + 1))
  echo "shard${k}: 从 batch ${C} 起推窗口 ${WINDOW}（GPU ${k}）"
  CUDA_VISIBLE_DEVICES="${k}" "${PYTHON}" inference/infer_from_checkpoint.py \
    --checkpoint "${CKPT}" --split "${SPLIT}" \
    --shard-index "${k}" --shard-count "${NGPU}" \
    --skip-batches "${C}" --max-batches "${WINDOW}" \
    --device cuda:0 --output-dir "${OUT_DIR}" \
    "$@" >> "${OUT_DIR}/shard${k}.log" 2>&1 &
  PIDS+=($!)
done

[ "${LAUNCHED}" -gt 0 ] || { echo "全部分片已达目标批数"; exit 0; }
FAIL=0
for i in "${!PIDS[@]}"; do
  wait "${PIDS[$i]}" || { echo "shard${i} 本轮异常退出（续跑即恢复）" >&2; FAIL=1; }
done

for k in $(seq 0 $((NGPU - 1))); do
  echo "shard${k}: $(contig "$k") chunks"
done
[ "${FAIL}" -eq 0 ] || exit 3
exit 0
