#!/usr/bin/env bash
# 多卡分片推理编排：对自描述 checkpoint 在 N 张卡上并行推一个 split，
# 各片写 shardK_records.pkl（round-robin 不相交），最后 CPU 合并并走
# infer_from_checkpoint.py 原生评估块（与单进程同一代码路径）。
#
# 用法: infer_test_sharded.sh <checkpoint.pth> <split(test|validation|custom)> \
#                             [ngpu=4] [透传给 infer_from_checkpoint.py 的额外参数...]
# 例:   infer_test_sharded.sh .../best_model.pth test 4 --batch-size 4
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAINLINE="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${MAINLINE}/scripts/load_environment.sh"

CKPT="$1"; SPLIT="$2"; NGPU="${3:-4}"; shift 3 || true

OUT_DIR="$(dirname "${CKPT}")/inference_${SPLIT}"
mkdir -p "${OUT_DIR}"

PIDS=()
for k in $(seq 0 $((NGPU - 1))); do
  CUDA_VISIBLE_DEVICES="${k}" setsid nohup "${PYTHON}" \
    inference/infer_from_checkpoint.py \
    --checkpoint "${CKPT}" \
    --split "${SPLIT}" \
    --shard-index "${k}" --shard-count "${NGPU}" \
    --device cuda:0 \
    --output-dir "${OUT_DIR}" \
    "$@" \
    > "${OUT_DIR}/shard${k}.log" 2>&1 < /dev/null &
  PIDS+=($!)
  echo "shard ${k}/${NGPU} pid ${PIDS[-1]} (GPU ${k})"
done

FAIL=0
DEADLINE=$(( $(date +%s) + 7200 ))
while :; do
  DONE=0
  for k in $(seq 0 $((NGPU - 1))); do
    [ -f "${OUT_DIR}/shard${k}_manifest.json" ] && DONE=$((DONE + 1))
  done
  [ "${DONE}" -eq "${NGPU}" ] && break
  ALIVE=0
  for pid in "${PIDS[@]}"; do kill -0 "${pid}" 2>/dev/null && ALIVE=$((ALIVE + 1)); done
  if [ "${ALIVE}" -eq 0 ] && [ "${DONE}" -lt "${NGPU}" ]; then
    echo "全部分片进程退出但仍缺 manifest（DONE=${DONE}/${NGPU}）" >&2; FAIL=1; break
  fi
  if [ "$(date +%s)" -gt "${DEADLINE}" ]; then
    echo "分片等待超时 2h（DONE=${DONE}/${NGPU}）" >&2; FAIL=1; break
  fi
  sleep 30
done
[ "${FAIL}" -eq 0 ] || { echo "有分片未完成，不进入合并（可重跑单分片）" >&2; exit 2; }

echo "全部 ${NGPU} 分片完成，CPU 合并评估..."
"${PYTHON}" inference/infer_from_checkpoint.py \
  --checkpoint "${CKPT}" \
  --split "${SPLIT}" \
  --merge-shards "${NGPU}" \
  --device cpu \
  --output-dir "${OUT_DIR}" \
  "$@" \
  2>&1 | tee "${OUT_DIR}/merge.log" | grep -viE "font|matplotlib"
