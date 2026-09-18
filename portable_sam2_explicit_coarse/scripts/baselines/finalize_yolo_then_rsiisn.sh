#!/bin/bash
# 收尾链: 等 YOLO 训练完 -> YOLO test 评估 -> RSIISN 正式训练(GPU0)
# 前提: RSIISN smoke 已通过(日志含训练 iter 且无 Traceback)
set -u
PROJ=/home/wangcheng/project/new/portable_sam2_explicit_coarse
LOGDIR=${PROJ}/logs/baselines

YOLO_LOG=${LOGDIR}/yolo11s_seg_whu1024_default_train_20260829_224153_pid1208212.log

echo "[$(date '+%F %T')] waiting for YOLO train to finish..."
for i in $(seq 1 240); do
  grep -q "^===== done" "${YOLO_LOG}" 2>/dev/null && break
  sleep 300
done
grep -q "^===== done" "${YOLO_LOG}" 2>/dev/null || { echo "yolo train not finished in 20h, giving up"; exit 1; }
echo "[$(date '+%F %T')] YOLO train done."

# YOLO test 评估(GPU0)
bash ${PROJ}/scripts/baselines/infer_whu1024.sh yolo11 0
echo "[$(date '+%F %T')] yolo test exit=$?"

# RSIISN smoke 是否通过
SMOKE_LOG=$(ls -t ${LOGDIR}/rsiisn*smoke*.log 2>/dev/null | grep -v gpu | head -1)
if [ -n "${SMOKE_LOG}" ] && grep -q "Epoch" "${SMOKE_LOG}" && ! grep -q "Traceback" "${SMOKE_LOG}"; then
  echo "[$(date '+%F %T')] RSIISN smoke OK (${SMOKE_LOG}), launching train on GPU0"
  bash ${PROJ}/scripts/baselines/run_rsiisn_whu1024.sh train 0
else
  echo "[$(date '+%F %T')] RSIISN smoke 未通过或未运行, 不自动启动训练"
  exit 2
fi
echo "[$(date '+%F %T')] finalize chain done"
