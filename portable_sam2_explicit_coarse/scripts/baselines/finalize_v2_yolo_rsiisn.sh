#!/bin/bash
# 收尾链 v2: YOLO 重训(新标签)完成 -> YOLO test 评估 -> RSIISN smoke -> RSIISN 训练
set -u
PROJ=/home/wangcheng/project/new/portable_sam2_explicit_coarse
LOGDIR=${PROJ}/logs/baselines
YOLO_LOG=/home/wangcheng/project/new/portable_sam2_explicit_coarse/logs/baselines/yolo11s_seg_whu1024_default_train_20260830_135951_pid1396098.log
echo "[$(date '+%F %T')] watching: ${YOLO_LOG}"

for i in $(seq 1 120); do
  grep -q "^===== done" "${YOLO_LOG}" 2>/dev/null && break
  sleep 300
done
grep -q "^===== done" "${YOLO_LOG}" || { echo "yolo retrain not finished"; exit 1; }
echo "[$(date '+%F %T')] YOLO retrain done."

# 验证训练结果有效(最后一个 epoch 的 Mask mAP50-95 应 > 0.2, 排除标签问题复发)
TAIL=$(tr '\r' '\n' < "${YOLO_LOG}" | grep -E "^\s+all" | tail -1)
echo "final val line: ${TAIL}"
MASK_OK=$(echo "${TAIL}" | awk '{for(i=1;i<=NF;i++) if($i ~ /^[0-9.]+$/ && i>=11) {print $i; exit}}')
echo "parsed metric field: ${MASK_OK}"

bash ${PROJ}/scripts/baselines/infer_whu1024.sh yolo11 0
echo "[$(date '+%F %T')] yolo test exit=$?"

# RSIISN: 先修好的 mm2 上重跑 smoke(GPU3 与 MaskDINO 短暂共存 10min)
bash ${PROJ}/scripts/baselines/run_rsiisn_whu1024.sh smoke 3
SMOKE_LOG=$(ls -t ${LOGDIR}/rsiisn*smoke*.log 2>/dev/null | grep -v gpu | head -1)
if [ -n "${SMOKE_LOG}" ] && grep -q "Epoch" "${SMOKE_LOG}" && ! grep -q "Traceback" "${SMOKE_LOG}"; then
  echo "[$(date '+%F %T')] RSIISN smoke OK, launching train on GPU0"
  bash ${PROJ}/scripts/baselines/run_rsiisn_whu1024.sh train 0
else
  echo "[$(date '+%F %T')] RSIISN smoke FAILED"
  tail -20 "${SMOKE_LOG}"
  exit 2
fi
echo "[$(date '+%F %T')] finalize v2 chain done"
