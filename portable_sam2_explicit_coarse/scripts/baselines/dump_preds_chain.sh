#!/bin/bash
# 4 个基线全量预测落盘(串行 GPU3): maskrcnn htc msrcnn condinst
set -u
PROJ=/home/wangcheng/project/new/portable_sam2_explicit_coarse
CK=/data1/wangcheng/checkpoint/whu1024_baselines
declare -A DUMP=(
  [maskrcnn]=${CK}/maskrcnn/mask-rcnn_r50_fpn_1x_whu1024_bs8ai2/test_out/pred_full
  [htc]=${CK}/htc/htc-without-semantic_r50_fpn_1x_whu1024_bs4ai4/test_out/pred_full
  [msrcnn]=${CK}/msrcnn/ms-rcnn_r50-caffe_fpn_1x_whu1024_bs4ai4/test_out/pred_full
  [condinst]=${CK}/condinst/condinst_MS_R_50_1x_whu1024_bs8lr05m/test_out/pred_full
)
for M in maskrcnn htc msrcnn condinst; do
  echo "[$(date '+%F %T')] === dump ${M} ==="
  DUMP_PREFIX="${DUMP[$M]}" bash ${PROJ}/scripts/baselines/infer_whu1024.sh ${M} 3 0 >/dev/null 2>&1
  f="${DUMP[$M]}.segm.json"
  [ -f "$f" ] && echo "  OK $(du -h "$f" | cut -f1)" || echo "  FAIL"
done
echo "[$(date '+%F %T')] dump chain done"
