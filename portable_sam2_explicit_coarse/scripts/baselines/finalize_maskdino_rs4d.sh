#!/bin/bash
# 收尾链: MaskDINO + RS4D test(带 dump) -> boundary 双口径 -> 产出报告
set -u
PROJ=/home/wangcheng/project/new/portable_sam2_explicit_coarse
CKPT_BASE=/data1/wangcheng/checkpoint/whu1024_baselines
PY=/data2/wangcheng/envs/cvt2/bin/python
cd ${PROJ}/scripts/baselines

echo "[$(date '+%F %T')] MaskDINO test (GPU3, with dump)"
DUMP_PREFIX=${CKPT_BASE}/maskdino/maskdino_R50_whu1024_bs2ai8_300ep/test_out/pred \
  bash ${PROJ}/scripts/baselines/run_maskdino_whu1024.sh test 3 \
  ${CKPT_BASE}/maskdino/maskdino_R50_whu1024_bs2ai8_300ep/model_final.pth \
  || { echo "maskdino test FAIL"; exit 1; }
# dump 落盘文件重命名对齐 pred_coco.json 约定(直接用 pred.segm.json)

echo "[$(date '+%F %T')] RS4D test (GPU3, with dump)"
DUMP_PREFIX=${CKPT_BASE}/rs4d/rs4d_bbox_whu1024_bs8_bf16_800ep/test_out/pred \
  bash ${PROJ}/scripts/baselines/infer_whu1024.sh rs4d 3 \
  || { echo "rs4d test FAIL"; exit 1; }

for MODE in coco fix4; do
  ${PY} boundary_eval.py ${CKPT_BASE}/maskdino/maskdino_R50_whu1024_bs2ai8_300ep/test_out/pred.segm.json --mode ${MODE} --tag MaskDINO || true
  ${PY} boundary_eval.py ${CKPT_BASE}/rs4d/rs4d_bbox_whu1024_bs8_bf16_800ep/test_out/pred.segm.json --mode ${MODE} --tag RS4D || true
done
echo "[$(date '+%F %T')] FINALIZE DONE"
