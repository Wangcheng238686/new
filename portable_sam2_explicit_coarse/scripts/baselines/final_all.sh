#!/bin/bash
# 最终收尾: MaskDINO test(修复后) + RSPrompter test(ep300) + 双方 boundary
set -u
PROJ=/home/wangcheng/project/new/portable_sam2_explicit_coarse
CKPT_BASE=/data1/wangcheng/checkpoint/whu1024_baselines
PY=/data2/wangcheng/envs/cvt2/bin/python
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
cd ${PROJ}/scripts/baselines

echo "[$(date '+%F %T')] ===== MaskDINO test ====="
DUMP_PREFIX=${CKPT_BASE}/maskdino/maskdino_R50_whu1024_bs2ai8_300ep/test_out/pred \
  bash ${PROJ}/scripts/baselines/run_maskdino_whu1024.sh test 3 \
  ${CKPT_BASE}/maskdino/maskdino_R50_whu1024_bs2ai8_300ep/model_final.pth
echo "[$(date '+%F %T')] maskdino test exit=$?"

echo "[$(date '+%F %T')] ===== RSPrompter test (epoch_300, final) ====="
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=/home/wangcheng/project/RSPrompter-release
RP_CKPT=${CKPT_BASE}/rsprompter/rsprompter_anchor_whu1024_bs1ai2_300ep/epoch_300.pth
RP_DIR=${CKPT_BASE}/rsprompter/rsprompter_anchor_whu1024_bs1ai2_300ep/test_out
mkdir -p ${RP_DIR}
${PY} ${PROJ}/scripts/baselines/whu_test_mmdet.py \
  /home/wangcheng/project/RSPrompter-release/configs/rsprompter/rsprompter_anchor-whu-1024.py \
  ${RP_CKPT} --out-dir ${RP_DIR} \
  --dump-prefix ${RP_DIR}/pred
echo "[$(date '+%F %T')] rsprompter test exit=$?"

for MODE in coco fix4; do
  ${PY} boundary_eval.py ${CKPT_BASE}/maskdino/maskdino_R50_whu1024_bs2ai8_300ep/test_out/pred.segm.json --mode ${MODE} --tag MaskDINO || true
  ${PY} boundary_eval.py ${RP_DIR}/pred.segm.json --mode ${MODE} --tag RSPrompter || true
done
echo "[$(date '+%F %T')] ALL FINAL DONE"
