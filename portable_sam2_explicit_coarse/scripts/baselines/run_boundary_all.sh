#!/bin/bash
# =============================================================================
# 边界指标(Boundary AP)全模型串行驱动: GPU1 重推 8 个已有模型(带预测落盘)
#   + 每个跑完立即评估 boundary(coco 口径 r=2 + fix4 口径)
# 已有 pred json 免重推: rsiisn / yolo11(已评估)
# 后续 maskdino/rs4d 训完 test 时另带 dump 再补
# 用法: nohup bash run_boundary_all.sh 1 > /tmp/boundary_all.log 2>&1 &
# =============================================================================
set -u
GPU="${1:-1}"
PROJ=/home/wangcheng/project/new/portable_sam2_explicit_coarse
CKPT_BASE=/data1/wangcheng/checkpoint/whu1024_baselines
PY=/data2/wangcheng/envs/cvt2/bin/python
cd ${PROJ}/scripts/baselines

# model -> work_dir 中 pred.segm.json 的定位
declare -A WD=(
  [maskrcnn]=maskrcnn/mask-rcnn_r50_fpn_1x_whu1024_bs8ai2
  [msrcnn]=msrcnn/ms-rcnn_r50-caffe_fpn_1x_whu1024_bs4ai4
  [htc]=htc/htc-without-semantic_r50_fpn_1x_whu1024_bs4ai4
  [catnet]=catnet/cat_mask_rcnn_r50_3x_whu1024_bs4ai2
  [mask2former]=mask2former/mask2former_r50_whu1024_bs4ai4_300ep
  [rtmdet]=rtmdet/rtmdet_ins_s_whu1024_bs4ai64_300ep
  [condinst]=condinst/condinst_MS_R_50_1x_whu1024_bs8lr05m
  [solov2]=solov2/solov2_R50_1x_whu1024_bs8lr05m
)

for M in maskrcnn msrcnn htc catnet mask2former rtmdet condinst solov2; do
  DIR=${CKPT_BASE}/${WD[$M]}/test_out
  echo "[$(date '+%F %T')] ===== ${M}: infer+dump ====="
  if [ "${M}" = "mask2former" ]; then
    CKPT=${CKPT_BASE}/${WD[$M]}/iter_220725.pth DUMP_PREFIX=${DIR}/pred \
      bash ${PROJ}/scripts/baselines/infer_whu1024.sh ${M} ${GPU} || { echo "infer FAIL ${M}"; continue; }
  else
    DUMP_PREFIX=${DIR}/pred \
      bash ${PROJ}/scripts/baselines/infer_whu1024.sh ${M} ${GPU} || { echo "infer FAIL ${M}"; continue; }
  fi
  for MODE in coco fix4; do
    echo "[$(date '+%F %T')] ${M} boundary (${MODE})"
    ${PY} boundary_eval.py ${DIR}/pred.segm.json --mode ${MODE} --tag ${M} \
      || echo "boundary FAIL ${M} ${MODE}"
  done
done

# 已有 pred json 的两个补跑(若尚无 boundary 结果)
for PAIR in "rsiisn:${CKPT_BASE}/rsiisn/rsiisn_cascade_swinT_whu1024_bs2ai4_12ep/test_out" "yolo11:${CKPT_BASE}/yolo11seg/yolo11s_seg_whu1024_default/yolo11s_seg_whu1024_default/test_out"; do
  M=${PAIR%%:*}; DIR=${PAIR#*:}
  for MODE in coco fix4; do
    [ -f ${DIR}/boundary_${MODE}.json ] && continue
    ${PY} boundary_eval.py ${DIR}/pred_coco.json --mode ${MODE} --tag ${M} || true
  done
done
echo "[$(date '+%F %T')] ALL DONE"
