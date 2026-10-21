#!/bin/bash
# 终版论文网格: 等 dump 链完成 -> 9 列(raw|gt|6基线|Ours) x 3 行
set -u
PROJ=/home/wangcheng/project/new/portable_sam2_explicit_coarse
CK=/data1/wangcheng/checkpoint/whu1024_baselines
DUMPC=${PROJ}/scripts/baselines

# 等 dump 链结束
while pgrep -f "dump_preds_chain.sh" >/dev/null 2>&1; do sleep 120; done

# 确认 4 个 segm json 齐
for f in maskrcnn/mask-rcnn_r50_fpn_1x_whu1024_bs8ai2 htc/htc-without-semantic_r50_fpn_1x_whu1024_bs4ai4 \
         msrcnn/ms-rcnn_r50-caffe_fpn_1x_whu1024_bs4ai4 condinst/condinst_MS_R_50_1x_whu1024_bs8lr05m; do
  [ -f "${CK}/${f}/test_out/pred_full.segm.json" ] || { echo "missing ${f}"; exit 1; }
done
echo "[$(date '+%F %T')] all dumps ready, generating final grid"

cd ${DUMPC}
/data2/wangcheng/envs/cvt2/bin/python layout_grid.py \
  --gt "/data1/wangcheng/dataset/WHU/2.4 annotation/annotation/test.json" \
  --images-dir "/data1/wangcheng/dataset/WHU/2.2 test/test" \
  --ours ours=/home/wangcheng/project/new/viz_compare_whu_test/ours_raw/predictions.json \
  --baseline catnet=${CK}/catnet/cat_mask_rcnn_r50_3x_whu1024_bs4ai2/test_out/pred_full.segm.json \
  --baseline condinst=${CK}/condinst/condinst_MS_R_50_1x_whu1024_bs8lr05m/test_out/pred_full.segm.json \
  --baseline htc=${CK}/htc/htc-without-semantic_r50_fpn_1x_whu1024_bs4ai4/test_out/pred_full.segm.json \
  --baseline maskrcnn=${CK}/maskrcnn/mask-rcnn_r50_fpn_1x_whu1024_bs8ai2/test_out/pred_full.segm.json \
  --baseline msrcnn=${CK}/msrcnn/ms-rcnn_r50-caffe_fpn_1x_whu1024_bs4ai4/test_out/pred_full.segm.json \
  --baseline yolo=${CK}/yolo11seg/yolo11s_seg_whu1024_default/yolo11s_seg_whu1024_default/test_out/pred_coco.json \
  --rival catnet --ours-min 0.9 --min-neighbors 6 --density-window 220 --rows 3 --cell 256 --out /home/wangcheng/project/new/viz_grid_final 2>&1 | tail -4
echo "[$(date '+%F %T')] final grid done"
