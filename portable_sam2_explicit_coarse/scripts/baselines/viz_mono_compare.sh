#!/bin/bash
# 可视化 v3: 每模型单一深色 / 全实例同色 / alpha 0.55 / 无框无置信度
set -u
PROJ=/home/wangcheng/project/new/portable_sam2_explicit_coarse
V2=/home/wangcheng/project/new/viz_compare_whu_test_v2
CK=/data1/wangcheng/checkpoint/whu1024_baselines
GT="/data1/wangcheng/dataset/WHU/2.4 annotation/annotation/test.json"
IMG="/data1/wangcheng/dataset/WHU/2.2 test/test"
mkdir -p "${V2}"
export VIZ_ALPHA=0.55

declare -A VISDIR=(
  [catnet]=${CK}/catnet/cat_mask_rcnn_r50_3x_whu1024_bs4ai2/test_out/vis
  [maskrcnn]=${CK}/maskrcnn/mask-rcnn_r50_fpn_1x_whu1024_bs8ai2/test_out/vis
  [htc]=${CK}/htc/htc-without-semantic_r50_fpn_1x_whu1024_bs4ai4/test_out/vis
  [msrcnn]=${CK}/msrcnn/ms-rcnn_r50-caffe_fpn_1x_whu1024_bs4ai4/test_out/vis
  [condinst]=${CK}/condinst/condinst_MS_R_50_1x_whu1024_bs8lr05m/test_out/vis
  [solov2]=${CK}/solov2/solov2_R50_1x_whu1024_bs8lr05m/test_out/vis
  [yolo11]=${CK}/yolo11seg/yolo11s_seg_whu1024_default/yolo11s_seg_whu1024_default/test_out/vis
)
declare -A COLOR=(   # BGR 深色, 每模型一色
  [gt]="0,0,170" [ours]="170,0,0" [yolo11]="0,80,170" [catnet]="0,130,0"
  [maskrcnn]="120,0,120" [htc]="40,70,160" [msrcnn]="150,70,0"
  [condinst]="0,110,150" [solov2]="100,100,0"
)

for M in catnet maskrcnn htc msrcnn condinst solov2 yolo11; do
  echo "[$(date '+%F %T')] === ${M} (color ${COLOR[$M]}) ==="
  V="${VISDIR[$M]}"; rm -rf "${V}"; mkdir -p "${V}"
  VIZ_MONO_BGR="${COLOR[$M]}" MAX_IMG=30 VIS_LIMIT=25 SCORE_THR=0.5 \
    bash ${PROJ}/scripts/baselines/infer_whu1024.sh ${M} 3 1 >/dev/null 2>&1
  if ls "${V}"/*.png >/dev/null 2>&1; then
    mkdir -p "${V2}/${M}"; cp "${V}"/*.png "${V2}/${M}/"
    echo "  $(ls ${V2}/${M} | wc -l) pngs"
  else
    echo "  WARN: no pngs at ${V}"
  fi
done

# Ours: 已有全量 predictions.json, 直接单色重画
/data2/wangcheng/envs/cvt2/bin/python ${PROJ}/scripts/visualize_instances.py pred \
  --pred-json /home/wangcheng/project/new/viz_compare_whu_test/ours_raw/predictions.json \
  --images-dir "${IMG}" --gt-ann-json "${GT}" \
  --out-dir "${V2}/ours" --mask-only --mono-color "${COLOR[ours]}" \
  --mask-alpha 0.55 --score-thr 0.5 --limit 25 2>&1 | tail -1
# GT: 同风格单色
/data2/wangcheng/envs/cvt2/bin/python ${PROJ}/scripts/visualize_instances.py gt \
  --ann-json "${GT}" --images-dir "${IMG}" \
  --out-dir "${V2}/gt" --mask-only --mono-color "${COLOR[gt]}" \
  --mask-alpha 0.55 --limit 50 2>&1 | tail -1
echo "[$(date '+%F %T')] viz v3 done -> ${V2}"
