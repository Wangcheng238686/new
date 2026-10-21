#!/bin/bash
# =============================================================================
# WHU-1024 对比算法统一测试集推理脚本 (13 个模型)
#   输出: bbox/segm 的 mAP + AP50/75/s/m/l -> work_dir/test_out/test_metrics.json
#   可视化(可选): 仅 mask 叠加原图 / 无置信度 / 每实例独立颜色 -> test_vis/
#
# 用法: bash infer_whu1024.sh <model> [gpu] [vis 0|1]
#   model: maskrcnn | msrcnn | htc | scnet | mask2former | rs4d | catnet
#          | condinst | solov2 | rtmdet | maskdino | yolo11 | rsiisn
#   环境变量: MAX_IMG=N(调试: 只推 N 张) VIS_LIMIT=N(只可视化前 N 张)
#             CKPT=<path>(手动指定权重, 默认自动选 val best)
#             SCORE_THR=0.5  VIS_DIR=<dir>
# =============================================================================
set -uo pipefail

MODEL="${1:?用法: infer_whu1024.sh <model> [gpu] [vis]}"
GPU="${2:-0}"
VIS="${3:-0}"

PROJ_ROOT="/home/wangcheng/project/new/portable_sam2_explicit_coarse"
LOGDIR="${PROJ_ROOT}/logs/baselines"
CKPT_BASE="/data1/wangcheng/checkpoint/whu1024_baselines"
PY_DIR="${PROJ_ROOT}/scripts/baselines"
mkdir -p "${LOGDIR}"
# torch>=2.6 默认 weights_only=True 会拒载含 mmengine meta 的 ckpt(resume 需要)
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_VISIBLE_DEVICES="${GPU}"

TS=$(date +%Y%m%d_%H%M%S)
VISSUF=$([ "${VIS}" = "1" ] && echo "_vis" || echo "")
LOG="${LOGDIR}/${MODEL}_whu1024_infer${VISSUF}_${TS}_pid$$.log"

best_adet_ckpt() {  # 从 detectron2 metrics.json 选 val segm/AP 最高的 ckpt
  python3 - "$1" << 'PYEOF'
import json, sys, os, glob
od = sys.argv[1]
best_iter, best_ap = None, -1
for mj in glob.glob(os.path.join(od, 'metrics.json')):
    for line in open(mj):
        try: d = json.loads(line)
        except Exception: continue
        ap = d.get('segm/AP')
        if ap is not None and 'iteration' in d and ap > best_ap:
            best_ap, best_iter = ap, d['iteration']
if best_iter is None: sys.exit(1)
c = sorted(glob.glob(os.path.join(od, f'model_*{best_iter}.pth')))
print(c[-1] if c else '')
PYEOF
}

case "${MODEL}" in
  maskrcnn)
    ENVV="cvt2"; ENTRY="${PY_DIR}/whu_test_mmdet.py"
    CFG="/home/wangcheng/project/mmdetection/configs/whu1024/mask-rcnn_r50_fpn_1x_whu1024.py"
    WORKDIR="${CKPT_BASE}/maskrcnn/mask-rcnn_r50_fpn_1x_whu1024_bs8ai2";;
  msrcnn)
    ENVV="cvt2"; ENTRY="${PY_DIR}/whu_test_mmdet.py"
    CFG="/home/wangcheng/project/mmdetection/configs/whu1024/ms-rcnn_r50-caffe_fpn_1x_whu1024.py"
    WORKDIR="${CKPT_BASE}/msrcnn/ms-rcnn_r50-caffe_fpn_1x_whu1024_bs4ai4";;
  htc)
    ENVV="cvt2"; ENTRY="${PY_DIR}/whu_test_mmdet.py"
    CFG="/home/wangcheng/project/mmdetection/configs/whu1024/htc-without-semantic_r50_fpn_1x_whu1024.py"
    WORKDIR="${CKPT_BASE}/htc/htc-without-semantic_r50_fpn_1x_whu1024_bs4ai4";;
  scnet)
    ENVV="cvt2"; ENTRY="${PY_DIR}/whu_test_mmdet.py"
    CFG="/home/wangcheng/project/mmdetection/configs/whu1024/scnet_r50_fpn_1x_whu1024.py"
    WORKDIR="${CKPT_BASE}/scnet/scnet_r50_fpn_1x_whu1024_bs8ai2";;
  mask2former)
    ENVV="cvt2"; ENTRY="${PY_DIR}/whu_test_mmdet.py"
    CFG="/home/wangcheng/project/mmdetection/configs/whu1024/mask2former_r50_whu1024.py"
    WORKDIR="${CKPT_BASE}/mask2former/mask2former_r50_whu1024_bs4ai4_300ep";;
  rs4d)
    ENVV="cvt2"; ENTRY="${PY_DIR}/whu_test_mmdet.py"
    CFG="/home/wangcheng/project/RS4D/configs/rs4d/rs4d_bbox-whu-1024.py"
    WORKDIR="${CKPT_BASE}/rs4d/rs4d_bbox_whu1024_bs8_bf16_800ep"
    export PYTHONPATH="/home/wangcheng/project/RS4D";;
  catnet)
    ENVV="cvt2"; ENTRY="${PY_DIR}/whu_test_mmdet.py"
    CFG="/home/wangcheng/project/CATNet/configs/whu/cat_mask_rcnn_r50_3x_whu1024.py"
    WORKDIR="${CKPT_BASE}/catnet/cat_mask_rcnn_r50_3x_whu1024_bs4ai2"
    export PYTHONPATH="/home/wangcheng/project/CATNet";;
  condinst)
    ENVV="dt2"; ENTRY="${PY_DIR}/whu_test_adet.py"
    CFG="/home/wangcheng/project/AdelaiDet/configs/WHU/condinst_MS_R_50_1x_whu1024.yaml"
    WORKDIR="${CKPT_BASE}/condinst/condinst_MS_R_50_1x_whu1024_bs8lr05m";;
  solov2)
    ENVV="dt2"; ENTRY="${PY_DIR}/whu_test_adet.py"
    CFG="/home/wangcheng/project/AdelaiDet/configs/WHU/solov2_R50_1x_whu1024.yaml"
    WORKDIR="${CKPT_BASE}/solov2/solov2_R50_1x_whu1024_bs8lr05m";;
  rtmdet)
    ENVV="cvt2"; ENTRY="${PY_DIR}/whu_test_mmdet.py"
    CFG="/home/wangcheng/project/mmdetection/configs/whu1024/rtmdet-ins_s_300e_whu1024.py"
    WORKDIR="${CKPT_BASE}/rtmdet/rtmdet_ins_s_whu1024_bs4ai64_300ep";;
  maskdino)
    ENVV="mdino"; ENTRY="${PY_DIR}/eval_maskdino_whu.py"
    CFG="/home/wangcheng/project/MaskDINO/configs/whu/maskdino_R50_whu1024.yaml"
    WORKDIR="${CKPT_BASE}/maskdino/maskdino_R50_whu1024_bs2ai8_300ep";;
  yolo11)
    ENVV="cvt2"; ENTRY="${PY_DIR}/eval_yolo_whu.py"
    WORKDIR="${CKPT_BASE}/yolo11seg/yolo11s_seg_whu1024_default/yolo11s_seg_whu1024_default";;
  rsiisn)
    ENVV="mm2"; ENTRY="/data2/wangcheng/envs/mm2/lib/python3.9/site-packages/mmdet/.mim/tools/test.py"
    CFG="/home/wangcheng/project/RSIISN/configs/whu1024/rsiisn_cascade_swinT_whu1024.py"
    WORKDIR="${CKPT_BASE}/rsiisn/rsiisn_cascade_swinT_whu1024_bs2ai4_12ep";;
  *) echo "未知 model: ${MODEL}"; exit 1;;
esac

source /home/wangcheng/miniconda3/etc/profile.d/conda.sh
case "${ENVV}" in
  cvt2) conda activate cvt2;;
  dt2)  conda activate /data2/wangcheng/envs/dt2;;
  mdino) conda activate /data2/wangcheng/envs/mdino;;   # 后台链路 activate 偶发失效, 下方用绝对 PY
  mm2)  conda activate /data2/wangcheng/envs/mm2;;
esac
PY=python
[ "${ENVV}" = "mdino" ] && PY=/data2/wangcheng/envs/mdino/bin/python
[ "${ENVV}" = "mm2" ] && PY=/data2/wangcheng/envs/mm2/bin/python

# 权重选择: CKPT 环境变量 > 自动 best
if [ -z "${CKPT:-}" ]; then
  case "${MODEL}" in
    yolo11)
      CKPT="${WORKDIR}/weights/best.pt";;
    rsiisn)
      CKPT=$(ls -t "${WORKDIR}"/best_segm_mAP*.pth 2>/dev/null | head -1);;
    maskdino)
      CKPT=$(${PY} - "${WORKDIR}" << 'PYEOF'
import json, sys, os, glob
od = sys.argv[1]
best_iter, best_ap = None, -1
for mj in glob.glob(os.path.join(od, 'metrics.json')):
    for line in open(mj):
        try: d = json.loads(line)
        except Exception: continue
        ap = d.get('segm/AP')
        if ap is not None and 'iteration' in d and ap > best_ap:
            best_ap, best_iter = ap, d['iteration']
if best_iter is None: sys.exit(1)
c = sorted(glob.glob(os.path.join(od, f'model_*{best_iter}.pth')))
print(c[-1] if c else '')
PYEOF
);;
    condinst|solov2)
      CKPT=$(best_adet_ckpt "${WORKDIR}") || { echo "未找到 adet best ckpt"; exit 1; };;
    *)
      CKPT=$(ls -t "${WORKDIR}"/best_coco_segm_mAP*.pth 2>/dev/null | head -1);;
  esac
  [ -z "${CKPT}" ] && { echo "未找到 best ckpt"; exit 1; }
fi

VIS_ARGS=""
if [ "${VIS}" = "1" ]; then
  VIS_ARGS="--vis"
  [ -n "${VIS_DIR:-}" ] && VIS_ARGS="${VIS_ARGS} --vis-dir ${VIS_DIR}"
fi
MAX_ARGS=""
[ -n "${MAX_IMG:-}" ] && MAX_ARGS="${MAX_ARGS} --max-images ${MAX_IMG}"
[ -n "${VIS_LIMIT:-}" ] && MAX_ARGS="${MAX_ARGS} --vis-limit ${VIS_LIMIT}"
[ -n "${SCORE_THR:-}" ] && MAX_ARGS="${MAX_ARGS} --score-thr ${SCORE_THR}"
[ -n "${DUMP_PREFIX:-}" ] && { [ "${ENTRY##*/}" = "whu_test_mmdet.py" ] || [ "${ENTRY##*/}" = "whu_test_adet.py" ] || [ "${ENTRY##*/}" = "eval_maskdino_whu.py" ]; } && MAX_ARGS="${MAX_ARGS} --dump-prefix ${DUMP_PREFIX}"

{
  echo "===== runtag: ${MODEL}_whu1024_infer${VISSUF} ====="
  echo "date: $(date '+%F %T')  gpu: ${GPU}  vis: ${VIS}"
  echo "config: ${CFG}"
  echo "ckpt: ${CKPT}"
  echo "env: ${ENVV}  python: $(which python)"
} | tee "${LOG}"

mkdir -p "${WORKDIR}/test_out"

case "${MODEL}" in
  yolo11)
    cd /home/wangcheng/project >/dev/null
    ${PY} "${ENTRY}" "${CKPT}" --out-dir "${WORKDIR}/test_out" ${VIS_ARGS} ${MAX_ARGS} 2>&1 | tee -a "${LOG}"
    ;;
  maskdino)
    cd /home/wangcheng/project/MaskDINO >/dev/null
    ${PY} "${ENTRY}" "${CKPT}" --config "${CFG}" --out-dir "${WORKDIR}/test_out" ${VIS_ARGS} ${MAX_ARGS} 2>&1 | tee -a "${LOG}"
    ;;
  rsiisn)
    cd /home/wangcheng/project/RSIISN >/dev/null
    ${PY} "${ENTRY}" "${CFG}" "${CKPT}" --work-dir "${WORKDIR}/test_out" \
      --out "${WORKDIR}/test_out/result.pkl" --eval bbox segm 2>&1 | tee -a "${LOG}"
    ${PY} - "${LOG}" "${WORKDIR}/test_out/test_metrics.json" << 'PYEOF' | tee -a "${LOG}"
import json, re, sys
log = open(sys.argv[1], errors='ignore').read()
vals = [float(v) for v in re.findall(r'maxDets=100 \] = ([0-9.]+)', log)][-12:]
assert len(vals) == 12, f'expect 12 metric lines, got {len(vals)}'
m = {}
for ti, t in enumerate(['bbox', 'segm']):
    for ni, n in enumerate(['mAP', 'mAP50', 'mAP75', 'mAPs', 'mAPm', 'mAPl']):
        m[f'{t}_{n}'] = round(vals[ti*6+ni], 4)
json.dump(m, open(sys.argv[2], 'w'), indent=2)
print('saved', sys.argv[2], m)
PYEOF
    ;;
  *)
    cd "$(dirname "${CFG}")" >/dev/null
    if [ "${ENVV}" = "dt2" ]; then
      TASKS=$([ "${MODEL}" = "solov2" ] && echo "segm" || echo "bbox,segm")
      DP=""
      [ -n "${DUMP_PREFIX:-}" ] && DP="--dump-prefix ${DUMP_PREFIX}"
      python "${ENTRY}" "${CFG}" "${CKPT}" --out-dir "${WORKDIR}/test_out" \
        --tasks "${TASKS}" ${DP} ${VIS_ARGS} ${MAX_ARGS} 2>&1 | tee -a "${LOG}"
    else
      python "${ENTRY}" "${CFG}" "${CKPT}" --out-dir "${WORKDIR}/test_out" \
        ${VIS_ARGS} ${MAX_ARGS} 2>&1 | tee -a "${LOG}"
    fi
    ;;
esac
rc=${PIPESTATUS[0]}

echo "[infer] exit=${rc} | metrics: ${WORKDIR}/test_out/test_metrics.json" | tee -a "${LOG}"
echo "===== done $(date '+%F %T') =====" | tee -a "${LOG}"
exit ${rc}
