#!/bin/bash
# 等 mm2 环境就绪后: 安装 RSIISN 官方补丁 -> 验证 -> GPU3 smoke
set -u
MP=/data2/wangcheng/envs/mm2/bin/python
RSIISN=/home/wangcheng/project/RSIISN

echo "[$(date '+%F %T')] waiting for mm2 env (torch/mmcv/mmdet)..."
for i in $(seq 1 240); do
  ${MP} -c "import torch, mmcv, mmdet" 2>/dev/null && break
  sleep 60
done
${MP} -c "import torch, mmcv, mmdet" 2>/dev/null || { echo "mm2 still not ready, giving up"; exit 1; }
echo "[$(date '+%F %T')] mm2 ready: $(${MP} -c 'import torch,mmcv,mmdet;print(torch.__version__, mmcv.__version__, mmdet.__version__)')"

MMD2=$(${MP} -c "import mmdet,os;print(os.path.dirname(mmdet.__file__))")
echo "mmdet dir: ${MMD2}"

# 官方补丁文件 -> mmdet 源码位置(先备份 .orig)
install_patch() {
  src=$1; dst=$2
  [ -f "${dst}.orig" ] || cp "${dst}" "${dst}.orig"
  cp "${src}" "${dst}"
  echo "patched: ${dst} ($(diff -q ${dst}.orig ${dst} >/dev/null && echo SAME || echo CHANGED))"
}
install_patch ${RSIISN}/cascade-swin/code/class_names.py    ${MMD2}/datasets/class_names.py
install_patch ${RSIISN}/cascade-swin/code/coco.py           ${MMD2}/datasets/coco.py
install_patch ${RSIISN}/cascade-swin/code/coco_instance.py  ${MMD2}/datasets/coco_instance.py
install_patch ${RSIISN}/cascade-swin/code/fcn_mask_head.py  ${MMD2}/models/roi_heads/mask_heads/fcn_mask_head.py
install_patch ${RSIISN}/cascade-swin/code/fpn.py            ${MMD2}/models/necks/fpn.py

# 验证补丁后的 mmdet 可导入
cd ${RSIISN}
${MP} -c "
import mmdet.models.necks.fpn as f
import mmdet.models.roi_heads.mask_heads.fcn_mask_head as h
import mmdet.datasets
from mmcv.cnn import GeneralizedAttention
print('patched mmdet imports OK')" || exit 2

# GPU3 smoke(10min 超时截断)
bash /home/wangcheng/project/new/portable_sam2_explicit_coarse/scripts/baselines/run_rsiisn_whu1024.sh smoke 3
echo "[$(date '+%F %T')] chain done"
