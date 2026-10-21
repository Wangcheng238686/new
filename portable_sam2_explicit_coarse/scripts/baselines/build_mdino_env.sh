#!/bin/bash
# =============================================================================
# mdino 环境搭建：MaskDINO (IDEA-Research, CVPR2023, detectron2 v0.6 栈)
# 路径: /data2/wangcheng/envs/mdino | GPU: 4090(sm_89) + cuda11.7(8.6+PTX)
# 不动 dt2 的 detectron2@9eb4831(AdelaiDet 依赖)
# 网络绕过失效代理; github 走 ghfast.top 镜像; CUDA_HOME 复用 dt2 内的 toolkit
# =============================================================================
set -x
source /home/wangcheng/miniconda3/etc/profile.d/conda.sh

ENV=/data2/wangcheng/envs/mdino

conda create -p "$ENV" python=3.9 -y || exit 1
conda activate "$ENV" || exit 1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

# torch 1.13.1 + cu117 (与 dt2 一致, MSDeformAttn/detectron2 v0.6 匹配)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
  --extra-index-url https://download.pytorch.org/whl/cu117 || exit 2

# detectron2 v0.6 (2021-11, 含 file_io/AMPTrainer/projects.point_rend)
export CUDA_HOME=/data2/wangcheng/envs/dt2
export PATH="$CUDA_HOME/bin:$PATH"
cd /tmp || exit 3
if [ ! -d detectron2-v0.6 ]; then
  curl -sL --connect-timeout 20 --max-time 600 -o d2v06.tar.gz \
    "https://ghfast.top/https://github.com/facebookresearch/detectron2/archive/refs/tags/v0.6.tar.gz" || exit 3
  tar xzf d2v06.tar.gz || exit 3
fi
cd detectron2-0.6 || exit 3
FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.6+PTX" pip install -e . --no-build-isolation || exit 3

# MaskDINO 依赖 + MSDeformAttn CUDA 扩展
cd /home/wangcheng/project/MaskDINO || exit 4
pip install -r requirements.txt --no-deps -q || exit 4
pip install pycocotools opencv-python-headless shapely timm h5py scikit-image submitit cython scipy || exit 4
cd maskdino/modeling/pixel_decoder/ops || exit 4
FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.6+PTX" python setup.py build install || exit 4
python -c "import MultiScaleDeformableAttention as MSDA; assert hasattr(MSDA,'ms_deform_attn_forward'); print('mdino env OK')"
