#!/bin/bash
# =============================================================================
# dt2 环境搭建：CondInst / SOLOv2 (AdelaiDet, detectron2@9eb4831)
# 路径: /data2/wangcheng/envs/dt2  |  GPU: 4090(sm_89) + cuda11.7(8.6+PTX)
# 注意: 网络需绕过失效代理；github 走 ghfast.top 镜像
# =============================================================================
set -x
source /home/wangcheng/miniconda3/etc/profile.d/conda.sh

ENV=/data2/wangcheng/envs/dt2
LOGDIR=/home/wangcheng/project/new/portable_sam2_explicit_coarse/logs/baselines
mkdir -p "$LOGDIR"

# 1. 创建环境
conda create -p "$ENV" python=3.9 -y || exit 1
conda activate "$ENV" || exit 1

# 2. torch 1.13.1 + cu117 (4090 经 PTX JIT 可用)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
  --extra-index-url https://download.pytorch.org/whl/cu117 || exit 2

# 3. cuda-toolkit 11.7 (提供与 torch 匹配的 nvcc，编译 detectron2)
conda install -y -c nvidia/label/cuda-11.7.1 cuda-toolkit || exit 3
export CUDA_HOME="$ENV"
export PATH="$ENV/bin:$PATH"
nvcc --version || exit 3

# 4. detectron2 @ 9eb4831 (AdelaiDet 推荐 commit)，经镜像下载源码
cd /tmp
if [ ! -d detectron2-9eb4831 ]; then
  env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    curl -sL --connect-timeout 20 --max-time 600 -o d2.tar.gz \
    "https://ghfast.top/https://github.com/facebookresearch/detectron2/archive/9eb4831f742ae6a13b8edb61d07b619392fb6543.tar.gz" || exit 4
  tar xzf d2.tar.gz || exit 4
fi
cd detectron2-9eb4831* || exit 4
FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.6+PTX" pip install -e . --no-build-isolation || exit 4

# 5. AdelaiDet (CUDA 扩展)
cd /home/wangcheng/project/AdelaiDet || exit 5
FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.6+PTX" pip install -e . --no-build-isolation || exit 5

# 6. 常用依赖
pip install pycocotools opencv-python-headless || exit 6

# 7. 验证
python -c "
import torch, detectron2, adet
print('torch', torch.__version__, '| d2', detectron2.__version__, '| cuda', torch.cuda.is_available())
from adet.modeling.condinst import CondInst
from adet.modeling.solov2 import SOLOv2
print('CondInst / SOLOv2 import OK')
"
echo "DT2_ENV_BUILD_DONE rc=$?"
