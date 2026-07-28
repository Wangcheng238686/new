# SAM2 安装说明

本目录是项目依赖的 **SAM2 (Segment Anything 2)** 源码，以 editable（可编辑）方式安装。
来源：`/data/wangcheng/myproject/sam2-main`（原项目环境的 editable 安装路径）。

## 包含内容

```
sam2/
├── sam2/                  # 核心 Python 包（modeling/, configs/, utils/, csrc/）
│   ├── _C.so              # 编译好的 CUDA 扩展（connected_components）
│   ├── sam2_hiera_*.yaml  # 模型配置（含 b+/l/s/t）
│   └── ...
├── setup.py               # 安装/编译入口
├── pyproject.toml
├── MANIFEST.in
├── INSTALL.md             # SAM2 官方安装文档
├── LICENSE
└── README_sam2.md
```

## 安装

### 情况 A：目标机器环境与当前一致（Python 3.9 + torch 2.8.0 + CUDA 12.9）

可直接 editable 安装，复用已编译的 `_C.so`：

```bash
conda activate cvt2          # 或你的目标环境
cd sam2
pip install -e .
```

### 情况 B：目标机器 Python/torch/CUDA 版本不同

`_C.so` 是针对 **Python 3.9 + torch 2.8.0 + CUDA 12.9** 编译的，版本不匹配时会加载失败。
需要重新编译 CUDA 扩展（源码在 `sam2/csrc/connected_components.cu`）：

```bash
conda activate <你的环境>     # 需有匹配的 CUDA toolkit + 编译器 (nvcc/gcc)
cd sam2
pip install -e .             # setup.py 会自动检测并重编译 _C.so
```

> 若 `import sam2` 报 `OSError: ... _C.so ... undefined symbol` 或 `cannot open shared object`，
> 即为版本不匹配，按情况 B 重编译。

## 与本项目的衔接

本项目的 `rsprompter/models_sam2.py` 和 `uav/sam2_backbone.py` 通过
`SAM2_SOURCE_PATH` 环境变量定位 sam2 源码目录：

```bash
# 若 sam2 已通过 `pip install -e .` 装入环境，可不设此变量（import sam2 直接可用）
# 否则指向本目录的上一层（包含 sam2/ 子目录的目录）：
export SAM2_SOURCE_PATH=/path/to/whu1024_reproduce_bundle/sam2
```

## SAM2 预训练权重（不在本目录内）

训练/推理需要 `sam2_hiera_base_plus.pt`（约 323 MB），需另行获取：

```bash
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt \
  -O /path/to/sam2_hiera_base_plus.pt
export SAM2_CKPT=/path/to/sam2_hiera_base_plus.pt
```
