# WHU1024 最强实验复现包

基于 RSPrompter + SAM2(base_plus) + LoRA 的 WHU 建筑实例分割，**1024 原生分辨率、单类别（building）**。
本包封装了项目在 WHU 数据集上取得的**最高 bbox / segm 指标**两个实验的完整、可移植代码。

## 1. 复现目标与指标

两个实验代码**完全相同**（同一 git commit），仅训练超参不同：

| 实验 | 封装脚本 | bbox/mAP | segm/mAP | 关键设置 | 峰值 epoch |
|---|---|---|---|---|---|
| **whu1024_baseline_noms** | `run_baseline_noms.sh` | **0.788** | 0.748 | IIMR OFF, 无 multi-scale, bs2×accum1 | 78/80 |
| **whu1024_bs1a2_fixddp** | `run_bs1a2_fixddp.sh` | 0.767 | **0.754** | IIMR OFF, 无 multi-scale, bs1×accum2 | 17（早停@27）|

> COCO 评估，IoU=0.50:0.95，maxDets=100，WHU validation 集。详细各档位指标见 `CODE_VERSION.md`。

## 2. 代码基线

- **Git commit**：`eee7517`（master 分支，2026-06-14）
- 两个实验的 `.params` 快照均记录 `git_commit=eee7517 git_branch=master`，本包源码即从该 commit 用 `git archive` 导出，源码层面零改动。
- 完整来源与超参差异记录见 `CODE_VERSION.md`。

## 3. 目录结构

```
whu1024_reproduce_bundle/
├── README.md                      # 本文件
├── CODE_VERSION.md                # git 版本、超参、COCO 指标明细
├── cvt2_env.yml                   # conda 环境定义 (name: cvt2)
├── portable_sam2_fusion_new/      # 项目源码（基于 eee7517）
│   ├── configs/                   # 含 whu1024 config 及其 _base_
│   ├── models/ rsprompter/ data/ utils/ train/ inference/ uav/
│   ├── scripts/
│   │   ├── run_baseline_noms.sh            # ★ bbox 最强一键复现
│   │   ├── run_bs1a2_fixddp.sh             # ★ segm 最强一键复现
│   │   ├── run_hparam_whu1024_baseplus_4gpu.sh  # 通用训练脚本（被上面封装）
│   │   └── inference_whu1024_baseplus.sh   # 推理脚本
│   └── tools/mask_to_coco_whu512.py        # mask→COCO 标注转换工具
└── sam2/                          # SAM2 依赖源码（精简，含 _C.so）
    ├── sam2/                      # 核心 Python 包
    ├── setup.py  pyproject.toml   # editable 安装 / _C.so 重编译
    └── INSTALL_NOTE.md
```

## 4. 环境安装

### 4.1 创建 conda 环境

```bash
conda env create -f cvt2_env.yml
conda activate cvt2
```

关键依赖：Python 3.9、torch 2.8.0、torchvision 0.23.0、mmdet 3.3.0、mmengine 0.10.7、mmcv 2.1.0、pycocotools、einops、opencv-python（CUDA 12.9）。

### 4.2 安装 SAM2 依赖

`sam2/` 目录是本项目依赖的 SAM2 源码（editable 安装）。注意其中的 `sam2/_C.so` 是在**当前机器（Python 3.9 + torch 2.8 + CUDA 12.9）编译的**，换到版本不同的机器需重新编译：

```bash
cd sam2
# 若 Python/torch/CUDA 版本一致，可直接 editable 安装复用 _C.so：
pip install -e .
# 若版本不同，需重编译 CUDA 扩展（源码在 sam2/csrc/）：
# pip install -e .   # setup.py 会自动重编译
```

详见 `sam2/INSTALL_NOTE.md`。

## 5. 数据集组织

WHU 数据集需按如下结构放置（COCO 格式标注，单类 building）：

```
$WHU_DATA_ROOT/
├── 2.1 train/train/                 # 训练图像 (*.tif/*.png)
├── 2.2 test/test/                   # 测试图像
├── 2.3 valid/validation/           # 验证图像
└── 2.4 annotation/annotation/
    ├── train.json                   # COCO 标注
    ├── validation.json
    └── test.json
```

若只有二值 mask，可用 `tools/mask_to_coco_whu512.py` 生成 COCO 标注（虽名字含 512，1024 同样适用）。

## 6. 训练复现

所有路径默认指向当前机器（`/data/wangcheng/...`），**可通过环境变量覆盖**（见第 8 节）。

### 6.1 复现 bbox 最强（baseline_noms）

```bash
cd portable_sam2_fusion_new
bash scripts/run_baseline_noms.sh
```

### 6.2 复现 segm 最强（bs1a2_fixddp）

```bash
cd portable_sam2_fusion_new
bash scripts/run_bs1a2_fixddp.sh
```

默认 4 卡 DDP 后台运行，日志写入 `logs/ablation_hyperparam_whu1024_baseplus/`，权重写入 `$CKPT_BASE/whu1024_<RUN_TAG>/`。

### 6.3 自定义

两个封装脚本仅 export 超参后委托给 `run_hparam_whu1024_baseplus_4gpu.sh`。可直接覆盖任意变量，例如：

```bash
GPU_LIST=0,1 NUM_GPUS=2 BEST_BS=2 BEST_ACCUM=2 bash scripts/run_baseline_noms.sh
```

## 7. 推理

```bash
cd portable_sam2_fusion_new

# 用 baseline_noms 的权重推理（默认 CHECKPOINT_PATH 已指向它）
bash scripts/inference_whu1024_baseplus.sh

# 用 bs1a2 权重推理：
CHECKPOINT_PATH=/data/wangcheng/checkpoint/ablation_hyperparam_whu1024_baseplus/whu1024_bs1a2_fixddp/best_model.pth \
  bash scripts/inference_whu1024_baseplus.sh
```

> ⚠️ 本包**不含权重文件**。权重路径：
> - baseline_noms：`/data/wangcheng/checkpoint/ablation_hyperparam_whu1024_baseplus/whu1024_baseline_noms_4gpu/best_model.pth`
> - bs1a2：`/data/wangcheng/checkpoint/ablation_hyperparam_whu1024_baseplus/whu1024_bs1a2_fixddp/best_model.pth`

## 8. 可覆盖的环境变量（路径/超参）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `PYTHON` | `/data/wangcheng/envs/cvt2/bin/python` | Python 解释器 |
| `WHU_DATA_ROOT` | `/data/wangcheng/dataset/WHU` | WHU 数据集根 |
| `SAM2_CKPT` | `/data/wangcheng/pretrained-models/sam2/sam2_hiera_base_plus.pt` | SAM2 预训练权重 |
| `SAM2_SOURCE_PATH` | `/data/wangcheng/myproject/sam2-main` | SAM2 源码目录（import sam2 用） |
| `CKPT_BASE` | `/data/wangcheng/checkpoint/ablation_hyperparam_whu1024_baseplus` | 训练权重输出根 |
| `CHECKPOINT_PATH` | (推理脚本) baseline_noms 的 best_model.pth | 推理加载的权重 |
| `GPU_LIST` / `NUM_GPUS` | `0,1,2,3` / `4` | GPU 编号与数量 |
| `BEST_BS` / `BEST_ACCUM` | 见各脚本 | per-GPU batch / 梯度累积步 |
| `BEST_LR` / `BEST_MULT` | `5e-4` / `1.0` | 基础学习率 / backbone lr 倍率 |
| `MAX_EPOCHS` | `80` | 最大训练轮数 |

> **迁移到其他机器**：至少需设置 `PYTHON`、`WHU_DATA_ROOT`、`SAM2_CKPT`、`SAM2_SOURCE_PATH`、`CKPT_BASE`，并按第 4 节安装环境与 sam2。
