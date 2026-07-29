# Portable SAM2 Explicit Coarse

这是一个独立派生项目，目标是在已验证的 WHU-1024 / SAM2 Base+ 基线上，
干净地集成以下四项：

1. SAM2 四层特征上的 PAFPN；
2. 显式、ROI-local、带独立监督的 coarse mask；
3. 从同型号 SAM2 checkpoint 严格加载并冻结的原生 PromptEncoder；
4. PromptEncoder 之前、只修正局部边界的 P2BoundaryRefiner（实验性 C5-v2）。

主代码位于 `portable_sam2_explicit_coarse/`。IIMR、多轮 mask memory 和
topology-token 路线已从主代码删除。`legacy_baseline/` 是旧项目复现包的冻结
副本，不参与新主线 import，只用于验证和复现历史基线。

完整的模型接线、目录结构、逐文件用途和参数说明见
[`PROJECT_ARCHITECTURE.md`](PROJECT_ARCHITECTURE.md)。

## 历史基线复现

```bash
bash scripts/verify_legacy_baseline.sh
bash scripts/reproduce_legacy_segm.sh
```

冻结快照对应来源 commit：

`eee751738d0c4e9f5fa7fdc017cec56bafe11584`

已记录的 WHU validation 指标：

- `baseline_noms`: bbox mAP 0.788，segm mAP 0.748；
- `bs1a2_fixddp`: bbox mAP 0.767，segm mAP 0.754。

复现脚本、环境、配置、源码和 SHA256 清单均保存在 `legacy_baseline/`。

## 新主线

```bash
cd portable_sam2_explicit_coarse

# 组件与接线 smoke test
bash scripts/smoke_test_components.sh

# 默认：PAFPN + points+box+dense，P2BoundaryRefiner 关闭
bash scripts/run_whu1024_explicit_coarse_4gpu.sh

# 官方 64×64 严格对照与唯一增量 C5-v2
bash scripts/ablations/r1_c4_pafpn_coarse_points_box_dense_emb64.sh
bash scripts/ablations/c5v2_pafpn_coarse_p2_boundary_refiner_emb64.sh

# 消融
EXPLICIT_PROMPT_MODE=points bash scripts/run_whu1024_explicit_coarse_4gpu.sh
EXPLICIT_PROMPT_MODE=points_box bash scripts/run_whu1024_explicit_coarse_4gpu.sh
```

默认数据与预训练权重路径保持旧基线口径，并可通过 `WHU1024_DATA_ROOT`、
`SAM2_CKPT`、`SAM2_REPO`、`CHECKPOINT_DIR` 覆盖。

## 从权重推理

新 checkpoint 会嵌入解析后的完整模型配置、训练超参、数据协议和 SAM2 运行时
信息。推理脚本只需指定 checkpoint，默认在完整 WHU validation 上输出 bbox/segm
COCO 指标和预测 JSON：

```bash
cd portable_sam2_explicit_coarse

bash scripts/infer_whu_checkpoint.sh \
  /data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/EXPERIMENT/best_model.pth
```

使用 WHU test 或自定义 COCO 测试集：

```bash
bash scripts/infer_whu_checkpoint.sh /path/to/model.pth --split test

bash scripts/infer_whu_checkpoint.sh /path/to/model.pth \
  --split custom \
  --data-root /path/to/dataset \
  --ann-file annotations/test.json \
  --image-subdir test/images
```

旧 checkpoint 若没有嵌入 `model_config`，需额外传入
`--config configs/对应配置.py`。

当前 B0/B1、M0/M1、C1–C4、R0、R1-C4、C5-v2 都支持该推理入口。后续所有影响模型结构、模块开关、
张量形状或 forward/predict 行为的新参数，都必须由配置解析进 `cfg.model`，确保
checkpoint 能完整记录并由推理器无歧义重建；训练、数据和运行时参数分别保存在
`training_args`、`data_config` 和 `runtime_config`。

## 消融矩阵

B0/B1保留旧基线的零初始化可训练MLP image PE；公共矩阵的detector loss默认在
全部epoch保持权重1.0。训练日志会同步打印最终mask的ROI target填充率和logit/
概率统计，用于快速排除全空mask回归。

消融入口统一放在
`portable_sam2_explicit_coarse/scripts/ablations/`，覆盖 aggregator/PAFPN、
旧 MLP/coarse、box、dense prompt 和 P2BoundaryRefiner。每次启动前都会检查解析后的配置
是否与脚本声明一致。

```bash
cd portable_sam2_explicit_coarse

# 全矩阵 smoke；不会开始训练
bash scripts/ablations/smoke_all.sh

# 消融当前默认使用 20% train / 100% validation
bash scripts/ablations/c4_pafpn_coarse_points_box_dense.sh

# MLP final-mask 坐标契约消融：B0/B1 的 full-image 对照
bash scripts/ablations/m0_aggregator_mlp_full_image.sh
bash scripts/ablations/m1_pafpn_mlp_full_image.sh

# SAM2 image embedding 分辨率与 P2 边界细化严格对照
bash scripts/ablations/r0_b0_aggregator_mlp_emb64.sh
bash scripts/ablations/r1_c4_pafpn_coarse_points_box_dense_emb64.sh
bash scripts/ablations/c5v2_pafpn_coarse_p2_boundary_refiner_emb64.sh

# 默认 nohup + setsid 后台运行；前台调试时显式关闭
RUN_IN_BACKGROUND=0 bash scripts/ablations/c4_pafpn_coarse_points_box_dense.sh

# 比例仍可独立覆盖；全量训练需显式设置 TRAIN_SUBSET_RATIO=1.0
TRAIN_SUBSET_RATIO=1.0 VAL_SUBSET_RATIO=1.0 \
  bash scripts/ablations/c4_pafpn_coarse_points_box_dense.sh
```

矩阵定义、参数覆盖和 dry-run 用法见
`portable_sam2_explicit_coarse/scripts/ablations/README.md`。

机器相关路径统一维护在
`portable_sam2_explicit_coarse/configs/environment.sh`。迁移到其他机器时，只需
修改其中的 Python、SAM2 checkpoint、WHU 数据集、checkpoint/log/tmp 根目录和
GPU 默认值；所有训练、消融、推理、smoke 与监视器入口都会自动加载。也可设置
`PORTABLE_SAM2_ENV_FILE=/absolute/path/to/environment.sh` 使用仓库外配置。

每个训练入口默认通过 `nohup setsid` 脱离终端运行，启动后打印后台 PID
和日志路径并立即返回，但不创建 PID 文件；关闭终端不会终止 torchrun。stdout/stderr 自动保存到项目内
`portable_sam2_explicit_coarse/logs/ablations/`。设置 `RUN_IN_BACKGROUND=0` 可恢复
前台运行。日志开头包含解析后的架构、
数据、优化器、EMA、初始化/续训路径、有效全局 batch size、git commit 和完整
torchrun 命令。可用 `LOG_DIR` 覆盖日志目录，或用 `LOG_FILE` 指定精确文件。
并行实验可从最外层脚本传入独立端口，例如
`bash scripts/ablations/b0_aggregator_mlp.sh --master-port 29601`；命令行端口优先于
`MASTER_PORT` 环境变量，二者都未提供时自动派生端口。
所有训练入口还通过公共运行器共享 `TORCH_DDP_TIMEOUT_SECONDS=1800`，避免
rank-0-only 完整验证和 COCO 评估超过 PyTorch 默认 600 秒；可在最外围脚本前设置
该环境变量覆盖。

全部新主线实验默认使用物理 GPU `1,2` 的两卡 DDP、每卡 batch `1`、梯度累积
`4`，有效全局 batch 保持 `8`。validation 继续保持 WHU1024 历史基线口径：
`batch_size=1`，只由 rank 0 遍历完整验证集，其余 rank 等待同步；空 GT 图像仍进入
COCO 评估，bbox/segm 都按 detector score 排序，bbox 主摘要和 best-bbox
checkpoint 默认使用 `bbox/mAP`（`bbox/mAP_75` 仍保留在详细指标中）。这样既保持指标可比性，也避免
每个 rank 重复累计 1024 分辨率 mask 导致主机内存 OOM。

## 坐标与梯度契约

- coarse mask：ROI-local，默认 `64×64`；
- PromptEncoder mask canvas：full-image；legacy 32×32 image embedding 对应
  `128×128`，官方64×64变体对应 `256×256`；
- 最终 SAM2 mask logits：B0/B1 保留旧 ROI-local target + bbox paste 复现口径；
  M0/M1 与 C1–C5 使用 SAM2 原生 full-image 网格监督，推理只 resize 到图像尺寸一次；
  validation 与 checkpoint 推理按 checkpoint 中的坐标配置共用对应路径；
- 训练 prompt box：Shared2FC 正样本 proposal；
- 推理 prompt box：最终检测框；
- 2P2N 点挖掘对 coarse logits 使用 stop-gradient；
- fixed 2P2N 强制四点互斥；warm-up 在训练与验证使用同一 epoch 阶段；无效槽在
  PromptEncoder 后显式置零，整批无有效点时直接使用空 sparse prompt；
- dense mask 经冻结 PromptEncoder 时不使用 `torch.no_grad()`，因此最终 mask
  loss 仍可回传到 coarse head 和 P2BoundaryRefiner；
- raw coarse 保留独立 `0.10*(BCE+Dice)`；refined coarse 是 2P2N 和 dense canvas
  的来源，P2BoundaryRefiner 另受权重 `0.05` 的局部边界 BCE 监督。
  dense embedding 的变化被限制在 proposal box 内，box 外保持 no-mask 基底。
