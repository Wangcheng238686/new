# Portable SAM2 Explicit Coarse

这是一个独立派生项目，目标是在已验证的 WHU-1024 / SAM2 Base+ 基线上，
干净地集成以下四项：

1. SAM2 四层特征上的 PAFPN；
2. 显式、ROI-local、带独立监督的 coarse mask；
3. 从同型号 SAM2 checkpoint 严格加载并冻结的原生 PromptEncoder；
4. PromptEncoder 之前、ROI-local logit 空间内的 DenseBR。

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

# 默认：PAFPN + points+box+dense，DenseBR 关闭
bash scripts/run_whu1024_explicit_coarse_4gpu.sh

# 开启 DenseBR
DENSEBR_ENABLED=1 bash scripts/run_whu1024_explicit_coarse_4gpu.sh

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

当前 B0–C5 全部消融路线均支持该推理入口。后续所有影响模型结构、模块开关、
张量形状或 forward/predict 行为的新参数，都必须由配置解析进 `cfg.model`，确保
checkpoint 能完整记录并由推理器无歧义重建；训练、数据和运行时参数分别保存在
`training_args`、`data_config` 和 `runtime_config`。

## 消融矩阵

消融入口统一放在
`portable_sam2_explicit_coarse/scripts/ablations/`，覆盖 aggregator/PAFPN、
旧 MLP/coarse、box、dense prompt 和 DenseBR。每次启动前都会检查解析后的配置
是否与脚本声明一致。

```bash
cd portable_sam2_explicit_coarse

# 全矩阵 smoke；不会开始训练
bash scripts/ablations/smoke_all.sh

# 消融默认使用 10% train / 100% validation
bash scripts/ablations/c4_pafpn_coarse_points_box_dense.sh

# 比例仍可独立覆盖；全量训练需显式设置 TRAIN_SUBSET_RATIO=1.0
TRAIN_SUBSET_RATIO=1.0 VAL_SUBSET_RATIO=1.0 \
  bash scripts/ablations/c4_pafpn_coarse_points_box_dense.sh
```

矩阵定义、参数覆盖和 dry-run 用法见
`portable_sam2_explicit_coarse/scripts/ablations/README.md`。

每个消融入口会把 stdout/stderr 同时输出到终端并自动保存到项目内
`portable_sam2_explicit_coarse/logs/ablations/`。日志开头包含解析后的架构、
数据、优化器、EMA、初始化/续训路径、有效全局 batch size、git commit 和完整
torchrun 命令。可用 `LOG_DIR` 覆盖日志目录，或用 `LOG_FILE` 指定精确文件。

全部新主线实验的 validation 默认保持 WHU1024 历史基线口径：`batch_size=1`，
四卡 DDP 只由 rank 0 遍历完整验证集，其余 rank 等待同步；空 GT 图像仍进入
COCO 评估，bbox/segm 都按 detector score 排序。这样既保持指标可比性，也避免
每个 rank 重复累计 1024 分辨率 mask 导致主机内存 OOM。

## 坐标与梯度契约

- coarse mask：ROI-local，默认 `64×64`；
- PromptEncoder mask canvas：full-image，1024 输入时为 `128×128`；
- 最终 SAM2 mask：full-image low-resolution mask；
- 训练 prompt box：Shared2FC 正样本 proposal；
- 推理 prompt box：最终检测框；
- 2P2N 点挖掘对 coarse logits 使用 stop-gradient；
- dense mask 经冻结 PromptEncoder 时不使用 `torch.no_grad()`，因此最终 mask
  loss 仍可回传到 coarse head 和 DenseBR；
- DenseBR 输出是 coarse loss、2P2N 和 dense canvas 的唯一 coarse-logit 来源。
