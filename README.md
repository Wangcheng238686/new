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

## 消融矩阵

消融入口统一放在
`portable_sam2_explicit_coarse/scripts/ablations/`，覆盖 aggregator/PAFPN、
旧 MLP/coarse、box、dense prompt 和 DenseBR。每次启动前都会检查解析后的配置
是否与脚本声明一致。

```bash
cd portable_sam2_explicit_coarse

# 全矩阵 smoke；不会开始训练
bash scripts/ablations/smoke_all.sh

# 默认使用完整 train/validation；也可以独立抽取确定性子集
TRAIN_SUBSET_RATIO=0.1 VAL_SUBSET_RATIO=0.2 \
  bash scripts/ablations/c4_pafpn_coarse_points_box_dense.sh
```

矩阵定义、参数覆盖和 dry-run 用法见
`portable_sam2_explicit_coarse/scripts/ablations/README.md`。

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
