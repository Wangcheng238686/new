# 旧版模型训练与推理指标归档（paper_promptminer_rd_p2_whu_full）

> 归档时间：2026-08-25。本文档记录 100-epoch 复现实验（旧版）的完整训练/推理指标，
> 所有日志路径相对 `portable_sam2_explicit_coarse/`。

## 1. 代码版本（可追溯性）

| 阶段 | git 状态 | 说明 |
|---|---|---|
| 首次启动（08-22 16:52） | `4ffd61d` | epoch 1 起 |
| 续训重启（08-24 23:27） | `5fff72f` + 工作区未提交改动 | `5fff72f`（"续训暂存状态"）之上恢复了 `rsprompter/models.py` 等训练代码（相对 `3b456c6` 精简版的还原），训练全程未再变更 |
| **实际执行代码的固化 commit** | **`machine2/fast-whu150` 分支 `c82cdd1`** | "baseline: 固化续训工作区代码状态"，即续训进程实际运行的代码逐字快照 |

训练日志中的记录：首段 `git_commit=4ffd61d4...`，续训段 `git_commit=5fff72fe...`。

## 2. 训练协议

- 架构：`c5v2_pafpn_coarse_p2_boundary_refiner_emb64`（PAFPN + stride16/emb64 + points_box_dense + raw-detach dense + P2-BRR），架构指纹 `36549a00a828...`
- 数据：WHU 全量训练集（2943 图，tr1.0），验证集 619 图（va1.0），seed=44
- 优化：lr 5e-4，cosine（T_max=36800 optimizer steps），warmup 100 步，weight_decay 0.05，有效 batch 8（1/rank × 2 GPU × accum 4），fp32（AMP 关）
- 监督：mask_size=1024（旧协议），maxDet=100 逐 epoch 验证选 best
- 早停：patience=10 / start=20 / smoothed(5)，**全程未触发**

## 3. 训练指标（验证集，maxDet=100）

| 里程碑 | epoch | segm/mAP |
|---|---|---|
| 起点 | 1 | 0.4489 |
| 破 0.70 | 27 | 0.7041 |
| 进入退火段后 | 57 | 0.7302 |
| **best（最终采用）** | **98** | **0.7417** |

- best bbox/mAP = 0.7779（epoch 81）
- checkpoint：`/data1/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/paper_promptminer_rd_p2_whu_full_tr1.0_va1.0/`（`best_model.pth` → epoch98；`best_bbox_model.pth` → epoch81；`last_checkpoint.pth` → epoch100）

## 4. 测试集推理指标（held-out test，2220 图 / 70,063 实例，epoch98 best segm 权重）

### segm

| 指标 | maxDet=100 | maxDet=150 |
|---|---|---|
| AP (mAP) | 0.7288 | **0.7440** |
| AP@50 / AP@75 | 0.9005 / 0.8402 | 0.9194 / 0.8580 |
| AP s / m / l | 0.4354 / 0.7785 / 0.8107 | 0.4748 / 0.7879 / 0.8136 |
| AR@cap / AR s | 0.7689 / 0.5148 | 0.7854 / 0.5676 |

### bbox

| 指标 | maxDet=100 | maxDet=150 |
|---|---|---|
| AP (mAP) | 0.7553 | **0.7705** |
| AP@50 / AP@75 | 0.8995 / 0.8389 | 0.9184 / 0.8568 |
| AP s / m / l | 0.4457 / 0.7996 / 0.8619 | 0.4870 / 0.8098 / 0.8619 |
| AR@cap / AR s | 0.7970 / 0.5280 | 0.8141 / 0.5837 |

完整 12 项统计见本目录 `test_metrics_md100.txt` / `test_metrics_md150.txt`（及同名 `.json`）。

**口径说明**：验证集 best 0.7417 是 maxDet=100 且 best 在验证集上选取（带乐观偏置）；
测试集同口径 0.7288，差 1.3 点属正常选型偏差。maxDet 100→150 的提升集中在小目标
（AP-small +3.9~4.1，large 基本不变），源于密集图（>100 实例/图）recall 释放。
测试集比验证集更密集（31.8 vs 25.7 实例/图）。

辅助历史数据：epoch83 权重 @100 = 0.7285；epoch90 权重 @150 = 0.7427（见第 6 节日志）。

## 5. 指标与可视化文件（本目录）

| 文件 | 内容 |
|---|---|
| `test_metrics_md100.json` / `.txt` | maxDet=100 完整指标（含口径元数据） |
| `test_metrics_md150.json` / `.txt` | maxDet=150 完整指标 |
| `inference_log_md100.log` / `inference_log_md150.log` | 两次推理的完整日志副本 |

可视化（掩码叠加，各 50 张）：`logs/viz/testonly_e98_md100/images/`、`logs/viz/testonly_e98_md150/images/`；
对应 `predictions.json` 在各自上级目录。

## 6. 原始日志索引（logs/ablations/）

| 日志 | 内容 |
|---|---|
| `paper_promptminer_rd_p2_whu_full_tr1.0_va1.0_20260822_165203_pid153415.log` | 训练首段（epoch 1 起，4ffd61d） |
| `paper_promptminer_rd_p2_whu_full_tr1.0_va1.0_20260824_232758_pid752347.log` | 训练续训段至 100 epoch 完成 |
| `testonly_e98_md100_tr1.0_va1.0_*.log` | epoch98 @maxDet100 推理（完整指标+可视化） |
| `testonly_e98_md150_tr1.0_va1.0_*.log` | epoch98 @maxDet150 推理（完整指标+可视化） |
| `testonly_best_model_epoch98_tr1.0_va1.0_20260825_190453_pid995766.log` | epoch98 @maxDet100 推理（旧代码，无逐项指标） |
| `testeval_promptminer_rd_p2_*.log`（08-25 上午） | epoch83 @100 / @150 早期口径探索 |
| `testeval_fixcheck3_*.log` | epoch90 @150（maxDets 重算修复验证） |

## 7. 后续

加速版 150-epoch 复跑入口：`scripts/ablations/paper_promptminer_rd_p2_whu_full_fast.sh`
（分支 `machine2/fast-whu150`，AMP/mask256/batch2×2/val2/benchmark，best 选择 @maxDet150 +
@100 兼容曲线）。本目录由 commit 固化，供新旧结果对账。
