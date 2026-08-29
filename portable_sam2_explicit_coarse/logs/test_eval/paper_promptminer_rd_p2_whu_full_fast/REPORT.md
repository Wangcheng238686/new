# fast-150（AMP + 增强 + EMA）训练与推理指标归档

## 1. 运行标识

- 分支/HEAD：`machine2/fast-whu150` @ `7d5f538`（训练时钉在 `20505d8`，日志 git_commit 可追溯）
- 入口：`scripts/ablations/paper_promptminer_rd_p2_whu_full_fast.sh`
- 训练日志：`logs/ablations/paper_promptminer_rd_p2_whu_full_fast_tr1.0_va1.0_20260826_234149_pid41629.log`
- checkpoint：`/data/wangcheng/checkpoint/.../paper_promptminer_rd_p2_whu_full_fast_tr1.0_va1.0/`
  （best_model_epoch150.pth + last_checkpoint.pth，best-only 保留）

## 2. 与论文复现 run 的协议差异（对比时须注明）

| 项 | 论文复现 run | 本 run |
|---|---|---|
| 精度 | fp32 | AMP fp16（18ep A/B 均值 \|ΔmAP\|=0.021） |
| 调度 | 100ep cosine，2 卡 batch1×4 | 150ep cosine，4 卡 batch1×2（有效 batch 均 8） |
| 增强 | hflip 0.5 | hflip 0.5 + vflip 0.5 + 多尺度(1024±12.5%, p=0.5, value 模式) |
| 选优口径 | maxDet 100 | maxDet 100（统一） |
| EMA | 无 | 影子跟踪（EMA_EVAL=0），事后对比选优 |
| 早停 | 未触发（ep98 best） | 未触发（ep150 best，恰好最后一 epoch） |

## 3. 验证集对比（best=epoch150，同 checkpoint 两套权重）

| 指标 | raw | EMA（合并修正版） |
|---|---|---|
| segm/mAP | 0.7493 | **0.7499**（胜出，+0.0006） |
| bbox/mAP | 0.7860 | 0.7860 |

EMA 说明：best 恰为末 epoch（LR→5e-7），EMA 已收敛至 raw，故差异极小；
EMA 权重加载触发 no_mask_embed 契约（冻结张量舍入漂移 1e-6），用合并修正
checkpoint（best_model_epoch150_ema_merged.pth：冻结张量回贴 raw、可训练
张量保留 EMA）评估。EMA 类已在 `7d5f538` 修复（冻结张量直接拷贝）。

## 4. 测试集（held-out test，2220 图 / 70,063 实例，@maxDet100）

| 指标 | 论文复现 run (ep98) | raw (ep150) | EMA 合并版 (ep150) |
|---|---|---|---|
| **segm/mAP** | 0.7288 | **0.7342** | **0.7342** |
| segm/mAP_50 | 0.9005 | 0.9131 | 0.9131 |
| segm/mAP_75 | 0.8402 | 0.8545 | 0.8546 |
| segm/mAP_s | 0.4354 | 0.4502 | 0.4501 |
| segm/mAP_l | — | 0.8123 | 0.8118 |
| segm/AR@100 | 0.7689 | 0.7680 | 0.7680 |
| bbox/mAP | 0.7553 | 0.7644 | 0.7643 |

raw 与 EMA 在 test 上差异 ≤0.0005（mAP_l），主指标四舍五入后同值——
best 恰为末 epoch（LR→5e-7）、EMA 已收敛至 raw 所致；验证集 tie-break
（0.7499 vs 0.7493）名义上 EMA 胜出，两套权重任选其一出报告均可，
本文档以 EMA 合并版为归档胜者。

## 5. 结论

AMP + 150ep + vflip/多尺度增强 + EMA 影子的组合在 test 集 segm/mAP 上
净胜 +0.54 个点（bbox +0.90），高 IoU 档（mAP_75 +1.44）与小目标
（mAP_s +1.47）收益最大——边界质量提升与 P2-BRR 的边界精修方向一致。
文件：`test_metrics_ema_md100.json`（test）、`val_metrics_{raw,ema_merged}.json`（验证）。
