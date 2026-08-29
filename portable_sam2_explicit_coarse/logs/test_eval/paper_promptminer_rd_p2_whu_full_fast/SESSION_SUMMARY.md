# 会话总结：fast-150 提指标 run（2026-08-26 ~ 08-29，主机 lthpc）

目标从"复现论文"转向"提升指标"：在统一 maxDet=100 口径下训练
150-epoch AMP 版模型并超越论文复现基线。**已达成：test segm/mAP
0.7342 vs 基线 0.7288（+0.54），高 IoU 档与小目标收益最大。**

## 1. 配置演进（本会话 10 个 commit）

| 分组 | commit | 内容 |
|---|---|---|
| 评估口径 | `1f4e79c` | TEST_MAX_PER_IMG 150→100，全项目统一 COCO 默认口径；移除冗余的 VAL_COMPAT 双口径 |
| 训练配方 | `fe06405` | EMA 影子跟踪（EMA_ENABLED=1/EVAL=0）+ best-only checkpoint 保留（原子写入+读回校验+删旧留新为 trainer 既有行为）；禁用 bbox-best；runner `${VAR:-}`→`${VAR-}` 允许显式空值 |
| 机器配置 | `ca0e9b4` | **仅限 lthpc**：environment2.sh（environment.local.sh 符号链接激活，gitignored 不随仓库迁移，其他机器自动回退 environment.sh） |
| 训练配方 | `95bd20b` | 增强环境变量化（TRAIN_VFLIP_PROB/TRAIN_MULTI_SCALE_*）；fast 接入 vflip 0.5 + 多尺度 1024±12.5% p=0.5（value 模式五档，固定 1024 画布、SAM2 输入不变）；random_erasing 保持禁用（实现有标签缺陷） |
| 并行扩展 | `3655034` | 2 卡→4 卡（0,1,2,3），有效 batch 8 不变 |
| 启动修复 | `da92f6f` | e21479c 引入的 git 状态采集在干净树下静默退出（裸 grep + pipefail），首次干净树启动触发 |
| 启动修复 | `20505d8` | 24G 卡 OOM：batch 2（128 ROI 解码器前向 4.3GB/张量，40G 卡验证档位）不可迁移；改 batch1×accum2，加 expandable_segments |
| 事后修复 | `7d5f538` | EMA 冻结张量舍入漂移（55200 步 ~1e-6）触发 no_mask_embed 防篡改契约；冻结张量改直接拷贝；附 eval_test_winner.sh |

另：`83ed0e9`/`be91517` 为结果归档（REPORT.md + 全量 metrics json）。

## 2. 训练运行标识

- 入口/时间：`paper_promptminer_rd_p2_whu_full_fast.sh`，08-26 23:41 → 08-29 10:10（58.5h，零事故）
- 训练日志：`logs/ablations/paper_promptminer_rd_p2_whu_full_fast_tr1.0_va1.0_20260826_234149_pid41629.log`（git_commit 钉定 20505d8）
- checkpoint：`/data/wangcheng/.../paper_promptminer_rd_p2_whu_full_fast_tr1.0_va1.0/`
  - `best_model_epoch150.pth`（raw + EMA 影子，1.2G）
  - `best_model_epoch150_ema_merged.pth`（冻结张量回贴 raw 的 EMA 合并版，1.2G）
  - `last_checkpoint.pth`（1.2G）
- 最终协议：4×3090，batch 1×accum 2（有效 8），AMP fp16，150ep cosine，
  maxDet 100 选优，早停 20/90 未触发（best 恰为 epoch 150）

## 3. 结果

验证集（best=ep150）：raw 0.7493 / EMA 0.7499（EMA 微胜）；两套权重
test 完全等价（差异 ≤0.0005）——best 为末 epoch、LR→5e-7、EMA 已收敛
至 raw 所致；EMA 价值在未来"中途取 best"的 run。

test（2220 图 / 70,063 实例，maxDet=100，raw 与 EMA 同值）：

| 指标 | 论文复现(ep98) | 本 run(ep150) | Δ |
|---|---|---|---|
| segm/mAP | 0.7288 | **0.7342** | **+0.54** |
| segm/mAP_75 | 0.8402 | 0.8546 | +1.44 |
| segm/mAP_s | 0.4354 | 0.4502 | +1.47 |
| bbox/mAP | 0.7553 | 0.7644 | +0.90 |

高 IoU/小目标收益与 P2-BRR 边界精修方向一致。论文对比须注明配方差异：
AMP fp16、150ep、vflip+多尺度增强（REPORT.md §2 有完整差异表）。

## 4. 过程事件记录（已修复，留档备查）

1. 干净树启动静默退出（`da92f6f`）：失败日志 233452/233635 可复现现场
2. 24G OOM（`20505d8`）：失败日志 233807 含 128-ROI 解码器 OOM 栈
3. EMA 权重推理被契约拒绝（`7d5f538`）：eval_raw_vs_ema_val 日志含拒绝栈；
   本次 run 用合并修正版绕过，未来 run 可直接 `--weights ema`

## 5. 遗留与建议

- [ ] `machine2/fast-whu150` 分支 12 个 commit（cc68988 之后）尚未 push
- [ ] 可选：test 掩码可视化（visualize_instances.py --mask-only，与对比算法口径一致）
- [ ] 可选：清理 last_checkpoint.pth（-1.2G）——best 已定格，续训场景已不存在
- EMA 结论：本次等价、无损失；若未来 run 的 best 出现在中段，EMA 影子
  （已默认开启于 fast 协议）值得事后对比
