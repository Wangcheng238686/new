# R3 前置探针：canvas 通道 oracle 读数（matrix300 A0 E300 权重）

日期：2026-09-10。权重：`vhr10_p2v2_matrix300_a0_pbm_d5b_tr1.0_va1.0/last_model_epoch300.pth`。
工具：`inference/probes/canvas_accuracy_probe.py`（本日按实现审计修复 P0+3×P1 后运行）。
产物：`refactor_golden/canvas_probe_a0_matrix300.json` +
`diagnostics/a0_matrix300_canvas_oracle/`（7 pass，含逐 cell gt/dt/manifest）+
`diagnostics/a0_matrix300_canvas_oracle/bootstrap10cls_support_current_vs_learned.json`。

## 读数（segm/mAP，冻结权重、matched ROI 714/767=93%）

| cell | mAP | Δ vs learned/current |
|---|---|---|
| learned, gate=0.674（=标准推理，逐位自检过） | 0.6331 | 0 |
| **support_matched(GT), gate=0.674** | 0.6463 | **+0.0132** |
| full_gt, gate=0.674 | 0.6468 | +0.0137 |
| learned, gate=1.0 | 0.6302 | −0.0029（噪声内，无门控增益） |
| support_matched(GT), gate=1.0 | 0.6498 | +0.0167 |
| full_gt, gate=1.0 | 0.6523 | +0.0192 |

配对 bootstrap（10 类契约，500 重采样，seed 44）：support_matched@current vs learned@current
点估计 +0.0132，**CI95 [+0.0082, +0.0187]，p(Δ>0)=1.0**。

画布精度（P0 修复后的口径：支撑域内、outside_fill 为背景）：IoU 均值 0.782 / 中位 0.793；
coarse 源 0.892 Dice（≈0.81 IoU 当量）→ 传输损耗同口径仅 ~0.02，**到 GT 总余量 ~0.22**。

## 结论审计（子代理 #2，判决 TRUSTWORTHY-WITH-CAVEATS）后的修正措辞

1. 数字全部核验：逐 cell mAP 与 metrics/manifest 逐位一致；审计者用仓库协议从原始 gt/dt
   独立重算 COCO 逐位复现；matched 714/767 独立重数吻合；门控 sigmoid(0.7254)=0.6738
   与 summary 逐位一致；logit_span=8.0 来自 clamp_range（±4 旧问题已修）。
2. **修正一（Dice≠IoU）**：coarse→画布的同口径传输损耗 ~0.02，不是 ~0.11（0.892 是 Dice，
   IoU 当量 0.81）。
3. **修正二（匹配率）**：728/918 是 09-07 dev100 旧数；本 run 714/767=93%，稀释比记忆中小。
4. **修正三（斜率）**：+0.0006/0.01IoU 是两点平均斜率；按线性外推过 +0.005 采纳线需要
   画布 IoU ≈ **0.87**（不是 0.85）。
5. oracle 是冻结权重上限，不是训练渲染器的预测；unmatched ROI（7%）的画布行为未测。

## R3 判读（Conditional GO：进入可行性阶段，不等于采纳）

支持：通道上限 +0.013~+0.019（CI 下界 +0.0082 仍过 +0.005 采纳线的 1.6 倍）；效果由
**内容**驱动而非门控（gate=1 单独 −0.003）；支撑域限制几乎免费（full−support +0.0005）；
方向跨两个 checkpoint 复现（dev100 E36 与 matrix300 E300）。
门槛（立项后须满足才谈采纳）：渲染器画布 IoU 目标 ≈0.87；unmatched-ROI 风险显式管理；
oracle 数仅作上限参考。附带背景：matrix300 三级矩阵中 dense 包对 PB 为 −0.005
（P 0.6258 / PB 0.6395 / A0 0.6342 平台），R3 是 dense 包翻盘的主要候选路径。

## 补充：A0 best（E105）权重上的同探针（2026-09-10 10:33，应用户指令即时执行）

产物：`refactor_golden/canvas_probe_a0_best.json` + `diagnostics/a0_best_canvas_oracle/`。

| | best E105 | last E300 |
|---|---|---|
| 标准 mAP（自检） | 0.6446 | 0.6331 |
| 画布 IoU（支撑域口径） | 0.766 | 0.782 |
| 当前门控 | 0.662 | 0.674 |
| GT 画布@当前门控 | **+0.0162** | +0.0132 |
| GT 画布@门控全开 | **+0.0245** | +0.0167 |
| 仅门控全开（画布不变） | **+0.0062** | −0.0029 |

两点解读：(1) **通道上限对权重选择稳健**（+0.013~+0.026，两个 checkpoint 都远超采纳线）；
(2) 门控-内容协同的方向随训练阶段翻转——E105 时开大门控单独就有 +0.006（内容尚新鲜、
门控欠开），E300 时变为 −0.003（内容封顶、门控追高）——与"依赖加深但画布质量停滞"的
主线一致，也提示 R3 若把内容做好，门控端还有额外可兑现的协同量。

## 运行事故记录（诚实账）

A3 双发（等待器半杀致孤儿脚本解锁+我手动并发）→ 杀一份；存活份 08:57 被探针共卡 OOM
打死（A3 每卡显存重于 a0，共卡先例不可迁移）；09:18 干净重挂 A3（无共卡纪律恢复）。
探针自身 09:00 完成，产物完整。
