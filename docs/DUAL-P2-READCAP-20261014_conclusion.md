# DUAL-P2-READCAP-20261014 结论：P2_NULL（语义细分：融合表示瓶颈，几何信号增强）

## 完整性

正式 run 于 2026-10-14 执行（HEAD `8707db3`，探针实现经独立首次定版审计
APPROVED + 增量复审 APPROVED，见
`docs/reviews/DUAL-P2-READCAP_implementation_review.md`）。输入为 P1-V2 R1a
冻结缓存（contract SHA `af04fc44…`，只读），fit 555 / select 139 / val 100
一次性判定，test 未读取（P1-V2 一次性 lock 不受影响）。工件：
`/data1/wangcheng/checkpoint/surfbev_b0/probes/DUAL-P2-READCAP-20261014/`
（split/selection/telemetry_fit/summary/per_scene/run.log + 增补 posthoc）。
预注册勘误（判定格补全、种子登记）在任何 val 结果产生前完成。

## 判定

**P2_NULL**（勘误判定格 (0,0,0)：三对门均未成对通过）。但语义必须细分：

| 差值（MLP 读出，val 100 场） | mean | 95% CI | 门 |
| --- | ---: | --- | --- |
| correct−SAT F1 | −0.245 | [−0.262, −0.229] | ✗ |
| correct−sham90 F1 | +0.172 | [+0.155, +0.189] | ✓ |
| correct−SAT AUROC | −0.117 | [−0.125, −0.108] | ✗ |
| correct−sham90 AUROC | +0.084 | [+0.079, +0.090] | ✓ |
| correct−SAT Spearman | −0.199 | [−0.246, −0.151] | ✗ |
| correct−sham90 Spearman | +0.090 | [+0.040, +0.137] | ✓ |

- **correct−sham90 三门首次全部通过**（P1-V2 线性下 rho CI 跨 0、auc 门挂）：
  非线性读出揭示的几何特异信息比线性可见的更强。
- **correct−SAT 三门大负**：与 P1-V2 线性（+0.007 F1）方向反转。
- parity 哨兵通过：SAT 臂 MLP F1 0.812 vs 线性 0.672（容量假设在 SAT 臂成立，
  非 READCAP_DEGRADED）。

## 解释（含 post-hoc 登记）

组合臂特征经过第二次随机投影（[SAT64+BEV120]→64），把 SAT 已投影结构再次
搅动；MLP 能从单投影 SAT 特征读出 0.81 F1 的非线性结构，但该结构在融合臂的
二次投影中被破坏，BEV 信息补不回损失。**post-hoc（非门，仅解释）**：各臂在
val 上各自最优阈值下排序不变（MLP：SAT 0.814 / correct 0.563 / sham90
0.412；线性：0.672 / 0.573 / 0.508），排除公共阈值伪影。
`posthoc_threshold_scan.py`/`posthoc_threshold_scan.json` 同工件目录。

## 结论边界

三项仍是冻结特征可读性度量，非 mask/bbox AP。本结论不主张"UAV 无用"或
"BEV 无信息"——恰相反：correct−sham 三门全过是至今最强的几何特异证据；挂掉
的是**融合表示**（随机二次投影）而非信息本身。不修改 P1-V2 任何门或结论。

## 下一步（按 P2_NULL 判定表）

判定表要求"改表示，禁止在此读出上调参"。下一步 DUAL-P3：**旁路表示读出**——
读出输入改为 `[x_SAT ‖ x_correct]`（纯 SAT 64 维与融合 64 维并排拼接，
128 维），特征本体仍为同一冻结缓存（视为判定表中"分通道"的读出侧实现；
若 P3 亦 NULL，则必须重新 harvest 升维/分通道，不再允许读出侧动作）。
P3 直接检验：无损保留 SAT 通道时，BEV 信息能否让非线性读出全面获益——
这是"learned 双流融合模块应保留流身份"设计原则的冻结版检验。
