# DUAL-P3-SEPSTR-20261014 结论：P3_NULL（语义细分：表示修复成功，BEV 边际信号微弱且实例质量为负）

## 完整性

正式 run 于 2026-10-14 执行（HEAD `882b384`；实现经增量审计 APPROVED——
concat_arm 与预注册一致、其余协议与已审 P2 实现逐字一致、池化等价性成立、
无新遮蔽/泄漏；审计对话归档于增补节）。输入为同一 P1-V2 R1a 冻结缓存
（contract SHA `af04fc44…`），fit 555/select 139/val 100 一次性判定，
test 未读取。工件：
`/data1/wangcheng/checkpoint/surfbev_b0/probes/DUAL-P3-SEPSTR-20261014/`。

## 判定与数字

**P3_NULL**（(0,0,0)：三对门均未成对通过）。

| 差值（MLP，val 100） | mean | 95% CI | 门 |
| --- | ---: | --- | --- |
| correct−SAT F1 | −0.013 | [−0.017, −0.009] | ✗ |
| correct−sham90 F1 | +0.015 | [+0.013, +0.017] | ✓ |
| correct−SAT AUROC | −0.003 | [−0.004, −0.003] | ✗ |
| correct−sham90 AUROC | +0.003 | [+0.003, +0.004] | ✓ |
| correct−SAT Spearman | −0.074 | [−0.116, −0.034] | ✗ |
| correct−sham90 Spearman | −0.043 | [−0.083, −0.004] | ✗ |

三臂 MLP：SAT f1/auc/rho = 0.812/0.981/0.381；correct = 0.799/0.977/0.307；
sham90 = 0.784/0.974/0.350。parity 哨兵通过（SAT MLP 0.812 ≥ 线性 0.661）。

## 解释

1. **P2 的表示诊断被证实**：旁路拼接后 correct 臂 F1 从 P2 的 0.567 恢复到
   0.799——二次随机投影确实是 P2 中 correct−SAT 大负的机制。
2. **几何特异信号在健康表示下仍在，但很弱**：boundary F1 与 AUROC 的
   correct−sham90 门过（+0.015 / +0.003）——三次探针（P1-V2 线性、P2、P3）
   跨表示一致为正，方向稳定。
3. **实例质量预测 BEV 为负贡献**：correct−SAT rho −0.074 且 **correct−sham90
   rho 也为负**（正确几何比假几何更伤实例质量预测）——该 BEV 描述子在 cell
   级不含可用实例质量信息，且其额外结构引入的是过拟合噪声。
4. correct 从未在任何指标上超过 SAT：读出层面 BEV 的边际价值 ≤ +0.015 F1。

## 结论边界

三项仍是冻结特征可读性度量非 AP；本结论只关闭"当前 64 维融合缓存上的读出
调参"（P3 预注册条款），不声称 SurfaceBEV 方向终结。

## 下一步（按 P3 判定表）

读出侧动作用尽 → **DUAL-P4：重新 harvest，真分通道**——val/train 逐场落盘
纯通道特征（SAT 单投影升维 + BEV 120 维原始通道不与 SAT 混合），在无二次
投影、无维度上限污染的表示上重跑同协议读出。这是该描述子读出路线的最后
一个未排除混杂（投影维度上限）；P4 若在 boundary 门仍 NULL，则关闭
"cell 级 SurfaceBEV 描述子"路线，双流机制转向 RoI 级多视角条件化并另立
预注册族。
