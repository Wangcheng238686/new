# R3 matrix300 结果诊断

日期：2026-09-11。状态：R3 v1 **不通过** dense-source 采纳门；不修改既有
P/PB/A0/A3 结果，也不据此启动重训。

## 同协议结果

所有数值来自 NWPU/VHR-10 matrix300、4 GPU、300 epoch、val/5、last E300。

| arm | segm/mAP | bbox/mAP | composite | best segm/mAP |
|---|---:|---:|---:|---:|
| PB | 0.6395 | 0.7215 | 0.6805 | 0.6530 |
| A0/PBM | 0.6330 | 0.7218 | 0.6774 | 0.6447 |
| R3/PB+R3 | 0.6273 | 0.7127 | 0.6700 | 0.6434 |

故 R3 对 A0 的 last 差为 -0.0057 segm/mAP、-0.0091 bbox/mAP；对 PB 的
last 差为 -0.0122 segm/mAP。单 seed 结果不是显著性检验，但已明确不满足预注册的
正向采纳预期。

## 冻结 canvas probe

R3 last checkpoint 的标准推理由 probe parity hash 验证，matched/unmatched RoI 为
707/54。与同口径 A0 last 比较：

| checkpoint | learned canvas IoU | current gate | learned mAP | matched GT canvas @ current | Δ |
|---|---:|---:|---:|---:|---:|
| A0 last | 0.7821 | 0.6738 | 0.6331 | 0.6463 | +0.0132 |
| R3 last | 0.7614 | 0.6234 | 0.6271 | 0.6364 | +0.0092 |

R3 既未达到设计时约 0.87 的探索性内容标尺，也低于旧 coarse canvas。将 R3
gate 强制为 1.0 会使 learned mAP 再降 0.00146，但 matched GT canvas + gate=1
可达 0.6415（相对 learned/current +0.01433）。结论是：当前全局 gate 正在抑制
不可靠 R3 canvas；问题主要是内容质量，不是 gate 没有打开。

## 原因排序与结论边界

1. **已证实主因：R3 dense source 未优于旧 coarse。** R3 正 RoI 的 train target
   可拟合（E300 BCE=0.0889、Dice loss=0.0543），但 validation matched delivered
   IoU 仍低；这是 train positive ROI 到 inference proposal 的泛化/表示问题，不是
   loss 未接通。
2. **高风险设计缺口：训练只监督 positive RoI。** 推理为所有 detection RoI 渲染，
   54 个 unmatched RoI 没有 canvas background 监督；当前 probe 只量化其数量，不能
   证明其安全性。
3. **遥测缺口：** `dense_*` prompt-pathway 组当前只覆盖 ShapePriorInjector+dense
   gate，不含 `canvas_renderer`。R3 日志中的低 dense final-mask gradient 因而不能
   判为 renderer 无最终任务梯度；R3 参数确实在 `sat_other` optimizer group 内更新，
   但尚无独立 renderer gradient/update telemetry。

因此不能写“R3 为 dense 通道提供了比 coarse 更好的信息”，也不能把它作为论文创新
主结论。下一步应先做只读/冻结分解（matched vs unmatched canvas occupancy、canvas
错配/置零依赖 probe、R3 renderer final-mask-gradient telemetry），再决定是否另立
R3 v2；不得把 gate=1 或新增启发式 fallback 直接当作修复。
