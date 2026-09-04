# V2 Prompt Mining 改造实施方案 v2.1（代码审查修订版）

> 交接文档。v2.0 经代码级审查后修订：v2.0 的"128 画布+偏移+负点GT判据"一步式
> 方案因因果变量过多与代码不兼容被否决；本版采用审查方提议的小步序列，并追加
> 两个审查补充（S1.5 oracle 门、消费端语义风险控制）。
> 分支 `machine2/fast-whu150`，主机 lthpc（4×RTX 3090）。
> 背景证据（通路审计、decoder 只听 sparse、负点 17% 误落、airplane 阴影复合
> 标注）见 §0——与 v2.0 相同，此处不重复，实施前必读 v2.0 §0 或本文附录 A。

## 0. v2.1 相对 v2.0 的关键修正（为什么改）

| v2.0 的错误 | 证据（审查确认） | v2.1 的处理 |
|---|---|---|
| 负点"GT 外 ε 判据"直接换采样规则 | 推理无 GT，会造 train-inference gap | GT 只作训练期 teacher/损失；推理全程预测驱动 |
| 对硬挖点加 hinge 即可降误落率 | miner 在 no_grad 下输出整数索引，损失无法修正点位 | 需连续 offset 分支 + grid_sample 在 GT 合法区图上取损失；hinge 只能作统计 |
| 点损失在 PromptEncoder 前实现 | coarse GT 现在在 mask forward 后才构造 | 先把 jittered-ROI 对齐的 coarse target 前移到 _mask_forward 前并复用 |
| "64→128=stride 升档" | coarse 是 ROI-local，SmallMaskDecoder 上采样后插值到 64；改 128 只是改插值栅格 | 128 臂改为 bilinear(raw64)+zero-init residual（读更高频 RoI/P2 特征） |
| 恒等初始化=上采样旧权重 | 不可执行 | 恒等性由 bilinear(raw64)+zero-init residual 保证 |
| P2BR "重新接线" | P2BR 硬编码 coarse_size=64，改画布必然牵连 | 128 臂显式关旧 P2BR，单独消融，不与画布收益混淆 |
| 正点移入边界带混入主臂 | 与"锚点不变"矛盾且改变 decoder 共适应的 prompt 语言 | 独立联合训练臂 |
| STE 预案 | 不需要：PromptEncoder 接受浮点坐标，全程可微 | 删除 STE；锚点 stop-gradient，偏移 clamp 到有效槽位/图界 |
| dense α 上限 0.5 | ft200 的 α 可能已 >0.5，加载即改变基线，破坏恒等起点 | 先记录 checkpoint 实际 α；限幅仅作为独立变体臂 |
| 80ep 只对比 ft200 历史最佳 0.6617 | 混淆训练时长与干预效应 | 必须加 C0 对照臂（同初始化/同 80ep/同 seed 的原架构续训） |
| G5 换点掉分=证明方案有效 | 只证明依赖性 | 主证据=新臂 vs C0 配对差 + bootstrap CI；换点仅作依赖性证据 |
| 阶段 0 死通路清理优先 | 现配置本就 proposal box + jitter_prob=0（已是关闭态） | 降级为记账项，不在关键路径 |

## 1. 实施序列（v2.1 主线）

### S1 coarse GT 前移与复用（前置工程，无训练）
把 jittered-ROI 对齐的 coarse target 构造从 mask forward 之后前移到 _mask_forward
之前，同一份 target 复用于：coarse loss、点语义损失（后续步骤）、DEBUG 统计。
验收：损失数值与现实现逐位一致（开关切换零差异）。

### S1.5（审查补充）推理期 oracle 负点门（一次推理，零训练）
在 ft200 best 上做推理期实验：将被 GT 判定非法的负点（~17%）替换为 GT 合法
负点（仅在推理时替换，模型不动），测 val mAP / airplane 逐类变化。
- 依据：8/24 的 bolt-on 失败针对的是**新增 token**；本实验是**移动既有 token 的
  位置**，扰动性质温和得多，未被既有证据覆盖；
- 判定：ΔmAP ≥ +0.3 或 airplane ≥ +3pt → 负点链条有实测天花板，进入 S2；
  Δ≈0 或为负 → decoder 对负点位置不敏感，**负点链条降级**，主线转 S3 偏移头。
- 这是全方案最便宜的 go/no-go，必须在任何训练投入前完成。

### S2 消费端训练：PromptRobustifier 臂（+C0 对照）
- 训练期：clean 点 + GT 语义保护（非法负点替换为合法位/或按 GT 语义重标），
  固定 2P2N，**推理完全预测驱动**（miner 不动）；
- C0：同 ft200 初始化、同 80ep、同 seed/数据/优化器的原架构续训；
- **（审查补充）语义风险控制**：decoder 学会"信任负点"后，推理期 miner 仍产生
  17% 非法负点，可能放大而非缩小 train-inference gap。因此配对评估必须同时报：
  (a) 训练臂 vs C0（miner 推理点，主证据）；(b) 两臂各自喂 oracle 合法负点的
  推理差（量化 gap 与消费端就绪度）。
- 验收：新臂 vs C0 配对差（逐图配对 + bootstrap 95% CI）> 0 且 (b) 中 gap 收窄。

### S3 有界连续偏移头（仅当 S1.5/S2 链条为正）
- 锚点=现启发式（stop-gradient）；偏移头从 64 coarse/P2 RoI 特征回归连续偏移，
  zero-init、clamp（槽位语义与图界内）；GT 合法区/signed-distance 图上用
  grid_sample 对连续坐标取损失；掩码损失端到端流经坐标（无需 STE）；
- 首版不做 128。验收：误落率 17%→<2%（可训练口径）；配对差同 S2。

### S4 128 residual coarse（独立因素，最后做）
- coarse128 = bilinear(raw64) + zero-init residual（residual 读更高频 RoI/P2 特征）；
- 旧 P2BR 显式关闭（coarse_size=64 硬编码耦合），与"P2BR-off + 64"C0 做正交消融；
- α 限幅（若做）：独立变体臂，先记录 ft200 实际 α。
- 正点入边界带：独立臂，绝不混入恒等主臂。

## 2. 统一实验规范
1. 每个训练臂必配 C0（同初始化/同长/同 seed）；主证据=臂 vs C0 逐图配对差 +
   bootstrap CI；oracle/换点类推理实验仅作依赖性与天花板证据；
2. DEBUG 新统计分清 rank-local sum/count 与 DDP epoch ratio（现有 DEBUG-SP 是
   detached 诊断，不可当训练监督）；
3. 恒等初始化验收：加载后续训首 epoch 指标 ≈ 基线（ft200 先例：0.6399 vs 0.6417）；
4. 其余协议、仓库规范、工具坐标、审查节点沿用 v2.0 §5-§7（入口架构导出段、
   契约注册、C4 事故教训、归档纪律均不变）。

## 附录 A：背景证据摘要（同 v2.0 §0）
画布 64 饱和 0.954 IoU；P2 三死通路（画布精修 Δ≈0、框精修/jitter 全程 0）；
decoder 只消费 sparse（去点 0.73→0.16，dense/s0s1 无效）；负点 17% 误落 GT 前景
（airplane 阴影被当背景）；airplane AP50 0.95→AP75 0.01 断崖（飞机+阴影复合标注）。
基线 checkpoint：ft200 0.6617（主对照初始化）。
