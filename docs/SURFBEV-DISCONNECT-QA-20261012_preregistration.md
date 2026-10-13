# SURFBEV-DISCONNECT-QA-20261012：非连通实例数据探针预注册

## 边界

只读 v3 标签、v2 空间隔离 split 中的 train/val 场景，不读 test 图像/标签、不改模型、不改标签、不启动训练。本探针属于数据诊断，不是 AP 证据；`EXP-ID` 和热启动/全量门不适用。若未来决定重建实例 ID，另立数据版本并重审 global ID 的 split 隔离。

## 固定指标与口径

在原生 512² raster 上按 local ID 做 8 邻域连通域。逐实例保存面积、连通块数、最大/第二大块面积及比例、两大块间最小像素距离、各自 bbox、是否触边、global ID、UE `grouping_mode`/anchor。候选定义在计算前冻结：

- 宽口径 `M2`：至少两个连通块，第二大块面积 ≥64 px 且占总面积 ≥10%。
- 类例 `M2-strong`：`M2` 且第二大块面积 ≥128 px、占总面积 ≥20%、两大块间最小距离 ≥8 px。
- 伴随报告第二大块 ≥32/64/128/256 px、占比 ≥10/20/30%、间距 ≥0/4/8/16 px 的分布，以及 train/val 场景数、scene-local ID 数、去重 global ID 数、bbox 空白比例。阈值是排查分层，不宣称物理真值。

主数为 train/val 的 `M2` 和 `M2-strong` scene-local 实例数；伴随为独立场景数、global ID 去重数和分层分布。一次完整扫描，禁止事后按结果改阈值；无模型权重或 best 选择。样本不确定性只对**固定抽样的语义误合并比例**报告，使用 global-ID 层面的 Wilson 95% 区间，不能把候选数当作确认错标数。

## 视觉抽样与判定

从 `M2-strong` 按 train/val 和 unique global ID 固定种子 20261012 最多抽 32 例；再从 `M2` 非 strong 最多抽 16 例作边界对照。每例展示 RGB、两大块染色叠加和局部/完整 bbox。视觉类别仅为：`two_buildings`、`one_building_disconnected`、`uncertain`；无证据不得强判。用户确认的 `19064360d2c1ae7c` 作**已知正例校验**，不参与随机抽样比例估计。对样本还核对 `instances.json` 的 anchor/related actors，注意单一 UE actor 可能是资产分组而非物理实例。

成功门为 794 场 train/val 全部读通，来源/派生 SHA 与 v3 manifest 一致，已知正例进入预设候选层，独立复核计数和抽样隔离；否则只交失败原因，不报语义数量。最终只区分“精确形态候选数”和“抽样支持的物理误合并估计”，不自动拆分或清洗。

实施审计修正：初版 `SURFBEV-DISCONNECT-QA-20261012` 的两抽样臂各自去重，但臂间有 4 个重复 global ID，因此初版样本不得用于语义比例估计。保留初版产物，R2 使用跨两臂共享 global-ID 去重集合，并从随机样本中排除已知正例的**整个** global ID。候选定义与全量计数不变，R2 输出另立目录，不覆盖初版。

R2 仍是 scene-local 行先随机再按 global ID 去重，出现次数多的对象被抽中概率较大，且跨场景有 strong 的 global ID 可被弱层抽取；因此 R2 仍不得用于比例估计。R3 修复为**先以 global ID 定层（任一场 strong 则为 strong），每个 global ID 选第二大连通块面积最大的代表场景，再对 global ID 均匀抽样**。已知正例的整个 global ID 排除。R1/R2 产物留存，形态候选定义不变。
