# DUAL-P4-CHANS-20261014：真分通道重新 harvest 探针（预注册）

## 路由与谱系

A 路线冻结诊断探针，DUAL-P3（P3_NULL）判定表指定的唯一允许后续；不训练任何
模型、不读 test。P3 关闭了"64 维融合缓存上的读作侧调参"；P4 排除最后一个
读出级混杂：**随机投影维度上限与 SAT/BEV 混合**。P4 重新逐场 harvest 纯通道
特征，BEV 通道不与 SAT 混合、不经二次投影。

## 冻结锚点与输入

- 卫星锚点与 P1-V2 完全一致：EXP-0006 seed44
  `last_model_epoch100.pth`，SHA256 必须等于
  `fc3ead0e9490ffd9249cdbc4ce2ee9556bed5252efc3e475dfcc0f2aace65145`；
  SAM2 权重按 identity manifest 校验；height bin 边界沿用 H0 冻结值。
- 数据：`ue_raw_hero_uav_pairset_20261011` + `surfbev_b0_split_v2.json`
  （694/100/151）；**train+val 采集，test 全程不读取**。
- UAV 输入约束、SurfaceBEV 构造、z-buffer、sham90 定义与 P1-V2 逐字一致
  （复用其 rasterizer/encoder/相机代码路径，实现指纹落盘）。

## 通道定义与臂组装

- `sat`：B0 P3（512 维/cell）经新随机投影（seed 20261014）到 **128 维**，
  单次投影，无二次混合；
- `bev_c` / `bev_s`：correct / sham90 SurfaceBEV 的 120 维/场景通道描述子
  （12 高度 bin × [4 维随机投影 UAV RGB + 6 维原始统计]），**不与 SAT 混合、
  无全局降维**（仅保留 P1 同式的 per-bin RGB 4 维投影，seed 20261014）；
- 臂：`SAT`=sat128；`correct`=[sat128 ‖ bev_c]（248 维）；
  `sham90`=[sat128 ‖ bev_s]。三臂共享同一 sat128。

## 协议（与 DUAL-P2/P3 逐字一致）

fit 555 / select 139（seed 20261014 同规则置换）/ val 100 一次性判定；MLP
网格、公共配置选择、患病率校正阈值（correct 臂、π=select yb 均值）、线性
parity 锚、DEGRADED 哨兵、三指标口径、bootstrap（10k、seed 20261014）、
六门与勘误判定格全部同 P2 预注册。逐场采样行规则沿用 P1（(SEED, scene,
target) 独立确定，256正+256负）。iou 池化 = sat/bev 分别均值池化后拼接。

## 假设与判定

H-P4：去除维度上限与混合后，`correct−SAT` 与 `correct−sham90` 六门在 val
全部 CI 下界 >0（P4_PASS）→ 放行 DUAL-H1 型热启动筛查（learned 双流融合、
保留流身份、参与度遥测）。其余组合按判定格落判。

**方向终止条款（承 P3 表）**：若 P4 的 boundary 两门（correct−SAT F1 与
correct−sham90 F1）均未过，则 **cell 级 SurfaceBEV 描述子路线整体关闭**
（P1-V2、P2、P3、P4 四连证），双流机制转向 RoI 级多视角条件化并另立
DUAL-R 系列预注册；不再对该描述子做任何读出或 harvest 探针。

## test 政策与遥测

不读 test（val 为第四次使用，如实登记；选参仍全在 train 内部）。遥测三件套
同 P2/P3；harvest 阶段逐场落盘 surface 契约与预测哈希遥测，fail-closed。

## 工件

`/data1/wangcheng/checkpoint/surfbev_b0/probes/DUAL-P4-CHANS-20261014/`
（cache/ + split/selection/telemetry/summary/per_scene/run.log，六件套同 P2）
；结论 MD 与实现审计落 docs/、docs/reviews/。

## 补充（2026-10-14，harvest smoke 后、任何正式数据产生前）：锚点加载链勘误

实现时发现：现行 `ue/features.build_baseline` 要求 checkpoint 内嵌
`ue_baseline` 封印元数据，该封印在 P1-V2 运行（807d220）**之后**才引入；
EXP-0006 锚点（SHA `fc3ead0e…`，即 H0/R1a 实际使用并登记的权重）保存于
封印存在之前，因此现行 build_baseline 无法载入该锚点——现行身份链对历史
锚点是断链（登记为独立待办，不在本探针内修复）。P4 的加载采用与 R1a 运行
时（807d220）等价的钉定加载：checkpoint SHA 钉死 + snapshot architecture
contract 与封存 WHU 契约比对 + 严格加载（missing/unexpected 均空）+ SAM2
权重 SHA 与 snapshot runtime config 比对。该加载强度与 R1a 缓存的身份
约束等价；此偏差在任何 P4 正式数据产生前登记。
