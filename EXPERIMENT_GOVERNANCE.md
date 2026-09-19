# 实验管控规范（收敛版 v1.0）

> 形态：双层。本文档是**原则层**（人读，判定标准与红线）；执行层是 skill
> `experiment-governance`（AI 接到实验类任务时加载的操作规程）。两层同源，
> 冲突时以本文档为准。
>
> 适用：模型训练、消融、策略迭代类实验。产出背景：NWPU 消融矩阵项目实操
> （2026-09），每条规则都有对应的事故或教训，标注为 ⚠。

## 0. 前提：角色模型

单人研究者 + AI 助手协作，无 PR receiver。传统流程中的"评审"由**独立子代理
复核**承担；复核人不参与被审材料的设计与开发。跨机实验以单一**规范机**的
口径为准，跨机直接比较无效。

## 1. 算力分级准入（分型双路由）

所有新增实验方向按改动类型分流，立项时声明路由并说明理由，禁止跳级：

**A. 模块叠加型**（在既有基座上加模块，冻结基座可验证其价值）：
```
冻结探针（oracle/干预，分钟级） → 热启动筛查（冻结基座单训模块，小时级）
→ 全量训练（正式预算）
```
每级设有预注册判定门；门未过 → 暂停，复盘假设后重新探针，不得进入下一级。

**B. 基座/训练动力学型**（改动对象就是基座训练本身，冻结基座无从验证）⚠：
```
冻结诊断探针 → 直接全量
```
（例：梯度隔离类改动改的就是训练动力学，热启动层对它无定义。）

失败是正式结果：负结果须**证据先行关账**（冻结证伪/判定门落判之后才停），
归档与正结果同级。探针多次不过 → 终止该方向。

## 2. 指标四要素预注册（所有层级强制）⚠

启动前写死，禁止跑完再定标准挑结论：

1. **主指标** + **伴随指标**（禁止单指标报喜——"segm +0.012 藏着 bbox
   −0.022"是一次真实事故）；
2. **选择口径**（last / 平台窗口 / best，预先指定；未指定口径的 best 尖峰
   是选择伪影，不得作为结论依据）；
3. **显著性方法**（image-paired bootstrap CI / 双 seed / 效应主张线，
   预先约定并区分其覆盖的不确定性类型）。

## 3. 遥测三件套 ⚠

遥测方案在编码前预定义，且必须同步交付：

1. **判读边界**：每个字段写明能证明什么、不能证明什么、DDP 聚合口径；
2. **标定校验**：遥测声称度量的决策面/配置必须与部署配置交叉验证
   （计数器锚 logit-0 而部署判定线在 −0.4055 的误读是一次真实事故）；
3. **机制遥测不得替代结局验证**：训练侧字段只作机制证据，结论以完整
   validation + 推理为准。

## 4. 独立子代理复核（节点触发）

三个强制节点，复核记录**全文落盘**（`docs/reviews/`，含 VERDICT 行与
发现清单，结论 MD 引用其路径）：

1. **全量拉起前**：实施+接线审计（臂隔离、指纹回归、单变量键集）；
2. **探针首次定版时**：审实现（对照 cell、sham、不变量）；
3. **结论写回论文口径前**：审结果（数字复现、预注册门执行、无越界表述）。

日常探针迭代与筛查自验（smoke + 不变量）即可，不触发复核。

## 5. 训练启动与实验臂隔离

1. 一切训练经统一 wrapper 启动；契约 echo 自描述：git commit/dirty、全部
   超参、架构指纹、协议、运行标签；
2. 臂 = 配置（env 组合），不是代码分支。臂间隔离由检查保证 ⚠：新臂接入后
   必须跑**全臂 harness**——遗留臂指纹逐一不变 + 新旧臂 resolved config 的
   两两 diff 恰好等于预期键集。"单变量"是检查出来的，不是声明出来的；
3. checkpoint 自描述（内嵌 config snapshot），推理入口按快照重建。

## 6. Git 管控

1. **单分支为体**；`exp/` 分支仅用于协议级大改（改协议/大重构/依赖升级），
   评审后合入；
2. **脏代码分级** ⚠：探针/筛查允许 dirty 启动（wrapper 记录 commit+dirty）；
   **全量 run 一律先 commit，无例外**——论文口径数字的复现性必须钉死在
   具体 commit 上；
3. commit 信息：`[类型] 描述 | EXP-ID`，单次提交单一主题；中文携带关键数字；
4. **EXP-ID**：进入全量时在 `docs/EXP_REGISTRY.md` 登记发号（每全量 run
   一个，全局唯一，`EXP-XXXX`，只发不复用）；commit message 携带，并在该
   commit 上打 tag。臂名继续承担谱系语义，探针/筛查以臂名+日期引用；
5. 禁止 force push 覆盖已用于实验的 commit；
6. 资产：权重/探针工件不进 git，放数据盘目录并在结论 MD 登记路径；配置
   身份 = env 解析 + checkpoint 快照 + 契约 echo 三重保障，不设独立配置目录。

## 7. 终止、早停与归档

- **早停纪律** ⚠：平台判定成立（多窗口无新高 + 遥测走平）后才停；best
  checkpoint 已存盘才杀；按 PID 树杀进程，防自匹配与孤儿双拉；
- 中途发现代码 bug → 直接终止，不继续烧算力；
- **归档包**（每实验，含探针/筛查/全量）：结论 MD（预注册四要素的判定
  原文与落判）、复核记录落盘件、遥测日志路径、工件/checkpoint 路径、
  EXP-ID 关联。

## 附：明确删除项及理由

- GPU 共租纪律 → 属硬件运维而非实验规范，留在项目实操约定中；
- PR 评审流程 → 无 receiver，由节点触发的子代理复核替代；
- DVC/对象存储、`config/experiments/` 目录制 → 与臂 env 体系重叠，
  由自描述 checkpoint + 契约 echo 覆盖其职责。

---

## 附录 A：立项记录 MHBEV-ROI-20260914（多高度 BEV；深度阶段隔离）

**状态：预注册，尚未授权启动训练。** 本记录把下面的技术方向约束为一个
模块叠加型实验；它不授权将 UAV 深度、单目相对深度、真实渲染深度或任何新的
监督项混入本轮。深度分支须在本轮证据关账后另立项、另编号、另走准入。

### A.1 路由、锚点与问题陈述

- **路由：A，模块叠加型。** 冻结的单流基座可独立运行，新增的几何查询模块可以
  零残差旁路，因此依次执行冻结探针 → 热启动筛查 → 全量训练，禁止跳级。
- **冻结锚点：**
  `ue_single_c5v2_control_tr128_va259_s44_r1/best_model_epoch86.pth`；官方 UE
  val，259 scenes / 5,937 instances，输入 1024²，seed 44，单流
  segm/mAP `0.7312098569877999`、bbox/mAP `0.7208712043697307`。
- **失败证据：**该锚点的实例探针中，COCO-small 为 332 个，`88.25%` 非良好
  匹配（177 missed、50 partial-or-merge、66 boundary error）；这些失败小实例在
  z=0 时 `89.42%`、z=22m 时 `87.03%` 至少有两路 UAV 几何可见。该覆盖只说明
  “可查询”，不说明 UAV 内容必然有语义增益。
- **目标：**优先恢复小实例的 proposal recall / instance separation，同时让 UAV
  的高度选择进入局部实例特征；不是把高度 argmax 解释为真实建筑高度。

### A.2 不可变数据与几何契约

1. 使用官方 `ue_coco_with_uav` train/val split；训练为既有 128 场景子集、val
   为全部 259 场景、seed 44。卫星实例标签是唯一任务监督。
2. 每景固定五张 512² UAV RGB；`K:[5,3,3]`，`E:[5,4,4]` 是 **W2C column-vector**
   矩阵，世界坐标 UE cm、Z-up。投影计算必须 FP32 且 autocast disabled。
3. 首轮卫星—世界映射唯一采用当前运行代码的 360m tile contract：
   `X=(0.5-v/1024)*360m`、`Y=(u/1024-0.5)*360m`。论文草稿中的 236.5m affine
   未经独立标定，禁止作为本轮隐式替换。
4. 高度集合固定为 `H={0,4,7,10,15,22,32}m`；局部偏移固定为
   `D={-20,0,+20}m × {-20,0,+20}m`。候选在少于两路有效视角时硬掩蔽；全无效时
   UAV residual 必须严格为零。

### A.3 实施边界与实验臂

**共同基座。** 单流 C5v2 + frozen UAV SAM2 encoder。新增参数单独归入
`drone_guidance`；基座参数在热启动阶段冻结。所有新增 residual 的输出投影为
零初始化，确保关闭 gate 时严格回到锚点行为。

| 臂 | 唯一变化 | 目的 |
|---|---|---|
| A0 | 单流，UAV bypass | 复现锚点与训练/推理 parity |
| A1 | 高分辨率 P2、单层 `z=0` 几何 residual | 分离“新的 P2/RPN 接入”与高度收益 |
| A2 | A1 + `H` 多高度 BEV，按高度注意力 collapse 到 P2 | 检验多高度是否提升小 proposal recall |
| A3 | A2 + 对 bbox/mask ROI 的 `H×D` 局部检索 | 检验高度/偏移是否改善粘连与轮廓 |

实现顺序固定：先 A1，再 A2，最后 A3；任一前序臂未过门，不实现后续臂。P2
分支服务 RPN 的小 proposal recall；ROI 分支只服务已经产生的候选的分类、分离和
mask。禁止把全局低分辨率 128² BEV 上采样后宣称解决小目标。

### A.4 预注册指标四要素

**主指标：**UE val `segm/mAP_s`（COCO area `<32²`，选择口径为 **last checkpoint
的标准推理**）。

**伴随指标：**`bbox/mAP_s`、`segm/mAP`、`bbox/mAP`、小实例的 mask-IoU≥0.5
匹配召回、以及 single-stream failure probe 中 small `missed + partial_or_merge`
比例。不得只报告主指标。

**显著性：**所有比较使用同一 259 images 的 image-paired bootstrap，10,000 次、
bootstrap seed 20260914；报告差值 95% percentile CI。热启动只作筛查，正式结论
还要求两个独立训练 seed 的同方向效应；CI 仅量化固定模型在 val 图像上的抽样
不确定性，不能代替训练种子不确定性。

**判定门：**

1. 冻结探针门：A0 bypass 必须与锚点逐预测一致；正确 UAV 几何候选相较于
   `XY+80m` spatial-shift sham 和 scene-permuted UAV sham，均保持相同的候选数和
   detector 输入。冻结探针中三种条件均须生成非零的候选 UAV feature，以便比较其
   统计性质；残差注入门在三种条件均固定为零，因此 detector 输出必须完全不变。
   只有后续热启动中的 correct geometry 可被学习为非零注入。small failure 集合的
   ≥2-view 可查询率须 ≥0.85 于 `H` 的至少一个高度。任何 parity、掩蔽或
   W2C 标定失败均停止。

**A1-Frozen-R1（2026-09-14，重新立项的数值标定审计）：**A1-Frozen 的其余门已
覆盖 259 val 场并通过；原 `E @ inv(E)` 审计在 float32 大尺度厘米坐标下出现消减
误差，故不得以该结果放行，也不得回改旧结果。本 R1 只替换标定**审计精度**：模型
运行时的 `K/E`、360m 映射、FP32 投影与 autocast-disabled 维持不变；相机中心的
矩阵自洽性固定以 float64 计算，要求所有场景 `max|E64 @ inv(E64)[:,3]-[0,0,0,1]|`
小于 `1e-6 cm`。此外，必须对同一组由 FP32 生成的 P2 查询点，验证 FP32 与 FP64
投影的 valid mask 完全相同，且二者共同有效点的最大像素偏差不超过 `0.5 px`。R1
仍须重跑完整 259 场的 parity、三种 sham 和 hard-mask 门，并再次独立复核；任一项
失败即停止，不进入热启动。

**A1-Frozen-R2（2026-09-14，边界舍入审计）：**R1 的 FP64 外参自洽和共同有效
投影偏差均通过，但 FP32/FP64 在严格图像边界（`u/v=0/512`）的 valid 判定不完全
一致，故 R1 不放行。本 R2 不改运行时 FP32 投影、采样坐标、`<2-view` hard mask 或
模型；只将 FP64 比较定义为数值审计：对每一个 view-query，若 FP32 或 FP64 投影到
任意画面边缘的距离 `≤0.5 px`，记为 boundary band。门要求：(i) boundary band 外的
FP32/FP64 valid 必须逐位一致；(ii) 所有 P2 query 的 `≥2-view` support mask 必须逐位
一致（包括 boundary band）；(iii) 所有 valid 不一致只能发生在 boundary band；(iv)
共同有效点的最大坐标差 `≤0.5 px`；(v) R1 的 float64 W2C 门、259-scene parity、三种
sham 与 hard-mask 门仍全部重跑并通过。R2 的 boundary mismatch 数和比例必须归档，
仅用于证明数值边界行为，不能作为 UAV 语义增益证据。任一项失败即停止。

**A1-HotStart-S1（2026-09-14）：**R2 PASS 后的唯一筛查训练臂。由单流锚点
`best_model_epoch86.pth` 通过 `--init-from --allow-cross-arch-init` 初始化；仅允许
新增 `a1_p2.{proj,gamma}` 训练，卫星 backbone/neck/RPN/ROI/SAM/P2-refiner 与 UAV
encoder 均冻结，UAV encoder 处于 eval/no-grad。A1 只在 P2 256² 使用 z=0、360m、
FP32 autocast-disabled 投影，要求 `≥2-view` hard mask；不得触碰 P3、SAM embedding
或 ROI。`proj` 保持默认随机初始化而 `gamma=0`，以保证锚点 parity 且使 gamma 有
梯度（禁止 projection 与 gamma 同时零初始化）。两个独立 seed 为 44、45，各 30
epoch，128 train 场景、259 val 场景、batch=1/accum=8；选择口径为每 seed 的 last
checkpoint，筛查判定仍使用既定的最后 5 个 val epoch 均值门。启动前 harness 必须
断言 source unexpected=0、shape mismatch=0，target missing 仅
`drone_encoder.*`/`a1_p2.*`，且 optimizer 仅含 `a1_p2.*`。 
2. 热启动筛查门（A1→A2，A2→A3）：采用两个 seed，最后 5 个验证 epoch 的均值
   相对相应前序臂满足：`Δsegm/mAP_s ≥ +0.015`，`Δbbox/mAP_s ≥ 0`，且
   `Δsegm/mAP ≥ -0.002`、`Δbbox/mAP ≥ -0.002`；两个 seed 同方向。未通过即停止
   当前后续臂并归档负结果。
3. 全量门：两个 seed 的 mean `Δsegm/mAP_s ≥ +0.020`，paired-bootstrap 95% CI
   下界 `>0`；同时 mean `Δbbox/mAP_s ≥ 0`，`Δsegm/mAP` 与 `Δbbox/mAP` 都
   `≥-0.002`。所有结论以 each seed 的 **last** 标准推理与两 seed mean 为准，
   `best` 仅作为 checkpoint 保存用途，不得替代口径。

### A.5 探针与遥测三件套

**探针（编码后、筛查前）：**

- zero-gate / UAV bypass parity：输出实例、scores、bbox、mask 必须逐位一致；
- 360m 映射的 tile center 投影、`E@[camera_center,1]=[0,0,0,1]`、5-view K/E
  shape 与现有 `diagnose_multihigh_geometry.py` 一致；
- correct / `XY+80m` shifted / scene-permuted UAV sham：三者共享卫星输入及
  detector proposal，sham 不得通过改变 proposal 数制造假阳性；
- detector 字段污染检查：仅 ROI arm 的 probe 不得改变 pre-ROI RPN proposals；
  P2 arm 则必须显式报告并允许 proposal 改变。

**遥测字段与判读边界：**

| 字段 | DDP 口径 | 能证明 | 不能证明 |
|---|---|---|---|
| `candidate_valid`, `two_view_ratio`, height/offset masked ratio | 所有 rank 的 valid candidate 总数/总数 | 几何查询未死、无效候选被屏蔽 | UAV 内容有语义价值 |
| height entropy、expected height、expected offset、cross-view variance | 有效 ROI 的全局均值及 p10/p50/p90 | 模块确实区分候选而非恒定取一个 bin | 高度是真实建筑高度 |
| `gamma_p2`, `gamma_bbox_roi`, `gamma_mask_roi` 与 residual/base norm | 参数全局值；ratio 全 rank mean | 残差路径打开且量级可审计 | AP 改善由该路径因果造成 |
| small proposal recall、small matched recall、small miss/merge rate | rank0 汇总所有 val image | 与预注册结局直接对应 | 跨 seed 泛化 |

所有阈值、坐标映射、mask 判定线必须同部署配置交叉验证；机制遥测永不替代完整
val 标准推理。首次探针定版后必须有独立子代理将对照、sham、parity 与字段接线
审计落盘至 `docs/reviews/`。

### A.6 资源、停止与归档

- 探针：分钟级、无训练；热启动：冻结 A0，仅训新增模块，小时级，达到平台门或
  多窗口无新高后停止；全量前必须按第 6 节 commit/EXP-ID/tag/全臂 harness/独立
  审计要求执行。
- 发现 NaN、零 gate 长期不动、有效候选率与探针标定不符、sham 优于正确配对，
  立即按 PID 树停止并归档，不以调参掩盖。
- 每臂归档：预注册门原文、resolved config diff、probe/telemetry、四权重标准
  推理、bootstrap 工件、结论 MD；失败臂同样归档。

### A.7 后续深度阶段（明确不属于 MHBEV-ROI-20260914）

只有 A2/A3 的多高度几何路线完成关账后，才可另立“UAV-depth”方向。届时先决定
深度来源：UE rendered metric depth、可校准的 metric-depth 模型或仅 relative-depth
prior；relative depth 禁止直接当 cm/m 距离反投影。新方向必须重新预注册深度监督、
遮挡/z-buffer 契约、domain-gap probe 与所有判定门。

---

## 附录 B：立项记录 MHBEV-DIAG-20260915（small failure 归因；冻结、只读）

**状态：预注册；冻结诊断，不训练、不改权重、不授权任何新模型臂。** A1-HotStart-S1
失败后，不得把 P2 z=0 residual 的负结果改述为 “UAV 无效”，也不得直接转入 A2。
本诊断只回答：单流小目标的主要可修复瓶颈位于 RPN proposal，还是位于已有候选后的
ROI/mask；UAV 几何覆盖只作为后续可查询性的必要条件，不是语义增益证据。

### B.1 路由、锚点与不可变契约

- **路由：冻结诊断。** 不存在可训练模块；输出只读 JSON/CSV，不启动热启动或全量。
- **锚点：** A.1 的同一单流 checkpoint，官方 UE val 的完整 259 scenes / 5,937
  instances，1024² 输入；不得替换 split、阈值或 checkpoint。
- **small：** GT raster mask 面积 `<32²`（与既有 failure probe 一致）。
- **RPN proposal：** 部署 `rpn_head.predict(extract_feat(...))` 的未送入 ROI 的
  `bboxes/scores`，按原 test config、`rescale=False`。对一个 GT，proposal recall@t
  是任一 RPN box 与 GT box IoU `>=t`；分别报告 `t={0.3,0.5}`。它不等同于最终检测
  recall，不能证明 UAV 会提高 recall。
- **最终 failure：** 复用既有 failure probe 的 final mask one-to-one Hungarian matching
  和 IoU 0.5：`missed` 或 `partial_or_merge` 为失败。RPN 只读抽取绝不写入
  `data_sample.proposals`，也不运行/替换 ROI。

### B.2 预注册四要素与对照

- **主指标：** small final-failure 集合的 RPN proposal recall@0.5。
- **伴随指标：** small 全集 recall@0.3/@0.5、final-failure 的 proposal-present 比例、
  各 failure 类型计数，以及 correct / `XY+80m` / scene-permuted geometry 下失败
  small GT center 的 ≥2-view 覆盖率。
- **选择口径：** 完整 259 val 的唯一一次只读扫描；无 best/挑 epoch。
- **不确定性：** image-paired bootstrap 10,000 次、seed 20260915，报告 failure-subset
  recall@0.5 与 matched-good-subset recall@0.5 的差值 95% percentile CI。该 CI 仅
  描述 val images，不外推训练种子。
- **不变量 / 阴性对照：** 每景先标准 `model.predict`，再只读 `extract_feat+rpn_head.predict`
  并再次标准预测；bbox、score、label、mask hash 必须逐位一致。重复 RPN readout 的
  boxes/scores hash 必须逐位一致。几何覆盖必须报告 correct、XY+80m 和 scene-permuted
  三条件，且三条件共享同一 GT / RPN / final prediction；不得因 sham 改变 proposal 数。
- **判定规则：** 若 final-failure small 的 recall@0.5 `<0.50`，且相对 matched-good
  small 至少低 `0.15`，并且该差值 bootstrap 95% CI 上界 `<0`，则下一假设应优先针对
  **proposal/objectness**；若 failure-subset recall@0.5 `>=0.75`，则 RPN 不是主要
  瓶颈，下一假设应优先针对 **ROI classification / instance separation / mask**；其余
  为 **AMBIGUOUS**，不得据本探针启动训练。任何 parity、readout stability 或数据完整性
  失败均为 **INVALID**。

### B.3 遥测边界与归档

- RPN telemetry 由 rank0 单进程扫描完整 val 汇总；每 GT 一行保存 GT box、area bucket、
  final failure、max proposal IoU、recall flags 和三个几何 coverage。它能定位基线瓶颈，
  不能证明干预后的因果增益。
- geometry 复用 A.2 的 W2C、360m、FP32 部署投影和 ≥2-view 判据；仅对 GT center
  做可查询性统计，不能当做实例完整可见或语义相似度。
- 归档包含 manifest、instances CSV、summary JSON、bootstrap JSON、实施复核与结论 MD。
  首次定版后须独立复核，结论不得超出上述判定规则。

---

## 附录 C：立项记录 MHBEV-DIAG-20260915-B1（proposal-present small failure 分解；冻结、只读）

**状态：预注册；冻结诊断，不训练、不改权重、不授权任何模型臂。** B0 已证明 227
个 final-failure small 中 120 个有 RPN IoU≥0.5 proposal、107 个没有，按 B0 的
预注册门合法落判为 `AMBIGUOUS`。B1 只对这 120 个 **proposal-present final-failure
small** 分解 post-RPN 的失败位置；不得把 B1 用于重写 B0 门或放行 A2/A3。

### C.1 锚点、样本与测量契约

- **路由：冻结诊断。** 使用 B0 同一 checkpoint、UE val、1024²、正常部署
  `model.predict(rescale=False)`；不调用训练、不给 data sample 写 proposals，也不修改
  model/config。
- **目标集合：** 从 B0 `instances.csv` 读取并硬校验：`area_bucket=small`、
  `final_failure∈{missed,partial_or_merge}`、`rpn_recall_iou05=True`。其预注册大小为
  120；B0 的 checkpoint、259 scenes / 5,937 rows、所有 B0 hash invariants 必须一致，
  否则本实验 `INVALID`。
- **final box reach：** 同一个 GT 对所有部署最终 `pred_instances.bboxes` 的最大
  xyxy IoU。`>=.5` 表示有最终检测框到达该 GT；它不意味着分类标签、实例归属或 mask
  已正确。
- **final mask failure：** 同一正常预测的最大 mask IoU `<.5`；必须重算并与 B0
  target 标记一致。按定义每个目标均失败。
- **两类互斥且穷尽的结果：**
  1. `post_rpn_box_lost_or_mislocalized`：final box reach `<.5`；它涵盖 ROI 分类
     抑制、NMS、box refinement 偏移等，**不得**狭义命名为分类错误。
  2. `mask_only_failure`：final box reach `>=.5` 且 final mask failure；它说明该
     路存在定位到达而 mask 未达 IoU 0.5，不能单独证明 SAM/UAV 哪个子模块是原因。

### C.2 预注册四要素、对照与判定

- **主指标：** 120 个 target 中 `mask_only_failure` 的比例；伴随指标：
  `post_rpn_box_lost_or_mislocalized` 比例、各 target 的 max final box/mask IoU、
  final detection 数、按类型的 B0 correct/XY+80m/scene-permuted ≥2-view 覆盖。
- **选择口径：** B0 明确的 target rows；每 target 所在 scene 一次正常 predict，
  无 checkpoint/阈值挑选。
- **显著性：** image-paired bootstrap 10,000 次、seed 20260916，对
  `mask_only_fraction - box_lost_fraction` 报 95% percentile CI；缺 target 的 scene
  可被重抽，比例以 pooled numerator/denominator 计算。
- **阴性对照与不变量：** 每个 target scene 正常预测两次，bbox/scores/labels/masks
  的 bytes+shape+dtype hash 必须逐字段一致；B1 绝不向 detector 前向注入 UAV 或 sham，
  因而 B0 已归档的 correct/XY+80m/scene-permuted geometry row 仅重分组报告，不能改变
  B1 的模型输出、proposal 或类别。B0 的 target membership 和 B1 重算的 mask failure
  必须逐行一致。
- **判定门：** 若 `mask_only_fraction >= .60` 且 bootstrap CI 下界 `>0`，下一假设可
  优先针对 ROI-local mask separation；若 `box_lost_fraction >= .60` 且 CI 上界 `<0`，
  下一假设可优先针对 post-RPN ROI/box path；其余 `MIXED`，不得启动训练。任一 B0
  lineage、数据完整性、target 一致性或 parity 失败为 `INVALID`。

### C.3 遥测边界与归档

- 单进程 rank0 汇总，逐 target CSV 保存 B0 标识、max final box/mask IoU、类别、
  final detection 数与 inherited geometry view counts。该测量可定位 failure stage，
  不能证明任一未来 UAV 注入的因果效果。
- 归档 `summary.json`、targets CSV、bootstrap JSON、run manifest、stdout log、独立
  实施/结果复核与结论 MD。任何 `MIXED/INVALID` 与正向分流同级关账。

---

## 附录 D：立项记录 MHBEV-DIAG-20260915-B2（post-RPN box-lost 机制分解；冻结、只读）

**状态：预注册；冻结诊断，不训练、不改权重、不授权任何模型臂。** B1 已合法判定
101 个 proposal-present final-failure small 是 post-RPN box-lost/mislocalized。B2 只
分析这 101 个 B1 target，分离 bbox refinement 将可用 proposal 推离 GT，与仍有对齐
decoded box 却在 score/NMS/rank 后消失的情况。

### D.1 锚点与可审计测量

- **路由：冻结诊断。** B0/B1 同一 checkpoint、UE val、1024²。B2 每个 target scene
  先后执行两次正常 `model.predict(rescale=False)`，中间在新 data sample 上只读执行部署
  同源的 `extract_feat → rpn_head.predict → roi_head._bbox_forward`；禁止写 proposals、
  禁止运行训练或注入 UAV/sham。
- **固定 target：** B1 `targets.csv` 中 `category=post_rpn_box_lost_or_mislocalized`
  的 101 行。B1 lineage（checkpoint、B0 lineage、94/120 complete、所有 invariant）和
  target 身份必须硬校验，否则 `INVALID`。
- **P\*：** 对一个 GT，RPN 1,000 proposals 中与 GT box IoU 最大的 proposal；B2 要求
  P\* IoU `>=.5`，与 B0 target 身份一致。
- **decoded box：** 将 `roi_head._bbox_forward` 的 bbox prediction 按部署
  `bbox_head.bbox_coder.decode` 在相同 image shape 解码，但**不**运行 NMS。当前锚点的
  bbox head 是 `Shared2FCBBoxHead + DeltaXYWHBBoxCoder`；ROI foreground score 按部署分类
  logits 的 softmax 和单类前景列计算。部署 score threshold、NMS 与 max-per-image 均从 live
  `roi_head.test_cfg` 记录，不硬编码。

### D.2 预注册四要素、对照与判定门

- **主指标：** 101 target 中 `bbox_refinement_lost` 比例；伴随：P\* pre/decode IoU、
  any decoded aligned box（IoU≥.5）比例、score-rejected 比例、NMS/rank-loss
  比例和 raw foreground score 相对于部署阈值的 margin。
- **互斥且穷尽类别：**
  1. `bbox_refinement_lost`：所有 decoded pre-NMS boxes 对 GT 的最大 IoU `<.5`；
  2. `roi_score_rejected`：存在 decoded IoU≥.5 box，但这些对齐 boxes 的最大 raw
  foreground score `<=` live `score_thr`；
  3. `nms_or_rank_loss`：存在 decoded IoU≥.5 且至少一个 score 达 threshold，
     但 B1 已证实最终 post-NMS box 未到达。
  第三类包括 NMS、max-per-image ranking 等，不得狭义写成 NMS。
- **选择口径：** B1 固定 101 target；每 target scene 唯一部署预测读数，无阈值/权重
  挑选。
- **显著性：** image-paired bootstrap，完整 259 val scene universe，10,000 次，seed
  20260917；对最大类比例减去其余两类比例报告 95% percentile CI。
- **阴性对照 / 不变量：** 两次正常 prediction 的 bbox/scores/labels/masks typed hash
  必须逐字段一致；中间 readout 不得给 detector 写 proposals。B2 不进入 UAV/sham 前向，
  B0 的 correct/XY+80m/scene-permuted 几何字段只按类别继承报告，不能改变任何 detector
  输出。P\* 预 IoU 与 B0 target membership、B1 final box failure 必须逐行一致。
- **判定门：** 仅当某一类别比例 `>=.60` 且其“该类 − 其余两类” bootstrap 95% CI
  下界 `>0` 时，才按该类别分流为 `BBOX_REFINEMENT`、`ROI_SCORE` 或
  `NMS_OR_RANK`；否则 `MIXED`。任何 lineage、data completeness、parity、P\*
  或类别穷尽失败均 `INVALID`。任何非 INVALID 也只定位单流失败机制，不授权训练。

### D.3 遥测边界与归档

- 单进程 rank0，每 target 保存 P\* index/IoU、max decoded IoU、对齐 decoded 数、
  对齐最高 foreground score 与 score margin、live test_cfg、类别和 inherited geometry counts。它能定位
  deployed ROI/bbox 后处理链中的断点，不能证明未来 UAV 特征会改善该断点。
- 归档 summary、targets CSV、bootstrap、manifest、log 和独立实现/结果复核；只允许在
  B2 结论后新立与对应机制一致的冻结干预探针。

---

## 附录 E：立项记录 MHBEV-DIAG-20260915-B3（ROI-score UAV frozen linear readout；不训练 detector）

**状态：预注册；冻结特征诊断，不训练或修改 detector/UAV encoder，不授权模型训练。**
B2 已将 87 个 target 定位为 `roi_score_rejected`：存在 decoded IoU≥.5 box，但 ROI
foreground score 未过 `.05`。B3 检验经过正确几何查询的 frozen UAV 局部特征是否含有
独立于已匹配卫星 ROI 分数的 candidate targetness 信号。它是一个**离线线性 readout
probe**，不是双流模型，也不能作为 AP 增益证据。

### E.1 数据、特征与模型不可变契约

- **冻结锚点与 val：** B0/B1/B2 同一单流 checkpoint；B2 的固定 87 个
  `roi_score_rejected` val target，来自官方 259-scene val。B2 lineage、84/101 complete、
  `Shared2FCBBoxHead + DeltaXYWHBBoxCoder`、score threshold `.05` 以及所有 B2
  invariants 均须硬校验。
- **readout training split：** 锚点训练时的同一官方 UE train 128-scene subset
  （`ratio=0.1797752808988764, seed=44`）；该 subset 与 val 无 scene 重叠。训练仅拟合
  sklearn `StandardScaler + LogisticRegression(C=1, class_weight=balanced, max_iter=1000,
  random_state=20260918)` 到冻结 descriptor；绝不反传/写入 detector 或 UAV encoder。
- **candidate pair：** 正例为一个 small GT 的 decoded pre-NMS IoU 最大且 `>=.5` 的
  RPN candidate；负例为同场景、与任一 GT decoded IoU `<.1`、且 raw foreground score
  距正例最接近的 candidate。val 正例仅限 B2 的 87 target 对应候选。pair 仅在
  `|score_pos-score_neg|<=.02` 时纳入评估；报告纳入数和 score gap。负例选择不看 UAV。
- **frozen UAV ROI descriptor：** 对 candidate decoded box 的固定 `3×3` 卫星局部网格，
  使用部署 360m、z=0、W2C cm、FP32/autocast-disabled 投影，在 frozen UAV SAM2 finest
  FPN 上 bilinear sampling；每个 grid point 平均 ≥2 个有效 view 的 256-D feature，再对
  9 点平均。candidate 只有 9/9 grid points 都 ≥2-view 时有 descriptor。
- **三条件：** `correct` 使用当前场景 UAV FPN/K/E；`XY+80m` 只把同一世界 query 的
  X 偏移 +80m；`scene_permuted` 只以同 split 的下一场 UAV FPN 替换视觉内容，K/E 和所有
  candidate/labels 保持当前 scene。三条件都不进入 detector 前向。

### E.2 预注册四要素、对照与判定

- **主指标：** held-out val pairwise win rate：同一冻结正确几何 readout 的正例分数是否
  大于其 score-matched hard negative。
- **伴随指标：** `XY+80m`/scene-permuted win rate、correct 相对两个 sham 的差、三条件
  共同可用 coverage、mean absolute satellite-score gap、train pair 数与类别平衡。
- **选择口径：** 唯一的 87 B2 target 与一次固定 logistic fit；不得调 C、选择 FPN 层、
  改 3×3 网格、阈值或选择 best readout。
- **显著性：** image-paired bootstrap 10,000 次、seed 20260918，对 `correct - XY+80m`
  与 `correct - scene_permuted` win rate 报 95% percentile CI；完整 259-val-scene
  universe，非 target scene 以零 pair 参与重抽。
- **不变量 / 阴性对照：** 每个 readout scene 正常 detector prediction 前后 bbox/scores/
  labels/masks 的 typed hash 必须一致；RPN/ROI readout 不得写 proposals；correct 与 sham
  的 pair identity、satellite score、decoded boxes 和 label 必须逐位一致。所有 3 条件的
  descriptor 仅在共同有效的 pair 上评估。UAV encoder eval/no-grad，参数 hash 前后相同。
- **判定门：** 若共同有效 val pair coverage `>=.80`、correct win rate `>=.60`、且两个
  `correct - sham` bootstrap 95% CI 下界均 `>+.05`，则落判
  `FROZEN_UAV_ROI_SCORE_SIGNAL`；否则 `NO_FROZEN_SIGNAL`。任何 lineage、split overlap、
  descriptor/condition identity、detector parity、encoder hash 或数据完整性失败为
  `INVALID`。即使 PASS，也只授权再立 **ROI-score zero-gate frozen intervention**，不
  授权热启动或全量训练。

### E.3 遥测边界与归档

- rank0 汇总；逐 pair 保存 scene/GT/candidate indices、satellite scores、三条件 support、
  三条件 readout scores、是否共同有效及 win bits。它能说明 frozen UAV content 是否带有
  与 targetness 一致的可读信号；不能证明该信号能被端到端检测器利用或提高 AP。
- 归档 summary、train/eval pair CSV、bootstrap、manifest、run log、独立实施/结果审计和
  结论 MD。`NO_FROZEN_SIGNAL/INVALID` 同级关账，不以调参或替换 readout 挽救方向。

---

## 附录 F：立项记录 MHBEV-DIAG-20260915-C1（RPN dense preselection UAV frozen readout；不训练 detector）

**状态：预注册；冻结诊断，不训练或修改 detector/UAV encoder。** B3 已在 ROI score
路径落 `NO_FROZEN_SIGNAL`，因此本立项不复用 B3 的 ROI candidate、门或结论。B0 的
227 个 final-failure small 中有 107 个在部署的 post-NMS RPN proposals 内没有 IoU≥.5
候选；C1 只问一个更窄的问题：在 RPN 的 dense、pre-`nms_pre`/pre-NMS decoded
candidates 中，正确几何的 frozen UAV 内容是否含有可把潜在正 candidate 排在
satellite-score-matched hard negative 之前的 targetness 信号。

### F.1 路由、固定数据与 dense-RPN 契约

- **路由：模块叠加型的冻结诊断。** 任何 UAV→P2/RPN 注入、热启动或训练均须等
  C1 通过；C1 PASS 也只授权另立 *RPN zero-gate frozen intervention*，不授权训练或
  AP 声明。
- **固定锚点与 lineage：** B0 的同一单流 checkpoint；B0 `instances.csv` 中
  `area_bucket=small`、`final_failure∈{missed,partial_or_merge}`、
  `rpn_recall_iou05=False` 的 107 个唯一 `(scene_id,gt_index)` target。B0 的完整
  259-val/5,937-instance、prediction/RPN-repeat invariants 与 post-NMS `max_rpn_box_iou`
  必须逐 target fail-closed 复验。
- **dense RPN candidate：** 用冻结 detector 的标准 `extract_feat` 及
  `rpn_head.forward` 得到部署 anchor 上的 raw cls/delta；只读地以同一 bbox coder
  decode 所有 level 的 anchors，`score_thr=0,nms_pre=-1,with_nms=False`。这只是
  *preselection readout*，不替换部署 RPN，也不写 `data_sample.proposals`。正例是该 GT
  IoU 最大的 decoded raw candidate，要求 IoU≥.5；负例是同场景、与任一 GT IoU<.1、且
  raw sigmoid objectness 与正例绝对差最小的 raw candidate。pair 选择不看 UAV；val
  仅在该 objectness gap≤.02 时评估。若 raw P* 未达 .5，记为
  `raw_geometry_unavailable`，不虚构可由排序恢复的 candidate。
- **feature / split：** 训练为官方 128-scene anchor train subset
  (`ratio=.1797752808988764,seed=44`) 的所有满足 raw-P* IoU≥.5 small GT；验证仅为
  上述 B0 107 target。冻结 UAV SAM2 finest FPN；对 decoded raw box 固定 3×3 网格，
  z=0、360m、W2C cm、FP32/autocast-disabled、bilinear sampling，每点至少两视角、
  9/9 点有效才产生 256-D descriptor。线性 readout 固定为
  `StandardScaler + LogisticRegression(C=1,class_weight=balanced,max_iter=1000,random_state=20260919)`，
  仅 fit train。
- **三条件：** correct；只将世界 X 偏移 +80m 的 `XY+80m`；以及下一 val scene 的
  UAV FPN 替换视觉内容、但保留当前 K/E 的 scene-permuted。三条件的 detector、dense
  pair identity、raw objectness、decoded boxes、labels 必须逐位一致；三条件均不得进入
  detector/RPN 前向。

### F.2 四要素、门与遥测

- **主指标：** held-out 107 B0 target 上 correct 的 pairwise win rate。
- **伴随指标：** 两个 sham win rate、correct-sham 差、三条件共同有效 coverage、
  raw-geometry-available 比例、score-match inclusion、mean raw objectness gap、train pair
  数与类别平衡。
- **选择与显著性：** 一次固定 readout；完整 259-scene image-paired bootstrap，10,000
  次、seed 20260919。非 target scene 以零 pair 纳入重抽。
- **不变量 / 遥测：** 每个 readout scene 在 dense-RPN/UAV/sham 前后运行标准
  `model.predict`，bbox/scores/labels/masks typed hash 必须一致；post-NMS RPN readout
  前后 hash 必须一致；detector/UAV state hashes 不变。逐 target 保存 B0 row、post-NMS
  replay、dense positive/negative indices、boxes、IoU、raw objectness、三条件
  9-point supports、continuous readout scores、win、排除原因及 immutable pair SHA。
- **判定门：** 完整数据和全部不变量成立时，三条件共同有效 coverage `>=.80`、correct
  win `>=.60`，且 `correct-XY+80m`、`correct-scene-permuted` bootstrap 95% CI 的下界
  均 `>+.05`，才判 `FROZEN_UAV_RPN_PRESELECTION_SIGNAL`；否则
  `NO_FROZEN_SIGNAL`。任一 lineage/split/identity/parity/hash/data 完整性失败为
  `INVALID`。`raw_geometry_unavailable` 和 descriptor-invalid 必须单列，不能从分母
  静默删除。

### F.3 边界与归档

C1 能证明的上限是：冻结 UAV 信息能否在已存在的 dense raw RPN candidate 上提供与
targetness 一致、对 sham 有特异性的排序信号；它不能证明真实 end-to-end proposal
recall 或 AP 提升。B3 的 `NO_FROZEN_SIGNAL` 不可被 C1 改写；C1 的任何结论也不替代
B0 的 post-NMS recall 诊断。归档 summary、train/val pair CSV、bootstrap、run manifest、
log、独立实现/结果复核和结论 MD；失败与成功同级关账。

---

## 附录 G：立项记录 MHBEV-DIAG-20260917-D1（卫星端 RPN stage attribution；冻结、只读）

**状态：预注册；冻结诊断，不使用 UAV，不训练或修改 detector。** B3 与 C1 已分别
关闭 UAV→ROI score 与 UAV→dense-RPN objectness 路径。D1 不试图挽救它们，而是把 B0
固定的 107 个 post-NMS RPN-miss small target 按部署 RPN 的真实丢失阶段分解，给下一条
卫星端模型假设选择可证伪的对象。

### G.1 固定谱系与四个嵌套阶段

- **固定 target：** B0 `instances.csv` 中 `area_bucket=small`、
  `final_failure∈{missed,partial_or_merge}`、`rpn_recall_iou05=False` 的 107 个唯一
  `(scene_id,gt_index)`。B0 的 checkpoint、259-val/5,937-instance 完整性、normal-predict
  与 RPN-repeat hash，以及每个 target 的部署 post-NMS max IoU 都须 fail-closed 重放。
- **readout：** 冻结的 `extract_feat → rpn_head.forward`。所有 bbox 均以部署
  `DeltaXYWHBBoxCoder`、相同 anchors、相同 image metainfo decode；不向 sample 写
  proposals。对一个 target，四个嵌套、只读阶段的“对齐”定义为任一 box IoU>=.5：
  1. **raw dense：** 所有 level、全部 decoded anchors，未做 score threshold、`nms_pre` 或
     NMS；
  2. **deployed pre-NMS：** 逐位复现本部署版 `RPNHead` 的 sigmoid 与每 level `nms_pre`
     排序/截断；该版本 RPN 没有 `score_thr`，且其公开 readout 不支持 `with_nms=False`；
  3. **NMS-unlimited：** 同一 live pre-NMS 选择、live `min_bbox_size`/NMS 参数，但将
     `max_per_img` 设为严格大于 pre-NMS candidate 总数的安全上限（本版本的 `-1` 会切掉末项，
     因而禁止使用）；
  4. **deployed final：** 完全 live RPN `predict`，应与 B0 post-NMS replay 一致。
- **互斥穷尽 stage：**
  `raw_geometry_unavailable`（raw 不对齐）；
  `score_nms_pre_rejected`（raw 对齐而 deployed-pre-NMS 无对齐）；
  `nms_suppressed`（pre-NMS 对齐而 NMS-unlimited 无对齐）；
  `max_per_img_truncated`（NMS-unlimited 对齐而 deployed final 无对齐）。其余状态为
  replay/invariant drift，立即 `INVALID`，不得另归类。

### G.2 四要素、对照与判定

- **主指标：** 四类 target 的比例与最大类。
- **伴随指标：** 每类 target 数、raw/pre-NMS/NMS-unlimited/final 对齐率、每层 raw anchor
  数、live `score_thr/nms_pre/NMS/max_per_img` 合同、以及每个阶段的 max IoU。
- **选择与显著性：** 唯一全量 107-target 口径；完整 259-scene image-paired bootstrap
  10,000 次、seed 20260920，对最大类比例减其余三类总比例给 percentile 95% CI。
- **对照 / 不变量：** 每场景 normal prediction 在所有 readout 前后 typed hash 一致；
  deployed final RPN 与 B0 逐 target IoU 一致；重复 raw 和逐位复现的 deployed pre-NMS readout
  的 typed hashes 一致；四阶段的输入 feature/anchors/decoded raw boxes hash 一致，且仅
  阶段配置中预注册的 `nms_pre/NMS/max_per_img` 控制开关不同。任何 readout 写 proposals、
  违反 nested monotonicity 或 live-config drift 均 `INVALID`。
- **判定门：** 完整性与不变量均成立，且最大类比例 `>=.60`、其相对其余三类总比例的
  bootstrap 95% CI 下界 `>0`，才以该 stage 作为唯一 `D1_*` 合法结论；否则 `D1_MIXED`。
  诊断结论只授权后续针对该 satellite RPN stage 的**新**冻结干预立项，绝不授权训练。

### G.3 解释边界

D1 能定位 baseline RPN 在该固定 failure slice 上的阶段性断点，不能证明任何改动必然提升
proposal recall/AP，也不构成 UAV 负结论的替代。若是 raw geometry，则下一假设才可讨论
anchor/bbox geometry；若是 score/nms_pre，则只可讨论 satellite objectness/selection；若是
NMS 或 truncation，才可讨论保留策略。所有结果、CSV、bootstrap、run manifest、日志、
独立实施/结果复核与结论 MD 均须归档。

---

## 附录 H：立项记录 MHBEV-DIAG-20260918-D2（RPN rank-profile；冻结、只读）

**状态：预注册；模块叠加型路径的冻结诊断，不使用 UAV，不训练或修改 detector。** D1 的
59/107 `score_nms_pre_rejected` 并未达到“唯一主导 stage”的门；D2 只问一个更窄且可证伪的
问题：这 59 个 target 是否都只是靠近部署每层 `nms_pre=1000` 截止线的候选保留问题，还是
多数正确 anchor 排名远低于该边界。D2 不计算 AP、不改 deployed proposal，也不授权训练。

### H.1 固定谱系与 readout

- **固定 target / 对照：** 唯一输入为 D1 fullval 的 107 个 target CSV 和类别：59 个
  `score_nms_pre_rejected` 为主分析集；23 个 `raw_geometry_unavailable`、2 个
  `nms_suppressed` 与 23 个 `max_per_img_truncated` 是预先指定对照，target key 集须与 B0
  一致。D1 完整性、checkpoint 与 nine invariants 均须 fail-closed；D2 重放 B0 final RPN
  IoU 和 D1 的类别/阶段 IoU，漂移即 `INVALID`。
- **rank 契约：** 冻结 `extract_feat → rpn_head.forward`；每一 FPN level 逐位复现部署
  sigmoid 与 `torch.sort(descending=True)`。对一个 target 的每个 raw decoded IoU>=.5
  candidate，记录其 level-local one-based rank；`min_positive_rank` 为所有正确 candidate 的
  最小 rank。部署 anchor 的 selection 定义唯一为 `rank<=1000`，没有 score threshold。原始
  geometry 对照不得有正确 candidate；D1 的 nms/max 对照必须 `min_positive_rank<=1000`；59
  主分析 target 必须 `min_positive_rank>1000`。违反任一等价即 `INVALID`。
- **预注册反事实 cap：** `1250, 1500, 2000, 4000`，均为**每层** rank cap；仅以
  `min_positive_rank<=cap` 读取“有正确 candidate 可被保留”，不写 proposal、不跑 ROI。
  `cap=2000` 是唯一主判读 cap；其他 cap 仅为 rank-profile 伴随遥测。

### H.2 四要素、对照与判定

- **主指标：** 59 个主分析 target 在 cap=2000 的 candidate-recovery proportion。
- **伴随指标：** 主集 `min_positive_rank` 的 median/p90、各 cap recovery、正确 candidate
  所在 FPN level、rank 距 1000 的 margin；四个 D1 category 的 count 与三个对照合约通过率。
- **选择与显著性：** 唯一 59-target 主集；按 85 个 D1 affected scenes 成对 bootstrap
  10,000 次、seed 20260921，对 cap=2000 recovery proportion 给 percentile 95% CI；任何
  全零 draw 只重抽该 draw 并报告次数。无 best/last 选择。
- **sham / 不变量：** normal prediction 与 deployed RPN 在 readout 前后 typed hash 一致；
  feature/logit/anchor/decoded-box typed hashes 重复一致；部署 rank=1000 membership 与 D1
  pre-NMS IoU 分类精确一致；固定 negative control 是 raw-geometry 23 target 的零正确
  candidate，正向 separation control 是 D1 nms/max 25 target 的 rank<=1000。不得写
  sample proposals 或触及 UAV/detector 字段。
- **判定门：** 只有完整性与所有不变量成立、主集 cap=2000 recovery `>=.70` 且其 paired
  bootstrap 95% CI 下界 `>=.50`，才为 `D2_BOUNDARY_RECOVERABLE`，仅授权一个独立的
  satellite-RPN **config-only proposal-recall** 立项（仍不授权训练/UAV）。否则为
  `D2_BROAD_RANK_DEFICIT`；任何合约失败为 `INVALID`。该冻结结论绝不等同 bbox/segm AP
  改善。

### H.3 解释边界

D2 能证/证伪的是“固定 B0 RPN-miss slice 上的正确 anchor 是否靠近 per-level 保留边界”；
不能证目标检测最终指标、实际显存/吞吐可接受性、或 UAV 融合收益。若通过门，下一步也只能
以部署同权重的 cap config 做独立 proposal-recall/compute 对照；若未通过，候选 cap 扩张不应
成为下一条改动主线，应回到 anchor geometry/objectness 假设另立项。

---

## 附录 I：立项记录 MHBEV-DIAG-20260918-D3（卫星 RPN frozen objectness readout）

**状态：预注册；模块叠加型的冻结诊断，不使用 UAV，不训练或修改 detector。** D2 已证伪
“仅增加 per-level candidate cap”能恢复大部分 D1 59 个 `score_nms_pre_rejected` target。
D3 问更严格的问题：冻结的卫星 FPN 特征是否仍含有、但当前 RPN score 未有效使用的 targetness
信号，足以在**raw-score-matched、同 level** hard pair 中把正确 candidate 置于背景 candidate 之上。

### I.1 固定谱系与 descriptor

- **主验证集：** D1 fullval 的 59 个 `score_nms_pre_rejected` target，key 和 D1 四 stage
  IoU / category 逐一重放；B0→D1→D3 不一致立即 `INVALID`。每个 target 取 raw decoded
  IoU>=.5 candidate 中 raw score 最高者为正例；在相同 FPN level、与任一 GT IoU<.1 的
  candidates 中取 raw score 绝对差最小者为负例。仅 `|score_pos-score_neg|<=.02` 的 pair 进入
  判读，报告 coverage 和 gap；pair identity（scene/gt/level/index/box/score）哈希冻结。
- **训练集：** 唯一的既有 128-scene train subset（ratio=.1797752808988764, seed=44），和
  val 无 scene 重叠。其所有 small GT 按同一 level、raw-score 最近的 pair 规则构造冻结
  descriptor 二分类样本，**训练不施加** val 的 `<=.02` score-gap 纳入阈值（否则监督可能
  稀疏到无法拟合）；score 平衡只由固定 59-target val 的主判读保证。不从 val 拟合任何
  scaler/classifier 参数。
- **descriptor：** 当前 RPN 输入的对应 PAFPN level feature map，在 anchor 所属 spatial cell
  抽取的 `rpn_head.in_channels`-d feature（该基线为 512-d；禁止误取 backbone 的 256-d map）；anchor-grid 与 feature H/W、每 cell anchors 及 index→cell 映射
  必须 cardinality/assert 对齐。模型只拟合
  `StandardScaler + LogisticRegression(C=1, class_weight=balanced, max_iter=1000,
  random_state=20260922)`，绝不反传或写 detector/RPN proposal。

### I.2 四要素、对照与判定

- **主指标：** 59-target val 的 raw-score-matched valid pairs 上，冻结 satellite descriptor
  classifier 的 pairwise win rate（正例概率 > 负例概率）。
- **伴随指标：** pair coverage、score gap、训练 pair 数/正负平衡、raw score-only tie baseline、
  正例的 level/rank，及各条件的准确率。
- **sham / 显著性：** 使用同一训练 feature 和 pair，但在 train 内以 seed 20260923 全局置换
  标签拟合 sham classifier；val 按 85 个 D1 scenes paired bootstrap 10,000 次、seed 20260924，
  报告 correct-minus-sham accuracy 95% CI。normal prediction、RPN、feature/descriptor、pair
  identity 的 typed hash 在 readout 前后/重复时必须一致；raw score-only tie baseline 与 label
  permutation 都不得系统性偏向正确方向。不得使用 UAV 字段。
- **score-match 标定门：** 同一 85-scene bootstrap 还计算 raw-score win rate 减 .5；其 95%
  CI 上界必须 `<=.15`，否则认定 `|score_pos-score_neg|<=.02` 未形成足够的 raw-score
  平衡，结果 `INVALID`。sham 须归档原/置换 label typed hash、class count，并 fail-closed
  确认置换不是恒等；anchor index→cell 用 `sparse_priors` 与 `grid_priors` 在运行时逐位核验。
- **判定门：** 完整性/所有不变量通过，valid coverage `>=.80`（>=48/59），正确 win rate
  `>=.65`，且 correct-minus-sham paired-bootstrap 95% CI 下界 `>0`，才为
  `D3_FROZEN_SAT_OBJECTNESS_SIGNAL`；否则为 `D3_NO_FROZEN_SAT_OBJECTNESS_SIGNAL`。正结果
  仅授权“satellite RPN objectness head/quality-loss”的新模块立项与热启动筛查，绝不证明 AP、
  不授权全量训练或任何 UAV 改动；负结果则关闭该 frozen satellite readout 路线。

### I.3 解释边界

D3 只能测试冻结 FPN 内容是否存在超出已匹配 raw score 的线性可读 targetness，不测试训练动力学、
anchor 几何或检测终局指标。D3 不是 D2 candidate cap 的替代验证，也不能挽回 B3/C1 的 UAV
负结果。

## 附录 J：立项记录 MHBEV-FUS-20260919（UAV 边际信息冻结探针；冻结、只读、不训练 detector）

**状态：已执行；结果已落盘，待独立复核（见 `docs/reviews/MHBEV-FUS-20260919_probe_results.md`）。**
本探针由 dual-stream 工作树会话发起，回答 r2/r3（FPN P2/P3 与 image-embedding 零门控注入）
parity 之后"采取何种融合方式"的决策问题。不训练、不修改任何 detector 或数据；只加载既有
checkpoint 做冻结前向与线性读出。

### J.1 谱系与输入

- **卫星特征谱系：** `ue_single_c5v2_control_tr128_va259_s44_r1/best_model_epoch86.pth`
  （原始数据单流对照，best segm 0.7312）。
- **双流分支谱系：** `ue_dual_c5v2_p2p3_emb_tr128_va259_s44_r1/best_model_epoch85.pth`
  （r3，best segm 0.7306；geometry_bev/drone_encoder/embedding_gate 权重）。
- **D2 臂：** 卫星特征改用 `ue_d2_single_tr128_va259_s44/best_model_epoch83.pth`，
  数据根 `/data1/wangcheng/dataset/ue_coco_with_uav_d2`；UAV 与标定不变。
- **划分：** 训练拟合 = 128-scene train subset（ratio=.1797752808988764, seed 44）前 80 场景；
  评估 = val 前 60 场景；与全部既有臂的 train/val 隔离一致。
- **实现：** `portable_sam2_explicit_coarse/scripts/ue/probe_fusion_information.py`
  （commit `d40be42`）。特征：SAT=卫星 P3 512-d；LBEV=学习式 GeometrySparseBEV 128-d；
  DBEV=免学习直接几何采样 BEV（4 level×256-d）；DBEVmis=地面点绕场景中心旋转 90°
  的标定错配负对照。探针=逐 cell 线性读出（标准化 + 单层 Linear，balanced 1:1 拟合）。

### J.2 四个臂与预注册判读

- **E1 占据率读出（原始数据）：** 逐 cell 建筑占据二分类。判读：SAT+BEV 相对 SAT 的
  逐场景 AP 差（paired）是否显著>0 —— 回答"UAV 是否有边际信息"。
- **E1-D2（降质数据）：** 同上，卫星支路换 D2 谱系。回答"降质是否制造可填补缺口"。
- **E2 门梯度 SNR：** 冻结全部参数、仅放开三个 γ，在 10 个真实训练 batch 上取 dL/dγ；
  与 weight-decay 拉力 `wd·|γ|` 比较。回答"零门能否自举"。
- **E3 注入空间结构：** 逐 cell 注入残差幅度与 GT 占据的秩相关、与基底特征的方向余弦。
- **E4 边界距离回归（后补，占位注册于结果落盘时）：** 逐 cell 到建筑边界的符号距离回归
  （GT 距离变换 /8），逐场景 Pearson r。回答"UAV 的边际信息是否在边界通道"。

### J.3 结果摘要（详见 review 文件）

E1：SAT AP .8848；SAT+DBEV ΔAP=+.0019（p=.10，n.s.）；DBEVmis .3268 vs DBEV .5395
（几何依赖成立）。E1-D2：SAT .8744；Δ=+.0017（p=.28，n.s.）。E2：三门梯度 SNR
.10–.59，低于 wd 拉力 1–2 个数量级。E3：注入幅度与占据 Spearman −0.27。E4：SAT
r=.519；SAT+DBEV r=.611，Δ=+.0919（t=13.27，p=2.3e-19，58/60 场景改善）。

### J.4 解释边界

探针只测线性可读信息与单谱系 checkpoint，不构成端到端增益承诺；E4 的正结果支持
"BEV→边界精修通路 + 非零门初始化"的机制方向，但不保证检测器训练后增益成立。在任何
后续实现/训练被授权前，本附录与 review 文件须先通过独立复核。
