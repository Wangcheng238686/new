# DUAL-P2-READCAP-20261014：冻结特征读出容量探针（预注册）

## 路由与谱系

A 路线（模块叠加型）谱系上的冻结诊断探针：不训练 detector、UAV encoder 或任何
融合模块；不重新 harvest；不读取 test。上一级 SURFBEV-P1-V2（2026-10-13 落判
P1_FAIL）为同锚点同特征的前置信息门；本探针是它失败后的定向复盘，不修改 P1
的任何门或结论，不重复利用其 test 选参。

## 动机与假设

P1-V2 六项门中四过两挂：挂的是 `correct−SAT` 的 error-AUROC（均值为负）与
`correct−sham90` 的 IoU-Spearman（CI 跨 0）；而 boundary-F1 两门干净通过且
sham90 反而低于 SAT——几何特异信号存在但两项伴随读出受限。假设 H-P2：
**线性读出（StandardScaler+Logistic/Ridge）容量不足是主因**；在完全相同的
冻结 64 维三臂特征上，把读出换成容量受控的非线性 MLP 后，六项门将在 val 上
全部恢复为 CI 下界 >0。

## 冻结输入

- 唯一输入：`/data1/wangcheng/checkpoint/surfbev_b0/probes/SURFBEV-P1-V2-R1a-20261012/cache/`，
  其 `contract.json` SHA256
  `af04fc445252b4b8902636eabe975559bfc18a86b21066841aeb4d3afb2e1732`；
  train 694 / val 100 个逐场 npz，特征与标签逐字节复用（三臂 SAT/correct/sham90，
  boundary/error 各 512 采样 cell + iou 逐实例池化，val 另有全 16384 cell 特征
  `x_*`、全标签 `yb/ye`、预测 `pred` 与真 IoU `qi`）。缓存目录只读。
- 锚点身份由该 contract 链条保证（B0 = EXP-0006 seed44 last，checkpoint SHA
  见 contract）；本探针不再加载任何模型权重。

## 划分与选择（全部不触碰 val）

- train 694 场以 seed 20261014 固定置换：**fit 555 场 / select 139 场**；val
  100 场仅作最终一次性判定。划分按缓存文件序（scene index）+ 种子置换产生并
  落盘 `split.json`（含 scene_id 双向查重，fit/select/val 三集交集必须为空）。
- 读出模型：`StandardScaler（只在 fit 上拟合）→ MLP`。分类目标（boundary、
  error）BCEWithLogits，回归目标（iou）MSE。结构网格：hidden ∈ {(64,), (128,),
  (64,64)}，weight decay ∈ {1e-4, 1e-3}，共 6 配置；Adam lr 1e-3、batch 4096、
  最多 50 epoch、每 epoch 在 select 上评损失、patience 5 早停；torch 种子
  20261014。**每个目标选一个三臂共用的配置**（按 SAT 与 correct 两臂 select
  损失的均值取最小；iou 目标同理）。三臂结构、优化器、种子完全相同。
- boundary 阈值：三臂共用，在 select 上按 correct 臂**患病率校正 F1** 最大选取，
  网格 {0.1,…,0.9}。校正公式（平衡采样下估计自然患病率的 F1）：
  `R=TP/(TP+FN)`、`FPR=FP/(FP+TN)`（select 采样 cell 上逐阈值计算），
  `π=select 各场 yb 均值`（只读标签），
  `F1=2R / (2R + (1−R) + ((1−π)/π)·FPR)`。P1-V2 在 val 上选阈值，本探针更严。
- **parity 线性锚**：同 fit/select 重跑 P1-V2 式读出（Logistic C∈{0.01,0.1,1,10}
  / Ridge α=1/C，同规则选择；阈值同上）。目的：容量对照与防退化校验。

## 指标、显著性与判定门（val 100 场，一次性）

三指标与 P1-V2 逐字一致：每场 boundary-F1（16384 cell，共用阈值）、error-map
AUROC（同 cell 集）、逐实例 IoU 预测 Spearman（池化特征→回归→与 qi）。
scene-paired percentile bootstrap 10,000 draws、seed 20261014，六组差值
（correct−SAT、correct−sham90 × {f1, auc, rho}）95% CI。

- **P2_PASS**：六项 CI 下界全 >0。含义：几何信息在冻结表示中可被非线性读出
  完整恢复 → 放行 DUAL-H1 型热启动筛查（learned 融合模块 + 强制参与度遥测）。
- **P2_PARTIAL_BOUNDARY**：f1 两门过、auc/rho 至少一门未过 → 几何信号仅支持
  边界/掩码精化路线 → 下一步只预注册边界精化方向的条件化筛查。
- **P2_PARTIAL_QUALITY**：rho 两门过、f1 未全过 → 仅支持实例质量预测路线。
- **P2_NULL**：无任何同指标两门同过 → 信息不在当前 64 维冻结表示中 → 后续
  探针必须改 harvest（升维/分通道/实例级），禁止继续在此读出上调参。
- **READCAP_DEGRADED**（完整性哨兵）：MLP 的 SAT 臂 val f1 < 线性锚 SAT 臂
  val f1 − 0.005 时判定失效——容量增大却把基线读垮，任何"恢复"不可信，
  结论只报 DEGRADED 不报上述四判。

## test 政策与历史如实登记

本探针**不读取 151 场 test**（P1-V2 的一次性 lock 不受影响）。val 100 场在
P1-V2 中曾被用于 C/threshold 选择（历史事实登记）；本探针选参全在 train 内部，
val 仅一次判定。任何 AP 级或读出级的 test 开封须另案预注册。

## 遥测三件套

1. 判读边界：三项指标仍是冻结特征可读性度量，非 detector AP；MLP 对线性的
   差别只能解释为读出容量差别，不能解释为"信息变多"。
2. 标定校验：parity 线性锚与 MLP 共用划分/阈值规则；选择只动 select；val
   只评一次；三臂同结构同种子。
3. 机制不替代结局：即使 P2_PASS，只放行筛查，不声称 mask/bbox AP 收益。

## 勘误（2026-10-14，正式 run 前、任何 val 结果产生前，依首次定版审计）

1. **判定格补全（审计 E3）**：原四判未穷尽 (f1,auc,rho) 三对门的 8 种组合。补全
   判定格：设三对门各自通过状态为布尔三元组 (f1,auc,rho)。
   - (1,1,1) → P2_PASS；
   - (0,0,0) → P2_NULL；
   - 其余六种 → `P2_PARTIAL(<通过的门列表>)`，如 (1,0,1) → P2_PARTIAL(f1,rho)。
     语义：通过的门所对应的路线获得冻结表示层面的支持，未通过的门对应路线
     不获支持；下游预注册只允许瞄准通过的门的路线。原 P2_PARTIAL_BOUNDARY /
     P2_PARTIAL_QUALITY 分别等价于 (1,0,0)/(0,0,1)，保留名称兼容。
2. **种子登记（审计 A6）**：实现中 MLP 配置网格的 torch 种子为
   `20261014 + 10·hidden_index + wd_index`（无碰撞；三臂同目标同配置共享同一种子
   以保证配对），与本预注册"torch 种子 20261014"的表述存在偏差，如实登记，
   不改变任何判定条款。
3. **审计处置记录**：F3（split.json 补 val 名单 + 三集交集断言）、C3（空预测
   实例防护，与 P1 口径逐字对齐）、F6（out≠cache 拒绝、线性锚全网格选择轨迹
   落盘、死代码清理）已实现；独立审计全文见
   `docs/reviews/DUAL-P2-READCAP_implementation_review.md`，VERDICT: APPROVED。

## 不变量与工件

- cache 只读；fit/select/val 三集 scene_id 交集为空；每臂每目标的选择轨迹与
  select 损失曲线落盘可审计。
- 工件目录：`/data1/wangcheng/checkpoint/surfbev_b0/probes/DUAL-P2-READCAP-20261014/`
  （`split.json`、`selection.json`、`telemetry_fit.jsonl`、`summary.json`、
  `per_scene.jsonl`、`run.log`）。结论 MD 与独立实现审计（首次定版节点）
  落 `docs/`、`docs/reviews/`。
