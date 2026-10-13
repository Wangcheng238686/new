> 2026-10-13 自 EXPERIMENT_GOVERNANCE.md「附录 I（迁出时第 680–731 行）」原文迁出；除本行外内容未改。

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

