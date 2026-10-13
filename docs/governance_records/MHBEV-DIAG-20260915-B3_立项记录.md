> 2026-10-13 自 EXPERIMENT_GOVERNANCE.md「附录 E（迁出时第 440–501 行）」原文迁出；除本行外内容未改。

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

