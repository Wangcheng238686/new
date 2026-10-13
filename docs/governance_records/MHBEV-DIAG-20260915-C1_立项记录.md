> 2026-10-13 自 EXPERIMENT_GOVERNANCE.md「附录 F（迁出时第 502–570 行）」原文迁出；除本行外内容未改。

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

