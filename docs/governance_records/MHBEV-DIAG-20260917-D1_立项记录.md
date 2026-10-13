> 2026-10-13 自 EXPERIMENT_GOVERNANCE.md「附录 G（迁出时第 571–627 行）」原文迁出；除本行外内容未改。

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

