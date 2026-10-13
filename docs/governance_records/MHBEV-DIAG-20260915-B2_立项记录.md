> 2026-10-13 自 EXPERIMENT_GOVERNANCE.md「附录 D（迁出时第 381–439 行）」原文迁出；除本行外内容未改。

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

