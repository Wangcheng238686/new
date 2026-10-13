> 2026-10-13 自 EXPERIMENT_GOVERNANCE.md「附录 C（迁出时第 324–380 行）」原文迁出；除本行外内容未改。

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

