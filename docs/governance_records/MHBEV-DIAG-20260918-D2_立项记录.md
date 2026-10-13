> 2026-10-13 自 EXPERIMENT_GOVERNANCE.md「附录 H（迁出时第 628–679 行）」原文迁出；除本行外内容未改。

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

