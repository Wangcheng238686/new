> 2026-10-13 自 EXPERIMENT_GOVERNANCE.md「附录 B（迁出时第 270–323 行）」原文迁出；除本行外内容未改。

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

