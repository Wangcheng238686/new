# 历史方案归档

以下文档已被《prompt 消费鲁棒化与 P2 边界提示联合改造实施指南》的
"当前执行决定（2026-09-04）"覆盖或降级，仅作决策历史保留。

- `v2_prompt_mining_plan.md`：V2 prompt mining 实施方案 v2.4（S1→S4 门控路线）。其
  PromptRobustifier / P2PointRefiner 新模块路径被 guide 的 D1/D2 → 四行矩阵路线取代，
  保留为历史候选（见 guide「当前执行决定」末段）。经 4 轮外部评审收敛，评审史见
  git log 962f5e0..84f2395。
- `p2_point_refiner_design.md`：P2PointRefiner 设计（2026-09-03 降级）。离线学习探针
  仅有 -3.1% 点误差改善（logs/learn_probe/fit_report.json），不满足直接实施门槛。

主线以 `docs/prompt_consumption_and_p2_refinement_implementation_guide.md` 为准。
