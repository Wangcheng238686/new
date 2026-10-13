# SURFBEV-MASK-V3-20261012：数据侧自动清洗结论

## 判定

**v3 数据构建门和预注册抽样视觉门通过；可作为下一版数据输入候选。**
这不是模型 AP 提升结论；尚未改 B0 默认 reader，也未启动 v3 训练。
`summary.json` 中的 `BUILT_UNREVIEWED` 是物化时冻结的状态快照；事后审计和抽样判定记录在本文，不回写原始产物。

用户要求的流程已转为：自动规则覆盖全量数据，人工只看固定分层抽样，模型侧不做 `IGNORE`。原始数据保持只读，派生 v3 标签在独立目录中物化；不逐例人工审批。旧的 113 例全量人工审阅前置要求被 v3 预注册替代。

## 本次范围

规则在原始 512² ID PNG 上判定：只有 `touch_border AND (area < 16 OR min_bbox_side <= 2)` 的 local ID 被整实例置为背景 0，其余像素逐位不变。1,200/1,200 场成功物化。train/val/test/excluded 分别剔除 569/44/165/294 个实例；合计 1,072 个、21,360 像素。test 只被确定性处理，没有用于定阈值或抽样视觉评判。

独立审计对全部 1,200 场重新逐像素计算，0 契约错误；源和派生哈希、删去 ID 与像素数均对得上。预注册固定种子抽取 48 场 train/val（每个 split 的变更/未变更各 12 场）。已查看 24 个变更场景的完整图与代表被删除 ID 的局部前后对照，以及 24 个未变更场景的保留屋顶前后对照；所见被删区域是图边极薄/极小可见片段，未见完整可辨的小楼被删；未变更对照中完整屋顶保留。多 ID 场景只对面积最大的被删除 ID 做单独放大，其他被删 ID 在完整图标记，因此这不是逐 ID 人工复核，也不能推断总体误删率为 0。

## 产物与后续边界

- 清洗标签与逐场来源/输出哈希：`portable_sam2_explicit_coarse/outputs/SURFBEV-MASK-V3-20261012/labels/`、`manifest.jsonl`、`summary.json`。
- 48 场抽样前后图：`portable_sam2_explicit_coarse/outputs/SURFBEV-MASK-V3-20261012/spotcheck_visual/index.html`。
- 自动构建脚本：`portable_sam2_explicit_coarse/scripts/ue/build_surfacebev_mask_v3.py`；抽样渲染脚本：`portable_sam2_explicit_coarse/scripts/ue/render_surfacebev_mask_v3_spotcheck.py`。
- 独立审计：`docs/reviews/SURFBEV-MASK-V3-20261012_independent_audit.md`。

规则只能清除高置信的图边碎片，不能解决全部标注问题，例如一个 local ID 涵盖多块屋顶的物理归属。v3 中被移除的真实可见碎片不再是 GT；模型若预测它们可能记为 FP，这是新的任务定义代价。v2 与 v3 的 AP 因 GT 改变不能直接解释为模型改进。下一步若要训练，应先单独登记 v3 B0，令数据 reader 显式读取这批派生 PNG，完成代码提交/审核后再启动；不能沿用旧 reader 悄悄评测。
