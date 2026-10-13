# V6 单图实例审图轮：独立只读复核

**VERDICT: PASS for diagnostic provenance/rendering; PENDING for label semantics and release.**

独立复核者未编辑脚本、标签或审计对象。复核对象是 `portable_sam2_explicit_coarse/scripts/ue/render_surfacebev_v6_intraimage_review.py`、其 `outputs/SURFBEV-V6-INTRAIMAGE-REVIEW-20261012/index.json` 和对应 PNG。脚本改进 `003085` 近景与新增 `003092` 重叠视角后，复核者再次审查关键分支。

## 核对结果

- M3 manifest SHA `3e36bd8ac37dc1e0fb60bfa2182eac407fb6f8eb3e5353d6adda3ce3916e1261`、R3 risk SHA `df57b4fc3ca65deae52f7a953ef483dcddd2945f02d16f0b75e1451ab5bf816d` 与索引一致；M3 manifest 中 1,200 个 V4/V5 PNG SHA 均核对通过。
- 22 场有 action，21 场改变单图实例 partition，1 场仅前景变化；后者 `002894` 在 `(x=4,y=1)` 删 1 px，原 local 26 的其余部分改为 local 38。结果与索引一致。
- `003085`：旧 local 4 的 514 px 与旧 local 44 的 461 px 合到新 local 61，总 975 px；近景的青、粉、黄三色与数据一致。两旧来源在图像右边缘相邻；视觉仍不足以单独断言物理归属。
- 新增 RGB 模板匹配完全不使用标签作对齐：`003085→003092` 得分 0.888076、平移 `(-110,-55)` px；映射 975 个来源像素在邻图中 711 px 落 local 11、264 px 落背景。两图均为 train。该结果仅辅助人工审图：显示的两个 crop 分别居中，不是逐像素注册的并排图，也不是物理身份真值。
- 脚本只读取 V3/V4/M3 PNG、RGB 与只读 JSON；只写 V6 审阅目录中的 PNG/index，未改原标签、reader、split 或模型。

## 选择与解释限制

- C01–C12 是 R3 中按面积选择的非裁边、不同场景代表，非随机总体样本；N01–N12 按随机场景顺序抽取合格的单连通未命中实例，11 train/1 val，非实例均匀抽样。两组都读 V3 标签作风险排查，**不能称为 M3 的语义验收或误并率估计**。
- 四张 `uncertain_connected_*.png` 是初筛后探索性挑出，非预注册概率样本。形状命中多数看似正常狭长屋顶，不能依据填充率直接批量拆分。
- 本次不对模型 AP、清洗完成率或“一栋一实例”发布作正面结论。尚需用户审 `003085`；其他 M3 改动的语义与相连误并召回仍待后续固定抽样/规则审计。
