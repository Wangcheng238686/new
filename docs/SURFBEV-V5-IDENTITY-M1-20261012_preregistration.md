# V5-M1：全量派生候选标签物化（未放行）

目标是先建立**1,200 场物理文件和完整训练元数据**的候选版本，解决已审 V4 的 1 个稳定误并（17 个 val 拆分场景）与 V5-R6B 的 18 组全部来源场景通过的身份串号（24 个来源组件场景）。它不是“误并已全清”的发布版。输入锁定：v3 manifest SHA `8d3da5addd302451337954191dccbf1b76f588a0790f1073415e30439015922f`；v4 manifest SHA `49b7f9268db3f767cef4d5cd56265d674da59c54808ca9650140d2d3c5e1884f`；R4 身份清单 SHA `a5346fa10ed2cb26c37c25174e1efe83c6b628600764c4f8ef3a4734eae04975`；R6B 逐格清单 SHA `910a1525c0b04566936032d8233b7dc5e2fc9342a1e5a227073f2be649294bc7`。

每张图以 v4 PNG 为底，train/val 已通过身份门的孤立组件从原 local ID 中拆出，赋不与**原 `instances.json` 声明 ID、v4 图内 ID、同景新 ID**冲突的 local ID；其 global ID 指向跨景一致的**已有**目标建筑，不能捏造 child。V4 的两个 child global ID 原样保留。输出每张 512² 派生 PNG、仅可见实例的 local/global 映射及 bbox/area、来源和输出哈希、操作明细。test/excluded 不用来定规则；同规则覆盖全部 1,200 场，未命中者保留像素。

硬不变量：1,200 场唯一覆盖；所有派生前景与 v3 完全相同；v4 之外的像素变化只发生在 R6B 全来源场景 PASS 的组件；非目标像素不变；local ID 唯一且映射完整；global ID 不跨 train/val/test；SHA 可复算。`532000→402005` 整对 HOLD，V4 excluded 的 6 个 HOLD 原样记录，95 个相连形状风险 global 仍待判定。首次产物状态 `BUILT_UNREVIEWED`，不接训练 reader、不运行模型；独立审计和固定视觉抽样通过后才讨论发布。

## M1 失败与 M2 技术修订

M1 物化到 `Coord_003085...` 时因两个不同来源 ID `102003`、`131004` 均应归到同一目标 `106001` 而被“目标已出现”保护条件停止，留下部分写出，未生成 manifest，不可使用。M2 保留这一失败目录，在新输出目录把**同景同一目标 global 的多个纠错组件合并为同一个新 local ID**；若目标在原底图已可见仍报错。其他输入、18 组准入、像素与 split 规则不变。
