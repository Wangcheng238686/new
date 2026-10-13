# V5-M3 第一轮身份纠错候选：独立审计结论

**技术审计通过，语义发布未通过。** M3 覆盖完整 1,200 场（train 694、val 100、test 151、excluded 255），每场派生 PNG 与可见实例 local/global、bbox、面积元数据及 SHA 可复算。以已审 V4 为底，18 组跨场景身份纠错中有 23 个来源组件最终重赋到已有目标 global，覆盖 21 场、13,977 px；另一个来源组件仅 1 px、距图边 1 px，被数据侧删除，不再制造 1 px 训练实例。已审 V4 的 17 场 val 物理拆分保留。M3 相对 M2 除这一像素外，其余 1,199 场 PNG/metadata 内容完全相同；保留新增实例最小 133 px。

独立全量审计确认：1,200 个 PNG/metadata/manifest 哈希正确；action 变更域、可见 local 唯一性、global 映射、bbox/面积、train/val/test 跨 split 身份隔离均通过。`summary.json` 的 `identity_source_scene_components=24`、`identity_scene_count=22`、`identity_pixels_changed_from_v4=13978` 包含 **1 个 DROP**；真正的重赋口径是 23 个组件、21 场、13,977 px，另删 1 px。原始、V3、V4 与 M2 均保留；没有改变默认 reader，也没有训练/AP 结论。

不放行原因：`532000→402005` 一对仍 HOLD；V4 的 6 个 excluded 场景仍 HOLD；相连但多栋的风险还没有可验证的自动分界规则。R3 的 95 个 connected-risk global 只是形状筛查，固定全体代表图视觉检查显示许多是合法屋顶条带/裁边，不能按低填充率批量拆；它也不是所有相连误并的完备检出。故 M3 是**文件完整、可复核、尚未达到“一栋物理建筑一个实例”验收门**的候选集，不应作为默认训练数据。
