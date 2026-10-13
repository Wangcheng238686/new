# V6 单图实例审图：用户判读记录

## 003085

- 场景：`Coord_003085_pt_gx_7_gy_-2_s_014_gx_7_gy_-2_s_14_X_157500_Y_-30000`，train。
- 待判问题：V4 local 4 中的 514 px 与 local 44 中的 461 px，能否在 M3 合并成新 local 61（975 px）。
- 用户在本轮审阅了两源组件放大图与重叠场景 `003092` 对照后答复：**“003085看起来算同一栋建筑”。**
- 记录判定：`ACCEPT_M3_MERGE_FOR_003085`。本条只批准单图建筑实例语义上的此次合并；不把 RGB 匹配分数或邻景同 ID 当作独立真值，也不推及其他 20 个 partition 改动。
- 审阅图 SHA256：`m3_003085_two_sources_detail.png` = `40cecdc156c0db3d973733124ad419ce6efda4208507b2a0423bdd075c91e1ef`；`m3_003085_overlap_003092.png` = `4d2b84d573adc771120633f8f31a8480ee61e331166c1a2f7730072762a0ecac`。
- V6 审阅索引 SHA256：`a9b372bd2f9e212477df8baf0f4f59ac61a2a766f0aecb1f524de965aec2159d`。

## 使用边界

本记录不改 M3 PNG/metadata、原始数据、reader、split 或模型。六个 V4 excluded/HOLD 场景的单图掩码保持不变。M3 仍是候选标签：其他图内 partition 改动、相连误并召回以及固定分层视觉验收尚未完成，不能据此宣布完整可训练清洗版或启动新 B0。形态学修补仍另立轮次。
