# 既有产物相邻建筑诊断：计算前协议

2026-10-16。仅CPU、既有153验证图、四套冻结EMA预测（single_best、dual_correct、dual_zero、dual_shift）；不作新forward/拟合/训练。该开发子集已用于选择权重，不是独立test。本诊断不重新选择模型，不替代原COCO AP。

## 固定定义

- GT为已有clean可见instance mask，不是地面足迹。每图不同global ID的无序对；最近前景像素中心欧氏距离d。接触：d≤sqrt(2)（8邻域）；近邻：sqrt(2)<d≤5px。两层单列及合计。不根据结果调阈值。GT零重叠须核对。保留全mask；同时记录两栋最大8连通分量是否仍在5px内，检查碎片敏感性。
- 米数仅名义Z=0平面换算：卫星高度/fx或fy；核对俯视、方形像素。不是屋顶间真实距离；d为像素中心距离，不是连续边缘间隙。
- 主描述量：IoU≥0.5的图内全局一对一mask匹配后，两栋均对应不同预测的比例；伴随0.75、bbox匹配、非邻接GT、原AP。匹配先最大化合格边数量，再最大化IoU和（合格边权N_GT+1+IoU、dummy为0、不合格为负值），同分由固定输入序决定。另存仅该对的两预测可行性，暴露第三栋竞争影响。
- 所有保存的生产最终预测均参与，不事后调score；核对每图数量≤100以对齐既有top100边界。mask IoU用RLE，bbox为COCO xywh。
- 失败对分解：mask配对成功；mask失败且bbox配对成功；两者均失败。另单列mask成功但bbox失败，不把差异硬解释为因果。
- 重叠标签：一个预测覆盖两栋各≥50% GT像素（跨栋覆盖，非必然错误合并）；bbox各自匹配后，对应mask覆盖邻栋像素比例（分母邻栋GT面积），同时报告≥10%和≥50%、连续均值；两栋各自框已匹配但对应mask任一IoU<阈值。
- 候选摘要以任一对成员在RPN/raw/eligible/final的best bbox IoU是否跨过固定阈值记录：RPN无覆盖、regression后丢覆盖、score后丢覆盖、NMS后丢覆盖、最终独立最佳覆盖但一对一不足。它们可以重叠；不能恢复全部候选身份，不能声称pre-NMS可一对一分开或已识别直接抑制者。
- 分母包括图像数、pair appearances、unique physical pair数、GT appearances、unique building数。重复物理对先对其出现等权平均再对物理对等权，仅作为敏感性描述；不同物理对/图仍可能相关。
- 单流→双流正确的改善/退化为同一pair配对计数。图像配对bootstrap固定seed20261016、2000次，对pair appearance比率差报告95%探索性区间；不能覆盖跨图重复建筑相关性、seed不确定性和开发集选择偏差，不作显著泛化主张。
- 典型图按类别各选字典序第一对（image_id,GT slots）：双流改善、双流退化、两者失败且dual bbox成功、两者失败且dual NMS覆盖丢失、两者失败且dual RPN缺覆盖、两者成功。只用于检查定义/展示，不代表频率。

## 身份和验收

四臂GT/图像ID须一致，GT mask与clean标签逐像素一致，stage bbox与GT框及global映射一致；固定JSON SHA256保存。所有配对assignment唯一且满足阈值；合成例检查一个大预测不得匹配两栋、最大匹配数量优先于IoU和。统计独立审核后归档结论。候选原缓存信息不足的地方如实留缺口，不追加GPU探针。
