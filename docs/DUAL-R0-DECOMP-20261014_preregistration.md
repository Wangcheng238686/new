# DUAL-R0-DECOMP-20261014：AP 亏损分解与三预言机上限探针（预注册）

## 路由与谱系

A 路线冻结诊断探针，不训练任何模型、不读 test。承接 DUAL-P4 决议与用户决定：
**双流期望 = bbox AP 与 mask AP 双指标提升** → RoI-only 条件化路线因无法触及检测
召回上限而出局；剩余候选路线（UAV 提案增补 vs 其他）的选择由本探针的分解证据
决定。锚点与 DUAL-P2/P3/P4 同源：EXP-0006 seed44 last（SHA `fc3ead0e…`），
推理在复原的 R1a 运行时（27b42d1 工作树，加载方式同 P4 预注册补充）。

## 数据与口径

- v2 split **val 100 场**（历史使用如实登记：B0 早停 + P1/P2/P3/P4 判定；本探针
  无任何选参动作，纯诊断）。test 151 场全程不读。
- 评测：pycocotools COCOeval，bbox 与 segm 两个 modality，单类（category_id=1），
  AP[.50:.95] 为主、AP50/AP75 伴随，按实例面积三分位（small/all）分层。
  GT box 由 GT mask 派生（一笔画包络），iscrowd=0。
- 匹配：标准 COCO 贪心（按分数降序、IoU≥0.5 起逐档），bbox 与 segm 独立匹配。

## 测量与三个预言机（全部为确定性上限，非统计估计）

一次锚点推理（val 100 场，落盘 box/score/mask/GT）后：

1. **基线**：锚点 val bbox AP 与 segm AP（及 small 分层）。
2. **miss 分解**：IoU=0.5 下未匹配 GT 实例的数量与面积分层。
3. **预言机 A（召回上限，route-2 相关）**：为每个未匹配 GT 注入完美预言机预测
   （box/mask=GT，score=s），重算 AP；ΔAP_miss = 召回补全的天花板。
   headline 固定 s=0.7；s∈{0.9,0.5} 作敏感性描述。
4. **预言机 B（排序上限）**：已匹配预测的 score 全部置 1.0，重算 ΔAP_rescore。
5. **预言机 C（掩码定位上限，route-1 相关）**：已匹配预测的 mask 替换为 GT mask
   （box/score 不变），重算 ΔsegmAP_mask。
6. **UAV 两视角可见率（miss 质量的可兑现份额）**：对每个未匹配 GT 实例，将其
   卫星足迹按 360m tile 契约反投影到世界 z=0 平面，再投到 5 路 UAV 视图；某视图
   "可见" = 足迹投影像素中 ≥5% 通过 DeviceZ 深度测试（射线深度与渲染深度差
   ≤ P4/H0 同源的 z-buffer 噪声阈）；"可查询" = ≥2 个视图可见。报告未匹配 GT
   中可查询比例（数量与面积加权；z=22m 平面作敏感性）。

## 四要素

- 主指标：ΔbboxAP_miss(s=0.7)、ΔsegmAP_miss(s=0.7)、可查询率；伴随：
  ΔAP_rescore、ΔsegmAP_mask、AP50/AP75、small 分层全部上述量。
- 选择口径：无模型/无超参选择；s 与可见阈按上文固定。
- 显著性：无 bootstrap（确定性上限）；逐场分布落盘供后续参考。
- 判定门（预注册）：
  - **R0_ROUTE2**：ΔbboxAP_miss ≥ +2.0 AP 且可查询率 ≥ 50% → 立项 DUAL-R1
    （UAV 几何提案增补的冻结探针族）；
  - **R0_ROUTE1**：ΔbboxAP_miss < +1.0 且 ΔsegmAP_mask ≥ +2.0 → 精化路线仍可
    立项（但按双指标期望只能作次级组合）；
  - **R0_RETHINK**：三个上限全部 < +1.0 AP → 任何机制都无法兑现双指标期望，
    带数字回到用户重谈期望；
  - 其余组合 → R0_MIXED，按主导成分给路线建议但不自动立项。

## 遥测三件套

判读边界：三个预言机是**上限**不是可实现值，真实机制回收率必然低于上限；
可查询率测的是几何可见性而非语义可分性。标定：evaluator 参数固定落盘、
匹配规则与 COCO 口径一致；推理与 R1a 同被动校验（预测摘要前后一致）。
机制不替代结局：本探针不产生任何 AP 增益主张。

## 工件

`/data1/wangcheng/checkpoint/surfbev_b0/probes/DUAL-R0-DECOMP-20261014/`
（inference 缓存 + summary.json + per_scene.jsonl + run.log + COCOeval 输出原文）；
结论 MD 与首次定版实现审计落 docs/、docs/reviews/。
