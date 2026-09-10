# A0 邻域负点受限 Oracle

## 预注册协议（2026-09-09）

目的：停止 coarse 修正路线后，验证邻域排他性提示是否值得学习。不是训练模块，
不是 decoder 输出择优的最大上限，也不证明视觉特征可预测这些点。

- 底座：NWPU dev100 A0 epoch36 raw weights；完整 validation 130 图。
- 三次前向：unhooked standard、identity raw、neighbor；全程 eval、不训练。
- 只替换现有有效 N2（slot3，label=0）。其余三个点、全部 labels、box、dense 输入
  SHA256 必须不变；无有效 N2、无同类 IoU≥0.5 target 或无候选时原样回退。
- 候选为 proposal 1.5x 扩张框上的固定 32×32 单元中心，超图像者丢弃。GT 只用于
  匹配 target 和筛选其他实例内部、target 外部的候选（3×3 独占内部，排除重叠歧义）。
  与已有有效点距离至少一图像像素；取距 proposal 中心最近候选，同行列顺序打破平局。
- GT、proposal、PE 点均处于预处理后的全图坐标；不调用 ROI mask-target 变换。
- standard 须复现 canonical A0 records/mAP；raw mask hash 必须与 standard 恒等；
  全臂 detector hash 恒等。按图导出 COCO records、真实处理 ID 与诊断分子/分母。
- 机制指标在固定 eligible proposals 上统计：其他实例误覆盖像素/其他实例像素，
  以及当前 target 覆盖像素/target 像素。实例可能被多个 proposal 匹配，统计是
  proposal-pooled 而非去重 GT 统计；图像级 bootstrap 保留这种相关性。

## 决策

完整 val 的 paired image bootstrap（500次，seed44）若 ΔmAP 95% CI 下界>0，
且邻居误覆盖减少、target 覆盖不出现明确损害，才支持进入学习型负点模块设计。
否则停止本次固定候选与单负点配方；不外推到所有提示模块无效。
eligible 比例过低时，应首先归因为该机制覆盖面不足，不能称消费端失效。
此验证集已用于 A0 权重选择和多轮开发，CI 仅是固定 checkpoint 条件下的不确定性，
不是独立测试集泛化或跨种子稳定性证明。

## 状态

完整三次前向已结束，canonical records/mAP 精确复现、raw hook 恒等、detector
和其余提示 SHA256 不变量全部通过。CPU selector 回归通过。

工件根目录：`/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/a0_neighbor_oracle_full_20260909/`。
`summary.json` 保存逐臂 hash；各臂 `diagnostics.json` 保存逐图分子/分母。

| 指标 | raw | neighbor |
|---|---:|---:|
| 全 130 图 segm/mAP | 0.6669291021 | 0.6659555972 |
| eligible proposal 的其他实例误覆盖像素 | 74014 | 79295 |
| eligible proposal 的 target 覆盖像素 | 2520659 | 2523797 |

918 proposals 中 728 匹配，298 eligible（分布于 61/130 图），没有 invalid N2。
其他实例像素分母 21208330、target 分母 2622247，两臂逐图完全相等。
其他实例误覆盖率 0.348986%→0.373886%；target 覆盖率 96.125918%→96.245586%。
500 次配对图像 bootstrap（NumPy default_rng seed44、ratio-of-pooled-counts）给出的
前者差值 CI 为 [-0.000003531,+0.001106237]，后者为
[-0.00041853,+0.00437770]（均为比例而非百分数）。二者均跨零。

目前点估计不支持训练这个配方：mAP Δ=-0.000973505，邻居误覆盖也未减少。
不能称其显著变差，不能将固定几何 Oracle 称为所有负点策略的最大上限。
500 次 COCO mAP paired-bootstrap 已完成（seed44、130图、10类、执行 ID 强校验）；
`bootstrap500.json` 的 baseline/treatment 点估计精确复现推理结果。
ΔmAP 95% CI **[-0.002948891,+0.000349853]**，bootstrap 平均差
-0.000976798，正差样本比例 0.142（非显著性检验 p 值）。
**预注册晋级门 fail：停止此固定几何单负点配方，不启动对应训练。**
该结果未证明显著变差；区间仍允许很小的正效应，但没有支持晋级的正向证据。

## 下一步方向与限制

在用户要求不再走 coarse 细化、且 pre-PE 新模块必须有贡献证据的条件下，当前
**不启动邻域负点学习器，也不把它注册为新的正式消融行**。先保留已验证的
PBM 与 decoder-tail 主矩阵；这不是用 tail 替代用户想要的 pre-PE 模块，而是
承认当前还没有足够证据推荐一个新的 pre-PE 训练模块。

若后续继续探索 pre-PE，下一项应是单独立项的“任务损失驱动的提示位置效用”诊断：
先检验当前 decoder 是否存在有用的点位置，再研究可学习选择器；不能继续仅凭
GT 成员身份或视觉源可读性推断最终分割收益。本轮不启动该新诊断或额外训练。

本探针只替换 N2、仅用一套固定扩框网格和几何排序，并保留 A0 在原点分布上训练的
decoder；因此阴性结果同时可能来自替换位置策略、点间协同或分布改变，无法区分
这些原因。邻居误覆盖分母包含所有其他实例，不仅是被负点选中的实例，属于
全图排他性指标，不能包装为选中邻居的局部误覆盖率。
