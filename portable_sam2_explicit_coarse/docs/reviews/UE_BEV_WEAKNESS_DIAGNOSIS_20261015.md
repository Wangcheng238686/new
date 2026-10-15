# BEV读出能力与错误分层：主代理独立复核

2026-10-15，审核者root；未参与两份子代理脚本实现。

VERDICT：PASS，允许作为当前开发子集的描述性定位证据；不改变上一轮总体工程门FAIL，不授权训练或追加GPU。

## 实施审核

核对ue_readout_capacity_audit.py：全部既有缓存SHA逐项验证；split分别计算协方差/局部空间变异，已有标准化/系数固定，未重新拟合监督模型；UAV对共享变量的Schur补只用协方差，是无监督线性冗余描述，不当作任务有用性。岭谱收缩只用训练SU标准化协方差与实际lambda=.01，不能把验证谱当新拟合。原预测复现与旧val_predictions核对。

核对ue_bev_error_strata.py：所有coverage和粗错误分位cut只由train确定，val按固定cut分组；oracle阈值0/.002/.02预定义，仅后验描述，不宣称confirmatory证据。联合身份按image_id/gt_slot/global_id核验；区域同时保留nROI/nCell、cell等权/ROI等权及建筑等权，不把混合占据格当真实细边界。多组无新门/多重显著性选择。空组明确返回null。源完整SHA已记录。

## 原缓存独立数值重算

另写BEV_DIAGNOSIS_ROOT_AUDIT.py，从281原缓存分批累积训练/验证sat/UAV协方差、ROI内方差，独立复算32维相关participation ratio和SU训练岭谱保留/自由度，与子代理结果容差1e−8内一致；并从val_predictions及原缓存目标重算SG/SX/S64的all/coverage0/原错误高/GTdense空间大与非正组的n/均值，及三类占据区域的nCell/nROI、cell等权和ROI等权，均与summary差<1e−12。train分位cut独立重算一致。输出bev_diagnosis_root_audit.json。没有再次拟合模型、GPU或加载原始图像。

这些只排除明显数值退化、确认分布异质性；不能证明某个几何设计导致因果损失，也不能把协方差秩/空间方差当有效语义。高oracle空间子组出现等权收益强、建筑等权下降，必须同时报告，不能重新判定主门成功。小目标有coverage后仍弱于错位/额外卫星控制，零覆盖不是唯一原因，但不等于已定位原因。

## 几何复核分工

独立子代理readout_capacity_audit未参与原始几何脚本开发，另行核验六图保存NPZ、原cache/source SHA和全部分位/候选比率，全文见UE_BEV_GEOMETRY_QUALITY_20261015.md。主代理审阅脚本/协议/原始表与该复核，确认raw/selected-bin、sat-frame/all两种分母不得混用；4m中心位移不能当实际quad投影误差。没有独立重载30EXR，不冒充重复原始生成验收。
