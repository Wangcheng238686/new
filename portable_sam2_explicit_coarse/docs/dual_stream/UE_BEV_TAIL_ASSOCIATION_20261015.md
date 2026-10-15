# 高度尾部与卫星分割误差：六图CPU关联

2026-10-15，EXP-0006/0007；HEAD 3ab5718；固定1218/950/306/1216/948/505，164个GT框出现。E72 dual EMA correct与E66 single EMA共同GT框，复用现有RLE/IoU/GT和六图两层attention，不新GPU/网络前向/fit/训练/干预/AP。

**状态：计算、主代理复算与最终独立结果审核PASS。** 子代理服务恢复后独立复核36地图、164组RLE、656ROI、1968区域及逐图相关均通过。结果为开发子集描述，不能用作因果或训练准入。

## 主要结果

主X1=actual meanabsdz−同source候选均匀meanabsdz；X2=1−height相容mass。按实际mass*quality及卫星target分母投影到SAM stride16，再在GT框内D>0区域按像素交叠面积平均。Y=singleIoU−dualIoU，正为双流退化。每图相关后六图算术等权，无p/CI/性能门。

|主stride16|层|退化Spearman|退化Pearson|dual绝对错误Spearman|
|---|---:|---:|---:|---:|
|X1高度距离超出uniform|0|−.02130|+.03120|+.26277|
|X1高度距离超出uniform|1|+.09369|−.05033|+.13928|
|X2高度不相容质量|0|−.04896|+.01134|+.08962|
|X2高度不相容质量|1|+.05577|+.03430|+.08127|

每图有效框26/26/7/49/24/28，共160；4框无支持单列。两层同一几何支持。全部164框图等权退化−.000198938，不能忽略无支持框而改称总体收益。stride8伴随方向同样弱/混合，完整逐图、GT内部、建筑聚合、混杂与区域误差见JSON。

**没有看到稳定的高度尾部→双流额外退化关系。** X1与dual绝对错误存在部分正描述关系，但绝对难度不等于UAV造成退化；逐图方向、层间与Pearson/Spearman并不一致。不能据此优先硬筛/压制高度不相容候选，也不能据零附近关联排除其非线性或局部影响。六图、固定已训练表征、框内混合暴露和不同训练历史均限制解释。

## 数值与实现验收

36份地图保留source重复票，按生产project_pixel归桶与mass*quality归一。D0 metric为NaN；coverage独立按rawmass/stride²，raw-positive且D0实际0格。stride16主、8伴随、1仅几何诊断。投影质量加权不含learned门、残差或decoder，不称最终贡献。

首轮FP64-vs-production FP32原allclose门FAIL，立即停审并保存首版/日志。两个首图失败由220/256票累加舍入解释，独立顺序FP32输出与生产bitexact。执行前明确v2：全36map全部12字段及quality/coverage用独立np.add.at FP32逐entry累计，对真实生产模块逐数组bitexact，全通过；不经验放宽常数。原FP64门完整运行14map失败，maxabs .000449615米，记录原门FAIL，不说原门通过。FP64主地图定义不改，差描述记录。

关联脚本首次在bool clipped分位数报NumPy TypeError，主代理只改stat数组dtype=float，保留原脚本/失败log。现有最终RLE对GT的历史IoU全部exact、共同框SHA/order一致；主代理独立scipy相关和bbox overlap暴露复算最大相关差3.89e−16。FP32伴随暴露与FP64的最大ROI差：X1 .0000638756米、X2 .000001080；图相关判断不变。该复算本身不是独立子代理审核；后续独立审核已另行完成，见全文。

## 边界与后续

仅六图重复场景/建筑，ROI与像素不独立。图相关算术平均不是总体相关估计。region统计为每ROI区域错误率→每图ROI均值→图等权，非pooled像素错误率；coverage以完整框面积为分母，quality仅D>0分母。建筑聚合仅敏感性描述，缺失支持可能使X/Y出现集合不同。

独立审核已补齐，本轮不支持把高度尾部作为优先干预对象；更值得回到局部形状表达的保留与消费，例如既有4m BEV格/stride16入口能否提供目标边界变化。该候选尚未预注册或实施，不自动启动新训练。

## 可追溯工件

- [协议及显式v2](probes/results_20261015/bev_tail_PREREGISTRATION.md)、[关联汇总](probes/results_20261015/bev_tail_association_summary.json)、[映射manifest](probes/results_20261015/bev_tail_mapping_manifest.json)。
- [独立审核全文PASS](../reviews/UE_BEV_TAIL_ASSOCIATION_20261015.md)。主代理两份复算脚本与JSON同目录归档。
- 原始地图、ROI/区域记录、首失败日志：`/tmp/ue_bev_tail_association/`。地图NPZ及大逐ROI记录不进git；精确分析脚本在`scripts/analysis/ue_bev_tail_*.py`，沿既有固定路径，非通用入口。生产接线/配置/遥测未改。
