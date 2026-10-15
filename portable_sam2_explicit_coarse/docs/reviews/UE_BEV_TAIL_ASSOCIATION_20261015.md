# 高度尾部与卫星分割误差关联：独立最终审核

VERDICT：PASS。未发现当前结果的阻断项。原FP64数值门失败须保留；PASS针对明确修订的v2及当前CPU描述关联，不构成训练、干预、新探针或性能准入。

审核者frozen_probe_audit，未参与mapper和关联实现。已读工作树`UE_BEV_TAIL_ASSOCIATION_20261015.md`及前序审核，独立以已有工件CPU重算全部36份map、164组dual/single GT框mask、656条尺度层ROI记录、1968条区域记录和两尺度两层全部主关联。无GPU、网络前向、fit、训练、生产修改或新增样本；仅/tmp写审核文件。

## 1. 来源与投影数学

全部mapper/协议/attention manifest/geometry summary/production projection源码SHA、36个map及其attention源和cache SHA独立吻合。production定义为按fine pixel整数行列分桶，weight=mass*quality，metric=sum(weight*source_metric)/max(D,1e−6)，D=sum(weight)；quality=D/max(rawmass,1e−6)，coverage=clip(rawmass/stride²,0,1)。真实SAM/P4 stride16为主、P3 stride8伴随；stride1仅几何描述。

独立从cache与attention NPZ重建36map全部12字段、分母、rawmass、coverage、quality、missing和两个主X，FP64保存数组逐项一致；source重复票保留。独立numpy逐entry FP32累计对CPU真实BEVToSatellite全部36map的全部metric/quality/coverage逐数组bitexact，确认v2实际通过，非只相信manifest布尔项。

D0指标NaN不补0；raw-positive且D0的边界在本次36map实际为0格，已独立重算。该计数不使两种支持定义在一般情形等价：quality全0时raw coverage仍可正。归一化采用D/max(D,1e−6)，不将极小D误当严格归一总和1。

## 2. 原门失败与修复记录

首轮FP64-vs-production FP32固定allclose门确实FAIL并停止。独立复现首图两个正值absdz字段target572/740失败，220/256票FP32累计舍入，非signed抵消或整数归桶错误；逐项FP32参考对生产bitexact。完整v2结果中原FP64门共14map失败，最大绝对差0.0004496154484456838，独立重算吻合，不能报告原门通过。

v2在完整映射前显式修订：全部FP32计算对production要求bitexact，FP64差保留描述，不经验放宽常数；主关联仍用预定FP64数学定义，固定另存productionFP32主X作数值伴随、不后选。首版mapper、首版协议、失败日志与诊断保留。此修订只改变数值验收方式，未改变图像、几何阈值、主X/Y、尺度、模型或结局。

关联脚本首轮bool clipped分位数触发TypeError后停止，旧脚本与失败log保留。独立diff确认唯一修复为stat内部`np.array(..., dtype=float)`；不改有效集合、数据、统计定义或排序。该修复解决NumPy bool插值运算，不是结果筛选。

## 3. GT共同框及像素复核

六图各26/27/9/49/24/29框，共164次出现。独立解码全部164组dual/single RLE，对GT512 mask重新算IoU，均逐项精确等历史保存值；共同框boxes/hash相同。不同训练历史的E72 dual和E66 single不是隔离UAV分支的反事实。

全部656个ROI尺度层记录的bbox像素交叠面积、D>0 covered面积、两个主X bbox平均及GT内部伴随平均独立复现，主Y=singleIoU−dualIoU、dual absolute error=1−dualIoU精确正确。stride16有效框26/26/7/49/24/28，共160，4框无支持单列；stride8有效26/25/7/48/24/27，共157。两层同尺度几何支持相同，缺失X为None而非0。

全部1968条区域记录的固定7×7形态学边界带、GT内部、非边界背景的total/covered/uncovered面积、covered dual/single错误率及差独立从像素重算吻合。区域是每GT框条件内统计，彼此可重叠不同目标框，不能像素伪重复推断；区域汇总先ROI错误率均值、再图等权，不是pooled像素率。

## 4. 关联与解释

独立scipy Spearman/Pearson重算32/64两尺度、两层、两个X、两个Y的逐图全部关联及六图算术等权均值，与summary误差<1e−12。平均ties Spearman、常数/不足样本None及有效n定义正确；没有p值、CI、拟合或性能门。主stride16图等权退化Spearman依次为X1层0 −.02130、层1 +.09369，X2层0 −.04896、层1 +.05577，方向弱且混合。不能由这些相关宣布高度尾部是双流退化原因，也不能由近零相关排除非线性/局部作用。

工作树结论“未看到稳定的高度尾部→双流额外退化关系”与证据范围相符。X1对dual绝对错误的部分正关联涉及目标本身难度，不等于UAV导致退化。coverage按整框面积均值、quality仅D>0面积均值，必须区分分母。大小、截断、支持与边界比例相关可能混杂，不以六图多变量回归消除；建筑聚合只描述重复目标影响，X/Y可因无支持用不同有效出现集合。

FP32伴随暴露差异和主代理完整混杂/建筑汇总由root另有独立复算。本审核重点独立覆盖production一致性、FP64实际map语义、真实RLE/ROI/区域分母及所有逐图主关联，不将root复算冒称审核者独立执行。

## 5. 最终边界

只有六个固定图像cluster，164次出现并非164独立场景/建筑；GT共同框排除检测误差，不等于完整detector AP。几何height相容不是语义/屋顶真值。map暴露未含后续CVT残差/FFN、learned fuser门、alpha和decoder，因此不叫最终信息或错误贡献。没有新GT用于UAV屋顶标签，没有概率mask复原，没有GPU或性能训练。

可归档本轮描述结果及失败修订记录。不能据此优先硬屏蔽高度不相容候选、宣布分支污染/净收益、否定所有尾部作用或启动训练。后续形状探针需另审固定预算、匹配支持、防泄漏和测量终点；本PASS不自动授权新运行。
