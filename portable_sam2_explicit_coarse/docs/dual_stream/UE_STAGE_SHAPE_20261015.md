# 等权几何候选与learned CVT的形状修正读出

2026-10-15，关联EXP-0006/0007；冻结E72 strict EMA correct UAV。原模型不更新、不运行正式训练、不计算detector AP。复用旧hash选128train与153validation名单，12621/4496 GT框出现；新16²占据任务，不与旧8²Brier直接相减。验证集已反复开发且用于E72选择，不能视独立test。

## 这轮具体回答什么

同次冻结前向，DepthCVT.source之后的五视角tokens，分别按原candidate有效anchor slot等权聚合成U，或经过现有两层learned CVT成C。U/C都是256维、同hascandidate、同真实source/pixel/massquality投影到SAM stride16，再相同RoIAlign16²、同一固定256→32矩阵。U已经是聚合，不是全部原tokens无损保留；C含learned query、PE、残差与FFN，差不能归因attention单点或不可逆信息损失。

四个U/C原/错位map均用共同I=(D>0)&roll(D>0,8,8)，公共q/c也乘I。错位为投影32²图roll8格=128卫星像素，两轴同时，是人为内容对照，非真实采集误差；共同I可能减少支持，全部ROI主分母保留，支持零单列。公共sat沿生产pre-UAV neck/PE/multilevelRoI14→bilinear16，与两stage直接stride16 RoI16不同，不称全路径同sampler；卫星额外32维作为容量对照。

SG、S64、C/CX、U/UX六cell，除SG有效变量较少外输入同71维（sat32+second32+公共7维）。Linear1×1仅参照，主Nonlinear3×3(71→32)+GELU+1×1共20513参数；各cell独立权重，同架构初始化与每step样本身份一致。train图均匀→ROI均匀有放回，固定seed1831/1000step/batch64/AdamW lr.001 wd.0001。训练未clip residual MSE，最后step唯一评估clip(p_sat+f,0,1) Brier；归一化train-only，无val早停/调参/续训。

四主compare controlBrier−stageBrier：C对S64/CX，U对S64/UX，均须>=.001且1000次图像配对bootstrap Bonferroni98.75%区间下界>0；每stage两门均过才支持。C−U95%区间仅描述，linear/GT64固定3px边界pool16/二值IoU/支持/尺寸/建筑等权全部伴随，不后选过门。

## 结果：四主门全部FAIL

|主非线性读出|train Brier|val Brier↓|val边界Brier↓|
|---|---:|---:|---:|
|SG|.026564|.046344|.091285|
|S64容量对照|.026459|.046002|.090583|
|C learned CVT|.026597|.045819|.090142|
|CX CVT错位|.026623|.046418|.091841|
|U候选等权|.026579|.046512|.091550|
|UX等权错位|.026612|.046613|.091975|

主四比较（control−stage Brier，正为改善）：

|比较|改善|98.75%配对图像区间|.001幅度且下界>0|
|---|---:|---|---|
|C对S64|+.000182947|[.000064897,.000331673]|FAIL|
|C对CX|+.000598295|[.000478591,.000737042]|FAIL|
|U对S64|−.000509908|[−.000654292,−.000381931]|FAIL|
|U对UX|+.000100496|[−.000025870,.000210506]|FAIL|

非线性C优于U .000692855，预注册描述95%区间[.000573324,.000829413]；不能绕过主门宣称成功。线性C对U仅+.000027324，95%区间跨0。完整六cell两arch、unclipped/clipped与clamp比例、支持/尺寸/建筑分组在JSON。共同支持为零val91目标保留全体主结果；它不同旧184个支持零数，采样分辨率和intersection定义已改变，不能相减解释改善。

**本轮没有证据支持“CVT比简单候选等权更差、等权聚合能恢复强形状信号”。** 当前CVT在固定非线性读出下比等权更可读，但仍只有微弱的容量对照优势；两stage都未达支持门。不能据此证明CVT保留所有原始UAV信息，U本身也已经聚合且共同stride16池化。训练目标是16²occupancy残差，不是最终mask/AP。

所有变量仍受32维随机压缩、同一stride16池化和固定1000步优化限制；train-val差与非线性clamp比例约34–42%均完整保留，不据此追加训练/调参数。train256特征协方差有效秩U约39.60/C约28.01，仅描述无全坍塌，不代表形状语义量。

## 实施、验收与审核边界

首smoke在散列可空PE=None时报AttributeError，模型forward已成功、尚无cache；保留extract_stage_pair_first_attempt.py与extraction_smoke_first_failure.log。修复仅feature_hash忽略None，与原attention已验证逻辑一致，独立静态复审后重跑。6图240ROI的correct hookon/off全features exact、16个确定性query候选均值CPU double手算、原zero-UAV coarse exact均通过；模型参数版本/原输入/geometry/hook清理通过。固定3train/3val smoke耗时21.38秒、2.76GiB。训练only32step timing无valtarget，独立norm/sample表复核PASS。

唯一正式提取281图614.031秒，峰allocated2.16GiB，cache2,879,598,544字节。12621train/4496val，2239/237可见建筑互斥。独立审核者写好的完整cache检查器由主代理执行PASS；这不是审核者亲自正式结果审核。12小头fitloop28.49秒，完整load/fit/finaleval92.75秒，峰小头allocated约32.4MiB。无earlystop/续训/选best，无生产模型/遥测修改。

主代理从预测另脚本独立复算Brier、边界与建筑均值、四主contrast/配对bootstrap，数值PASS；没有重新拟合或调参。**最终独立结果审核PASS**：服务恢复后审核者亲自完成281cache、主门/CI及固定12小头4图48输出重放；GPU同环境重放全部bitexact。CPU异构卷积重放maxabs8.27e−5及Brier差约1.04e−6已如实记录，无重新拟合。

## 决定与后续

按预注册，本轮两stage未过门，停止扩大该候选等权/读出方向，不启动正式模型训练、不追加hidden/channel/seed/steps。独立结果审核已补齐PASS。继续定位应提出独立的尺度池化对照：同一CVT输出在真实stride16与更细投影下比较可读性，保持读出与支持预算；该候选本轮未实施，不能称已定位投影瓶颈。完整五阶段信息保留尚未打通，本轮只关闭其中一个最小可比问题。

## 来源与归档

[预注册](probes/results_20261015/STAGE_SHAPE_PREREGISTRATION.md)、[结果JSON](probes/results_20261015/stage_shape_summary.json)、[提取manifest](probes/results_20261015/stage_shape_extraction_manifest.json)、[拟合manifest](probes/results_20261015/stage_shape_fit_manifest.json)、[审核全文](../reviews/UE_STAGE_SHAPE_20261015.md)。精确脚本与cache检查/root复算脚本同任务归档。协议原文含“草案/未执行”标题以保持执行前SHA，不改写为事后版本，以manifest为实际执行证据。

原始目录`/tmp/ue_stage_shape_probe/`含features、fit小头权重/预测/归一化/采样表、全部运行log、失败首版；大缓存/权重不进git。脚本沿固定/tmp helper路径，归档命名变更不改内容，不是通用生产入口。关联EXP-0006/0007，sourcegit884b381。
