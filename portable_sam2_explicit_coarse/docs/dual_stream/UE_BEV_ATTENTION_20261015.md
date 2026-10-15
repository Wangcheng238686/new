# 有效投影query的CVT实际注意力来源

2026-10-15，EXP-0006/0007；基于HEAD a41d1ef。固定六图1218、950、306、1216、948、505，与前次raw-depth几何审计同集。E72 strict EMA、correct UAV；无GT/拟合/训练/optimizer/候选干预/AP。仅开发子集6图描述，不是完整153图或独立test。

## 实施与准入

三个子代理分别实现GPU运行器、独立实施/产物审查、CPU汇总，主代理审协议/代码并独立重算统计。两层CVT各观察完整16384个query、8head/width32。以只读forward prehook获取实际query/tokens/payload，按生产norm/query/key/value、anchor position+view embedding、clamped relative MLP bias/距离惩罚/invalidmask重算。FP32无autocast，每256query有界处理，不替换任何输入或输出。

每层按projected、heightvalid有候选未投影、heightinvalid有候选、无候选四组，各确定性最多4点保存Q/K/V/bias/权重/SDPA输出供独立数值验收。门为explicit-vs-production SDPA maxabs<=1e−5或relativeL2<=1e−5；两者同时完整记录。无候选dummy输出归零、浮点质量字段NaN，不能作为真实观测。

一图1218smoke两层各16样本，生产SDPA误差maxabs<1.4e−6；独立CPU float64样本重算同样通过。hook-on/off全FPN/SAM/PE/high特征SHA逐值相同，payload/query/token/模型版本/hook清理通过。6960个最终投影query从缓存独立复建一致。正式六图不重复关闭hook，字段null表示未重复。

[预注册](probes/results_20261015/BEV_ATTENTION_PREREGISTRATION.md)、[结果JSON](probes/results_20261015/bev_attention_summary.json)、[运行manifest](probes/results_20261015/bev_attention_run_manifest.json)、[独立审核全文](../reviews/UE_BEV_ATTENTION_20261015.md)。

## 统计定义

- height mass：|anchor_z−reference_z|<=2m的真实attention质量；XYcore：|dx|和|dy|均<=2m；joint两者同时。先每head softmax再平均。高度相容不是屋顶语义正确，非本格也可能是合法context。
- uniform基线：每query有效候选等权，然后按同一query集合汇总；candidate-vote等权另报，不能混淆两种分母。
- entropy：每head H/log(candidate_count)后平均，count1设0；headmean分布的entropy另存，不混同。signedoffset有抵消，meanabsdistance没有，分别报告。
- 主描述组为最终unique(project_source)，另列hascandidate、observed&heightvalid&hascandidate及invalidheight组。不把旧256规则采样当投影query统计。
- 分层/逐图和pooledquery等权分别报告；投影rawmass和mass*quality按source累积加权，后者尚未按卫星target归一化，不能叫实际下游贡献。
- 所有指标纯描述，无显著性或工程增益门，不根据某层/某图结果后选方案。

## 结论

**实际attention没有表现为“整体抛弃局部高度证据”：在有效最终投影query上，高度相容mass相对同query均匀候选略低，但XY本格与高度同时相容的joint mass反而更高。另一方面，加权绝对高度距离更大，提示存在较远高度上下文尾部被加权的线索。两者应并列，不能挑一个统计宣布attention污染或完全正确。**

投影query等权，L0/L1高度mass .795887/.796110，而均匀候选为.804386，下降约.85/.83个百分点；XY本格mass .178875/.178586比均匀.163334高约1.55/1.53个百分点；joint .136983/.139467比.126385高约1.06/1.31个百分点。加权abs dz为5.47797/5.11113m，而均匀4.36852m。signed z均值仅−.81965/−.34135m，存在抵消，不能代替绝对距离。

因此：可疑点更接近“高度尾部上下文及单格内表达如何影响形状”，目前没有证据把跨视角融合弱归结为attention大规模选错局部证据。几何相容不等于屋顶/建筑语义正确；本格之外可能是合法上下文。下一步不能据此直接硬屏蔽不相容候选或启动性能训练。

## 验收与分母

两层各6×16384=98304个query；每层has-candidate 71878，无候选26426；height-valid 38741；最终投影34050。observed support正60975，含无效高度组，不混当有效投影。候选有但height-invalid 33137独立报告，其参考z不可靠，不当真实物理误差。

projected source raw mass总计1111723，mass×quality累计907372.43717。rawmass与massquality均按source累计；仅是投影前源权重，卫星target还会归一，不能称真实最终BEV/SAM贡献或独立样本。每query有效candidate均匀先算比例再汇总；另列候选票等权，分母不同。

全部source/helper/checkpoint（8MB流式）/manifest/cache/NPZ SHA、历史architecture/geometry核定通过；每层FP32、8head×32dim、全query、输入/参数/geometry/cache/hooks不变。正式不重复on/off，smoke 1218全部FPN/SAM/PE/high features逐值SHA一致。12组SDPA sample最大abs误差1.37091e−6、最大relativeL2 9.88445e−7，通过原数值门1e−5；未修改该门。

无候选attention字段保存NaN，CPU完全排除质量均值，不把dummy softmax当观测。entropy为每head H/logN再平均，headmean分布熵另列。完整均值/分位/每图/每图等权/投影加权见summary.json。

## 同集合的投影加权对照

|加权口径|层|高度mass actual/uniform|XY本格 actual/uniform|joint actual/uniform|abs dz actual/uniform(m)|
|---|---|---|---|---|---|
|query等权|0|.795887/.804386|.178875/.163334|.136983/.126385|5.47797/4.36852|
|query等权|1|.796110/.804386|.178586/.163334|.139467/.126385|5.11113/4.36852|
|source rawmass|0|.796301/.805553|.182526/.166909|.139004/.128601|5.49674/4.31765|
|source rawmass|1|.797103/.805553|.182219/.166909|.141510/.128601|5.07376/4.31765|
|source massquality|0|.846139/.850882|.155512/.139877|.137768/.124480|3.54879/2.88096|
|source massquality|1|.844365/.850882|.155903/.139877|.139262/.124480|3.37926/2.88096|

候选票等权高度/XY/joint为.7972968/.1589810/.1213859，精确对应上一轮855882候选票口径；与本轮query均匀.804386/.163334/.126385不同，不能混用分母做学习前后比较。六图先内部query均值、再图等权的height实际L0/L1=.779904/.780987（uniform .790099），joint .137942/.140462（uniform .128297），absdz6.14792/5.71950m，方向与pooledquery一致但幅度不同。

## 逐图异质性（每图projected query等权）

|图|层|n|高度mass actual/uniform|XYmass actual/uniform|joint actual/uniform|absdz actual/uniform(m)|
|---|---|---:|---|---|---|---|
|1218|0|6960|.895958/.894446|.149659/.133205|.133511/.118070|1.44177/1.33085|
|1218|1|6960|.889391/.894446|.152117/.133205|.136238/.118070|1.47097/1.33085|
|950|0|4623|.661901/.682641|.220583/.207190|.141486/.138170|10.56143/8.32032|
|950|1|4623|.668845/.682641|.217485/.207190|.143507/.138170|9.73867/8.32032|
|306|0|6655|.911470/.911996|.151581/.133813|.135835/.119119|2.61127/1.95242|
|306|1|6655|.909222/.911996|.156213/.133813|.139883/.119119|2.29740/1.95242|
|1216|0|6543|.782909/.787549|.164067/.147716|.128706/.116718|3.74315/3.30695|
|1216|1|6543|.781600/.787549|.159095/.147716|.128735/.116718|3.72212/3.30695|
|948|0|4601|.692543/.714516|.221027/.208833|.150521/.147415|10.48248/7.84799|
|948|1|4601|.699186/.714516|.219967/.208833|.154287/.147415|9.47284/7.84799|
|505|0|4668|.734645/.749442|.199254/.183959|.137594/.130290|8.04744/6.48698|
|505|1|4668|.737678/.749442|.197953/.183959|.140119/.130290|7.61498/6.48698|

950/948/505的高度兼容较低且absdz尾部更重，1218/306较高；全部12图层joint与XY质量均高于均匀，全部12图层absdz均高于均匀。不能把重复场景query当独立统计显著性，也未据图ID重新筛结果。

## 熵、视角与高度分组

projected每head归一熵L0/L1 .770827/.741523，headmean分布熵.963297/.963957；头平均分布较分散不等于各head都均匀。最大viewmass .464771/.426248是headmean后视角分布的最大值，不是每head最大值平均，也没有语义好坏判据。

全部has-candidate高度mass .537663/.538084，远低于projected .795887/.796110；height-invalid组只有.230507/.231074且mean absdz28.0622/27.7033m。有效height组则.800387/.800685，absdz5.67204/5.32991m。说明旧hascandidate统计混入无效参考高度，不能将旧训练E72 raw .508与当前EMA投影统计直接相减；这是不同集合/状态/口径，不能叫改善了约30个百分点。

## 数值验收容差记录

CPU首次数学不变量absdz>=abs(signeddz)用固定绝对tol1e−5过严，FP32不同归约造成573行轻微反向差：572 heightinvalid、1 heightvalid但nonprojected，最终projected全部0行；最大绝对差6.10352e−5m、相对差约2.2e−7（依归一化分母略有差异）。改为1e−5+1e−6×abs(signedz)后通过，XY同理。只修改CPU代数验收容差，不改SDPA门/高度2m/XY2m/任何模型或统计。首次失败log从工具原输出保存于summary_first_failure.log，文件说明原summary.log被重跑覆盖的事实。


## 主代理独立重算

直接读取12份原始NPZ，独立重算各组数量、候选票、三种加权的8项关键均值、相对uniform差、候选票等权比例及图等权均值，绝对差<1e−12。脚本和PASS结果归档于probes/results_20261015/BEV_ATTENTION_ROOT_AUDIT.py及bev_attention_root_numeric_audit.json。首次主代理脚本的图均值使用FP32归约，与正式FP64汇总在严苛1e−12门下不一致；改为FP64后通过，未修改正式汇总或原始产物。

后续优先把高度尾部query映射到卫星位置，与已有分割误差关联；若确有定位关系，再另行设计冻结权重的软几何偏置干预，以最终mask质量判定。当前尚未实施该后续。


## 局限与后续

attention几何相容不等于有用性或语义正确；两层output还经过残差、learned query与FFN，某层权重不能完全分解最终BEV内容。无外部标定/roof标签，不能推断屋顶错误率、空间错投或融合污染。

不将当前EMA六图与训练raw E72旧规则采样.508直接相减：参数状态、样本、集合和聚合不同。本轮不调整任何权重，不能因某几何比率低就直接硬筛候选或宣称改动会改善AP。下一步若干预必须另预注册保持固定协议的结局比较。

原始工件`/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis/bev_attention_audit/`，含smoke/full日志、两层逐query NPZ与小样本QKV、manifest及summary。完整大张量不进git，轻量汇总/协议/精确脚本/来源SHA与审核归档。脚本固定运行路径和/tmp helper依赖保留，非通用生产入口。无生产模型或遥测定义修改。
