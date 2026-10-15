# 分阶段形状留存：工程可比性核定（未执行）

## 现有缓存能回答和不能回答什么

已有缓存位置：/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis/uav_complementarity/features，共128train+153validation。原冻结提取281图357.76秒。仅保留卫星512→32两投影、UAV256→32和错位32、8²标签/概率/7几何字段及身份；没有full256、多视角tokens、CVT前后或SAM融合前后。随机压缩与8²不可逆，不能据旧cache重建新16²两阶段比较。可复用：固定imageID选择、manifest/checkpoint哈希、GT boxes/建筑身份核验规则，以及旧结果作历史背景；新主16²标签/概率和stride16特征必须一致重算，不能把旧8²指标混作新baseline。

## 各候选阶段的混杂

|候选阶段|实际表示|支持/分辨率|可比性限制|
|---|---|---|---|
|UAV输入特征|detailed stride8 + resized semantic，经DepthCVT.source Conv1×1/GN/SiLU|5×64²×256，token世界anchors可一token多anchor|五view拼接会给读出5倍容量，GT选view/选anchor造成泄漏；不能直接与单BEV向量比较|
|几何候选等权读出（本轮）|相同source token按生产candidate有效slot等权成256/query|128² BEV、每query候选支持；无候选0|已经是聚合，不能叫“无损聚合前”；无learned query/position/view/relativebias/FFN，与CVT差异是整个表示设计|
|learned CVT（本轮）|两层CVT最后输出，含query残差/位置/attention/FFN|128²×256，无候选0|更强读出可访问性不能单独归因attention；阴性不证明无信息或不可逆损失|
|投影后BEV（两臂共用）|production BEVToSatellite massquality归一|真实SAM stride16 32²，固定project_source/pixel/mass/quality|两臂严格同空间支持；不同source内容仍受同投影量化，不在本轮改变尺度|
|SAM融合前/后（后续）|sat256与sat+gatedproject(UAV)256|32²、融合门依sat与geometry|后者含卫星，不能与纯UAV直接比“信息保留”；应另做same-satellite residual可读性/实际mask冻结对照|

## 本轮收敛方案

只比较几何候选等权读出U与learned CVT C。两者均来自同次E72 strict EMA冻结正确UAV前向，均256channels，使用同一payload、hascandidate mask、project_source/massquality和stride16生产投影。两臂用同一256→32固定Gaussian矩阵seed1813，无PCA/GT学映射，无view选优、无5view concatenation。错位为同32²卫星投影图roll(8,8)，等于128卫星像素，几何/框不变；这个人为错位控制不是物理UAV干预。

主读出空间16² RoI，两个stage均同RoiAlign(spatial_scale1/16, output16, aligned/sampling_ratio沿生产flags)，公共satellite分支使用相同RoI目标网格但来自pre-UAV neck原层选择（512channels，使用相同预注册投影seed1811/1812）。公共GT64局部target及coarse sigmoid64直接area16，所有256gridcells用同标签/概率/局部xy/boxwh/quality/coverage。公共支持从相同projection采样，不按GT或stage质量选样；缺失/低覆盖描述单列，全样本主分母不后筛。注意sat原neck带PE而U/C不带PE，六cells同一sat+geometry控制可控制部分变量，但不能把绝对variance差当语义丢失。

六cells Sbase（卫星+公共几何）、Scap（额外卫星32）、U/Ushift、C/Cshift。每cell同追加32槽、相同train标准化和12个固定小读出头预算（6cell×Linear/Nonlinear，两架构全报）；主标签16² softoccupancy，不是最终SAM mask/AP。evaluator固定统计与工程门，不能据结果加非线性/改seed。

## 原256诊断及资源

仅train共同16²读出单位（每RoI256点）流式累计两臂原256维sum与crossproduct，不保存大原张量。输出每channel mean/std、协方差谱/entropy有效秩，明确读出域/样本重复，不用作PCA特征或调seed；可同时记录支持正域但预先固定两口径。32维阴性不能否定full256表示，数值秩不是形状语义。

建议一次281图重提取而不先做更大多阶段套件；旧281为5.96分钟，新candidate均值每query40slots×256channel分chunk256、矩阵投影/16²ROI约4倍旧缓存数据，预计≤2倍旧时但须smoke实测。root预算full15分钟、峰显存<20GB、输出<5GB，超限停审非按结果停止。缓存/权重仅/tmp，home不写大文件。先3train+3val smoke，严格源不变、同stagegeometry、全GT身份顺序、candidate平均独立手工样本核对、原CVT/投影on-off features exact、params_version和hooks清理，再独立审+root明确full准入。此文与代码均未GPU执行。
