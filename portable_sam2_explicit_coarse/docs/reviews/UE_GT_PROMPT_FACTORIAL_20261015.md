# GT点/密集提示2×2交换独立审核全文

## 实施静态审核

VERDICT：PASS，仅建议主代理准入固定三图smoke，不自行启动GPU或准入153正式。

审核者/root/gt_prompt_evaluator未编写GPU运行器，独立编写CPU验收汇总。审核对象/tmp/ue_gt_prompt_factorial_probe.py及gt_prompt_factorial/PREREGISTRATION.md，对照前轮GT oracle运行器与原mask head实际SAM decoder调用。

1. 四组明确固定PP/GP/PG/GG。P/G只在同一图、同一组已提取视觉特征上获得，G沿上轮相同GT64有限±4替换与生产点/dense构建；两live对角逐GT最终IoU必须精确复现前轮。
2. Capture截取实际mask decoder输入sparse/dense，未调用Observer的补零统计副本。源sparse真实(N,2或6,256)，dense真实(N,256,32,32)，有shape/finite断言。G与P最后两个box token逐值一致，其他SAM image/PE/high/args/kwargs哈希一致。
3. 两交叉仅在SAM decoder prehook替换选定真实源sparse/dense。进入时原输入必须与P源逐值相同，替换后与指定源逐值相同，其他参数和kwargs不改。保留实际2/6序列长度，.to(other_tensor)仅转设备/dtype，不补slot。每GT实际SHA来自Exchange现场计算，并与源SHA精确核验。
4. point valid/correct/labels/coords/local_yx按实际sparse来源映射，与前轮live源逐项对账；交叉上游coarse仍为P，不将上游P点误标GT。效应包括无效slot与可能的序列布局差异，不能声称纯点坐标效应。
5. GT框SHA、每chunk实际boxes顺序与生产32切分精确；源GT ShapeReplacement记录target ranges供CPU验收。sourceprompt CPU克隆结束后SHA不变，避免原地污染。参数_version、特征hash、RGB/深度/几何payload hash及hooks清理有明确断言。
6. dense support=0两源均必须精确为no-mask基底，dense-only最终paste mask必须逐像素等baseline。这个阴性对照保留，不修正support。三图smoke还将在两个对角通过同源交换做sham，最终paste必须逐像素精确。
7. 预注册两主contrast各使用seed1770、1000 image-paired sum/count bootstrap的97.5%区间(.0125,.9875)，Bonferroni对应两主同时区间；各门mean>=.005且lower>0。交互与条件差仅描述95%区间，不后选主结论。
8. 固定±4同时改变形状、置信度与分布，GT框条件排除了检测误差。无训练/optimizer/RPN/AP，也不等价UAV可达净增益、数学上限或解除detach训练收益。

静态修正均为验收字段澄清：actual SHA改由现场Exchange输出记录；补记录GT target ranges与显式真实prompt shape/finite断言。未改变四组干预或指标门，没有扩展实验。

smoke与正式审核待追加。

静态定版补记：作者在GPU启动前自审发现main内dense源字母ds覆盖dataset变量ds，已改为dsrc。审核者重新检索全部ds/dsrc引用，确认dataset只保留用于samples/索引/manifest，dense源全部改用dsrc。此问题未生成任何GPU结果。以上PASS针对修复后的当前运行器。

## 固定三图smoke独立验收

VERDICT：PASS，建议主代理准入唯一153图四组正式冻结运行；未自行启动GPU。

独立运行/tmp/ue_gt_prompt_factorial_summary.py对smoke产物验收，exit 0，生成smoke_summary.json。3图287/288/295、45条GT身份/顺序/面积/层级正确，两live对角逐GT IoU与上轮oracle精确一致；source点metadata、dense support及GT64前景面积均与历史逐项一致。checkpoint使用8MB块流式重算SHA，runner/base/oracle/trace/GT manifest/PREREG依赖SHA全部通过。

四组actual sparse/dense SHA逐GT等于指定现场源，actual_point_source/actual_dense_source/upstream_coarse_source映射正确，actual token布局45例均6（无补0副本）。mask阈值.4/logit−0.4054651081081643、prompt形状contract 2或6×256与256×32×32精确核验。源最后2个box token及visual args/kwargs固定的明确checks全部true，所有source prompt/feature/payload/parameter/hook不变量通过。

三个chunk分别13/14/18，actual GT target ranges与slots精确对应。2例support0实际dense在四组均精确no-mask基底，dense-only paste XOR严格0并逐GT IoU等baseline。两个对角同源交换sham均逐像素exact。smoke无多chunk与2token实例，不制造已覆盖声明；正式保留逐chunk切分/ranges及真实布局核验。

两主门均NOT_EVALUATED_SMOKE，不据三图效果调幅度、换cell或后选方向。保持原四组、固定±4和完整153准入边界。

## 正式153图独立结果审核

VERDICT：PASS。四组冻结产物与预注册协议一致；GT点only主门FAIL（有正向信号但均值未达0.005），GT dense-only主门PASS。不能把FAIL改称点无价值，也不能把PASS外推为训练/UAV/AP收益。

独立CPU汇总实际运行exit 0，生成gt_prompt_factorial/summary.json。153图、4496次出现、237栋建筑身份顺序一致；42张多chunk图共197个chunk，每个实际GT框/slot/GT target range精确符合生产32切分。所有checkpoint（流式）、runner/base/oracle/trace/manifest/PREREG SHA复算通过。两live对角GT IoU、点metadata、GT64前景面积与support均与上轮oracle逐项精确重放；两主区间seed1770/1000 paired image sum/count，quantile(.0125,.9875)按Bonferroni执行，没有根据结果换门。

### 四组共同GT框最终mask IoU

|组|IoU|
|---|---:|
|PP baseline|0.841768524767971|
|GP GT point-only|0.8463085911226222|
|PG GT dense-only|0.8573875912505794|
|GG joint GT|0.8608314016667015|

### 两主contrast

|contrast|均值差|同时控制的各97.5%区间|门|
|---|---:|---|---|
|GP−PP|+0.004540066354651249|[+0.0037024010618297463,+0.005494398035985416]|FAIL，均值<0.005|
|PG−PP|+0.015619066482608386|[+0.013482854423995267,+0.01815635503443887]|PASS|

建筑等权差分别+0.005671541683435764、+0.015193073658535195，不能替代预注册出现次数加权主指标或据此改GP落判。GP有2209例改善>0.002、1563例退化<-0.002；PG分别2281、1142例。

### 全部固定分组（仅描述）

|组|n|GP−PP|PG−PP|
|---|---:|---:|---:|
|small|881|+0.009407965822731007|+0.006095172306845282|
|medium|2510|+0.004459163462916624|+0.012258435819363438|
|large|1105|+0.0008427331663034106|+0.030845974838800347|
|P2|3657|+0.005501949571610866|+0.01187067758573913|
|nonP2|839|+0.00034744785116933247|+0.03195739567909333|
|dense support=0|197|+0.00956340967093728|0|
|dense support>0|4299|+0.004309873604405064|+0.01633480411858742|

### 预定次要contrast（95%区间，仅描述）

- GG−PP：+0.019062876898730412，CI[+0.0171028149265492,+0.021546481730948]。均值精确重放上轮；区间种子本轮1770不同于上轮1760，因此区间稍异，不是端点漂移。
- 交互GG−GP−PG+PP：−0.0010962559385292211，CI[−0.0015880661968478032,−0.0005939003245668515]。
- GTdense下点条件差GG−PG：+0.003443810416122027，CI[+0.002672023907248404,+0.004374261981853884]。
- GTpoint下dense条件差GG−GP：+0.014522810544079166，CI[+0.012660032527967848,+0.016658444706978675]。

交互轻微为负，联合收益小于两单独收益之和，不能直接命名为架构协同或通道冗余。GP/PG可能存在跨来源提示语义矛盾，主效应是另一通道保留P时的条件效应，不是通道内在价值排序。summary内交互的improved_gt002/worse_gt002只是contrast符号阈值计数，不解释成mask质量改善/退化。

### 真实输入与阴性对照

四组实际decoder sparse/dense逐GT SHA均对应预注册现场源；last2 box token、所有image/PE/high/其余args和kwargs不变，source tensors结束后SHA不变，features/UAVRGB/深度/几何/参数与hooks清理检查全部通过。实际四组4496条都是6token，没有真实2token实例；实现保留2/6结构，但本子集未覆盖2token情况。

GP/GG的有效点正确数为P1 4496/4496、P2 4496/4496、N1 4442/4442、N2 4396/4396；PP/PG仍为4469/4496、4466/4496、2923/4479、4005/4460，证实交叉metadata按actual source读取。点有效槽减少与54个GT64全前景等情况相关，不能忽略有效性而只比正确率。

最终paste改变数GP/PG/GG分别4397/4231/4386。197个support0目标在P/G dense源都是精确no-mask基底，PG全部paste逐像素等PP，IoU差严格0；GP/GG均有130例paste改变、平均差+0.00956340967093728。仅是现有规则下阴性对照通过，不证明修复支持规则必有收益。

### 允许结论与边界

允许：在当前冻结模型、GT框、GT64±4源与原support/alpha条件下，dense替换具有更大的平均条件改善；点替换也有正向信号，未达到预注册0.005效应门，小目标描述分组响应较强。提示路径仍有空间，下一步应围绕如何产生有益提示谨慎定位。

禁止：不推断UAV能够提供同等GT提示、不推断解除detach或重训必提高、不推断检测/实例AP净增益；不是数学上限，也不是纯形状精度效应（固定±4同时改变形状/置信/分布）。不删除点通道、不按small/P2后选主结论，不把两个条件效应当全局重要性排序。数据仅开发子集，图像bootstrap未覆盖空间相关、checkpoint选择、单seed与完整生产分布不确定性。本轮无训练、无AP、无额外探针。

本轮summary最后补强明确mask_roi_chunk_size==32及交互计数语义；CPU重验三图smoke仍PASS，两主门仍NOT_EVALUATED_SMOKE。未修改GPU runner或协议，所有SHA与最终CPU汇总代码对应。

## 主代理汇总与结论审查

VERDICT：PASS。主代理逐段审查运行器/协议和独立CPU汇总，要求实际SHA来自Exchange现场、source身份与真实token布局、精确chunk32、实际阈值和阴性对照字段核验；全部完成。

主代理以独立脚本直接读取四组原始逐GT记录，未调用汇总函数，重算六contrast均值、seed1770两主97.5%/四次要95%配对图像区间、八分组均值与符号计数、237建筑等权、四组点有效/正确及paste变化数，结果与summary差<1e−12。197例support0的PG−PP逐目标IoU精确为0、paste XOR为0；另GP和GG的真实sparse/dense SHA一致且终点IoU逐值精确相同。审计脚本及JSON为GT_PROMPT_FACTORIAL_ROOT_AUDIT.py与gt_prompt_factorial_root_numeric_audit.json。

结论审核：点only有正信号但未达.005幅度门，不叫无效；denseonly门通过。dense在原点和GT点两种条件下的平均效应均较大，优先定位其质量/构造问题有依据，但不是通道普遍价值排序。小目标对GT点仍有改善；负点低正确率不能单独证明它是总体瓶颈。不宣称UAV可达、AP改善或解除detach训练因果，不按描述分组替换主指标。固定±4同时改变形状/置信度，后续应先区分这两个因素，且不自动追加实验。
