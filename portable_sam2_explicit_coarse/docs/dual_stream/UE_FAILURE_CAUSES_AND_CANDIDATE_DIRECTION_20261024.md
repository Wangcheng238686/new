# UAV失败归因与候选质量方向（2026-10-24）

当前摘要：已完成两个独立训练seed、全test与三项固定输入控制，在当前强纯SAT之上Mask净+.405～.464pp，匹配内容差+.183～.241pp，区间均为正。输入归因、同数量控制、来源及冻结审计全部通过。[最终结论](UE_UAV_EFFECTIVE_DIRECTION_VERDICT_20261024.md)。下文按时间保留探索过程，当时的未完成/等待状态不代表最终状态。

## 目标与当前判定

目标是找出失败原因，并用真实UAV内容带来超过纯卫星的实例分割净收益。该阶段目标现已通过总验收；不能以候选数增加的卫星收益代替UAV收益，最终方案因此同时检查强纯SAT、常量、实际候选数及场景错配控制。

既有的单位、矩阵回环、来源复算、初始化恒等、参数冻结及梯度门，没有发现能解释全部失败的公共cm/m或反传断路。它们不认证真实跨视角同表面对应。详见来源与读取审查。

## 一、证据支持的限制

1. **任务入口与主要错误不一致。** 旧mask-only分支的检测框、分数保持不变，无法恢复漏框。检测审计发现小目标在Top-K及RoI NMS损失明显；这些框不存在时，后续mask改得再好也不能恢复该实例。
2. **强基座与弱内容差。** 全train2142图卫星Mask AP90.65093；旧N2正确输入90.64475，shared90.57448，正确不优于同权重常量/donor。不支持“训练内已获得UAV收益，仅在test分布偏移后丢失”这一单因解释。train饱和仍可能限制新分支动力。
3. **跨视角表示与接收特征语义受限。** 旧conv_s0接收重复的全图特征，而UAV残差采用RoI局部坐标；RGB格内最多8候选、步长4、没有完整卫星遮挡认证。C/D虽增强融合消费，但常量RGB训练几乎复制其全部AP上涨。不能将新增容量/几何适配解释为真实RGB贡献。
4. **现有辅助目标不对准实例收益。** S²DB D加入建筑union辅助BCE后Mask AP由64.79991降至64.77605；辅助语义可解码不等于能恢复实例、排序正确或修好精细边界。
5. **几何RPN输入与监督过弱。** 第一轮只在原anchor中心取四个字段，以原anchor分类目标训练。没有读取回归候选区域、完整表面形状、RGB或多层结构；标签基于原anchor，不是回归后候选质量。下面的新诊断量化后两点。

上述不是互斥因果分解，尚无单项消融证明各项解释多少AP。

## 二、最强可操作证据：候选预算

固定SAT E73，纯卫星RPN每层pre1000→2000，post1000、原NMS与final250不变，无训练：

- test Mask AP64.55098→65.69030，+1.13932pp，名义配对95% CI[+0.87424,+1.26824]pp；bbox +1.13811pp。
- val Mask AP85.45525→85.53770，+0.08245pp。
- 真几何RPN相对SAT仅+0.00241pp，相对常量训练+0.00088pp，主Mask区间均跨0。

这证明当前卫星筛选损失可以传到最终AP；不证明UAV能优于扩大预算后的卫星。

## 三、新增459图覆盖审查

脚本`scripts/analysis/ue_candidate_evidence_audit_20261024.py`，产物`/data/wangcheng/checkpoint/ue_candidate_evidence_20261024`。读取已完成baseline/budget预测及历史native缓存，不进行新test推理、不改缓存。GT与图像逐项一致，输入/缓存/源码SHA记录于summary。

44988个GT，以任意候选框IoU≥.5为覆盖，不是一对一检测匹配或AP分解：

|组|GT数|有任意实例mask内支持|至少50%实例mask有支持|bbox中心有支持|
|---|---:|---:|---:|---:|
|原Top-K丢失|1920|1878|1633|1468|
|2000预算在Top-K救回|1065|1048|922|847|
|最终覆盖新恢复|952|944|848|784|
|预算2000后仍未覆盖|2606|2565|2349|2154|

最终新恢复952，其中947为mask小目标；最终新增丢失26，净增加926个任意覆盖GT。原Top-K丢失1920不是旧报告“最终漏检的最后损失Top-K1838”：筛选条件不同，不能混用。

实例支持是已投影深度存在，不认证屋顶、同物体或卫星可见性。bbox中心是单点读取的代理，不是某个真实候选anchor中心。尽管如此，这批可恢复目标通常并不缺观测，直接否定“主要因为根本没有UAV覆盖”的简单解释。

## 四、新增anchor监督审查

脚本`scripts/probes/ue_rpn_supervision_audit_20261024.py`，产物`/data/wangcheng/checkpoint/ue_rpn_supervision_audit_20261024`。固定32train/16test密度中分位样本，FP32无增强，无优化，完整卫星SHA未变。不是全split统计，test仅诊断；每图仅一次确定随机anchor抽样。

- train1885GT，4007个抽样正anchor，3765个logit>10（93.96%）；4185负anchor，4165个logit<−10（99.52%）。平均每图抽样原始BCE约5.67e−5，不含原loss权重与训练AMP/增强效应。
- train26个Top-K丢失GT，22有有效支持的好回归候选，18有原assigner正标签的好回归候选，12在本次抽样中得到好候选正监督。
- 这些GT对应964个GT–候选好匹配，原assigner正61、负452、忽略/边界无效451；仅19个抽样正匹配、1个抽样负匹配。重复候选可跨GT计数，不是独立anchor数。
- test1572GT，Top-K丢失69；67有支持好候选，64有assigned-positive好候选，24在本次抽样得到好候选正监督。

支持判断：现有目标主要继续巩固已非常确信的anchor分类，而不是直接学习回归候选质量或挽救漏检；绝大部分有用回归候选没有作为正样本训练。不能说这些候选都被负监督压制：实际抽样负项很少，缺少针对性监督比强烈反向监督更符合结果。一次抽样没有正项不代表10轮从未抽到。

## 五、下一方向及正在执行的桥接检查

方向：**基于UAV区域表面的候选质量判断／排序**，直接监督回归后候选价值。保持原卫星与SAM/UDPR冻结，先验证几何内容是否具有额外区分能力，不继续同一anchor分类残差长训。

桥接脚本`scripts/probes/ue_candidate_quality_bridge_20261024.py`，独立根`/data/wangcheng/checkpoint/ue_candidate_quality_bridge_20261024`：

- 固定64train/32val密度中分位图，不读取test。候选池固定原每层sigmoid top2000，不改回归框。
- 公共描述：冻结RPN hidden固定随机投影32维并归一化、原logit、框宽高、层级；几何读取回归框内3×3网格（nearest），保留9点支持，不填孔。
- 三个同配置表格分类器：常量区域（train-only均值+真支持）、真几何中心重复9次、真几何区域。后两臂同时改变支持采样范围，区域−中心含字段与支持上下文；真实区域−常量区域才是同支持下字段内容比较。
- 监督：回归框maxIoU≥.5为正、<.3为负，中间忽略。每train图最多2048正+2048负，确定随机抽样；HistGradientBoosting固定100轮、15叶、min_leaf50、L2=1、LR=.1、无early stopping，seed44。
- val先比较固定每层1000预算下任意GT覆盖、新恢复/新增丢失和候选分类性能。覆盖不是最终AP；这是信息可解码性的快速检查，不是正式双流神经模型、训练重复性或完整矩阵。

后续门：只有真实字段相对同支持常量控制在固定val候选保留上呈合理净改善，才进入正式可训练轻量候选头和完整AP评价。若只比原SAT1000好、却不比同预算SAT2000好，结论限于压缩预算的效率价值；若只有常量臂也同样好，不归因UAV字段。不得按test继续调读取范围、分类器超参或权重。

未来正式收益需同时检查完整检测/分割AP、匹配容量对照、正确与错配几何、多训练seed及独立区域/新版本评价。当前不保证该方向必然成功；其优先级来自实际可恢复错误、UAV覆盖和监督缺口三条证据，而不是理论上能重建3D的直觉。

## 六、桥接检查已完成：直接替换排序不通过

96图特征缓存经独立检查，81维描述有限、公共卫星描述相同、孔洞字段零、IoU列与GT数一致。64train抽样250792候选，其中130555为正，三臂已全部拟合完成。32val共1046GT：原Top1000覆盖1035，整个Top2000池覆盖1041。

|描述|候选分类ROC AUC|候选PR AP（不是Mask AP）|重排Top1000覆盖|新恢复／新增丢失|
|---|---:|---:|---:|---:|
|常量区域+真支持|.97847|.97775|1025|0／10|
|真几何中心|.98079|.98114|1027|0／8|
|真几何区域|.98132|.98189|1026|0／9|

真实几何略提高候选二分类，但区域描述没有胜过中心的覆盖；三种全部覆盖下降，不能以AUC上涨宣称“找到了方向”。不会据此启动替换全局RPN排序的长训。val里可额外恢复的GT只有6，进一步限制该小样本对微弱收益的判断。

额外排查sigmoid舍入：这96图各层池的第1000名阈值没有任何一处等于1；val阈值相等候选最多1，train最多40。即使大量抽样正logit>10，也不意味着Top-K阈值饱和；当前不支持“sigmoid都舍入到1导致Top-K失败”的解释。

## 七、收紧为直接实例候选补充的可学习性试验

新增`scripts/probes/ue_geometry_proposal_pilot_20261024.py`，独立根`/data/wangcheng/checkpoint/ue_geometry_proposal_pilot_20261024`。不再使用现有分类分数重排已保留的卫星候选；先训练一个独立几何实例候选生成器，验证字段能否直接预测实例，再决定是否接入冻结SAT。

- 输入native512四字段+support；两个臂为真字段与train-only常量字段，support相同，完整初始化相同。不加载任何卫星网络，不用UAV RGB。
- 小型多尺度卷积：512→256(16ch)→128(32)→64(64)→32(64)，后两层上采样到128与32ch拼接，32ch解码后输出中心heatmap、中心offset、log宽高。GroupNorm，无EMA。
- 实例监督直接取卫星GT框，128网格Gaussian中心focal，offset smoothL1和0.1×log-wh smoothL1；同cell碰撞按最小面积框回归，中心损失分母为唯一正中心cell数。不是union语义辅助loss，也不通过冻结RPN离散筛选反传。
- 两臂固定全部2142train、batch8、drop_last6、10epoch/2670updates、AdamW1e-3/wd.01、warmup50/cosine至.1、clip.5、AMP512、seed44与epoch1044000+e。新网络从随机初始化学习；此LR与旧冻结适配器1e-4不同，不称严格同训练合同。没有光学/空间增强；几何输入与GT原坐标一致。
- 固定E10 last；只评全部459val，FP32、局部3×3中心峰、同支持过滤、top250候选。输出any-box IoU .5/.75覆盖及完整候选；不声称唯一实例匹配、bbox AP或Mask AP。验证前不选阈值、轮次或较好seed。
- init/gate/train三个命令；保存common_init、输入档案/源码SHA、完整optimizer/scaler/Python/NumPy/Torch/CUDA/DataLoader generator。gate检查AMP有限梯度、支持空洞不发候选、常量保留支持、原初始化不变，实测batch2峰值reserved156MiB；正式batch8两个臂共租GPU0，单进程allocator上限约5.5GiB，不抢现有任务。

工程门通过后两臂已实际启动，PID记录于launch.json。它是针对“能否直接从几何学出实例候选”的小型训练，成功后仍需：与强SAT2000对照的新增候选有效性、去重/误检、冻结RoI/SAM消费、完整AP及内容干预；没有跨过这些门不能称完成用户目标。若真几何不优于常量，不立即堆RGB/更大骨干，应回到字段是否保留建筑可辨识结构的检查。

`first_epoch_audit.json`已核定两臂E1的2136图顺序完全一致；实际检查点为相同初始指纹、E3/801、AMP scale512及完整五类RNG（含DataLoader generator）。训练仍运行，不能将其写成最终验收。

独立CPU`scripts/analysis/ue_geometry_proposal_complement_20261024.py`按launch.json已验证的两个真实训练PID等待结果（不是等待空GPU），随后只读完整val459输出，计算真几何−常量的IoU .5/.75任意覆盖、最大一对一匹配容量，以及相对已有SAT2000的RPN cap／最终检测未覆盖GT可补充数量。同图bootstrap500、seed20261021，差值分母为全部val GT。该“可补充”尚未实际合并推理，不等于AP上涨；v1 val较稀疏，先作为可学习性与互补性证据。

## 八、直接实例监督已产生强内容信号，尚待消费成AP

两臂均完成E10/2670及完整459val，14980GT。`sample_order_audit.json`核对全部10轮顺序完全一致，AMP scale由512按默认growth interval在第2000个有限更新后升至1024，两臂相同；没有按val选轮。

|独立候选指标|真几何|常量+真支持|真−常量|
|---|---:|---:|---:|
|IoU≥.5、top250任意覆盖|12867（85.89%）|9257（61.80%）|3610（+24.10pp）|
|IoU≥.5最大一对一容量|12867|9257|3610|
|IoU≥.75、top250任意覆盖|6607|1939|4668|
|补充SAT2000的RPN cap未覆盖GT，IoU≥.5|15|7|8|
|补充SAT2000最终未覆盖GT，IoU≥.5|142|97|45|

独立候选一对一容量差95%图像bootstrap[+22.67,+25.50]pp；最终未覆盖GT的可补差45个，按全部GT分母为+0.3004pp，名义区间[+0.1759,+0.4352]pp。IoU≥.75最终可补为112对33，差79。不是AP，也不是多seed泛化确认。

这支持“现有几何字段包含可学习的实例位置/形状信息”，反对从旧适配器失败推出数据完全无有效信息。两臂都含真support，所以字段收益不是仅由有无支持造成。常量臂本身61.80%也说明support携带结构，不能称纯卫星。

重要瓶颈迁移：几何框能补最终漏失142GT，却只能补RPN池15GT；大多数待补GT在原RPN池已有候选，之后才失去覆盖。不能默认把新几何框再送进原RoI回归/分类后仍能保留其优势。

## 九、已启动冻结下游消费2×2（仅val，不新训）

新增`scripts/probes/ue_geometry_proposal_consume_20261024.py`，根`/data/wangcheng/checkpoint/ue_geometry_proposal_consume_20261024`，四臂固定为内容{真几何、常量}×框{原几何框、原RoI回归后的框}。E10候选已经锁定，不按其val结果调框范围或模型。

- 基座SAT pre2000/post1000/final250，逐图重放原RPN/RoI，核对全部原bbox/score/label与已有SAT2000记录逐值相同，复用已有原mask RLE，保证原预测完整保留。
- 两种框均以同一输入几何框调用冻结RoI分类器；regressed臂额外使用原bbox head的框回归结果，raw臂保留几何原框。不是修改mask网络。
- 用原前景阈值和原NMS阈值0.5：拒绝与原SAT最终框IoU>0.5的新增框，对新增框按原cfg自身NMS，再填充250上限剩余名额。原检测优先，不重排或挤出它们。无GT参与准入。
- 只对新增框调用冻结SAM/UDPR；不重新学习SAM、RoI或候选分支。全459val FP32计算Mask/bbox AP250，保存完整记录供后续Boundary和配对分析。
- 每臂先独立2图gate，再完整val；初始真几何raw门通过，原检测/原mask恒等、模型SHA不变，峰值reserved1242MiB。四个gate→val进程已共租GPU0/2启动，PID及source/候选/基座SHA均存独立根。原训练/旧source不动，不访问test标签或推理。

只有四臂完整结果及真实−常量、真实−SAT2000净差，才能判断上述内容信号是否真正被消费成实例AP。若只有候选回归前可补数正，不得宣布完成用户目标；若regressed与raw分歧大，才进一步离线定位框回归破坏，而不从候选互补数直接定因。

## 十、冻结分类器否决证据与置信分数补充试验

上述四臂均已完成全val459。SAT2000 Mask85.537698；真几何raw85.530225、常量raw85.525969；真几何regressed85.537686、常量regressed85.537680。四个均没有Mask AP净改善，不能把第八节候选内容信号直接称为分割收益。

新增只读`scripts/analysis/ue_geometry_box_consumer_audit_20261024.py`，产物消费根`box_consumer_audit.json`，以CPU float64 IoU追踪raw框：

- 真几何可补142GT，排除与原SAT框IoU>0.5后仍119，经过原分类器等后续筛选的实际新增框只覆盖其中8个。常量对应97→84→4。
- 真几何原114705框，novelty后98068，最终接纳567；常量114702→101104→957。novelty不是损失绝大多数可补GT的原因；剩余111GT损失包含原分类阈值、新增框自身NMS和最终名额，需细分时再离线回放，不能唯一归因某个算子。
- 回归臂各只接纳5个框。raw与regressed完整AP都无净收益，说明直接把几何候选交给原RoI链并未打通消费。
- GT oracle允许相交对应框中挑更好几何框，14587个已匹配SAT GT只有195能改善IoU>0.05，90个可跨.75；实际按与SAT框最高IoU找几何对应，有180改善而11229恶化>0.05，平均IoU降.1788。不能粗暴用几何框替换卫星框，也不能把oracle当可执行收益。

接下来固定三臂，不新训：`scripts/probes/ue_geometry_score_band_20261024.py`，根`/data/wangcheng/checkpoint/ue_geometry_score_band_20261024`。内容为真几何、常量字段、纯SAT自己被抑制的RoI假设。

- 真/常量几何框使用各自E10中心heatmap置信度，不接受冻结SAT分类器否决，也不经其框回归。原SAT全部框和mask保留。
- 新框仍按原IoU阈值novelty、自身NMS及250剩余槽位；用`extra_score=原score_thr × 分支objectness`，objectness∈[0,1]。所有原检测分数>原阈值0.05，故原检测在全局排序中完整先于新增假设。
- 这使原precision-recall前缀及匹配保留，AP不下降是设计上的保证，本身不是UAV有用的证据。必须看真几何是否超过常量及纯SAT补充对照。
- 纯SAT对照读取原post-RPN全1000框的RoI回归框及前景概率，包含原分类阈值/NMS未保留的框，使用相同novelty/NMS/剩余槽位及低分数带。它具有比几何250更大的初始候选池，作为更强预算控制，避免把放宽SAT分类阈值就能得到的收益归于UAV。
- 三臂固定全459val FP32，逐图原检测/原mask恒等与冻结SHA；每臂先2图gate再完整val，原始真几何门已通过。独立source/协议/worker PID已保存，正式三臂共租GPU0/2启动。无test推理或新标签使用，不按test选分数缩放。

当前仍未达到用户目标；下一判据是完整AP的真几何−常量、真几何−SAT-only增量与配对统计，然后才决定是否做锁定结构的独立test和新训练seed确认。

## 十一、三臂完整val与强基线条件增益

score-band三臂均完成全459val，以下单位AP×100：

|策略|Mask|bbox|
|---|---:|---:|
|SAT2000原前景阈值|85.537698|83.394341|
|SAT2000+常量字段候选|85.682871|83.484256|
|SAT2000+真实几何候选|85.816187|83.627862|
|SAT2000+纯SAT被抑制的RoI假设|86.299126|83.982185|

Mask配对500次，seed20261021，已完成的单指标统计：

- 真几何−SAT2000：+0.278489pp，95%[+0.142792,+0.435340]pp。
- 真几何−常量：+0.133317pp，95%[+0.040657,+0.204281]pp。
- 纯SAT补充−SAT2000：+0.761428pp，95%[+0.574939,+0.895745]pp。

首次正内容差不等于胜过更强SAT策略；真几何单独分支仍低于纯SAT补充约0.483pp。不能忽略强控制来宣布成功。

新增`scripts/probes/ue_geometry_score_band_analysis_owner_20261024.py`在完整identity落盘后启动固定五个比较，最多2个CPU统计同时运行，使用既有向量化累积器逐evaluator前三个抽样与原函数校准≤1e−12，全部四指标仍按原500样本。其余Boundary统计继续，不根据中间结果改变已固定的分支。

新增`scripts/analysis/ue_geometry_conditional_complement_20261024.py`，独立根`ue_geometry_conditional_complement_20261024`，只读已有预测：保留更强纯SAT补充结果的完整前缀，将其分数严格单调映射到>.5（原rank逐值核对），几何新增假设置于≤.25；对全部强前缀按原IoU.5去重并填250剩余槽位。不新训或调用GPU，使用已缓存几何mask子集；此前被几何自身250上限截断的尾部候选不会补回，不将其误称完整新推理。

|强前缀之上的val组合|Mask|bbox|
|---|---:|---:|
|仅强SAT前缀|86.299126|83.982185|
|强SAT+常量几何|86.299126|84.021126|
|强SAT+真几何|86.323688|84.031361|

Mask真−强前缀+0.024562pp，95%[0,+0.065977]pp；真−常量+0.024562pp，95%[−0.022504,+0.029697]pp。未达到原明确增益参考门槛，不认定稳定条件收益。两个完整四指标统计分别运行，源记录、区间与校准见该根analysis/。

## 十二、锁定在线与测试核验

新增`scripts/probes/ue_geometry_conditional_online_20261024.py`，根`ue_geometry_conditional_online_20261024`：同一E10/2670几何头在原生字段上在线前向，原SAT重放和新增mask均FP32。每个内容臂的前2张val门要求框、类别、RLE逐值相同、分数误差≤1e−12，复现第十一节缓存组合，两臂均已通过。然后固定全部459test，不根据test改准入、分数带或权重。原SAT prefix和combined完整记录/指标分别保存，G/K的纯SAT prefix必须完全相同。

新增`scripts/probes/ue_geometry_branch_test_20261024.py`，根`ue_geometry_branch_test_20261024`：确认第十一节相对原SAT2000已val正向的直接几何分支，仍固定E10/2670与所有准入规则；两臂先2val gate，要求原score-band缓存的框/score/RLE精确重现，再全459test。该输出与更强条件分支一起公开，不根据test选择较好路线。前者在GPU1/3共租，条件分支两臂在GPU0共租，不抢占原attention训练。

新增`scripts/probes/ue_geometry_confirmation_analysis_owner_20261024.py`，独立根`ue_geometry_confirmation_analysis_20261024`，只在各evaluation identity完整落盘后计算五个预定test比较：直接真−SAT2000、直接真−常量、强SAT−SAT2000、条件真−强SAT、条件真−条件常量。使用相同500次同图bootstrap与原累计等价校准。五项CPU比较可并发，每项限制OpenBLAS/OMP为1；不是等待GPU配额。对两个条件臂的prefix JSON做逐字节恒等审计。

两类test运行根均已保存新依赖source快照及SHA，并复核原RPN研究冻结49份source全部未变（source_audit.json）。原卫星权重/缓冲区SHA由工程门及最终推理审计；没有新模型训练、EMA、UDPR或coarse64修改。当前测试还在推理/CPU指标阶段，未形成测试收益结论。

注意：v1 test已在此前大量探索中暴露，这里是锁定方案的开发测试确认，不是独立未见基准；单训练seed和空间相关限制保留。若效果只相对较弱SAT策略成立，必须明确其边界，而不能回避已知更强纯SAT比较。

## 十三、两个独立训练seed的完整主比较均为正

seed44四项test及五个配对报告已全部完成。强SAT原分数策略修复后Mask66.087884，比SAT2000的65.690299仍高；主方案为该强前缀上的独立几何实例补充，不以较弱原SAT作为唯一参考。

|训练seed|强SAT|强SAT+常量|强SAT+真几何|真−强SAT pp及95%区间|真−常量 pp及95%区间|
|---|---:|---:|---:|---|---|
|44|66.087884|66.310435|66.551599|+0.463715，[+0.293334,+0.586733]|+0.241164，[+0.092121,+0.305968]|
|45|66.087884|66.309953|66.493020|+0.405136，[+0.277193,+0.514552]|+0.183067，[+0.047673,+0.266730]|

每个seed的net增量均超过0.2pp且图像配对下界>0；内容差均为正区间，seed45的内容点值0.183pp小于0.2pp幅度参考，完整披露，不混称其也超过该参考幅度。两个seed不是训练重复性置信区间；这是两次固定训练的重复验证。

完整四指标差值pp：

|seed/比较|Mask|bbox|Boundary2%|Boundary1%|
|---|---:|---:|---:|---:|
|44，真−强SAT|+.46372|+.30625|+.51838|+.47471|
|44，真−常量|+.24116|+.14640|+.24770|+.28140|
|45，真−强SAT|+.40514|+.30522|+.45784|+.36594|
|45，真−常量|+.18307|+.19020|+.16518|+.23445|

seed45由`scripts/probes/ue_geometry_proposal_repeat_20261024.py`只改变初始化46、data45与epoch1045000+e，从同一个新初始化分别训练G/K，仍E10/2670。`scripts/probes/ue_geometry_conditional_configured_20261024.py`是数值同引擎的checkpoint/out-root参数化版本，先以seed44模型精确复现两张缓存val，再运行seed45工程门/全test。`scripts/probes/ue_geometry_repeat_owner_20261024.py`核查10轮配对顺序、不同于seed44的完整初始化、全test前缀跨内容及跨seed逐字节相同，最后固定net/content两项500次四指标分析。该owner已完成，不是仅提交了训练任务。

## 十四、数据来源与实际纠错机制

`scripts/analysis/ue_probe_input_identity_audit_20261024.py`已完成：36734个归档文件SHA全部匹配，3060个native缓存的相机+五个depth来源均匹配同一归档，cache版本/来源无错误。输出在`ue_probe_input_identity_audit_20261024`。这排除了归档字节变化/串缓存来源的解释，不认证全库RGB-depth同表面配准、遮挡或标签语义。

`scripts/analysis/ue_geometry_new_instance_audit_20261024.py`已完成全459test的COCO segm一对一匹配核验。保留原强SAT检测的匹配/ignore在全部十个IoU阈值逐项相同；新增假设实际带来的新增GT匹配：

|指标|真几何|常量控制|
|---|---:|---:|
|IoU .5新匹配GT|811|476|
|IoU .75新匹配GT|253|176|
|.75新匹配、原来.5已匹配|75|93|
|.75新匹配、原来.5也缺失|178|83|
|>100实例图中的.5新匹配|688|380|

真几何.5新匹配中，标准bbox-area small451、medium359、large1；密集图占688/811。不是对旧mask进行边界修正，旧mask RLE完全不动，主要在补充实例假设和相应掩码。

原始新增候选真51354、常量50846；.5未匹配分别50543/50370，含背景、重复、定位不足，不能全称背景误检，也不能把低分数带raw精度称部署工作点精度。

为排除输出个数差异，`scripts/analysis/ue_geometry_count_matched_20261024.py`按每图min(G,K)的新增个数、各自最高分截断，使每图新增数严格相同（共50838），不使用GT准入。Mask点值仍为66.551599对66.310435，bbox也不变。该项是事后预算稳健性诊断，非按test选择的新模型策略；不额外宣称匹配后bootstrap重复结论。

## 十五、固定权重的运行时内容归因

新增`scripts/probes/ue_geometry_conditional_intervention_20261024.py`，固定seed44真几何E10，checkpoint不变：

- constant：仅四字段替换为train-only均值，保留真实support。
- shuffle：支持区域内联合置换四维向量，保留support和向量集合，seed20261024+pointID；局部物理一致性被破坏，作为OOD诊断。
- 引擎的correct模式先精确复现两张原缓存val，然后每个干预各做2val原SAT恒等门和全459test。源及PID在`ue_geometry_content_intervention_20261024`。

完整点值：正确66.551599，constant输入66.230773，shuffle输入66.229936。它们是同一模型的输入干预，区别于第十三节的独立常量训练控制；本轮没有出现旧native尾部路径中“零字段推理更好”的现象。

因前两项保留真实support，另加`scripts/probes/ue_geometry_whole_map_donor_20261024.py`：整个四字段+support图取固定半循环229偏移的另一test样本。内部图保持真实结构，但与当前SAT场景错配；源点XY距离中位878.2m。offset0先通过原缓存精确复现门，再固定229全量验证；不随结果挑donor。输出根`ue_geometry_whole_map_donor_20261024`。这是场景配对诊断，不是相机位姿误差的精确物理反事实。

`scripts/probes/ue_geometry_intervention_analysis_20261024.py`共用正确输入evaluator，固定同一500次采样，只针对上述功能依赖问题计算Mask AP配对差（主net/content报告已覆盖四指标）。验证相同checkpoint、GT/图像、纯SAT前缀逐字节相同，每evaluator前三抽样校准原函数≤1e−12。正确−constant已完成+0.320826pp，95%[+0.239626,+0.452999]pp；shuffle和整图donor区间仍运行。所有干预都不参与新模型/阈值选择。

架构与证据概要另见[几何实例候选补充架构](UE_GEOMETRY_INSTANCE_SUPPLEMENT_ARCHITECTURE_20261024.md)。本节结束时仍待输入归因与总验收，最终完成状态见下一节。

## 十六、最终收口

三项固定输入归因已完成：正确−constant +.320826pp，95%[+.239626,+.452999]；正确−shuffle +.321663pp，[+.241482,+.451779]；正确−完整donor图 +.378593pp，[+.248073,+.480274]。相同checkpoint、图像/GT、纯SAT前缀、固定500次配对及参考累计等价核定均通过。

新增可重跑总验收`scripts/analysis/ue_geometry_direction_final_audit_20261024.py`，核对四个实际E10权重、202581参数、每参数2670 AdamW更新、20轮配对样本顺序及跨seed不同初始化/数据顺序、两seed四指标结果/24次累计校准、三输入控制、每图同数量控制、实际新增一对一掩码匹配、全部输入来源和旧49源不变。实际运行通过，报告`ue_geometry_direction_final_audit_20261024/audit.json`。

交付：[最终阶段结论](UE_UAV_EFFECTIVE_DIRECTION_VERDICT_20261024.md)及[完整架构](UE_GEOMETRY_INSTANCE_SUPPLEMENT_ARCHITECTURE_20261024.md)。这关闭了当前“找到有实际效果的方向”的阶段目标；正式共同初始化联合训练、UAV RGB/CVT增量及独立区域泛化仍按未来开发问题记录，不宣称已经完成。
