# 真实UAV内容收益：独立复核与定向诊断

## 当前判断

用户要求派独立子代理解释内容归因失败并实施必要探针。本次派出内容归因、读取/消费、下一步方案三路。三者共同认为：A/B/C/D未达成正确RGB的整体Mask AP增益判据，不应把新增适配器对卫星的上涨当作正确RGB贡献。当前没有理由再重复同类concat/offset训练。

D正确训练Mask AP64.7999，匹配常量RGB训练64.7850，差仅0.0149个百分点，名义95%图像配对区间跨零。同一个D E10正确/常量/donor输入的AP分别64.7999/64.7851/64.7852。预测发生变化说明接线生效，却未证明变化具有足够正向质量。常量RGB仍保留目标深度、相机、候选几何和支持，不能称完全没有UAV信息。

### 几何训练价值的正证据

`ue_geometry_content_priority_20261023`四个真实/非零常量字段配对已完成；第三子代理独立核对全部459图记录SHA、GT/检测不变量、初始化、数据合同和逐轮样本顺序。

|采样/消费|真实−常量训练Mask差（百分点）|名义95%区间|真实模型−纯卫星（百分点）|
|---|---:|---|---:|
|bilinear/concat|+0.2718|[+0.0652,+0.2922]|+0.1817|
|nearest/concat|+0.2658|[+0.0678,+0.2871]|+0.1761|
|bilinear/difference|+0.1283|[+0.0284,+0.2198]|+0.0306|
|nearest/difference|+0.0364|[+0.0198,+0.1885]|−0.0727|

这支持几何字段的训练价值，不能称所有UAV内容均无证据。常量训练臂低于卫星，真实模型均未达到既定对卫星+0.2个百分点参考门槛；同检查点置零不劣的既有诊断仍限制对推理几何依赖的解释。单seed与开发test的多路线探索限制保持。

## 已在运行的针对性训练

本次核对实际PID和train.jsonl；这些是已存在的实验，不是本轮新启动：

- D辅助监督正确/常量/zero：检验表征是否缺少直接训练动力。
- N2正确/常量：将注入移至最后GELU，检验消费者路径。
- N2-shared正确/常量：显式让SAM logits与UDPR feature读取同一个增强tensor。
- S1-R正确/常量：按真实roi_local→bbox映射在原UDPR选点读取UAV证据，保持原选点/MLP/损失。
- Attention ladder既有内容配对继续按原协议运行。

原conv_s0接收全图repeat、最终mask受roi_local监督是混合语义；它构成需要检验的消费者限制，不能仅凭输入布局判定最后输出坐标错。旧S1整图选点假设确有错误，已独立退役；本轮只使用S1-R。

## 本轮新增的两个诊断

### E：RGB内容的净纠错

新入口`scripts/analysis/ue_rgb_content_error_audit_20261023.py`。

固定A/D E10，仅读已有完整459 test正确、常量、循环donor预测，无新训练或GPU推理。核定GT/框/分数身份，采用固定bbox关联，比较真实RGB相对干预的逐实例IoU、0.50:0.05:0.95阈值跨越，以及边界/内部/背景的纠错与新增错误。公开支持分区、匹配/未匹配和所有区域分母；重复实例像素不可解释成整图唯一像素。

意图：区分无方向扰动、正负抵消、只改善极少位置，以及改善不足以跨AP阈值。阈值跨越和固定框关联不代替COCO按分数匹配的AP；不据test分区调门控/范围。

### G：内容残差的损失方向与传递

新入口`scripts/probes/ue_rgb_content_direction_audit_20261023.py`。

固定A/D E10、4张预定训练密度分位样本，FP32、无增强/dropout、无optimizer更新。正确/常量/循环train donor成对，在实际注入张量处测内容差与总注入的RMS比例，分别对原mask和tail loss求梯度方向内积，并跟踪rawSAM和UDPR最终logits差异。

内容差定义为同权重correct−intervention。梯度内积符号必须结合参考点披露：在intervention处，负内积预测局部移向correct可降低该loss；在correct处，正内积预测移回intervention可降低该loss。实际配对loss另列，不能将一阶近似当AP或全训练动力证明。离散采样/点选择不连续时仅作局部诊断。

源编码/聚合可no_grad，receiver建立叶子；这只测消费者对已有内容差的敏感性，不测源参数训练梯度。所有实际mask调用逐一累计，不能沿用末调用统计冒充整图。源码、checkpoint、模型/adapter前后SHA以及optimizer_steps=0进入审计。资源先验收后共租，不抢占既有训练。

## 下一步判定

1. 若真实内容的纠错和新增近乎抵消，问题在证据质量/位置或拒绝能力；总残差RMS与非零梯度不能替代质量判断。
2. 若真实内容使loss下降，却很少跨IoU阈值，优先检查实例大小、冻结框/排序与修正范围，不急于加深融合层。
3. 若主mask方向有利但tail方向冲突或UDPR覆盖改善，结合N2-shared/S1-R实际配对结果判断消费路径；不凭4图梯度立即调loss。
4. 若D辅助让前景IoU改善而内容Mask差仍弱，说明语义可解码不等于实例任务可消费；不继续用辅助准确率宣布成功。
5. 首先要求正确内容相对纯卫星与匹配内容控制都有正向净收益，再用预先固定结构的训练seed重复/独立数据检验稳定性。没有架构能预先保证AP提升。

本轮观察全部独立离线保存，不新增训练逐步遥测，不改运行中的冻结源码或实验选权合同。

## 独立验收补充

第三代理实读两份新脚本，提出并采纳：E需独立后验绑定evaluation_identity/CP及source身份，核定净纠错恒等与两种支持互补分区守恒；其Hungarian实际为最大化总bbox IoU后筛≥.5，不是先最大化合格匹配数。原np.arange阈值有浮点边界差，长CPU计算保留原始产物，由已有逐实例IoU以canonical np.linspace(.5,.95,10)重算阈值计数并披露差异，不覆盖原记录。G补无leaf/有leaf同seed的raw/final logits、框和loss恒等门，以及门后重新读显存准入；修订前自有诊断进程停止，既有训练不受影响。

稀疏保留锚点唯一像素覆盖与native全深度观测覆盖、候选4×4格支持不同。原stage1全部落入low覆盖组不能说明UAV整体观测不足；原固定分组保持，不能按test重新分位挑阈值。

## 本轮实际取得的结果

### G已完整完成

A/D各4个固定train图×3种RGB内容，8个样本组/24次正式前向；工程门另计。raw/final logits、框、loss的leaf回放精确恒等；正式source未变、模型/adapter SHA前后不变，所有输出数值有限，无optimizer更新。A正式峰4334 MiB，D全程峰4342 MiB。

- A correct−constant内容差RMS/总注入RMS为0.616–0.668；D为1.090–1.139。不能将比值当信息贡献比例；它表明内容差并不接近零。
- 对train donor，A内容差/总注入0.412–0.502，D为0.399–0.534。该诊断仍保留目标几何，属于不一致输入。
- 24次前向形成的所有配对中，UDPR后与rawSAM内容差RMS比为1.0008–1.0129，没有出现总差被显著压小的现象。原生选点随内容改变，因此不是强制同点UDPR因果实验。
- A正确RGB的mask loss相对constant在4图均低（平均−0.001584）；对donor为3低1高（平均−0.000356）。D相对constant和donor均3低1高，平均−0.000574/−0.000083。
- tail方向并不总与mask一致：D相对constant/donor的平均tail loss差为+0.000230/+0.000514。仅4图且局部方向/选择有变化，不足以据此调整tail权重或认定其造成测试失败。

这排除了本组样本上的“RGB完全不改变特征”“变化在UDPR前被整体抹掉”解释。训练图上mask loss偏好correct与测试AP未占优并存，剩余重点为泛化、修改质量和loss→AP转化；高维梯度cos小没有独立基准，不能单独判无信息。

G完整产物：`/data/wangcheng/checkpoint/ue_rgb_content_direction_audit_20261023/`。

### E：A完整459图已完成，D继续运行

A固定bbox关联40363实例，未匹配GT4625/DT569；根代理独立核定全部区域净纠错恒等和两种支持互补分区守恒。

- correct−constant平均实例IoU为−0.000904：改善17186，恶化22266，不变911。边界纠正332282/新增393601，净−61319；背景净−49879，内部净+13865。
- correct−donor平均IoU为+0.0000723：边界纠正117818/新增116864，净+954，正负几乎抵消；背景净−4342。
- mask面积small组27846匹配实例的correct−constant平均IoU为−0.001189；固定>100实例密度组为−0.001104。均为描述性分组，不能与canonical bbox-area尺寸组混称。

这些像素均按匹配实例计、区域有裁剪且实例可重复，不是整图唯一像素；不能把阈值跨越/像素净纠错等同AP。A证据已经说明部分内容修改质量不足，不只有幅度太小。D全459及独立canonical阈值验收继续计算，不在此提前下D净纠错结论。

## E最终完成与本轮继续实施

A/D均已完成459图，`verification.json`通过全部身份、匹配计数、区域守恒和canonical阈值重算验收。D correct−constant平均匹配实例IoU +0.0003254，边界净纠正36084、内部+482、背景−2887；correct−donor平均IoU +0.0003199，边界净+18575、内部+366、背景−115。这是微弱的GT净纠错信号，不能替代整体AP或训练内容配对证据。

D的mask面积small占匹配实例69%，correct−constant平均IoU仅+0.0000326；medium约+0.000980，large约+0.000892。固定密度>100组仅+0.000180。由此保留“大量小实例修不动、部分正负抵消”的描述性线索。前文D继续运行是历史进度，以本节及最终验收为准。

### 新增G2：跨划分内容方向（用户授权继续实施）

- 新独立worker：`scripts/probes/ue_rgb_content_generalization_audit_20261023.py`；从已验收G复制，唯一任务变化是按CLI指定val/test取数据、记录split和对应donor。原train脚本及冻结source不改。
- 持久串行owner/collect：`scripts/probes/ue_rgb_content_generalization_finish_20261023.py`。
- 根目录：`/data/wangcheng/checkpoint/ue_rgb_content_generalization_audit_20261023`。
- 每个val/test划分按GT实例数稳定排序，提前固定.125/.375/.625/.875四个分位索引；A/D E10、correct/constant/本划分循环donor，共48次正式前向与下游反向；工程门另计。
- 沿用FP32、eval、无增强、无optimizer更新、源编码no_grad、实际receiver叶子方向、分mask/tail loss、全部调用统计与raw/final/选点变化。每划分先单图恒等和容量门，再按实测预算加2GiB余量重新准入；不要求独占GPU。
- source/checkpoint SHA、抽样规则与用途先写plan/manifest；不按诊断结果选样本、改结构、选权重或调loss。
- 自动汇总既有train4图与新增val4/test4图；**不同划分不是配对图，不计算跨划分的配对CI**，均值仅作描述。test为已经反复使用的开发test；本诊断使用GT正proposal计算loss，不是部署AP，也不是独立泛化确认。

G2意图是检查“训练样本通常有利、测试净收益有限”的loss方向是否随数据划分改变。E已有完整分组，优先用这些结果解释G2与运行中的S1-R/N2-shared/辅助配对，不重复计算完整459图形态学。
