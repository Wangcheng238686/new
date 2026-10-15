# 有效投影query实际attention来源独立审核

审核者：frozen_probe_audit。只读核验生产源码与实验工件；不参与runner或汇总开发，不运行GPU或训练，不扩大固定六图。

## 准备核验

已核对HEAD `a41d1efbe0888df5c1357d2f134221875882ec5f`，并阅读全文：`UE_BEV_WEAKNESS_DIAGNOSIS_20261015.md`、`UE_BEV_GEOMETRY_QUALITY_20261015.md`、`rsprompter/uav/cvt.py`与`diagnostics.py`。

生产注意力为query经过norm/query线性层，key为token key加anchor position/view编码；relative xyz按radius/radius/10缩放后clamp，再形成平方xy距离，relative MLP输出减0.5 distance。按head width平方根缩放点积，mask转为query dtype后交SDPA。旧diagnostics为最多256个has-candidate网格query、近似FP32重算，不能代替本轮有效投影全query观察。

必须区分signed offset与加权绝对/欧氏距离、候选数量比例与attention mass、query等权与投影mass加权。height-compatible不等于语义正确。六图固定为1218、950、306、1216、948、505；不得自动扩153。

## 初版实施独立审核

已阅读`/tmp/ue_bev_attention_probe.py`及`bev_attention_audit/PREREGISTRATION.md`。未参与runner或summary开发。

VERDICT：初版核心数学与观测实现通过静态核验；最终一图smoke准入仍待下列定版收尾。未运行GPU。

1. 两个CVT layer prehook读取实际输入，每次观察返回None；不替换forward/attend，额外调用的Linear/LayerNorm/SiLU无运行状态更新。每层全16384query按256chunk处理，临时候选张量有界；没有保存全量head权重矩阵。每层独立query/token/payload前后hash，模型参数version、全features on/off SHA及hook清理覆盖合理。
2. 重建query norm/linear、keys/values、anchor position/view编码、relative clamp、distance、learned bias和点积head scale顺序与生产相同。当前无autocast且原模型FP32时等价；建议明确断言并记录q/key/bias实际FP32，以便证明生产`bias.to(q.dtype)`语义，而非依赖隐含状态。
3. invalid候选mask=-inf，无候选仅dummy首slot置0，manual与actual均乘has归零。无候选浮点描述置NaN；candidate count/fraction留0，不能当观测entropy0。sample包括no-candidate时仍可核对dummy数值，不会把dummy归入实际attention来源。
4. 每head softmax后平均用于线性mass/offset，视角max作用于headmean视角分布。entropy_per_head=H(head)/ln(validcount)，先逐head规范化再mean；另存headmean分布entropy，validcount1为0。uniform按valid候选等权。height/core/joint候选比例与实际mass分别保存，未混淆。
5. project_source unique定义投影query；project_mass、mass*quality按source分别bincount，meanquality为加权量之比，fine票数另记。projected不按quality筛。signed z/xy offset、offset norm、mean absolute z及欧氏xy距离分别保存，避免抵消偏置冒充平均距离。四类sample按固定组内linspace，最多各4，空组不补；bounded sample用于SDPA数值核对而非主统计样本。
6. geometry高度bin及半cell均2m已有强assert。建议同时固定实际allgrid16384、headwidth32，并断言projected均height_valid且has_candidate，令预注册有效投影语义与工件身份直接一致。
7. 来源已记录checkpoint流式SHA、runner/helper/部分生产源、manifest、payload所有非RGB数组hash与RGB hash。建议补全实际几何与读取依赖的source SHA：candidates/surfaces/footprints/cameras、ue_geometry_cache/ue_rgbd_dataset及dual_detector/fusion。无需新增rawdepth重算或GPU；当前payload数组hash足以钉住本次模型实际输入。已向root/runner报告，审核者不改runner。
8. 本轮只观察actual current EMA、固定六图，两层全query后由CPU按有效集合汇总；旧256grid训练统计不代替本轮。height相容不是roof正确，低相容不等于污染，query平均与投影mass加权不得混用；任何结果均不授权训练、候选屏蔽或扩大153图。

待定版收尾核验后，由root批准1218唯一一图smoke；审核者随后只从保存Q/K/V/bias/weights小样本CPU独立重算，并验收on/off全feature精确回放。

## 定版实施复审

VERDICT：PASS。建议root准入唯一一图1218 smoke；六图正式仍须等待smoke工件独立验收与root准入。不授权153图或训练。

已复核更新后的runner，前述收尾均落实：q/keys/values/position与bias明确FP32断言，记录reconstruction_dtype；固定16384query/8heads/width32；project_source unique均height_valid且has_candidate；source SHA补全几何候选/表面/footprint/相机、fusion、数据读取/cache/payload、dual_detector及间接helper。增加上轮rawgeometry summary身份、固定六图与manifest一致检查，当前cache SHA需等于上轮，loaded payload各cache数组逐值相同；每图记录cache路径/SHA/exact。审核者CPU读取1218实际cache确认所有数组为可转换数值/布尔类型，与检查方式兼容。

补项只约束dtype/来源/集合，不改变观测数学定义或生产模型。预注册公式、dummy语义、entropy、uniform对照及query/mass加权边界与定版实现一致。尚未执行GPU或验证实际SDPA误差；这些是smoke的实际准入条件，不能提前记为结果PASS。

## 1218一图smoke独立验收

VERDICT：PASS。建议root准入预注册固定六图唯一完整运行，顺序1218、950、306、1216、948、505。不得扩大153图、训练或候选干预。

审核者只使用保存工件CPU复核，无新增GPU。manifest显示smoke_complete=true、complete=false、image_ids=[1218]，strict EMA加载无缺键、多键，无optimizer/训练，parameter_versions_unchanged=true。全部记录source SHA、预注册SHA、两层attention/sample NPZ SHA及历史cache SHA独立重算吻合；loaded payload各数值数组哈希独立从cache重算均匹配manifest。全FPN/SAM/PE/high hook-on/off feature SHA精确相同的运行断言及记录通过，hooks清理、payload/RGB和层输入前后不变；当前HEAD clean a41d1ef。

两层各完整16384query、8heads、width32、FP32、256chunk；有效投影unique query均6960。独立从历史cache全部candidate/anchor/reference重建has_candidate、project_source unique、height/core/joint候选计数与比例、source rawmass与massquality及票数，逐项符合attention工件；projected均height_valid且has_candidate。

两层各16个数值样本，四个预定几何组各4个；独立按组内linspace从cache重建所选query IDs，全部吻合。CPU float64从保存Q/K/V/bias重算点积、sqrt32缩放、softmax、headmean、attention输出：权重与保存FP32权重最大容差2e−6内相同；对实际生产SDPA输出层0 maxabs=1.3362802e−6、relativeL2=3.2742785e−7，层1 maxabs=9.2955536e−7、relativeL2=2.3891407e−7，均低于预注册门。运行自身FP32对账maxabs分别1.3709068e−6/9.5367432e−7，两者均达门。

用相同CPU重算权重独立复现样本height/core/joint attention mass、signed z/xy offset、absolute dz及逐head规范化entropy；headmean entropy聚合与记录一致。无候选样本manual/actual输出均严格0，全grid无候选attention描述NaN、有候选有限，未把dummy当有效观测。candidate fractions保持0，与浮点attention描述NaN区别正确。

CPU汇总脚本静态检查未发现阻断项：单query候选均匀比例与candidate票等权比例分列，projected query等权、rawmass、massquality及图像等权分别报告；massquality未冒充实际下游按target归一化贡献。entropy先head后均值及headmean分布另存；no-candidate不入质量均值，height-invalid单列。smoke不能支持六图机制结论，正式结果仍待独立验收。

本次CPU数值重算以保存的Q/K/V/bias为输入；norm/query/key/position/relative MLP生成语义已通过生产源码静态对照，并非重新运行整个GPU模型。没有从GT、AP或训练日志选择样本，也没有增益门。

## 固定六图正式结果独立验收全文

VERDICT：PASS。未发现实施或结果工件阻断项。结论仅限固定六图当前EMA的注意力几何来源描述，不产生训练、性能、污染或因果判断；不授权进一步样本扩展、候选干预或模型变更。

审核者frozen_probe_audit，未参与runner与CPU汇总开发。审核只读CPU检查正式manifest、12份全query attention NPZ及12份数值sample NPZ；不新跑GPU、fit、GT评价或AP。实际六图顺序为1218、950、306、1216、948、505，与预注册一致；complete=true、smoke_complete=false，elapsed26.5735秒。strict E72 EMA无missing/unexpected，无训练/optimizer、参数version不变、HEAD clean a41d1ef。

**来源和无扰动。** 独立流式重算checkpoint SHA，全部runner/helper/生产依赖/历史rawgeometry summary SHA、预注册SHA、24个NPZ SHA、六份cache SHA均吻合manifest。逐cache数组重算带dtype/shape的payload hash，与实际输入记录一致。六图记录hooks_removed、payload_unchanged、geometry_cache_arrays_exact及两层input_hashes_unchanged均True。正式hookon_off_features_exact=null按预注册表示未重复off；一图smoke全FPN/SAM/PE/high逐值SHA精确回放已验收，null不是失败。没有参数更新或生产路径替换。

**全query和分母。** 两层各六图×16384query，8head、width32、FP32、256有界chunk；绝非旧256固定采样。按cache重新计算candidate_index/anchor/reference几何、project_source unique、source rawmass/massquality及票数，所有保存结果逐项吻合。六图投影query分别6960/4623/6655/6543/4601/4668，合计34050；跨图不按世界位置去重，不能视为34050独立物理表面。全部projected均height_valid且has_candidate。候选票855882，height-compatible票比例.7972968236275562，与上轮几何审计一致。

**数值小样本。** 独立重建每层四个预注册组内linspace样本IDs，每组4个，12层工件共192个query样本；不把这192个当主统计集合。CPU float64从保存Q/K/V/bias计算点积/sqrt32、softmax、headmean及attention输出，权重在2e−6绝对容差内等于保存FP32权重。对生产SDPA输出最大绝对误差1.3840099332e−6，最大relative L2 3.2742784797e−7；全部低于预注册门。逐样本独立重算height/core/joint质量、view mass、signed xy/z offset、mean绝对dz/欧氏xy距离、逐head normalized entropy均吻合。全部无候选sample实际/manual输出严格0，全grid无候选attention描述NaN、有候选有限；未将dummy当观测。生成Q/K/position/relative MLP的上游语义通过源码静态核对，本次CPU重算未再次执行整个模型。

**独立关键汇总。** projected query等权层0/1的height mass为.795887086823/.796110182528，同集合uniform为.804386179293；xycore mass为.178875150821/.178585620297，uniform为.163334428134。不能把候选票比例.7972968代替uniform query均值.8043862；前者按候选数加权query，后者每query先归一化再等权。

按source rawmass加权，height mass层0/1为.796300522152/.797102903268，uniform .805553157029；xycore .182526364096/.182218523988，uniform .166908668869。按source massquality加权，height mass为.846138894752/.844364694491，uniform .850881645785；xycore .155512445614/.155902925639，uniform .139876707441。以上三种分母不同，均为同集合描述；massquality是投影前source累计，不等于按卫星target归一化后的实际下游贡献。query pooled与每图等权也须分列，不能混称总体attention。

**解释与局限。** 实际height mass相对同分母uniform略低、xycore略高是当前权重如何使用候选几何上下文的描述；不能由此宣称网络学习变坏、约20%污染、屋顶错误率或损失语义。height-compatible不是语义正确，局部外与异高候选可包含合法立面/上下文。signed offset存在抵消，不能替代meanabsdistance或外部配准误差。height-invalid query的参考z不可靠，只能描述模型坐标输入，不能当真实高度误差。旧E72 raw训练最后层的.508来自不同参数/数据/采样/聚合，不与本轮数值直接相减。

两层结果分别报告，不挑层/图；单图smoke用于数值和无扰动准入，六图正式仅完成同一预注册范围。没有外部UAV表面真值、GT标签使用、训练、干预、AP或工程收益门。注意力质量不是表征语义质量或最终贡献。正式CPU汇总由另一代理负责，主代理将独立复算其完整图像等权/分位数；本审核独立覆盖原工件、192数值样本、全query几何/来源分母及上述关键均值。

### CPU数学一致性验收容差补记

最终summary validation=PASS，主代理已完成全部group均值、三种权重、uniform差、candidate票比例和图像等权重算，记录`root_numeric_audit.json`。

独立检查全12份NPZ发现原绝对容差`weighted_abs_dz + 1e−5 >= abs(weighted_signed_z)`在573条query上触发：572条height-invalid、1条height-valid但未投影；有效projected为0条。最大FP32差6.103515625e−5；独立以`gap/max(abs(signedz),1e−12)`计算最大约2.20e−7（分母/筛选口径稍异不影响量级）。修订为`1e−5 + 1e−6*abs(signedz)`后全部通过，容差与FP32大幅值求和舍入匹配。该检查是三角不等式数值验收，不是模型统计或机制门。

此修订没有改变原始attention、NPZ、任何均值/权重/样本选择、FP32重建或预注册SDPA误差门；无需重新GPU。较大绝对offset主要在高度无效query，不把它解释为物理错高。该容差修订及原触发行数应随结果保留，不能隐藏为首次即通过。

最终VERDICT维持PASS，未发现须新增GPU或改变结果的阻断项。


## 主代理统计复算

另以不导入模型源码的BEV_ATTENTION_ROOT_AUDIT.py，读取原始NPZ独立重算关键均值、分组/候选票、三种加权及图等权，PASS（差<1e−12）；不是全部分位数的独立复算。最初图等权使用FP32导致严格容差失败，修正独立检查器为FP64后通过，正式汇总不变。具体PASS结果随probes/results_20261015归档。
