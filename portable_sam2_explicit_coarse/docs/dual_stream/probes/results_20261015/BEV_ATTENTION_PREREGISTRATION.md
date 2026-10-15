# 有效投影query的实际CVT attention来源，只读冻结诊断

固定六图顺序[1218,950,306,1216,948,505]，与上轮raw geometry相同；E72 strict EMA correct UAV，当前开发子集非独立test。先独立静态审查/root准入1图1218 smoke，再独立验收/root准入唯一六图完整。无AP、训练、fit、权重干预、候选硬筛或新样本，不改生产源码。

每层forward prehook获取实际query/tokens/payload，按生产源码完全复算norm/query/key/value、anchor position+view embedding、clamped relative几何MLP bias与−0.5 distance、invalid−inf、无候选dummy，head8/width32/scale sqrt(width)。FP32/inference_mode/noautocast，与本冻结生产forward一致。按256query bounded chunk重算全部16384query，不使用旧256固定采样。每head softmax后headmean为主线性指标；dummy无候选的描述指标NaN，不能计入组mean或当观测。保存轻量perquery NPZ，绝不存完整权重矩阵。

## 数值准入

每层确定性最多16个sample query：projected、heightvalid&hascandidate&nonprojected、heightinvalid&hascandidate、nocandidate四组各最多4个，组内ID linspace取整；空组不补样，保存各组实际数量。sample的nocandidate保留生产dummy softmax（head weight在dummy为1）与−inf bias，最终manual/actual均乘hascandidate归零，不能当观测；sample保存实际投影Q/K/V/bias、8head weights/candidate_index/delta和explicit/production layer.attend输出小块，独立CPU可重算。样本explicit attention output对生产F.scaled_dot_product_attention路径layer.attend对账：maxabs<=1e−5 **或** relativeL2<=1e−5；两者完整报告，不因结果改门。失败停审。smoke hook开启/移除后全FPN/SAM/PE/high features SHA逐值相同；features相同以bitexact SHA，不用近似。

## perquery schema

attention_IMAGE_layerL.npz，每层分开，query_id完整0..16383：
- candidate_count与height/xycore/joint_candidate_count和fraction，valid candidate为分母。
- height_mass(|dz|<=2m)、xycore_mass(|dx|及|dy|<=2m)、joint_mass；真实8head softmax后mean。
- view_mass[N,5]与view_dominance=max(headmean viewmass)，不是每head max再平均。
- entropy_per_head[N,8]=H(head)/ln(validcount)，validcount=1时0；entropy_normalized为8head该值平均。entropy_headmean_distribution另存，不混同两定义。
- weighted_signed_z_m=sum(w*dz)，weighted_abs_dz_m=sum(w*abs(dz))；weighted_signed_xy_m二维、其norm与mean abs XYdistance分别存，不把有抵消偏置当平均距离/配准错误。
- uniform_*同套height/XY/joint/view/offset指标和uniform_entropy_normalized（count>1为1、count1为0），uniform按有效候选均匀，便于同集合候选比例与实际权重比较，不作因果或roof精度指标。
- observation_support、height_valid、has_candidate、projected_mask各自保存。projected_mask=unique(project_source)，不按quality新筛。reference_xyz保存。
- project_source_mass为project_mass按source累计；project_source_mass_quality为mass*quality累计；mean_quality为后者/前者；project_source_vote_count为fine pixel票数。它们分别用于描述等query/原mass/massquality加权，不能称独立样本或真实误差。

无候选所有浮点attention描述字段NaN（candidate counts/fractions保留0），不伪报entropy0；candidate有但height invalid的query另组报告，不当有效高度支持。height compatible不是屋顶正确，低兼容不是必然污染；xy外可能是合法上下文，无UAV表面真值。

## 不变量、输出与解释

manifest含checkpoint流式SHA、全部runner/helper/CVT/diagnostics/projection/geometry/contract源码SHA、dataset manifest SHA、模型architecture/geometry contract、git SHA/dirty、每图payload/RGB SHA和完整feature SHA、参数_version不变、hook清理、输入query/token/payload哈希前后一致、两层sample误差、各query NPZ SHA。smoke保存on/off exact；正式不重复off，字段null为未重复，不是假失败。预算六图<10分钟；若成本明显超预算或检查异常先报告root。

统计由另一代理CPU实施：两层分开，明确allhascandidate、heightvalid&hascandidate、projected_unique组；等query/每图等权以及投影rawmass/massquality描述分开。不做显著性/工程收益门，不按某层/某图结果后选，不把训练日志旧采样与EMA投影子集直接相减。定位只允许“候选pool包含哪些几何上下文，实际attention在本组如何分配”；不能据此断言UAV branch净收益/污染、冻结权重因果或修改注意力必增AP。

静态审后准入收尾：现场assert q/keys/values/position/bias全float32、无autocast、nquery16384/width32，report.reconstruction_dtype='float32'；projected必须heightvalid&hascandidate。sourceSHA包括全部geometry依赖、data cache/dataset/payload、dual_detector/fusion及间接helper和rawgeometry summary。每图cache path/SHA精确匹配上轮rawgeometry记录，当前loaded payload全部cache数组逐值相等，保存geometry_cache身份和arrays_exact。不额外读取原始EXR或重建rawgeometry。
