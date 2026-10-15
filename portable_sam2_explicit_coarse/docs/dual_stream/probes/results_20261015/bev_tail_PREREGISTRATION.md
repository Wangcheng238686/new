# 六图高度尾部与最终GT框mask误差：CPU描述关联预注册

固定六图[1218,950,306,1216,948,505]；两CVT层完整报告。复用E72 EMA correct attention和dual、E66 EMA single的现有GT共同框最终mask RLE/IoU、GT标签、geometry缓存；不新forward/GPU/fit/训练/干预/AP，不改生产。

映射由runner独立实施：按真实project_source/project_pixel，source连续指标经sum(project_mass*quality*metric)/max(sum(weight),1e-6)映到卫星target。stride16(32²，真实SAM入口)主，stride8(64²，P3)伴随，stride1只几何诊断不作为性能入口。map必须另存denominator/rawmass/coverage/quality/missing；未覆盖不能填0参与关联。实际pixel对target归桶、源重复票保留，与生产一致。具体字段/不变量由mapper作者补充，不改变以下统计。

## 固定主关联

X1=actual weighted_abs_dz_m−uniform weighted_abs_dz_m；X2=1−actual height_mass（|dz|<=2m），两层各自完整，不合层后择优。每GT框取covered单元，与实际bbox交叠像素面积加权平均；不用插值或阈值筛tail。分母为covered bbox交叠面积，0支持RoI的X=None单列，不当0。框内格映射是空间暴露描述，不等于该GT独占source。GT建筑mask内部covered像素均值另作伴随，不与框内均值混用。

Y_degradation=singleIoU−dualIoU，正值为双流退化；Y_dual_error=1−dualIoU。主每图Spearman（平均rank ties）及Pearson，图内ROI描述后六图等权汇总相关系数（不Fisher，不p值/CI/工程门）；相关未定义(count<3或常数)为None并报实际图数。pooled ROI只描述，另全六图先按global_id聚合均值再相关展示重复目标影响，不当独立建筑推断。

## 伴随与混杂

逐图/图等权报告X分布、单/双流IoU差及绝对错误，bbox面积、clipped(x1<=0或y1<=0或x2>=512或y2>=512，仅触图边界不作完整率推断)、covered面积比例、bbox平均coverage/quality和GT边界比例。每主X与这些混杂的连续相关仅诊断并列，不做六场景多变量拟合或事后阈值/分组搜索。

复用最终二值RLE（没有概率）与GT512 mask重新计算历史IoU逐项exact。伴随像素误差仅在GT共同框内：预测与GT XOR，dual−single XOR为+1表示新增错；GT边界带固定为二值GT膨胀3px与腐蚀3px之差（7×7方形、边界外0），报告boundary/interior/非边界background中实际covered交叠单元的误差率差、dual绝对错误与X1/X2平均；框内pixel算到格内像素数后统计，像素仅重复测量不作独立样本显著性。不会依据这三个区域择优变主终点，不将GT边界当UAV屋顶真值。无覆盖像素分开报，不人为复制覆盖信号到未覆盖区域。

两流权重与优化历史不同；关联不等于attention/分支因果，GT共同框排除检测误差、不等完整detector AP。几何相容不是语义正确，layer残差/FFN与source混合使暴露不是完整贡献。六图可能重叠同建筑，结论仅这一固定子集；不与训练raw遥测比较。

## provenance与验收

保存脚本/协议/映射/attention/geometry/RLE/GT/身份文件SHA；GT slots、bboxhash、area/globalid历史逐项对齐；mask IoU独立CPU复算精确，不写新mask。地图与production加权分母/coverage等对账由独立审计；统计数组finite、无支持None、分母与有效n全部报告。源文件只读；工件仅/tmp/ue_bev_tail_association及/tmp脚本。独立审核通过后由主代理归档，不自动干预。

## 映射实现与数值验收（runner补充，执行前固定）

映射脚本/tmp/ue_bev_tail_map.py仅CPU，不建网络。固定六图两个layer各完整attention NPZ，源query的连续几何指标读取不改。每个缓存投影entry e含source/pixel/mass/quality。stride s∈{16,8,1}，target=(pixel//512//s)*(512//s)+(pixel%512//s)，w_e=mass_e*quality_e。每target D=sum_e w_e、rawmass=sum_e mass_e，metric=sum_e(w_e*a_source)/max(D,1e−6)。这是几何权重映射，未包含learned融合门、残差、FFN或decoder处理，不能称最终信息/误差贡献。

主stride16真实SAM/P4入口32²；stride8 P3真实入口64²伴随；stride1 512²只细像素几何描述，不作为新模型入口/主关联。所有map每layer独立，不按结果合层/选层。metric字段actual/uniform weighted_abs_dz_m、height_mass、xycore_mass、joint_mass、signedz、XYoffsetnorm，另abs_dz_excess=actualabsdz−uniformabsdz和height_incompatible_mass=1−heightmass。source均uniform candidate的定义沿前attention；先每source各自计算，再生产权重投影，不做新拟合。

mean_quality=D/max(rawmass,1e−6)，coverage=clip(rawmass/s²,0,1)；missing_mask=(D==0)。所有attention metric在missing处NaN，rawmass/D/coverage保留原值，不把无覆盖当零尾部。即使某格有mass但quality0也按D0 missing，不能把其metric填0进行关联。当前源候选无效query的NaN仅在未被project_source使用的dense feature占位转0以便调用CPU生产池化，所有实际source均assert hascandidate&heightvalid，不能消费此占位。

每图cache file SHA与前rawgeometry记录精确一致；cache所有数组tensorSHA逐项等attention运行manifest payload SHA；reference/flags/hascandidate/projectedmask和source唯一集合从attention NPZ逐数组精确核对。inputs缓存前后原字节SHA保持。所有attention NPZ SHA精确核manifest，datasetmanifest/checkpointSHA及source脚本/合同来源锚保存。

数值：独立numpy FP64 bincount累积vs实际CPU生产BEVToSatellite(torch FP32 segment_reduce)，所有metric逐数组allclose(atol=2e−5,rtol=2e−6)，meanquality/coverage各atol=2e−6,rtol=2e−6。两者不声称bitwise等同，完整记录maxabs/maxrelative；有任何越门先停审不改阈值重跑。missing在生产metric/meanquality应0；coverage仅rawmass为0时必须0，D0但rawmass>0时仍可正；同时记录target entry归一权重和应为D/max(D,1e−6)的误差，若D<1e−6不误称sumweight=1。无新GPU、训练、模型干预、AP或performance门。

## 映射数值验收v2明确修订（首轮失败后、完整映射前）

首轮1218/L0/32在原FP64-vs-FP32 allclose门FAIL并停审：仅weighted_abs_dz两值，target572/740，各220/256重复票，差4.42951e−5/6.82692e−5m，相对3.71e−6/4.76e−6。独立numpy按稳定target内部原票顺序FP32逐项累加分子/分母，两个输出与生产bitexact；分母乘积量化误差0，主要为FP32串行累加舍入。原门、原脚本、协议和失败log保存在mapper_first_attempt.py、PREREGISTRATION_first_attempt.md、mapping_first_failure.log；诊断见roundoff_diagnosis.json/roundoff_sequential_detail.json及精确脚本，不覆盖或称原门PASS。

经主代理与独立审同意，v2主实现验收改为**全36map、全部12字段**独立numpy np.add.at无缓冲FP32逐entry累计，保持每target与生产stable sort相同原票顺序，包含mass*quality和numerator逐元素FP32乘法、分子/分母/rawmass的逐项FP32 sum、最终max-clamp除法与quality/coverage；对真实CPU生产模块metric/quality/coverage要求逐数组bitexact。若不相同先停解释，不经验改常数。无效target按生产0处理。

原FP64地图保持相同数学定义作为预定关联主输入，明确非生产bitexact；其与生产差完整报告，不再设置经验放宽门，原门pass/fail与超门数仍每map记录。固定另存production_fp32_abs_dz_excess及production_fp32_height_incompatible_mass作为伴随数值敏感性，不按结果择优主输入，不做新fit/新模型/性能检验。纯数值验收修订不改变样本、几何阈值、关联指标、统计方向、层/尺度或结局。
