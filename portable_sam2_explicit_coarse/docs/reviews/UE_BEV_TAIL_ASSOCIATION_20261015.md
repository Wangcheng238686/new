# 高度尾部与卫星mask误差关联独立审核

审核者frozen_probe_audit，只读生产源码和既有工件，不参与mapper或关联实现，不运行GPU、训练、拟合、干预或生产修改。审核文件仅写/tmp。

## 前置调查及定义协调

已核对生产`BEVToSatellite`：target由fine pixel的行列按目标stride整数分桶；weight=project_mass*project_quality，输出=sum(weight*source_feature)/clamp(sum(weight),1e−6)，meanquality=sum(weight)/clamp(sum(mass),1e−6)，coverage=clamp(sum(mass)/(sx*sy),0,1)。真实部署FPN P3/P4为64²/32²，SAM embedding为32²；512²仅细像素描述。无支持target不得以tail=0混入有支持关联。

既有`full_fixed_box_audit/{dual_correct,single_best}.json`含真实GT共同框boxes/hash、IoU及每目标512² RLE masks，可CPU核对，不需重新模型forward。`full_stage_audit`含gt_slot/global_id/bbox_area供身份关联。

已向runner/evaluator协调：连续actual weighted absdz与actual−uniform差分别保留，1−heightmass伴随；标量投影描述不包含后续fuser gate/alpha或跨层变换，不叫下游因果贡献。主结局dual−single GT共同框IoU差及dual绝对错误1−IoU；两模型不同权重，不是隔离UAV分支的反事实。需固定区域、支持分母、层/尺度策略，不后验选图/阈值/关联。六图只有6个场景，不以像素或重复RoI制造独立显著性；大小、边界截断、coverage/quality可能混杂，相关不等于污染或因果。

VERDICT：PENDING。协议与实现待定版；目前仅完成定义调查与协调，无新结果。

主代理随后统一冻结口径：主Y=`singleIoU−dualIoU`，正为dual退化；`dual−single`若保留只作明确反号副字段。主stride16/32²，stride8/64²伴随，stride1/512²仅描述；不做六图小样本多参数混杂拟合。此口径覆盖上述初始协调的delta方向，正式审核按本段执行。

## 映射实施静态审核

VERDICT：PASS。允许runner执行预注册36份CPU几何标量映射，不运行网络/GPU。统计关联实现另待审核。

已核对`/tmp/ue_bev_tail_map.py`和执行前完整协议。六图×两层×32/64/512，共36份map。source来源、fine pixel整数分桶、massquality权重、D分母clamp1e−6、rawmass/coverage/quality与生产一致。独立FP64 bincount与直接CPU生产BEVToSatellite FP32逐数组对账门预先固定，未新增参数或模型forward。source字段全有限；无候选dense占位0仅用于生产池化接口，实际所有project_source均为height_valid&hascandidate，不能消费占位。attention/cache/ref/flags/projected身份与SHA严格对账、缓存前后字节不变。

执行前发现并关闭一处边界问题：D=0但rawmass>0、quality全0时，metric和meanquality应0而coverage可正，初版断言missing coverage=0不普适。runner与审核者独立发现后改为仅rawmass=0要求coverage0，协议同步保留D0正coverage语义。此修正发生在执行前，未改映射定义/数据/数值门。归一权重和按D/max(D,1e−6)核对，未将极小D误称sumweight1。

主X为actual absdz−uniform absdz及1−heightmass；先source自身指标，再生产权重映射，不能把mapped offsetnorm当投影后signed vector的norm。只是对源几何指标做空间暴露描述，未含learned fusion gate/alpha、FFN或decoder，不能称物理错位或真实输出贡献。无支持map指标NaN，保留独立missing/coverage，不让0暴露混入关联。

统计协议静态边界合理：固定框交叠面积权重，主Y为single−dual，每图相关后图等权，不p值/CI/性能门；size、clip、support、边界比例并列作混杂描述，不六图多参数拟合。需待关联代码确认是否实际按此实现。

## 关联实现静态审核

VERDICT：PASS。允许evaluator在mapper完成后执行原预注册六图CPU统计。未授权增加样本、拟合、模型forward或干预。

已阅读全文`/tmp/ue_bev_tail_association.py`。每GT slot对齐实际GT bbox、共同框SHA、历史身份、dual/single RLE和IoU逐值；binary RLE直接解码不伪造概率。bbox与fine像素区间精确交叠权支持fractional box，按target网格reshape求各格实际交叠面积；map指标NaN自然排除D0，covered area按D>0定义，0支持X=None。GT内部伴随独立，不与bbox主平均混用。

主Y single−dual、absolute error 1−dual；主X excessabsdz与incompatiblemass，图内Pearson/平均ties Spearman后六图等权汇总，无p/CI/性能门。n<3或常数相关为None且实际n保留；两层32²主/64²伴随全报告，pooled ROI/globalid内部平均再相关仅描述。region三部分按固定7×7膨胀/腐蚀边界、内部、非边界背景，GT共同框内covered与uncovered分列，像素不当独立推断。

分母说明：mean_projection_coverage是整个bbox交叠面积均值，含D0可能存在的raw coverage；mean_projection_quality只D>0区域均值。building汇聚X若部分出现无支持，则其有效出现集合可少于Y，需只作描述敏感性分析并保留corr有效n，不能说独立建筑因果。源文件/GT/RLE/maps SHA均保存或通过上游映射manifest继承钉住。正式结果另待独立数值抽查。

## 首次CPU映射停止及数值诊断

原预注册FP64 vs production FP32的allclose门在1218/layer0/32²首map失败，runner即时停止，尚无正式map或关联结果；不能将此前静态PASS写成原数值门通过。原脚本/协议/失败log已保留first_attempt/first_failure。

审核者独立CPU重建实际生产投影，确认仅2项失败，均为weighted_abs_dz_m正值累计而非signed抵消：target572，FP64 11.9333205730254，FP32 11.9333648681641，差4.42951386646e−5；target740，FP64 14.3566079954321，FP32 14.3565397262573，差6.82691748022e−5。对应220/256个entry，D约192.1413/230.8083。整数归桶/归一化公式未见错误。

已读runner进一步诊断：独立numpy原稳定序逐项FP32 numerator/denominator累计，对上述两个production target bitexact；主要是FP32顺序归约舍入（分母累积差−.0003473/+.0006444，分子+.0044655/−.0065377），不是模型/映射bug。审核建议采用root允许的显式协议v2：保留原门FAIL，全部36map改为独立稳定分组FP32逐项累计对production逐数组核验，FP64差只描述记录，不经验放宽常数。当前仍等待root确认修订及完整参考实现复审，不能继续关联或声称结果PASS。

## v2执行前静态复审

VERDICT：PASS，可按root明确授权继续唯一36map CPU映射。root已确认简洁v2，无gamma界或经验放宽allclose。当前脚本独立numpy np.add.at无缓冲依原entry序FP32累计，与production target stable sort组内顺序一致；weight与numerator FP32乘法、den/rawmass/numerator累计、clamp除法、quality/coverage全部逐数组np.array_equal对账。旧FP64门pass/fail/超门数量保留为描述，全部FP64地图仍为预定关联主输入；固定productionFP32两个主X伴随字段不后选。协议显式保留首FAIL及原脚本/协议/log，并记录修订在完整映射前发生；样本/几何/关联终点不变。实际结果仍待审核。


## 最终状态：独立结果审核待完成

三位子代理因服务额度限制退出。此前PASS仅覆盖协议、mapper v2与关联原脚本静态审核，不覆盖正式36map或最终关联结果。主代理接手：stat bool数组分位数触发NumPy TypeError，已保留原脚本与失败log，将统计数组显式dtype=float后执行完成；未改变终点/集合/方向/统计定义。主代理以另脚本复算框面积重叠暴露及scipy pearsonr/spearmanr相关，通过<1e−12，并核FP32伴随；这些是主代理复算，不是独立结果审核。最终VERDICT：PENDING（独立结果审核未完成）。不得用于论文机制结论或训练准入。
