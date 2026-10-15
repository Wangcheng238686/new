# 冻结UAV表征的粗形状残差线性读出诊断

2026-10-15；关联EXP-0006/0007。独立实施/smoke/正式产物审核及主代理五组重拟合均PASS；预注册工程支持门FAIL。

## 问题与身份

本轮在固定E72 EMA和数据清单下，检查已对齐BEV特征能否帮助预测卫星粗形状错误。卫星侧取双流训练过的backbone/neck在UAV融合前的特征，经原PE、RoI和粗头；不是独立单流E66。训练侧固定hash抽128图，仅拟合71变量的岭回归，网络没有更新。153验证图只评估，GT框为共同条件。

## 主要判据

8×8 RoI occupancy Brier为主；对照SG保留卫星、几何和原粗概率，SU增加对齐UAV，SX增加错位UAV。SU对SG、SX须均降低Brier至少0.001且两主比较Bonferroni区间下界大于0。S64额外卫星32维、SN固定噪声32维为容量描述对照。全部及支持非零组都报告；二值8²IoU是伴随描述，非最终mask或AP。

## 不变量

六图smoke的240GT raw coarse logits与原zeroUAV head精确一致；source neck/BEV、卫星输入和UAV payload/RGB未被修改，hook已移除，模型parameter versions不变。实际RoI aligned=True、sampling_ratio=0、14²输出，BEV和投影均256×128²。所有NPZ身份、shape、finite、hash通过。正式281图只提取一次，不运行SAM、RPN或bbox。

## 结果

唯一281图提取exit0，357.76秒，17117个RoI：train12621，val4496。固定5cell CPU拟合一次，线程2，无调参；训练807744网格行（不是807744独立建筑）。SG活跃输入39维，其余71维，所有系数形状71+截距；S64/SU/SX均64维视觉输入，SN额外32维噪声。

| cell | 训练Brier | 验证Brier↓ | 验证二值8²IoU↑ | 验证建筑等权Brier↓ |
|---|---:|---:|---:|---:|
| 未修正卫星粗概率 | .01921017 | .03809660 | .87424205 | — |
| SG：卫星32＋几何 | .01816086 | .03537434 | .88078572 | .04085614 |
| S64：卫星64＋几何 | .01808286 | .03535637 | .88098433 | .04082998 |
| SU：卫星32＋对齐UAV32＋几何 | .01810302 | .03501472 | .88163556 | .04050438 |
| SX：卫星32＋错位UAV32＋几何 | .01812449 | .03520150 | .88109014 | .04064158 |
| SN：卫星32＋噪声32＋几何 | .01815998 | .03537466 | .88074017 | .04085727 |

两主比较（controlBrier−SU为正）：

- SU相对SG：+.000359613，Bonferroni两主比较区间[.000308873,.000418876]；建筑等权+.000351759；二值8²IoU差+.000849840。
- SU相对SX：+.000186775，区间[.000144564,.000234352]；建筑等权+.000137203；二值8²IoU差+.000545412。

虽然两个区间都高于0，均未达到预注册绝对Brier改善.001门，两个主门均FAIL、总体FAIL，不进入模型训练。S64/SN描述上也略差于SU，但不将这个小幅条件差值包装成正式模型增益。相对SG的Brier减少约1.02%，相对SX约0.53%，效应小。

支持非零组4312RoI：SU−SG Brier改善+.000371687、SU−SX+.000206914，依然低于同一工程幅度；不能事后换组过门。其余184个是UAV投影RoI采样覆盖为零，**不同于前轮197个SAM dense框支持为空目标**。

尺寸描述（不设组内门、不挑最好组）：small881：SU−SG+.000162214、SU−SX−.000074083；medium2510：+.000240685/+.000145177；large1105：+.000787139/+.000489243。小目标相对S64 Brier变化亦为负（改善量-0.000033961），并非统一受益，不能从全体微弱均值推断小目标细节已经改善。

## 数据隔离与空间身份

manifest源SHA495841b2174242c87661e6b40dba2ddcca13d64b13798116efc8c933c679ca59，完整train627/val153的已知buildingID2523/237、交集0。此次选中128train的实际可见建筑2239，val237，实际交集也0。选中训练图与验证图manifest `footprint_xy_m` AABB重叠对0；每张验证图到最近选中训练原点最小912.5m，中位1352.14m。此处仅审manifest footprint AABB与原点，未重新审计producer字段具体视野定义，不将其径称整张卫星图AABB，不宣称所有UAV可见对象绝无交叉。验证237栋建筑在4496次实例出现中重复，building等权与imagepaired区间均已报告，但空间和重复相关不被普通图像bootstrap完整覆盖。

## 验收与工件

features/run_manifest.json complete=True，281图身份/来源/矩阵/所有NPZSHA与shape/finite/gt_slot唯一、真实可见建筑隔离、train-onlyfit和参数version检查均通过。来源包括checkpoint streaming SHA、脚本/base helper/protocol SHA、architecture与历史验证、RoI对齐配置；CPU汇总先检查再拟合。模型只有5×71个系数加截距，ridge_models.json保存训练均值/标准差/系数。val_predictions.npz保存五cell与基线/GT8、image/globalID供独立重算；summary.json保存全/支持/尺寸/建筑等权/主CI与门。脚本/tmp/ue_complementarity_probe.py、/tmp/ue_complementarity_summary.py，完整运行日志full.log、summary.log；root已从全部cache独立重新拟合五cell标准化/系数，复算Brier/IoU/建筑等权和两主CI，结果差<1e−12、beta差<1e−10，结论审核PASS；全文由root另行归档。

## 当前决定

存在微弱可读出的条件互补信号，超过几何、噪声和本轮卫星额外维度描述对照，但工程支持门未通过。现有证据不支持把后续大训练资源押在这个线性RoI补丁上。其余两个并行诊断（dense编码因素、训练梯度结构）可以提供不同证据，再共同确定下一步；不把本轮负门解释为UAV无价值，也不拿微弱阳性区间解释为设计成功。

## 解释边界

阳性只说明小容量读出能够访问冻结表征中的条件互补信号，不证明原融合器能用好、不证明UAV净增益、finalmask/AP或完整生产数据可推广。阴性只否定本轮投影维数、8²读出、岭回归形式、128训练图预算的支持门，不否定UAV信息。train/val建筑ID与可见ID、空间邻接、建筑等权须随完整结果报告；E72已用此验证集选优，独立测试和训练seed不确定性未覆盖。

错位对照是在投影后的128²UAV特征上循环平移32cell，几何图保持原位，会制造特征与支持不一致，不能视为真实采集错位或最优标定测试。8²occupancy是软标签粗形状诊断，无法观察细边界；固定单次随机投影与线性读出也可能漏掉非线性或高维互补。

## 归档索引

[预注册](probes/results_20261015/UAV_COMPLEMENTARITY_PREREGISTRATION.md)、[结果JSON](probes/results_20261015/uav_complementarity_summary.json)、[来源manifest](probes/results_20261015/uav_complementarity_run_manifest.json)、[字段语义](probes/results_20261015/UAV_COMPLEMENTARITY_FIELD_SEMANTICS.md)、[缓存schema](probes/results_20261015/uav_complementarity_schema.json)、[读出系数（原始工件）](/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis/uav_complementarity/ridge_models.json)、[独立审核与主代理复核](../reviews/UE_UAV_COMPLEMENTARITY_20261015.md)。

原始目录`/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis/uav_complementarity/`，包含约610MB feature NPZ、smoke/full/summary日志与val_predictions；缓存和完整预测不进git。精确脚本快照为`scripts/analysis/ue_complementarity_probe.py`与`ue_complementarity_summary.py`，保留原/tmp/helper路径以匹配SHA，非通用生产入口。无生产遥测修改。
