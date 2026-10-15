# Dense提示的来源形状与幅度拆分

2026-10-15；EXP-0006/0007。当前开发子集153图、4496次出现、237栋建筑；非完整生产集/独立test。三组冻结E72、correct UAV、GT共同框，实际点/sparse全固定原P（含固定box token），仅dense来源改变。无训练、反传、optimizer、RPN或AP。

| cell | dense来源 | 最终GT框mask IoU |
|---|---|---:|
| baseline | 原粗logits | 0.841768525 |
| pred_binary_pm4 | 原预测64²二值形状，固定±4 | 0.841319969 |
| gt_binary_pm4 | GT64²二值形状，同固定±4 | 0.857387591 |

## 预注册主比较与落判

1000次配对图像sum/count bootstrap seed1780，两主各97.5%边际区间（Bonferroni .0125/.9875），每门要求均值>=.005且下界>0。

| 对比 | 均值差 | 97.5%区间 | 门 |
|---|---:|---|---|
| 预测±4−原logits | -0.000448556 | [-0.000923192,+0.000023090] | FAIL |
| GT±4−预测±4 | +0.016067622 | [+0.013620570,+0.018728376] | PASS |

**本次固定二值±4变换没有带来总体改善；同幅度下GT来源形状仍明显更好。** 说明不能把原GTdense条件收益主要解释为这一次幅度/量化调整本身。不能推广成所有置信校准都无效。

237建筑等权：两主差分别−.000269268、+.015462342。全部预定尺寸、P2、support分组如下；不用于后选主结果。

| 组 | n | 预测±4−原logits | GT±4−预测±4 |
|---|---:|---:|---:|
| small | 881 | +0.000707606 | +0.005387567 |
| medium | 2510 | -0.000225859 | +0.012484295 |
| large | 1105 | -0.001876200 | +0.032722175 |
| P2 | 3657 | -0.000126514 | +0.011997191 |
| nonP2 | 839 | -0.001852260 | +0.033809655 |
| dense_support_zero | 197 | +0.000000000 | +0.000000000 |
| dense_support_nonzero | 4299 | -0.000469111 | +0.016803915 |

## 隔离与验收

- 两历史端点逐目标IoU和sparse/dense SHA精确复现上轮PP/PG；全三组原sparse逐值固定，且decoder embedding/PE/high/flags核定未变。
- 153图全部验收通过，42图多chunk，总197chunk，实际全6token；未声称测试到2token。197support0实例三组dense基底、最终mask逐像素及IoU均不变。
- 三图smoke的各组实际输入sham prompts和mask均精确；正式不重复sham，null表示未重复。模型版本/特征/输入hash与hook清理通过。
- 正式session95861，exit0，246.30秒；独立静态/smoke/原始产物审与主代理数字重算PASS。诊断耗时不是纯推理benchmark。

## 解释边界与下一步

预测±4严格保持源64² logit0二值形状，但会去除形状内部置信梯度；双线性paste后下游sign边界也可能变化。因此它不是纯温度校准/纯置信度因果效应，GT±4−预测±4也只是相同有限幅度下形状来源的条件差。原点与GTdense仍可能不一致。

GT使用真实get_coarse_targets的crop/nearest64，GT coarseIoU=1只是注入构造，非学得提升。没有测UAV是否能生成该形状、最终AP、detach训练收益或完整数据集泛化。图像区间不涵盖空间/选模/seed方差。

下一步选择须结合UAV互补表征诊断；不再盲扫幅度或自动增加FPN层，也不自动进入训练。工程梯度核定见[梯度与设计报告](UE_GRADIENT_DESIGN_AUDIT_20261015.md)。

## 来源

[预注册](probes/results_20261015/DENSE_SHAPE_CONFIDENCE_PREREGISTRATION.md)、[方法补充](probes/results_20261015/DENSE_SHAPE_CONFIDENCE_METHOD_NOTES.md)、[结果JSON](probes/results_20261015/dense_shape_confidence_summary.json)、[manifest](probes/results_20261015/dense_shape_confidence_run_manifest.json)、[独立审核与主代理复核](../reviews/UE_DENSE_SHAPE_CONFIDENCE_20261015.md)。

原始工件`/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis/dense_shape_confidence/`，smoke.log/full.log/summary.log及逐GT结果；脚本精确副本`scripts/analysis/ue_dense_shape_confidence_probe.py`、`ue_dense_shape_confidence_summary.py`，保留运行时/tmp/helper路径以匹配SHA，非通用生产入口。无新增生产遥测。
