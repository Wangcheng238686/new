# GT形状提示冻结条件诊断

日期2026-10-15；关联EXP-0006/0007。当前冻结开发子集153验证图、4496次出现、237栋建筑；不是完整生产集或独立test。

## 协议

用户两段描述是同一核心对照，两个子代理分别实施GPU干预运行器与CPU独立评估/运行器审核，共用唯一两组GPU推理；主代理审查协议、汇总程序和结论并直接复算关键数字。

baseline为E72 strict EMA、正确UAV与原GT共同框；joint_gt_prompt在shape_injector.mask_decoder输出处，以真实get_coarse_targets生成的GT局部64²二值形状映射成固定±4 logits替换预测粗形状，再由原流程联合生成点与dense提示。SAM图像embedding/high_res/PE、框、RoI、参数、点规则、dense alpha=.32814547和32²框支持规则保持不变。没有训练、反传、optimizer、RPN/bbox/AP计算或生产模型/遥测修改。

[预注册](probes/results_20261015/GT_PROMPT_ORACLE_PREREGISTRATION.md)、[轻量汇总](probes/results_20261015/gt_prompt_oracle_summary.json)、[运行manifest](probes/results_20261015/gt_prompt_oracle_run_manifest.json)、[逐图不变量](probes/results_20261015/gt_prompt_oracle_invariance_checks.json)、[完整审核](../reviews/UE_GT_PROMPT_ORACLE_20261015.md)。

预注册主比较为全部4496个共同GT框最终paste mask IoU的joint_gt_prompt−baseline，1000次image-paired sum/count bootstrap seed1760；探索性提示改善空间门为均值>=.005且95%区间下界>0。全部尺寸、官方RoI层级、dense支持为空/非空分组与237建筑等权均报告，不后选。

## 验收

三图287/288/295的45GT：observer开/关最终mask逐像素相同；原coarse输出clone的sham在全部stage与最终mask上精确相同。GT替换hook先于observer注册，真实GT输出进入原point/dense路径。正式只跑baseline和oracle，不重复sham/off；正式对应字段为null，表示未重复，不是失败。

## 完整153图结果与落判

| 对照 | 共同GT框最终mask IoU |
|---|---:|
| 原预测形状提示 | 0.841768525 |
| GT64联合形状提示 | 0.860831402 |

配对差 **+0.019062877**（约+1.9063个IoU百分点），1000次配对图像sum/count bootstrap 95%区间 **[+0.017034467,+0.021300709]**。均值>=.005且区间下界>0，预注册提示改善空间门 **PASS**。不是AP提升。

237栋建筑内部平均再等权，差+0.020008996。4496次出现中2624次改善>.002，1335次退化<−.002，其余537次变化介于两者之间，平均改善不代表所有目标改善。

| 分组 | n | 最终IoU差 |
|---|---:|---:|
| 小目标 | 881 | +0.013708274 |
| 中目标 | 2510 | +0.015556400 |
| 大目标 | 1105 | +0.031296961 |
| 直接读取P2 | 3657 | +0.015899706 |
| 非P2（本集P3） | 839 | +0.032850381 |
| dense框支持为空 | 197 | +0.009563410 |
| dense框支持非空 | 4299 | +0.019498186 |

上述分组均为预定描述，不用来后选主结论；尺寸分组按GT框面积，非UAV覆盖分组。

## 机制伴随与完整验收

- 正式耗时252.17秒，GPU执行session78444，exit0；仅一次153图两组推理。独立实施/smoke/正式审核均PASS，主代理关键数字直接重算也PASS。
- 153图全部框/身份/顺序与历史基线一致，基线IoU、coarse/BCE、points/tokens/support逐项精确复现。参数version、特征及RGB/几何hash不变；GT替换发生前的模型原始粗logits仍与基线精确相同，全部RoI固定。
- 42图实际覆盖多chunk，共197个chunk；逐图生产32切片、target ranges/GT slots精确一致。
- GT64空目标0、全前景54。oracle coarse IoU全1来自注入构造。每个目标canvas/PE dense均发生数值变化，4299个实际dense变化，4494个点提示变化，4386个最终二值mask变化。
- baseline P1/P2有效点准确率99.40%/99.33%，N1/N2为65.26%/89.80%；GT替换后四槽有效点准确率均100%，有效负点数量从4479/4460降至4442/4396。正确率只针对GT64采样、只以各槽有效数为分母；全前景目标局部本无负点。
- 197例dense框支持为空的目标，两组dense均为原no-mask基底，仍有195例点提示改变和130例最终mask改变。该组平均IoU改善+.00956341；不能据此推断修正dense支持规则将有多大收益。
- 原预测logits各RoI的min/p05/p50/p95/max再平均为[−11.0282,−7.7717,3.54488,9.71622,11.30155]，RoI内std再平均6.02968；±4替换确实改变了置信分布。这里只做描述，没有据此调整幅度。

## 对下一步设计的判断

**当前SAM仍能利用这套更准确形状来源的提示，不能把P2弱响应解释成后续提示路径没有价值。** 结合上一轮“P2扰动已到达decoder但粗质量未提高”，更应先定位如何生成有用提示。

下一候选是固定权重下拆分GT点/原dense和原点/GTdense，与本轮两组构成四组合，判断该改善主要通过哪个提示通道。N1原有效点中约34.74%落在GT64前景，是值得检查的线索，但本轮联合替换无法证明它就是瓶颈。

尚不确定：原有UAV/BEV特征是否足以生成这些提示；改P2融合、解除dense detach或改变辅助监督能否学到有效改善。没有这方面训练证据，本轮不扩全FPN、不启动训练，也不自动追加新GPU探针。


## 解释边界

- 此干预是冻结权重、GT共同框下的条件诊断，不是完整检测AP，不是UAV净收益。GT64的形状IoU为1仅是注入构造校验，不能称模型学得更好的粗分割。
- 形状与有限logit置信度同时改变；GT crop/nearest的64²形状并非完整512真值，点挖掘、画布采样、冻结PE、原alpha/框支持与提示分布迁移均影响响应。因此oracle不是所有提示实现的数学上限。
- 联合替换不分别隔离点与dense因果，也没有修复small目标支持；GT框下的support0比例不能直接代表正式预测框。
- 正结果只能说明当前decoder在此固定条件下能够利用GT形状来源的提示；不能证明UAV能生成这些提示、取消detach就会改善，或增加P2/全FPN必然有效。负结果也不能否定所有更好的提示。
- 配对图像区间不涵盖相邻图/建筑重复的空间相关、验证选模与训练seed不确定性；建筑等权仅补充描述。

## 工件与复现

原始目录：`/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis/gt_prompt_oracle/`，日志`smoke.log`/`full.log`及两组逐GT标量/点记录位于其中。没有保存大中间特征。

脚本精确副本归档于`scripts/analysis/ue_gt_prompt_oracle_probe.py`与`ue_gt_prompt_oracle_summary.py`；为保留来源SHA，维持运行时固定路径，包括/tmp下ue_rgb_content_probe和ue_p2_prompt_trace依赖。它们是本次实验快照，不是通用生产入口。代码/依赖/数据/预注册/检查点SHA见manifest。没有新增生产遥测字段，DEBUG_FIELDS定义不变。
