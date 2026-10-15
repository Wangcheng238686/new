# 冻结GT点与密集提示交换：通道定位

日期2026-10-15；EXP-0006/0007。范围仍为当前冻结开发子集153验证图、4496次建筑出现、237栋建筑，非完整生产集或独立test。

## 对照设计

承接GT联合提示条件改善，用户授权追加点only与denseonly冻结对照。两个子代理分别实现GPU交换运行器、CPU评估与独立运行器审核；主代理审代码/协议、直接复算主数字和结论。

固定E72 strict EMA、正确UAV、GT共同框、FPN/SAM/PE/high特征、所有参数和后处理；原模型和GT64±4来源各执行一次，捕获decoder实际输入，而后交换两种提示。GT来源沿用原点挖掘和dense转换/gate/support，不修复小框支持，不调幅度。

| 组 | 实际点/sparse来源 | 实际dense来源 |
|---|---|---|
| baseline（PP） | 原预测P | 原预测P |
| gt_point_only（GP） | GT来源G | 原预测P |
| gt_dense_only（PG） | 原预测P | GT来源G |
| joint_gt_prompt（GG） | GT来源G | GT来源G |

sparse包含固定box token，P/G最后两个box token逐值相同；仅点token及其有效性/实际序列结构可能不同。直接交换真实2或6token的sparse和实际gate/support后dense，不把补0统计副本喂模型。交叉两组的上游coarse仍为P，点诊断按实际sparse来源记录。

两对角live运行用于取得源提示，并逐GT精确复现上轮GT提示探针；这四组是本轮唯一153图GPU运行。没有训练、反传、optimizer、RPN/bbox/AP计算，没有生产结构/遥测改动。

[预注册](probes/results_20261015/GT_PROMPT_FACTORIAL_PREREGISTRATION.md)、[轻量结果](probes/results_20261015/gt_prompt_factorial_summary.json)、[manifest](probes/results_20261015/gt_prompt_factorial_run_manifest.json)、[逐图不变量](probes/results_20261015/gt_prompt_factorial_invariance_checks.json)、[审核全文](../reviews/UE_GT_PROMPT_FACTORIAL_20261015.md)。

## 预注册判定与验收

两个主比较为GP−PP与PG−PP，全部4496共同GT框最终paste mask IoU。1000次image-paired sum/count bootstrap seed1770，两主比较使用Bonferroni校正：各97.5%边际区间（分位数.0125/.9875），目标家族覆盖95%；这是有限重采样的探索性区间。每个主比较支持门为均值>=.005且该区间下界>0。

GG−PP、交互GG−GP−PG+PP、GTdense下点作用GG−PG、GTpoint下dense作用GG−GP均作为预定描述，给95%区间而不用于后选主结果。交互正负计数只是对比值方向，不是实例mask改善计数。尺寸/P2层级/support0分组、237建筑等权全部报告。

三图287/288/295共45GT：双对角的实际提示原值交换sham最终mask逐像素相同；两个support0目标仅换dense时最终mask逐像素等于baseline；source/actual哈希、框token、视觉输入/kwargs、源提示metadata和参数不变检查通过。smoke只有6token、单chunk图，未伪称覆盖所有形状；正式继续强制chunk/shape/source检查。正式不重复sham，对应字段null表示未重复。

## 完整结果与预注册落判

| 对照 | 最终GT框mask IoU | 相对baseline |
|---|---:|---:|
| 原点＋原dense | 0.841768525 | +0.000000000 |
| GT点＋原dense | 0.846308591 | +0.004540066 |
| 原点＋GTdense | 0.857387591 | +0.015619066 |
| GT点＋GTdense | 0.860831402 | +0.019062877 |

| 主比较 | 均值差 | 两项校正后各97.5%区间 | 幅度/区间门 |
|---|---:|---|---|
| 仅换GT点 | +0.004540066 | [+0.003702401,+0.005494398] | FAIL |
| 仅换GTdense | +0.015619066 | [+0.013482854,+0.018156355] | PASS |

**点only有正信号，但均值+.004540未达到预定.005幅度门；不能解释为点提示无效。denseonly+.015619通过门。** 两主区间均在0以上，预注册门同时要求效果幅度和区间，未后验放宽阈值。

建筑等权：点only差+0.005671542，denseonly差+0.015193074。不以建筑等权替换原实例等权主指标。

### 全部预定分组（描述）

| 分组 | n | 仅GT点−baseline | 仅GTdense−baseline |
|---|---:|---:|---:|
| 小目标 | 881 | +0.009407966 | +0.006095172 |
| 中目标 | 2510 | +0.004459163 | +0.012258436 |
| 大目标 | 1105 | +0.000842733 | +0.030845975 |
| 直接读取P2 | 3657 | +0.005501950 | +0.011870678 |
| 非P2（本集P3） | 839 | +0.000347448 | +0.031957396 |
| dense框支持为空 | 197 | +0.009563410 | +0.000000000 |
| dense框支持非空 | 4299 | +0.004309874 | +0.016334804 |

总体dense效应较大；小目标中点only+.009408，高于本组denseonly+.006095。这是预定分组描述，不能后选小目标宣布整体点门通过，也不证明两个通道的普遍价值排序。

### 条件差与交互（次要描述）

| 对比 | 均值差 | 描述性95%区间 |
|---|---:|---|
| 联合−baseline | +0.019062877 | [+0.017102815,+0.021546482] |
| 交互 GG−GP−PG+PP | -0.001096256 | [-0.001588066,-0.000593900] |
| GTdense条件下加GT点 GG−PG | +0.003443810 | [+0.002672024,+0.004374262] |
| GT点条件下加GTdense GG−GP | +0.014522811 | [+0.012660033,+0.016658445] |

联合效果比两者单独改善之和低约.001096，不能用单通道改善相加来分配联合收益。dense在原点/GT点两种条件下均有约.0145–.0156的平均条件改善；GT点在原dense/GTdense下约.0034–.0045。交互可能包含提示一致性和非线性，不能据此宣称架构冲突或训练协同。

## 完整验收与资源口径

153图/4496出现均完成，42图实际多chunk，总197个chunk，四组实际均为6token（没有实际2token样本，不能宣称验证2token部署行为）。两live对角逐GT精确重放上轮。实际session89849，exit0，耗时2205.27秒（约36.75分钟），显著超过首批估计；含大量逐chunk提示/视觉张量诊断校验，未profile，不将耗时定量归因某一步或作为纯模型推理性能基准。

- 全部source/actual提示SHA、固定box token、视觉args/kwargs、特征与RGB/几何hash、参数version、source prompt不变和hook清理检查通过。
- 197例support0：仅换GTdense最终mask逐像素等baseline，逐GT IoU精确不变；GT点only与联合组的actual sparse/dense SHA及最终IoU也逐GT相同。该组点only平均改善+.009563，dense仍无注入。
- 原点源与GT点源的valid/correct/坐标/标签/token数量均精确复现上轮，不将交叉组上游P点误作实际GT点。原GT点源有效负点减少的口径保留；正确率只按有效槽。
- 相对baseline，点only有4397个最终mask变化、denseonly4231个、联合4386个。变化数量本身不是质量结论。
- 独立实施、smoke、正式产物审核均PASS；主代理直接复算六contrast与各分组也PASS。所有检查在同一既定运行及其CPU工件上完成，没有追加GPU测试。

## 对模型设计的判断

**当前冻结构造下，总体更大的条件改善空间位于粗形状经dense提示进入SAM的通道。** 提示链并非完全无法有效传递信息；上轮P2扰动没有提高粗质量，也不能单纯归因链路长。

第一负点GT64一致率约65.26%仍是线索，但点only总体改善较小，不足以据此将点挖掘认定为首要总体瓶颈。小目标的点提示作用应保留，不主张删除点分支。

下一步优先低成本拆分dense来源的形状与置信度：在原预测二值形状上同样使用固定±4，再与原预测logits及GT形状±4比较，才能判断本轮改善有多少来自形状、多少可能来自置信/分布。此为后续建议，本轮未运行。

最终仍需核定UAV对齐特征是否能帮助预测/修正卫星粗形状误差；GT来源提示改善不是UAV可学习性的证据。解除dense detach、提高粗loss或新训P2均尚无因果训练证据，不据此自动训练或扩全FPN。


## 解释边界

- 这是固定GT框与冻结权重下的提示通道干预，不是可部署GT方法、完整AP或UAV净收益。
- GT64±4来源同时改变形状、置信度和提示分布。即使一通道较强，也不能推出所有该类提示设计都更优，或另一路没有价值。
- 稀疏提示包括坐标、标签有效性和真实token布局，效应不能缩成纯点坐标精度；box token保持原值。
- dense支持为空属于32²框支持规则，与UAV几何coverage不同。此组仅换dense的pixel不变是阴性验收，不验证修正support的收益。
- 交叉组可能出现P点与G dense或G点与P dense语义矛盾。两个主效应以另一通道维持P为条件，不是通道的内在价值排序；交互可能包含提示一致性和非线性响应。
- 交互是固定四组平均IoU的差中差，不能作为可加模块贡献、融合架构协同或训练净收益。
- 不从此推断解除detach、改辅助loss或扩P2/全FPN必然有效。图像区间不覆盖空间相关、验证选模或训练seed方差。

## 复现与归档

原始工件：`/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis/gt_prompt_factorial/`，日志`smoke.log`/`full.log`，四组逐GT结果和manifest在smoke/full子目录。源提示中间张量按图释放，不保存大张量。

脚本精确副本为`scripts/analysis/ue_gt_prompt_factorial_probe.py`、`ue_gt_prompt_factorial_summary.py`；保留运行固定路径和/tmp中ue_rgb_content_probe/ue_gt_prompt_oracle_probe及trace helper依赖以匹配来源SHA，不是通用生产入口。没有新增生产遥测，DEBUG_FIELDS定义不变。
