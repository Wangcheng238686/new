# 4090 N系列核查与本机补充方向（2026-10-23）

## 读取来源与状态

通过SSH只读核查4090实际运行目录 `/data2/wangcheng/remote-lthpc/`，不是应用里仍停留在10月13日的旧SurfaceBEV会话。运行包为其下的 `portable_sam2_explicit_coarse`，不是旧 `dual-stream-dev` 工作树。

本机证据快照：`/data/wangcheng/checkpoint/ue_remote_n_review_20261023/`。
其中 `snapshot.json` 保存六臂冻结合同摘要与最新训练行；两个 `.py` 是读取时的远端源码副本。远端仍在工作，后续结果不由本快照代表。

| 方向 | correct最新训练行 | constant最新训练行 | 完整AP对照 |
|---|---|---|---|
| N1 | E7 / update1645 | E7 / update1655 | 尚无 |
| N2 | E5 / update1301 | E5 / update1316 | 尚无 |
| N3 | E5 / update1331 | E5 / update1326 | 尚无 |

progress.json只在部分边界更新，PID还指向原进程；以上采用train.jsonl最新行，并与实际训练命令交叉检查。不能以训练loss、alpha或新增参数直接推断AP增益。

六臂基座文件SHA均为 `dd9106f4afe8ba72936316d7efa9621ca0c05d24836fa28c72ea322bed84ace1`，即本机EXP-0017。远端N1合同为10epoch、AdamW LR1e-4、accum8、seed44、reseed_from_epoch=0。本机L系列逐epoch重设1044000+epoch，因此不可把跨机器两个训练结果直接当作仅改一个机制的严格对照。

## 实现实际检验的问题

### N1：聚合特征的前景与边界辅助监督

在attention聚合后、output投影前的32维pooled特征上接两层3×3 Conv/GN/ReLU与2通道头，预测卫星实例掩码的前景并集及其边界。训练RoI网格采样标签：并集经3×3结构元素膨胀/腐蚀各3次，二者差为边界带。loss为 `.05 * [(BCE+Dice)_union + 2*(BCE+Dice)_boundary]`。

这个辅助梯度绕过alpha与冻结decoder，但仍通过attention的卫星Q及几何K/V；源码没有分离卫星Q梯度，不能称为只由UAV外观驱动。边界是并集边界，会丢失接触实例之间的内部边，不是逐实例边界监督。

本机D_aux也对聚合证据做前景辅助监督，但采用128²全局网格、detach Q（含offset输入）、BCE及系数.1。两者部分重叠、非同一实验；远端N1同时改变标签/损失/网格/梯度路径，不能把差异全部归因于边界标签。

### N2：将注入移到最后GELU之后

旧入口：`conv_s0` 输出上加残差，随后进入 `GELU(dc2(upscaled) + feat_s0)`。
新入口：hook `mask_decoder.decoder.output_upscaling[-1]`，在该最后GELU的输出上加同维32通道残差。

已核对本机实际SAM2源码：启用high-res时Sequential整体虽不直接调用，解包出的act2子模块仍被调用，因此hook有效。后面立即是mask token生成的hypernetwork向量与高分辨率特征的矩阵乘法。到这一层原始mask logits的路径对新增特征是线性的；后续完整掩码流程及UDPR不能因此称为线性。

它检验相同证据是否在现有注入位置被非线性变换削弱或改写；不预设GELU已造成失败。没有新增预测头，也没有替换UDPR。该方向与本机L6开放消费者LoRA不同，也与已有early注入不同。

### N3：可学习attention温度

将 `4*cos(Q,K)` 改为 `exp(log_tau)*cos(Q,K)`，log_tau初始化ln4，首步与固定4一致。仅增加一个参数，检验候选权重是否需要更集中或更平滑；不修正几何坐标、不增加候选与多头。

远端实现没有额外温度范围限制，且当前通用latest_forward没有tau字段。若本机补做，应在离线检查点审计中读取tau和候选attention熵；不为观察而给每步训练追加大量同步统计。

## 本机建议：优先一个新机制，避免再铺完整三组

1. **优先N2式后激活注入**，同一全新归一化attention基准下correct/constant成对训练；其他源特征、候选、投影、LoRA、损失、支持区域、训练顺序和seed均固定。与L0正确/常量两臂形成2×2设计；L0身份/初始化/顺序严格相同时才可复用，不能复用仅名字相近的历史A或远端b0。
2. **N3后置**：远端已有完整成对训练正在进行。本机没有证据支持现在重复消耗两臂训练；若远端内容差优于基准，优先做独立种子复核，而非同seed搬运一次。
3. **N1暂不复制**：先等待本机D_aux与远端N1完成。若需区分边界监督，后续在同一个辅助头/网格/梯度路径上只改变边界项权重0→固定非零；不能把两个辅助实现的整体差值叫作边界监督效果。

N2最低统计交付：

- post_correct−卫星基座：是否实际提高Mask AP250及Boundary AP。
- post_correct−post_constant：是否需要正确UAV RGB。
- (post_correct−post_constant)−(pre_correct−pre_constant)：更换入口是否改善内容利用。
- 正确训练的同一post检查点额外constant/donor RGB推理，仅作依赖诊断；不代替成对训练内容差。
- bbox/score/顺序严格一致，FP32完整预测，500次相同图像抽样的配对区间；已有test为开发集，单seed区间不代表跨seed稳定性。

工程准入须包括FP32/AMP alpha=0精确回放、无支持精确回放、只挂一次且实际触发、250 RoI推理与高实例训练反向资源测量、原卫星参数与buffer指纹不变。后激活改动会改变梯度与激活分布，不能套用旧conv_s0的显存或梯度结论。遥测沿用原字段，在离线审计中标记注入位置后再比较applied/base RMS，跨位置RMS不能直接比较。

本轮完成只读核查与本机试验收紧方案；**未新开本机训练**。若进入实施，仅GPU1/2/3的额外独立slot排队，不抢已有owner的任务，也不改已冻结L系列源码。
