# UAV几何辅助RPN排序：独立首轮方案（2026-10-23）

状态：独立设计建议，尚非实现或运行结果。主代理负责工程核定、冻结合同、实施及文档索引。

## 1. 推荐及证据边界

首轮推荐**三个训练臂的native几何RPN logit残差**，RGB编码/LoRA后置，不同时改RoI NMS。目标是检验：UAV深度导出的空间信息能否帮助原检测器把有价值的小建筑候选保留到下游。

理由：检测阶段459图独立验收已通过。44988个GT中，全部回归RPN框任意IoU≥.5覆盖44969；每层top-k后43049，最终41456，一对一容量40376。最终无覆盖3532个GT中，最后丢在top-k的1838个（small1833），RoI NMS841个（small832）；final250上限损失0。阶段路径不是单步骤干预的因果收益估计。

已有几何真实字段对常量字段训练的四项比较有小幅正信号；RGB A–D整体Mask AP内容差弱，N2-shared只有小边界信号。首轮无需再引入RGB聚合、LoRA、学习偏移、辅助分割头。这不否定RGB价值，而是先检查新的检测消费入口。

## 2. 冻结与唯一改动

- 沿用EXP-0017 E73/v1，所有原卫星模块、RPN卷积/分类/框回归、RoI检测、SAM/UDPR参数与buffer冻结并eval。
- 不加载旧mask适配器；新增模块是唯一可训练参数。不同于此前“只训练UAV LoRA”的RGB合同，本轮明确没有RGB编码或LoRA。
- 在`model.rpn_head.rpn_cls`的真实分类logits后加残差，输入query使用该层输入的冻结RPN hidden（已经过原shared conv/激活）。不要复制近似RPN前向，不要改原bbox deltas。
- 修改后的同一份logits同时送入原`loss_by_feat`和`predict_by_feat`。原top-k、RPN NMS排序及post-cap必须贯穿使用新分数，不能只替换top-k索引又恢复旧分数。
- RPN每层pre-topk1000、postcap1000、原两级NMS/分数阈值、final250不变。RoI分类分数仍由原头重新计算，不能把RPN新分数当最终检测score。

只改RPN排序可能补不回RoI NMS后损失。若RPN覆盖上升而最终AP不上升，要记录瓶颈迁移，不自动再加NMS模块。

## 3. 几何输入及轻量结构

复用已审计缓存`/data/wangcheng/cache/ue_v1_native_geometry`，4字段为height_m/100、clip(dx_m/10,−5,5)、clip(dy_m/10,−5,5)、nearest1m内视角数/5，独立512² support；真实字段严格复用旧float16缓存转float32，不重建、更换筛选或扩展覆盖。

在各实际RPN FPN尺度H×W上，先从真实`grid_priors`按location/anchor顺序解析每个中心(u,v)，核定同location的A个anchor中心一致，不用`interpolate`均分图像推定中心：

1. 以`grid=(2*(u+.5)/512−1,2*(v+.5)/512−1)`显式`grid_sample(...,mode="nearest",align_corners=False,padding_mode="zeros")`读取support，得c∈{0,1}；越界中心必须显式c=0。
2. 同一grid读取四字段得g，c=0处g=0。保持真实512栅格点读取，不填深度孔洞，不引入区域池化半径。实际中心0,4,…与area bin中心不一致，不能以area resize冒充anchor对齐。
3. `q=Conv1x1(C_hidden→32)(hidden.detach())`，对每位置32通道无affine LayerNorm。
4. 拼接[q,g,c]共37通道，`Conv1x1(37→32)→GELU→Conv1x1(32→A)`，A由原rpn_cls输出anchor数核定，空间/anchor展开顺序完全沿用原头。各层共享参数。
5. 建议固定`delta=2*tanh(alpha)*tanh(h)*1[c>0]`，`logits'=logits+delta.to(logits.dtype)`；alpha单一可训练标量零初始化。h末层保持确定的随机非零初始化，避免alpha与末层同时零导致梯度锁死。范围±2为预定logit单位限制，不据开发test调大。

这是RPN真实anchor中心位置的native点字段读取，不是anchor框内三维重建，也不宣称该点代表整栋建筑。中心无支持即不修改该位置；粗层中心稀疏可能遗漏局部结构，这是首轮归因边界。512²到H×W按照实际FPN全图坐标，禁止套用旧RoI局部坐标。保留support输入和掩码，但不把support称遮挡认证。若当期训练含空间增强，必须先同步变换geometry/support；不能单独增强卫星。首轮沿用现有噪声/颜色增强而不增空间增强。

## 4. 三个匹配训练臂

|臂|4个字段|support|query/容量|
|---|---|---|---|
|G真实|真实native字段|原图保留|相同|
|K常量|支持位置固定train-only有效像素均值，约[.273104,−.001392,.003337,.498688]，无支持0|原图保留|相同|
|Z零|全部0|原图保留|相同|

主代理工程采纳修订：K复用`/data/wangcheng/checkpoint/ue_geometry_mechanism_20261022/common_constants.json`，由2142个train样本486813240个有效native像素按像素加权求四通道均值，完整精度与文件SHA进入manifest，不读取val/test均值。原独立初稿建议[1,1,1,1]；其相当于100m高度、10m梯度等较大常量，可能人为削弱对照，因此采用train-only均值作为更合理、更强的匹配内容控制。三个臂复制完整同一初始化快照，独立相同数据顺序、anchor采样及RNG恢复；新增模块初始化在fork_rng内完成，不改变原数据/采样随机流。K/Z仍包含真实UAV观测support，不是纯卫星或完全无UAV信息。G>K支持字段内容价值；G>Z排除同容量query+support调整的大部分解释，但G/K/Z与原SAT之差仍不能单独归因RGB或精确三维几何。

固定检查点另做真实/常量/零/联合4通道空间置换输入。置换仅在supported像素内联合置换向量、保持support与联合值多重集合；固定seed预先记录，不据结果挑置换。干预含分布及query关联变化，不能用“置换变差”单独证明正确几何贡献。

## 5. 监督与完整预算

首轮唯一训练目标为原RPN anchor objectness分类loss，使用原GT bbox、assigner、sampler、正负标签、ignore规则、loss权重与平均分母。不新增mask/边界/辅助loss、不变更anchor标签为decoded IoU标签。

- 正常调用原RPN `loss_by_feat(cls_modified,bbox_frozen,...)`，只汇总原分类项训练新增模块；框回归项可计算后离线核验但不参与优化。原冻结网络不得因为loss函数默认路径而被train()。
- 不依赖top-k/NMS的离散路径反传mask loss：那会让目标梯度无法直接到达新排序分支。只做AP推理后训练mask loss不构成有效的RPN训练监督。
- 沿用10epoch/2670次更新、2142train/459val/459test、batch1累积8、AdamW1e−4/wd.01、warmup50/cosine到.1、clip.5、seed44及各轮显式epoch seed=1044000+epoch、AMP训练/FP32评价、无EMA，固定E10 last。
- 工程实际采用从E1起每轮1044000+epoch重置，与旧S²DB E1–E4的RNG协议不同；三臂彼此严格一致，不能称与旧RGB实验完全同训练合同。此处“同预算”指三个新臂一致；不能把本轮RPN训练loss与此前mask loss训练结果视为同一个干预。
- train只读train；val诊断；已反复使用的test为开发有效性评价，不能作为独立泛化确认。保存Python/NumPy/Torch/CUDA/DataLoader generator并做续训样本与anchor采样顺序核定。

## 6. 训练前的纯卫星预算诊断

独立固定SAT无训练，完整推理两项：原pre-topk1000基线、仅pre-topk2000（postcap1000及其余不变）。基线优先复用已逐值绑定记录，2000单独跑。记录RPN top-k/postNMS/postcap、RoI decoded/score/NMS各级覆盖和一对一容量、bbox/Mask AP与耗时峰值。

额外在固定train工程样本记录：每GT达到IoU≥.5的原候选最高logit、其所在层排名及第1000名logit差。±2限制下，若所有合格候选比对应层阈值低超过4 logits，则仅靠本分支不可能把其排进top-k；此可达上界用于解释，不据开发test调范围。

目的：判断原排序遗漏是否能够通过较宽候选预筛选实际传到下游。新增训练仍固定1000；2000只能称候选预算诊断，不能称UAV收益或混作G对照。更宽预算不改善不排除学习排序改善，但会提示下游压制。无需首轮同时调RoI NMS阈值或最终250上限。

## 7. 验收门

- 基座/数据/cache source SHA与缓存版本绑定，字段单位与support恒等；所有比较保存同一完整初始化指纹。
- alpha=0时原RPN logits、bbox deltas、实际proposals、最终框/score/mask逐值恒等（FP32）；AMP检查dtype与有限值。
- 全无support时任意参数alpha都恒等；c=0处不能query-only改分数；valid zero-height地面不被当孔洞。
- 合成坐标ramp、已知4字段、孤立支持像素/孔洞，核对真实prior中心→源像素、实际FPN尺寸/stride、anchor通道及flatten顺序。原中心0必须读取pixel0而非1.5；边缘/越界/空GT/空有效support测试。
- 人工非零residual确认top-k及NMS分数都改变，原bbox delta不变；mask logits在不同框上当然可变化，不要求新模型框score不变。
- alpha=0首次backward只保证alpha梯度可达；在一次合法alpha更新后，证明query/mixer梯度可达。不能把零初始化首步query无梯度当故障。
- 原RPN标签/weights及随机anchor样本与无adapter同seed逐值一致；原网络buffer/参数SHA全程不变。
- 最大238GT图验证250推理实例路径及资源峰值；先短工程门，不用微小测试loss作为选结构标准。独立续训短跑确认样本顺序与更新等价。

## 8. 评价、机制与成功判据

完整E10主评价：SAT/G/K/Z的Mask AP250、AP75、Boundary AP2%/1%、bbox AP/AR；同图bootstrap500，seed20261021，名义95%区间。预先主对比G−SAT及G−K，Z容量比较支持解释；维持Mask AP≥+.002且区间下界>0的明确净收益参考门槛，并完整报告较小信号。训练种子重复和独立数据检验后才能称稳定。

检测机制重跑已有阶段漏失probe，但不可沿用其“最终框必须等于SAT”的assert：新框/score应变化，恒等assert只用于alpha0工程门。每级公开GT总数44988及各尺寸/密度分母；记录RPN/最终any覆盖与一对一容量、lost与recovered，不将AP和容量混称。small组按真实mask面积定义，canonical AP尺寸组沿用旧bbox-area合同并并列披露。

离线优先统计：支持率、各层delta/logit幅度/达到预定上限95%比例、top-k候选替换率、被换入/换出的GT覆盖、GT的最高正候选排名变化、RPN阶段增益保留到RoI NMS后的比例。GT排名诊断每图单独，不将共享框当多实例成功。

结论分级：接通并改变排序；真实字段比匹配内容控制更好；最终超过纯卫星。只有RPN recall上升属于机制正信号，不能替代最终Mask AP收益。

## 9. 资源与后续

native四字段缓存和约小型1×1网络，不需要五视角RGB编码；冻结SAT特征可no_grad，训练仅RPN分类，无SAM反传，应明显低于既有RGB训练预算，但必须实测。工程门后按当前可用显存+实测峰值+2GiB余量共租三臂；不抢占现有任务，不等独占空卡。训练和推理分别测峰值；离线梯度审计不要套用推理预算。

若G在RPN阶段优于K/Z但最终被RoI NMS吞掉，第二轮只检查密集实例保留，另立冻结合同。若G不优于控制，先查中心支持覆盖/anchor监督和排序目标是否匹配；不立即叠加RGB/可学习BEV。若G形成净AP收益，固定结构先重复训练种子，再考虑RGB在同一检测入口的增量，避免跨阶段同时改读、融合、监督和NMS。

## 10. 实施源码只读审查（正式训练前）

已读`rsprompter/uav/rpn_geometry_probe.py`及`scripts/probes/ue_rpn_geometry_probe_20261023.py`初版。没有发现明确的旧score覆盖、分类loss梯度断路、hook级索引错误或原参数解冻：RPN前hook重置level并生成真实priors；rpn_cls hook按level读取hidden，输出直接进入原loss_by_feat及predict_by_feat；bbox delta未改；eval允许新框/分数变化，只有工程门要求alpha0恒等。native字段最近邻读取与真实整数anchor中心一致。

主代理仍需补齐验收证据：无adapter/有adapter同seed的anchor labels/weights与采样顺序逐值一致；实际epoch边界续训等价短跑；工程门的空支持与坐标孔洞、边缘检查；新排序的train可达率。现有restore函数和梯度门存在不代表这些门已经通过。初版正式eval只保存topk/postcap/final三个阶段，原检测probe的更全阶段回放应按实施承诺单独产出，才能判断收益是否被RoI NMS吞掉。
