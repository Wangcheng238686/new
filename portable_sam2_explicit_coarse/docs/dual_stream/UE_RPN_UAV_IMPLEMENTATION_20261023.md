# UAV几何辅助RPN排序：实施与工程验收

## 独立设计与实际实现

用户要求独立子代理给方案、主代理负责实施。[独立方案](UE_RPN_UAV_PROBE_INDEPENDENT_PLAN_20261023.md)已交付并经实际源码只读复核。首轮检验深度导出的几何信息，RGB与LoRA留待该检测入口成立后检查。

新增：

|文件|实际职责|
|---|---|
|`rsprompter/uav/rpn_geometry_probe.py`|冻结RPN分类层后的有界logit残差；只拥有新增query/mixer/alpha参数，weakref引用原RPN|
|`scripts/probes/ue_rpn_geometry_probe_20261023.py`|独立init/gate/train/eval、完整RNG续训、原RPN分类loss、完整检测阶段与实例推理|
|`scripts/probes/ue_rpn_geometry_schedule_20261023.py`|工程门后冻结来源、共租三臂、预算与干预评价、CPU配对分析及最终检查点审计|
|`scripts/probes/ue_rpn_geometry_analysis_20261023.py`|允许检测输出改变、严格同图同GT的bbox/Mask/Boundary AP配对bootstrap|
|`scripts/probes/ue_rpn_rank_reachability_20261023.py`|固定4train图的原候选logit差与±2分数残差可达范围诊断|

复用的旧runner、native缓存、检测阶段trace和AP估计器仅读，不修改它们的冻结源码。完整SAT/RPN/框回归/RoI/SAM/UDPR参数与buffer继续冻结eval。

## 唯一模型干预

原RPN每个特征层`rpn_cls`输入hidden投影512→32、逐位置无affine LN；真实priors解析各位置anchor中心，nearest读取原native512四字段和support；拼接37通道，经1×1 Conv(37→32)/GELU/1×1 Conv(32→A)。实际配置每位置2个scale×3个ratio，A=6，非硬编码3。

`logit'=logit+2*tanh(alpha)*tanh(head)*support`，alpha零初始化。残差单位为objectness logit，绝对上限2；原bbox deltas不变。修改的同一logit进入原RPN分类loss和predict_by_feat，贯穿sigmoid、top-k、RPN NMS和1000 postcap。原RoI分类头重新生成最终scores，不把RPN分数冒充最终分数。

native栅格按原projection floor-bin定义，当前真实anchor中心为整数位置，用`(u+.5)/256−1`的nearest读取对应栅格。stride64大anchor的FP32中心抵消误差实测最多1.5259e−5px，原1e−5一致性门过严；改为1e−4px后精确native bin不变。第一次失败日志和数值检查保留，正式训练前修正，不改几何筛选或加半格位移。

## 三臂与固定合同

- G：真实native四字段。
- K：支持内固定为**train-only有效像素均值**，值与SHA直接复用已有`common_constants.json`，不是人为全1。
- Z：四字段为0。三臂都保留原support、卫星query与相同容量，不能称K/Z完全没有UAV。

共同完整初始化，固定EXP-0017 E73/v1归档SHA。只训练新增模块，唯一loss为原RPN anchor分类loss，原assigner/sampler不变；不通过离散top-k向新分支反传mask损失，不训练框回归。

10epoch/2670次更新尝试、batch1/累积8、AdamW LR1e−4/wd.01、warmup50/cosine至.1、clip.5、AMP scale512、无EMA、E10 last。每轮2136图，E1起1044000+epoch，与最近geometry/ladder合同一致，区别旧S²DB E1–4随机流；不把旧模型当单因素训练对照。卫星噪声/ColorJitter保留，无空间增强。DataLoader generator/Python/NumPy/Torch/CUDA与optimizer/scaler一并保存恢复。

## 已通过工程门

最大train样本index875，208个GT：

- FP32/AMP零alpha原RPN logits及bbox回归、最终框/score/mask逐值恒等。
- 非零alpha、全无support时恒等；实际anchor中心坐标ramp读取正确。
- 同seed无分支/有分支原anchor labels、weights和随机采样结果一致。
- 合法alpha更新后query/mixer梯度可达；首步仅alpha梯度为预期。
- 新logit确实改变原proposal scores，原bbox deltas逐值不变。
- 250个最终RoI的FP32冻结mask路径有限，不以RPN-only路径替代资源门。
- 两个真实增强mini epoch（各8图、双worker、累积8）的边界续训：顺序、输入、loss、更新后adapter权重和generator终态完全一致。它是短工程回放，不称267次更新的正式epoch复跑。
- 全部原模型参数及buffer SHA不变；门中临时参数更新恢复共同初始化。

峰值reserved3342MiB、allocated1984.84MiB；新owner工作预算4177MiB，另留每卡2048MiB余量，一张卡最多一个本研究GPU子任务，可与既有研究共租。

## 已启动任务与后处理

根目录：`/data/wangcheng/checkpoint/ue_rpn_geometry_20261023`。
耐久owner：`ue_rpn_geometry_schedule_20261023.py`；实际PID以owner.pid/state.json为准。

三臂均已产生正式训练更新，非仅启动命令。固定4train候选可达诊断已完成。纯卫星pre-topk1000基座及2000预算诊断分别跑val/test，postcap1000、NMS、final250均保持。该预算诊断不属于UAV收益。

三臂E10均完整val/test；G固定CP另外constant/zero/shuffle字段推理（保留原support，属于输入干预）。训练/评价/统计全量固定，不按开发test选权、改范围或停止某臂。

新的评价保留原检测trace全部10阶段，不沿用“新框必须等于旧卫星框”的assert；仅要求trace内部独立回放与实际当前RPN/RoI一致。每图保存最大IoU、任意覆盖与一对一容量，结合最终bbox/Mask/Boundary AP分析RPN改善能否穿过原RoI NMS。

GT保持同图同标签；检测框、分数、实例数允许变化。配对分析先核定459图GT完全一致和AP身份重累积，再500次同图bootstrap(seed20261021)。G−SAT和G−K为主比较，G−Z、K/Z对SAT、预算及固定CP内容干预完整单列。Mask AP+.002且名义区间下界>0是净收益参考，仍须多训练seed/独立数据才能称稳定。

完成审计会实际读取三份E10/2670权重、完整更新日志、逐epoch样本顺序、源码快照与各评价身份，并核对三臂初始化和样本顺序一致。门通过和训练loss下降均不等于检测或分割收益。
