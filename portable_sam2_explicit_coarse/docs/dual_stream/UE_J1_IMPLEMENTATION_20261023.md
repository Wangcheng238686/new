# J1编码器内部交互：实现与启动

产物根：`/data/wangcheng/checkpoint/ue_joint_encoding_20261023`。设计依据与论文事实见UE_RS2_SAM2_JOINT_ENCODING_PROBE_20261023.md。

## 模型与梯度

- `rsprompter/uav/joint_encoding_probe.py`：JointEncodingAdapter，实际Hiera Base+ blocks[4]输出 `[1,64,64,224]`，后续blocks[5..23]正常处理增强输入。没有修改原Hiera/卫星源码。
- UAV P0投影32维、逐token LN，按已有L1连续source_uv合同读取16384×8候选。group_four_cells以dy、dx、原候选顺序把2×2旧格拼成4096×32，不resize、不筛选、不去重。
- reverse attention：每个UAV候选读卫星3×3邻域（边界mask、固定2D位置编码）；更新后U以系数.2进入forward attention，卫星读取32个更新候选。single head、cosine temperature4。四子格另有固定位置编码，原几何属性保留。
- 输出32→224映射零初始化，无额外零alpha；无支持在输出映射后严格屏蔽bias。两方向路径闭合，不丢弃reverse输出。
- 原SAT（含已训LoRA、FPN/检测/CSPM/SAM/UDPR）参数全部冻结。原模型eval，新增adapter及UAV LoRA独立train。encode UAV在卫星extract前完成，整段卫星extract保持autograd；前段无可训练参数且无输入梯度自然不建图，融合后图正常保留。
- 数据与几何维持v1/EXP-0017：RGB正确/常量成对、目标深度/姿态/支持不变、EXR R camera-Z cm→m；不使用UAV标签。

## 独立runner与训练目标

`scripts/probes/ue_joint_encoding_probe.py`独立复制已验证的训练/FP32推理框架，新增SOURCES与J1合同，不动在跑的其它source。

10epoch/2670更新、batch1累积8、2136图/epoch、seed45初始、逐epoch1044000+epoch、AdamW LR1e-4/wd.01、warmup50/cosine、clip.5、AMP scale512、无EMA、固定E10 last。两臂严格加载同一common_init.pth。

提前融合后，RPN/检测/coarse等损失可能有到adapter的路径。objective递归收集key含loss的tensor及list/tuple项，只跳过无梯度项，避免旧尾部runner过滤tensor而漏掉RPN列表损失。独立objective_gate验证列表RPN项有梯度、accuracy类字段不被计入、冻结常数项不参与优化。不能因当前原头冻结就丢掉其对新特征的监督。

## 已通过的门

1. CPU：四格32候选顺序及边界，零输出恒等、非零bias无支持恒等、forward/reverse/source梯度非零。
2. CUDA：最大GT训练样本三步反向；FP32/AMP零映射精确复现raw mask logits/masks/bbox/scores；非零映射无支持恒等；250RoI FP32有限；原SAT参数/buffer精确不变。
3. 接线：非零临时bias时block5接收的输入就是被替换的block4 tensor；stride4 lateral精确不变；stride8、SAM embedding、检测neck确有变化。临时状态丢弃，正式重新加载初始化。
4. 训练峰值allocated4400.93MiB/reserved4522MiB，250RoI FP32 allocated2286.53MiB/reserved4522MiB（allocator高水位保留，并非纯推理净开销）。GPU门在5500MiB硬allocator限制内通过。

接线门首轮因pool.kernel_size/stride为tuple、检查脚本按int比较而失败；仅修检查脚本，重跑通过。初次日志另存wiring_gate_initial_failure.log；模型未变。

## 资源准入与启动

`ue_joint_encoding_schedule.py`独立owner3587086，GPU0–3可用，每卡本组最多1个GPU任务。已有其它owner不抢占。

本组显存策略为**实测通过的硬allocator上限5500MiB + 外部余量1536MiB**，准入free≥7036MiB。由独立资源包装器ue_joint_encoding_bounded_worker.py设置，与门测相同；原runner数学不变。上限低于实际物理余量，超限时本任务失败停新准入，不能侵占整卡剩余空间。该策略与旧组“3GiB+1.25×实测reserved”不同，明确写入本组manifest及admission，不把旧策略假装成已满足。

正式correct3587110在GPU2，constant3587115在GPU3，disabled基座推理3587120在GPU1。两臂均已完成至少13次真实更新，loss/grad有限；卫星与原消费者冻结。两臂初始指纹与除内容/运行参数外的合同逐项一致，运行中数值源指纹已复核不变。启动不代表收益。后续以state.json/train.jsonl为准；首次startup模型重新载入共同随机初始状态，无门测优化器/临时投影继承。

## 端到端评估

ue_joint_encoding_analysis.py独立于旧分析脚本：保留459张图像顺序和GT身份核对，**允许预测bbox/score/数量变化**。每个比较输出Mask AP250、BBox AP250、Boundary AP2%/1%，500次seed20261021配对图像bootstrap，并做恒等重积累校准。

队列包含两臂各val；test基座disabled、正确CP correct/constant/donor、常量CP constant；各自对基座、训练内容差、固定正确CP的constant/donor依赖五项比较。原始run_coco_eval里历史AR@100字段实际maxDets250，报告时解释为AR@250，不改冻结公用源。

v1 test已为开发集；常量臂保留几何/支持，不能称完全没有UAV。两臂可以验证本接线的RGB内容增益，不能单独归因为双向优于单向；论文全量consumer微调与本探针冻结consumer不同，负结果不能排除所有联合微调。
