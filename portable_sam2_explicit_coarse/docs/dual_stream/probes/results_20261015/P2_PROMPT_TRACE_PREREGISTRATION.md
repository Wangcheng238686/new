# P2→粗掩码→提示→最终mask：冻结读取路径追踪

日期2026-10-15。关联EXP-0006/0007。仅离线观测，不训练、不改默认结构/强度，不新增P2方案，不追加RPN/bbox/AP计算。

## 范围/对照

当前冻结开发子集全部153验证图、4496次出现、237栋建筑。固定E72 strict EMA和上轮GT共同框/GT顺序。复现上轮P2迁移四组：baseline、native_correct_q（.25主对照）、native_donor_q、native_correct_one（压力对照）；不重复上采样，因为当前目标是定位P2后续链路，已知其终点持平且本轮不做纯分辨率主张。

正确和既定donor_a仍按同一目标几何提取BEV，借原P3融合器构造R2，仅改变mask输入x[0]；其余FPN、SAM/PE/high不变。每组GT共同框最终IoU必须逐项精确等于上轮p2_transfer_audit；非P2 GT的RoI/coarse及最终paste mask必须不变；若其他中间量因全invalid批次的token长度变化而变，需单列批处理结构效应，不当作P2直接内容作用。

## 只读观测位置

只读forward/pre hooks，返回None、不得修改输入输出；按正式mask_roi_chunk_size拼接GT顺序：
1. mask_head输入RoI特征[N,512,14,14]；
2. shape_injector.mask_decoder输出粗logits64²；
3. point_miner实际coords/labels和local_yx（四槽，忽略两侧均invalid时的无效坐标变化）；
4. prompt_encoder实际mask canvas（预计128²）及PE原始dense输出；
5. mask_decoder实际sparse/dense输入、raw低分辨率logits（tail前）；
6. mask_head最终lowres logits（tail后）和正式paste后的二值mask。

强制检查captured ROI等于coarse decoder输入，captured coarse等于MaskHeadOutput.coarse_outputs.raw_logits，point_miner输入就是同一coarse；chunk数/每chunk GT数量/boxes顺序相同；hook移除及参数version/特征hash不变。3图smoke各组hook开启/关闭逐像素mask比较，正式153只跑开启hooks避免重复算力；完整端点IoU精确对账上轮。所有张量有限，正反例与阈值按当前部署确认。

## 离线字段与分母（不写生产遥测）

每GT记录各连续阶段delta RMS、mean abs、relative L2=||variant−baseline||/max(||baseline||,1e−6)、严格数值变化/变化>1e−6；不同stage不同维度/尺度，不能把relative ratio之间直接当作信息保真率。

粗掩码：sigmoid后prob MAE、>=.5二值翻转比例；GT目标使用实际mask_head.get_coarse_targets同一floor/ceil crop+nearest resize64，不注入模型提示。记录粗IoU、BCE及变化，用于判断变化方向是否有用，而非仅是否非零。

点：四槽label有效性、image coords、local_yx；记录按有效槽的坐标/label变化比例、移动距离，以及按coarse GT target采样的前景正点/背景负点正确数与有效数。标签无效槽不算错误；它只是GT条件诊断，非部署输入。

Dense：canvas相对变化、PE dense变化、送入decoder的最终dense变化。记录mask_head实际shape-dense gate（与UAV gate不同）及32²下bbox支持cell数；support=0时最终dense必须为原no-mask基底。PE sparse若因all-invalid从6token变为2token，保留实际长度，统计比对时仅用左侧补0到6的编码副本，不改变输入；不能把补齐距离当完全等价的序列行为。

Decoder：raw/tail后logits RMS变化、部署logit阈值对应的翻转率（mask_thr_binary=.4时logit≈−.405465，不用logit0替代）；最终paste mask像素翻转及IoU变化。用于区分连续变化和阈值化后无变化。

## 汇总/不确定性/解释门

主定位口径：固定GT框下阶段变化到达比例及粗IoU/点正确率/最终IoU变化。分P2 RoI3657/非P2 839、small881/medium2510/large1105、32² dense support为0/非0分组；237建筑先内部均值再等权。

1000次配对图像sum/count bootstrap seed1750，仅粗IoU和最终IoU的variant−baseline，全部四组报告。输出纯描述性阶段计数，不按最大的中间变化选择方案，无新“增益门”或最优层决策。

定位用定义：连续变化RMS>1e−6才计数值到达；粗二值变化按>=.5；点变化按有效性和坐标；最终变化使用实际paste二值。若某阶段不变可定位该冻结输入的阻断位置；若各阶段都变但质量不升，只能称未产生稳定有益变化，不能强说decoder压制了有用信息。

关键训练边界：代码和checkpoint应核定roi_only、dense detach_input、point no_grad、coarse auxiliary weight。此轮不做反传，静态路径只能说明这些配置下final-mask loss没有沿coarse提示回传，不能量化训练动力或反事实重训收益；SAM图像embedding直连segm梯度仍存在。沿用继承单流的契约不叫新实现bug。

无optimizer/模型更新，无全量训练；机制字段不能替代结局指标，上轮条件AP结论保留。所有结论仅当前冻结开发集，非独立test或完整生产集。
