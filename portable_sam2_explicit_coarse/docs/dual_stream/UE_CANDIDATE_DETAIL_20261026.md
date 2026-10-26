# 候选先选择与细粒度几何直接消费

## 问题和本轮干预

用户授权解决查询前候选汇总、单层几何摘要/空间降采样的限制，并在四卡余量内共租。原100轮四臂和20261025机制矩阵保持原源码与预算。本轮不是S²DB-Net或RS2-SAM2复现。

Q：在卫星Hiera block1的128²/112维特征处，用卫星query读取各格最多8个独立候选。key分别由32维源RGB描述子及7维几何属性线性生成；value为原RGB特征。masked softmax后才聚合，替换原先的候选等权mean；融合后的U仍走原block4路径。key零初始化，因此初值为原mean。归因是“卫星条件化RGB/几何候选读取”，不是只有某个单一attention算子的功效。

G：各保留候选的7维几何先分别经过7→32→32 MLP，再由同一128²卫星网格的独立query读取；32→112零初始化残差直接加到block1。没有先将候选压成最近单表面高度图，也没有先经过64²平均池化。无有效候选/无原观测支持时残差严格为零。保留原单表面CNN与block4交互作为共同上下文路径，故G是额外细节旁路，尚未证明可以删去旧CNN。

两者共同使用原FP64源锚点和连续特征读取，没有平移特征假装真实3D对应。源RGB仍只编码P0，256→32，最多8候选、每视角最多2、原排序和空间格不变。Q/G不能恢复缓存未保留的表面或源像素，不能称完整三维重建、全深度无损编码或完整可见性认证。最终仍必须压成有限通道；区别是查询先于此处汇总。

## 几何单位与格点

`data/ue_candidate_evidence.py`从相机JSON和现有锚点独立复算世界高度，不读取卫星标签。7维为：

1. SAT投影u/v相对block1 stride4名义特征中心(4x,4y)的偏移/4，分别为两个通道。
2. 世界Z减采样中心Z，单位m，再/100；不是AGL。
3. source camera-Z，单位m，再/1000；不是欧氏射线长度。
4. source view yaw的cos/sin，两个通道。
5. SAT camera-Z相对同格最近保留锚点的差，单位m，再/10；不是遮挡真值。

前6维与旧geom6仅前两项差+.5，实际最大训练图核对的float16旧缓存量化误差0.0002441。block1中心0,4,...,508，与源P0既有kernel7/stride4/pad3合同一致；bin为[4x,4x+4)，描述子公开其格内偏移，不把整格当同一个三维点。原512²height/dx/dy/count/support及旧归一化不改。

## 固定实验和共同初始化

| 结构 | 正确RGB | 常量RGB | 参考平面＋正确RGB |
|---|---|---|---|
| 原compact | 复评既有E10 | 复评既有E10 | 复评既有E10 |
| Q | 训练 | 训练 | — |
| G | 训练 | 训练 | — |
| Q+G | 训练 | 训练 | 训练 |

共7条新训练。顺序先Q正确/G正确/QG正确/QG常量，随后其余匹配对照；能否并发由实际显存准入决定，不设固定槽数。每臂10epoch/2670次成功更新，按原100轮LR日程前缀运行，seed44、AdamW5e-4/wd.05/warmup100/globalclip.5/AMP512/accum8，无EMA。每轮2136唯一训练图，顺序与原对应臂同epoch逐值核对。卫星增强、源LoRA、原损失、直接中心框损失、候选NMS/打分/250上限及UDPR不变。

训练初始化只复制共同通用SAM2和新头初始化common_init，绝不加载E73或任何已训练任务权重。新增模块在fork_rng内seed2060，匹配臂共享同一初始化，恢复全局seed44。卫星原可训练范围继续开放；原冻结参数必须不变。旧E10任务CP只用于独立参考推理，不能混淆为训练热启动。

固定E10，不按val/test挑epoch。每轮val作为观察；全部7条训练完成后，对所有锁定臂执行完整459图开发test。QG正确权重另做常量/donor RGB干预，保持目标几何，donor为split顺序跨半圈，明确RGB-D不一致。纯SAT与旧compact三种内容条件E10在val/test同口径复评。所有新产物独立根`/data/wangcheng/checkpoint/ue_candidate_detail_20261026`。

## 验收和结果约束

已实际通过三个方向各3次accum8 AMP更新、208GT训练图、FP32正式推理及强制250实例掩码解码。Q/G/QG峰值reserved分别5716/5722/5724MiB。初始卫星各层输出与旧路径最大差0；Q/QG中心头最大差1.1921e-6、G为0。新增有效模块和源LoRA有实际参数更新；冻结353–362张量逐值不变。非空optimizer/scaler恢复、DataLoader generator恢复通过。CPU门验证无锚点为零、初始化恒等、候选共同置换不变和可学习选择敏感性。门不证明AP收益。

首个GPU门在通过前向/更新后，因restore.pth父目录未创建而在落盘时报错；已保留`engineering_debug/attempt1`原脚本/日志，只修复建目录并全量重跑上述三门，没有动原任务。

分析固定原路径/合并路径的Mask/bbox/Boundary2%/1%与500次配对图像bootstrap(seed20261021)，报告Q/G效应、交互项、匹配常量内容差/差中差、QG实测深度差和固定CP内容依赖。不能将内容差中的参数/优化差异混杂完全排除。

困难分布另外报告：GT和DT均按真实RLE面积的COCO尺寸AP/AR250；固定图像密度<=50/51–100/>100的Mask AP；bbox和mask IoU>=.5按score贪心匹配的TP/FN/FP、small TP/FN和固定FP/图预算下small recall。背景(maxIoU<.1)/定位(.1–.5)/重复(>=.5但已占用)均为代理，不称人工语义真值，也不是AP精确可加分解。补算真实mask面积分组须验证all-area AP不变。对照失去/找回的逐实例归因仍需结合最终共同匹配离线核查，不能单凭FP/FN总数推断。

v1 test已开发暴露，单seed、空间相关、多路线名义区间限制保留；不因测试结果调结构/轮次。若正确RGB不优于同结构常量RGB，不以非零梯度、attention非均匀或几何旁路有收益认定RGB成功。前缀结果不替代最终全预算训练与成本审查。

## 文件

- `configs/ablations/ue_candidate_detail_protocol.json`：锁定矩阵/预算/入口/单位/选权。
- `data/ue_candidate_evidence.py`：每候选7维几何，不使用标签或新表面过滤。
- `rsprompter/uav/candidate_detail_interaction.py`：Q/G独立可选，额外block1 hook，原block4消费者不改。
- `scripts/probes/ue_candidate_detail_run_20261026.py`：CPU/GPU门、共同初始化训练、完整推理、显式epoch边界resume。
- `scripts/probes/ue_candidate_detail_owner_20261026.py`：门和来源绑定、非抢占共租、全部训练完成再test、失败停止新增调度。
- `scripts/probes/ue_candidate_detail_analysis_20261026.py`：完整配对效应、真实面积尺寸、密度和检错/FROC诊断。

资源预算使用7800MiB allocator上限+512MiB上下文预留，设备另留2048MiB；统计CPU单任务，数值源/driver归档并每轮核对。逐步遥测不增加，定义见DEBUG_FIELDS.md。收益和最终交付尚待实际完整结果。
