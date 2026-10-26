# 实例监督驱动的编码器内部回写探针

状态：实现与工程验收中；本协议先于本轮任何AP结果固定。用户已授权完整验证实施流程。

## 目的与证据边界

借鉴RS2-SAM2编码中交互、更新表征继续编码的机制；不是图文模型复现。旧J1已做过单入口双向交互，未确认RGB内容收益。本轮保留已证实有用的深度几何内容，并让直接实例损失穿过卫星回写。

主问题是：**更新后的特征继续进入卫星编码链，是否比仅供新增实例头使用产生额外净收益？** 两个入口均计算双向attention，所以不把这个比较称为双向attention对无attention的消融。

## 固定架构

- 五camera-Z深度与内外参生成既有native512几何：相对高度/100、邻接高度差/10裁至±5、最近表面1m内视角数/5，加独立support。沿用已审计缓存，不改变cm→m、坐标轴、表面选择。
- 原几何CNN产生128²×32特征，保留高分辨率旁路。2×2平均池化得到64²，名义中心8j+2，与Hiera首次Q-pool的名义格点一致；并不认证精确3D可见点。
- Hiera Base+ block4后，卫星64²×224降维32。U读取卫星3×3邻域，再卫星读取更新U的3×3邻域；中心无支持时回写严格为零。
- 回写224维投影零初始化；没有第二个零alpha阻断。更新U增量上采样追加至原128²特征。
- 中心头读取 `[U128更新特征, Align(Project(S′))]`，轻量卷积后预测中心／偏移／logwh。**中心损失必须训练到S回写投影和attention。**
- head入口：S′只供新中心头，原S继续进入block5；internal入口：同一个S′同时进入block5及后续检测／掩码链。其余相同。
- RGB扩展：五视角SAM2基础权重编码器＋LoRA，32维无affine归一化特征按旧L1连续源坐标读取，保留旧最多8候选/每视角2上限，masked mean后与几何128²特征拼接卷积。对应关系与几何数值不替换。RGB编码器原参数eval，LoRA dropout训练期开放。
- 原SAM解码器、UDPR、coarse64不改；不增加边界loss。所有原卫星参数和buffer冻结eval，融合后保留autograd。

## 预定训练矩阵：14臂

阶段G：`{head,internal} × {真实几何,常量几何保留support} × seed{44,45}`，8臂。

阶段RGB：`{head,internal} × {真实几何+正确RGB,真实几何+常量RGB,常量几何字段+正确RGB} × seed44`，6臂。

冻结前独立审查补充两条常量几何字段RGB臂，以检验完整RGB-D结构内几何数值贡献。此修订先于本轮任何AP或训练结果；原12臂尚未启动，不根据结果扩矩阵。保留真实深度对应与support，不称无深度。

两阶段均无条件执行，不根据test排名决定是否加入RGB或第二seed。不新增第二交互入口，避免更改当前问题。几何阶段和RGB阶段分别复制各自完整初始化；RGB新增模块使用独立RNG，原共享模块初始化逐值相同。不同seed初始化不同。RGB仅消费source P0，实际有效LoRA范围按梯度/optimizer state披露，不能声称整个UAV编码器均被主损失训练。

共同合同：v1 train2142/val459/test459，EXP0017 E73冻结；10epoch、每轮2136图、batch1/accum8、2670次实际更新；AdamW lr1e-4 wd.01，warmup50 cosine至.1，clip.5，AMP init512，无EMA，E10 last。逐轮seed=1000000+1000×seed+epoch。保留卫星高斯噪声p.5/std.02与既定ColorJitter，无源空间增强。原任务全部loss加1.0×直接实例loss；后者与成功pilot相同的中心focal＋offset SmoothL1＋.1 logwh SmoothL1，分母唯一正中心格数。LR与先前独立几何CNN的1e-3不同，不作为对旧实验的单因素比较。

## 推理与归因

FP32，全459图，RPN pre2000，maxDets250。保持已经固定的原RoI假设补充、几何独立objectness、novelty/NMS=.5与优先分数规则；不经过冻结SAT分类器否决新几何框。无支持不增加UAV框。

每臂完整val和test。真实训练臂固定E10检查点补常量字段、整图几何donor；RGB真实臂补constant RGB/donor RGB。几何donor同时换support，固定半循环；RGB臂此时仍保留目标RGB及其候选anchor，是混合不一致输入，不是完整物理一致的RGB-D场景donor。RGBdonor只换RGB、保留目标深度/pose，同样明确输入不一致性。纯卫星来自disabled运行，与历史强SAT逐值回放；新的正式disabled路径先全459图回放，未通过不得启动引用历史基线的统计。

每个评价同时保存：

1. `satellite_records`：原检测/掩码路径及其RoI假设补充；**internal时已受交互条件化，不是纯卫星**。
2. `combined_records`：上述条件化前缀＋几何实例头补充。

分别报告条件化原路径−disabled、合并−条件化原路径、internal−head、真实−常量及RGB真实−常量。内容及入口比较同时在条件化原路径和合并结果上计算，不能只凭新增框收益声称原卫星路径得到内容增益。原冻结前缀AP不降性质不适用于internal对独立纯卫星。

Mask AP250、AP75、bbox/AR与标准面积组；Boundary AP2%/1%；500同图配对bootstrap、seed20261021、名义95%区间，沿用原函数校准。Mask +.002 raw且下界>0为明确净收益参考，完整报告小信号与负结果。不把图像bootstrap当训练重复或独立区域验证；v1 test已用于开发，不作为未暴露最终基准。

## 验收与审计

长训前：缓存来源与旧审核绑定；64/128线性ramp与边缘；FP32/AMP零回写原logits恒等、非零权重无支持恒等；同权重head/internal中心输出相同且block5接线不同；直接中心梯度达卫星回写、双向attention、几何CNN与RGB LoRA；最大训练实例＋250FP32实例显存；原卫星参数buffer SHA不变；实际续训恢复五类RNG和loader顺序。

新目录 `/data/wangcheng/checkpoint/ue_instance_interaction_20261024/`；不修改旧source/缓存/产物、不抢占旧任务，实测显存峰值＋余量按free显存共租。失败保留日志，不因AP较差提前终止。训练仅epoch损失均值与低频进度，采样/注入/梯度机制优先离线。

最终审计检查14个实际E10/2670 CP、实际有效参数optimizer步骤、初始化、每轮顺序、源码快照、冻结SHA、全部评价身份和配对统计。当前固定119个训练/推理/四指标统计任务，不含工程门。当前协议不是完成记录。共同基础初始化的最终全预算联合模型仍属后续正式训练；不能把E73冻结探针当作该验证已完成。
