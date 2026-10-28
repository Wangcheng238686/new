# 现有探针的梯度拓扑核查（2026-10-23）

## 已有证据

核对 `LocalUAVAdapter`、Unit、S2DB、AttentionLadder、GeometryMechanism及本机SAM2/UDPR实现，并读取L1/L2/L3实际GPU gate的三步反向记录。

- 三组step0：alpha梯度非零；source/Q/K/V/几何投影梯度为0。alpha零初始化使 `dL/dtheta = alpha * dL/d(applied) * ddelta/dtheta`，属于预期的初始延迟。
- 三组step1/2：上述投影梯度非零，UAV encoder的部分LoRA有非零梯度，L3 null_logit也非零。因此这些路径不是持续断梯度；三步单样本门不证明整段训练梯度充分。
- 冻结卫星参数不等于禁用decoder自动求导。runner仅以no_grad缓存卫星extract_feat；新增残差进入decoder之后仍可向UAV反传。不能将冻结消费者自动解释为断梯度。
- UDPR仅用detach的logits选择离散点；所选logits、高分辨率feature以及最终mask路径未全部detach。离散选点不可导，不等于最终mask梯度断开。

## 剩余盲区与已覆盖范围

| 问题 | 当前覆盖 | 剩余限制 |
|---|---|---|
| alpha=0起步延迟 | L1–3已见随后上游梯度；4090 P2改变非零初始化 | 不必仅凭第一步0梯度再开全组；AdamW状态也使原始梯度大小不直接等于更新幅度 |
| GELU前注入的消费者敏感性 | 多数本机分支共享conv_s0入口；4090 N2改变入口 | 本机L1–5、采样/concat/difference对照不能独立检验该入口；有梯度不代表更新能改善边界 |
| 冻结消费者适配 | L6指定decoder Q/V LoRA成对训练设计 | L6不是N2；额外消费者容量仍需常量对照，且未完成结果不能先解释为成功 |
| 主/辅助损失方向冲突 | S2DB已有独立gradients审计脚本、固定4个训练样本 | 脚本存在不是结果完成；当前根目录未找到其gradients.json。现脚本将多种主损失求和、参数组求总范数，仍可能掩盖局部冲突 |
| 卫星Q捷径 | constant训练、native Q/G分解、difference及aux detach-Q | 多数RGB注意力仍可通过Q和几何生成残差；总分支梯度不能单独证实RGB内容被学习 |
| 固定投影候选/可见性/离散选择 | L1连续读取、L2视角组织、L3拒绝、L4分离打分、L5深度偏置 | 固定缓存的坐标/候选集合本来不可学习；分割梯度不能找回已丢候选。不是通过增加反向传播就能修复 |
| 检测任务 | 冻结候选框/分数 | 后端mask探针设计上不能改善bbox召回，不能用bbox无增益指认梯度故障 |

LayerNorm去掉幅度、softmax候选竞争等会改变梯度方向/敏感性，但不应未实测就称为梯度消失。L3已专门避免输出LN抹掉拒绝概率；不能重复以同一理由立项。

## 收紧后的补充建议

先补**离线梯度诊断G0**，再补唯一新训练机制N2式后激活注入的correct/constant一对。暂不追加一整批初始化、辅助loss、解冻和新head训练。

G0不更新权重，不改变已冻结source，固定训练集密度分位的4个样本及同一checkpoint/同一随机状态，FP32/dropout关闭。建议覆盖归一化attention基准、concat最强臂、几何difference；辅助臂可用后再补。起始与早期/末期checkpoint分开报告，不能以临时改alpha的反向替代实际训练轨迹。

诊断分三层：

1. **路径**：逐项实际loss对source、Q、K、V、geom-K/V、UAV有效LoRA、alpha的梯度；区分None、精确0、非零。UAV只取P0时没有参与前向的深层参数另标inactive，不归为断路。记录未缩放、未裁剪梯度；不跨不同参数量模块只比总范数。
2. **消费者**：对注入残差retain_grad，比较GELU前后对实际mask loss的梯度RMS/空间分布，正确与常量成对读取。可用固定微小扰动验证自动微分的局部方向，但不做optimizer更新、不据此宣布AP收益。边界/内区/背景诊断分区仅使用训练GT，且应分别定义逐实例边界与前景并集边界。
3. **冲突与内容**：主mask损失、tail损失、aux损失分别计算共享模块梯度余弦；Q模块与source/LoRA分别列出。非零或余弦为负均非充分失败证据，应结合固定CP正确/常量干预及内容训练对照解释。已有AP归因失败结论不因存在新诊断而撤销。

N2训练以同协议L0为基准，保持候选、源坐标、维度、alpha初始化、LoRA和loss不变，只将同一残差从conv_s0输出移至最后GELU输出。要求完整成对内容差和2×2入口×内容交互，见远端N核查。N2自身亦有alpha延迟、固定消费者和候选限制，不能称为完全消除梯度盲区。

本记录为源码与现有门结果核查；G0正式诊断和N2新增训练尚未执行。新增观察使用独立离线artifact，不往现有训练循环加每步高开销遥测。
