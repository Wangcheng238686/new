# 几何采样与消费机制：8臂固定探针合同

用户授权子代理并行实施，显存充足可与现有探针共卡，不抢停既有任务，避免OOM。此轮不根据结果增减训练臂或选测试最佳轮次。

## 因子与估计目标

| 因子 | 水平 |
|---|---|
| 几何采样 | bilinear：原公式；nearest：最近像素，未知不填洞 |
| 残差 | concat：H(Q,G)；difference：H(Q,G)−H(Q,0) |
| 训练内容 | geometry；constant：训练有效像素全局固定四通道均值 |

共8独立训练，所有参数数目均19681、同common_init。卫星query采样始终bilinear；有效支持始终原nearest support，采样因子仅改变几何取值。
原四字段及native缓存版本不变，未顺便增加法向/相机z通道或更改投影。这只检验现有表面插值与可绕过几何的消费问题，不认证真实可见性。

constant固定为[.27310444867236794,−.0013918533714920775,.0033367500561123213,.4986884076865656]，来自train2142图486813240个有效像素，按像素加权。support保留、无效像素0，训练/评价都不借助GT生成字段。constant不是彻底无UAV，非零常量允许模型做条件卫星适配，是difference路线必须保留的容量控制。
统计来源逐cache SHA及split/annotation文件SHA在产物common_constants.json，未用val/test统计。

## 训练固定合同

- sealed EXP-0017 E73完整卫星冻结，SAM2/检测/CSPM/UDPR保持eval；仅新native模块训练，无RGB编码器，无额外loss/新修边头。
- v1数据2142train/459val/459test，coarse64，maxDets250，非EMA。原高斯与ColorJitter增强不变。
- 10epochs，2136参与更新/epoch，batch1累积8，267steps/epoch，固定2670steps，E10 last。每轮随机丢尾6图，不按指标提前停止。
- AdamW lr1e−4 wd.01，warmup50，cosine到.1倍，clip.5，AMP初始scale512；评价FP32原scores。
- 参数初始化seed45，训练seed44；本轮E1–10统一epoch_seed=1044000+epoch，所有8臂同数据/增强流，并保存实际样本顺序与完整RNG。
- 与旧native恢复实验RNG协议不同，因此旧结果仅历史参照，不当本轮单因素对照。

## 工程门与遥测纪律

CPU门已覆盖旧bilinear+concat逐值等价、支持不变、阶跃nearest不平均、边界/越界/未知与0高度区分、difference非零alpha下zero精确恒等及checkpoint反传。
GPU门四配置全部完成后才train：FP32/FP16 AMP原始logits零alpha恒等；非零alpha无支持恒等；difference zero字段恒等；真实损失梯度及冻结参数/buffer SHA；训练图选择train GT实例数最多（非AP选择）；强制250RoI FP32沿32分块正式路径。
记录本进程peak allocated/reserved、整卡及进程显存。训练沿用原lastmicrobatch/latest_forward语义，不加逐步新字段；sample_order/完整冻结状态/初始化身份是epoch离线审计。

每GPU本owner至多一个额外任务，可与其他owner共租。准入要求当前空闲≥3072MiB+ceil(1.25×对应配置最大reserved峰值)。统计实际全部进程占用，不用“卡非空”阻止启动；PID/命令身份核对防重复。OOM/非零退出诚实终止该臂，不静默降低batch/实例上限。

## 固定评价与归因

所有8臂完整test主输入：geometry臂correct，constant臂constant。geometry四臂另做constant/shuffle/zero，共20分支推理。
复用旧卫星baseline前必须核对权重、数据、GT、scores、max250和FP32身份；否则重新基座评价。
四个匹配训练内容差为主要归因；八臂各对卫星比较。额外correct/constant/shuffle固定检查点比较是依赖诊断，不代替训练控制。shuffle只在支持内联合置换四字段，非物理场景，保留分布外限制。
MaskAP主指标，Boundary2%/1%辅助，同图500bootstrap seed20261021；单seed/空间相关/多路线探索名义区间限制保留。
nearest−bilinear、difference−concat及交互应区分真实内容与常量内容；若只有点差就明示无交互区间，不能比较两条CI来宣称交互显著。
difference zero恒等由结构保证，不可把zero掉点当内容成功。只有真实内容优于匹配常量，并有相对卫星净增益，才能支持几何消费；一次seed仍只是机制证据。
不按test选结构或权重；本轮test仍开发域，正式结论另需独立区域/训练重复。

## 产物与完成标准

产物根`/data/wangcheng/checkpoint/ue_geometry_mechanism_20261022`。模型native_geometry_mechanism.py、CPU门、probe、schedule及collect独立保存，不改旧任务。
实际计划、source快照、共同初始化、constant SHA在首个test输出前冻结。
最终必须核对实际8×E10权重/2670更新、各轮样本顺序、训练内容/算子/初始化/源码合同、全部459图预测及检测/GT一致、全部计划分析后生成完整审计与报告。工程门或训练启动不计收益。
