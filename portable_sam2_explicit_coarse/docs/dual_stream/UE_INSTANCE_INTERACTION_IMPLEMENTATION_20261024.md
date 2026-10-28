# 实例监督内部回写：实施与工程验收

## 当前状态

2026-10-24最新人工核查快照：14个训练臂及45项计划推理均完成，101/119任务完成；其余为统计，12个并发进程存活、无失败。当前owner243139、finish bridge243143；原161872/161891为历史控制面。完整统计及最终审计尚未齐，下文早期进度是按时间保留的历史。

- [固定科学合同](UE_INSTANCE_INTERACTION_PROTOCOL_20261024.md)
- 产物根：`/data/wangcheng/checkpoint/ue_instance_interaction_20261024`
- 初始owner PID81973；当前由显存准入控制面161872接管，子任务明细与实时状态见`state.json`/`launches/`。
- `study_manifest.json`保存23份源码/协议SHA、三个完整初始化SHA、14臂/119任务定义、两套共6120缓存文件SHA清单身份。
- `study_manifest_prelaunch_draft12.json`只是启动前草案；14臂修订先于任何本轮训练/AP，独立复核补入完整RGB-D中常量几何字段控制。

## 独立复核采纳

两位子代理只读审核，未发现必须阻止训练的坐标/反向错误。采纳以下要求：

1. head与internal都含双向交互，唯一开关是S′是否继续进入原卫星编码链；不叫无attention对照。
2. 中心头必须读取S′，工程门单独以中心loss验证回写投影和attention梯度。
3. `satellite_records`在internal时是条件化原路径，不是纯卫星。内容差、入口差同时在该路径与combined上评价，区分实例头贡献。
4. 新的正式disabled路径全459图与历史强SAT逐框/score/RLE/GT一致后，才允许引用历史基线的配对统计。
5. RGB中整图几何donor仍保留目标RGB/anchors，是混合不一致干预；常量几何保留对应/support，不叫无深度。
6. RGB只读P0，只有实际损失可达的LoRA得到训练；无RGB增量不能推出RGB本身无信息。

## 已完成工程证据

|家族/入口|最大GT训练实例|FP32推理实例|峰值reserved MiB|
|---|---:|---:|---:|
|geometry/head|208|250|3294|
|geometry/internal|208|250|4060|
|RGB-D/head|208|250|4826|
|RGB-D/internal|208|250|5400|

每格`gate.json`核定FP32/AMP零回写原始logits恒等、非零回写权重且无支持时恒等、三步有限梯度、直接中心梯度可达、卫星参数与buffer SHA不变。RGB另检查有效LoRA梯度与原编码参数冻结。

`cpu_gate.json`核定U128名义中心4i、2×2池化中心8j+2、align_corners=False上采样的内部线性ramp精确、边缘延拓与3×3邻域不跨行；这不是外部三维标定认证。

每格`extended_gate.json`核定同一非零权重切换入口时中心输出完全相同、block5输入只在internal改变、stride4不变且后续预期消费者改变；正式推理函数的零回写前缀与无支持完整结果恒等。

### 实际续训重放边界

实际AdamW/AMP状态经磁盘保存恢复，Python/NumPy/Torch/CUDA/DataLoader generator均恢复；下一轮真实两worker loader前4图顺序和增强输入SHA逐值一致。CUDA原算子反向有数值噪声，权重更新不是逐位相同；独立三次同代码内存重放标定后，磁盘恢复差异均在2倍标定界内。原始失败与修正记录保留在各gate日志，不删除失败历史。

## 运行与交付纪律

owner按实测峰值＋2048MiB余量共租，不抢占现有attention任务。每轮保存全部新增参数/optimizer/scaler/五类RNG与实际2136图顺序；E10 last固定，非有限/AMP跳步会明确失败。失败不按AP解释，不取消其它活跃任务。

全部执行任务完成仍不等于科学结论完成：需运行`scripts/analysis/ue_instance_interaction_audit_20261024.py`，核查真实14个CP/2670步、初始化、配对顺序、有效optimizer状态、冻结RGB原参数、全部459图输出、来源及四指标统计校准，再报告内部回写、几何内容和RGB各自的证据。

本轮仍是冻结E73探针，最终共同基础初始化、全预算联合训练及独立区域泛化均未被本轮工程验收证明。

## 自动收尾已接通

`ue_instance_interaction_finish_20261024.py`绑定owner PID81973及实际进程起始ticks，独立PID84334等待执行完成；失败/owner意外退出会明确停止收尾并保留状态，不凭旧state文件认定仍在运行。结束后先执行冻结严格审计，再重核6120缓存内容SHA、各评价实际权重SHA及完整训练合同，生成机器汇总与报告。自动产物完成不代替前台最终证据审阅。

报告接线在临时fixture上核定14臂、60项四指标比较和四组证据字段，临时文件已删除；这不是真实AP或实验完成证明。训练与原119项任务合同没有改变，新增收尾源码单独归档。

## 首轮实际检查点核验

内部回写真实/常量几何两臂已完成E1并进入E2。`epoch1_internal_pair_audit.json`实际加载两个`epoch1.pth`，核定epoch=1、267次AdamW更新、五类RNG齐全、所有新增权重有限；2136图唯一样本顺序与完整初始化指纹逐值匹配，卫星冻结SHA保持原值，AMP scale均为512。

按2136个microbatch等权平均，直接实例loss为真实2.000738、常量2.268973；参与反传的原任务loss分别0.201356/0.201046。两臂在各自训练路径上统计，未做固定权重反事实评价；这里只记录训练状态与合同，不能把loss差当成泛化内容收益或实例AP证据。预算与E10 last未改变。

随后head两臂也完成E1，`epoch1_geometry_factorial_audit.json`实际核对完整几何四格：每个CP的AdamW状态均为267步、同一初始化、2136图顺序跨四臂逐值一致，五类RNG、有限权重、原SAT SHA和AMP512均通过。head真实/常量中心loss为2.001560/2.261342；其`original=0`表示没有原任务可导路径，不表示原任务错误为零。四个几何臂均已进入E2，RGB阶段及后续完整验证仍待完成。

### RGB首轮实际可达参数核验

`ue_instance_interaction_rgb_path_audit_20261024.py --epoch 1`已核查当时已落盘的3/6个RGB臂（internal正确RGB、internal常量RGB、head正确RGB）；未落盘的三个臂在产物中明确列为missing，不称全部完成。三者实际267次AdamW更新、2136图顺序与几何臂完全相同，原source非LoRA状态逐值不变。

P0读取对应的实际更新范围为`patch_embed.proj`和Hiera blocks0/1，10个LoRA张量共25648参数；声明可训练的source LoRA总量为1061424。优化器状态证实更新，不证实RGB内容收益。源后续stage未参与损失路径，不能声称整个UAV编码器被训练，也不能用本读取方式的负结果否定RGB全部信息价值。早期审计把patch_embed LoRA一并纳入，避免仅列blocks0/1而漏报。

## 训练全部启动与资源准入修订

按用户此前显存允许即共租的授权，原每卡2个本轮进程上限改为实测工作集准入。新控制面161872接管原12个训练/统计子进程，保留PID及起始ticks；训练和分析均未被终止。两个seed45 head臂在GPU2/3追加，并均已完成25次实际更新。14个训练臂至此全部启动，5个已完成。

工作集采用同家族/入口已完成epoch最大allocator峰值与工程门峰值的最大值，加512MiB上下文预算；每卡统一保留2048MiB余量，并为尚未实际分配的在途进程预留工作集。两个新增head的准入工作集均5466MiB。没有修改23份冻结源码/协议、任何初始化、随机种子、预算或评价命令；仍为119项固定任务。

`capacity_handoff.json`记录接管与资源政策；原控制面/finish身份、finish v1源归档`control_history/`。finish v2 PID161891绑定新owner，终点附加审计检查新控制面源SHA。原81973/84334均为历史控制面PID，不再用它们存活与否判断训练状态。

首个RGB E10（head正确RGB）也已完成实际可达参数审计：2670次有效更新，source非LoRA状态逐值不变，P0有效LoRA范围仍25648参数；产物`epoch10_rgb_effective_path_audit.json`明确仅1/6臂可用，不称全部RGB完成。


## CPU统计准入接管

新增analysis_owner/analysis_finish脚本，48核及约116GiB可用内存下统计并发4→12，4GiB/任务工作集与8GiB主机余量。仅当剩余未启动任务全为analysis时接管；原子进程按PID/起始ticks保留、119命令与科学源不改。analysis_capacity_handoff.json记录新控制面，第一次capacity_handoff保留；旧启动身份归档control_history/pre_cpu12_*。新bridge先核CPU控制面源与119计划，再调用未改的原finish --audit-only，完成后附加新接管审计SHA。
