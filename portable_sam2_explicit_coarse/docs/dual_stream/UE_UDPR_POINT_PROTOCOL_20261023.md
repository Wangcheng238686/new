# S1：UAV证据进入原UDPR不确定点特征

> **已作废并停止（2026-10-23）**：本文件下文“全图mask网格”合同错误。实际roi_local监督与bbox paste要求通过proposal框映射选点。旧训练/点级诊断不能用于正确对齐UAV结论；源码、日志保留供审计。修正与新训练见 [S1-R与N2-shared](UE_CONSUMER_BLINDSPOT_PROBES_20261023.md)。以下为历史记录，非当前有效协议。

## 实现与范围

产物：`/data/wangcheng/checkpoint/ue_udpr_point_20261023`。

`rsprompter/uav/udpr_point_probe.py`只hook原UDPR的forward输入。原卫星模型生成raw logits、mask token、高分辨率特征；原_select取K=64点，不改变规则。对这些点的32维原特征做线性query，在同一卫星坐标的8个几何候选上读取5视角UAV特征，用32维cosine attention聚合。源特征归一化、深度/几何K/V与已有Unit基础路径一致，RGB采用整数token读取；没有增加L1连续读取、局部偏移或新损失。

`f_point_new = f_point_sat + supported * W(LN(pooled_UAV))`。W为32→32线性映射，weight/bias均零初始化；不在W后再加LN。不引入第二个可训练alpha，保留alpha=1的冻结参数供回放门。W首步有梯度，source/Q/K/V及UAV LoRA在W打开后获得梯度。原image-query卷积不使用并冻结，新增point_query使用当前实例的原UDPR高分辨率点特征。

只向原高分辨率张量的选定位置scatter_add，原raw logits/tokens保持原对象，随后原UDPR照常读取这些点并写回有限logit残差。卫星及UDPR全部原参数冻结，没有密集UAV注入、第二个mask头、decoder LoRA或辅助loss。不能将S1与N2之差称为只移动同一入口：S1同时使用实例点query和稀疏读取组织；其主要归因对照是同结构的正确/常量训练。

## 坐标与深度合同

实际SAM2 wrapper重复完整图像embedding，无RoI裁剪；UDPR点索引在全图mask网格。图像边缘坐标为 `u=(col+.5)*512/W, v=(row+.5)*512/H`，再定位128²候选格。不得根据bbox缩放点坐标。CPU检查穷举128²/256²网格；128²时mask格与候选格逐一对应。

深度通过已冻结的投影候选及几何属性参与K/V，不预测新深度、不使用UAV标签。常量RGB训练仍保留深度/姿态/候选支持，因此它不是“完全没有UAV信息”。

## 固定训练与评估

独立runner `scripts/probes/ue_udpr_point_probe.py`，独立owner `ue_udpr_point_schedule.py`。严格读取本组common_init.pth，两臂同一随机初始参数；卫星EXP-0017/v1 split不变。10epoch×267更新，2136 microbatch/epoch、batch1累积8、seed45初始化、每epoch1044000+epoch、AdamW LR1e-4/wd.01、warmup50/cosine、clip.5、AMP scale512、无EMA、固定E10 last。正确与常量RGB分别训练，不热启动另一臂。

完整test：基座disabled、正确训练CP的correct/constant/donor、常量训练CP的constant；各臂完整val。5个固定统计比较：两臂对基座、正确训练−常量训练、正确CP下correct−constant/donor。MaskAP250、BoundaryAP2%/1%，500次seed20261021配对图像bootstrap，严格核对459图GT/bbox/labels/scores/顺序。v1 test为开发集，单seed结果不涵盖训练方差。

## 点级有效性诊断

仅评估时，在同一次forward、同raw logits/选点/token上，额外直接调用原UDPR.forward，用未增强的卫星特征计算参考写入；直接调用绕过hooks，不递归。比较原UDPR与UAV条件化UDPR的refined_selected_logits，以原部署boundary_logit二值化，记录corrected/damaged/changed和old/new_correct。无任何optimizer更新。

每个预测RoI通过最大GT bbox IoU≥.5关联实例，GT按选中全图格中心最近邻读取；未匹配RoI排除。重复RoI/点重复计数，属于局部诊断而非一对一实例匹配或AP；不涵盖漏检/未选点。一个图像的计数在所有mask head调用累积，其他latest_forward统计仍仅最后调用。恒等核对：new_correct−old_correct=corrected−damaged。

## 门与启动证据

- CPU gate：全图坐标、零W恒等、非零W/bias无支持恒等、仅选点写入、raw logits/tokens与选择索引保持、source/query梯度非零。第一版测试fixture每格只有一个候选，Q梯度必为0；修为两个不同候选后通过，未改变模型。此为测试fixture修正，无正式训练被重跑。
- GPU gate：FP32/AMP零alpha与零W原始mask logits/masks/bbox/scores精确回放，非零W/bias无支持恒等；最大GT样本三步反向；250RoI FP32；原卫星参数/buffer不变。
- GPU梯度step0 W norm=.02617，point_query=0；step1/2 point_query梯度非零；有效encoder LoRA检查通过。没有把零W第一步上游0误判成断路。
- 训练allocated2776.52MiB/reserved3044MiB；250RoI FP32 allocated2340.74MiB/reserved3044MiB。准入至少6877MiB空闲，仅GPU1/2/3，每卡本owner最多一个任务，不中断已有owner。
- 单张val评估诊断门：零W时1280个匹配点均无额外改变；临时非零bias产生16个改变，恒等关系正确。临时bias不是训练权重，其方向/效果不构成任何收益证据。
- owner3378228脱离会话启动，correct在GPU1、constant在GPU2、baseline在GPU3；两条训练均已完成真实参数更新且loss/grad有限。初始指纹及除内容/运行参数外合同逐项相同。后续PID/状态以state.json/日志为准。

## 文件

- 模型：udpr_point_probe.py。
- 独立训练/评估/资源门：ue_udpr_point_probe.py。
- 解析坐标/写入门：ue_udpr_point_cpu_gate.py。
- 点级评估门：ue_udpr_point_evaluation_gate.py。
- 独立队列：ue_udpr_point_schedule.py。
- CPU只读汇总：ue_udpr_point_summary.py（PID3379439），等待完整test后将图像累计点级诊断汇总为point_summary.json，不占GPU。

没有改动原SAT/UDPR源码或正在运行的N2/L系列数值source。本组训练不加逐步高开销诊断，点级参考forward只在离线评估开启。
