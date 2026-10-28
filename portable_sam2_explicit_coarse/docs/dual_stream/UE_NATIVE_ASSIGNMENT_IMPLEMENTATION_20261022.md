# 已审核六条件Native空间赋值诊断：实施记录

执行依据：[修订计划](UE_NEXT_PROBE_PLAN_20261022.md)P1。P2真实来源核查已完成，原8图历史缓存逐值一致；没有确认需要暂停的系统性冲突，但视觉对应均未获独立认证。

## 与历史五条件诊断分开

已有`ue_native_spatial_diagnostic.py`及`ue_native_spatial_diagnostic_20261022/`是较早的五条件（含RMS匹配），与本轮mean/三固定置换并不相同。保留原文件、权重、记录与统计；本轮不从五条件挑选有利结果，也不替换本轮模式。

新代码：

- `scripts/probes/ue_native_assignment_probe.py`：六条件确定性CPU干预、8图原始logits回放门、FP32冻结推理和低优先级owner。
- `scripts/probes/ue_native_assignment_analysis.py`：固定8图残差变化/余弦，六条件完整指标、共同Mask AP bootstrap、固定bbox纠错分母和预定支持/密度分组、完整审计报告。

产物：`/data/wangcheng/checkpoint/ue_native_assignment_probe_20261022`。

## 固定模式与统计

correct /zero_eval /mean_eval /permute_s1 /permute_s2 /permute_s3。
置换是RoI读取前支持内的四维向量联合置换，PCG64 SeedSequence([20261023或24或25, archive_image_id])。
mean为每通道支持内float64均值后转float32，保持孔洞外0。不重训、不做RMS匹配。

主指标Mask AP250，仅correct−mean及correct−三次置换AP均值两个主比较；共同500次图像重采样seed20261021，报告95%和Bonferroni97.5%区间。Boundary/AP75/尺寸/分组/纠错是描述性次要结果，不另挑显著值作确认结论。97.5%每侧尾部约6个样本，披露区间端点粗糙性。

输出新correct全459测试并逐预测核对旧native E10归档，不直接凭8图回放替代全量身份核对。
旧SAT及独立训练native_zero仍是不同问题的控制，不能与zero_eval混读。

## 工程门与资源

合成扰动确定性/联合向量集合/孔洞0及固定8图六条件CPU门已通过。
GPU启动后，先用P2固定8图比较原adapter与新测量hook的原始logits、mask、bbox、score、label逐值回放，确认记录全部调用平方和没有改变预测。
完整测试每图核对GT/框/score/类别，前后卫星完整state SHA一致；记录原cache及输入映射SHA。

owner只在GPU1实际空闲且旧19项test推理的预约结束后占用GPU，不抢占旧训练。
以短生命周期独立CUDA占用进程保持串行诊断之间的可见GPU预约，防止原空闲调度器在子进程间隙重复派任务；CPU分析前释放该进程。
若旧任务未释放GPU则明确排队，不将排队视为已推理或工程门已通过。

最终完整产物为`RESULTS.md`及`complete_audit_summary.json`；不存在前不宣称本轮完成。
空间置换破坏频谱、物理相容性及query相关，结果只支持固定模型空间赋值敏感性；不独立认证精确几何或可重复净收益。
