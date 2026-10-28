# 几何机制8臂：工程验收与启动

## 已通过的工程门

四个sampling/fusion组合均通过同一真实冻结基座GPU门：
FP32及FP16 AMP零alpha原始mask logits逐值恒等；非零alpha无支持恒等；difference零字段恒等。
实际mask/UDPR损失3步梯度通过，alpha及mixer可更新；卫星所有参数与buffer SHA不变、无梯度。
资源门选train最多208实例图；实际AMP训练正反向，以及250RoI FP32沿32分块正式mask路径均通过。
CPU门通过旧模型逐值等价、采样/支持、差分/阶跃/边缘/越界及checkpoint反传。

| 配置 | AMP训练峰值allocated MiB | 最大reserved MiB | FP32 250RoI峰值allocated MiB |
|---|---:|---:|---:|
| bilinear_concat | 2380.82 | 2606 | 1922.35 |
| bilinear_difference | 2384.20 | 2568 | 1923.85 |
| nearest_concat | 2380.82 | 2606 | 1922.35 |
| nearest_difference | 2384.20 | 2568 | 1923.85 |

这些是本进程PyTorch allocator数值，不是整卡占用或所有数据长期训练峰值。
调度要求当前空闲至少3072MiB+ceil(1.25×实测reserved)，约6282–6330MiB；每GPU本owner最多一个新任务，允许外owner共租。
不声称能绝对保证其他owner异步启动时仍不OOM，资源准入逐次刷新，失败诚实记录、不静默缩batch或截断实例。

## 冻结计划与启动

完整协议见[固定8臂协议](UE_GEOMETRY_MECHANISM_PROTOCOL_20261022.md)。
产物根`/data/wangcheng/checkpoint/ue_geometry_mechanism_20261022`，manifest.json及job_plan.json在任何新test输出前已冻结。
8个训练，20分支test推理+1新基座test，8个val诊断，24项配对比较，最后实际CP/更新/样本顺序/身份/预测验收。
四个真实几何配置先训练，匹配constant控制随后按固定顺序训练；无val/test指标启停筛选。
新基座test复现只检验固定已知基座，不参与结构选择。
owner PID2923487为独立持久进程；实际最新状态以state.json、活PID/命令身份及train.jsonl为准。
训练成功启动不等于完成或获得指标收益。最终以complete_audit_summary.json和FINAL_REPORT.md为准。

## 文件索引

- rsprompter/uav/native_geometry_mechanism.py：因子模型，19681参数。
- scripts/probes/ue_geometry_mechanism_cpu_gate.py：CPU工程门。
- scripts/probes/ue_geometry_mechanism_probe.py：独立初始化、GPU门、训练、显式输入评价。
- scripts/probes/ue_geometry_mechanism_schedule.py：共卡显存准入、固定矩阵、进程身份、失败不自动重试。
- scripts/probes/ue_geometry_mechanism_collect.py：实际8×E10、完整合同、全部更新/源快照/样本顺序、评价及比较审计。

工程通过不认证外部表面可见性，也不保证nearest或difference具有AP收益。

## 实际启动核对

独立owner已PPID1运行，基座test在GPU1执行，bilinear_concat_geometry/GPU0、bilinear_difference_geometry/GPU3、nearest_concat_geometry/GPU2均实际产生train.jsonl更新（首轮至少8次），loss/梯度有限、AMP scale512。nearest_difference及四条constant按固定计划排队，非等待空卡；当前仅每GPU本owner一个槽。实际PID/cmdline与首批更新记录在launch_verified.json。此记录是启动证据，不是训练完成证据。
