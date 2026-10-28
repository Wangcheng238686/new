# Attention L1/L2/L3首发：工程验收与队列

## 已完成工程验收

CPU九项解析/数值门覆盖L0–L6共同初始化、strict state、归一化基准等价、源坐标实际PatchEmbed中心delta核、L2分视角聚合、L3真正拒绝/梯度、L4分离几何及L5卫星Z偏置。
L6 decoder LoRA另由runner实现，未做GPU资源门，不将CPU门当L6显存证明。

L1/L2/L3通过实际冻结SAT CUDA门：FP32/FP16零alpha原始logits精确恒等、非零alpha无候选恒等，实际208实例train图三步梯度可达source及UAV LoRA；L3 null_logit梯度非零。SAT原参数与buffer完整SHA不变、无梯度。
250RoI FP32按正式32分块路径通过。

| 方向 | AMP训练峰值allocated MiB | 最大reserved MiB | FP32 250RoI allocated MiB | 首发卡 |
|---|---:|---:|---:|---:|
| L1 | 2877.29 | 3380 | 2424.49 | 1 |
| L2 | 2873.53 | 3228 | 2420.02 | 2 |
| L3 | 2873.63 | 3444 | 2420.62 | 3 |

上述为本进程allocator，不是整卡；按当前整卡全部进程占用准入，训练需空闲≥3072MiB+ceil1.25×对应reserved峰值，L1/2/3分别7297/7107/7377MiB。工程门开始时三卡空闲约8053–8153MiB，实测无OOM。
L4/L5/L0普通资源门待首发槽释放后安排，至少8GiB空闲；L6资源门单独至少12GiB空闲，不足等待。
每卡本owner最多一项，GPU0禁止。不会因为别处有空闲就额外叠加其他训练。

## L1映射核定

实际导入sam2位于`/home/wangcheng2021/project/new-uav-dual/sam2/`，实际Hiera patch/neck源文件及SHA保存在每个gpu_gate的source_lattice字段。
核定PatchEmbed kernel7/stride4/pad3，第i格中心源整数4i，FPN0 lateral1×1且不含topdown重采样。
L1 grid=(source_uv+2)/256−1，align_corners=False，border；先沿Unit对token作LN再读取，不增加offset或源后归一化。
该合同不认证世界配准，不更改旧运行任务。

## 工程修复记录

第一次GPU门只在模块匹配自检失败：PEFT同时暴露base_layer/lora_A/lora_B卷积，自检误计三条主patch投影。未进行正式前向/训练/test。
只修为排除lora_卷积的base投影匹配；旧manifest、source和失败gates完整归档engineering_attempt1_patch_module_match。
修复后在任何训练/test输出前重新冻结，再跑三项门，均通过。非指标驱动修改。

## 固定队列和证据

协议：[固定六方向合同](UE_ATTENTION_LADDER_PROTOCOL_20261022.md)。总103项：14训练、4后续资源门、43推理（29test含baseline、14val）、41配对比较及最终验收。
owner PID2964566独立PPID1运行；首发L1correct/GPU1、L2correct/GPU2、L3correct/GPU3。
主训练固定E10/2670step；后续常量对照/L0共同基准/L4–L6等待slot与显存门，不按AP筛选。
资源registry允许后续gate追加，但每次admission冻结实际gate身份和整卡快照；训练源码/模型/初始化/数据/分析计划已注册冻结。

产物根`/data/wangcheng/checkpoint/ue_attention_ladder_20261022`。初始CPU/GPU门、初始化及manifest已存在；源码改动在正式训练后禁止静默兼容。
最终以actual14CP、2670行/arm、每轮实际样本顺序、冻结卫星/L6新LoRA、全部459GT/原det/scores及41项比较核对为准。
启动/工程通过不等于有收益，当前没有本轮E10任务指标。

## 实际首批训练核对

L1correct/PID2964589/GPU1、L2correct/PID2964594/GPU2、L3correct/PID2964599/GPU3均已实际产生至少4条优化更新，loss/grad有限、AMPscale512，无OOM。owner PPID1，命令身份/准入门/源码SHA与首批更新保存在launch_verified.json。三卡整卡约19.7–19.8GiB，含所有旧任务；GPU0未追加。本记录是启动核定，不是E10完成或收益结论。
