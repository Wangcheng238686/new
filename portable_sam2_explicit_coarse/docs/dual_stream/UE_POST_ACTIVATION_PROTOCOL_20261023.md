# N2后激活注入与G0离线梯度诊断

## 工程与冻结方案

产物根：`/data/wangcheng/checkpoint/ue_post_activation_20261023`。

- `rsprompter/uav/post_activation_probe.py`继承AttentionLadder的L0，state_dict与共同初始化完全相同；N2仅把receiver由conv_s0移至decoder最后GELU输出，拒绝重复attach。
- `scripts/probes/ue_post_activation_probe.py`为独立runner，沿用ladder训练/FP32评估/资源门；新路径、study、injection_site进入合同。旧source未修改。
- `scripts/probes/ue_post_activation_schedule.py`为独立owner：正确/常量两臂、基座、val/test、成对bootstrap及N2 E1/E10离线诊断。仅GPU1/2/3，每卡本owner一个GPU子任务；失败保留returncode并停止新准入，不中断原任务。
- `scripts/probes/ue_gradient_topology_audit.py`不更新权重，固定训练密度分位.125/.375/.625/.875四图，各做正确/常量RGB，FP32、无增强/dropout。逐实际loss分别计算模块梯度、receiver/最后激活输出梯度及损失间余弦，区分None/0/非零。卫星及adapter张量指纹前后不变。

共同初始化直接严格读取ladder的common_init.pth；不是继续训练某个已训练臂。EXP-0017/v1 split不变。10epoch、2670更新、有效batch8、AdamW LR1e-4/wd.01、warmup50、cosine、clip.5、AMP scale512、无EMA、E10 last；每epoch相同1044000+epoch重设，2136训练microbatch。两臂深度/相机/支持保留，唯一内容差为RGB正确/常量。

test预测为基座disabled、正确训练CP的correct/constant/donor、常量训练CP的constant；两臂各val。固定五比较：各自对基座、训练内容差、同CP的constant/donor干预。要求459图GT/bbox/labels/scores/顺序精确一致，MaskAP250/BoundaryAP2%/1%，500次同seed20261021图像配对bootstrap。analysis目录提前创建。

L0来自仍在运行的ladder队列。共同初始化、优化配方、样本顺序设计相同；最终须核对实际合同和各epoch顺序，计算 `(N2_correct−N2_constant)−(L0_correct−L0_constant)`。不得拿旧A或远端b0替代，不通过比较各自CI是否跨零来判断交互。独立CPU等待器 `ue_post_activation_interaction.py`（PID3359374）已启动，等待四臂完整后核验合同/10epoch顺序/检测输出，直接计算三指标500次四臂配对交互及保存向量；不占GPU。

## 资源门与启动

gate/gpu_gate.json已通过：FP32/AMP零alpha原始logits/masks/bbox/scores精确回放；非零alpha无支持精确回放；最大GT样本三步反向source/有效encoder LoRA非零；250RoI FP32有限；卫星参数/buffer精确不变。

训练峰值allocated3095.63/reserved3334MiB；250RoI FP32 allocated2617.25/reserved3334MiB。准入空闲门为3072+ceil(1.25*3334)=7240MiB。门中临时训练不会写入正式初始化。

独立owner3357860已脱离会话；首次correct在GPU2、constant在GPU3、baseline在GPU1。两条正式训练均已完成update10，loss/grad有限；实际后续状态以state.json及日志为准，不将启动/显存等同于效果。

## G0首份结果：L1 E10

gradient_L1_E10/gradients.json完成4图×2输入=8次前反向，optimizer_steps=0、参数指纹不变。实际反传loss_mask与loss_decoder_tail均能到source/Q/K/V/有效LoRA，排除全支断路。

正确RGB四图：tail对encoder LoRA的组RMS为mask的约4.6–11.2倍，对source约5.1–27.4倍。部分图在source/value/output或encoder方向相反，非全部样本冲突。这支持继续检查目标梯度平衡，不能据此认定tail导致AP失败或立即调整权重。梯度范数也不是AdamW实际更新量。

同一实际前向的GELU前receiver梯度RMS为最后GELU输出的约0.48–0.53。局部导数衰减不是N2收益预测；移动入口同时改变前向表示和训练轨迹。

空间分区是训练GT每实例1px内边、并集1px内边、前景内部和背景，最近邻读取到RoI网格。包含支持/无支持位置与重复RoI，不是GT覆盖率，也不是Boundary AP2%/1%带宽。L1含连续读取，不能代替纯L0入口对照。

## 尚未完成

N2两臂训练/完整AP、队列中的N2 E1/E10梯度诊断、L0齐备后的入口×内容交互。v1 test已是开发集；图像bootstrap不覆盖跨seed方差/空间相关。

本轮无新辅助loss、调权、消费者解冻或训练每步遥测。所有新增梯度观察均为独立离线artifact。

## 后续源码核查：UDPR捕获顺序

SAM2DecoderWrapper构建时先在act2注册_capture_upscaled_embedding，N2 attach随后在同一act2注册替换output的hook。按PyTorch hook顺序，wrapper保存的是N2残差注入前张量，hypernetwork矩阵乘法得到的是注入后张量。因此现N2改变原始mask logits，并经这些logits影响UDPR选点/属性/损失，但UDPR的upscaled feature通道仍读取注入前版本。

旧conv_s0注入在act2之前，其增强特征会进入wrapper捕获，故UDPR已间接消费旧路UAV特征。不能把当前N2概括为“所有下游消费者共享同一增强高分辨率特征”。该差异限制纯GELU位置归因，入口交互仍描述实际这两条接线的差值。正在运行的source不改；共享增强特征或UDPR显式点证据读取须另立版本/对照，不能静默修hook后接续本组。
