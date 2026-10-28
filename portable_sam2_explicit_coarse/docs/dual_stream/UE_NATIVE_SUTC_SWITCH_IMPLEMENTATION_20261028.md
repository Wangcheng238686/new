# S/U/T/C 独立开关与纯卫星 UBR

日期：2026-10-28。工程实现；不自动发射新的完整预算训练。

## 1. 本轮核定

原 `B_zero/C_correct` 只有尾部证据开关；两者都始终保留 G、UAV 实例头/loss 和补框。原 `A_nohead` 也有上游 UAV，不是纯卫星。旧 ABC 不可重命名为完整四格。

新合同 `configs/ablations/ue_native_sutc_contract_20261028.json`：

| 组 | U：上游增强包 | T：UBR UAV 证据 | 尾部结构 |
|---|---|---|---|
| S | 关 | 关 | 同一个 native UBR，RGB/几何端口为零 |
| U | 开 | 关 | 同一个 native UBR，RGB/几何端口为零 |
| T | 关 | 开 | 同一个 native UBR，读取独立提取的 RGB/几何 |
| C | 开 | 开 | 同一个 native UBR，读取 G 融合前 RGB/几何 |

UBR 关闭的是 UAV 证据端口，不是整个修正头。S 的 UBR 读取原解码器 32 通道特征和停止梯度的概率，同样由原 mask 任务监督学习残差；不会因为没有 UAV 而整头失效。四组均保留 70 通道输入投影、相同空间模块和零初始化输出。native128、mask supervision512、coarse64 不变。

U 是整体包：早期几何写回、双向交互、实例头及辅助目标、UAV 候选补充。U−S、C−T 不能单独解释为某种注意力或融合算子的效应。

## 2. 实际接线

新增 `rsprompter/uav/native_sutc.py`：不可变开关表、独立 tail installer、P0 证据提取器、S/T 模型封装。新 installer 不把 S/T provider 修改成 G；mask/RoI head 仍沿用已有 native 包装方法，不复制参数。

S 不构造 UAV 编码器、实例头或 UAV 数据 loader，不读取样本 RGB/深度/相机/候选；传入非空 UAV payload 报错。共同初始化文件本身含历史 UAV 权重字典，但 S 不装载这些权重到模型或使用它们运算。

T 只构造 P0 RGB 编码器与 project32，严格加载共同初始化的对应子键。读取顺序保持 C 的 detail 路线：逐视角编码 → project32 → 源 token 无 affine LN → 原连续采样 → 有效候选均值。RGB 支持来自 index>=0.any，几何保留独立 raw512 geometry5 支持。对应仍来自实测深度和相机。未读取/消费候选7维描述；无几何 CNN、无卫星写回 hook、无双向交互、无实例头/loss、无 UAV 补框。

T 编码器固定 eval，仅 LoRA dropout 随 train 切换。仅尾部读取不会在固定权重推理时直接改变 bbox/score 或增加候选；联合训练可经共享卫星参数间接改变检测。

新增 `scripts/probes/ue_ubr_native_sutc_runtime_20261028.py`：

- U/C 委托原 B/C 构造路径，共同源初始化与任务目标不变。
- S/T 用独立模型封装，仅执行原卫星目标。`objective` 经 `network` 调用，可由 DDP 包装，不绕开 DDP。
- `payload` 对 S 返回空字典，对 T 只加载所需 measured RGB-D 对应和 geometry5。
- `inference` 四组均沿用原卫星候选/评分/掩码 consumer，只有 U 开启时允许 UAV 候选。
- 运行该文件只打印合同；没有训练入口或自动排队副作用。不能将旧 ABC runner 的无条件 payload/实例目标代码直接用于 S/T。

## 3. UDPR 退出范围

新四组构建前使用 `native_model_config` 将 `decoder_tail_refiner_cfg.enabled=False`；从共同卫星状态中过滤旧 UDPR 键后严格加载，其余缺失/额外键拒绝。运行时断言旧模块不存在、同构 UBR 存在、证据与补框开关匹配。无新增 UBR loss。

UDPR 从本轮后续 SUTC 的模型实例和优化路径移除。历史实现及旧配置保留供旧结果复现；不修改当前全100轮 ABC 的冻结 worktree，不把其训练中途切换成新合同。物理删除历史源码将破坏旧导入与复现，因此本轮不做跨历史清理。

## 4. 验收状态与资源限制

29 项 CPU 测试通过，覆盖开关表、S 无 UAV 路径、T 的读取等价/梯度/训练模式、独立 installer、原始任务/候选消费路由及既有 native UBR 回归。

另用真实 generic common_init 在 CPU 构建 T 的 P0：44 个共享状态张量逐值相同，可训练参数 33,872，全部为 P0 LoRA/project；无上游模块。收据：`/data/wangcheng/checkpoint/ue_native_sutc_engineering_20261028/real_P0_initialization_CPU.json`。此检查不含真实前向、GPU、DDP 或 AP。

真实 S/T 四卡 AMP 前反传、DDP 空实例、250 实例推理、初始化 SHA、恢复五种 RNG 和资源峰值仍须独立工程门。旧 ABC 门不能替代此项。本轮检查时四卡仅约 1.3–1.4 GiB 空闲，不抢占既有训练，也未发射 S/T 训练。

完整训练发射仍需独立源码快照、四卡 wrapper、按臂隔离目录及新门收据；本文件不认定 AP 或训练复现已通过。
