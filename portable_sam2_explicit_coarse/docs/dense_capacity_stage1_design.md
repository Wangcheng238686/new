# Dense-only Residual Capacity Adapter：阶段一冻结可行性规格

日期：2026-09-11。状态：已接线；阶段一 dev100 于 2026-09-12 启动中，不属于既有
matrix300 默认矩阵。

## 目的与非目的

本实验只检验：在 A0 的点、box、raw-logit canvas 编码、PromptEncoder、SAM2 decoder
均不变时，一个更有容量的 **dense-only** source residual 是否能提高 canvas 内容和最终
segmentation。它不宣称改进共享 coarse head，也不重跑或改写 P/PB/A0/A3。

## 前向与梯度契约

设原 ShapePriorInjector 为 `H0`，RoI feature 为 `x`：

```text
z_point = H0(x)                              -> ShapePointMiner -> 2P2N
z_dense = stopgrad(z_point) + H_delta(stopgrad(x))
                                             -> existing paste/gate/PE/decoder
```

`H_delta` 是 `SmallMaskDecoder(512, 256, 1, 64)`；最终 3×3 output convolution 零初始化。
故启用模块的 step-0 `z_dense == z_point`，并且对原 A0 的 point source、RoI feature 和
其它参数均无 adapter gradient。R3、P2、UDPR 均禁用；不改变 64-grid、proposal box、
2P2N、raw logits、dense gate 或 mask-downscaling。

训练只允许 adapter 参数可训练，且模型整体置 eval、仅 adapter 置 train，避免冻结 A0 的
BatchNorm buffer 漂移。loss 为既有 final mask loss 与既有 64-grid coarse BCE+Dice 的
adapter 输出版本（固定权重 0.10）；没有新损失类型。
无 positive RoI 的 DDP rank 仍返回每个 adapter 参数相连的零 `loss_dense_capacity`，确保
其 reducer 路径与非空 rank 一致。

## 启动与验收

入口为 `vhr10_p2v2_dev.sh densecap64`，从 matrix300 A0 E300-last 以显式跨架构 INIT
加载。加载白名单仅允许 `dense_capacity_head.*` 缺失；任一其他 missing、unexpected、
shape mismatch 或 migration 均失败。该 arm 不登记进 `matrix300.sh` 或默认 series。

启动前必须通过：

1. CPU tensor smoke：零残差 identity，adapter 梯度不泄漏到 `x` 或 `z_point`；
2. 真实模型 contract audit：adapter width、detach 与旧 arm config 均正确；
3. A0 checkpoint heat-start：仅 adapter state key 缺失；
4. step-0 真实 validation 前向的 detector、points、dense canvas 和最终 prediction hash
   与 A0 恒等。

第 4 项由 `scripts/smoke/preflight_densecap64_step0.py` 执行；它同时从同一 A0 checkpoint
构建 A0 与 densecap，严格要求 state-dict 缺失项恰好为全部
`roi_head.mask_head.dense_capacity_head.*` keys。2026-09-12 在 GPU0、A0 matrix300
E300-last、首个真实 validation batch（2 images）通过：18/18 adapter keys 缺失且无
unexpected，real-RoI residual 为零，detector、PE point slots、dense canvas 和所有最终
prediction fields 均逐位恒等。该记录仅证明 step-0 接线，不是训练或效用结果。

训练后应在同一 NWPU-130 评估并导出 records。继续到正式 matrix 的必要条件是：
`z_point`/point slots 恒等、natural support-canvas IoU 的 paired CI 正向、最终 mAP 的
paired CI 正向。否则停止该路线。

正式 matrix300 注册成对的从头训练行 `a0_sg` 与 `a0_sg_densecap64`，两者均设置
`SHAPE_DENSE_DETACH=1`，使原 coarse source 均不接收 final-mask 梯度。前者是唯一的
source-gradient-isolated A0 control，后者只额外启用 H_delta。因此二者的 paired mAP 是
DenseCap 的唯一干净正式比较；不得用后者直接减通常 A0。默认 P/PB/A0/A3 序列仍不改。

正式行还将 `DENSECAP/roi_count`、`delta_abs_mean`、`delta_nonzero_ratio` 同时写入
`PROMPT_DEBUG_STATS_INTERVAL` snapshot 和跨 DDP epoch 均值，避免将通用
`DENSE/*` gate 读数错误归因于 H_delta。

## 运行记录

- 2026-09-12：启动 `vhr10_p2v2_dev.sh densecap64`，输出目录
  `/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/vhr10_p2v2_dev100_densecap64_a0e300init_tr1.0_va1.0/`，
  日志 `logs/ablations/vhr10_p2v2_dev100_densecap64_a0e300init_tr1.0_va1.0_20260912_000343_pid2444951.log`。
  启动审计为 18 个 adapter tensors 可训练、其余 A0 参数 0 trainable；A0 E300
  heat-start 为 loaded=665、missing=18、unexpected=0、shape-mismatch=0，首次非零 LR
  update 的跨 DDP parameter deviation=`0.000e+00`。运行中，尚无效用结论。

- 2026-09-12：在不干预上述 dev100 任务的前提下，已 arm 持久队列
  `scripts/ablations/queue_vhr10_a0_sg_densecap.sh`（tmux 会话
  `densecap_matrix300_queue`）。它在 GPU 0--3 全部空闲后先重放 matrix300 DRY_RUN，再依次
  启动 `a0_sg` 与 `a0_sg_densecap64`；队列日志为
  `/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/vhr10_a0_sg_densecap_queue.log`。

- 2026-09-12：用当前工作树的 `check_forward_equivalence.py` 从历史 A0 matrix300
  E300-last checkpoint 的内嵌 `model_config` 严格重建模型并严格加载 665/665 model keys；同一
  validation batch 连续两次 forward 的 bbox、label、score、mask hash 逐位一致。产物为
  `refactor_golden/a0_matrix300_current_code_forward.json`。这证明当前推理可忠实消费历史
  A0 权重；它不替代对训练期代码/优化动态的单独等价性证明。
