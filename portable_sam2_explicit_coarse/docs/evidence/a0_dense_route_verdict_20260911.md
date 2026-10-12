# A0-last dense 路线裁决：gate 与 source-resolution

日期：2026-09-11。底座为 matrix300 A0/PBM 的 `last_model_epoch300.pth`，NWPU
VHR-10 validation 全 130 图、十类 COCO 合同。本文是冻结诊断与下一步准入记录，
不是新的训练结果或论文精度声明。

## 1. 共同不变量

所有 A0-last 前向均 strict-load 同一 checkpoint。unhooked standard、hooked
learned/current 的完整 mask 输出 SHA256 均为 `b32c…816c`，检测 SHA256 均为
`9ff8…62f4`，并与 production inference 的 COCO records 精确一致。所有干预保持
proposal、bbox、类别、分数、排序和 130 个执行 image ID 不变。

## 2. 实例级 dense gate：Oracle 通过，train-only readout 失败

current gate 为 0.6738；强制为 1.0 使 mAP 从 0.633074 降至 0.630163。对 714 个
同类 IoU>=0.5 匹配 detection，GT 后验 selector 在 339 个检测中选择 alpha=1，并得到
0.636589。三项 500 次 image-paired bootstrap（seed44）均通过：

| treatment - baseline | Δ mAP | 95% CI |
|---|---:|---:|
| Oracle - current | +0.003515 | [+0.002002, +0.005511] |
| Oracle - forced alpha=1 | +0.006426 | [+0.003362, +0.009026] |
| Oracle - same-image/count random-00 | +0.003385 | [+0.001609, +0.005370] |

这只证明存在可选择的 dense 可靠性差异。可部署 readout 以 train-520 的 GT 后验标签拟合，
在 val-130 上不读取标签；full pre-PromptEncoder 特征的 matched AUC=0.5701，低于
raw-statistics 的 0.5773（置乱对照 0.5644）。固定阈值 0.5 时 relative-to-current
mAP 为 raw -0.000993、full -0.001923、permuted -0.002700。再以 train-only Oracle
正例率校准阈值后，raw 仍为 -0.000246，full/permuted 不变为 -0.001923/-0.002700。

因此预注册的“val selector 胜 raw 与 permutation，且胜 current”的必要条件失败；**不实现、
不训练实例 dense gate 模块**。Oracle 数字只能作为 GT 上限保留。

## 3. 64→128 source-resolution：停止

为分离 resolution 与表示，matched proposal 内的 GT mask 先按训练同构的 crop+nearest
规则栅格化至 64 或 128，再通过 A0 生产的 bilinear paste 写入完全相同的 PE support；
points、box、current gate、PromptEncoder、decoder 和检测字段都不变。

| GT source grid | segm/mAP | 相对 A0 raw |
|---|---:|---:|
| 64 | 0.648166 | +0.015091 |
| 128 | 0.647944 | +0.014870 |

GT128- GT64 = -0.000221；200 次 image-paired bootstrap（seed44）CI 为
`[-0.000944,+0.000772]`，`P(Δ>0)=0.32`。因此不为 `COARSE_MASK_OUTPUT_SIZE=128` 启动
dev100 或 matrix300。64-grid 的 GT 仍有 +0.0151
上限，说明 dense 消费端和 source 内容空间都存在；被否定的是“加密栅格本身”这一解释。

## 4. 最终裁决与唯一下一门

本轮停止：signed-EDT、global/instance gate、R3 v1、当前 P2、以及 64→128 resolution；
均不得新增 dev100 训练臂。也不增加 loss。

若继续追求 dense 收益，唯一未被本轮证伪的对象是**固定 64-grid 上 ShapePrior source 的
预测质量**，而不是其表示、gate 或栅格尺寸。下一步必须先做一个训练集内的 source-capacity
feasibility spike：在完全固定的 proposal geometry、64-grid target、BCE+Dice、dense paste 和
decoder 下，仅扩大 `SmallMaskDecoder` 的中间容量；盲评 val proposal 的 support-canvas IoU。
只有它相对 A0 的 0.7821 有预注册的正向 image-paired CI，才允许新增唯一全模型训练臂
`A0-Capacity64`。若该 spike 也失败，则停止新增 dense-source 模块，转向非-dense 的
decoder-tail/论文主线，而不再以无证据的结构搜索消耗矩阵预算。

## 5. 工件

- gate current/forced features 与 records：
  `/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/diagnostics/a0_e300_dense_gate_audit_20260911_v2/`
- gate Oracle 与三份 bootstrap：
  `/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/diagnostics/a0_e300_dense_gate_selector_20260911/`
- GT64 / GT128 records：
  `/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/diagnostics/a0_e300_gtgrid64_20260911/` 与
  `/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/diagnostics/a0_e300_gtgrid128_20260911/`。
