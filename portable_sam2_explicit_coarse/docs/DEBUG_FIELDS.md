# Explicit Coarse 训练调试字段字典

本字典是 `portable_sam2_explicit_coarse/` 训练日志中机制字段的唯一解释来源。它区分
单步 latest-forward 快照与跨全部 DDP rank 的 epoch 统计；不得混用两者比较结论。

> 2026-10 a3sgm-only 清理修剪：P2BR / R3 / DenseCap / prompt-pathway 梯度探针 /
> 置信门尾（A4/A5）相关字段已随死臂删除，此处不再收录；历史字段定义可在 git
> 历史（提交 7080811 之前）查到。

## 1. 输出位置与统计范围

| 日志形式 | 触发与范围 | 用途 |
|---|---|---|
| `[DEBUG-DENSE]`、`[DEBUG-COARSE]`、`[DEBUG-SP]` 等 | `--prompt-debug-stats-interval` 的 rank-0 latest-forward 快照 | 排查当前 batch 是否接线、数值是否异常；不能当 epoch 平均。 |
| `Epoch N dense residual monitor` | 每个训练 epoch；所有 rank 的有效 forward 算术均值 | 判断 dense embedding 实际注入强度。 |
| `DDP parameter-sync audit after first nonzero-LR update` | 仅 DDP 训练；首次计划学习率非零的 optimizer update 后，逐 trainable tensor 比较各 rank 参数与跨-rank 均值的最大绝对偏差 | 训练接线自检。正常应为 `0` 或仅有极小浮点误差；明显非零时，该次多卡实验不可用于比较。 |
| `TAIL/*` | UDPR 启用时的 latest-forward / selected-point loss 字段；前者由 debug-interval 输出，后者也进入 epoch detailed losses | 仅说明 tail 的选点和残差/监督实际发生；不能替代 paired validation 效用。 |

`loss_mask` 指最终 SAM2 MaskDecoder 的 mask loss；它不包含 `loss_shape_prior` 或
`loss_decoder_tail`。第二轮重构后模型 loss 通道只含纯损失键（loss_rpn_*/loss_cls/
loss_bbox/loss_mask/loss_decoder_tail/loss_shape_prior），训练器对损失通道无过滤
全量求和；mmdet 的 `acc` 等指标键以 `DET/acc` 进入遥测通道。

## 2. Dense prompt 字段

| 字段 | 范围 / 公式 | 正确解读 |
|---|---|---|
| `DENSE/residual_alpha` | latest-forward 或 epoch 平均；有效全局 sigmoid gate `alpha` | 非零仅说明 gate 打开，不代表 decoder 已受益。 |
| `DENSE/source_delta_norm` | `‖dense_pe − base_dense‖₂` 的实例均值 | PromptEncoder 编码后的原始 dense 差异。 |
| `DENSE/applied_delta_norm` | `‖dense_embeddings − base_dense‖₂` 的实例均值 | 真正送入 MaskDecoder 的 dense 改变量。 |
| `DENSE/source_delta_ratio` / `applied_delta_ratio` | 对应 norm 除以 `‖base_dense‖₂` | `applied_delta_ratio>0` 才证明 dense 注入非空；过大不等于更好。 |
| `GATE/shape_dense_alpha_{mean,min,max}` | latest-forward 的 gate 有效值分布 | 与 `DENSE/residual_alpha` 同源；训练是否把 gate 推离初值 0.25。 |
| `DENSE/canvas_{mean,std,min,max}` / `nonzero_ratio` | latest-forward 的 PE mask canvas 统计 | 画布是否非空、是否塌缩到常数。 |
| `SP/*` 计数字段 | 每次挖掘的 rank-local 计数（四槽 valid、空/全前景、点对距离/重叠） | 由 roi head 的 `_ratio` 派生成 `SP/*_ratio`；只进日志不进 loss。 |
| `SP/p*_in_gt_fg` / `SP/n*_in_gt_bg` | coarse-loss 路径上以 GT 采样的点位命中率 | 挖出的点是否落在真前景/真背景；训练期机制诊断专用。 |

dense 通路验收（D1 判读）：`applied_delta_ratio>0`，且 Point+Box+Mask 的验证趋势
优于 Point+Box。机制字段只用于筛选；模块效用由同协议独立训练的 paired validation 确认。

## 3. UDPR decoder-tail 字段（margin 容器）

| 字段 | 范围 / 公式 | 正确解读 |
|---|---|---|
| `TAIL/selected_count` | 当前 forward 所有正 RoI × 固定 K 的选点数 | K64 时应为 `64×RoI数`；零说明 RoI/接线而非"模块无效"。 |
| `TAIL/selected_abs_logit` | selector 后 `mean(|z|)`，z 为写回前的 native logits | 越低表示 selector 确实落在 native 不确定点；不是错误率。 |
| `TAIL/selected_pos` | loss batch 中 selected target 的正点总数 | 必须同 `selected_count` 一起读；极端稀疏时依赖正/负独立归一化。 |
| `TAIL/selected_bce` | selected point 的正/负 BCE：每类的**脱离梯度的 numerator 与计数均跨 DDP 求和**，各自归一后再等权平均 | 可跨 GPU 解读的单 batch 监督遥测；不等于反传的局部 DDP-scaled loss。margin 模式下它只是 detached 可比监测值，不进入 loss。 |
| `TAIL/delta_abs` / `TAIL/delta_max` | selected native logits 的残差绝对均值/最大值 | 零初始化第一步为 0；非零仅证明 tail 写回，不证明指标提升。 |
| `TAIL/selected_error_fraction` | 在 selected K 点中，写回前 native hard label（`logit>=0`）与训练 assignment GT 不一致的比例 | selector 是否把预算投向 pre-tail 错误点；不是全图错误率。GT 仅用于训练期统计，绝不参与 selector。 |
| `TAIL/selected_error_coverage` | `selected_error_count / all_native_grid_error_count`（同一正 RoI 完整 native 网格） | 当前固定 K 覆盖了多少可纠正错误池；须与 `selected_error_fraction` 联合读。 |
| `TAIL/changed_fraction` | selected 点中 `abs(delta)>1e-6` 的比例 | 细化器实际修改点的范围；零初始化时应为 0。 |
| `TAIL/error_direction_agreement` | 分母=selected error 点；分子=其中 `abs(delta)>1e-6` 且 delta 符号朝 GT 修正方向的点 | 训练期方向性机制证据；未改动的错误点按 0 计入。 |
| `TAIL/correct_flip_fraction` | selected 点中 pre-tail 错误、写回后变为正确的比例 | 真正跨阈值的局部修正量；不是 soft improvement。 |
| `TAIL/destroy_fraction` | selected 点中 pre-tail 原本正确、写回后变错的比例 | 副作用；应远小于 `correct_flip_fraction`。 |
| `TAIL/boundary_margin_violation` | margin 违约项的 DDP 全局归一值（正/负点各自按全局计数归一再等权） | margin 主目标的直接读数；下降才表明残差在兑现跨界目标。 |
| `TAIL/boundary_logit` | 本次配置的锚定值（thr 0.4 → −0.405465） | 配方常数；推理首帧自检会校验它与部署阈值一致。 |

所有字段仅用于接线/机制诊断，不能代替完整 validation paired bootstrap；epoch
`detailed losses` 是 rank-0 的时间均值，而非严格的全局 epoch 加权均值。

## COCO-eval 单一实现说明

2026-09 起合并为单轨：训练侧强版（自定义 maxDets 主 AP 重算 + rles 快速通路）收至
`portable_sam2_explicit_coarse/utils/coco_eval_utils.py`，trainer、checkpoint 推理与
评测脚本共用。checkpoint 推理只走默认 maxDets 路径，行为与其历史输出一致。
