# Explicit Coarse 训练调试字段字典

本字典是 `portable_sam2_explicit_coarse/` 训练日志中机制字段的唯一解释来源。它区分
单步 latest-forward 快照、rank-local 累积量和跨全部 DDP rank 的 epoch 统计；不得混用三者
比较结论。

## 1. 输出位置与统计范围

| 日志形式 | 触发与范围 | 用途 |
|---|---|---|
| `[DEBUG-DENSE]`、`[DEBUG-P2BR]`、`[DEBUG-COARSE]` | `PROMPT_DEBUG_STATS_INTERVAL` 的 rank-0 latest-forward 快照 | 排查当前 batch 是否接线、数值是否异常；不能当 epoch 平均。 |
| `Epoch N dense residual monitor` | 每个训练 epoch；所有 rank 的有效 forward 算术均值 | 判断 dense embedding 实际注入强度。 |
| `Epoch N P2 boundary refiner/support/gain` | 每个训练 epoch；按 ROI 或像素分母跨 DDP 汇总 | 判断 P2BR 覆盖、残差及 raw→refined 的局部收益。 |
| `Epoch N P2 R1 supervision` / `gradient probe` | 仅 `P2_BOUNDARY_REFINER_LOSS_MODE=correction_keep`；全部 step 和 DDP rank 汇总后再除分母，gradient probe 每 epoch 每 rank 首个有限 batch 一次 | 审计 R1 实际选择的错误/低置信/保持像素、两项辅助损失及其共享 P2BR 参数梯度关系；不能直接替代验证指标。 |
| `Epoch N prompt pathway` | 每个训练 epoch；每 rank 首个有限训练 batch 的 `loss_mask` autograd probe 经 DDP group-L2 RMS/计数汇总，参数更新量为整个 epoch 的 rank-wise group-L2 RMS | 判断最终 mask loss 是否真的训练 dense/P2BR，及 optimizer 是否改变这些参数。 |
| `DDP parameter-sync audit after first nonzero-LR update` | 仅 DDP 训练；首次计划学习率非零的 optimizer update 后，逐 trainable tensor 比较各 rank 参数与跨-rank 均值的最大绝对偏差 | 训练接线自检。正常应为 `0` 或仅有极小浮点误差；明显非零时，该次多卡实验不可用于比较。 |

`loss_mask` 指最终 SAM2 MaskDecoder 的 mask loss；它不包含 `loss_shape_prior` 或
`loss_p2_boundary_refiner`。因此 pathway 的 gradient 字段可区分最终任务训练与辅助 loss 训练。

## 2. Dense prompt 字段

| 字段 | 范围 / 公式 | 正确解读 |
|---|---|---|
| `DENSE/residual_alpha` | latest-forward 或 epoch 平均；有效全局 sigmoid gate `alpha` | 非零仅说明 gate 打开，不代表 decoder 已受益。 |
| `DENSE/source_delta_norm` | `||dense_pe - base_dense||₂` 的实例均值 | PromptEncoder 编码后的原始 dense 差异。 |
| `DENSE/applied_delta_norm` | `||dense_embeddings - base_dense||₂` 的实例均值 | 真正送入 MaskDecoder 的 dense 改变量。 |
| `DENSE/source_delta_ratio` / `applied_delta_ratio` | 对应 norm 除以 `||base_dense||₂` | `applied_delta_ratio>0` 才证明 dense 注入非空；过大不等于更好。 |
| `dense_final_mask_grad_group_l2_rms` | 每 rank 首个有限训练 batch 上，`loss_mask` 对 ShapePriorInjector + trainable dense gate 的参数组 L2 范数；跨 rank 后取 RMS。probe 使用未 scale、未 accumulation-divide 的 local `loss_mask` | D1 dense 行应非零；Point+Box control 理应为零或数值噪声，因为硬挖点不可微。它是 shape-injector+dense-gate 的代理，不是 dense-off control 中不存在模块的“dense 参数梯度”。 |
| `dense_final_mask_grad_nonzero_param_ratio` | 获得有限、非零 `loss_mask` 梯度的参数 tensor 数 / probe 参数 tensor 数 | 诊断是否只激活局部参数；不是元素级比例。 |
| `dense_parameter_update_group_l2_rms` | 各 rank 的 epoch 前后同一组参数总 L2 差，再跨 rank 取 RMS | 包含 optimizer 的全部更新效应（含 weight decay）；须与 final-mask gradient 联合解读。 |
| `pe_mask_downscaling_*`（同构三件套） | PromptEncoder mask 下采样卷积栈（`PROMPT_ENCODER_TRAIN_MASK_DOWNSCALING=1` 时 10 张量/4,684 参数；否则该分支为空、字段恒 0）的 final-mask 梯度 group-L2 RMS / 非零比例 / epoch 前后更新 group-L2 RMS；与 `dense`/`p2br` 分支同构同分母 | D5-B 的 PE 适配独立验收：`update_group_l2_rms>0` 且 `final_mask_grad` 非零才证明读取端真的在学习；**不得**用 dense 组的活动冒充（两组参数不相交）。optimizer 组审计行（`Optimizer group prompt_encoder`）给出入组参数数与实际 LR（应为 基础 LR×`PROMPT_ENCODER_LR_MULT`）。 |

## 3. P2BoundaryRefiner 字段

| 字段 | 范围 / 公式 | 正确解读 |
|---|---|---|
| `valid` / `rejected` | search support 有效 / 被 coverage gate 拒绝的 ROI 比例 | P2BR 的可作用 ROI 范围。 |
| `coverage` | 有效 search support 像素占 ROI 的比例 | 过高可能触发整 ROI 拒绝。 |
| `delta_abs` | 每 ROI `mean(abs(delta_logits))` 的 DDP ROI 均值 | P2BR 实际残差幅度。 |
| `sat_thr`（`saturation_threshold`） | 本次训练自己的残差上限 `0.99 × beta × delta_logit_max`（黄金跑 0.2×2.0 下恒等于 0.396；D2 0.1×0.5 下为 0.0495），epoch 值 = 阈值按 ROI 计数加权后的 DDP 均值 | 每个 run 的饱和判据随上限自校准；`run begin` 后首次出现时先核对该值再读饱和统计。 |
| `nonzero` / `saturated` | 非零 delta / 全部 coarse 像素中 `abs(delta) ≥ sat_thr` 的比例（每 ROI 均值再跨 DDP 取均值） | 前者为作用区域；后者是**全像素口径**的饱和比例，保留用于与支持域口径对照。 |
| `saturated_in_support`（`delta_saturated_support_ratio`） | 聚合式：`Σ 满足 abs(delta)≥sat_thr 且在 search_support 内的像素数 / Σ search_support 像素数`，分子分母先分别跨 step 与跨 DDP rank 求和、最后再相除 | **唯一推荐的饱和读数**：拒绝 ROI / 空支持域对分子分母同时贡献零，既不稀释也不产生 NaN；比值应对照 `sat_thr` 判读（接近 1 说明残差顶到本次上限）。 |
| ⚠️ 历史失效数据 | 2026-09-05（commit 624165c）之前，`saturated` 判据为硬编码 `abs(delta) ≥ 0.396`——按黄金跑上限（0.2×2.0=0.40）标定 | 对小上限配置（如 D2 上限 0.05）该读数**结构性恒 0**；D1/D1b/D2 历史日志中的 `delta_saturated_sum=0` 一律不可作为"未饱和"证据，`P2BR/delta_saturated_support_ratio_sum`（624165c 引入的中间口径）亦已被上表两个字段取代。 |
| `raw_dice`、`refined_dice`、`raw_iou`、`refined_iou` | 对同一 coarse ROI target 的 DDP ROI 均值 | 比较 P2 注入前后的 coarse 局部质量。 |
| `raw_boundary_f1`、`refined_boundary_f1` | 由全局 TP/FP/FN 汇总得到的 F1 | 不可把各 rank F1 直接平均。 |
| `P2 refinement gain` | `refined - raw` 的 Dice、IoU、boundary-F1 | D2 最低机制指标：不应持续为负。 |
| `p2br_final_mask_grad_group_l2_rms` | 每 rank 首个有限训练 batch 上，`loss_mask` 对 P2BR 全部可训练参数的参数组 L2 范数，跨 rank 后取 RMS；probe 使用未 scale、未 accumulation-divide 的 local `loss_mask` | 仅证明最终 decoder loss 的梯度，不包含 boundary auxiliary loss。 |
| `p2br_final_mask_grad_nonzero_param_ratio` | 同 dense 的 tensor 级定义 | Full 行应大于零；P2-off control 为零且参数数为零。 |
| `p2br_parameter_update_group_l2_rms` | 各 rank 的 P2BR 参数 epoch 前后总 L2 差，再跨 rank 取 RMS | 辅助 loss 也会推动更新，因此不能单独替代 final-mask gradient 字段。 |
| `P2 R1 supervision: corrective_support` | `Σ error_or_low_confidence ∩ loss_support / Σ loss_support`；先跨全部 step/DDP 求和。`error_or_low_confidence` 是 detached raw-logit selector | R1 实际给予 BCE 的支持域比例；它与 `keep_support` 应加和为约 100%。 |
| `error_support` / `error_low_confidence` | 分别为 `Σ error ∩ loss_support / Σ loss_support` 与 `Σ error ∩ abs(raw_logit)<correction_margin ∩ loss_support / Σ loss_support` | 将 `corrective_support` 拆回真实错误和错误∩低置信的交集。由四项可推导：错误高置信=`error_support-error_low_confidence`，正确低置信=`low_confidence_support-error_low_confidence`；它们均应非负。 |
| `low_confidence_support` | `Σ abs(raw_logit)<correction_margin 且在 loss_support / Σ loss_support` | 可与 corrective selector 重叠（错误且低置信），故不得与前两项相加。它说明 margin 是否真正扩大了纠错候选。 |
| `keep_support` | `Σ correct 且非低置信 且在 loss_support / Σ loss_support` | R1 仅施加残差保持惩罚的像素比例。与 corrective_support 构成支持域分区。 |
| `corrective_bce` / `keep_penalty` | 分别为每个有效 ROI 的 `BCE(error_or_low_confidence)` 与 `keep_loss_weight×mean(abs(delta), keep)`，先跨 step/DDP 求和后除有效 ROI 数 | 两项正是 R1 auxiliary loss 的可加组成，只能比较**标量 loss 量级**，不可单独声称 keep 压制或不压制 corrective。 |
| `P2 R1 gradient probe: corrective_l2_rms` / `keep_l2_rms` | 每 epoch 每 rank 首个有限 batch；按生产的 auxiliary weight 与 DDP 全局有效-ROI归一化后，分别对全部 P2BR 参数做 autograd；各 rank 的 group-L2² 先汇总、后取 RMS | 两项真实进入参数更新的梯度量级。keep 明显大于 corrective 才构成“可能主导更新”的必要但非充分证据。probe 不增加 loss、不执行 optimizer step。 |
| `P2 R1 gradient probe: cosine` | 上述两项梯度的参数内积与平方范数先跨所有 probe/DDP rank 汇总，再计算 `dot/sqrt(norm²_corrective×norm²_keep)` | `<0` 表示两项在共享参数空间存在相互抵消方向，接近 `+1` 表示同向；须同时看两项 L2，任一近零时 cosine 不应过度解读。 |

## 4. 共同字段与 D1/D2 判读

`params` 为该分支可训练 parameter tensor 个数；`probes` 为每 rank 成功执行的 final-mask
gradient probe 次数。固定训练中通常为 1。probe 取 epoch 首个有限训练 batch，用于控制额外
`autograd.grad` 开销；它是可重复的机制诊断，不是整 epoch 梯度平均。

D1 通过前至少应同时观察到：dense `applied_delta_ratio>0`、`dense_final_mask_grad_group_l2_rms>0`、
`dense_parameter_update_group_l2_rms>0`，且 Point+Box+Mask 的验证趋势优于 Point+Box。

D2 通过前至少应同时观察到：P2 support/delta 非零、`p2br_final_mask_grad_group_l2_rms>0`、
`p2br_parameter_update_group_l2_rms>0`、`P2 refinement gain` 不持续为负，且 Full 的验证趋势优于
冻结的 Mask control。

首次 update 若仍处于 warm-up 的零学习率阶段，两个 `parameter_update_group_l2_rms` 可以为零；
应结合启动日志中的学习率与后续 epoch 判断，不能把这一项单独判成死通路。

这些字段只用于机制筛选；最终模块效用仍由相同训练协议下独立训练的 paired validation 指标确认。


## COCO-eval 单一实现说明

2026-09 起合并为单轨：训练侧强版（自定义 maxDets 主 AP 重算 + rles 快速通路）上收至
portable_sam2_explicit_coarse/utils/coco_eval_utils.py，trainer、checkpoint 推理与探针共用。
checkpoint 推理/探针只走默认 maxDets 路径，行为与其历史输出一致；自定义 maxDets 行为仅训练侧调用。
