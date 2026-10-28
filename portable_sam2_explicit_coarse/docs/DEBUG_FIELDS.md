# Explicit Coarse 训练调试字段字典

## 2026-10-27：原RGB-D双seed全预算四卡训练

`ue_rgbd_two_seed_ddp_20261027.py`的`progress.json`仅rank0每25次成功更新写入：`update`为该seed四rank同步完成的AdamW更新数，AMP失败尝试不计；`effective_batch=8`，不是每卡batch8。`preclip_norm`为DDP归约、AMP unscale后clip.5前的联合梯度范数；`AMP_retries_epoch`是本epoch同步回退重算次数。
逐epoch的`mean_microbatch_loss`先对四rank每个8图更新求loss总和，再对267次累积求和、除全局2136图；不是仅rank0均值或4个独立训练的均值。`sample_order`是2136图全局顺序，`rank_sample_order`各534图，必须重建一致。
`best.score/epoch`仅来自全459val的combined Mask AP250；四rank预测按图序在CPU汇总后统一算AP，不平均rank AP。所有新字段仅记录状态，不参与loss、路由或额外选权；仍按val-best正式规则选权。

## 2026-10-26：固定E67测试集五组输入／路由干预

`ue_joint_counterfactual_20261026.py`使用独立输出，不扩展训练日志。`target_fields_preserved`为同组已处理图中geometry/index/anchor原对象保留的图数（A/B/C）；不是几何可靠像素数。`disabled_inference_checks`为E中确认两开关关闭、context/outputs为空、UAV追加候选为零及原score映射保留的图数。两者全量分母均459，单图gate分母1。
`input_files_sha256`仅列本进程实际打开的UAV输入/几何缓存文件；C的`donor_rgb_sha256`明确列实际替换使用的2295个RGB，目标RGB可能被原loader读取后丢弃，不能按读取次数推断内容消费。D禁止实测cache读取，E禁止任何UAV输入读取。
OpenCV imread另行包装核验；`identity_metadata_sha256`列合同检查读取的manifest/BUILD_COMPLETE元数据，和实际NPZ/RGB消费分开，E允许这些身份检查。
`A_minus_B/C/D/E.MaskAP_delta`为0–1合并AP差，名义95%同图区间500次；不是训练配对消融或种子稳定性。`mean_appended_candidates`为459图UAV追加候选数均值，不含原卫星推理本身的候选；E应为0。

## 2026-10-26：四臂固定best的训练集／测试集补充评价

`ue_joint_snapshot_eval_20261026.py`只在独立运行目录写评价遥测，不改训练字段。
`progress.images_done/images_total`为当前arm、当前split已完成图数／总评价图数（train2142、test459），每25图及最后一图写入；不是batch均值或跨rank汇总。`elapsed_seconds`含该评价循环的输入读取与推理，不能当隔离GPU延迟benchmark。
`selected_epoch`是被固定的val-best权重epoch；`completed_epochs_at_snapshot`是固化时该臂已完成训练预算，两者不能混用。`optimizer_updates=0/state_unchanged=true`仅认证只读评价，不认证内容净收益。
`analysis.comparisons.*.MaskAP_delta`用0–1 AP单位，区间为500次同图bootstrap名义95%；仅合并Mask AP有该区间。训练集区间不能当泛化证据，不同预算和单seed限制保留。Boundary AP带宽仍为512图像对角线2%／1%，约14／7px。

## J1编码器内部交互（2026-10-23）

latest_forward.target_support_fraction为整张64²目标token中32个候选至少1个有效的比例；source_delta_rms是224维输出残差RMS，base_stage_rms是Hiera block4原224维张量RMS。不是RoI/UDPR点统计，不能与N2/S1残差RMS直接比较。target_tokens=4096、candidates_per_token=32为维度合同。
无可训练alpha，输出映射零初始化。loss为所有有梯度loss项之和（包括RPN list/tuple项），grad_norm为AMP unscale之后、clip.5之前的全adapter范数；不可与旧尾部只连接mask/tail目标的loss直接比大小。每epoch loss_microbatch_mean和样本顺序/指纹沿用原审计字段。
所有统计仍是当前forward快照，checkpoint重算时覆盖含义不变；新增字段不引入逐步全参数/attention分布同步。

## S1稀疏UDPR点读取（2026-10-23）

**旧S1已作废**：其全图坐标假设与实际roi_local监督不符，点级GT诊断不得用于结论。当前S1-R同名统计先经proposal框把native点映射到图像，再读取UAV/GT；以UE_CONSUMER_BLINDSPOT_PROBES_20261023.md为准。原始输入feature未裁剪不代表输出mask是全图语义。

沿用训练日志节奏。latest_forward.point_support_fraction为最后一次UDPR调用中选中点有候选的比例，包含重复RoI，不是全图覆盖率；base_point_rms只统计原选中点的32维特征。source_delta_rms/applied_delta_rms也仅对应选中点，不能与密集N2的整张张量RMS直接比较。rois/points_per_roi说明该次范围。
S1没有可学习alpha，冻结值为1，W零初始化；不得用缺少alpha梯度判为断路。
仅离线评估的read_observation.point_diagnostics跨当前图全部UDPR调用累积：matched_rois/points为最佳GT框IoU≥.5的RoI/点数（重复计数）；supported_matched_points为有UAV候选的子集。corrected/damaged相对相同raw logits和未增强特征的原UDPR结果定义，阈值为实际boundary_logit。old_correct/new_correct满足差值=corrected−damaged。该统计不是AP、Boundary AP或训练标签覆盖率。

## 2026-10-23：G0离线梯度诊断

独立脚本ue_gradient_topology_audit.py写gradients.json，不增加训练每步字段。
parameters.status区分None（该loss无路径）、zero、nonzero；norm为FP32未缩放未裁剪梯度L2。
groups.rms_all_parameters以该组全部可训练参数数目归一，connected_numel单列；深层未用于P0的LoRA不可自动视为故障。
receiver_gradients/post_activation_gradients统计实际mask调用张量的梯度RMS，含重复RoI和无支持区；zones为训练GT每实例/并集1px内边及前景内部/背景，不是Boundary AP带宽。
cosines为同次前向不同实际loss在各参数组的梯度余弦；任一范数为0时为null。负值不自动判定训练失败，范数不等于AdamW更新。
optimizer_steps=0、satellite_state_unchanged和adapter_state_unchanged为离线只读检查；dropout/augmentation关闭，不能充作整段训练梯度分布。

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


## 4. UE UAV 双流遥测（2026-10-14）

生产者为 `rsprompter/detection/dual_detector.py`、`uav/fusion.py`、`uav/diagnostics.py`；
训练汇总与梯度由 `train/ue_telemetry.py` 提供。字段只用于日志，不加入 loss、门控决策或选模。

### 输出、分母和聚合

- `[DEBUG-UAV]`：默认每50个 **mini-batch** 的 rank0 latest-forward；不是跨卡均值。
- `UAV/grad/<module>`：同一间隔且恰逢 optimizer update 时打印；rank0、累积梯度经过 AMP unscale 后、裁剪前。禁用 prompt-debug 时不打印。
- `[UE-EPOCH]`：每个 UE train epoch 输出一个 JSON，所有 rank 在 epoch 尾共同参与。
  每个 UAV 字段独立记录 `mean` 和 `finite_forwards`，即跨卡有限值之和 / 有限 forward 数。
  只计入全局 loss 有限的训练 forward，不含 validation。字段 NaN/Inf 排除并减少自身分母；
  不把缺失字段补成有效的零。统计是 **forward 等权均值**，不是像素、建筑或 cell 加权均值。
  正式 batch1 时每 forward 对应一张图；DistributedSampler 补齐的重复样本也会计入。
- 单流没有 `UAV/*`，epoch JSON 只记录 runtime。所有 runtime 的显存值都是当前训练进程的
  PyTorch allocator 读数，不包括 UE 渲染进程；显存峰值在 train epoch 开始重置，结束读取，
  不包含本轮后续 validation。

### 支持、几何和注意力

| 字段（均带 `UAV/`） | 公式 / 单位 / 分母 | 判读边界 |
|---|---|---|
| `candidate_available` | 每图 `(candidate_index>=0).any` 的 BEV 全格比例，再作 batch 均值 | 仅表示8m邻域可读 token；不是本格真实覆盖。 |
| `observation_support` | 每图真实深度观测支持的全格比例，batch均值 | 不表示卫星可见性。 |
| `height_valid` | 每图满足代表高度规则的全格比例，batch均值 | 有高度不等于高度真值正确。 |
| `geometry/upper_fraction` | 所选上表面高度桶观测 / 本格观测；batch中所有有效高度格等权均值 | 不表示建筑前景占比。没有有效高度格时字段缺席。 |
| `geometry/selected_bucket_sigma_m` | 所选高度桶内高度标准差，恢复为米；同上分母 | 不是整格所有高度的离散度。 |
| `geometry/nearest_xy_m` | 最近候选 XY 距离，恢复为米；同上分母 | 小距离不证明跨视角语义对齐。 |
| `attention/sampled_queries` | 最后 CVT 层最多16×16规则采样，排除无候选 query；batch均值 | 支持诊断样本量，空支持为0。不是全网 query 数。 |
| `attention/local_mass` | 每采样query：head均值attention在 `abs(deltaXY)<=cell/2` 的权重和，再均值 | 本格上下文占比；立面可作为合法语义上下文。 |
| `attention/height_compatible_mass` | 每采样query：`abs(deltaZ)<=height_bin_m`（首版2m）的attention权重和，再均值 | 软相容程度，不作为硬过滤。 |
| `attention/xy_bias_m` | `norm(sum(attention * deltaXY))` 的采样query均值，米 | 有正负抵消；不是 `sum(attention*norm(deltaXY))`，更不是对应真值误差。 |
| `attention/height_bias_m` | `abs(sum(attention * deltaZ))` 的采样query均值，米 | 同上抵消边界；参考是观测 Zrep。 |

attention 遥测关闭 autocast/cache，用 FP32 近似重算 logits；只看最后层，排除无候选行。
空样本时 mass/bias=0，必须结合 `sampled_queries` 读，不能把0解释成完美对齐。
AMP+checkpoint 校验启停诊断的输出及梯度相同，避免共享权重缓存影响正式训练。

### 融合入口

`<route>` 为 `fpn_s8`、`fpn_s16`、`sam`。每个入口有 `UAV/<route>/<field>`：

| 字段 | 公式 / 分母 | 判读边界 |
|---|---|---|
| `alpha` | 可训练标量原值，初始0.05，允许负值 | 符号/幅度表明残差缩放，非收益。 |
| `gate` | 全batch全网格 sigmoid gate 均值 | 未覆盖格也计入，不能与覆盖区均值混读。 |
| `confidence` | 全网格几何质量均值，无覆盖格为0 | 覆盖少会拉低该值。 |
| `coverage` | 每网格细像素投影 mass / 网格像素数，截断到[0,1]，再全网格均值 | mass近似不是精确可见像素占比。 |
| `supported_fraction` | `(coverage>0)` 的全batch全网格比例 | 有投影证据的网格比例，不是 coverage 的同义字段。 |
| `gate_supported` | `sum(gate * valid) / count(valid)` | 覆盖区域的 gate；不把低覆盖误当门控关闭。 |
| `confidence_supported` | `sum(confidence * valid) / count(valid)` | 覆盖区域的几何质量。 |
| `effective_gain_supported` | `sum(alpha * confidence * gate * valid) / count(valid)` | 实际标量缩放强度，可为负；不是特征增益或 AP 增益。 |
| `residual_ratio` | `norm(applied.float()) / max(norm(satellite.float()),1e-6)`，全batch全通道网格L2 | 真正注入相对强度；不证明有效贡献。 |

`valid=coverage>0`；无支持时三个 supported 均值以分母max(count,1)输出0，必须同
`supported_fraction` 一起读。零支持forward的这些0仍计入epoch的finite_forwards；epoch均值不是只对非空forward的条件均值。新增统计都 detach，不改变融合输出、梯度或模型指纹。

### 梯度与运行状态

| 字段 | 口径 / 校验 |
|---|---|
| `UAV/grad/{uav_backbone,uav_bev,fpn_fusion,sam_fusion}` | 对应**实际独立clip分区**（整个FPN ModuleList同一组，SAM另组）；不把 optimizer 的合并fusion组误当一组裁剪。 |
| `used` / `nonzero` | grad非None张量数 / 至少有一个非零元素的张量数；不是元素比例。LoRA初始化部分张量零梯度合法。 |
| `finite` / `norm` | 全组grad均有限 / unscale后全组L2范数；Inf/NaN表示数值异常。 |
| `clip_scale` | `min(1,0.5/(norm+1e-6))`，与本项目max_norm0.5、L2裁剪公式标定 | 对有限梯度预测裁剪缩放，不是优化器最终参数更新比例。 |
| `lr` | 当前 optimizer 对应组学习率；UAV/CVT单独组，FPN/SAM共享fusion LR | 用来区别无梯度和零LR，不能以梯度非零证明参数更新。 |
| `runtime/rank_step_attempts` | 所有rank本epoch scaler.step 调用次数之和 | 不是全局更新数，4卡时每轮通常除4；末组可能只有1个mini-batch。 |
| `runtime/rank_amp_skips` / `amp_skip_fraction` | 每rank `scale_after < scale_before` 的step次数之和 / rank_step_attempts | 标准GradScaler overflow降scale对应跳步；禁用AMP为0。现有scheduler/EMA计数仍按尝试推进，不能把尝试计数称为成功更新。 |
| `runtime/amp_scale_rank0` | epoch结束rank0 GradScaler scale | 非跨卡均值；周期增长正常，持续下降提示数值风险。 |
| `runtime/max_rank_{allocated,reserved}_mib` | 各rank训练epoch峰值的最大值，MiB | reserved是缓存；不包括其它进程，也不表示全机可用显存。 |

机制曲线用于排查支持、注入与优化。最终增益依赖独立训练的A/D完整卫星segm AP100，
伴随bbox AP100/AP75与AP300另报；zero/shift干预仅用于依赖诊断。

本轮628训练样本、4rank各157个mini-batch、累积2：每epoch79次更新尝试，末次全局batch4（常规8）。覆盖区新增均值使用FP32归约，避免AMP半精度求和误差。

标签审计后627训练样本的重启：训练DistributedSampler(drop_last=True)每epoch丢3个、4rank各156mini-batch、78次尝试、全部累积组2；validation不drop。前述628→157→79仅适用于旧清单。


## 2026-10-21 citysample_minimal评估身份

仅新minimal路线：主AP上限250，`bbox/AR@250`、`segm/AR@250`替代本路线旧命名AR@100，
由同一次COCOeval maxDets=[1,10,250]得到；mAP等主键仍保留便于验证选模，快照与推理manifest
均携带evaluation_contract明确AP250。不可与历史AP100直接比较。
checkpoint推理现在按快照primary_max_dets评估（旧无合同的checkpoint仍默认100）。
新增train_augmentation配置身份记录噪声概率/标准差与ColorJitter参数、顺序，非随机触发次数遥测。
原loss/UDPR字段的聚合范围不变；本路线禁用EMA，所有val均model权重。

## 2026-10-21 v1冻结适配器筛查（独立runner）

`scripts/probes/ue_v1_adapter.py`的`train.jsonl`不进入生产训练遥测。`latest_forward`
仅指每次累积8张后最后一次microbatch；无DDP、不是epoch均值。`alpha`为无界单个残差系数，
`source_delta_rms`/`applied_delta_rms`为该次所有RoI空间通道的RMS，`base_rms`是conv_s0输出RMS；
`support_fraction`分母为128²卫星候选格，不是GT/边界覆盖率。`rois`为该次正RoI或推理RoI数。
`grad_norm`为累积后裁剪前参数梯度L2范数；`loss`是末microbatch参与梯度的原掩码及UDPR损失之和。
字段只用于接线/数值诊断，不参与选权或替代完整验证。实际对齐与关闭分支由smoke数值校验。

独立probe直接调用的canonical evaluator仍把第三个AR槽命名为`AR@100`；实际本协议传入
`max_dets=[1,10,250]`，其数值对应AR@250。原始缓存不改写；最终collect汇总显式重命名为
`AR@250`，避免与旧maxDets100实验混读。这是字段命名兼容问题，不改变匹配或AP数值。

### native_geometry / native_zero独立筛查

沿用RMS/alpha的latest-forward定义；`support_grid=512`明确支持率分母为原生512²像素，
不能与adapter候选128²格支持率直接比较。两臂不使用RGB编码器/LoRA，只比较同support下
真实几何内容和零内容。高度与梯度来自最近UAV已观测表面，不是目标视角可见性真值。

恢复训练行新增`epoch_seed`与`recovery_parent_epoch`：前者为恢复后逐轮确定的随机种子，后者标识
恢复前缀epoch；它们是来源字段，不是性能遥测。旧前缀行缺这些键属正常。epoch checkpoint包含
RNG与parent合同，逐轮satellite_state_sha256核定冻结；最终更新日志必须连续且不能混入归档的未继承尾部。

unit/unit_geometry的`source_delta_rms`仍是原公式，但输出无affine LayerNorm将有效位置的通道RMS
限制在约1（无效位置0，验收容差1.01）；实际注入RMS由alpha与有效支持共同决定。训练幅度
稳定不能替代Mask/Boundary AP及真实RGB对常量RGB的净贡献比较。`zero_init_logits_exact`为
smoke原始mask logits逐值恒等，强于只检查二值mask；输出dtype校准不改变mask阈值。

终点审计不以state.stage=done代替证据：实际加载last.pth核对epoch=10、step=2670及完整合同，
逐行核对10轮各267次更新。常量内容训练对照遵守同一审计；这些检查是离线验收，不新增训练遥测。

用户改为test主要评价后，eval新增离线`evaluation_identity.json`：`split`、`images`为实际数据范围；
`base_sha256`/`adapter_sha256`固定权重身份；`mode`为correct/constant/donor/disabled，
`score_key=scores`、`max_dets=[1,10,250]`、`inference_precision=fp32`为评价合同。
这不新增训练遥测、不读test训练。test/test_state.json的完成条件是19项推理与15项完整配对分析；
frozen_test_plan.json记录结果产生前固定的十臂/五个内容对照，不能用val旧gate替代test判断。

推理records中的`stats`是该图**最后一次mask head调用**留下的统计。anchor_roi_head推理默认
按最多32个RoI分块，所以stats.rois不是整图检测实例数，RMS也只覆盖最后分块；训练的
latest_forward仍沿用末microbatch/末调用定义。对459图stats取中位数必须明确为“末分块统计的
跨图中位数”，不能写为整图/所有RoI或像素加权平均。alpha属于全模型单个系数，与分块无关。

## 2026-10-22 S²DB机制探针（独立产物）

`ue_s2db_probe.py`沿用旧latest_forward及更新行loss语义；loss仍为最后microbatch原掩码+UDPR，
不将aux混入该字段。每epoch独立summary的main_loss_microbatch_mean/aux_bce_microbatch_mean
分别以实际2136个microbatch为分母；aux_valid_pixel_sum是逐图唯一锚点投影像素数之和，
不是跨图去重，也不是4×4格覆盖。无支持的aux BCE为0。辅助总loss系数固定.1。

采样统计只在离线评价开启，不新增逐步字段：valid_anchors是保留候选数，offset_mean_abs/
offset_p95_abs是按源特征像素计的每轴绝对偏移；axis_at_95pct_count为每轴abs(offset)≥.95计数；
accepted/fallback以valid_anchors为分母；depth_jump_accepted是接受样本中深度差
>max(1m,.05*anchor_z)的计数，报告比例时须写分母accepted。
这些统计在每图encode计算一次，各RoI复用；与末mask调用RMS的聚合范围不同。
深度突变不参与拒绝，偏移只描述外观上下文。记录不参与训练loss/门控/选模。

`epochN_summary.json`的sample_order为实际参与更新的2136张顺序，SHA仅作续训审计；
卫星SHA包括参数及缓冲区，逐轮核对。完整RNG包括DataLoader generator；CPU回放门同时检验worker增强随机流。
工程gate文件不代表完整2670次更新或AP提升；完成审计必须实际读取E10权重、训练行及固定评价记录。

### 2026-10-22 离线遥测阶段核定

完整数值见[阶段报告](dual_stream/UE_V1_PROBE_STAGE_CONCLUSIONS_20261022.md)。applied/base为每图末mask调用RMS比值后取跨图中位数，不是两个中位数相除；alpha无约束可为负。原始RGB实际注入约75%接收RMS，unit约58%，原生几何约36%。这些比例只说明扰动强度，内容收益必须结合独立训练对照。未新增训练字段。

### Native空间诊断离线字段

`all_call_telemetry.json`按图保存全部mask调用，`rois`为每调用实例数；records.stats仍为末调用。`raw_intervention_rms`为幅度匹配前残差RMS，`reference_delta_rms`为相同query/boxes下正确字段残差RMS，`rms_match_scale`为应用的每调用单标量，`residual_cosine_to_correct`为未匹配残差与正确残差的展平余弦，`residual_change_rms`为最终残差相对正确残差差值RMS。仅离线诊断、不写训练日志、不用于门控或选模。

### 2026-10-22 测试弱点离线审计

新增`ue_test_*_20261022.py`仅分析已有数据/预测，不向训练日志增加字段。尺寸组须区分：
canonical GT `area=bbox_w*bbox_h`，现有s/m/l按bbox面积的COCO阈值划分；
离线补充将GT area换为mask面积，all-area AP回放不变，不覆盖冻结评价记录。
`errors/summary.json`的Hungarian按bbox IoU≥.5优先最大匹配数，再最大总IoU；
`support/summary.json`先最大化总bbox IoU再筛≥.5，两者固定各自配对且不是COCO AP匹配。
COCO分数顺序匹配单列；其ignore flag用logical_not，不对整型flag做按位取反。
像素计数公开各自区域分母，重叠实例可重复计数；错误审计的OpenCV默认形态学图外值与
支持审计显式背景值不同，两者边界分母不可混用。支持审计union FN不反映实例区分质量。
保留点覆盖由float16 geom重建时是近似值；32图原深度重建只作独立子集校准。

### 2026-10-22 读取有效率的解释边界

S²DB离线`accepted`在`original_source_uv+4*offset`上检查深度有效性，特征读取在离散token中心偏移。
两者起点不同，因此该比例是“锚点平移的深度代理有效率”，不是精确新3D对应率或全特征感受野有效率。
实际4×4格token中心与source_uv差异的CPU核对见读取审查报告；此说明不改变现有字段公式。
接收base RMS来自原全图repeat特征，残差RMS来自RoI局部网格；RMS比值用于数值强度，不能证明地面点对齐。

### 2026-10-22 Native来源P2（离线，不新增训练字段）

`*_cache_check.json`的mismatch计数分母分别为512² bool support元素和4×512² float16字段元素，
是历史缓存逐值核对，不是几何准确率；8张均cached-reference。
`source_camera_roundtrip_max_px`为原始有效射线按同一矩阵回放的最大像素误差，属于内部一致性。
`visual_review.json`为固定24点视觉记录；全部无法判断不等于校准通过，也不等于24点错误。
`root_verification.json`是独立厘米制原始射线复算，计数8图24点，仍共享渲染相机与深度，不提供外部真值。

### native路径分解与像素错误审计

path_decomposition的first_conv_*_rms在固定整图box128²网格分解，非实际检测RoI、非末mask调用；geometry_term_per_field_rms为每字段单独卷积，不能视作因果贡献。matched_error_audit固定32图、bboxIoU≥.5关联，区域像素按匹配实例计、可重叠；漏检不计入平均IoU。形态学2px区域不是Boundary AP的14/7px带。仅离线产物，无新增训练字段。

query四格诊断沿用原records.stats的末mask调用字段；query关闭包括投影bias，但mixer bias与支持仍保留。factorial_identity明确干预位置，paired_mask_analysis为固定权重内容比较，不是重新训练效果。无新增训练字段。

### 六条件Native赋值诊断（独立离线产物）

`all_call_squares.json`每图累加所有实际mask调用：elements是32通道×128²×实际RoI数，
base/residual/injected_square_sum是FP64累加平方和。整图/全集RMS=sqrt(平方和之和/elements之和)，
不是RMS或比值的算术平均；image injected_to_base_rms_ratio是同图平方和比值开根，base为0记null。
`input_mapping_identity.json`记录整图支持内四维联合向量置换的映射SHA、原/改fields SHA、cache SHA，不含GT决定的参数。
`delta8.json`只覆盖固定8trainval所有实际mask调用，记录未缩放残差差值平方和、点积和余弦，零范数null；不参与loss/门控。
主检验仅Mask AP两个对比，500次同图bootstrap；Boundary及尺寸/纠错为描述性，无额外确认主张。

### 几何采样审计离线统计

ue_native_sampling_audit的supported_samples为重复计数的RoI网格采样点，非整图唯一像素。material neighbor要求有效且归一化双线性权重≥.05；mixed_height_gt2m/5m按这些邻居高度跨度统计，是不连续代理而非错误表面真值。GT边界2px由独立实例带取并集，与Boundary AP无关。numpy_vs_torch_max_error_m验证float64采样公式，不认证外部配准。无新增训练字段。

### 2026-10-22 共租调度状态（不属于训练遥测）

`ue_probe_coresident_20261022/state.json`每5秒按整机刷新：`physical_memory`为nvidia-smi每卡MiB；`projected_memory_MiB`在实际占用上补足每个受管运行进程的工作预算差额，并包含旧GPU1评价的未来预算。它不是峰值实测，也不是GPU利用率。`running`仅列该owner接管的进程，整机其他进程仍计入实际显存。字段只用于资源调度，不参与loss、门控或选模；训练日志未扩展。

### 几何机制8臂统计合同

训练沿用loss末microbatch、grad_norm累积unscale后clip前、latest_forward末调用；difference source_delta_rms指相减后的最终残差，不能与任一H单支幅度混淆。epoch_summary记录实际样本顺序及均值loss；GPU gate分别记录training AMP和250RoI FP32峰值allocated/reserved，以及整卡快照，不能把本进程峰值当整卡总占用。非零constant来自train全有效像素均值，字段zero不同于constant。

### attention ladder观测纪律

沿用原latest_forward末mask调用及训练末microbatch语义，不新增逐步字段。L3 null质量/拒绝与各方向候选观测只在离线observe开启；null probability必须对应输出LN后真实残差幅度。L6额外decoderLoRA的变化单列，原SAT参数buffer冻结证明不包含新增adapter-owned LoRA；不能写整消费者冻结。源坐标单位为整数像素中心，L1 feature grid由patch kernel/stride/padding解析确定。

L3 read_observation只存eval/gate，训练latest_forward不合并null字段；当前null统计scope为最后adapter内部4RoI chunk（可能由checkpoint重算覆盖），不是全部图/实例平均。read_observation须与stats最后mask调用（最多32RoI）区分；解释时明确范围，不将其称总体拒绝率。

### 2026-10-23 RGB内容净纠错与方向诊断（独立产物）

`ue_rgb_content_error_audit_20261023`比较同权重correct与constant/donor；`corrected`是干预错而correct对，`introduced`相反，`net_corrected=corrected-introduced`。分母为固定bbox关联实例的区域像素，实例间可重复，不是整图唯一像素。boundary为GT两次3×3形态学带，内部/背景互斥，均限于GT/DT框并集外扩8像素。cell支持与实际保留锚点唯一投影支持分别划分，二者均非遮挡真值。IoU阈值上/下跨不是AP分解；canonical阈值与计数守恒由独立verification核定，原始计算保留来源。

`ue_rgb_content_direction_audit_20261023`只覆盖固定4train图、FP32、关闭增强/dropout、无更新：`content_difference_rms`是全部实际conv_s0调用correct−intervention数值注入差的RMS；总注入RMS分母为相同调用所有RoI×32×H×W元素，按平方和累计，不对调用RMS求平均。`content_difference_over_correct_total_rms`不能解释成信息贡献比例。loss按model.loss实际加权key分别记录，receiver叶子只测下游敏感性，不测源参数梯度。`dot`为各调用梯度与内容差点积之和；正值表示在对应端点沿intervention→correct方向局部增大该loss；零范数cosine为null。实际loss差另列，离散选点可变，不能把一阶dot等同于全跨度loss差。raw/final差异RMS比及选点rank/set变化率只描述原生部署路径，不是强制同点UDPR因果消融；全部不参与训练loss、门控或选模。

### 跨划分内容方向补测

`ue_rgb_content_generalization_audit_20261023`与原train方向诊断公式相同，增加显式`split`、`sample_index`、`donor_sample_index`；不沿用train_index命名。每划分按实际GT实例数稳定排序取四分位位置。collect的`mean_delta`为该划分四图实际加权loss差的等权均值、`favorable`为差<0的图数（分母4），不是跨划分配对估计、不是按RoI或像素加权，也不代表完整验证集均值。使用GT正proposal的诊断不能代替部署Mask AP；所有结果仅离线产物。

### 检测阶段漏失定位（独立产物）

`ue_detection_attrition_20261023/records.json`每图每阶段：`candidates`是候选框数；`best_iou`是每GT在该阶段的最大bbox IoU、无候选为0；`one_to_one_capacity`是IoU≥.5/.75稀疏二部图的最大匹配数量，不是分数顺序COCO匹配。summary的分组recall分母为该组GT总数，以任意框覆盖计；不同GT可共享框，须结合整图最大基数容量阅读。

`transitions`逐相邻阶段分别计lost/recovered；回归可能恢复。`final_uncovered_last_transition`只对最终无框覆盖的GT，按最后一次有覆盖后的下个阶段分类；从未被覆盖者记never_covered，总和必须等于最终无覆盖GT数。它不认证某步骤的反事实AP贡献。`max_foreground_score_among_good_decoded_boxes`为RoI回归后IoU达标候选的最高前景分数，无候选为−1。

`final_dt_proxies.no_gt_overlap`为最终DT没有任何IoU达标GT的数量；`unmatched_capacity`为DT数减最大匹配容量；均非背景误检/重复的唯一判据。全部尺寸组为真实mask面积，密度阈值固定。resource峰值是本诊断进程allocator读数，不含其它GPU进程；state/progress只记录诊断，不参与训练、门控或选模。

检测漏失的`verification.json`另记`final_coverage_fp32_vs_independent_coco_float64_threshold_disagreements`，分母为全部44988GT，计数为最终任意覆盖判定在相同bbox IoU阈值下的布尔差异；只核定数值边界，不覆盖主FP32统计。不能将稀疏最大基数容量与旧Hungarian最大基数再最大IoU的匹配身份混用。

离线梯度审计调度修订：`reserve_MiB=20000`为FP32多loss审计的准入工作预算，不是实测峰值；仍加2048MiB设备余量。旧3500MiB只适用于原推理预算，不能用于该梯度审计。previous_attempts/旧OOM日志保留，重排不改变训练或统计合同；state失败不等于模型效果差，恢复CPU统计也不等于GPU审计成功。

### UAV几何RPN objectness探针

2026-10-24独立离线诊断：candidate_evidence的mask_support_fraction按单GT mask像素计，group均值按实例等权，pixel_weighted按重复实例像素加权；final recovered/lost为任意框IoU≥.5，不是一对一或AP归因。rpn_supervision_audit的counts按唯一anchor抽样，实例内good_*按GT–回归候选匹配计，可跨GT重复；assigned正负按原anchor，sampled按一次确定sampler，ignore含原ignore及图外无效。saturated正/负分别logit>10/<−10，分母为抽样正/负数；mean_image_sampled_bce为每图抽样未乘原loss权重BCE均值再图像等权，不同于正式AMP增强训练loss。candidate_quality_bridge的candidate_average_precision是候选二分类PR积分，不是实例Mask AP；top1000/pool2000是NMS前任意GT框IoU≥.5覆盖。所有新字段仅离线文件，不参与现有loss/门控/选模，不新增逐步训练遥测。

geometry_proposal_pilot仅写每epoch汇总，不增加既有逐步训练日志。mean_batch_loss为267个batch训练损失等权均值；mean_batch_components.heat/offset/size均先按当batch唯一GT中心cell总数归一，其中size尚未乘总loss的0.1系数；amp_scale是该epoch末缩放器值。sample_order为2136个实际图ID，step每轮267次更新尝试（若AMP跳步由scale体现，不冒称每次均成功参数更新）。独立val totals的any50@K/any75@250按GT任意候选IoU覆盖计，一个候选可覆盖多GT，不是实例AP或一对一召回。两个内容臂同一投影support；常量臂不等于无UAV。

score-band/conditional均为独立离线/推理产物，不扩充生产训练日志。score-band新增分数=原score_thr×head_objectness，原高置信SAT scores保持；条件组合为完整强SAT前缀的严格单调分数映射(.5,1]及几何新增≤.25，不能把层级分数当经过概率校准的共同置信度。conditional的admission.original、satellite_supplement、geometry_cached_equivalent、geometry_accepted均按当图候选个数计，不是独立GT或像素；geometry_cached_equivalent为先复现原几何mask批次、再条件筛选前的个数。prefix raw记录可与纯SAT补充直接比较，combined保留prefix排名与mask但其数值分数已重新编码。AP非下降是原全局前缀保留的设计性质，只有匹配内容差能回答UAV字段是否额外有用。

geometry_new_instance_audit的new50/new75是全area、max250、COCO segm贪心一对一新增GT匹配，分母44988；old dtMatches/dtIgnore在10阈值全部不变才出报告。additional_unmatched包含重复/背景/定位失败，不能直接当背景误检。bbox_small/medium/large按序列化GT bbox面积阈值1024/9216，不等于真实mask面积分组；density按GT数≤50、51–100、>100固定。count_matched按每图min(G,K)新增数量截断各自分数尾部，共50838，GT不参与配额，是离线诊断不是新推理策略。

独立seed45复用同mean_batch_*分母与10×267更新，新增epoch_seed=1045000+e，初始化46；seed44为1044000+e、初始化45。RNG保存含Python/NumPy/Torch/CUDA/DataLoader generator，但当前重复为不中断运行，不借此宣称新CNN中断续训已实测。固定input_mode constant/shuffle保留support；whole_map_donor的donor_offset229同时替换四字段和support，input_point_id记录真实输入点。干预均是OOD/配对诊断，不改变训练loss或选模；其置信区间为同图bootstrap，并非训练seed方差。

新runner的train.jsonl中`loss`仅为累积8图后末microbatch的原RPN分类loss（各FPN层之和），不含bbox/Mask/UDPR；`grad_norm`仍为unscale后clip前新增模块总L2。`latest_forward.levels`覆盖该末microbatch实际全部5个RPN层：locations为该层H×W，supported为nearest中心有支持位置数；elements为B×6×H×W；delta/base_square_sum为该层实际logit残差/原logit的平方和。整层/全图RMS须sqrt(平方和之和/elements之和)，不对各层RMS作均值；不是检测AP或几何准确率。alpha无界原标量，实际增益因2*tanh(alpha)有界，最终残差还乘tanh(head)与support。

epoch summary记录2136图实际样本顺序、每轮267次更新尝试、mean分类loss、冻结SAT SHA。推理记录的stages沿用检测漏失10阶段口径，允许框/分数和检测数变化；同图bootstrap仅要求GT一致。K/Z均保留真实support，不是完全无UAV控制。排名可达诊断gap>4仅是±2单anchor分数残差无法把原好框送入每层top1000的充分条件；gap≤4不保证有support、共享小头能学到或NMS后保留。所有排名/GT机制统计只离线保存，不用于loss、门控或选模。

### 实例监督内部回写（2026-10-24）

独立`ue_instance_interaction`不向生产逐步日志增加字段。`epoch*.json.mean_microbatch_components`分母为2136个实际训练microbatch，等权均值；`center`是热图focal＋offset＋.1×size，三分项先按当图唯一正中心格数归一，`size`分项尚未乘.1。`original`只汇总原loss中可求导项；head入口原卫星没有可求导路径时为0，不能解释为原网络预测误差为0。监督覆盖全部GT，无支持区域也可能训练卫星旁路，不将整体loss下降解释为UAV学习成功。

`progress.update`为完成clip与scaler.step的全局更新数；非有限梯度或AMP降scale跳步即中止，保证宣称的2670次更新不是尝试数。每25更新写末microbatch的分项到stdout，不聚合成整段均值，不参与选择或门控。

仅评价/验收开启`observation`：support_tokens按64²中心格计、分母target_tokens=4096；delta/base_square_sum覆盖该图64²×224所有通道，分母elements，RMS=sqrt(sum/elements)。跨图RMS先合并平方和与元素数，不平均每图RMS比值。无支持中心回写为0，但该支持仅为观测代理。geometry_accepted为当图被接纳新增框数，不是新增GT数。

`satellite_records`保存本次特征条件下的原检测/掩码及SAT自身候选补充；internal时不是纯卫星。`combined_records`另加UAV实例框，score层级不是概率校准。常量几何保留support与RGB对应，常量RGB保留真实深度/pose；均不等于完全无UAV/无深度。固定donor几何换整图及support，RGBdonor只换RGB，两者不能混称同一种反事实。

extended_gate的resume_parameter_max_abs_error是磁盘恢复后一次实际AdamW/AMP更新对不中断参考的最大参数绝对差；replay_noise_calibration是三次同代码内存恢复的相同差值最大值，门限max(1e-7,2×噪声标定)。loader顺序与增强输入SHA逐值一致；该门不声称CUDA权重逐位一致，也不证明训练重复性。

独立收尾`finish_state.json`仅记录控制面：done/running为119个预定任务中已完成/当前运行个数，observed为观察时间戳，不是epoch比例。`study_summary.evidence_flags`的positive仅指对应Mask差值>0且名义图像bootstrap下界>0；net_clear_reference另外要求raw AP差值≥.002。不是多重比较校正或训练种子方差估计，不参与训练/门控/选权。原路径内容差与合并内容差分开，不把候选补充当内部表征改善。

`epoch*_rgb_effective_path_audit`仅离线：source_LoRA_parameters_with_optimizer_state按与实际optimizer参数槽对应、持有AdamW状态的source LoRA张量numel求和；source_declared_trainable_LoRA_parameters为工厂requires_grad声明总数。前者E1实测25648（patch_embed.proj和blocks0/1），后者1061424；有optimizer状态不证明有效RGB语义/梯度质量。actual_checkpoints与missing_checkpoints明确已核定CP数及缺失列表，all_RGB_arms_present仅当6个RGB臂均存在才真。原source非LoRA逐值对初始CP核定，更新步数来自实际AdamW state，不只读取日志。

完成状态需优先核对实际CP和`training_complete.json`：`progress.json`只每25更新写一次，可能在E10完成后仍显示2650。不能据该旧值判定任务尚未结束；也不能仅凭complete文件存在跳过实际CP/optimizer身份核验。`prefix_input_followup`是原路径固定E10输入干预的观察后离线分析，与原119项分开；负差表示正确字段输入低于该替代输入，不直接证明深度无信息或确定梯度冲突。

`capacity_handoff`为控制面证据，不是训练遥测：adopted保存PID/起始ticks；initial_extra_admission记录按MiB核算的额外准入。working_set=同家族/入口工程门和已完成epoch最大reserved峰值+512MiB上下文预算；available=NVML free−单卡2048MiB余量−Σmax(working_set−当前该PID NVML用量,0)。每次新启动还从available扣其工作集，防止模型加载前重复占额。该数值只影响资源调度，不进入loss/门控/选模；终点训练身份仍以实际CP和优化器状态核定。

`value_source_*`为固定E10的离线代数审计，不扩展训练日志。每图每输入记录一次：`uav`为卫星交互前U的线性value贡献，`satellite_return`为实际更新后邻域token减原U邻域的线性贡献，`bias`合并uv与output偏置，`actual`为原始实际注入，`base`为Hiera block4原特征。所有注入乘原中心支持，attention权重保持原观察值。`elements=4096×224`包括不支持格的零注入；跨图`pooled_rms=sqrt(Σsquare_sum/Σelements)`，`pooled_cosines=Σdot/sqrt(Σsquare_sum_a×Σsquare_sum_b)`，零分母为null。`supported_tokens`分母`target_tokens=4096×图数`，不是像素可见性认证。三项并不正交，RMS不是可相加百分比；UAV项包含support/几何CNN，不能作为独立RGB贡献。

`max_reconstruction_error`为三项相加相对原注入的最大绝对误差；FP32容限逐图`2e-6+2e-5×max_abs(actual)`。原模型、adapter、CP和冻结源身份在结束核验；完整split须`all_split_images=true`。这些观察不参与loss/门控/选模，也不等价于去掉某项的AP干预。

`p0_efficiency_gate`是独立工程门，不写训练日志：`checks`逐视角/FP32或AMP P0张量逐值比较；`normalized_candidate_reads_exact`包括32通道投影、无affine归一化和连续候选读取。训练梯度使用相同CUDA起始RNG，`replay_noise`是3次完整源重放相对参考的最大梯度绝对误差，`tolerance=max(1e-7,2×replay_noise)`；不是CUDA逐位相同。`end_rng_equal`明确删除无用层dropout会改变全局随机流。参数量只计共享一套source，五视角不乘5。

`five_views_cuda_ms`/`median_ms`为FP32 source-only连续五视角CUDA event耗时，预热2轮、测5轮，共享GPU运行可能受其它任务影响；`additional_peak_allocated_MiB`为重置peak后最大allocated减起点常驻allocated，两套源均可能常驻，不是单独部署显存，更不是端到端延迟。所有字段仅离线观察，不进loss/门控/选模。

`camera_variation_audit`中`K_unique_exact`按同split同视角[fx,fy,cx,cy]计唯一值；`relative_pose_unique_rounded_6`按satellite world_to_camera乘UAV camera_to_world后、平移cm转m的4×4矩阵保留6位计唯一值。全局世界平移消去不代表世界位置相同。固定K/rig意味着跨图错配这些量几乎不改输入，不能做任务必要性证据。

`p0_full_val_gate`每25图写进度，实际结束必须核对459个唯一ID及`source_P0_exact/fused_U128_exact/support_exact`，范围为FP32且只到交互输入，不是最终AP；P0和U128比较均为torch.equal而非容差。原adapter、CP与冻结源在结束核定身份。

`reference_plane_cache`进度每25图记录完成数/3060与PID，仅构建控制面，非训练遥测。每图`candidates`分母16384×8槽，`candidate_cells`分母16384格，`supported_pixels`分母512²像素；这些是参考平面投影支持，不是真实可见率。`measured_depth_read=false`与审计钩子禁止EXR/实测缓存读取配套，摄像机元数据和sampling-origin参考面仍被使用，不称完全无几何。全部完成须`BUILD_COMPLETE.json`含3060条及各缓存SHA、manifest来源身份，不能把进度或3图工程门当作无深度消融结果。

`compact_joint_foundation_gate`为仅train的3步工程门；`group_parameter_norm_sums`按satellite LoRA/decoder/UAV LoRA/geometry/writeback各组逐参数L2范数相加，非整体梯度L2，也不证明内容收益。`preclip_norm`为全模型联合梯度在global .5裁剪前的整体范数，`AMP_scale`为完成更新后的scale，降scale跳步即失败；无均值跨图聚合，3步独立列出。`actual_torch_load_paths`只核构造期加载源，全部为通用foundation；`common_satellite_initial_sha`为添加UAV前新鲜卫星状态，非E73权重身份。门不训练val/test，不声称续训或正式指标已核定。


紧凑联合runner每25次真实更新写progress：last_loss为该次最后microbatch未除累积8的原任务+实例loss，不是区间均值；preclip_norm为累积8后的全模型梯度L2、clip .5之前；epoch.mean_microbatch_loss为2136图等权均值。validation为整459图FP32 AP250，best仅按val combined segm/mAP严格较大更新，未读test。checkpoint_phase=after_validation在正式epoch边界使用；工程门epoch0/step1为隔离检查点，不能送正式train恢复。

optimizer_resume_gate对非空exp_avg/exp_avg_sq、参数组、模型、AMP scaler和五RNG逐值核对，再执行8图累积更新，不宣称CUDA后续轨迹逐位一致。runtime_gate的增强顺序门使用预定16图子集的8图ID/输入SHA；input_audit对全3060参考npz实际字节及camera逐项SHA，不能仅按计数认定完成。

analysis_capacity_handoff为资源控制面：CPU统计上限12，准入余量=MemAvailable−8192MiB−Σmax(4096MiB−当前统计RSS,0)，新进程再扣4096MiB；不进入估计器、loss或选模。原GPU handoff记录不改，CPU新源额外核验。owner161872/finish161891已退役，当前实际PID从owner_launch/finish_launch读取。

`loss_direction_audit`仅离线24个固定train函数：main/auxiliary_square按组全部参数的梯度元素平方和（unused按零），dot同坐标内积，norm=sqrt(square)，cosine=dot/(norm_main×norm_aux)，任一零则null；mean_cosine按8图的已定义余弦等权平均，不是合并梯度余弦。active_parameters表示autograd返回非None的参数元素数，不要求数值非零。epoch首图的per-loss writeback norm只覆盖output.weight，不能与包括bias的整组norm混用。

`compact_cost_audit`每种路径3图×3重复；wall_ms含eval GT/输入读取、预处理、网络和RLE，eval_input_wall_ms包含CPU缓存/PNG准备；CUDA stream事件间隔可能含host空隙，不称纯GPU算子时间。SAT/RGB-D输出实例数不同，源完整/紧凑输出指纹才逐值一致。additional_peak_allocated_MiB为共享所有常驻模型条件下相对起点新增的allocated峰值，不是单模型部署显存；中位数不作为统计置信区间。所有字段仅观察，不进loss/门控/选模。


### 2026-10-24 正式联合运行补充（取代工程门的跳步即失败规则）

`AMP_retries_epoch`是本epoch失败的累积尝试次数，分母为本epoch实际267个成功更新，不是失败图数或新增样本数。一次重试重算同8张已准备增强图，恢复该组loss前Python/NumPy/Torch/CUDA/DataLoader generator及模型buffer，保留降低后的scale；失败尝试不增加update/样本顺序/成功loss均值，最多4次重试。`progress.update`为跨epoch成功AdamW更新计数，每25次落盘；`last_microbatch`为本轮已收集图数；`epoch.mean_microbatch_loss`仅成功尝试2136图等权均值。仅溢出事件额外打印，不加逐步内容遥测。工程门的3步无跳步要求仍是该历史门的范围，不能混用于正式100轮的动态scale策略。

`segm/AR@250`、`bbox/AR@250`是COCO第三maxDets槽（250）的召回；旧共享工具命名AR@100在新runner本地更正，实际评价数值不变。选权仍为val combined segm/mAP，不使用AR或test。

`paired_analysis`仅离线：`baseline/candidate`各含AP/AP75/标准COCO小中大AP/AR250；`delta`为candidate−baseline原始AP（乘100才是百分点）；区间为500次459图同索引有放回重采样，seed20261021，未校正多重探索。`original_path`表示本模型交互条件下原卫星检测输出，不是纯卫星。`accumulation_reference_checks`应24条、绝对误差<1e-12；它证明累积实现一致，不证明预测或训练的科学有效性。读取效应/内容依赖/相对纯卫星净增益必须分开解释。该分析不参与loss、门控或选权。


`posttraining/state.json`仅为收尾控制面：waiting阶段的training_owner_pid/tick需要对应实际活句柄，done/running按10个收尾命令计数，不是训练epoch或收益比例；complete的scope仅含8比较＋2干预，final_checkpoint_and_scientific_audit_still_required保持true。固定RGB干预donor_map为459图循环映射，rgb_source_sha256仅donor PNG来源；model_state_unchanged检查推理前后网络参数/缓冲区，不能证明输入有用。raw/complete仅原评价结束，外层complete才包括额外内容/权重身份检查。


`geometry_encoding_train_audit`仅离线2142train缓存统计：四字段mean/RMS/std按全部有support的原始512像素元素等权合并，不平均每图RMS。difference.valid_pairs要求原始相邻两点均support；unknown_pairs以有support起点计缺邻点及外缘，unknown_fraction_of_supported_origins分母为有支持像素，unknown_fraction_of_zero_supported_origins分母为该通道数值等于零且有支持的起点，两者不得混用。clipped_valid_pairs以abs(normalized_dxdy)>=5计、分母valid_pairs；不是所有像素或边界像素比例。occupied64_raw_support_fraction_quantiles为全部非空8×8格的实际support数/64的合并分位数，是覆盖代理，不是CNN感受野/遮挡认证。统计不参与loss、门控或选权。


### 2026-10-25 checkpoint_audit

`optimizer_step_histogram`按实际optimizer参数ID与模型requires_grad顺序映射后统计张量个数，不按参数元素数计；缺状态列`optimizer_missing_state`，不自动判为训练失败（既有未用头/条件空分支可能没有状态）。canonical_parameter固定原RPN rpn_cls.weight，要求其实际step等于CP.step。`frozen_parameter_tensors_exact`为逐值等于共同初始化的冻结参数张量数；`trainable_tensors_changed`为逐值不同的可训练张量数，不代表非零梯度频率或信息价值。`changed_buffer_names`仅限state_dict持久buffer；`nonpersistent_buffers_not_checkpoint_auditable`列出CP无键者，不能解释为已经核验。`--epoch`是指定anchor的工程观察，`final=true`才要求四臂完训best/last，但也不证明AP净收益。日志/收据不进loss、门控或选权。


`epoch1_path_diagnostic`为E1 val早期只读分解。original_path的score取combined同一预测前缀，逐框bbox/label/RLE一致；appended_candidates_delta为同图同前缀score下合并−原路径AP，非对真实内容或训练动力的独立因果估计。original_count_mean/appended_count_mean均分母459图，appended_total为总新增候选数，不是新增正确实例数。delta_vs_satellite是不同训练模型E1验证点值，非test、非同CP干预且无CI；不用于选结构/提前停臂。


### 2026-10-25 joint_loss_direction

仅固定train8图的离线FP32/eval函数，comparisons四项为main_vs_auxiliary/detection_vs_segmentation/detection_vs_auxiliary/segmentation_vs_auxiliary；left/right含义按名字顺序。main由同一次前向的det+seg梯度逐参数相加，无第四次随机前向。left_square/right_square/dot按组全部参数元素合并，unused视作零；norm=sqrt(square)，cosine=dot/(left_norm*right_norm)，任一零为null。*_active_parameters表示autograd返回非None张量的参数元素数，不是非零梯度元素数。summary.mean_cosine为8图已定义余弦均值，negative_cosines分母defined_cosines；mean_left/right_norm为8图L2等权均值，不是合并向量范数，二者比值也不是每图比值均值。范数未经AMP/累积8/globalclip/AdamW，不等于实际更新份额。

分组卫星neck/检测头/原maskdecoder分别独立，aux对这三组梯度为零时余弦不可定义；不以0填余弦。geometry仅a/b/c/d/up；rgb_alignment_fusion为rgb_project/rgb_fusion；instance_head含head_satellite/head_fusion/heat/reg；其余UAV位置与双向query/key/value为attention。peak_reserved_MiB是本分析进程峰值，不含同卡训练进程；allocator_cap_GiB为14，对运行器预算另留512MiB上下文和每卡2048MiB余量。owner/state仅两个E3/E10分析命令的控制面，等待CP需实际RGB-D训练句柄存活；不计作训练epoch、不参与loss/门控/选权。


### 2026-10-25 收尾v2/固定干预/成本

`posttraining/state_v2`控制19个命令，owner_v2_launch为当前身份，state/owner_launch为v1历史；done不是训练轮次。成本依赖前16项完成的原因是测量隔离，不能套用该条件限制训练/功能任务共租。complete_v2的artifact_sha256逐项绑定19个marker，final_scientific_review_still_required保持true，不以工程通过认定模态贡献。

`fixed_rgb_intervention/*/paired_*`的delta为正确RGB−干预RGB的原始AP；干预为baseline、正确为candidate，500次同图bootstrap seed20261021。original_path仍为固定RGB-D权重下原卫星路径，不是纯卫星。source/donor身份校验不能证明输入物理一致，且未校正多比较/空间相关/训练方差。

`joint_cost`每case 3图×3重复。wall_ms含eval GT/输入读取；prepared_inference_wall_ms从输入就绪到RLE与CUDA完成；uav_prepare_stream/satellite_with_interaction_stream/detection_mask_RLE_stream为CUDA事件间隔，可能含host空隙。peak_allocated/reserved_MiB为单case进程各次前向峰值绝对量，summary取9次中位，非部署全进程显存（context/库不在allocator内）。original_instances/output_instances/appended_instances为每图预测数量，不是正确数量。retained_prefix_sha256用combined前缀完全相同score/bbox/label/RLE，v2结束必须对完整/原路径九组逐一相等，防止用不同预测冒充省时。成本不进loss/门控/选权。

`final_input_identity`的archive_files/geometry_cache_files分别是实际读字节SHA核对文件数（缓存应9180），不是模型样本数；archive_bytes/geometry_cache_bytes为哈希到的字节和，非内存占用或数据增强量。标签仅身份哈希，非测试评价。source/manifests/通过标志仍不能认证真实物理标定。


### 2026-10-25 原始影像对应离线字段

`rgb_depth_image_alignment`中at_zero/best_local/controls的count均为每view中央裁剪的深度结构边缘像素数；mean_clipped_distance为到RGB Canny最近边缘距离，12px截断，within2/4为原始计数，summary按全部点合并。局部偏移是加到深度边缘坐标后查RGB距离，argmin非真值/校正；同一批点用于所有offset。不是标定准确率，纹理/暗部/LOD会改变边缘集合。

`rgb_sat_image_correspondence`中common_pixels为每view真实/参考面warp支持交集再裁去40px边缘后的像素数；真实与参考面及32px控制都用同样起点。measured_ZNCC/plane_ZNCC为RGB逐通道去均值后拼接向量的相关系数；mean_view_*按40个view等权，不按像素池化。不同view共8scene，不是40独立场景。重投影只用最近已有表面，缺洞黑色，不认证完整遮挡。complete为数值计算完成；visual_review文件公开实际看过哪些图，不能据visual_review_pending历史值称全部或完全未看。两工具只读train影像，字段不进loss/门控/选权；无AP或训练重复性含义。


### 2026-10-25 retained_rgb_read_audit

valid_candidates分母为实际保留有效槽668301（非所有16384×8槽）；behind_nearest_observed_1m是锚点own-floor卫星像素的sat-Z残差>1m数量，front为[-.01,1]m；不是每格最近Z也不是遮挡真值。negative_residual_candidates为<−.01m，单列定位复算/离散化矛盾。mixed_front_and_behind_cells/cells_without_front分母supported_cells；camera_Z_spread的格仅含counts>0，格内多个不同子像素不可自动等同多表面错误。

candidate_color.RGB_MAE按该组全部候选×3颜色通道等权，0–255单位；组间位置/材质分布不同。common_cell_color所有方法同74631格，SAT参考为原4×4 RGB均值，各法格内采样位置/数量不同，mean_image_ZNCC跨8图等权；这不是学习特征保真率，也不与逐view影像ZNCC共分母。anchor_replay_max_errors的Z单位m、UV单位px；index严格检查view和源stride4格。仅只读观察，不改变模型候选/聚合/门控/选权。


`joint_loss_direction/owner`当done=[3,10]且complete.json的两个anchor/驱动SHA及实际epoch10记录均核定时，控制面已正常结束；最后state仍可能是waiting_or_running。完成后/proc owner PID不存在是正常终态，不应重新启动诊断。当前活控制面应分别核对实际训练owner和收尾v2 owner；完成marker不能免除CP/源码/8图记录绑定的前台核验。

`matched_validation_observation.json`仅汇总当前所有共同完成epoch，Mask_delta_pp是RGB-D−对应臂原始AP×100，单位百分点。跨臂顺序一致性按每轮2136唯一image_id逐值核对，不代表CUDA训练重复性；累计共同轮数不是选权轮次，不跨epoch均值取代既定val-best，不参与loss/门控/选模。

### 2026-10-25 RGB机制变体前缀

`ue_rgb_mechanism_20261025/<variant>/<content>/progress.json`每25成功updates输出；`variant/content`仅实验身份，`last_loss/preclip_norm/AMP_retries_epoch`沿用compact成功accum8更新的既有含义。`original_prefix_order_exact`为每epoch2136个唯一ID与原compact同epoch顺序逐值相等，不代表CUDA优化数值相同。`peak_reserved_MiB`为进程allocator累计峰值，不含整个GPU上其他进程或CUDA全部外部内存。

`gates/*/gpu_gate.json`中的changed_parameters只核对3个工程更新后参数是否改变，不是梯度强度/AP收益。no_sat_copy以output×0保留反向计算图，其reverse支路是零梯度而非None；这些参数仍可能因AdamW衰减改变，不可据改变声称反向复制路径被消费。`extra_trainable_parameters`为新增view_mix/spatial_mix参数数目，不是所有可达RGB参数总量。

`analysis/summary(.partial).json`的`content_interaction`为四臂同一次图像bootstrap的差中差；`delta`与区间为原始AP单位，展示pp时乘100。`paths.original_path`是条件化后的原检测/掩码路径，不能标为无UAV纯卫星；`combined`还含实例头新增候选。固定E10为前缀评价点，不是val-best，也不能比较100epoch最终结果。

### 2026-10-26 candidate_detail

新独立`ue_candidate_detail_20261026`沿用既有compact训练遥测：progress每25次成功更新，last_loss为末microbatch未除accum8的原任务+中心框目标；preclip_norm为8图累积后全模型L2/globalclip.5之前；epoch.mean_microbatch_loss分母2136唯一图；AMP_retries_epoch仅失败累积尝试次数。variant/content为身份，未增加逐步attention或深度统计，不进入loss/选权/门控。

Q/G门的initial_features_max_error、initial_head_max_error为相同初始化FP32旧/新路径各张量元素绝对差最大值，非平均误差。candidate_geometry_vs_legacy_six_max_error比较7维前6项(偏移减.5)与旧float16 geom6，单位已归一化；不是物理标定误差。frozen_tensors_exact计不训练参数张量，changed_parameters为3次工程更新后的逐值变化，不代表输入贡献。peak_reserved_MiB是本进程allocator累计峰值，不含外部CUDA内存/其它共租任务。formal_FP32和max_decoder_instances=250是工程覆盖，不是AP/召回结果。

分析的small_gt/small_TP/small_FN按GT实际RLE面积<1024分组；TP+FN=GT，TP+FP=输出DT，IoU>=.5且按score贪心一对一，bbox与mask两种匹配独立，不能混分母。duplicate/localization/background_proxy以失败DT对全部GT的最大IoU分别>=.5、[.1,.5)、<.1划分，不是人工背景标签或COCO AP分解。small_recall_at_FP_per_image使用全部尺寸共同FP预算/459图，分母为该split全部small GT；score阈值由各模型自身曲线产生，不用于训练/选模。mask_area_groups补算GT和DT实际mask面积；density_groups固定<=50/51–100/>100，val无>100图应为null，不以0代替。content_interaction为同一次500图像bootstrap的四臂差中差，不能相减两个区间。所有诊断只用于离线观察。

### 2026-10-26 四卡candidate_detail（取代本轮单卡运行字段）

新progress仅rank0每25次全局成功更新落盘：world_size=4、effective_batch=8；last_global_microbatch_mean是本次4rank各2图的原loss总和/8，区别于旧last_loss末图；preclip_norm为同步和累积后全模型clip前L2，各rank理论相同，门已核对真实副本权重一致。AMP_retries_epoch是全局失败尝试次数，每rank共同重试一次只计一次。epoch.mean_microbatch_loss分母全局2136图，rank_sample_order各534图，按每rank连续2图重组为sample_order，并逐值核对原合同。rank0_pid不能代表只有单卡训练；rank_launches必须有4个真实rank PID/local_rank和起始tick。

四卡门rank_peaks按各rank单独的allocator累计峰值，不能把4rank峰值当一张卡或跨卡均值；forced_decoder_instances=250为每rank实际解码数量。replicas_identical核对第3次成功更新的全模型SHA；门另在恢复后完成第4次更新。restored_all_five_rng包括各rank Python/NumPy/Torch CPU/当前CUDA/DataLoader generator，4rank合起来覆盖4张卡，不调用每rank get_rng_state_all。每epoch验证459图按rank分片/Gloo收集/恢复顺序后统一评价，单rank子集AP不是正式数值。scheduler的per_GPU_reservation_MiB为实际最大门峰值+1024，另留每卡2048；组准入同时检查四卡，并扣除启动中的未兑现预留，无固定组数上限。

### 2026-10-26 FROC 权威修正口径

`analysis_corrected_froc/{split}/*.json` 的 `small_recall_at_FP_per_image` 取完整同分组阈值端点，允许空预测；预算为全部尺寸误检数/整集图数，召回分母为真实掩码面积<1024的GT实例数，匹配仍为每图score稳定排序、IoU≥.5贪心一对一，bbox/mask分别计算。无small GT时为null。它仅替代冻结分析的FROC字段，不改变AP、总TP/FN/FP或训练。`images/detections/small_gt`公开整集分母；`records_sha256/identity_sha256/checkpoint_sha256/source_sha256`绑定来源。watcher每30秒检查24份既定完整评价，不新增逐步训练遥测，不参与loss、门控、选模。

### 2026-10-26 调度恢复状态

`owner_recovery_launch.json`记录新owner PID、源SHA及已核定完成job；`recovery_20261026/audit.json`记录原owner缺失、无活跃数值子进程、93源身份、完成receipt/CP核定及控制代码提交。`recovered_from_verified_receipt`仅表示重建控制面完成状态，不代表额外训练。旧progress每25更新，E10最后记录2650而完成receipt为2670是正常采样间隔；不能据progress判未完成。state可能为退出前快照，需同时核实PID与完成产物。无新增训练字段。

### 2026-10-27 UBR组件工程门（非训练遥测）

`ue_ubr_component_gate_20261027/*rank*_result.json`为4rank各自独立synthetic组件门，不是正式训练epoch。updates各row是一rank本地1图（rank3首步0实例但图分母仍1）的mask平均＋0.1×raw_ubr_loss；raw_ubr_loss为带内像素均值→有效实例均值→本图loss，weight在gate仅乘一次。grad_norm_before_clip是同步DDP梯度AMP unscale后的整体L2，clip_limit=.5；decoder_feature_leaf_grad_norm为synthetic输入叶子范数除AMP scale后的值，不是实际decoder参数梯度。replica_sha为每次更新后四rank核同的参数SHA。zero_init_exact_identity只对组件输入目标网格的连续logit，不能代替旧RoI probability-paste等价。

training_peak_allocated_mib/reserved为每rank进程allocator训练阶段峰值，非整卡/整模型总量。inference.num_instances=250是每rank顺序执行250个512² UBR实例、一次驻留1个输入；秒数包括CPU scipy SDT与设备传输，optimizer状态仍驻留。不是250实例同时反传峰值，也不包括SAT/UAV/SAM/候选生成。finite/outside_logits_and_binary_exact只检验该组件边界合同。source_sha绑定module/gate；complete仅4rank全部成功后写，异常保留failed_rank。字段不进入loss决策/门控/选权，不扩充逐步训练日志。

### 2026-10-27 UBR坐标适配器工程产物

`ue_ubr_adapter_gate_20261027/complete.json`是4rank数组，每rank独立三种incoming dtype（FP32/FP16/BF16），每种3个合成框、128²canvas，非数据集或训练遥测。`max_roundtrip_error`为该rank该dtype全部像素的max abs(sigmoid(finite_logit)-原paste概率)，epsilon1e-6、容差1.1e-6；不均值或跨rank平均。`exact_binary_and_uint8`逐像素对比真实FCN threshold .4/.5及负阈值uint8；`threshold_ties_exact`直接保存概率覆盖exact/adjacent .4、0/1。`gradient_finite_nonzero`是合成native logits/features输入叶子梯度，不是实际decoder参数。source/script SHA绑定本次源。无AP、无完整模型资源指标，不进入loss/门控/选权，不增加逐步日志。

### 2026-10-27 UBR真实检测器门与短训练口径

`ue_ubr_wiring_20261027/detector_foundation_gate_attempt3`是四rank真实模型工程门，不是完整训练/AP。parity对完整自然250pool按旧source段/chunk32真实head.predict_mask比较，probability_max_error为全部像素max abs，raw/binary/scores/boxes逐值比较。updates每rank本地1图，rank3首步显式空GT；gradient_norms为DDP聚合且unscale后范数，空rank非零不说明本地loss有贡献。active_count<=64为预测RoI数，可多RoI指同GT，不是唯一GT数。context_count是已选pool总数。nonempty_band_count为active里band非空数量，用于UBR有效实例平均。AMP_attempts记录同批恢复重试，失败不计成功更新；峰值MiB均rank进程allocator reserved。250压力fixture重复一真实框，非250自然实例；自然pool仅parity另测。

新短runner progress每25成功更新仅rank0写，last_global_microbatch_mean分母8图，preclip_norm同步累积AMP unscale后全参数L2；epoch mean_microbatch_loss分母全局2136图（含空图），rank_sample_order各534、sample_order重组2136逐值核原顺序。无新增训练逐步几何stats。resume_gate complete的rng_next_five_exact逐rank逐Python/NumPy/Torch/CUDA/loader五种各5个随机数，optimizer_moment为显式synthetic零AdamW状态；只证明恢复机制，不证明真实next-update轨迹。owner headroom扣实际驻留后的未兑现预约，reservation8066MiB/card，cap9216是上限、margin2048另计。字段不用于loss/门控/选权。

### 2026-10-27 三组共租控制面修订

owner_state.control_policy记录新控制器commit/SHA及原owner身份；numerical_commit仍5b719ee。per_card_reservation_mib改7042（实测6530＋CUDAcontext512），all_card_margin_mib仍2048，max_groups改3。admissible_free_mib为nvidia实际free减2048再减各running组尚未兑现的7042预约；不是实际free。allocator cap9216不变，不能将cap与预约混用。仅控制面变化，训练遥测/归一化不变。

### 2026-10-27 UBR效率门与正式显存口径

`ue_ubr_fast_gate_20261027/attempt4`是新源码最终工程门；旧attempt1/2失败和attempt3选型保留。execution_policy区分training_microbatch1与inference_microbatch4。cases.FP32/AMP的old/fast1为真实完整UAV/G/SAT提取但仅UBR loss的前反传诊断（非正式完整目标更新）；loss/gradient差异比较同weight、同image、同RNG；band/context/ID/assignment要求exact。profile各项秒数含cuda同步且inclusive重叠，不能相加当占比；speedup只取无插桩total比值。correctness_peak_reserved_mib包含FP32/AMP梯度诊断，official_fast_execution_peak_reserved_mib/max_peak_reserved_mib只取正式AMP完整目标及FP32推理峰值供owner预约，max_all_measured另列全部最大。full_objective_updates每rank1图，无accum的global4工程两更新，非正式global8；norm为DDP unscale后，全rank SHA须一致。250inference为真实自然pool，并区分zero exact概率与nonzero maxabs/binary一致。native_cache_bytes只计native logits/features原dtype，不计总模型/allocator。新字段仅离线门，不增加逐步训练遥测、不参与loss/门控/选权。

### 2026-10-27 Native RoI工程门口径

`ue_ubr_native_gate_20261027/attempt5`是最终候选4rank工程门，每次更新各rank1图无累积（global4）；empty_GT仅rank3 step1。updates.gradient为DDP同步、AMP unscale/clip后的参数组L2，空rank非零可能来自其他rank；native_out首步有梯度、native_hidden首步0符合末层zeroinit，第二步再核可达。decoder_calls为每本地mask解码前向次数，正常64pos RoI是一批1次，empty0；AMP重试另有attempts记录。主mask-only objective=原loss_mask＋0×其余loss保持DDP图，除主mask外贡献严格0；梯度不冒充独立UAV预测证据。main_mask_size512不变，native_hw128，loss_terms不得出现ubr或decoder_tail。inference250为强制框fixture、原consumerchunk32＝8次SAMdecode，秒数含此maskconsumer处理、输入特征已提取，不当整图端到端耗时。A/C wholeloss计时含真实UAV/G/SAT抽取与全部loss backward，无optimizerstep；CUDA同步且同图同RNG，非多轮吞吐。峰值为rank进程allocatorreserved/allocated。observedraw/RGB等tensor只工程observe开关、禁止塞逐步日志；关闭后清理。门字段不进入loss/门控/选权，不扩训练逐步遥测。

main_mask_only_precision公开autocast、gradient_scale、无optimizerstep及其他loss权重0；此独立诊断也必须scale/backward/unscale，不用未缩放FP16的零梯度判结构断接。新head整段FP32、samplerFP32，其余仍AMP；native_hidden只指input_proj参数，不等于所有隐藏块或RGB内容收益。attempt3低scale32下溢、attempt4正式512→256通过三更新但独立诊断漏scale失败，均保留失败记录，不作为最终通过产物。

最终attempt5 passed。max_peak_reserved_mib按producer只覆盖updates与inference（8558MiB）；计入diagnostic_full_objective_timing的全部已测峰值为8564MiB。updates.decoder_calls首步非空rank为2是512→256 retry两次前向、empty rank为0；后续每批1次，须结合attempts解释。A/C各一次且固定顺序，不用作加速证据。

### 2026-10-27 Native短训练与资源调度字段

native runner复用旧progress每25更新rank0采样：last_global_microbatch_mean分母8，preclip_norm是DDP全参数AMP unscale后裁剪前L2；epochs mean分母2136，样本每rank534。E1/E10完整459，测试固定E10。native_resource_gate每臂4rank×2次global8更新，memory是rank进程allocator/实际nvidia驻留，非每卡全部进程；inference250为固定框资源fixture，无AP含义。owner per_arm_reservation_mib含进程上下文，allocator_caps仅torch上限，admissible_free_mib=实际free−共同margin−运行组未兑现预留；资源只用于准入，不用于loss/选权。恢复gate synthetic moment仅机制验证。

### 2026-10-28 Native离线归因字段

attribution rank_progress.images分母为rank分片114/115图，最后gather全459；correct/zero/donor固定C框score/sourcechunk。model_state_unchanged是预测前后权重/缓冲SHA，不是单流冻结。donor_map来自全459顺序+229，错配含RGB支持/geometry支持差异。paired_image_bootstrap95为同图500次seed20261021名义区间，不能解释训练种子不确定性；BoundaryAP2%/1%内带radius14/7px，与双侧净纠错band定义分开。

### 2026-10-28 Native全100轮调度与恢复字段

全量progress仍每25更新rank0，last_global_microbatch_mean分母8，mean_microbatch_loss分母2136；full validation每epoch完整459。best.score为val segm/mAP，strictgreater且ties earlier；test绑定best_checkpoint_sha256，不是固定E10。owner.stage等待waiting_for_analysis、running、failed、complete；dependencies仅完成门，不读指标。gate epoch0为gate_only/phase gate_no_validation/validation_performed=false，正式CP after_validation/true；rank_rng五流，合成gate不等同真实next-update轨迹。日志按action+arm+attempt时间戳隔离，资源caps/预留/512margin语义沿native短研究不变。

### 2026-10-28 共租控制覆盖审计

launch_manifest.co_rent_override明确reason/released_dependencies/time/controller_source_sha256/numerical_commit_unchanged；仅调度依赖变化，不参与loss/门控/选权。control_overrides存旧manifest/state/override，release拒已有GPU任务或失败owner；恢复从已核状态继续。

### 2026-10-28 Native归因中期结果状态与纠错分母

paired*.partial.json 的metrics各项已经500次，但complete=false表示整份四指标尚未结束，不能当全部完成；root_verified_results_snapshot显式GPU/CPU状态并绑定SHA。固定C权重zero/donor AP点差是推理依赖干预，含分布变化，不称配对训练净增益；训练净差看C−B。净纠错helper按固定bbox一对一匹配、四互斥区域统计，分母matched-instance pixels（跨实例重复），未匹配不纳入；没有完成产物不得报告像素结果。

### 2026-10-28 检测／掩码方向与框误差字段

遥测位于full root/telemetry而非原progress。gradient stats的detection_norm/mask_norm是FP32无增强trainmode固定global8均值原已加权loss对预注册参数组的L2、preclip；dot/cos在固定union None0算，零norm时null/status undefined，joint_connected_only仅补充，instance_auxiliary_norm独立。rank rows shared_direction_unaveraged_image是单图未除8范数，勿与主global8比较。support=both/det_only/mask_only/neither张量/元素数，native_refiner det无梯度属预期。CPU detection views每行固定score .05/.25/.5与IoU .5/.75，各自459图分母；TP/FP/FN守恒，FPduplicate/localization/background代理，FN保留前/后score候选bestIoU分桶。TP_bbox_errors每字段count=TP，中心dx/dy px及GTwh归一化、宽高px/logratio，signed/abs/P50/P90；尺寸GTmask与FPbbox不混同。全部定义及score跨view校准限制见Native遥测文档；字段只观察，不用于loss、门控、选权。

### 2026-10-28 SUTC 开关审计字段

新增 runtime.initialization_audit 的 matrix_arm/upstream/tail_evidence/UBR_present/UAV_instance_loss/UAV_proposals 是模型构造与路由的布尔身份审计，不是训练统计或收益证据；UAV_instance_loss/UAV_proposals 同时跟随 U，不能只关 hook 留下辅助监督。source_subset 对 S 为 none，对 T 为共享 RGB 子集。S 无 UAV 样本 loader、参数或候选；T 的 enabled=False 仅指上游/补框关闭，独立 prepare_uav 仍提取尾部证据。无新增逐步训练日志、loss、门控或选模字段。CPU 门不能代替真实四卡 S/T 验收或 AP 结论。

### 2026-10-28 视角数量审计与CPU接线门

SUTC initialization_audit.uav_view_ids/uav_view_count 是实际相机ID/配置数量，S为[]/0；subset_cache_manifest_sha256绑定精确子集产物，默认旧五视角缓存为null，不代表无深度。view_count_denominator固定5，geometry第3通道为最近表面1m内实际所选观察者数/5；不是所选数量比例，geometry.shape的5是通道数。新增CPU门 candidates分母16384×8候选槽，supported_pixels分母512²；RGB_files_read是本次真实加载文件清单，未选RGB保持空槽且不进encoder。门只有单train图001234，不是分组覆盖率/AP/四卡资源证据，不参与loss/门控/选权，无新增逐步训练遥测。正式训练必须拒绝仅工程图的局部缓存。

### 2026-10-28 三阶段wrapper身份、工程门与模态控制

launch_manifest.recipes各含arm/view_ids/rgb_content/geometry_content及canonical recipeSHA；protocol/source/order/cache SHA用于拒漂移，selection_lock绑定val选择的base和训练收据，不读test排名。gate.initialization的satellite/head/RGB/full_model SHA是更新前模型状态，不比较描述性audit；结构S/U/T/C共享sat/head，U/T/C共享RGB，后续各臂full初始状态与base一致。gate.updates为4rank×各2图的3次global8工程更新（rank3首步空GT），main_mask_gradient为仅loss_mask权重1/其余0且AMP scale/unscale后的DDP梯度，非AP或内容必要性证据；forced250是重复框资源fixture，8次chunk32原consumer，不是自然检测250GT。restore_gate只验合成moment/RNG恢复，不证明真实next-update数值重放。

stage queue状态仅控制面：结构不自动test，lock后eval独立队列；modality/views在全阶段train完成后test，不跨阶段自动发射。unknown预约15872MiB/card，实测后max(reserved+1024,resident+256)再留全卡1024，启动时预扣未兑现预约；非allocator cap，默认cap0。modalities.audit明确geo5[0:4]/candidate7[2,3,6]清零而保留深度对应/support/offset/yaw，不称无深度。S cache.kind=pure_satellite只核8个卫星数据身份文件，不依赖UAV样本缓存。训练逐步字段沿原global8/2136分母口径，无新loss或逐步模态日志。analysis summary为459图500次seed20261021名义图像区间，结构交互项同一抽样，不代替训练种子重复；分析不参与loss/门控/选权。
