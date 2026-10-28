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
