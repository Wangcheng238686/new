# PromptMiner-SAM2 (a3sgm) 代码库导读

> 结构重组终态（cleanup3/package-structure，2026-10-13）。本文件是当前代码的唯一权威地图；
> `docs/a3sgm_architecture_guide.md` 的 `文件:行号` 引用对应清理前提交 add7482，
> 机制描述仍然有效，行号请以本文件为准。每轮变更清单见 `docs/cleanup_rounds.md`。

## 1. 项目架构说明

仓库三层：**论文资产**（`icassp2026_promptminer/`、`fig1_components/`、`fig1_rows.png`、
图件脚本）＋ **模型代码**（`portable_sam2_explicit_coarse/` + 顶层 `sam2/` vendored 官方仓）
＋ **实验证据**（`docs/`（含追踪表与 evidence/）、`logs/`、`paper-reviews/`）。

模型 = RSPrompter-anchor 单流两阶段检测器（SAM2 Hiera-B+ LoRA 编码器 → PAFPN → RPN/RoI）
＋ **CSPM**（粗形状提示挖掘：RoI 特征 → 64×64 coarse logits → 2P2N 点 + 框 + dense canvas
三模态提示喂冻结 SAM2 PromptEncoder）＋ **UDPR**（margin 尾：K=64 最小|z|选点、±2 tanh 残差、
部署阈值锚定损失）。两个正式入口：WHU-1024 全量 fast150（roi_local 监督、PE 冻结）与
NWPU VHR-10 matrix300（full_image + roi_balanced_dice、PE 下采样栈 0.1×LR 可训练）。

### rsprompter 包：按域分子包（2026-10-13 重组）

UAV 双流扩展（2026-10-14）位于 `detection/dual_detector.py` 与独立 `uav/` 子包；UE 数据、训练与推理适配分别位于 `data/ue_*`、`train/ue_*`、`inference/ue_rebuild.py`。统一入口为 `scripts/ue_rgbd.sh`。完整模块地图、数据合同、实验命令和验收边界见 [双流实现与入口](dual_stream/IMPLEMENTATION_AND_RUNBOOK.md)。旧架构合同与旧推理重建模块继续作为冻结区。

`MODELS.build` 按 `type` 字符串注册，包内文件布局不影响指纹与 ckpt；根 `__init__.py`
是唯一注册门面（`RSPROMPTER_LIGHT_IMPORT=1` 可跳过重链导入，供张量级单测用）。

| 模块 | 职责 | 依赖 |
|---|---|---|
| `rsprompter/__init__.py` | 注册门面：re-export 全部 registry 类与加载器符号（原 models/models_sam2/shape_prior 三门面并入） | 各子包 |
| `rsprompter/architecture_contract.py` | **冻结区**：架构 ID 与 cfg.model sha256 指纹 | — |
| `rsprompter/ckpt_utils.py` | 严格加载工具（跨子包共用） | — |
| `detection/anchor_detector.py` | 顶层检测器：特征抽取/损失/预测入口、stride16 选择 | backbone、neck（经 registry 组装） |
| `detection/anchor_roi_head.py` | RoI head：采样、bbox 损失(fp32 可选)、mask 前向、三大损失接线、空秩 DDP 补偿、分块推理 | cspm.mask_head（经 registry 组装） |
| `detection/sam2_neck.py` | RSSAM2PAFPN（stride 4/8/16/32/64，五层 512ch） | — |
| `backbone/sam2_vision.py` | LoRA 编码器加载/冻结、no_mask_embed 加载、PE 权重加载、SAM2 严格加载器 | ckpt_utils |
| `cspm/mask_head.py` | 因果链核心：Step4-11 前向（coarse→dense canvas→挖点→PE→dense embedding→decoder→尾）、双坐标 predict | injector, miner, tail, decoder, helpers, targets |
| `cspm/mask_head_init.py` | `_MaskHeadInitMixin`：墓碑拒载 + 7 个 `_init_*` 构造契约（方法体自 mask_head 逐字搬移） | 同 mask_head |
| `cspm/mask_head_helpers.py` | mixin：dense canvas 变换粘贴、PE 输入选择、无效点置零、gate 有效值、框内 support、GT 点位遥测 | dense_prompt_utils |
| `cspm/mask_head_targets.py` | mixin：最终 mask 目标（fi/roi_local 双契约）、roi_balanced_dice、coarse 目标、loss_and_target | — |
| `cspm/shape_prior_injector.py` | SmallMaskDecoder（coarse 头，roi_only） | — |
| `cspm/shape_point_miner.py` | 2P2N 挖掘（无参数纯函数式，平台中心选取+三级 fallback） | — |
| `cspm/dense_prompt_utils.py` | raw_logits 变换、框内粘贴、无效恢复基底 | — |
| `cspm/coarse_mask_loss.py` | coarse BCE+Dice（epoch 权重调度、boundary 指标遥测） | — |
| `decoder/sam2_decoder.py` | SAM2 MaskDecoder 严格加载包装 + upscaled hook | vendored sam2, backbone |
| `decoder/decoder_tail_refiner.py` | UDPR margin 尾（耦合容器：梯度回传主干） | — |

### train/ 与 inference/：入口名不变，内部按主题分模块

两处均以脚本直跑（`python train/train_rsprompter_fusion.py` / `python inference/infer_from_checkpoint.py`），
子模块平级导入（sys.path[0] = 脚本目录），torchrun 与 smoke 门同样成立。

| 模块 | 职责 |
|---|---|
| `train/train_rsprompter_fusion.py` | 入口：setup + main 编排（642 行；DDP 顺序骨架：epoch 头、EMA 恢复、停训广播） |
| `train/cli_args.py` | `_env_flag` + `_parse_args`（argparse 全树+校验） |
| `train/config_snapshot.py` | git 状态捕获 + `_build_config_snapshot`（自描述快照组装） |
| `train/optimizer.py` | 参数组/LR 倍率 + 解耦辅助裁剪分区（含 2026-09-14 审计注释） |
| `train/ema.py` | ExponentialMovingAverage（tracking-only） |
| `train/checkpointing.py` | verified-best 保存 / resume / init-from 加载 / epoch 戳 |
| `train/telemetry.py` | 指标读取（best 选择）、前向遥测、dense 残差监视、DDP 参数同步审计 |
| `train/distributed.py` | DDP init（控制面超时）/停训广播/rank 工具/DDP 解包 |
| `train/gt_samples.py` | 训练/验证共用 GT 构建 + 验证 loader 分片 |
| `train/loops.py` | `train_one_epoch` / `validate_one_epoch` / `_finalize_epoch`（函数体自 main 逐字搬移；跨 epoch 状态显式穿线） |
| `inference/infer_from_checkpoint.py` | 入口：parse_args + main 编排（419 行） |
| `inference/model_rebuild.py` | 快照解析 + **契约 compat 冻结区** + 注册构建 + 严格加载（搬移后函数文本逐字节不变） |
| `inference/eval_records.py` | 数据集契约 / loader / 样本构建 / RLE 记录 / checkpoint 报告 |

### 其余目录

| 文件 | 职责 |
|---|---|
| `data/` | WHU COCO 实例数据集 + 训练/验证 loader（含 VHR10 格式） |
| `utils/coco_eval_utils.py` | COCO GT/DT 构建与评测唯一实现 |
| `configs/a3sgm_{whu1024,vhr10}.py` | 静态模型配置（指纹=论文 ckpt 训练期契约） |
| `configs/ablations/vhr10_*.py` | 论文消融 P/B/P+B/P+B+M 的 delta 配置（`_base_` 加减键） |
| `configs/environment*.sh` | 机器路径（python/SAM2/数据/ckpt 根） |
| `scripts/ablations/*.sh` | 两入口瘦启动器（ABLATION_CONFIG 可指 delta） |
| `scripts/infer_{whu,nwpu}_checkpoint.sh` | 推理入口包装 |
| `scripts/smoke/check_a3sgm_*.py` | 等价性回归门（加载+前向哈希+DDP+resume，金样 json 同目录） |
| `tools/` | bootstrap 配对统计、Boundary-AP、原尺度评测、可视化 |

## 2. 代码阅读路线（建议顺序）

1. `docs/a3sgm_architecture_guide.md` §0-§1（一页定位 + 架构图）——机制全景。
2. `configs/a3sgm_vhr10.py` —— 模型长什么样（静态 dict，键名即模块语义）。
3. `rsprompter/detection/anchor_detector.py`（214 行）—— 数据怎么流进两阶段。
4. `rsprompter/detection/anchor_roi_head.py` —— 训练/推理在 RoI 层怎么组织（三大损失在哪接线）。
5. `rsprompter/cspm/mask_head.py`（728 行）—— **核心**：先读类 docstring 的 Step4-11 流程图，
   再按 `_forward_coarse → _forward_prompt_encode → _forward_dense_embeddings → forward →
   _predict_by_feat_single` 顺序读；构造期契约在 `mask_head_init.py`。
6. `rsprompter/cspm/shape_point_miner.py` 附录级细节（架构指南附录 A 有逐槽讲解）。
7. `rsprompter/decoder/decoder_tail_refiner.py`（228 行）—— margin 语义与 DDP 归一。
8. `train/train_rsprompter_fusion.py` 的 `main()` → `train/loops.py` —— 训练循环/选模/断点。
9. `inference/model_rebuild.py` 的 `_resolve_model_config` —— ckpt 自描述重建与契约校验。

## 3. 关键接口清单（入参/返回/副作用）

| 接口 | 入参 → 返回 | 副作用 |
|---|---|---|
| `RSPrompterAnchorMaskHeadSAM2.forward(x, image_embeddings, image_positional_embeddings, roi_img_ids, high_res_features, boxes, …)` | RoI 特征 [N,512,14,14] 等 → `MaskHeadOutput(mask_logits z′, coarse_outputs, tail_outputs)` dataclass | 写 `self._forward_telemetry`（遥测总线，detached） |
| `DecoderTailPointRefiner.forward(logits[N,1,H,W], upscaled[N,32,H,W], mask_tokens[N,1,256])` | → `(refined z′, {indices, delta, selected_abs, refined/base_selected_logits, base_logits_detached})` | 无（耦合容器：不 detach，梯度可达主干） |
| `DecoderTailPointRefiner.point_loss(tail_outputs, targets)` | → `(loss, TAIL/* 遥测 dict)`；DDP 全局正负点归一 | 两次 all_reduce（空秩经 `empty_point_loss` 镜像） |
| `ShapePointMiner.forward(coarse_logits[N,1,64,64], boxes[N,4], image_size)` | → `(coords[N,4,2], labels[N,4]∈{1,0,-1}, SP/* 统计)`；全程 no_grad | 写 `_last_local_yx/_labels`（GT 遥测用） |
| `_MaskHeadTargetsMixin.loss_and_target(mask_preds, sampling_results, batch_gt_instances, rcnn_train_cfg, final_mask_pos_priors)` | → `{loss_mask, mask_targets, debug_stats}`；fi/roi_local 双路径 | 无 |
| `RSPrompterAnchorRoIPromptHead.mask_loss(...)` | → `(losses, telemetry)`（损失通道纯损失；`DET/*` 精度键走遥测） | 写 `_last_coarse_stats` 等遥测 |
| `architecture_contract(model_config_dict)` | → `{schema_version, architecture_id, model_fingerprint}` | 纯函数，**冻结** |
| `train/loops.py train_one_epoch(...)` | args/model/loaders/优化器组 + 穿线计数器 → `(avg_loss, lr_groups, ema_update_steps, nonfinite_batches, ddp_sync_audited)` | 训练一个 epoch（含批内 collective） |
| `train/loops.py validate_one_epoch(...)` | loader/分片参数/GT 缓存 → `(val_loss, val_metrics, ema_applied, cached_gt_records)` | 验证（rank0-only 或分片 gather） |
| `train/loops.py _finalize_epoch(...)` | epoch 结果 + best-*/早停状态 + 存档上下文 → 更新后的跨 epoch 状态 + `stop_training` | rank0 写四别名 ckpt |
| `infer_from_checkpoint.py --checkpoint … --build-only` | 重建模型+strict load，打印契约 | 进程内设置 SAM2_REPO/CKPT env |
| 训练器 `--config configs/a3sgm_*.py` | DDP 训练，产出四别名 ckpt（best/bbox/composite/last + EMA state） | checkpoint 含 config_snapshot（自描述） |

## 4. 项目编码规范（新增代码请遵守）

1. **行为即契约**：任何影响 cfg.model 的改动都会改变架构指纹并拒载旧 ckpt——改模型形态前先跑
   `scripts/smoke/check_a3sgm_checkpoint_load.py` 与 `check_a3sgm_forward_equivalence.py`。
2. `rsprompter/architecture_contract.py` 与 `inference/model_rebuild.py` 的契约比对区
   （含两个 compat 变体与 `_is_legacy_pre_tail_id`）为冻结区，只加注释不改逻辑。
3. 死臂参数模式：`__init__` 参数保留（ckpt 嵌入 config 会传键），启用即 `raise ValueError`
   并写明属于哪个已删臂——禁止静默吞掉。
4. 遥测前缀约定：`SP/*`（点挖掘）`COARSE/*`（粗头）`GATE/*`（dense 门控）`DENSE/*`（dense 通道）
   `TAIL/*`（尾）`BOX/*`/`JITTER/*`（提示框总线）`DET/*`（检测精度键）；计数键由 roi_head 的
   `_ratio` 派生命比率；遥测只写日志不参与 loss。
5. 注释只写"为什么"（如 margin 必须锚定部署阈值 logit、空秩双 collective 防死锁、
   `**kwargs` 吞旧 ckpt 键的原因）；不复述代码在做什么。
6. 禁第三方新依赖、禁一行式压缩、单一职责、guard clause 提前返回、变量就近定义。
7. 配置改动走 `configs/ablations/` 的 `_base_` delta（加减键），不复活 env 开关。
8. GPU 上的数值等价验证必须先做同代码双跑标定噪声带（scatter 原子操作使 bit 级复现不可达）。
9. 结构搬移纪律（2026-10 新增）：函数体逐字搬移 + AST 等价对账；跨模块常量跟随唯一消费者；
   smoke 门用的 `/tmp` delta 配置内嵌 worktree 绝对路径，换 worktree 必须重生成。


## 5. 文档同步记录

- 2026-10-14：UE新增覆盖区门控/置信/有效缩放、按真实裁剪分区的梯度与LR、跨rank有限forward均值/分母、AMP跳步和训练峰值显存。生产者仍在 `uav/fusion.py`，训练汇总仍在 `train/ue_telemetry.py`；完整范围、公式与判读边界同步至 [DEBUG_FIELDS](DEBUG_FIELDS.md)。没有新增模型配置键或改变融合计算。

`scripts/smoke/check_ue_telemetry.py`：两rank缺失/非有限字段分母与AMP+零支持融合输出/梯度不变性检查。

- 2026-10-14 启动修正：UEEpochTelemetry将字符串或torch.device输入统一为torch.device，覆盖实际trainer接口，重训前同一分母smoke复验。

- 2026-10-14：`data/ue_labels.py`统一clean掩码/local-global映射校验；dataset与prepare共用，prepare在几何计算前排除标签合同失败样本。未知ID仍拒绝，不转成背景。

- 2026-10-15：完成EXP-0006/0007的80轮产物核验、完整153图strict EMA推理、correct/zero/shift依赖诊断与探索性图像bootstrap，结果和判读边界见 `docs/dual_stream/UE_AD_RESULTS_20261015.md`。无训练/模型/遥测生产者变更；字段字典保持原定义，raw训练遥测与EMA验证分开解释。

- 2026-10-15：补充7个典型改善/退化/持平案例，复用固定EMA推理产物、逐目标top100最大IoU与正式COCO逐图匹配，核对几何支持和预测框限制，见 `docs/dual_stream/UE_CASE_AUDIT_20261015.md`。仅新增分析文档，不改模型/标签/遥测定义。

- 2026-10-15：复用冻结EMA完成3例/42GT共同框mask诊断，12次原框逐像素复现；追加RPN/解码框/score-NMS阶段审计并核对最终框score完全复现，见 `docs/dual_stream/UE_FIXED_BOX_DIAGNOSIS_20261015.md`。仅分析文档，无生产模型/指标/遥测变更。

- 2026-10-15：完成当前冻结开发子集153验证图、4496次建筑出现的全候选阶段与三种共同框审计，修正先前42GT代表性判断，见 `docs/dual_stream/UE_DEV_SUBSET_AUDIT_20261015.md`。轻量JSON汇总与两项独立审核归档；仅分析文档，无生产模型/遥测变更。

- 2026-10-15：新增`scripts/analysis/ue_rgb_content_prepare.py`（冻结供体映射/预注册）、`ue_rgb_content_probe.py`（三模式strict EMA完整推理/共同框/不变量）、`ue_rgb_content_summary.py`（AP配对bootstrap与建筑/尺寸汇总），均为固定路径诊断脚本。153图RGB内容干预已完成，结果见`docs/dual_stream/UE_RGB_CONTENT_20261015.md`，仅支持配对RGB敏感性；生产模型、推理入口及遥测定义未改变。

- 2026-10-15：新增`scripts/analysis/ue_fusion_route_probe.py`（冻结FPN/SAM四组合特征交换、对角精确回放）、`ue_fusion_route_summary.py`（GT框IoU条件效应/交互、1000次配对图像bootstrap）。153图两套供体均完成，结果见`docs/dual_stream/UE_FUSION_ROUTE_20261015.md`；仅定位本干预主要沿SAM，未训练、未改生产接线/遥测。

- 2026-10-15：新增`scripts/analysis/ue_sam_strength_probe.py`（只缩放SAM融合有效残差0/.5/1，单次特征/RPN/bbox复用，完整mask及GT框验收）、`ue_sam_strength_summary.py`（预定半强度对比、AP/IoU配对bootstrap、尺寸/建筑汇总）。153图完成，不支持统一减弱；见`docs/dual_stream/UE_SAM_STRENGTH_20261015.md`。无训练或生产模型/遥测变更。

- 2026-10-15：新增`scripts/analysis/ue_spatial_local_probe.py`（SAM残差四方向±4px/对称平滑、相机内部代数和GT边界/支持统计）、`ue_spatial_local_summary.py`（8contrast同时区间、局部分组和半强度关联）。153图六对照完成，未达到方向/平滑改善门，见`docs/dual_stream/UE_SPATIAL_LOCAL_20261015.md`。P2列为待验证候选，未新增入口；无训练或生产模型/遥测变更。

- 2026-10-15：新增`scripts/analysis/ue_p2_transfer_probe.py`（借原P3融合器向P2原生/供体/上采样/压力对照，固定框只看mask，非P2精确不变验收）、`ue_p2_transfer_summary.py`（条件AP/GTIoU配对bootstrap和官方层级分组）。153图五组完成，三支持门未过，不追加完整检测；见`docs/dual_stream/UE_P2_TRANSFER_20261015.md`。未训练/改默认入口/新增生产遥测。

- 2026-10-15：新增`scripts/analysis/ue_p2_prompt_trace.py`（只读捕获RoI/coarse/点/dense/SAM/tail，精确回放上一轮四组GT框终点）、`ue_p2_prompt_trace_summary.py`（阶段到达、粗形状/点质量、密集框支持与配对图像区间）。153图完成，变化可达但质量未稳定改善，见`docs/dual_stream/UE_P2_PROMPT_TRACE_20261015.md`；无训练/生产接线/遥测变更。

- 2026-10-15：新增`scripts/analysis/ue_gt_prompt_oracle_probe.py`（固定GT64±4替换粗形状，原点/dense路径联合生成，视觉/框/权重不变）、`ue_gt_prompt_oracle_summary.py`（独立CPU身份/SHA/不变量/质量及seed1760配对图像区间）。153图完成，最终GT框IoU+.019063，提示条件改善空间门通过，见`docs/dual_stream/UE_GT_PROMPT_ORACLE_20261015.md`；主代理审计脚本与轻量数值归档在`docs/dual_stream/probes/results_20261015/GT_PROMPT_ORACLE_ROOT_AUDIT.py`及对应JSON。未训练/改生产接线/遥测。

- 2026-10-15：新增`scripts/analysis/ue_gt_prompt_factorial_probe.py`（真实decoder sparse/dense四组合交换，固定box tokens/视觉、两对角回放及空dense支持阴性）、`ue_gt_prompt_factorial_summary.py`（两个主contrast Bonferroni区间、条件差/交互及来源SHA验收）。153图完成，点only+.004540未过幅度门、denseonly+.015619过门，见`docs/dual_stream/UE_GT_PROMPT_FACTORIAL_20261015.md`。主代理独立审计脚本/轻量结果在`docs/dual_stream/probes/results_20261015/GT_PROMPT_FACTORIAL_ROOT_AUDIT.py`及对应JSON。无训练/生产接线/遥测变更。

- 2026-10-15：新增`scripts/analysis/ue_dense_shape_confidence_probe.py`及`ue_dense_shape_confidence_summary.py`（原点固定的原dense/预测±4/GT±4，153图，等幅形状差+.016068）；新增`ue_complementarity_probe.py`及`ue_complementarity_summary.py`（融合前卫星+对齐BEV随机投影、固定128train/153val特征、train-only五组ridge，UAV两个主Brier改善低于.001门）。精确脚本/轻量结果和主代理独立重算在`docs/dual_stream/probes/results_20261015/`可追溯；综合见`UE_PARALLEL_DIAGNOSIS_20261015.md`，源码梯度/optimizer/clip/合同见`UE_GRADIENT_DESIGN_AUDIT_20261015.md`。无正式训练/生产模型或遥测变更。

- 2026-10-15：新增`scripts/analysis/ue_readout_capacity_audit.py`（固定缓存谱/空间变异/旧读出复现）、`ue_bev_error_strata.py`（traincut分层/占据区域/GTdense空间关联）、`ue_bev_geometry_quality_audit.py`及`ue_bev_geometry_quality_summary.py`（固定六图rawdepth缓存重建/单表面量化和候选相容性）。本轮无新fit/网络forward/GPU/训练/生产遥测变更；综合见`UE_BEV_WEAKNESS_DIAGNOSIS_20261015.md`，原始几何仅/tmp保存NPZ，复核脚本与轻量JSON在probes/results_20261015。

- 2026-10-15：新增`scripts/analysis/ue_bev_attention_probe.py`（六图两层全query只读QKV/几何权重、生产SDPA样本对账和smoke特征逐值不变）、`ue_bev_attention_summary.py`（分组/逐图/query及source加权/uniform/熵与尾部CPU汇总）；主代理独立复算脚本、轻量JSON、协议和完整审核归档。结果见`docs/dual_stream/UE_BEV_ATTENTION_20261015.md`。固定路径依赖沿旧/tmp helper链，精确归档不作通用入口；无训练、候选干预、生产接线/配置/遥测变更。

- 2026-10-15：新增`scripts/analysis/ue_bev_tail_map.py`（真实投影归一化/独立FP32生产对账）、`ue_bev_tail_map_roundoff_diagnosis.py`（首次数值门失败定位）、`ue_bev_tail_association.py`（已有GT框RLE误差关联/覆盖分母与图等权）。结果见`docs/dual_stream/UE_BEV_TAIL_ASSOCIATION_20261015.md`；原FP64门FAIL、显式v2及bool分位数类型修复均留档。无GPU/新forward/fit/训练/生产遥测修改；独立结果审核PENDING。

- 2026-10-15：BEV高度尾部关联最终独立结果审核已补齐PASS；36maps、RLE/ROI/区域及主相关重算通过，替代此前PENDING状态，不改变原结果或生产代码。

- 2026-10-15：新增`scripts/analysis/ue_stage_shape_extract.py`、`ue_stage_shape_helpers.py`、`ue_stage_shape_fit.py`、`ue_stage_shape_summary.py`，共同支持下候选等权与CVT的冻结16²占据读出；仅12离线小头固定1000steps，正式模型不更新。SOURCE/protocol/manifest/轻量JSON/root复算归档，四主门FAIL；独立最终结果审核PENDING，见`UE_STAGE_SHAPE_20261015.md`，无生产配置/遥测变更。

- 2026-10-15：候选等权/CVT形状读出独立正式结果审核补齐PASS，完整cache/主CI及固定小头GPU重放48输出bitexact，原负门结论不变。

- 2026-10-15：新增`scripts/analysis/ue_projection_scale_extract.py`、`ue_projection_scale_fit.py`、`ue_projection_scale_summary.py`，同CVT粗/细池化重建读出，旧公共cache逐值复用；仅6图smoke已运行并独立PASS，full/fit未运行，资源竞争证据与恢复步骤已归档，见`UE_PROJECTION_SCALE_20261015.md`。生产模型/配置/遥测无改动。
