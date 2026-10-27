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

- 2026-10-27：新增 `ue_rgbd_two_seed_ddp_20261027.py`、`ue_rgbd_two_seed_owner_20261027.py`（均在scripts/probes）及`configs/ablations/ue_rgbd_two_seed_ddp_20261027.json`；原RGB-D模型44/45双seed各占四卡并可共租，100轮26700更新、全val选best、完训test，通用独立初始化。每epoch保存完整CP与四rank RNG，源码/环境/全100轮图序启动前固化。命令、调度和DDP与历史单卡不能逐位等价的边界见[两seed四卡复现](dual_stream/UE_RGBD_TWO_SEED_DDP_20261027.md)。既有在训源未修改。

- 2026-10-26：新增 `scripts/analysis/ue_joint_counterfactual_20261026.py`，固定正确RGB-D E67进行A正确/B均值RGB/C整组供体RGB/D独立参考面/E关闭全部UAV路由的459test诊断；复用已固化评价内核，原在训源保持不变。文件/命令、GPU2准入、输入文件审计与E的同检查点共同适应边界见[五组干预说明](dual_stream/UE_JOINT_COUNTERFACTUAL_E67_20261026.md)。先通过五组单图不变量门，再执行B–E全量并对A做同图配对统计。

- 2026-10-26：新增 `scripts/analysis/ue_joint_snapshot_eval_20261026.py`，按用户要求固化原四臂当前val-best并在GPU2串行评价完整train/test；新增[补充评价说明](dual_stream/UE_JOINT_TRAIN_TEST_SNAPSHOT_20261026.md)。复用原推理及COCO内核，不修改在训数值源。固定权重epoch/预算、预测/源码SHA及同图Mask AP区间独立保存；不同预算和训练集拟合的限制明确披露，不能冒充E60或原100轮最终结果。

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

- 2026-10-16：完成阶段性17项证据汇总和三名新子代理的独立量纲/对齐/消费审查，见 `docs/dual_stream/UE_STAGE_RETROSPECTIVE_20261016.md`。新增仅文档、固定CPU原始相机深度数值抽样脚本及轻量JSON；无训练、GPU、生产模型/缓存/遥测定义变更。

- 2026-10-16：新增H1数据优势/H2表征损失/H3共适应三路独立判断与交叉质疑，见 `docs/dual_stream/UE_HYPOTHESES_JUDGMENT_20261016.md`；仅定性原RGB接触板复核，无模型/训练/遥测定义变更。纠正将条件依赖写为已证能力让渡的归因措辞。

- 2026-10-16：根据用户澄清将核心目标对准相邻建筑实例粘连，纠正把跨视角信息优势收窄为GSD优势的表述；新增 `docs/dual_stream/UE_INSTANCE_SEPARATION_DIRECTION_20261016.md`。仅研究方向/评价定义草案，未计算新指标、修改模型或启动训练。


- 2026-10-16：新增`scripts/analysis/ue_adjacent_pair_audit.py`（既有153图GT/DT、全局一对一配对、接触/近邻及候选摘要定位）、`ue_adjacent_pair_cases.py`（六类首序案例轮廓图）、`ue_adjacent_pair_sensitivity.py`（明确事后的触边/small及多块覆盖代理）。结果见[邻接对定位](dual_stream/UE_ADJACENT_PAIR_DIAGNOSIS_20261016.md)，轻量身份/数字及独立复算存`dual_stream/probes/results_20261016/adjacent_pair/`。仅CPU已有产物；无新forward、fit、训练、生产接线/配置/遥测定义变更。数值/实施审核PASS，报告终审三处措辞已主代理修正，完整记录保留。


- 2026-10-16：完成两个子代理的RoI-UAV冻结可行性探针及第三代理独立复算/审图。仅CPU已有真实候选与原RGB-D，无forward/fit/训练/标签清洗或生产遥测变更；候选存在空间，但缺关键消费证据，不准入训练。见[综合报告](dual_stream/UE_ROI_UAV_FEASIBILITY_20261016.md)。精确脚本归档在`docs/dual_stream/probes/results_20261016/roi_feasibility/`：`candidates/probe.py`为有限真实候选去重/一对一/NMS兼容；`uav_evidence/freeze_cases.py`为固定六例选择；`uav_evidence/run_probe.py`为camera-Z投影/位置阴性/12图；`review_recompute.py`与`review_uav_recompute.py`为独立数值复算。它们固定历史/tmp路径，不是通用运行入口；大逐对JSON/图留/tmp。完整复核PASS。


## 2026-10-21 最小归档单卫星基座

统一入口新增 `scripts/ue_rgbd.sh minimal train|infer|run`；`run` 调用
`scripts/run_ue_minimal_experiment.py`，以单进程顺序完成全字节checksum预检、train/val训练、
封存val最佳segm权重、最后一次test推理。`data/ue_minimal_dataset.py`提供独立COCO RLE读取、
固定split和train-only噪声/ColorJitter；`configs/ablations/ue_minimal_single.py`保持封版coarse64、
仅覆盖max_per_img250；`scripts/smoke/check_ue_minimal.py`负责真实batch2与AP250校准。
新dataset CLI名为`ue_minimal`，新快照dataset_format为`citysample_minimal`，推理通过现有
`ue_rebuild.resolve_ue_model`严格按嵌入model_config重建，数据按归档SHA核定。
旧UE四臂、冻结architecture_contract与inference/model_rebuild.py不变。
训练100epoch、4×2×1、全组LR5e-4、globalclip.5、EMA关；val选segm AP250最佳。
详见 [训练协议](dual_stream/UE_MINIMAL_SINGLE_20261021.md)。

2026-10-21 数据再审计：新增[旧双流数据问题与方向](dual_stream/UE_DATASET_REAUDIT_AND_DIRECTION_20261021.md)，
`docs/dual_stream/probes/dataset_reaudit_20261021/`归档两份CPU分析脚本及逐图证据，不接训练入口。
包含新旧153图独立mask对比、固定32图深度观测支持和旧train→新test454图身份风险。
EXP-0017生产实现与数据保持不变，后续UAV仍待新单流固化。

### 2026-10-26 候选先选择与stride4几何旁路

新增`data/ue_candidate_evidence.py`（由FP64锚点/相机复算7维候选属性）、`rsprompter/uav/candidate_detail_interaction.py`（Q：SAT block1 query先选RGB候选；G：多候选几何在128²零初始化写回）、`configs/ablations/ue_candidate_detail_protocol.json`（7条固定E10训练与内容对照）。新运行/调度/分析分别为`scripts/probes/ue_candidate_detail_run_20261026.py`、`ue_candidate_detail_owner_20261026.py`、`ue_candidate_detail_analysis_20261026.py`。原单表面CNN、block4、候选解码、损失和全部在训数值源不改；新源与产物独立，旧E10仅推理参考。三个真实GPU门已通过，正式完整结果未出。详见[机制与实验合同](dual_stream/UE_CANDIDATE_DETAIL_20261026.md)。本轮是保留锚点的选择/消费干预，不是恢复所有深度点或完整3D；新加入真实mask面积尺寸、密度和固定误检预算的小目标召回评价。未改现有生产入口或冻结架构合同。

### 2026-10-26 用户修订：本轮每个实验四卡DDP

单卡新Q/G矩阵已显式停止并留档；由`ue_candidate_detail_ddp_20261026.py`接管新实验，四卡batch1×accum2=global8，rank CPU Gloo汇总预测后算整集AP，不平均rank AP；DDP基线也重跑，避免与单卡随机流/分布式归一化混淆。新增`ue_candidate_detail_ddp_owner_20261026.py`按四卡共同余量共租组/排队，`ue_candidate_detail_ddp_analysis_20261026.py`仅比较同四卡合同，`ue_candidate_detail_repro_20261026.py`核定运行时import/Git blob/环境/输入身份，配置为`ue_candidate_detail_ddp_protocol.json`。原100轮在训源保持不变。文件、10条矩阵、随机状态恢复及复现说明见[四卡实施](dual_stream/UE_CANDIDATE_DETAIL_DDP_20261026.md)。旧单卡启动记录为历史，不表示当前执行方式。

### 2026-10-26 独立审查与 FROC 同分组修正

独立审查建议继续四卡10臂；确认冻结分析器的逐检测前缀 FROC 可拆开同分预测。新增独立 `scripts/analysis/ue_candidate_detail_froc_ties_20261026.py`，只在完整同分组端点计算 score 阈值可实现的小目标召回；保留原匹配和所有训练/AP代码。`--watch-root ROOT --owner-pid PID`监听固定24份完整评价，在 `analysis_corrected_froc/{val,test}`输出权威 FROC，绑定 records/complete/checkpoint/source SHA；不覆盖原分析。合成同分反例、100次无同分等价和真实459图顺序反转门已通过。详见四卡实施文档的审查补充。

### 2026-10-26 Q/G完成与控制面恢复

Q、G正确内容训练各完成固定E10/2670更新，但原owner已消失且state仍标running；其余8训练未启动，不能标整轮完成。新增 `scripts/probes/ue_candidate_detail_ddp_recovery_20261026.py`，只恢复调度，不改93份冻结源及数值合同：独占原lock，核原owner/rank不存在、原manifest/DAG相等、完成receipt/CP SHA/10轮顺序记录后跳过完成臂；有部分训练权重或意外评价即停止要求专项审查。旧state/owner launch归档recovery_20261026；使用脱离会话的Popen重启owner，继续四卡共租与原固定E10评价。退出原因未获证据，不认定OOM或正常整轮完成。

### 2026-10-26 正向实验恢复归档

新增`scripts/analysis/ue_positive_experiment_recovery_20261026.py`，仅CPU审计与复制两组历史source/初始化/固定CP/关键预测/逐轮记录，不改训练源。实例内部交互14个E10及联合四臂E50可找回；E60仅指标、无独立权重。外部归档`ue_positive_recovery_20261026/recovery.json`绑定SHA/代码提交；原source未跟踪owner/统计/协议按原字节补入Git，不能把归档当实际训练/AP重现。详情见[正向实验恢复](dual_stream/UE_POSITIVE_EXPERIMENT_RECOVERY_20261026.md)。无新增训练遥测字段。

### 2026-10-26 原四臂后续验证独立讨论

三方向只读审查形成[后续验证决定](dual_stream/UE_FOUR_ARM_NEXT_VALIDATION_REVIEW_20261026.md)：完成原100轮与既定val-best/test，先拆原路径/追加候选净纠错，再读完已有Q/G同DDP矩阵，不追加重复短训练。纠正此前过度简化：原u128直达128²实例头，只有卫星block4回写路径经pool64；新高分辨率RGB入口是另一个独立干预。记录E67原四臂与QG E4观察及限制；无数值源、选权或训练遥测变更。

### 2026-10-27 UBR收敛提案

依据用户给定两份设计文档，新增[UBR收敛提案](dual_stream/UE_UBR_CONVERGED_PROPOSAL_20261027.md)：先用统一UBR替换尾部，off/constant/real三臂固定上游、几何端口单变量；保留coarse，margin换带内监督，不默认删实例发现头。G互补性和RGB候选读取分阶段处理。补连续合成/阈值、梯度分工及GroupNorm裁剪不等价风险。仅文档，不改在训合同/数值源/遥测。

### 2026-10-27 UBR修订保留实例头与loss

用户确认先替换UDPR、保留当前G补充实例头/中心框loss。同步修订统一设计、收敛提案及讨论备忘：双流上游固定，D0/Dc/Dr只变UBR几何端口；纯卫星S另列。固定权重检测各1/final1/coarse0.1/UBR0.1/实例1（size内部0.1），明确已有loss不二次加权和UBR带内分母。仅文档，不改数值源/配置或新增训练遥测；无任务发射。

### 2026-10-27 完整G＋UBR结构与独立审查

新增[完整结构展示](dual_stream/UE_G_UBR_FULL_ARCHITECTURE_REVIEW_20261027.md)及[独立审查](dual_stream/reviews/UE_G_UBR_INDEPENDENT_REVIEW_20261027.md)：明确已有G/旧RGB均值/双向交互/128²实例头与拟建40ch UBR；图区分候选7维几何与native单表面5ch。审查有条件推进，长训练前补邻居训练来源、补充框训练覆盖、初始化梯度、空满mask和真实资源门。仅设计记录，无活跃模型/配置/日志字段改变。

### 2026-10-27 G＋UBR首批组件实施

按用户授权由子代理实施、主代理审核，新增[分批实施计划](dual_stream/UE_G_UBR_IMPLEMENTATION_PLAN_20261027.md)和`configs/ablations/ue_g_ubr_development_contract_20261027.json`。P0新增`rsprompter/uav/unified_boundary_refiner.py`（40ch、无界残差、SDT与按图平均raw带内loss）、`rsprompter/uav/ubr_prediction_context.py`（metadata先NMS再只解码保留预测、ID/lineage/self排除）、两份对应tests及`scripts/probes/ue_ubr_component_gate_20261027.py`（四rank synthetic组件更新/250实例推理）。组合CPU16项通过；四卡门记录单独落盘，不能视为整个SAM/G接线/资源或AP成功。P1前必须核roi_local概率paste，P2使用原256/.25来源盲采样，不加来源quota；P3再改容量/读取。没有修改旧数值源或发射全量训练。

### 2026-10-27 P0四卡组件验收

`8104ed2`独立UBR与context组件CPU16项通过，四rank各4次AMP更新/参数SHA一致/空实例rank通过；每rank顺序250个512² UBR推理通过。原始`ue_ubr_component_gate_20261027/complete.json`与`reviewed_acceptance.json`绑定源码与提交。训练allocator reserved678MiB、推理394–396MiB为组件局部结果，不是完整双流资源/AP。进入P1前先解决roi_local probability-paste坐标门，尚未完整模型接线或长训练。

### 2026-10-27 UBR坐标适配与历史接线裁定

新增`rsprompter/uav/ubr_coordinate_adapter.py`：native32ch与sigmoid概率同bbox grid映射到图像，finite-logit epsilon1e-6，推理保留原概率恢复未改动像素；不接生产head。`tests/test_ubr_coordinate_adapter.py`对照真实FCN；`scripts/probes/ue_ubr_adapter_gate_20261027.py`四卡三精度合成适配门。CPU组合26项、四卡三精度通过；不是完整检测器/AP。新增审计`dual_stream/UE_UBR_WIRING_AUDIT_20261027.md`及reviews下history/independent review，配置与实施计划同步。历史明确不允许全图UBR输出送旧28² ROI loss；修订为保留native主mask监督，UBR直接全图band监督；active来自当前预测pool匹配，GT不构建邻居。P1实际head、正式几何、全pool+active反传资源仍待完成。

### 2026-10-27 G＋UBR实际接线与首轮四卡运行器

新增`rsprompter/uav/ubr_detector_integration.py`、`tests/test_ubr_detector_integration.py`、`scripts/probes/ue_ubr_runtime_20261027.py`及`ue_ubr_detector_gate_20261027.py`。原decoder三元组不改，构造前禁UDPR，native主loss保持，UBR全图band监督；所有参数在satellite/DDP注册。context/predict复现旧source段＋默认32批，active逐实例完整checkpoint。33项CPU门及真实四卡attempt3通过：250自然pool zeroUBR raw/prob/binary/box/score exact，真实更新/64匹配预测/空rank/250压力已测，非AP或续训证明。

新增`configs/ablations/ue_g_ubr_short_protocol_20261027.json`、`scripts/probes/ue_ubr_short_run_20261027.py`、`ue_ubr_short_owner_20261027.py`与[短研究说明](dual_stream/UE_G_UBR_SHORT_STUDY_20261027.md)：fresh共同初始化、三臂10轮机制诊断、每实验4卡，full459val E1/E10、全部结束后固定E10 test、不按test选权。每epoch after_epoch保存与5类rank RNG恢复；owner最多2组按4卡headroom共租、第三排队。旧93份数值源保持不变，未启动100轮。

### 2026-10-27 UBR短比较已发射

代码提交5b719ee，四卡恢复门实际通过（模型/optimizer synthetic moment/scaler/五类RNG）。owner1407943在`ue_g_ubr_short_20261027`启动geometry与no_tail两个四卡组，zero_geometry按合同队列；94份数值源已绑定，完整预算未启动。已见参考臂E1真实成功更新，尚无本轮AP结论。实际rank/状态见[短研究启动记录](dual_stream/UE_G_UBR_SHORT_STUDY_20261027.md)。

### 2026-10-27 用户要求三组四卡共租

新增`scripts/probes/ue_ubr_short_adopt_owner_20261027.py`，按用户指示移除原两组限制，审计后仅替换owner、保留已有训练rank，接管并发射zero_geometry第三四卡组。每组预约7042MiB＋每卡全局2048余量，仍按实际4卡余量准入；归档初始owner状态、数值提交5b719ee不变，控制器另有Git/SHA，test保持串行。

### 2026-10-27 UBR效率实现与新三臂重启

新增`rsprompter/uav/ubr_detector_fast.py`和`tests/test_ubr_detector_fast.py`：同模型/512网格/active64/原loss下，context只概率paste、邻居sum-self、trainSDT局部缓存、去嵌套checkpoint，固定训练1/推理4。`scripts/probes/ue_ubr_fast_gate_20261027.py`做真实四卡loss/grad/离散/250预测/完整目标资源；`ue_ubr_fast_runtime_20261027.py`和`ue_ubr_fast_run_20261027.py`独立复用相同参数与oldengine；`configs/ablations/ue_g_ubr_fast_protocol_20261027.json`保持10轮合同；`ue_ubr_fast_owner_20261027.py`修全repo源闭包含SAM2lazy依赖，按新实测峰值共租三组。`scripts/analysis/ue_ubr_fast_restart_audit_20261027.py`审计并停止旧owner/两组，原输出完整保留。旧no_tail E10真实完成，owner因漏vendor2源误报failed，非漂移。实现与验收/重启范围见[效率记录](dual_stream/UE_G_UBR_EFFICIENCY_20261027.md)。
