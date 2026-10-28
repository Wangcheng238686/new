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

2026-10-21 文献与开发设计：新增[现有数据证据约束下的文献与开发方向](dual_stream/UE_LITERATURE_AND_DEVELOPMENT_20261021.md)，覆盖Multi-View SAM、DeepViewAgg、CVT、HQ-SAM、SG-BEV/CRFFNet、S²DB-Net和Aerial Lifting；区分任务证据与接口借鉴，记录单入口局部查询方案及公平评价规则。仅文档，未启动训练或改生产实现。

2026-10-21：新增[v1冻结基座UAV适配器开发草案](dual_stream/UE_V1_FROZEN_ADAPTER_PROTOCOL_20261021.md)，核对EXP-0017完成/封存身份，明确v1开发边界、RoI局部feat_s0残差入口、冻结/梯度合同及10epoch建议筛查预算。未实现、未启动新任务。

2026-10-21：新增实验模块`rsprompter/uav/local_adapter_probe.py`和独立runner
`scripts/probes/ue_v1_adapter.py`；统一wrapper新增`adapter prepare|smoke|train|eval`路由。
通过mask_head前置hook取得真实RoI、conv_s0输出hook注入局部残差，原生产模型/冻结架构合同不改。
几何cache放数据盘，prepare只读train/val。smoke核定零初始化/无候选恒等、梯度与基座state不变。
协议与遥测解释见对应v1草案和DEBUG_FIELDS.md；原single/minimal/dual/fpn/sam路由不变。

`adapter`结果分析使用`scripts/probes/ue_v1_adapter_analysis.py`：从完整459图缓存重新计算
Mask AP和min(MaskIoU,BoundaryIoU)的Boundary AP，图像配对bootstrap复用COCO逐图匹配；
先校验相同GT、框/分数，以及恒等重采样AP复现。不读取test。

`ue_v1_adapter_finish.py`负责已授权筛查的训练后顺序执行：等待指定live训练PID退出，
校验last存在，正确/常量/错配验证及配对分析；正向信号触发同预算常量外观训练对照。
run_state仅作进度索引，活性以实际PID/进程和日志核实。

2026-10-21用户授权并行筛查后新增`local_adapter_variants.py`：early入口在two-way transformer前，
contextual源表示融合FPN0细节与FPN2语义；高分辨率fine原模块不改。独立smoke均通过恒等/冻结/梯度门。
`ue_v1_adapter_multi_finish.py`接管原训练PID，并行启动geometry/early/contextual臂，逐臂完成后评价，
统一使用原FP32基座缓存；新增真实UAV臂有正向信号时按同配置补常量外观训练归因。

`ue_v1_adapter_collect.py`在并行臂全部完成后核对2670连续更新、10轮、冻结状态、启动源码快照、
各模式459图覆盖及GT/框/分数逐值一致，交叉检查缓存AP与配对分析并计算checkpoint SHA。
汇总仅作为证据索引，最终结论仍需人工/主代理逐项核对原产物。

`ue_v1_adapter_interim.py`只在预先固定E1/E3检查点写完后进行完整开发验证，记录中途信号与
冻结检测不变量；不修改训练进程、超参、选权或10轮终点。

2026-10-22：`native_geometry_probe.py`提供512²表面高度/梯度/视角数及小型RoI局部残差适配器；
`ue_v1_native_geometry.py`复用基座严格重建、损失与训练预算，替换payload与适配器并独立记录源码身份。
统一wrapper新增nativegeo路由，原adapter及生产路由不变。该方向不编码RGB，以同support零几何内容作对照。

`ue_v1_native_finish.py`并行训练native_geometry/native_zero，随后在GPU1顺序完成FP32评价，
并计算各自对基座及geometry对zero内容的配对AP/Boundary AP分析。状态native_state.json独立于
原四臂run_state.json；最终collect必须同时核定两份状态与所有六臂产物，不能只凭四臂完成落判。

interim监督器支持`--family rgbd|native`，复用已完成记录并以`probe_gpu_lease.py`协调GPU2单个中途
评价任务；训练不停止、不改变终点。native中途进度独立保存在native_interim_state.json。

固定第1轮六臂结果记录于[中途观察](dual_stream/UE_V1_PARALLEL_PROBE_INTERIM_20261022.md)，
明确不是最终结论；全部原始缓存和后续终点仍保留在数据盘工件目录。

2026-10-22恢复：`ue_v1_probe_recover.py`以独立session管理六臂共同完整前缀续训及后处理。
通用probe train新增resume-from/reseed-from-epoch、原子checkpoint、RNG保存、parent合同链和逐轮
基座SHA检查；原架构/损失不改。原中断工件分目录保留，具体随机流修订与共同前缀见协议。

UnitLocalUAVAdapter在原局部查询路径增加无affine源/残差归一化，无新增可训练参数；通过source-mode
unit选择，保留原dtype。强化smoke记录soft logits恒等。八臂矩阵与尺度漂移背景见开发协议。

`ue_v1_unit_finish.py`以独立session管理unit/unit_geometry完整10轮、干预和配对内容比较；
unit_interim_state.json记录固定E1/E3。最终collect同时要求RGB-D恢复组、native组、unit组完成，
核对归一化范数界、恢复/逐轮随机种子、所有checkpoint来源与全部16项中途观察。

终点collect还实际加载各臂及触发的early/contextual常量训练对照的last.pth，核定E10/2670、
权重内合同与JSON快照一致、封版基座SHA及数据身份一致；对照同样检查完整逐轮预算、源码、
冻结state SHA和459图检测不变量，不能仅凭control目录存在或监督器done判断完成。
原生几何两臂的这组实际权重审计已通过；阶段结果文档已补E3、E10及内容对照区间。

2026-10-22用户将主要判断改为test：adapter/nativegeo eval新增`--eval-split val|test`，默认val，
非eval动作禁止test；不改训练损失/模型。`evaluation_identity.json`记录实际split、图数、
基座/分支权重SHA、FP32/scores/maxDets250。`ue_v1_test_finish.py`在独立test目录管理固定
十臂+基座共19次完整推理和15项配对比较；GPU1串行推理，最多3项CPU分析并行，不按test选权。
`ue_v1_contextual_control_finish.py`在GPU3无条件补齐contextual的同预算常量RGB训练对照，
状态contextual_control_state.json，不让旧val非正的触发规则遗漏新主要终点的归因控制。
新计划/修订记录在读取新增test结果前写入；旧val产物仍作为诊断保留，详情见开发协议末节。
`ue_v1_test_collect.py`要求19项test推理和15项配对比较实际产物、十个E10训练检查点、
459个test来源缓存审计、test与train/val图ID互斥、全部模式的检测/GT不变量及split/权重SHA一致；
生成complete_test_audit_summary.json。原val collect也增加test完成门与无条件contextual控制核定，
不再以val方向触发与否作为新主要终点的完成标准。结果前后线程/统计校准仍在8图重复COCO重建上误差0。

### 2026-10-22 S²DB-Net机制借鉴独立探针

- `rsprompter/uav/s2db_geometry.py`：扩展锚点元数据，逐值核定旧index/geom；源/目标camera-Z单位m。
- `rsprompter/uav/s2db_adapter.py`：固定或局部可学习读取、归一化或拼接残差、条件UAV辅助头。
- `scripts/probes/ue_s2db_probe.py`：独立prepare/init/gate/train/eval入口，固定v1/E10合同及完整RNG恢复。
- `scripts/probes/ue_s2db_gate.py`：CPU读取/边缘/续训回放与GPU原始logits/梯度/250RoI资源门。

10臂第一阶段及D固定的辅助六格合同见[预注册协议](dual_stream/UE_S2DB_MECHANISM_PROTOCOL_20261022.md)。
代码实现、工程验收、完整训练和指标结论分别记录；不将已实现模块描述为已取得收益。
此入口不改原高分辨率全图repeat路径，也不解冻decoder/UDPR。
- `scripts/probes/ue_s2db_cache_audit.py`：对全部train/val及独立test缓存复算原深度→世界→卫星投影、来源SHA和旧候选兼容性。
- `scripts/probes/ue_s2db_analysis.py`：固定记录的AP/边界AP、共同500次bootstrap向量、因子效应、匹配纠错分母及内容预测变化。
- `scripts/probes/ue_s2db_aux_audit.py`：独立进程离线辅助IoU及固定train样本主/辅梯度诊断。
- `scripts/probes/ue_s2db_schedule.py`：持久调度owner，尊重旧任务GPU预约；第一阶段核定后仅启动预定D辅助三臂。
- `scripts/probes/ue_s2db_collect.py`：实际权重/源码/初始化/冻结/样本顺序/全部推理审计及最终报告；不按test选结构。

命令示例：`/data/wangcheng/envs/cvt2/bin/python scripts/probes/ue_s2db_schedule.py --root /data/wangcheng/checkpoint/ue_s2db_probe_20261022`。
调度前准备train/val缓存并通过CPU/GPU门；默认新缓存`ue_v1_s2db_geometry_v2`以float64保存投影元数据。
新增加的采样统计和梯度分析只在离线产物中，解释见DEBUG_FIELDS.md；此轮是工程实现与运行中状态，收益待完整审计。

工程验收已通过，见[验收记录](dual_stream/UE_S2DB_ENGINEERING_20261022.md)。源码/初始化冻结后由持久owner运行；
新方案训练及AP结论须等待E10完整矩阵，不能用此验收替代。2026-10-22同步新增9个探针文件、固定合同及离线遥测定义。

### 2026-10-22 探针阶段结论索引

[阶段结论与遥测核定](dual_stream/UE_V1_PROBE_STAGE_CONCLUSIONS_20261022.md)整理原八臂完整test、匹配内容对照、末RoI分块遥测及下一步路线；early/contextual对照与独立S²DB矩阵未完，不作为最终验收。

独立可信度审查见[子代理报告](dual_stream/UE_V1_INDEPENDENT_EVIDENCE_REVIEW_20261022.md)：抽查原始比较、native完整更新行及源码合同，列出空间依赖缺口、固定矩阵盲点和有停止标准的下一步。

### Native固定权重空间诊断

`scripts/probes/ue_native_spatial_diagnostic.py`独立冻结权重推理、支持内字段置换、残差RMS匹配及配对分析；不修改旧runner或训练路径。见[固定合同](dual_stream/UE_NATIVE_SPATIAL_DIAGNOSTIC_20261022.md)。

### 2026-10-22 测试集弱点三路离线探索

- `scripts/analysis/ue_test_distribution_audit_20261022.py`：train/val/test密度、真实mask及bbox面积、触边、空间块及基座分层评价；独立校正GT面积分组并回放总体AP。
- `scripts/analysis/ue_test_error_modes_20261022.py`：固定框Hungarian失败类型、分割阈值纠错/新增及像素区域计数，单列COCO分数顺序匹配。
- `scripts/analysis/ue_test_support_audit_20261022.py`：已有459图候选/原生缓存的观测支持与错误关联，明确格、保留投影点邻域及可见性认证的差别。
- `scripts/analysis/ue_test_weakness_collect_20261022.py`：核对共同来源SHA、计数及面积回放，生成[测试弱点综合报告](dual_stream/UE_TEST_WEAKNESS_SYNTHESIS_20261022.md)。

独立CPU产物在`/data/wangcheng/checkpoint/ue_test_weakness_20261022/`；有限32图原深度重建及最终DT尾部/框内覆盖上界脚本按精确研究产物归档，不是训练入口。
本轮不改冻结探针模型、训练预算、选权、评价源码或缓存。下一步方向待当前S²DB完整E10矩阵与审计后，按综合报告的证据判定表再决定。

### 2026-10-22 读取/对齐/消费审查

新增`scripts/analysis/ue_read_consume_audit_20261022.py`，CPU坡度坐标合同、固定32train缓存的源token中心/实测像素原点核对与来源SHA。
[审查报告](dual_stream/UE_READ_ALIGN_CONSUMPTION_AUDIT_20261022.md)明确局部残差与全图接收特征、深度代理与特征偏移原点、native插值及RGB/几何捷径边界。
没有改动41份冻结源码、训练预算、缓存或评价规则；新检查不代表性能消融。

`scripts/probes/ue_native_spatial_collect.py`独立核对五组实际输出、源码/权重身份、冻结state、全部RoI调用覆盖及RMS匹配误差，生成诊断RESULTS.md及complete_audit_summary.json；这些是离线验收，不修改训练。

### 2026-10-22 下一步探针计划（待独立审核）

[计划草案](dual_stream/UE_NEXT_PROBE_PLAN_DRAFT_20261022.md)在旧17/19测试推理、11/15比较及当前S²DB矩阵未完的状态下，
限定下一步为native固定权重的字段空间依赖诊断及8图有限对应核查；不启动新训练、不修改运行合同。
用户要求先plan再独立审核，审核意见单列保存，最终采纳/修订在同一任务中同步；不将草案标成已执行。

下一步计划独立审核已给出[有条件通过意见](dual_stream/UE_NEXT_PROBE_PLAN_REVIEW_20261022.md)。
[修订计划](dual_stream/UE_NEXT_PROBE_PLAN_20261022.md)逐项采纳M1–M4：空间赋值敏感性的窄解释、native实际来源射线与缓存逐值对账、8图影像抽查的真值/代表性限制、Mask AP两项主检验及多重比较限制。
先CPU及来源影像核查、再空闲GPU的有限固定权重诊断；目前仅计划交付，无新增推理/训练。
修订版经同一独立代理定向复核通过，限定为有限机制诊断；原审核末节保存复核结果，无新增必须修改项。原始影像抽查不能替代外部标定认证，最终方法方向仍待完整矩阵与独立验证。

本轮空间诊断的耐久report owner在产物根finish_report.py中等待原计划全部分析完成，再执行独立collect、输出RESULTS及同步文档。当前状态与已完成点值见固定合同；不能将仍在计算的区间描述为已完成。

### 2026-10-22 Native真实来源核查（P2已实施）

新增`scripts/analysis/ue_native_source_provenance_20261022.py`追溯固定4train/4val的native实际最近表面射线、相邻高度差及近表面视角来源，历史cache逐值核对后选24个固定点并生成影像接触图。
新增`scripts/analysis/ue_native_source_root_verify_20261022.py`使用原始厘米制齐次矩阵独立复算选点与邻点，不调用子代理重建函数；8图缓存和24点来源核定通过。
[核查报告](dual_stream/UE_NATIVE_SOURCE_AUDIT_20261022.md)区分来源忠实、内部一致性与视觉无法判断；没有外部对应真值，不报整库正确率。
产物在`/data/wangcheng/checkpoint/ue_native_source_audit_20261022/`，含全部来源射线、历史SHA、24点、8图、视觉记录及主代理复核。
本轮仅CPU；未运行P1或新增训练，未改冻结源码与共享缓存。

### native置零现象定向排查

[报告](dual_stream/UE_NATIVE_ZERO_PARADOX_20261022.md)记录卫星query路径、几何首卷积项分解及固定32图纠错抵消。`scripts/probes/ue_native_path_diagnostic.py`只读8train+8val激活；`ue_native_error_diagnostic.py`只读32test既有预测，不启动训练或选结构。

`scripts/probes/ue_native_query_factorial.py`补齐新增适配器query/几何四格的两项固定权重推理，关闭query只影响新适配器，原SAT保持完整；见[诊断合同](dual_stream/UE_NATIVE_QUERY_FACTORIAL_20261022.md)。

### 2026-10-22 已审核六条件Native诊断实施

新增`scripts/probes/ue_native_assignment_probe.py`与`ue_native_assignment_analysis.py`，实施P2完成后的P1：correct/zero/mean/三组固定seed联合置换，原始logits回放门、全调用平方和、固定8图残差诊断、Mask AP两项主比较与最终审计。
与历史五条件含RMS匹配诊断完全独立，不改变其产物或本轮冻结模型。[实施与排队纪律](dual_stream/UE_NATIVE_ASSIGNMENT_IMPLEMENTATION_20261022.md)。
CPU门已过；GPU推理需等待实际空闲和旧owner预约结束，不能把排队当已完成。

[Native独立补充复审](dual_stream/UE_NATIVE_INDEPENDENT_FOLLOWUP_20261022.md)核定空间干预区间，收紧卫星路径、纠错抵消和高度RMS归因，并明确固定函数诊断的完成边界。

### Native归因诊断最终交付

[最终可信结论](dual_stream/UE_NATIVE_CREDIBLE_CONCLUSION_20261022.md)汇总空间干预、query四格及独立复审；`scripts/probes/ue_native_conclusion_audit.py`核对两轮实际预测、权重/源码身份、全部459图及500次配对比较，输出complete_conclusion_audit.json。固定模型问题已收口，不代表其他矩阵完成。

### 几何实现与消费调研

[调研与最小改进建议](dual_stream/UE_GEOMETRY_IMPLEMENTATION_AND_CONSUMPTION_RESEARCH_20261022.md)核查表面选择/插值风险、原始文献及差分残差假说。`scripts/probes/ue_native_sampling_audit.py`为固定8val的CPU只读审计，含NumPy/PyTorch公式回放与边界高度混合计数；未改模型或启动训练。

### 2026-10-22 用户授权GPU共租调度

新增`scripts/probes/ue_probe_coresident_owner.py`，按显存工作预算接管S²DB与六条件Native诊断，保留运行子进程及原冻结源码。
[共租修订](dual_stream/UE_PROBE_CORESIDENT_SCHEDULING_20261022.md)取代此前“等待独占空闲GPU”的资源纪律，科学依赖和训练/评价合同保持原样。实际启动与显存记录独立保存，不增加训练日志字段。

### 几何采样×消费8臂探针

[固定协议](dual_stream/UE_GEOMETRY_MECHANISM_PROTOCOL_20261022.md)规定bilinear/nearest×concat/difference×真实/非零常量训练控制。独立模型native_geometry_mechanism.py、CPU门及ue_geometry_mechanism_*脚本；不改旧训练入口。resource gate后按显存共卡，不按AP启动或筛选。

[几何机制工程验收](dual_stream/UE_GEOMETRY_MECHANISM_ENGINEERING_20261022.md)记录四个CUDA门、实际208实例训练及250RoI峰值和共卡准入。8臂调度与最终审计源码已冻结，当前运行中，未有本轮AP结论。

### 六方向归一化attention探针

[固定协议](dual_stream/UE_ATTENTION_LADDER_PROTOCOL_20261022.md)定义L0共同基准与L1–L6单机制干预/常量控制、GPU1–3首发限制。L1须依据实际patch lattice验证；L3须避免输出LN消除拒绝；L6仅指定decoder Q/V LoRA并单测资源。当前实现/验收中，未宣称训练或收益完成。

[Attention ladder工程记录](dual_stream/UE_ATTENTION_LADDER_ENGINEERING_20261022.md)列出L1–L3 CUDA门、实际patch坐标核定、峰值与GPU1–3固定slot。`attention_ladder.py`、`ue_attention_ladder_cpu_gate.py`、`ue_attention_ladder_probe.py`、schedule/collect均为独立入口；L4–L6未做GPU门，按队列后置。

[2026-10-23正向信号阶段状态](dual_stream/UE_PROBE_POSITIVE_SIGNAL_STATUS_20261023.md)记录当前E10 test点值/内容差与未完成项。几何组输出目录错误由产物根独立repair脚本恢复，不改已冻结数值source；运行状态不等于全部审计完成。

[已完成内容归因裁决](dual_stream/UE_COMPLETED_CONTENT_ATTRIBUTION_VERDICT_20261023.md)及`scripts/probes/ue_content_attribution_review.py`独立审计完成9CP（4对8臂及Czero）、固定500次向量配对。几何组原统计写盘错误已依据完整原日志恢复，保留原失败/来源/数值核对证明，不改冻结数值source。

`scripts/probes/ue_geometry_content_priority.py`只读已训权重/完整预测，优先输出固定四个几何内容对照的Mask AP配对区间；不等待其它离线图表、不占GPU、无新增训练字段。全指标与全队列最终验收仍保留。

[4090 N系列核查](dual_stream/UE_REMOTE_N_REVIEW_20261023.md)记录远端实际N1辅助边界监督、N2后激活注入、N3可学习温度及六臂训练状态；建议本机优先补后激活入口的成对内容验证。本轮只读核查，没有新增训练或遥测字段。

[梯度拓扑核查](dual_stream/UE_GRADIENT_TOPOLOGY_REVIEW_20261023.md)结合L1–3实际三步反向门排除持续全支断梯度，区分alpha起步延迟、消费者敏感性、辅助梯度冲突和固定候选不可导边界；建议离线分损失/分模块诊断及N2入口成对验证，未宣称其已执行。

[N2实施与G0结果](dual_stream/UE_POST_ACTIVATION_PROTOCOL_20261023.md)记录独立post_activation模型/runner/scheduler、已通过GPU门、两臂启动和L1 E10离线梯度证据。现有冻结source不变；梯度范数和局部衰减不代替AP内容对照。

[S1稀疏UDPR证据协议](dual_stream/UE_UDPR_POINT_PROTOCOL_20261023.md)：udpr_point_probe仅增强原UDPR选中点的32维输入，不改变原MLP/选点/损失。独立runner/schedule与CPU/GPU/评估门、零W初始化、全图mask格坐标、正确/常量训练及离线点改正/破坏统计均在文档中；不改原数值source。

**S1历史协议已作废**：全图输出格假设错误，旧任务已停止。当前有效为[S1-R与N2-shared](dual_stream/UE_CONSUMER_BLINDSPOT_PROBES_20261023.md)：udpr_point_roi_probe通过proposal映射roi_local选点；post_activation_shared_probe显式共享注入后tensor给UDPR。ue_consumer_blindspot_schedule统一在GPU0/1运行四臂，独立解析crop/共享tensor门通过，旧source保留。

[RS2-SAM2精读与J1设计](dual_stream/UE_RS2_SAM2_JOINT_ENCODING_PROBE_20261023.md)区分BEiT-3 union、Hiera内部BHFM、外部BC/提示通路及官方训练范围；定义Base+ block4的局部闭合双向交互、正确/常量内容对照和完整检测评估。实施状态见下文。

[J1实现与启动](dual_stream/UE_J1_IMPLEMENTATION_20261023.md)：joint_encoding_probe在Hiera block4执行局部闭合双向交互；独立runner保留融合后SAT自动求导、递归计入RPN列表损失，独立analysis允许检测框变化并增加bboxAP。CPU/CUDA/接线门通过，两臂已启动；bounded_worker设置实测通过的5500MiB allocator上限，资源策略记录于manifest。旧源不变。

[S1-R/N2-shared阶段结果](dual_stream/UE_CONSUMER_RESULTS_INTERIM_20261024.md)：两组完成，未确认整体Mask内容增益；N2-shared有Boundary内容差，不能扩展为正确配对特异性或共享机制优越性。J1仍在训练。旧N2离线梯度OOM通过产物根独立repair顺序恢复，不改冻结数值source。

[旧 N2 训练分布诊断](dual_stream/UE_TRAIN_DISTRIBUTION_AUDIT_20261024.md)：独立 `ue_train_distribution_audit.py` 复用冻结评估，完整 train2142 图、七次推理，对比基座、正确/常量训练与固定权重 donor；输出 bbox/Mask/Boundary AP，区分训练内拟合、容量和内容贡献。

[J1首轮结果](dual_stream/UE_J1_INTERIM_RESULTS_20261024.md)：训练及完整推理完成，正确Mask净变化+0.194pp、训练内容差+0.071pp，已得区间跨0；同权重正确优于常量/donor点值，归因分析继续排队。旧N2训练集诊断已完成，无Mask净收益。

`scripts/probes/ue_joint_encoding_fast_analysis.py`：CPU配对统计加速复核。保留原J1匹配、边界定义、500次采样和随机种子，只将COCO全区域precision envelope的Python循环替换为NumPy累计最大值；每个evaluator前三次重采样与原函数校准至1e-12，保存校准误差。独立产物 `ue_joint_encoding_20261023/priority_input_audit/`，不修改原source或原队列结果。

J1错配完整统计已完成：Mask/BBox/Boundary2%/Boundary1%正确−donor的500次配对区间全部跨0，24次原函数校准误差为0。完整数值和结论追加于上述J1结果文档；原队列常量/错配复核继续，不将小幅点值视作对应RGB收益已获确认。

[2026-10-25证据汇总与进展](dual_stream/UE_STAGE_SYNTHESIS_20261025.md)：J1已14/14且双实现统计完全相同；新实例内部交互有RGB内容及原路径增益，精确donor对应性仍有限；四臂共同基础联合训练进行中，同E23–27排名波动；L系列14条训练完成但统计目录缺失使队列停止。区分常量对照保留几何/容量、消费入口、局部可见性和训练方差。

[三代理RGB深审](dual_stream/UE_RGB_COMPRESSION_CONSUMPTION_REVIEW_20261025.md)：确认当前RGB候选先等权mean、融合后再2×2平均，查询无法恢复被抵消的视角/子格差异；SAT→U→SAT提供无纹理替代路径，但尚未证明其主导。当前joint梯度接通且source主辅范数接近，不能沿用旧冻结辅助压倒主任务解释。优先现权重逐阶段任务信息定位，保留弱Boundary配对信号与所有归因限制。

[RGB机制变体共租协议](dual_stream/UE_RGB_MECHANISM_VARIANTS_20261025.md)：独立view_preserve/spatial_preserve/no_sat_copy三方向，各正确/常量配对，6个共同基础初始化的10epoch前缀训练；固定E10 val对比原100epoch日程的E10参考，不用test挑结构。源文件全独立，资源门已通过，allocator7800MiB/free11000MiB准入，每卡最多新增1任务；自动补原路径/合并路径内容差中差及固定RGB干预。

[机制变体完整结果2026-10-26](dual_stream/UE_RGB_MECHANISM_RESULTS_20261026.md)：22/22完成。空间变体内容差+0.859pp但正确臂未超过原compact正确；view/no-copy的正确−donor区间为正；no-copy整体Mask上涨但常量训练仍更高。尚无单变体同时通过全部内容/对应性/超原基准比较。原四臂仍训练，最新共同E60 RGBD86.5553/常量86.1068，不能提前当最终test结果。

### 2026-10-23 真实RGB内容瓶颈定向诊断

[独立复核与定向诊断](dual_stream/UE_RGB_CONTENT_BOTTLENECK_20261023.md)更新几何四对已完成的训练内容正证据，并区分对SAT门槛与推理依赖。新增`scripts/analysis/ue_rgb_content_error_audit_20261023.py`分析A/D已有预测的内容净纠错；`scripts/probes/ue_rgb_content_direction_audit_20261023.py`在固定训练图、固定E10下测内容残差与真实loss方向及SAM→UDPR传递。均为独立离线诊断，非新训练方法；现有N2/N2-shared/S1-R/辅助配对继续，不重复启动近似架构。

### 2026-10-23 内容诊断收尾与跨划分补测

A/D全459内容净纠错及4train内容方向两项均已完成、独立验收。新增`scripts/probes/ue_rgb_content_generalization_audit_20261023.py`在固定val/test四图延续方向诊断，`ue_rgb_content_generalization_finish_20261023.py`串行共租调度并汇总train/val/test描述。独立plan/source SHA/原始logits恒等及资源门先于正式补测，无训练更新。分组结果及用途边界见[内容瓶颈记录](dual_stream/UE_RGB_CONTENT_BOTTLENECK_20261023.md)末节；不把不同划分当同图配对。

### 2026-10-23 检测阶段漏失定位

新增`scripts/probes/ue_detection_attrition_probe_20261023.py`：纯卫星固定检查点，独立回放全部RPN锚框/回归/top-k/尺寸/NMS/上限及RoI回归/分数/NMS/上限；逐值核定原RPN、RoI与历史完整基座检测输出。按GT任意覆盖和一对一最大匹配容量分开统计，回归恢复单列。见[固定合同](dual_stream/UE_DETECTION_ATTRITION_PROTOCOL_20261023.md)。门后共租执行459图，全部观察写独立产物，生产训练/检测阈值与旧探针source保持原样。

`scripts/analysis/ue_detection_attrition_verify_20261023.py`作为独立CPU完成验收器，核定完整459图来源、GT/mask面积/最终DT、阶段计数及FP32对pycocotools float64阈值差异；不修改诊断或原预测。最大238实例工程门通过，正式459图已实际推进。

### 2026-10-23 检测与内容阶段结果及离线资源修复

[阶段更新](dual_stream/UE_PROBE_STATUS_DETECTION_AND_CONTENT_20261023.md)记录459图检测漏失验收、跨划分内容方向、N2-shared边界正信号及S1-R内容配对未通过。`ue_probe_coresident_owner.py`按真实任务种类将offline gradient工作预算从误用的推理3500MiB修为20000MiB，保留训练和失败证据。新增`ue_post_analysis_recover_20261023.py`在GPU审计失败后独立恢复已就绪CPU AP比较；原数值source不改，GPU失败与CPU完成分别报告。

### 2026-10-23 UAV几何辅助RPN排序三臂

用户授权子代理设计、主代理实施后，新增`rsprompter/uav/rpn_geometry_probe.py`及`ue_rpn_geometry_probe_20261023.py`、`ue_rpn_geometry_schedule_20261023.py`、`ue_rpn_geometry_analysis_20261023.py`、`ue_rpn_rank_reachability_20261023.py`。三臂只训练native几何条件的RPN objectness残差，冻结原网络，直接用原anchor分类loss；实际A=6，真实anchor中心nearest读取。原两级NMS/回归/1000/250预算保持，单列SAT预筛2000诊断。工程门和双worker增强mini epoch续训已通过，三臂均实际更新。源码与架构/日志口径见[实施与验收](dual_stream/UE_RPN_UAV_IMPLEMENTATION_20261023.md)，独立方案与主代理收紧K对照的依据一并保留。

### 2026-10-24 UAV失败归因与候选质量桥接

新增只读`scripts/analysis/ue_candidate_evidence_audit_20261024.py`（459图已有候选预算输出与native支持关联）、`scripts/probes/ue_rpn_supervision_audit_20261024.py`（固定32train/16test原anchor分配/抽样与回归候选质量）、`scripts/probes/ue_candidate_quality_bridge_20261024.py`（独立64train/32val表格可解码性检查，extract/fit两阶段，无test访问）。未改原卫星、缓存、活跃探针source或训练遥测。新产物位于/data/wangcheng/checkpoint各同名独立根；[失败归因及候选质量方向](dual_stream/UE_FAILURE_CAUSES_AND_CANDIDATE_DIRECTION_20261024.md)明确SAT候选预算收益与真实UAV收益的区别、阶段计数分母及后续门。

桥接三臂全部完成：区域几何二分类AUC略高，但直接替换Top1000排序均丢失覆盖，不推进该实现。另增`scripts/probes/ue_geometry_proposal_pilot_20261024.py`（init/gate/train），独立几何五通道小CNN直接学习实例中心/offset/log-wh，真几何与常量共享支持/初始化，完整train10epoch后仅val评价候选覆盖；工程门通过，两个臂共租GPU0运行。没有加载或改动原SAT/SAM/UDPR，也没有最终实例AP结论。配置、学习率差异、资源与后续验收见同一失败归因报告第七节。

`scripts/analysis/ue_geometry_proposal_complement_20261024.py`仅在这两个固定PID实际产出val结果后做CPU互补性／匹配容量分析，SAT参考为已有pre2000/post1000输出；图像配对bootstrap500。没有新增GPU排队或test推理，不把潜在可补框数解释为AP。

proposal pilot两臂E10及全部val已完成：真几何top250一对一IoU.5容量12867，常量9257；相对SAT2000最终未覆盖可补142对97，尚非AP。新增`scripts/probes/ue_geometry_proposal_consume_20261024.py`执行真/常量×原框/原RoI回归框四臂冻结消费，仅val；逐值回放原检测并复用原mask，按原阈值拒绝重叠新增框、自身NMS、填final250剩余名额，仅新增框走原SAM。独立2图门后完整val，原SAT参数和预测保留。来源、资源和边界见失败归因报告第八/九节。

四臂完整val均无Mask净增益。新增只读`scripts/analysis/ue_geometry_box_consumer_audit_20261024.py`定位潜在可补GT142→novelty后119→原分类链实际8的漏失；oracle与直接对应框替换上界分开记录，不以oracle作收益。新增`scripts/probes/ue_geometry_score_band_20261024.py`进行真几何/常量/SAT自有低置信候选三臂，仅val、不新训；新增假设以原阈值乘分支置信度进入低分数带，绕过原分类否决/框回归、保留全部原预测，检查UAV相对同预算控制的实际净增益。AP非下降来自可信前缀设计，不能替代内容对照。见同一报告第十节。

score-band完整val：真几何Mask85.8162、常量85.6829、原SAT2000为85.5377，匹配内容差及净差区间为正；但更强SAT自身补充86.2991。`scripts/probes/ue_geometry_score_band_analysis_owner_20261024.py`执行固定五项配对分析。`scripts/analysis/ue_geometry_conditional_complement_20261024.py`对已有缓存做强SAT前缀条件增益（只读CPU），真几何86.3237、常量86.2991，微小增益区间包含零，不认定稳定。

`scripts/probes/ue_geometry_conditional_online_20261024.py`在线复现缓存组合后全test，保留原SAT2000+低置信补充前缀、独立E10几何头生成候选，固定分数层级和250；`scripts/probes/ue_geometry_branch_test_20261024.py`核验相对原SAT2000的直接分支。两者每内容臂先2val精确box/score/RLE门再459test，无重训或test选参。`scripts/probes/ue_geometry_confirmation_analysis_owner_20261024.py`等待真实结果identity后执行固定五项test配对比较并核对G/K的纯SAT prefix逐字节一致。原49源未改，新source快照/协议/PID在各独立运行根；详见失败归因报告第十一/十二节。

两seed主比较均已完成：seed44真几何Mask66.5516，对强SAT66.0879净+.4637pp；seed45真66.4930，净+.4051pp，配对net及内容区间均为正。`scripts/probes/ue_geometry_proposal_repeat_20261024.py`复用原Net/loss只改独立seed；`ue_geometry_conditional_configured_20261024.py`参数化checkpoint/out-root并通过原缓存精确回归；`ue_geometry_repeat_owner_20261024.py`连接E10训练审计、test与两项四指标分析。[架构说明](dual_stream/UE_GEOMETRY_INSTANCE_SUPPLEMENT_ARCHITECTURE_20261024.md)明确输入、直接实例监督、候选级消费、强SAT前缀与排序分数含义，当前不包含UAV RGB/CVT。

新增只读`scripts/analysis/ue_probe_input_identity_audit_20261024.py`完成36734归档文件与3060缓存来源核对；`ue_geometry_new_instance_audit_20261024.py`核对旧COCO匹配不变与新增GT匹配（真811/常量476）；`ue_geometry_count_matched_20261024.py`按每图同新增数50838总额诊断，Mask点值差不变。这些不是生产loss或选模字段。

`scripts/probes/ue_geometry_conditional_intervention_20261024.py`做固定真模型的constant/shuffle字段输入；`ue_geometry_whole_map_donor_20261024.py`做四字段+support半循环229错配，均先实际val门再全test。`ue_geometry_intervention_analysis_20261024.py`检查同权重/同SAT前缀并共享正确evaluator做500次Mask配对，只作推理内容归因。原source/缓存/SAT/UDPR保持不变，详见失败归因报告第十三至十五节。

最终总验收已通过：新增`scripts/analysis/ue_geometry_direction_final_audit_20261024.py`可重跑核对权重/初始化/实际优化步数/配对epoch顺序、两seed完整指标、三项输入控制、同数量控制与全输入来源，输出`ue_geometry_direction_final_audit_20261024/audit.json`。最终推荐路线、数值及限制集中于[阶段结论](dual_stream/UE_UAV_EFFECTIVE_DIRECTION_VERDICT_20261024.md)；架构文档同步为完成状态。该路线是实验脚本与冻结模型组合，主训练/推理入口尚未改成正式共同初始化联合训练模型，UAV RGB/CVT仍未获本实验正证据。

### 2026-10-24 实例监督驱动的编码器内部回写

[固定协议](dual_stream/UE_INSTANCE_INTERACTION_PROTOCOL_20261024.md)规定几何内容×head/internal×两seed八臂，以及RGB六臂，共14臂；冻结前独立审查补两条常量几何字段+正确RGB，未读取任何本轮AP，不依test选结构。不是RS2-SAM2复现。head入口也有双向交互，区别仅为S′是否继续进入后续Hiera，不称无attention对照。

- `rsprompter/uav/instance_interaction_probe.py`：native几何CNN高分辨率旁路、block4局部闭合交互、直接读取S′的中心/框头、可选五视角RGB LoRA及连续锚点读取。
- `scripts/probes/ue_instance_interaction_20261024.py`：独立init/gate/train/eval；冻结E73原参数/buffer并保留回写后autograd，原loss加直接实例loss；正式推理和验收共用`inference_outputs`。
- `scripts/probes/ue_instance_interaction_gate_20261024.py`：名义格点ramp、同权重入口开关、正式推理无支持回退、实际AdamW/AMP断点重放与两worker增强loader顺序。CUDA原算子反向存在噪声，采用独立三次内存重放标定，不宣称权重逐位续训一致。
- `scripts/probes/ue_instance_interaction_owner_20261024.py`：14臂/119个预定训练、评价和统计任务；实测显存峰值加2048MiB余量共租，flock防重复owner，失败保留并审查，不终止其它任务。
- `scripts/probes/ue_instance_interaction_baseline_20261024.py`：正式disabled推理全459图与历史强SAT逐框/分数/RLE/GT一致性门，未过门禁止引用历史基线的统计；不改变训练。
- `scripts/analysis/ue_instance_interaction_audit_20261024.py`：实际CP、有效optimizer步骤、初始化、配对顺序、来源、全部评价和四指标统计的严格完成审计，缺项即失败。
- `scripts/probes/ue_instance_interaction_finish_20261024.py`：绑定实际owner PID与进程起始ticks，等待119项产物；调用冻结最终审计，另核6120缓存内容SHA、评价权重SHA和完整合同一致，生成`study_summary.json`及`RESULTS.md`。不改变原调度、训练或选模；执行完成仍需前台审阅再确认目标完成。

四组工程门已通过：最大208GT训练、250实例FP32推理；直接中心梯度到卫星回写及两向attention，RGB门另核LoRA。产物独立根`/data/wangcheng/checkpoint/ue_instance_interaction_20261024`。这只证明工程接通，尚无新AP结论。`satellite_records`在internal条件下已被交互改变，纯卫星必须来自disabled或核定的历史强基座；不沿用旧前缀对纯卫星AP非下降保证。原正式训练入口和解码器/UDPR没有改动。

[实施与工程证据](dual_stream/UE_INSTANCE_INTERACTION_IMPLEMENTATION_20261024.md)记录独立复核采纳、四格实测峰值、数值续训重放边界、全459图历史强SAT精确回放和首批8进程启动。启动不等于E10或指标完成；最终审计缺任一CP/推理/统计即失败。

同日补齐自动收尾：真实owner与8个子进程持续存活，独立finish PID84334已启动。报告字段接线用隔离临时fixture核定14臂/60比较/四组证据项，不把fixture当真实指标或审计完成。新增报告只写独立产物根，不改变23份冻结源码/协议的身份。

内部回写真/常量两臂E1实际检查点已核验：267步、2136图同序、共同初始化、五类RNG、有限新增权重与原SAT SHA均通过，独立产物`epoch1_internal_pair_audit.json`。训练loss差仅用于观察，不作为E10内容增益或选权证据；完整数值追加于实施记录。

同轮几何四格E1均已落盘并完成`epoch1_geometry_factorial_audit.json`实际CP核对：四臂初始化/顺序一致，各267次AdamW更新与冻结记录通过；继续原固定预算，未增加中途test评价或按loss筛选。

`scripts/analysis/ue_instance_interaction_rgb_path_audit_20261024.py`为CPU只读有效参数范围审计，按实际optimizer参数槽映射命名参数、核对原source非LoRA状态与共同初始化、原SAT SHA及配对顺序，缺失CP单列不当通过。E1已有3/6个RGB臂通过：patch_embed.proj与blocks0/1共25648个source LoRA参数实际更新，非整个声明的1061424；不改变训练，仅完善RGB消费范围披露。产物`epoch1_rgb_effective_path_audit.json`。

[几何四格阶段结果](dual_stream/UE_INSTANCE_INTERACTION_INTERIM_20261024.md)记录实际四个E10/2670 CP、10轮样本顺序及已完成459图推理审计。真实internal−head合并Mask+.103pp有正区间，但常量入口差点值相近，不能归为几何特异收益。固定权重原路径正确输入点值反低于常量/donor，新增`scripts/analysis/ue_instance_interaction_prefix_followup_20261024.py`仅对已有预测做两项离线配对统计；原14臂/119项和冻结源不变，RGB/第二seed/其余分析继续。

### 2026-10-24 内部回写调度的显存准入修订

新增`scripts/probes/ue_instance_interaction_capacity_owner_20261024.py`，只替换原控制面，复用原`build_jobs`/119项命令与完成判据。取消每卡固定2个本轮进程的人工上限，工作集取该家族/入口的工程门及已完成epoch峰值最大值，加512MiB CUDA上下文预算；每卡保留一次2048MiB余量，并扣除已启动但未实际分配完的预留。CPU统计并发仍最多4个（独立探索性followup单列）。

原owner81973被替换，12个训练/统计子进程按PID/起始ticks接管，未中断；新owner161872。两个等待中的seed45 head臂在GPU2/3启动并均实际更新25步；14个训练臂至此全部启动，5个已完成。切换记录`capacity_handoff.json`，原控制面/finish身份和finish源码归档`control_history/`，23份冻结数值源码/协议未变。finish v2接受新控制面身份，当前PID161891；最终附加审计核对新控制面源码SHA与119项身份。资源修订不改实验初始化、样本顺序、训练预算或评价。

seed44的internal训练内容比较已完整校准：合并Mask+.152974pp CI[-.021586,+.190515]、原路径−.004567pp CI[-.147670,+.102384]，四指标均跨零；head内容正区间与该结果分开解释。原路径固定权重正确−常量字段四指标统计也完成，Mask−.166407pp CI[-.237367,+.055994]仍跨零，不以负点值宣称稳定伤害；donor、RGB与第二seed继续。阶段文档同步实际完整区间，原科学合同不改。

首个RGB-D head正确输入全459图test已完成：Mask66.477342；独立`rgb_head_first_test_partial_audit.json`核定E10权重SHA、FP32/max250、GT/ID及原SAT前缀逐值一致。同结构常量RGB控制尚未完成，此整模型点值不能归为RGB增量，阶段文档保留该边界。

内部回写真实字段相对强SAT的净Mask统计完整校准通过：+.484852pp，CI[+.308804,+.609847]，与训练内容比较跨零并存；不能混称无净增益或已证实字段数值贡献。常量控制仍含真实深度support，不是完全无UAV/纯容量控制；阶段文档明确此归因范围。

### 2026-10-24 回写value来源离线分解

新增`scripts/analysis/ue_instance_interaction_value_audit_20261024.py`，对固定seed44真实几何internal E10进行FP32只读审计。保留原观察到的attention权重，将线性value及输出映射的注入拆成交互前UAV表征、读取卫星后的token增量、两层线性偏置三项；在block4结束即停止，不运行检测或修改训练。逐图核对三项相加回放原始注入；全局RMS和夹角按平方和/点积合并，不能解释为可相加贡献百分比或AP因果效应。UAV项仍包含几何CNN对support的消费，不叫纯RGB信息。

独立产物`ue_instance_interaction_20261024/value_source_smoke`已完成2图×三种输入回放，最大误差8.94e-8；`value_source_val`为完整val的正确/常量字段/几何donor审计，完成须检查`complete.json`和实际459图。原14臂/119任务、冻结源和检查点不变。结构讨论优先保留现有实验完整性，未启动value分离新训练。

随后完整val459图×三输入审计完成并前台核对逐图记录及回放误差，最大1.0431e-7。正确输入U项/卫星返回项/偏置/实际总注入的合并RMS为.037042/.001342/.002989/.038531，未显示卫星value返回项幅度占主导。该观察下调对应结构猜测的优先级，不代替AP因果对照；既定RGB/第二seed继续，下一步优先监督梯度方向与有效读取检查。原路径正确−donor完整统计也已完成：Mask区间跨零，Boundary1%为负区间；详见阶段文档，不概括为所有指标受损。

2026-10-24新增RGB internal匹配训练结果：正确RGB相对常量RGB的合并Mask+.193716pp CI[+.061330,+.343377]，原路径+.142806pp CI[+.012801,+.265799]。已核对实际E10评价SHA、10轮各2136图共同顺序与初始化、冻结源及四指标各500次/24次校准；E10有效LoRA审计现覆盖4/6 RGB臂。该单seed正区间支持本检查点的RGB消费，但不等于训练稳定性、深度必要性或双向必要性。8/14训练和59/119项为本次快照，所有剩余任务继续；RGB固定输入诊断、head匹配与深度字段统计待齐。完整数值和限制见阶段文档。

2026-10-24新增[几何消费与正式化成本复核](dual_stream/UE_GEOMETRY_CONSUMPTION_REVIEW_20261024.md)：四字段整体收益不能单独归因高度/高度差，当前字段表示为相对中心Z、像素差分和近表面视角数。记录单表面栅格、差分有效性、U128→U64与观测支持的语义边界；实际几何CNN202416参数、完整无RGB交互头259893参数。五视角共享RGB编码器当前完整前向但仅P0被消费，后续优化必须验证同权重等价。联合预算准备不等于固化完整探针架构；正式保留需要内容对照和端到端增量成本审查，不因理论价值默认保留显式几何大分支。

2026-10-24按用户全模态且不冗余要求新增[联合模态角色与验收合同](dual_stream/UE_JOINT_MODALITY_CONTRACT_20261024.md)。`rsprompter/uav/rgb_p0_encoder.py`为独立可选P0源编码器（patch、位置、blocks0/1、lateral），未接入或修改冻结探针。`scripts/analysis/ue_rgb_p0_efficiency_gate_20261024.py`通过实际E10首val五视角FP32/AMP P0及候选读取逐值一致、有效LoRA梯度三次重放噪声界；70.17M→.404M源参数，源编码单样本中位104→15ms，不外推整模型速度/AP。训练丢弃后层dropout导致全运行RNG不同，不能声称旧轨迹续训等价。`scripts/analysis/ue_camera_variation_audit_20261024.py`核查3060相机：K固定、各视角相对rig固定，故跨图错配标定不是有效消融。真正无深度须重建候选/支持/属性，不能继续使用真实深度缓存。

2026-10-24后续工程推进：`scripts/analysis/ue_rgb_p0_full_val_gate_20261024.py`完成全459val×5视角P0及最终融合U128/support逐值一致门，原adapter/CP/冻结源不变；不是全训练轨迹或最终AP证书。`rsprompter/uav/geometry_inputs.py`是无I/O的实测/参考平面几何转换；`scripts/analysis/ue_geometry_input_gate_20261024.py`在预定3val图对旧实测缓存逐值复现，参考平面无文件读取及全无效回退通过。`scripts/probes/ue_reference_plane_cache_20261024.py`固定sampling-origin水平面为无实测深度先验，重建index/geom/anchor/native fields/support，独立全3060缓存已启动，运行时禁止EXR和实测缓存读取；尚无相应匹配训练/AP，不拿参考平面当真实地面。

2026-10-24新增[紧凑联合模型工程说明](dual_stream/UE_COMPACT_JOINT_IMPLEMENTATION_20261024.md)：`rsprompter/uav/compact_joint.py`保留原单流可训练范围并加入紧凑UAV，原任务和直接实例loss均可反传，不加载任务权重；设备迁移显式执行MMEngine子级接口。`data/ue_joint_inputs.py`将实测/参考平面及正确/常量RGB设为显式输入路径，来源逐图核对、参考缺失不回退。`scripts/analysis/ue_compact_joint_foundation_gate_20261024.py`构造期限制实际torch.load只读通用SAM2，最大208GT train图3次AMP门通过、冻结参数保持、峰值reserved5648MiB；最终forward源版本在v2门复核。它不是已启动的正式100epoch联合运行，推理/续训/匹配训练协议仍待完成，原冻结探针不改。


### 2026-10-24 紧凑联合运行器与CPU统计准入

`scripts/probes/ue_compact_joint_run_20261024.py` 与 `configs/ablations/ue_compact_joint_protocol.json` 固定四臂共同基础初始化、100ep/26700更新、单GPU batch1累积8、每轮val选best及完训一次test，尚未启动长训练。`data/ue_instance_targets.py` 从卫星框生成目标，避免无深度对照读旧实测x；`scripts/analysis/ue_compact_joint_runtime_gate_20261024.py` 已过四臂首val FP32、强制250掩码、reference-plane无实测读取的8图累积与五RNG/增强顺序恢复；`scripts/analysis/ue_compact_joint_optimizer_resume_gate_20261024.py` 补非空AdamW/AMP磁盘恢复。`scripts/analysis/ue_compact_joint_input_audit_20261024.py` 重核3060参考缓存/camera SHA及三图目标等价；`scripts/analysis/ue_compact_joint_engineering_acceptance_20261024.py` 已绑定门和当前源码。产物根 ue_compact_joint_20261024，详细范围见紧凑联合实施文档。

旧固定研究仅剩统计时，新增 `scripts/probes/ue_instance_interaction_analysis_owner_20261024.py`、`scripts/probes/ue_instance_interaction_analysis_finish_20261024.py`，并发4→12；每任务4GiB预算、主机8GiB余量，接管子PID不终止任务，原119命令/科学源不变。当前owner243139/finish243143，161872/161891已退役；第一次GPU handoff保持，新CPU接管另存analysis_capacity_handoff.json，bridge额外核新源后调用原finish全审计。101/119完成为本次快照，14训练及45推理已完成，剩余统计继续。

E10 RGB有效参数审计已覆盖6/6臂，均2670次实际更新、原source非LoRA保持；后层LoRA仍不在损失路径。该覆盖审计不代替完整研究最终验收。

2026-10-24新增[固定梯度与整链路成本审查](dual_stream/UE_COMPACT_GRADIENT_COST_REVIEW_20261024.md)。`scripts/analysis/ue_interaction_loss_direction_20261024.py`在8固定train图×E1/E3/E10做FP32 eval-mode原任务/中心梯度与首图分项，24行完成、无参数更新；早期中心对UAV表示梯度较强，E10 writeback平均余弦约+.0067，不认定单一冲突根因。`scripts/analysis/ue_compact_cost_audit_20261024.py`对3固定val图×3重复比较SAT/完整源/紧凑源，后两者预测逐值一致；wall中位364/826/728ms，源准备107→20ms，但新增掩码数量与I/O仍有成本。它是旧冻结权重评估，非最终联合模型部署benchmark，不因此调loss或更改原119项。


### 2026-10-24 四臂联合训练实际启动及收尾口径

最新运行状态见[紧凑联合实施](dual_stream/UE_COMPACT_JOINT_IMPLEMENTATION_20261024.md)顶部，取代此前“尚未启动”和旧研究101/119等历史快照。旧14臂119任务已全部完成，最终身份/缓存/固定源码审计通过，RGB内容有正向单seed证据；四字段内部增量区间跨零，不宣称高度或深度已独立成立。

新增运行控制面`scripts/probes/ue_compact_joint_owner_20261024.py`已实际启动四个独立100epoch训练，共同通用SAM2基础初始化；补齐`scripts/analysis/ue_compact_joint_amp_retry_gate_20261024.py`、`scripts/analysis/ue_compact_joint_launch_review_20261024.py`。AMP溢出对同8图恢复buffer与五RNG重算，失败尝试不计实际更新，最多4次重试；AR@250仅改原错误标签，未改数值。当前数值源及协议冻结，新分析入口不改变运行任务。

新增`scripts/analysis/ue_compact_joint_paired_20261024.py`：冻结8项val-best完训配对比较，原路径/合并路径分开，500次图像bootstrap及24次原估计器校准，另报AP75/标准尺寸组/AR250。全100轮顺序及val-best选权、两臂共同合同、CP与预测SHA不一致或任务不完整时拒绝分析；不读test调整任何结构。具体范围、路径、命令与不能证明的事项见实施文档。新增工具为只读收尾分析，最终联合收益与各输入必要性仍待实验。


2026-10-24补充只读收尾入口：`scripts/analysis/ue_compact_joint_rgb_intervention_20261024.py`在四臂训练/test全完成后，用固定RGB-D val-best执行常量/循环donor RGB干预，原几何及卫星不变；`scripts/analysis/ue_compact_joint_finish_owner_20261024.py`绑定实际训练owner句柄，等待8项训练/test完成后按资源准入运行8项配对分析＋2项干预，归档独立源码，不修改冻结训练源。全部scope与剩余科学审计见紧凑联合实施文档。


2026-10-24新增`scripts/analysis/ue_geometry_encoding_diagnostic_20261024.py`：完整2142train原始缓存的只读字段RMS、差分缺测零值/截断和64²原始覆盖代理统计，逐缓存SHA绑定、不读取标签及val/test。四字段RMS .37–.56，差分约.2%有效对截断；零差分中的未知占dx15.8%/dy25.6%，有支持起点未知约3.8%/4.3%。见[几何消费复核](dual_stream/UE_GEOMETRY_CONSUMPTION_REVIEW_20261024.md)新增实测节；不是AP损失成因证明，不改在训合同。


### 2026-10-25 联合检查点审计

新增`scripts/analysis/ue_compact_joint_checkpoint_audit_20261025.py`：CPU复建可训练参数顺序并对已保存实际CP逐项核对冻结参数/持久buffer、AdamW moment及逐参数step、五RNG、共同初始化与来源。可按anchor epoch或四臂完训后的best/last审计，非持久buffer公开为CP不可核验。首个卫星E1通过：323冻结张量不变、334参数实际267更新；原SAM未参与目标的IoU/obj-score头12参数无optimizer状态，未将其虚报为已训练。详细边界及首次分析程序非持久buffer键修订见紧凑联合实施文档。


2026-10-25四臂E1检查点审计完成：同轮样本顺序完全一致，全部冻结张量不变，UAV新增可训练张量均有267步AdamW状态。E1全val Mask为SAT67.6711、RGB-D66.4703、常量RGB68.4940、参考平面65.4744，尚无本轮test/稳定内容收益结论。新增`scripts/analysis/ue_compact_joint_epoch1_paths_20261025.py`只读快照并核对四臂E1原路径/合并候选的Mask与bbox，score从合并前缀原样取用，预测身份/点值校准后才输出完整诊断。见紧凑联合实施末节，不据此改变冻结实验合同。


2026-10-25 E1路径分解完成并归档：RGB-D原路径/合并Mask66.4292/66.4703，平均追加115.009候选只增加.041099pp；首轮bbox提高主要在原路径，Mask落差也已在原路径。三UV臂几乎填满250预算，候选成本收益需与内部交互分开核算。此为早期val、不同模型训练结果，未证明具体对齐或梯度冲突原因，不据此更改在训合同或宣称最终成功。


2026-10-25新增`scripts/analysis/ue_compact_joint_loss_direction_20261025.py`及`scripts/analysis/ue_compact_joint_gradient_owner_20261025.py`。前者在固定train 8图上用同一FP32/eval前向分离det/seg/aux三个梯度，按组计算四种配对；后者绑定真实RGB-D训练句柄并等E3/E10落盘，按显存准入复查同一驱动。E1已完成且CP/模型不变、无optimizer更新：det/seg的writeback/源LoRA/卫星LoRA平均余弦+.240/+.400/+.281，未支持普遍强冲突；seg/aux回写平均−.026、7/8负但幅度接近正交。完整范数、分母、空梯度及训练外推限制见紧凑联合实施末节和DEBUG_FIELDS；不据此自动调权。


### 2026-10-25 收尾控制面v2与新增只读入口

`scripts/analysis/ue_compact_joint_fixed_rgb_paired_20261025.py`补4个同CP正确−常量/donor、原路径/合并路径的完整配对区间；`scripts/analysis/ue_compact_joint_final_inputs_20261025.py`补最终实际archive与9180缓存的字节SHA；`scripts/analysis/ue_compact_joint_cost_20261025.py`补真实val-best纯SAT/完整RGB-D/只跳过追加候选解码的独立进程成本测量。`scripts/analysis/ue_compact_joint_finish_owner_v2_20261025.py`保留原10命令，扩为19项并检查9组成本预测前缀等价。新owner291123接替只有等待、无子进程的274663，四训练PID未动，28数值源逐项未变；旧控制面身份保留。当前权威posttraining状态为state_v2/owner_v2_launch，不再看旧PID判断完成。

仅最终成本benchmark要求同张空闲GPU顺序测量并等自身统计结束，训练和功能诊断继续按显存共租。完整合同/限制及尚未执行的科学核查见紧凑联合实施末节。E2早期全val的正确RGB-D Mask70.0818低于SAT73.0874、常量RGB75.2438与参考平面72.6691；不掩盖回撤，也不据此改变在训合同。


### 2026-10-25 独立原始影像对应核查

新增`scripts/analysis/ue_rgb_depth_image_alignment_20261025.py`：固定8train×5view的RGB/深度边缘内容代理、固定同点平移控制与40组原图。新增`scripts/analysis/ue_rgb_sat_image_correspondence_20261025.py`：不读模型/cache/标签，厘米坐标直接重投影UAV原RGB到实际卫星图；真实深度/参考面在相同像素交集做ZNCC。40view均真实深度优于参考面，均值.862/.298；10组源视图、3组目标棋盘图已实际查看，作用边界与原始记录见[原始影像对应报告](dual_stream/UE_RAW_RGBD_CORRESPONDENCE_20261025.md)。只下调严重全局错位猜测，不认证全部物理真值/互补信息/AP增益；没有改在训源。numpy汇总序列化失败原目录保留，整数cast重跑后40逐view记录逐值相同。


2026-10-25新增`scripts/analysis/ue_retained_rgb_read_audit_20261025.py`：同8train、668301锚点独立复算源view/源像素/EXR-Z/SAT投影，原色聚合的固定格对照。缓存对应一致，后方>1m约6.13%；硬过滤原色ZNCC .914低于当前平均.922（8/8），没有实施任何过滤/重排序/新训练。见原始对应报告C节。E3首次同轮val正向：RGB-D78.0419高于SAT75.2622、常量RGB77.2910和参考平面75.9853；同8图E3梯度复查完成，没有持续强冲突证据。仍保持100轮计划，不能当最终test/稳定净增益。


2026-10-25同步[联合模态合同](dual_stream/UE_JOINT_MODALITY_CONTRACT_20261024.md)中的过时“尚未启动/工程门待完成”状态，新增用户目标逐项证据矩阵：RGB/深度/标定实际角色、通用共同初始化、非空挂与成本、全预算净收益分开验收。当前仍在训练，表中未将E3点值/原图相关性/工程通过当作正式目标完成，固定rig识别限制保留。


2026-10-25记录E4完整同轮val：RGB-D78.1413、SAT77.8317、常量RGB78.2975、参考平面77.3177（Mask AP×100），RGB增量从E3正点转为−.1562pp；更新模态验收表，未将单轮正向提前判为稳定成功，训练数值源/预算未变。


2026-10-25记录E5完整同轮val：RGB-D79.3335高于SAT77.9739、常量RGB78.8230及参考平面77.7781（Mask AP×100），三点差均正；保留E4波动记录，不将其升级为最终/稳定收益，训练合同不变。


2026-10-25记录E6完整val三项Mask匹配点差均负，并把当前所有共同完成E1–E6逐轮指标与实际epoch文件SHA汇总至运行根`matched_validation_observation.json`；不挑阳性、不跨轮均值代替预定选权，训练/测试合同未变。详见紧凑联合实施E6节。


### 2026-10-25 11:20 联合中期核定

E1/E3/E10固定函数梯度诊断24项全部完成，driver/实际E10 CP SHA复核通过，owner287046正常退出；不能据其PID消失或末次state waiting快照误判失败。E10四CP冻结/optimizer/RNG审计通过（334/427状态均step2670、冻结张量保持），E1–E20同轮2136训练图顺序跨臂一致。全轨迹已更新，E20 Mask RGB-D83.5102高于SAT82.4716/常量RGB83.1073/参考平面82.1281；E10/E15内容负点保留。实际训练约E21–E31，无失败，100轮/test/19项收尾仍待完成。详见联合实施和模态合同的中期节。

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

### 2026-10-27 效率版三臂重启已核定

效率源码246e4aa，最终四卡attempt4及fast恢复门通过，旧两UBR/owner已停且完整归档。新root`ue_g_ubr_fast_20261027` owner1485153实际并行三组各4rank，三臂97数值源均在owner100来源闭包且同Gitblob；初始化/10轮2670/global8/train1infer4均一致。最新状态/性能界限见[效率记录](dual_stream/UE_G_UBR_EFFICIENCY_20261027.md)，尚无AP收益结论。

### 2026-10-27 Native RoI头与主任务监督接线

新增`rsprompter/uav/native_roi_refiner.py`（70ch native残差与显式RoI证据采样）、`ubr_native_integration.py`（prefusion缓存/局部decoder shim/空rank连参）、`scripts/probes/ue_ubr_native_runtime_20261027.py`（A/B/C共同初始、不构旧UBR）和`ue_ubr_native_gate_20261027.py`（真实四卡工程门）。新测试`tests/test_native_roi_refiner.py`及`test_ubr_native_integration.py`共15CPU通过。开发合同`configs/ablations/ue_native_roi_development_contract_20261027.json`与[实施说明](dual_stream/UE_UBR_NATIVE_ROI_IMPLEMENTATION_20261027.md)记录native128修正、原512主监督、独立RGB/geometry支持和无额外loss。当前实验保持停止，本轮不发射长训练。原全图/fast源97份均不改。

新轻量头完整FP32 island，原模型继续AMP，delta按接收logits dtype相加；解决低scale下新隐藏层梯度下溢，不改loss分母/初始化。正式训练恢复/调度仍需另行核定，工程门不等于长训练或AP证明。

最终真实四卡attempt5通过（3次AMP工程更新、空rank、主mask独立反传、250框8decode、zero恒等），81份import源零漂移；训练峰值8558MiB，含所有诊断8564MiB。250框maskconsumer .635–.688秒不含extract_feat，A/C单批顺序计时不作加速结论。15CPU通过，无AP或正式续训轨迹证明。

### 2026-10-27 Native三组训练接线与并行调度

新增native_run、native_owner、native_resource_gate和native_protocol；明确复用旧train/save/restore/五类RNG与原inference_outputs评价，旧源不改。A无头/B尾证据全零/C正确，每组四卡10轮2670/global8，E1/E10完整val、全部训练完固定E10全test。每臂资源cap仅外合同，源码commit与全repo import闭包绑定；详情见[三组说明](dual_stream/UE_UBR_NATIVE_ROI_SHORT_STUDY_20261027.md)。

### 2026-10-27 Native三组已并行发射

数值提交864b441、owner1518567；`ue_g_ubr_native_short_20261027`三组各四rank并行。A/B/C资源门与C四卡恢复门通过；owner绑定101源/runner98源、同数值合同与初始化，资源每臂5988/8596/8890MiB预留，512MiB/card共同余量。发射审计见root_launch_audit，尚无AP结论；详情见Native三组说明。

### 2026-10-28 Native固定权重与固定框归因

新增native_paired（98源/CP/顺序审计＋500配对bootstrap）、native_attribution（C尾部correct/zero/donor、正确精确复现、A/B同C框/score/sourcechunk重解码）及一次性fixed_box_queue。只分析固定E10完整459，无训练修改/新loss/选模；供体包含支持，联合错配不称RGB必要性。详见[归因合同](dual_stream/UE_NATIVE_TAIL_ATTRIBUTION_20261028.md)。

### 2026-10-28 Native归因队列和首批配对区间

新增独立`ue_ubr_native_attribution_queue_20261028.py`最多两CPU后续队列，以及`ue_ubr_native_fixed_box_errors_20261028.py`固定bbox匹配的四区域净纠错。C正确干预459图精确复现，A/B同C框四卡已发射；original_path MaskAP区间已出且C-A/C-B为正，combined及内容干预未完成不下最终结论。详细分母/中间数值见Native归因合同。

### 2026-10-28 UAV完整预算与消融候选规划

新增[按证据推进的实验规划](dual_stream/UE_UAV_FULL_BUDGET_EVIDENCE_GATED_PLAN_20261028.md)：区分S/U/T/C结构矩阵、全路径RGB/显式几何对照、深度对应与1/3/5视角敏感性；矩阵为候选，按既有归因→最小主比较→必要消融逐阶段准入。明确无增益时简化/暂停、单种子和开发测试限制，不预承诺所有模块成立。仅文档，不改配置、数值源、训练/归因队列，不发射完整预算。

### 2026-10-28 Native全预算wrapper与臂隔离

新增experiments/ue_native_full100.sh、ue_native_full100_control.py、README规范；独立full_run与full_protocol，固定100轮26700、fresh共同初始化、同序、每轮完整val-best、全部完成后test。CPUprepare创建不可改Git运行工作树与orders；queue等待现有归因后GPU门/三臂四卡，arm目录/锁/恢复合同隔离。所有后续全预算实验必须有一键wrapper。详情[100轮合同](dual_stream/UE_NATIVE_FULL100_WRAPPER_20261028.md)。

### 2026-10-28 Native全100轮已登记持久队列

数值/运行器e8fe406，owner1576874，root ue_g_ubr_native_full100_20261028，105来源与100轮orders固定；独立runtime_source工作树。当前waiting_for_analysis，不抢先训练。重复queue幂等、gate身份/失败依赖负向CPU核验已过；fullGPU恢复门待前置完成后执行，审计root_queue_acceptance.json。

### 2026-10-28 全100轮提前共租释放

wrapper新增release，用户明确授权后仅解除分析完成等待，归档旧控制合同、核验并停止空闲owner，原e8fe406固定数值树恢复；GPU门/四卡准入不变，不终止分析。新owner1578097，审计control_overrides，详情全100轮文档。

### 2026-10-28 全100轮三组共租已核实更新

用户授权提前共租后full恢复门通过，三组各四rank均已E1 update25，AMP重试0；owner1578097，三数值合同相同、固定105运行源零漂移。root_co_rent_launch.json为实际审计；数值仍e8fe406，控制release8032d6a，分析未停止。

### 2026-10-28 Native归因结果核定与记录

新增[归因结果](dual_stream/UE_NATIVE_TAIL_ATTRIBUTION_RESULTS_20261028.md)：完整test C−A MaskAP+.03435/CI正，C固定尾部zero/donor明显下降，同C框C仍较A/B高约.039/.040；bbox C较A下降必须保留。GPU全完成，CPU部分Boundary与干预/固定框bootstrap尚未全完。队列优先让固定框像素归因先算，500预算不变；机器快照与SHA绑定。100轮任务继续不改合同。

### 2026-10-28 检测与掩码遥测sidecar

新增gradient_math/gradient_telemetry/detection_telemetry/telemetry_owner四个analysis模块、ue_native_telemetry.sh一键start/status及两组11项CPUtests。逐epoch完整val独立框误差FPFN，E1/10/25/50/75/100四卡固定8图任务gradnorm/cos，未知loss拒绝、aux单列、None0固定union、buffer恢复及CP源SHA审计。只观察不参与训练/选权，原full运行树不改；GPU需空闲15872MiB/card，真模型验收待资源。详见[遥测定义](dual_stream/UE_NATIVE_TASK_TELEMETRY_20261028.md)。

### 2026-10-28 只读遥测已实际部署

observer1aff788、PID1642915，CPU逐epochval已回填9份无失败；11项CPUtests与A/E1 459图验收通过。full105源未改、ABC未重启。四卡梯度门因空闲1.3GiB/card不足仍排队，尚无梯度数值，审计telemetry/root_enable_receipt.json。

### 2026-10-28 SUTC 独立开关接线

新增 `rsprompter/uav/native_sutc.py`（不可变 U/T 开关、纯卫星/尾部证据模型和独立 UBR 安装）、`scripts/probes/ue_ubr_native_sutc_runtime_20261028.py`（SUTC 初始化/目标/正式候选消费路由）、`configs/ablations/ue_native_sutc_contract_20261028.json` 及 `tests/test_native_sutc.py`、`tests/test_native_sutc_runtime.py`。四组同构 native UBR；S 不构造或读取样本 UAV，T 独立 P0 证据不写回卫星且无实例 loss/补框；U/C 复用原 B/C。UDPR 在新四格构建时禁用，不物理删除历史复现源码。不修改在训 ABC 源码。新 runtime 为工程 API，不是训练 wrapper，真实 S/T 四卡门待执行。详见 [SUTC 实施合同](dual_stream/UE_NATIVE_SUTC_SWITCH_IMPLEMENTATION_20261028.md)。

### 2026-10-28 视角子集敏感性接线

新增 `rsprompter/uav/view_subset.py`（canonical槽位/所选源编码/G读取）、`data/ue_view_subset_inputs.py`（子集原始深度重建和严格loader）、`scripts/probes/ue_build_view_subset_cache_20261028.py`（隔离缓存CLI）、`scripts/probes/ue_view_subset_cpu_gate_20261028.py`（真实单train图I/O/旧缓存等价门）、`configs/ablations/ue_uav_view_sensitivity_20261028.json`、`tests/test_uav_view_subset.py`。SUTC工厂支持 `view_count=1/3/5` 或显式 `uav_view_ids`，全路径同步且不变容量；减少视角须完整子集缓存，S仍无UAV。几何五通道不是视角数，计数保持/5单位，未选RGB不编码。详见 [子集接线说明](dual_stream/UE_UAV_VIEW_SUBSET_WIRING_20261028.md)；38CPU测试及真实图001234门通过，真实四卡/全缓存/新训练未执行，在训ABC冻结源码不变。

### 2026-10-28 收紧后续实验路线

新增 [三方向分阶段实验路线](dual_stream/UE_COMPACT_EXPERIMENT_ROADMAP_20261028.md)：先S/U/T/C结构主比较；仅对保留结构新增常量RGB、关闭显式几何两组；有相对S可靠净收益后再新增1/3视角训练并复用5视角。禁止全结构×模态×视角展开，不安排双关/固定平面/逐入口模态矩阵，不据此声称RGB-几何交互或深度对应必要性。优先严格合同复用，模态入口仍待实施；本次仅记录，不发射训练、不改在训合同或遥测。

### 2026-10-28 三阶段完整wrapper与模态入口

新增 `scripts/experiments/ue_staged_uav.sh`、`ue_staged_uav_control.py`、`ue_staged_uav_matrix.py`，`scripts/probes/ue_staged_uav_run_20261028.py`，`scripts/analysis/ue_staged_uav_analysis_20261028.py`，两份staged协议/矩阵JSON，`rsprompter/uav/modality_controls.py`及对应5组CPU测试；SUTC工厂新增全路径RGB/显式几何控制。wrapper按structure/modality/views独立启动，阶段内所有gate通过并核初始状态后才训练；结构先val锁基座再显式test；模态/视角各只新增两组并复用基座。完整100轮数学复用旧full runner、source/checkpoint/五流恢复隔离；显存允许四卡任务共租。显式analyze只CPU并输出四指标500配对bootstrap。详见 [wrapper与单变量审查](dual_stream/UE_STAGED_UAV_WRAPPERS_20261028.md)。本次不启动正式训练或真实GPU门，在训ABC不改。

### 2026-10-28 三阶段wrapper CPU冻结验收完成

数值/控制提交a95a444；正式待启动root为`/data/wangcheng/checkpoint/ue_staged_uav_full100_20261028`，536源码/配置及100轮101份订单文件核定，88CPU测试通过，CUDA未初始化，未发射训练。CPU_PREPARE_AUDIT.json记录源码闭包与缺cache硬门；第一次伪_ops.py误判的失败产物单独归档prepare_attempt1保留，不作正式runtime。缓存/真实四卡门仍待后续start执行；本次仅文档更新，不改已冻结数值源。

### 2026-10-28 full100中期C臂检测分支负收益分析

新增 [检测分支负收益量化、梯度图与隔离方案](dual_stream/UE_NATIVE_DET_COST_GRADIENT_AND_ISOLATION_20261028.md)：只读分析在训full100（E59快照，val）。bbox负收益持续成立的是C−B（−0.2~−0.4pp，中大型承担，小目标bbox +0.35），C−A后期已追平；E10短研究的B−A −1.19pp结构代价在100轮日程消失。遥测证明损失不在框回归（TP误差三臂相同）与召回（C更好），而是combined视图背景FP增多=分数排序退化，original_path三臂不变。给出含行号的完整梯度图：C独有耦合=M损失经native_aligned_rgb（不detach，`ubr_native_integration.py:34`）训练UAV编码器，其输出aligned位于block4写回与实例头两条检测前向上游。隔离首选端口级detach（对应a3sgm的SHAPE_DENSE_DETACH先例）+预注册C_detach臂；四卡梯度余弦遥测应先出数。附C对纯单流（S臂）设计判断：mask预期薄而稳超越(+0.5~1.5pp)，bbox或平或负。全部为中途val、单seed、无配对CI，不作test结论，不改动在训合同。
