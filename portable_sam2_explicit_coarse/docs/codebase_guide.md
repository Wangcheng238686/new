# PromptMiner-SAM2 (a3sgm) 代码库导读

> 清理终态（cleanup/a3sgm-only，2026-10-07）。本文件是当前代码的唯一权威地图；
> `docs/a3sgm_architecture_guide.md` 的 `文件:行号` 引用对应清理前提交 add7482，
> 机制描述仍然有效，行号请以本文件为准。每轮清理的变更清单见 `docs/cleanup_rounds.md`。

## 1. 项目架构说明

仓库三层：**论文资产**（`icassp2026_promptminer/`、`fig1_components/`、`fig1_rows.png`、
图件脚本）＋ **模型代码**（`portable_sam2_explicit_coarse/` + 顶层 `sam2/` vendored 官方仓）
＋ **实验证据**（`docs/`、`logs/`、两张追踪表、`paper-reviews/`）。

模型 = RSPrompter-anchor 单流两阶段检测器（SAM2 Hiera-B+ LoRA 编码器 → PAFPN → RPN/RoI）
＋ **CSPM**（粗形状提示挖掘：RoI 特征 → 64×64 coarse logits → 2P2N 点 + 框 + dense canvas
三模态提示喂冻结 SAM2 PromptEncoder）＋ **UDPR**（margin 尾：K=64 最小|z|选点、±2 tanh 残差、
部署阈值锚定损失）。两个正式入口：WHU-1024 全量 fast150（roi_local 监督、PE 冻结）与
NWPU VHR-10 matrix300（full_image + roi_balanced_dice、PE 下采样栈 0.1×LR 可训练）。

### 文件职责与依赖（portable_sam2_explicit_coarse）

| 文件 | 职责 | 依赖 |
|---|---|---|
| `rsprompter/anchor_detector.py` | 顶层检测器：特征抽取/损失/预测入口、stride16 选择 | sam2_vision, sam2_neck |
| `rsprompter/anchor_roi_head.py` | RoI head：采样、bbox 损失(fp32 可选)、mask 前向、三大损失接线（最终 mask/尾 margin/coarse 辅助）、空秩 DDP 补偿、分块推理 | sam2_mask_head |
| `rsprompter/sam2_mask_head.py` | 因果链核心：__init__ 契约、Step4-10 前向（coarse→dense canvas→挖点→PE→dense embedding→decoder→尾）、双坐标 predict | injector, miner, tail, helpers, targets, sam2_decoder |
| `rsprompter/sam2_mask_head_helpers.py` | mixin：dense canvas 变换粘贴、PE 输入选择、无效点置零、gate 有效值、框内 support、GT 点位遥测 | dense_prompt_utils |
| `rsprompter/sam2_mask_head_targets.py` | mixin：最终 mask 目标（fi/roi_local 双契约）、roi_balanced_dice、coarse 目标、loss_and_target | — |
| `rsprompter/shape_prior_injector.py` | SmallMaskDecoder（coarse 头，roi_only） | — |
| `rsprompter/shape_point_miner.py` | 2P2N 挖掘（无参数纯函数式，平台中心选取+三级 fallback） | — |
| `rsprompter/dense_prompt_utils.py` | raw_logits 变换、框内粘贴、无效恢复基底 | — |
| `rsprompter/decoder_tail_refiner.py` | UDPR margin 尾（耦合容器：梯度回传主干） | — |
| `rsprompter/sam2_decoder.py` | SAM2 MaskDecoder 严格加载包装 + upscaled hook | vendored sam2 |
| `rsprompter/sam2_vision.py` | LoRA 编码器加载/冻结、no_mask_embed 加载、PE 权重加载 | ckpt_utils |
| `rsprompter/sam2_neck.py` | RSSAM2PAFPN（stride 4/8/16/32/64，五层 512ch） | — |
| `rsprompter/architecture_contract.py` | **冻结区**：架构 ID 与 cfg.model sha256 指纹 | — |
| `rsprompter/coarse_mask_loss.py` | coarse BCE+Dice（epoch 权重调度、boundary 指标遥测） | — |
| `rsprompter/ckpt_utils.py` / facades×3 | 严格加载工具；`models*.py`/`shape_prior.py` 为注册 facade | — |
| `train/train_rsprompter_fusion.py` | DDP 训练器：参数组/EMA/断点/四别名选模/早停/解耦裁剪/遥测聚合 | rsprompter, data, utils |
| `inference/infer_from_checkpoint.py` | 自描述 ckpt 重建+推理+评测（契约校验、compat 变体、旧 ID 规则） | rsprompter, data, utils |
| `data/` | WHU COCO 实例数据集 + 训练/验证 loader（含 VHR10 格式） | — |
| `utils/coco_eval_utils.py` | COCO GT/DT 构建与评测唯一实现 | — |
| `configs/a3sgm_{whu1024,vhr10}.py` | 静态模型配置（指纹=论文 ckpt 训练期契约） | — |
| `configs/ablations/vhr10_*.py` | 论文消融 P/B/P+B/P+B+M 的 delta 配置 | `_base_` 上述两文件 |
| `configs/environment*.sh` | 机器路径（python/SAM2/数据/ckpt 根） | — |
| `scripts/ablations/*.sh` | 两入口瘦启动器（env 只剩数据/调度策略） | configs |
| `scripts/infer_{whu,nwpu}_checkpoint.sh` | 推理入口包装 | inference/ |
| `scripts/smoke/check_a3sgm_*.py` | 等价性回归门（加载+前向哈希，金样在同目录 json） | — |
| `tools/` | bootstrap 配对统计、Boundary-AP、原尺度评测、可视化 | — |

## 2. 代码阅读路线（建议顺序）

1. `docs/a3sgm_architecture_guide.md` §0-§1（一页定位 + 架构图）——机制全景。
2. `configs/a3sgm_vhr10.py` —— 模型长什么样（静态 dict，键名即模块语义）。
3. `rsprompter/anchor_detector.py`（222 行）—— 数据怎么流进两阶段。
4. `rsprompter/anchor_roi_head.py` —— 训练/推理在 RoI 层怎么组织（三大损失在哪接线）。
5. `rsprompter/sam2_mask_head.py` —— **核心**：先读类 docstring 的 Step4-11 流程图，
   再按 `_forward_coarse → _forward_prompt_encode → _forward_dense_embeddings → forward →
   _predict_by_feat_single` 顺序读。
6. `rsprompter/shape_point_miner.py` 附录级细节（`docs/a3sgm_architecture_guide.md` 附录 A 有逐槽讲解）。
7. `rsprompter/decoder_tail_refiner.py`（228 行）—— margin 语义与 DDP 归一。
8. `train/train_rsprompter_fusion.py` 的 `main()` —— 训练循环/选模/断点。
9. `inference/infer_from_checkpoint.py` 的 `_resolve_model_config` —— ckpt 自描述重建与契约校验。

## 3. 关键接口清单（入参/返回/副作用）

| 接口 | 入参 → 返回 | 副作用 |
|---|---|---|
| `RSPrompterAnchorMaskHeadSAM2.forward(x, image_embeddings, image_positional_embeddings, roi_img_ids, high_res_features, boxes, …)` | RoI 特征 [N,512,14,14] 等 → 6 元组 `(mask_logits z′, iou, coarse_outputs, {}, quality=iou, mask_tokens)` | 写 `self._last_tail_outputs`（尾损失输入）与 `_last_uav_debug_stats`（遥测总线） |
| `DecoderTailPointRefiner.forward(logits[N,1,H,W], upscaled[N,32,H,W], mask_tokens[N,1,256])` | → `(refined z′, {indices, delta, selected_abs, refined/base_selected_logits, base_logits_detached})` | 无（耦合容器：不 detach，梯度可达主干） |
| `DecoderTailPointRefiner.point_loss(tail_outputs, targets)` | → `(loss, TAIL/* 遥测 dict)`；DDP 全局正负点归一 | 两次 all_reduce（空秩经 `empty_point_loss` 镜像） |
| `ShapePointMiner.forward(coarse_logits[N,1,64,64], boxes[N,4], image_size)` | → `(coords[N,4,2], labels[N,4]∈{1,0,-1}, SP/* 统计)`；全程 no_grad | 写 `_last_local_yx/_labels`（GT 遥测用） |
| `_MaskHeadTargetsMixin.loss_and_target(mask_preds, sampling_results, batch_gt_instances, rcnn_train_cfg, final_mask_pos_priors)` | → `{loss_mask, mask_targets, debug_stats}`；fi/roi_local 双路径 | 无 |
| `RSPrompterAnchorRoIPromptHead.mask_loss(...)` | → mask_results dict（含三大损失） | 写 `_last_coarse_stats` 等遥测 |
| `architecture_contract(model_config_dict)` | → `{schema_version, architecture_id, model_fingerprint}` | 纯函数，**冻结** |
| `infer_from_checkpoint.py --checkpoint … --build-only` | 重建模型+strict load，打印契约 | 进程内设置 SAM2_REPO/CKPT env |
| 训练器 `--config configs/a3sgm_*.py` | DDP 训练，产出四别名 ckpt（best/bbox/composite/last + EMA state） | checkpoint 含 config_snapshot（自描述） |

## 4. 项目编码规范（新增代码请遵守）

1. **行为即契约**：任何影响 cfg.model 的改动都会改变架构指纹并拒载旧 ckpt——改模型形态前先跑
   `scripts/smoke/check_a3sgm_checkpoint_load.py` 与 `check_a3sgm_forward_equivalence.py`。
2. `rsprompter/architecture_contract.py` 与 `inference/infer_from_checkpoint.py` 的契约比对区
   （含两个 compat 变体与 `_is_legacy_pre_tail_id`）为冻结区，只加注释不改逻辑。
3. 死臂参数模式：`__init__` 参数保留（ckpt 嵌入 config 会传键），启用即 `raise ValueError`
   并写明属于哪个已删臂——禁止静默吞掉。
4. 遥测前缀约定：`SP/*`（点挖掘）`COARSE/*`（粗头）`GATE/*`（dense 门控）`DENSE/*`（dense 通道）
   `TAIL/*`（尾）`BOX/*`/`JITTER/*`（提示框总线）；计数键由 roi_head 的 `_ratio` 派生命比率；
   遥测只写日志不参与 loss。
5. 注释只写"为什么"（如 margin 必须锚定部署阈值 logit、空秩双 collective 防死锁、
   `**kwargs` 吞旧 ckpt 键的原因）；不复述代码在做什么。
6. 禁第三方新依赖、禁一行式压缩、单一职责、guard clause 提前返回、变量就近定义。
7. 配置改动走 `configs/ablations/` 的 `_base_` delta（加减键），不复活 env 开关。
8. GPU 上的数值等价验证必须先做同代码双跑标定噪声带（scatter 原子操作使 bit 级复现不可达）。
