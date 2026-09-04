# V2 Prompt Mining 改造实施方案（高分辨率 Coarse + 轮廓点挖掘）

> 交接文档：由规划会话产出，交实施助手执行，规划方做审查。
> 分支 `machine2/fast-whu150`，主机 lthpc（4×RTX 3090, 24GB）。
> 本文档自包含：实施前不需要读其他会话记录，但**必须读完本文档再动手**。

## 0. 背景与已确证事实（实施者必读）

模型：RSPrompter-anchor 系后代 + SAM2（冻结 Hiera + LoRA），两阶段检测（RPN+ROI）+
prompt 挖掘（coarse 画布/形状点）→ SAM2 prompt encoder/decoder 出掩码。
当前 VHR-10（NWPU VHR-10，RSPrompter 520/130 划分，maxDet=100）最好成绩：

| run | 架构 | segm/mAP | checkpoint |
|---|---|---|---|
| vhr10_c5v2_400 | C5-v2 base+ | 0.6475 | `ablations/vhr10_c5v2_400_tr1.0_va1.0/best_model.pth` |
| vhr10_c5v2_ft200 | 上者+200ep@1e-4 | **0.6617** | `ablations/vhr10_c5v2_ft200_tr1.0_va1.0/best_model.pth` |
| vhr10_large600 | C5-v2 + large 编码器 600ep | 收官中(~0.653+) | `ablations/vhr10_large600_tr1.0_va1.0/` |
| （对照）RSPrompter-query 论文 | — | 0.675 | — |

**通路活性审计结论**（训练日志 DEBUG-* 统计 + `inference/probe_decoder_pathways.py`
反事实探针，两数据集核实）：

1. coarse 画布（64×64, stride-16）已达 **0.954 IoU**（VHR-10）——64 分辨率下饱和；
2. **P2-BRR 画布精修 Δ≈0 甚至为负**（WHU 0.853→0.848；VHR-10 0.954→0.953）；
   P2 框精修（refined_valid_count）与框抖动（num_jittered）**全程为 0**——死通路；
3. **decoder 只消费 sparse prompts**：去掉点/框 0.73→0.16；置零 dense 基底无变化
   （0.7320→0.7323）；置零高分辨率 s0/s1 特征**逐位不变**——dense 通道是聋的；
4. **负点病灶**：17% 的负点落在 GT 前景（n1_in_gt_bg=0.83）——负点从"预测画布背景"
   采样，airplane 阴影区被误当背景，等于在教 decoder 排除阴影。airplane 逐类 AP 仅
   0.27-0.34（AP50 0.95 / AP75 0.01 断崖 = "飞机+阴影"复合标注 vs 模型只分机体）。

**核心设计原则**（由上述证据决定，不可违背）：
- 价值只能经 **sparse 通道**（点）流动；dense 槽位是兼容性装饰；
- 一切新自由度必须 **恒等初始化、有界扰动、可单独回退**（zero-init + clamp + 分相训练）；
- 不许 bolt-on（8/24 研究 `docs/p2_decoder_side_findings.md` 已证明：对已共适应的
  decoder 临时加点/改点分布一律为负，oracle 点也 -2.3pt；消费端必须联合训练）。

## 1. 总体路线（三阶段，每阶段有验收门）

- **阶段 0**：死通路清理（纯减法，~0.5 天）
- **阶段 1**：高分画布 + 轮廓点挖掘（因果验证版，80ep 续训，3-4 天）★主体验证
- **阶段 2**（门控）：query-coarse 头替换生成器（仅当阶段 1 通过且暴露"单次生成"短板）

GPU 排程：large600 预计 9/4 下午收官；之后 4 卡空出按阶段顺序使用。

## 2. 阶段 0：死通路清理

代码位置：`rsprompter/models.py`（P2-BRR 实现/调用）、`rsprompter/models_sam2.py`。
要求：**全部做成环境变量开关，默认行为先不变**，避免破坏 WHU 侧复现：

| 开关 | 默认 | 说明 |
|---|---|---|
| `P2_BOX_REFINE_ENABLED` | 0（新增，默认关） | refined_valid_count 全程 0 的死通路 |
| `PROMPT_BOX_JITTER_ENABLED` | 0（新增，默认关） | num_jittered 全程 0 的死通路 |
| `P2_BOUNDARY_REFINER_ENABLED` | 保持 1 | 保留（叙事/契约），阶段 1 中其角色被重新接线 |

交付物：清理 commit + 一次 20ep 冒烟（协议不变跑通）+ DEBUG 统计确认无回归。
注意 `configs/whu1024_baseplus_explicit_coarse.py` 的架构契约（architecture_contract）
与 fingerprint 会随开关变化——新增开关必须能被 checkpoint 的 config_snapshot 重建
（沿用现有 env→config→snapshot 模式）。

## 3. 阶段 1：高分画布 + 轮廓点挖掘（本方案核心）

### 3.1 画布升级（stride-16 → stride-8）

- 位置：coarse 头（现 64×64 输出，`models.py` coarse 分支 + `models_sam2.py`）。
- 新输出：**128×128**（stride-8；不建议直接 256：显存与梯度噪声，先 128 验证）。
- **恒等初始化**：新头 = 现有头权重 + 双线性上采样最后一层（或 1×1 conv 复制），
  保证初始输出≈现有 64 画布的上采样（起始行为不变）。
- 监督：高分 GT（GT 掩码下采样到 128），BCE+dice，逐像素权重沿用现有 coarse 损失
  （含 foreground_threshold 等配置），损失权重不变。
- dense 槽位：128 下采样到 64 喂 SAM2 prompt encoder（decoder 无视 dense，已证），
  dense 残差通路（shape_dense_alpha 门）**保留但加 α 上限 clamp=0.5**（VHR-10 上
  曾涨到 0.876/Δratio 4.9，防它一家独大抢梯度）。

### 3.2 点挖掘改造（锚点 + 可学习偏移，2P2N 预算不变）

- 现状：shape-point 启发式从 64 画布取 2 正 2 负（`models.py` shape miner）。
- 改为：**锚点 = 现有启发式点（不变）；偏移头 = 从 128 画布局部特征回归每点偏移**，
  zero-init，clamp ±8px（1024 坐标系）；输出仍为 2 正 2 负（token 预算不动）。
- **负点 GT 合法性（必做，最高优先）**：负点采样/偏移的合法性判定改为"GT 掩码外
  ε 像素"（ε≈4px），替换现在的"预测画布背景"。监督：对落在 GT 内的负点施加
  惩罚项（margin hinge），目标 误落率 17% → <2%。
- 偏移监督两段：先用"到真轮廓最近距离"回归对齐（distance loss，权重 ~0.05 量级，
  参考 shape_prior 0.1 / boundary 0.05）；解冻后允许掩码损失端到端流经坐标
  （坐标保持浮点进 PositionEmbedding；若坐标在某处被 round/量化 → 改为
  straight-through（前向 round、反向恒等）；若工程受阻 → **回退：只用 distance
  监督**，仍交付大部分价值，端到端留作升级）。
- 正点至少 1 个从"边界带内"初始化（P2 边界带 machinery 在此服役：band 提取→带内
  取锚点），兑现"更好边界→更好点"叙事。

### 3.3 训练协议（沿用既定口径，勿改）

- 入口：以 `scripts/ablations/vhr10_fast400.sh` 为模板（**注意它包含完整架构导出段
  ——P2_BOUNDARY_REFINER_ENABLED=1 / SAM_IMAGE_EMBED_STRIDE=16 / SHAPE_DENSE_DETACH=1
  等，曾有入口漏导出导致整个 run 降配的事故，勿重蹈**）。新 RUN_TAG=vhr10_v2s1_80。
- 初始化：`--init-from` ft200 的 best（0.6617，base+ 编码器；不依赖 large——变量控制）。
- 调度：80ep，lr 1e-4，warmup 100 iter，其余全沿用（AMP、4 卡 batch1×2、增强、
  EMA 影子、maxDet100、best-only）。
- **两相训练**：第 1-25ep 冻结偏移头（只训高分画布 + 负点合法性惩罚）；
  第 26ep 起解冻偏移（偏移头 LR 为主 LR 的 0.3 倍）。

### 3.4 验收门（全部通过才算阶段 1 成立）

| 门 | 指标 | 阈值 | 工具 |
|---|---|---|---|
| G1 | 128 画布 IoU（train debug） | > 64 版画布 IoU（≥0.954→更高或至少持平） | DEBUG-COARSE |
| G2 | 负点 GT 误落率 | <2%（基线 17%） | DEBUG-SP |
| G3 | 点-真轮廓距离 | 较启发式点显著收窄 | DEBUG-SP 扩展 |
| G4 | val segm/mAP | ≥0.6617（不低于 ft200 基线）；偏移解冻允许 <1pt 暂态、10ep 内恢复 | 日志 |
| G5 | 反事实 | 轮廓点换回启发式点（推理期开关）→ mAP 掉 ≥0.5pt，证明 sparse 通道承载增益 | 新增探针开关 |
| G6 | airplane 逐类 | >0.34（ft200 水平）为佳 | 逐类评估脚本（见 6.2） |

止损线：val mAP 单边下行 >2pt 不回 → 冻结偏移只收画布+负点收益；画布损失不降 →
查高分监督数据管线（GT 下采样 nearest vs bilinear+0.5 阈值）。

### 3.5 阶段 1 通过后的决策树

- 通过且 airplane 显著涨 → 直接扩展为 200ep 完整版冲 0.675+；
- 通过但增益 <0.5pt → 检查 G5：sparse 通道没承载住 → 考虑阶段 2（query 生成器）；
- 不通过 → 回退收尾：负点修复 + 画布升档单独作为小改进保留（预计 +0.3~0.5）。

## 4. 阶段 2（门控）：query-coarse 头

**仅在阶段 1 通过且暴露"单次生成、无全局上下文"短板时启动**（例：airplane 阴影
轮廓画布仍画不全、集群实例画布粘连）。

- 结构：检测框初始化实例 query（box-conditioned），L=6 层 transformer 精修，
  掩码注意力（注意力在 stride-16，画布投影 stride-8），输出替换 coarse 头。
  Mask2Former 解码层从 mmdet 拆用；逐层深监督；deep supervision 权重按 mmdet 默认。
- P2 特征作为注意力最细输入层（"高分辨率影响生成"叙事由输入侧成立）；
- 初始化：从阶段 1 best 续训（`--init-from`），query 头 LR 1e-4、旧件 0.1×；
- 训练 120ep，两相（query 头先单独预热 20ep）。
- 验收：G1-G6 复测 + query 注意力可视化（airplane 含影轮廓定性图，论文素材）。

## 5. 工程与仓库规范（必须遵守）

1. **环境变量驱动、默认关闭**：所有新开关默认不改变现有行为（AMP/增强等先例模式）；
   每个开关必须写入 config_snapshot 可重建路径。
2. **架构契约**：新变体注册 architecture_contract 新 ID（如 `c6_*`），入口脚本显式
   导出全部架构变量（**C4 事故教训**：漏导出=整个 run 静默降配，日志契约行必查）。
3. **提交纪律**：每完成一个可验证单元即 commit（feat/fix/docs 前缀 + 中文说明 +
   关键数字）；训练日志与指标 json 归档 `logs/test_eval/`（沿用 vhr10_c5v2 模式）。
4. **DEBUG 统计扩展**：新增的度量（点-轮廓距离、负点误落率、128 画布 IoU）按现有
   DEBUG-* 模式进日志（每 50 optimizer step），保证审查方可远程判读。
5. 冒烟先行：任何训练前先 20ep 冒烟（`--max-train-batches` + 小子集），核对：
   契约行、损失构成、DEBUG 统计、显存（batch1 峰值应 <20GB）、负点误落率即时下降。
6. 评估口径：val（=报告集，RSPrompter 同构）+ maxDet100 + 逐类 AP（工具见 6.2）；
   raw/EMA 双测（`inference/infer_from_checkpoint.py --weights {model,ema}`）。

## 6. 现成工具与坐标（避免重复造轮子）

1. 训练/评估入口：`scripts/ablations/vhr10_fast400.sh`（完整协议模板）、
   `scripts/ablations/eval_vhr10_final.sh`、`tools/eval_vhr10_original_scale.py`。
2. 逐类/airplane 分析：参照 `logs/test_eval/vhr10_c5v2/REPORT.md` §4 的脚本逻辑
   （COCOeval precision 数组取逐类 + airplane 分 IoU 档；原尺度口径）。
3. 反事实探针：`inference/probe_decoder_pathways.py`（zero_base_dense/no_sparse 等
   开关，方法见 `docs/p2_decoder_side_findings.md`）。
4. 数据：`/data/wangcheng/dataset/NWPU VHR-10 dataset/`（coco_split/NWPU_instances_
   {train,val}.json + positive image set）；路径含空格，注意引号。
5. 断点续训：入口脚本支持 `RESUME_FROM=<ckpt>`（large600 已内置，其余脚本照抄）。
6. 通路统计基线值（对照用）：canvas64 IoU 0.954/0.853(VHR/WHU)、P2 refined-raw
   Δ≈0、n1_in_gt_bg 0.83、applied_delta_ratio 4.9/0.47。

## 7. 审查接口（规划方如何验收）

实施方在以下节点暂停并产出审查材料（日志片段 + 指标 json + commit 列表）：
A. 阶段 0 冒烟后；B. 阶段 1 训练第 30ep（两相切换点）与第 80ep（终值+全部验收门）；
C. 阶段 2 启动前（决策树判定材料）；D. 每次止损触发时。
审查重点：验收门数据的真实性（不许拿训练集指标充数）、恒等初始化证据
（续训首 epoch 指标应≈基线）、契约行完整、DEBUG 新统计的实现正确性。
