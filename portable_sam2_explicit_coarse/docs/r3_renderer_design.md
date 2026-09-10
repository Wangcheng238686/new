# R3：专职 dense 画布渲染头——设计提案 v0.1

日期：2026-09-10。状态：已实施，待 matrix300 训练。证据基础：[r3_canvas_oracle_matrix300.md](r3_canvas_oracle_matrix300.md)。
决策纪律：§5——新方向独立设计文档，开发可与外部机器并行，评测一律本机。

## 1. 证据 → 约束

| 实测（探针） | 设计含义 |
|---|---|
| GT 画布 @当前门控 +0.0132(E300)/+0.0162(E105)，CI95[+0.0082,+0.0187] | 通道价值成立，值得专用容量 |
| 画布 IoU 0.766-0.782；coarse 源 0.892 Dice≈0.81 IoU 当量；传输损 ~0.02 | 瓶颈=coarse 头内容；paste 管线复用不动 |
| 两点斜率：过 +0.005 采纳线需 IoU≈0.87（+0.09） | 分辨率+特征+容量三管齐下 |
| box 探针 AP75 增益≈2×mAP | 边界是价值所在 |
| E105 门控欠开(+0.0062)/E300 过开(−0.0029)；好内容下 gate1 更优(+0.0245) | 门控可学习、不调度；内容修好后有协同红利 |
| support 限制 +0.0005 免费 | 不动支撑域逻辑 |
| A2e 外溢教训（bbox −0.015） | 辅助权重守 0.10，盯 bbox 外溢 |

## 2. 模块设计（六轴）

1. **解耦**：现 coarse 头保留，仅喂 shape point miner（2P2N）；新 `CanvasRenderer` 专职画布。
2. **特征**：PAFPN 两级（P3 高分辨率 + P4 语义）各自 ROIAlign→1×1 投影到 128d→上采样对齐→
   相加融合（结构仿 PAFPN top-down，已验证的实现族）。
3. **头**：融合特征 → 2 层 3×3 conv（128d→64d）→ 1×1 出单通道 logit @128×128；
   **终层零初始化**只是稳定初始化，不能称为与 `masks=None` 等价的中性路径。
4. **监督**：per-positive-RoI Dice+BCE vs 渲染分辨率 GT（复用现有
   `get_coarse_targets` 的 floor/ceil crop 和 nearest resize 语义，grid=128）；唯一新增
   loss 的总权重固定 `0.05`。不加 boundary、distillation、gate 或负例 loss，避免扩大本轮
   消融变量；unmatched ROI 行为列为 checkpoint 后的审计探针，而非暗中加入的训练机制。
   监督点在渲染分辨率（pre-paste）——paste 传输损已证仅 ~0.02，端到端监督画布的
   复杂度（多 ROI 贴画交叠）不值得。
5. **消费端不动**：渲染 logits → 现有 `_shape_prior_to_prompt_mask`（transform→resize→
   paste→clamp ±8）→ 冻结 PE mask_downscaling（md 解冻）→ 门控 α（可学习 0.5 起）。
6. **臂定义**：`r3`（tag `pb_r3`）= PB(points_box) + CanvasRenderer + dense delivery + md 解冻；
   `r3_udpr`（tag `pb_r3_udprk64`）只在相同 R3 配置上增加既有 A3 UDPR-K64。
   - vs PB：R3 净增益（判据：matrix300 平台 ≥ +0.005）
   - vs A0(PBM)：画布质量叙事（A0 旧画布对 PB −0.005 作阴性对照）
   - A3(PBM+UDPR) 不受影响，独立行。

## 3. 筛选与验收（预登记）

- **筛选**：dev100 同协议 A/B：`a0_r3`（a0 配置但画布源换 CanvasRenderer）vs 既有 A0
  （0.6467 平台/0.6669 best）——同协议同 dense 家族，合法比较；判据：平台 ≥ +0.005。
- **机制验收遥测**（训练期可得，不等 mAP）：
  1. 渲染 Dice / 画布 IoU（探针口径，checkpoint 上可测）应爬向 ≥0.85、趋势向 0.87；
  2. 门控轨迹应随内容改善而开（E105 样板行为），E300 不应再现"内容平门控追高"的剪刀差；
  3. bbox 平台外溢 ≥ −0.005（A2e 红线）。
- **定稿**：过筛选后 matrix300 `pb_r3` 行；last/best 双口径 + 探针复查（训练后权重上
  drop_canvas/换 GT 的依赖结构变化——渲染器练满后 GT 替换的增量应显著缩小）。
- 风险登记：unmatched/false-positive ROI 的画布行为（oracle 未测，训练中监督只在正例上）；
  训练协同 vs 冻结 oracle 的差（oracle 是上限不是预测）；成本 +~10-15%/步（双 coarse 前向）。

## 4. 实现清单（批准后）

- `rsprompter/canvas_renderer.py`（新模块，多尺度融合+conv 头+零终层）
- `rsprompter/sam2_mask_head.py`：可选 `canvas_renderer_cfg`；缺失该键即不构建模块，旧路径不变。
- config 只在 `CANVAS_RENDERER_ENABLED=1` 时物化 R3 键；严格 checkpoint 重放据嵌入配置恢复该环境。
- `verify_p2v2_arms.py` 增加 `r3/r3_udpr` 构建与差分断言：`r3→r3_udpr` 仅差 UDPR config。
- `vhr10_p2v2_dev.sh`、eval wrapper 已接入两臂；matrix300 series 默认仍是冻结的 P/PB/A0/A3，
  必须显式传 `r3 r3_udpr` 才会新增实验，不重跑或改变历史行。

## 5. 不做

- 不动 paste/支撑域/embedding 管线（探针判定已够好）；
- 不做画布 dropout / 双路互备（box-dropout 是另一独立方案，见
  [box_prompt_dependence_vs_contribution.md](box_prompt_dependence_vs_contribution.md) §5，
  两者正交可叠加但不捆绑）；
- 不碰 P2/残差线（本机 2×2 已判死）。
