# Differentiable Candidate Prompt Miner：点挖掘后续路线

状态：设计冻结，未实施、未加入任何现有矩阵。本文只定义 P/PB 路线在 DenseCap 结论之后可单独研究的下一步，不改写既有 P、PB、A0、A3 或其已产出的结果。

## 1. 问题与目标

现有 `ShapePointMiner` 从 coarse logits `z=H0(x)` 以阈值、TopK、距离约束和 fallback 挖掘 2P2N。该过程在点坐标处离散，且 `z.detach()` 后才进入 miner：

```text
z -> stopgrad -> hard 2P2N coordinates -> PromptEncoder -> MaskDecoder -> L_mask
z -----------------------------------------------------------------> L_coarse
```

所以 `L_mask` 不能训练“选哪个点”；`H0` 仅由既有 coarse BCE+Dice 训练以生成点。目标不是否定这一稳定基线，而是在新模块中检验：最终 mask 监督能否学习更有 decoder 效用的 prompt 选择，同时保持正/负点几何语义和 SAM2 的原生硬点输入。

## 2. 设计原则

1. 现有 P 是不可变 hard-2P2N 对照；新路线不得替换其实现或重用其 tag。
2. 正、负、两两距离、safe-background 和 fallback 的几何约束优先于端到端可微性。
3. 前向仍向原生 PromptEncoder 送真实的四个离散 64-grid 点；只在反向使用连续近似。
4. 第一阶段不新增 loss 类型：保留 `L_det + L_mask + 0.1 L_coarse`。若发生塌缩，才把多点分散正则作为单独、显式变量研究。
5. 不与 A0/DenseCap 混跑；先只在 point-only P 路线检验点选择因果链。

## 3. 推荐模块：候选集上的 Straight-Through Gumbel selector

不采用全图 soft-argmax。先由当前 hard miner 的规则构造每个 RoI 的 detached 候选集合：

```text
C+ = high-confidence foreground / valid interior candidates
C- = safe background / distance-qualified candidates
```

候选集只承担几何安全边界，不直接从 `L_mask` 接收梯度。对每个 slot `k`（`p1,p2,n1,n2`），由 coarse logits 或一个新增但小型的 selector score head 产生候选分数 `s_k(i)`：

```text
p_k(i)  = softmax((s_k(i) + g_i) / tau),  i in C+/C-
y_hard  = one_hot(argmax_i p_k(i))
y_ST    = y_hard - stopgrad(p_k) + p_k
c_k     = sum_i y_ST(i) q_i
```

`q_i` 是候选的连续 1024-space 坐标。前向时 `y_ST` 数值等于 `y_hard`，故 PromptEncoder 收到一个真实、离散、与现行 P 同类型的 point prompt；反向时梯度按 `p_k` 回传：

```text
L_mask -> PromptEncoder(c_k) -> c_k -> p_k -> s_k -> H0
```

对同符号第二个 slot，先对第一 slot 邻域做 detached 或可微 soft suppression，再选择第二个，保持现有点间距语义并降低塌缩风险。温度 `tau`、Gumbel 随机性、候选规则和第二点抑制必须全部写入 `cfg.model`/architecture contract，不能依赖 wrapper 隐式默认。

## 4. 不推荐作为首实现的替代方案

| 方法 | 优点 | 当前项目的主要风险 |
|---|---|---|
| 全图 soft-argmax / DSNT | 最简单的连续坐标梯度 | 多峰平均导致点落在目标间；正点、负点和多点分散均易失效。 |
| expected prompt embedding | `sum p_i PE(q_i)` 梯度平滑 | 不再是原生“一个真实点”输入，改变 SAM2 prompt 语义和论文归因。 |
| REINFORCE / policy gradient | 严格处理离散动作 | 以 final mask 为延迟 reward，方差大、样本效率差，不适合 520 张训练集。 |
| 可微 top-K / Sinkhorn 全局排列 | 理论上可同时选 K 点 | 复杂、显存/数值预算高，且会把几何约束与选择机制混在一起。 |

## 5. 渐进实验与单变量对照

先不加入 matrix300。开发顺序为：

1. **P-hard**：现有 P，不改动；
2. **P-ST**：相同 coarse head、候选集合、2P2N 数量、box 状态、loss 和训练协议；仅将最终 hard TopK 换为 ST selector；
3. 若 P-ST 塌缩，单独新增 **P-ST+diversity**，只增加明确记录的 second-slot repulsion 或 diversity loss；不得把它混进 P-ST；
4. 只有 P-ST 的机制门和 paired dev 指标均正，才决定是否建立 PB-ST；DenseCap 与 UDPR 均保持关闭。

`P-hard -> P-ST` 是首个可声明的单变量比较。不要把 P-ST 与 A0 的 dense package、R3 或任何改变 dense source 的实验混作同一消融轴。

## 6. 预注册遥测与停止门

实施时新增独立 `DPM/*` 字段，并明确其 latest-forward 或 epoch/DDP 范围：

- `selector_final_mask_grad_group_l2_rms`：`L_mask` 对 selector/H0 selection-score 参数的梯度；为零表示没有实现目标链；
- `slot_entropy`、`temperature`：选择分布是否过早塌缩或长期过平；
- `hard_soft_coordinate_gap`：ST 前向硬点与 soft expectation 的距离；
- `same_sign_pair_distance`、`collapse_ratio`：两正/两负点的分散；
- 保留现有 `p*_in_gt_fg`、`n*_in_gt_bg`、fallback 与 invalid 比例，禁止用 mAP 掩盖几何退化；
- paired validation mAP 及 paired bootstrap CI。

停止条件：final-mask selector 梯度持续为零、点塌缩/GT 命中率明显恶化、或经预注册 dev 预算后 P-ST 相对 P 的 paired CI 仍不支持正增益。所有 telemetry 仅作机制证据，不能替代完整 validation 与 bootstrap。
