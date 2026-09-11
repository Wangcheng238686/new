# A0 prompt 依赖性探针：dense vs box（matrix300 双权重）

日期：2026-09-11 16:08。权重：`vhr10_p2v2_matrix300_a0_pbm_d5b_tr1.0_va1.0` 的
best(E105) 与 last(E300)。工具：`inference/probes/prompt_switch_probe.py` 四模式，
`--split validation` 全量 130 图。

动机：box token dropout 可行性分析（本日）预登记的判别实验——A0 的 dense 瓶颈在
**用量**（decoder 被 box 捷径遮蔽、不读画布）还是在**内容**（画布质量不足）。
判读规则事先固定：drop_dense ≈ 0 → 用量侧有解锁空间，dropout 立项依据强；
drop_dense 已是 −0.01 量级 → dense 已被使用，瓶颈在内容侧，应转内容程序。

## 结果（segm/mAP；baseline 与 16 权重表逐位一致，口径自检 ✓）

| 模式 | best E105 | Δ | last E300 | Δ |
|---|---:|---:|---:|---:|
| baseline（points+box+dense） | 0.6446 | — | 0.6331 | — |
| drop_box | 0.6134 | −0.0312 | 0.6214 | −0.0117 |
| **drop_dense** | 0.5808 | **−0.0639** | 0.6156 | **−0.0175** |
| drop_both | 0.4481 | −0.1966 | 0.5720 | −0.0611 |

参考：PB best(E155) drop_box −0.0357、drop_dense 0（无 dense 通路，阴性对照）。

## 判读

1. **"dense 装饰性/被 box 遮蔽"假设证伪。** E105 上 drop_dense（−0.064）是
   drop_box（−0.031）的两倍——decoder 早已重度读取画布，且比读 box 更重。
   用量侧不存在"解锁空间"：用量不但已拉满，E300 还回落到 −0.0175。
2. **依赖 ≠ 净贡献，且模式与 box 自身同构。** A0 平台 0.6342 ≈ PB 0.6395
   （dense 包净 −0.005）：dense 拿走了 decoder 的依赖却没有换来超过 box+crop
   的产出——**替代而非叠加**。这与 box 的"依赖 −0.036 vs 贡献 +0.014"是同一
   结构：prompt 模态的依赖度（分布偏移上界口径）与训练间净贡献是两个量。
3. **训练后段依赖整体回落、联合塌缩超加性稳定。** E105→E300：box −0.031→−0.012、
   dense −0.064→−0.017、both −0.197→−0.061；drop_both ≈ 2.1×（单模态之和）在
   两个阶段都成立——两模态互为部分补偿，裁剪内容路在后段承接了余下负载。
4. **A0 终盘几乎不依赖 box（−0.0117）。** 推论：A0 实际已成为"points+dense 为主、
   box 为辅"的模型，但仍不超过 PB——dense 替代 box 的角色而没有做得更好。

## 对 box token dropout 方案的判罚（按预登记规则落判）

E300 drop_dense = −0.0175，落在"−0.01 量级"分支 → **瓶颈在内容侧，转内容程序**。

- 路径 A（用量解锁）：被本探针直接证伪（判读 1）。
- 路径 B（用量带动内容：dropout → dense 梯度增强 → coarse 头改善）：同样被削弱——
  E105 的用量已经很高，dense 通路梯度已充分存在，而画布内容仍封顶在 coarse 源
  ~0.81 IoU 当量。封顶更可能是 coarse 头容量/信息 intrinsic 上限，不是梯度饥饿。
- dropout 的剩余价值仅剩 PB 双路机制收益（§5 原始动机：预期 ≥PB 绝对值），那是
  鲁棒性目标，**不是**"让 dense 生效"的候选路径；如仍要做，按 dev100 A/B 协议
  独立立项，不再挂 dense 叙事。

## 内容侧优先级（"让 dense 生效"的下一程）

oracle（`r3_canvas_oracle_matrix300.md`）已证 decoder 会把更好的画布兑现成 mAP
（GT 替换 +0.013~+0.016，CI 下界 +0.0082），本探针证明受众真实存在（依赖 −0.064）。
缺口是内容：learned 画布 IoU 0.782，过 +0.005 采纳线需 ≈0.87；coarse 源自身
~0.81 当量，传输损耗仅 ~0.02。候选方向（按证据强度排序）：

1. **coarse 源内改良**（R3 教训：换源 0.7614 更差，源内改良优于换源）：
   `COARSE_MASK_OUTPUT_SIZE` 64→128 / coarse 头容量 / 监督信号质量。
2. **训练-推理分布对齐**：训练正例多为 GT 框、推理画布渲染在 RPN 框上
   （0.81→0.782 的内容落差）——box-jitter 化的正例采样可直接收窄。
3. 门控协同在收敛段已死（E300 gate 全开 −0.0029），E105 型"早开大门"协同
   不可在终盘兑现，不作为独立方向。

## 工件

| 项 | 路径（本机未跟踪，协议 §4） |
|---|---|
| E105 探针 JSON | `refactor_golden/prompt_switch_a0_best_model_epoch105.json` |
| E300 探针 JSON | `refactor_golden/prompt_switch_a0_last_model_epoch300.json` |
| 探针脚本 | `inference/probes/prompt_switch_probe.py`（四模式，同 PB 探针） |
| 运行日志 | 本会话后台任务 stdout（2026-09-11 16:04–16:08，GPU0 空闲窗口） |

注意（判读边界）：drop_* 是分布偏移干预（训练从未见过缺模态输入），数值是
依赖性上界口径，不是模态信息量；单 seed 单权重，不做显著性宣称。
