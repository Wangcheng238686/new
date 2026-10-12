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

1. **固定 64-grid 的 coarse source capacity feasibility spike**：R3 教训表明另换源更差，
   而 GT64 仍有 +0.0151 上限；保持 proposal geometry、64-grid target、BCE+Dice、
   paste 与 decoder 不变，仅扩大 `SmallMaskDecoder` 的中间容量，先盲评 val support-canvas
   IoU，再决定是否有资格训练 A0-Capacity64。
2. `COARSE_MASK_OUTPUT_SIZE=64→128` 已由 GT64/GT128 同支持域冻结 Oracle 否决：
   GT128−GT64=-0.000221，CI `[-0.000944,+0.000772]`，不作为训练臂。
3. 当前训练和测试均使用 proposal prompt boxes（不是 GT-box→RPN-box mismatch），故
   box-jitter 若研究只能视为独立正则候选，不能包装成 dense 分布对齐修复；E300 全局 gate
   全开为 −0.0029，实例 gate 的 train-only readout 亦失败，二者均不作为独立方向。

完整 A0-last gate/readout 与 GT64/128 裁决见 `a0_dense_route_verdict_20260911.md`。

## 工件

| 项 | 路径（本机未跟踪，协议 §4） |
|---|---|
| E105 探针 JSON | `refactor_golden/prompt_switch_a0_best_model_epoch105.json` |
| E300 探针 JSON | `refactor_golden/prompt_switch_a0_last_model_epoch300.json` |
| 探针脚本 | `inference/probes/prompt_switch_probe.py`（四模式，同 PB 探针） |
| 运行日志 | 本会话后台任务 stdout（2026-09-11 16:04–16:08，GPU0 空闲窗口） |

注意（判读边界）：drop_* 是分布偏移干预（训练从未见过缺模态输入），数值是
依赖性上界口径，不是模态信息量；下文 bootstrap CI 只量化这 130 张图的抽样不确定性，
不覆盖训练 seed 或权重选择，不能外推为多训练重复的显著性宣称。

## 复核升级（已完成：A0-last 全量只读 audit）

2026-09-11，以 E300 last、NWPU validation 130 图完成正式复核。新增的 unhooked
standard pass 与 hooked baseline 的完整输出 SHA256 完全相同（`b32c…816c`）；并且
standard、baseline、drop_box、drop_dense、drop_both 五格的 detector SHA256 均为
`9ff8…62f4`。同时，standard 的 COCO records 与原生产 inference records 精确相同。
因此以下差异只来自 PromptEncoder 的 box/dense 输入切换，不混入 proposal、类别、分数、
排序或评测图像集合的变化。

| 对比（treatment − baseline） | 点估计 ΔmAP | image-paired 95% CI | 重采样 |
|---|---:|---:|---:|
| drop_box | −0.01171 | [−0.02195, −0.00139] | 500, seed44 |
| drop_dense | −0.01748 | [−0.03058, −0.00460] | 200, seed44 |
| drop_both | −0.06105 | [−0.08990, −0.03493] | 200, seed44 |

三个 CI 都不跨零，故 E300 的“dense 实际被读取、box 为辅、联合移除显著塌缩”可作为该
**冻结权重的提示依赖性**结论；其强弱仍不得解释为模态信息量或训练期因果贡献。尤其，
`drop_dense` 的稳定负效应不支持把 training-time box dropout 包装成“使 dense 生效”的
机制；若将来做 dropout，只能以独立的鲁棒性/缺提示训练实验立项。

复核工件位于 `refactor_golden/a0_prompt_switch_audit_last/`：各 cell 的 COCO records、
manifest、`summary.json`，及三个 `bootstrap_drop_*.json`。该 probe 不修改模型、训练配置、
checkpoint 或既有矩阵结果。
