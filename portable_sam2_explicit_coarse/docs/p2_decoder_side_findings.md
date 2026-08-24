# P2/边界信息路线 — 终局记录（2026-08-24）

本轮在 decoder 侧（PA-SAM 启发）的实验结果与此前全部测量的汇总。
基线：paper_promptminer_rd_p2 epoch 66 固定副本（probe_pinned_e66.pth），val segm/mAP 0.7320。

## 通路解剖（反事实探针，inference/probe_decoder_pathways.py）

| 通路 | mAP | 结论 |
|---|---|---|
| 置零 s0/s1 高分辨率特征 | 0.73203（逐位不变） | 死通路，decoder 已完全忽略 |
| 去掉点+框稀疏 prompt | 0.1624（mAP_75 0.008） | 命脉：prompt 驱动的分割器 |
| 置零 dense prompt 基底 | 0.73227 | 近乎无效 |

## E0：无训练版 PA-SAM 式二轮解码（inference/probe_e0_iterative.py）

第一轮出掩码 → 熵×(σ(decoded)−σ(coarse)) 分歧采样 4 正 4 负点追加 → 二轮解码：

| | baseline | E0 |
|---|---|---|
| segm/mAP | 0.7320 | **0.6904 (−4.2pt)** |
| mAP_50 | ~0.904 | 0.9043（粗定位不受影响） |
| mAP_75 | ~0.854 | 0.8210（边界被扰乱） |
| mAP_s | ~0.39 | 0.3127 |

结论：bolt-on 二轮提示**显著负向**，与 K 扫描的分布外点伤害一致（8P8N −3.8pt、E0 12-token −4.2pt）。

## 全路线终局表

| 路线 | oracle 天花板 | bolt-on 实测 | 训练版实测/预演 |
|---|---|---|---|
| 粗掩码 logit 精修（原 P2） | +0.003 | β=−0.2 无差异 | 57ep ≈ 0（已训） |
| 点位移校准 | +0.008~0.011 | — | 离线可学习性 ≈ 0（目标=GT形状函数） |
| 加点数/换点风格 | — | −0.012~−0.038 | — |
| s0/s1 注入 | 通路贡献 0 | — | 需连带重训消费端 |
| 二轮不确定度挖点（E0） | 未测 | **−0.042** | E1 需训不确定度头+选点器 |

## 机制结论（论文叙事素材）

1. 训练后的模型是一个**闭合的 prompt 驱动分割器**：稀疏 prompt 是命脉（0.73→0.16），
   dense 与高分辨率通路形同虚设；对 prompt 分布的一切 bolt-on 扰动都为负。
2. 所有 oracle 增益（+0.003~+0.011）都要求"以 GT 形状为函数"的信息——
   即修正者必须先解决分割本身；局部高频特征（无论单点采样还是 patch+conv）无法预测。
3. PA-SAM 的机制依赖其"冻结 SAM + 只训 adapter"的前提：decoder 从未与 2P2N 共适应。
   我们的 decoder 已共适应 66 epochs，免费追加任何风格的点都会被当作噪声外推。

## 建议

- E1（训练版 PA-SAM-lite：不确定度头 + Gumbel 选点 + 边界膨胀 GT 监督）是逻辑上的最后一步，
  但期望收益受 +0.008 级别天花板约束，工程量（训练代码 + 数小时训练）不小——
  投入前建议先做其 oracle（"完美二轮选点"上限测量）。
- 更稳健的方向：接受"prompt 路线饱和"作为论文结论；把测量体系
  （oracle 探针 / 通路反事实 / 离线可学习性）作为方法学贡献。

## 本轮产物

- inference/probe_decoder_pathways.py — 通路反事实探针
- inference/probe_e0_iterative.py — 二轮迭代解码探针
- inference/probe_p2_point_learnability.py — 离线可学习性探针（collect/fit/eval）
- inference/oracle_p2_probe.py — oracle 探针（--oracle-cap / --oracle-points / --oracle-points-k）
- logs/eval_p2_ab/ 下全部测量数据
