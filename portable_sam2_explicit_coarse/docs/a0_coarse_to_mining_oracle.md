# A0 冻结 coarse-to-mining Oracle

这是 A0（P2-off，points+box+dense）最佳权重上的推理诊断，不训练、不写入模型
参数、也不改变 architecture contract。它直接检验原设计的因果链：若一个 proposal 的
ROI-local coarse 更准确，现有的点挖掘和 dense canvas 两个消费者能否带来最终 mask 收益。

四格均使用同一冻结 A0、同一完整 NWPU validation、同类别且 box IoU ≥ 0.5 的 proposal
→GT 匹配；未匹配 proposal 永远保留 raw 输出：

| 格 | `ShapePointMiner` 输入 | dense canvas 的 coarse 来源 |
|---|---|---|
| `raw` | raw coarse | raw coarse |
| `points` | GT signed coarse | raw coarse |
| `dense` | raw coarse | GT signed coarse 经生产 `_shape_prior_to_prompt_mask` |
| `both` | GT signed coarse | GT signed coarse 经生产变换 |

GT signed coarse 只在 Oracle 内生成：先按训练同构的 proposal crop → nearest resize 映射到
粗网格，再写为 `±8` logits。它不参与 proposal 选择、检测、不匹配行，或任何训练/验证期
模型输入。`dense`/`both` 必须调用生产 `_shape_prior_to_prompt_mask`，所以仍包含原 transform、
bbox paste、outside fill 与 dense gate，而不是构造新的 canvas 分支。

运行前先做无 hook 的标准推理；完整运行还要求它和 A0 生产 `dt_records.json`、mAP 精确相等。
随后每格均断言 `image_id,bboxes,scores,labels` 的 SHA256 不变；`raw` hook 路径还必须与
无 hook 完整 mask 输出逐位相等。每格输出 GT/DT/images/manifest，供 paired bootstrap 使用。

```bash
CUDA_VISIBLE_DEVICES=3 "$PYTHON" inference/probes/coarse_to_mining_oracle_probe.py \
  --checkpoint /data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/vhr10_p2v2_dev100_a0_pbm_d5b_tr1.0_va1.0/best_model.pth \
  --device cuda:0 \
  --canonical-dir /data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/vhr10_p2v2_dev100_a0_pbm_d5b_tr1.0_va1.0/inference_best_vhr10 \
  --output-dir /data/wangcheng/checkpoint/portable_sam2_explicit_coarse/diagnostics/a0_coarse_to_mining_oracle_full
```

解释纪律：`points` 正向才证明“更好的 coarse 可经点挖掘被消费”；`dense` 正向才证明
“更好的 coarse 可经现有 dense 通道被消费”；`both` 是两者联合上限，不能将交互收益归给
单一路径。任何正向 Oracle 都只是后续 P2/coarse 候选的必要条件，不是方法收益声明。

## 2026-09-08 GPU3 全量结果（冻结 A0）

正式可用工件：
`/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/diagnostics/a0_coarse_to_mining_oracle_full_v2/summary.json`。

- A0 `best_model_epoch36.pth` 在 130 图 NWPU validation 的 unhooked standard 与既有
  `inference_best_vhr10` 的 `gt_records/dt_records/images`、所有 segm 指标均精确一致；
  `canonical_records_exact=true`。
- raw hook 与 unhooked 完整输出 SHA256 精确一致；四格的 detector SHA256 全相同，故
  所有差异只来自 mask 分支，未混入 proposal、类别、bbox 或检测分数变化。
- 每个 GT 干预格实际替换 728 个同类 IoU≥0.5 匹配 proposal 的 2,981,888 个 coarse
  像素；190 个未匹配 proposal 保持 raw。

| 格 | segm/mAP | 相对 raw |
|---|---:|---:|
| raw | 0.666929 | — |
| points | 0.667831 | +0.000902 |
| dense | 0.675280 | +0.008351 |
| both | 0.676067 | +0.009138 |

当前可得的机制结论是：在这一个冻结 A0 上，**完美 coarse 的可达收益主要位于既有 dense
消费者，而非 2P2N 点挖掘**；`both-dense=+0.000787` 也表明点路径提供的附加上限很小。
这些是 Oracle 上限，不是 P2、R1 或任何新训练模块的结果。首次输出漏写 manifest 的
`dataset.num_classes=10`，使 bootstrap 工具错误回退为单类；补齐后以 `_full_v2` 重跑完整
四格。其 `resamples=0` 十类重建已逐项精确复现 raw/points/dense/both 的上述 mAP，且使用
`manifest_processed_image_ids` 验证相同的 130 图执行覆盖；其大样本 bootstrap 结论见下节。

## 2026-09-08：500 次 paired image-bootstrap 裁决

使用 `_full_v2/bootstrap500/` 的完整十类、同 130 图、同 `processed_image_ids` 工件；每组
500 次有放回 image resample，种子 44。CI 是 treatment−baseline：

| 对比 | 点估计 | bootstrap 均值 | 95% CI | `P(Δ>0)` | 裁决 |
|---|---:|---:|---:|---:|---|
| points − raw | +0.000902 | +0.001004 | [−0.000882, +0.002867] | 0.850 | 不足以支持点挖掘路径 |
| dense − raw | +0.008351 | +0.007956 | **[+0.005167, +0.011141]** | 1.000 | dense consumer 存在明确可达空间 |
| both − dense | +0.000787 | +0.000931 | [−0.001625, +0.003721] | 0.736 | 点路径没有可分离附加证据 |

### 下一步方向（已收缩）

结论支撑的是原始因果链中的 **P2 / 图像证据 → 更准 ROI coarse → 生产 dense canvas →
冻结 PromptEncoder / MaskDecoder**，而不是“再设计选点”或“直接另造 canvas 信息源”。

1. 后续 P2 候选必须保持 `ShapePriorInjector` 的 coarse 输出作为唯一语义源；dense 仍经
   生产 `_shape_prior_to_prompt_mask`、既有 box support 与 gate，不在验证期以 GT 替换 canvas。
2. P2 的停止/进入训练门从“有非零 residual、边界统计好看”改为两层：先在 matched ROI 上
   证明 refined coarse 相对 raw 的 content/IoU（尤其 bbox-support 内的 canvas accuracy）提升；
   再证明 A2 相对 A0 的 paired full-val segm 改善。只改善 point telemetry 不构成继续投入理由。
3. 不再为 `ShapePointMiner` 增加边缘正负点、Gaussian 选点或新 sparse token；它们即使获得
   完美 coarse，在本 A0 契约下也没有统计上可分离的独立上限。
4. 该 Oracle 只证明 downstream dense 通路值得优化；它**不证明**现有 P2/R1 已能产生这样的
   coarse 改善，也不允许将 +0.00835 写作 P2 或训练模型收益。
