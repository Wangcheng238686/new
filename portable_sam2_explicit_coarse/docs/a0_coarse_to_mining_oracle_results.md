# A0 coarse-to-mining Oracle：正式结果与裁决

日期：2026-09-08。用途：冻结 A0 下的**诊断证据台账**，用于复查后续 P2 方向；不是训练
实验，也不是可报告的方法精度。

## 1. 可复现工件

- 代码：`inference/probes/coarse_to_mining_oracle_probe.py`
- checkpoint：
  `/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/vhr10_p2v2_dev100_a0_pbm_d5b_tr1.0_va1.0/best_model_epoch36.pth`
- canonical A0 推理：
  `/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/vhr10_p2v2_dev100_a0_pbm_d5b_tr1.0_va1.0/inference_best_vhr10/`
- 正式结果：
  `/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/diagnostics/a0_coarse_to_mining_oracle_full_v2/`
- bootstrap JSON：上述目录下的 `bootstrap500/`。

数据为 NWPU VHR-10 validation 全 130 图、10 类。模型完全冻结；GT 仅将同类且
proposal-box IoU≥0.5 的 ROI coarse 写为 signed `±8` logits。未匹配 proposal 保持 raw。

## 2. 四格与不变量

| 格 | 点挖掘输入 | dense canvas 来源 |
|---|---|---|
| raw | raw coarse | raw coarse |
| points | GT coarse | raw coarse |
| dense | raw coarse | GT coarse，经生产 `_shape_prior_to_prompt_mask` |
| both | GT coarse | GT coarse，经生产变换 |

已通过的硬不变量：

- unhooked A0 与 canonical 的 `gt_records/dt_records/images` 及全套 segm 指标精确一致；
- raw hook 与 unhooked 完整输出 SHA256 一致；
- 四格 `image_id,bboxes,scores,labels` 的 detector SHA256 完全相同；
- 每个 GT 干预格替换 728 个匹配 ROI、2,981,888 个 coarse 像素；190 个 unmatched ROI 未改；
- 每格均为 130 图、743 GT 实例、918 detections；导出 manifest 含十类数据契约，
  `resamples=0` bootstrap 精确复现正式 mAP。

## 3. 正式 mAP

| 格 | segm/mAP | 相对 raw |
|---|---:|---:|
| raw | 0.666929 | — |
| points | 0.667831 | +0.000902 |
| dense | 0.675280 | +0.008351 |
| both | 0.676067 | +0.009138 |

## 4. 500 次 paired image-bootstrap

同一 130 图有放回重采样，种子 44；CI 是 treatment−baseline。

| 比较 | 点估计 | 95% CI | `P(Δ>0)` | 判定 |
|---|---:|---:|---:|---|
| points − raw | +0.000902 | [−0.000882, +0.002867] | 0.850 | 不支持点路径独立收益 |
| dense − raw | +0.008351 | **[+0.005167, +0.011141]** | 1.000 | dense 路径有明确可达空间 |
| both − dense | +0.000787 | [−0.001625, +0.003721] | 0.736 | 点路径没有 dense 之外的附加证据 |

## 5. 设计裁决

唯一受本次证据支持的下一步主线是：

```text
P2 / 图像证据 → 更准 ROI coarse → 既有 dense canvas → 冻结 PromptEncoder / MaskDecoder
```

不再优先投入边缘正负点、Gaussian 选点或额外 sparse token。原因不是它们理论上不可能有效，
而是本 A0 契约下即使给出完美 coarse，也没有得到统计上可分离的独立增益。

后续 P2 候选必须依次通过两个门：

1. 在 matched ROI 上相对 raw 证明 refined coarse 的 content/IoU 与 bbox-support canvas
   accuracy 改善；
2. 在 A2 vs A0 的完整 validation paired bootstrap 中得到正向 CI。

本 Oracle 仅给出 downstream 上限；不能将 dense 的 +0.008351 归因于 P2、R1 或任何可训练模块。
