# NWPU P2-v2 dev100：A0 / A2 / A3 完整推理记录

日期：2026-09-08。本文是 NWPU VHR-10 开发协议的**本机四卡评测记录**，用于将
训练期验证与固定 checkpoint 的完整 COCO 推理解耦。它不是多 seed 统计结论，也不替代
正式 600ep 论文协议。

相关训练期过程、A1/A2e 的既有结果见
[`p2v2_dev100_a0_a1_a2e_results.md`](p2v2_dev100_a0_a1_a2e_results.md)；UDPR 的模块
定义与单变量约束见 [`udpr_decoder_tail_design.md`](udpr_decoder_tail_design.md)。

## 1. 评测口径与执行环境

- 数据/划分：NWPU VHR-10 validation，130 图，checkpoint 内嵌的 10 类数据集契约；
  `maxDet=100`。
- 入口：`scripts/ablations/vhr10_p2v2_eval.sh <a0|a2|a3> <best|last>`；均导出
  `metrics.json`、`dt_records.json`、`gt_records.json`、`images.json`、`run_manifest.json`
  以供后续图像级配对 bootstrap。
- **本机四卡协议（操作者调度记录）**：本次汇总按本机 4 张 CUDA GPU 的并行协议执行；
  A3 的 best/last 调度至物理 GPU0/GPU1，A2 的 best/last 调度至物理 GPU2/GPU3，A0
  已有同入口、同数据契约的完整输出。四份 manifest 均确认 `processed_images=130` 且
  `processed_image_ids` 长度为 130。**证据边界**：现存 manifest 只持久化逻辑设备
  `cuda:0`，未持久化 hostname 或 `CUDA_VISIBLE_DEVICES`；因此上述物理卡映射是本次
  执行调度记录，不能由现存评测工件单独复现。后续评测应在 manifest 追加这些字段。
- 权重口径：`best` 是训练时按验证 `segm/mAP` 选择的链接权重，`last` 固定为 epoch 100。
  它们回答不同问题，禁止将同一训练轨迹的 best 与 last 当成独立重复实验。
- 严格加载：A2/A3 checkpoint 均通过保存的 architecture ID 与 SHA256 fingerprint 精确
  匹配后才构建模型。为兼容保存时漏入 `model_config`、但仍存在于同一 checkpoint
  `mask_head_config` 的模块配置，推理端仅复原该配置并再次逐项比对 contract；不匹配即失败。

## 2. 结果

| 实验 | 模块差异（相对 A0） | best 权重 | best segm/mAP | best bbox/mAP | last 权重 | last segm/mAP | last bbox/mAP |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| A0 | PBM / D5-B，P2 off | E36 | 0.6669 | 0.6764 | E100 | 0.6464 | **0.7089** |
| A2 | A0 + P2-R1 | E63 | 0.6517 | **0.7005** | E100 | 0.6407 | 0.7015 |
| A3 | A0 + 零初始化 UDPR-K64；P2 off | E41 | **0.6953** | 0.6773 | E100 | **0.6594** | 0.6926 |

以 A0 为共同分母的 segmentation 差值：

| 对比 | best Δ segm/mAP | last Δ segm/mAP |
| --- | ---: | ---: |
| A2 − A0 | −0.0152 | −0.0057 |
| A3 − A0 | **+0.0283** | **+0.0129** |
| A3 − A2 | **+0.0436** | **+0.0187** |

补充 mask 指标：A0/A2/A3 的 best `segm/AR@100` 分别为 0.7078 / 0.6894 / **0.7357**；
last 分别为 0.6832 / 0.6799 / **0.6984**。该方向与 UDPR 位于 mask decoder tail、直接
改变 mask logits 的机制相符；不应据此宣称 bbox 提升。

## 3. 当前可说与不可说的结论

1. 在本机、相同 NWPU validation 与同一 seed 的 dev100 训练轨迹上，A2/P2-R1 未给出
   segmentation 正向证据；A3/UDPR 在 best 与 last 两种预先定义的权重口径下均高于 A0。
2. A3 best 位于 E41、明显高于其 E100 last，因此峰值与末轮收益必须分开汇报。last 的
   +0.0129 是更保守的训练后段比较，仍不是收敛性或跨 seed 的证明。
3. 下一步必须运行 A0↔A3 的图像级配对 bootstrap CI（best 与 last 各一组）。其 CI 仅量化
   130 张验证图的采样不确定性，不能替代独立训练 seed 复跑；在此之前，A3 只能称为
   **强开发候选**，不能作为“稳定有效”的论文主张。

## 4. 工件位置

| 实验 | 根目录 |
| --- | --- |
| A0 | `/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/vhr10_p2v2_dev100_a0_pbm_d5b_tr1.0_va1.0/` |
| A2 | `/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/vhr10_p2v2_dev100_a2_r1_tr1.0_va1.0/` |
| A3 | `/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/vhr10_p2v2_dev100_a3_pbm_udprk64_tr1.0_va1.0/` |

每个根目录下的 `inference_best_vhr10/` 与 `inference_last_vhr10/` 为本表对应的完整评测输出。
