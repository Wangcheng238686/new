# A0 P2 带外错误定位可读性门

状态：已实现，待在可用 GPU 上运行。它不是训练模型，也不是 GT Oracle mAP。

## 问题

支持域 Oracle 已证明：若已知纠错方向，撤去 accepted-S4 的 coverage hard reject 可以释放
显著 dense 路径上限。但 GT 在那个实验中提供了最难的信息——哪些像素错、应向哪个方向改。
本门只问一个更小且必要的问题：**冻结 A0 的真实 P2 特征，是否能在没有 GT 的前提下定位
accepted-S4 之外的 raw coarse 错误？**

## 协议

- 底座：P2-off A0 `best_model_epoch36.pth`，所有 A0 参数、proposal、box、类别、分数及
  final mask forward 均冻结且不改写。
- 空间域：从 A2 epoch63 保存配置构造 accepted S4；候选像素是该 support **之外**、但仍在
  同类且 proposal IoU≥0.5 匹配的 ROI-local 64×64 coarse 网格内。
- 标签：GT 只在 train-520 中生成 `raw_binary != target` 的 error 标签，及在 val-130 作盲评
  指标；不进入模型 forward、feature 或 selector。
- 读出器：固定随机种子、均衡采样的线性 logistic readout。`raw` 只读 `[raw_logit,|raw_logit|]`；
  `p2` 读同一 raw 特征加上以生产 P2 空间尺度（0.25）、aligned RoIAlign 对齐的冻结 P2
  通道。不能将 P2 projection/refiner 参数从 A2 搬入，以免混入已经训练失败的残差头。
- 负对照：验证时使用同一个训练好的 P2 readout，但把每个 ROI 的 P2 map 循环置换到另一 ROI；
  一张图只有一个 ROI 时以半幅空间平移替代，确保对照绝不保留其正确空间对齐。
- 部署形态指标：每 ROI 在候选域固定选择 top 10% error score。只为评估，把被选像素的 raw
  二值符号翻转；这不是部署时使用 GT direction，而是检验“预测为 error 后按 raw 的反号纠正”
  是否有足够低的 false-write 风险。报告 error recall、precision 和得到的 coarse IoU。

## 预注册裁决

在 130 张验证图上做 1000 次 paired image-bootstrap（seed44）。只有当下面六个 CI 下界均
严格大于 0 才通过：

1. aligned-P2 减 raw 的 recall、precision、coarse-IoU；
2. aligned-P2 减 permuted-P2 的 recall、precision、coarse-IoU。

若失败，不能把 support Oracle 的 GT 上限包装成 P2 模块收益，也不得以该方向启动热训练；应
回到 P2 特征本身的判别性/监督设计。若通过，才可实现“P2-conditioned adaptive correction
support”：连续、有预算、取代 whole-ROI hard reject，但 raw coarse 仍是唯一语义源。

## 命令

```bash
CUDA_VISIBLE_DEVICES=0 "$PYTHON" inference/probes/p2_error_readability_probe.py \
  --checkpoint /data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/vhr10_p2v2_dev100_a0_pbm_d5b_tr1.0_va1.0/best_model_epoch36.pth \
  --support-reference-checkpoint /data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/vhr10_p2v2_dev100_a2_r1_tr1.0_va1.0/best_model_epoch63.pth \
  --train-ann-file coco_split/NWPU_instances_train.json \
  --train-image-subdir 'positive image set' \
  --device cuda:0 \
  --output-dir /data/wangcheng/checkpoint/portable_sam2_explicit_coarse/diagnostics/a0_p2_error_readability_full_v1
```

输出 `summary.json`（预注册 gate 与 CI）和 `image_records.json`（按图聚合的计数）。仅在完整
train-520/val-130 运行、严格加载成功且没有错误退出后，才将实际数值补入本文档。
