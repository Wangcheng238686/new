# A0 P2 带外错误定位可读性门

状态：已完成（2026-09-09）。它不是训练模型，也不是 GT Oracle mAP。

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

## 2026-09-09 完整结果与裁决

完整运行已在与 PB 共存的 GPU0 上完成：A0 epoch36 strict load 的 missing/unexpected 均为
0；使用 NWPU train-520 拟合 77,465 个均衡采样候选像素（原始 positive fraction 0.3386），
再对独立 NWPU val-130 完成 1000 次 image-bootstrap。工件：
`/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/diagnostics/a0_p2_error_readability_full_v1/`。

| selector | error recall | error precision | fixed-10% signed-flip coarse IoU |
|---|---:|---:|---:|
| raw-only | 0.41874 | 0.29405 | 0.80075 |
| aligned P2 | 0.40819 | 0.28664 | 0.77252 |
| ROI-permuted P2 | 0.38034 | 0.26709 | 0.76798 |

| 对比 | recall 95% CI | precision 95% CI | coarse-IoU 95% CI |
|---|---:|---:|---:|
| aligned P2 − raw | [−0.01961, +0.00143] | [−0.01587, +0.00086] | [−0.02984, −0.02668] |
| aligned P2 − permuted | [+0.02093, +0.03737] | [+0.01515, +0.02430] | [+0.00319, +0.00632] |

**预注册 gate = fail。** 它相对 raw-only 的 recall/precision 均未越过零。因此，当前证据
不支持“P2 能提供超过 raw coarse 的、足以安全写入带外
区域的错误定位信息”。更不能把 support Oracle 中 GT direction 的增益归因给可训练 P2。

审计发现 v1 的 `raw` coarse-IoU 同时累计了未修正 baseline 与 raw-selector 修正结果，而其余
selector 只累计修正结果；故上表所有 coarse-IoU 数字与 IoU CI 均**作废、不得引用**。这一 bug
不改变 gate fail（recall/precision 已失败），但未来任何 IoU 结论必须使用修复后代码重跑。

这也不是 P2 特征永远不可用的证明：该门只否定了**当前 512-channel linear readout、当前
accepted-S4 complement、10% 写入预算**下的“直接替换 hard reject”路线。若继续，应把研究问题
收缩为：为什么 P2 对齐相较打乱可辨、却不及 raw-only——例如严格仅在 raw 低置信候选上评估、
或以训练集固定的正则化/降维 readout 控制 512 维过拟合。它们必须作为新的预注册诊断，不能
直接启动 P2-conditioned support 训练。

## 2026-09-09：SAM2 native image-embedding 冻结探针

在完全相同的 A0、accepted-S4 complement、train-520 拟合 77,465 样本、val-130 和 1000
image-bootstrap 协议下，`--feature-source=sam_image` 读取了 PromptEncoder 前、custom PAFPN 前
的官方 stride-16 `image_embeddings`。结果如下；表中 `aligned` 为 raw+正确 ROI 对齐 native
image embedding，`permuted` 为同一 readout 下 ROI 特征置换。

| selector | recall | precision | fixed-10% signed-flip coarse IoU |
|---|---:|---:|---:|
| raw-only | 0.41874 | 0.29405 | 作废，待重跑 |
| aligned native image | 0.40577 | 0.28495 | 作废，待重跑 |
| permuted native image | 0.40592 | 0.28505 | 作废，待重跑 |

aligned−raw 的 recall/precision CI 分别为 `[-0.01607,-0.00970]`、
`[-0.01156,-0.00626]`；旧 coarse-IoU CI 作废。aligned−permuted 的旧 CI 也不再用于解释。
因此该门 **fail**，且比 detector-P2 的结果更强地说明：这张 stride-16 native semantic map
在当前 ROI-local coarse 带外错误定位任务中没有可用的 recall/precision 增益。不能使用它构造 pre-PE
prompt adapter。唯一尚未测试、且分辨率上仍有机制依据的原生源是 backbone-FPN high-res s0/s1；
若该臂也失败，应停止“另换视觉特征源来改善 prompt 挖掘”的路线。

## 2026-09-09：SAM2 native high-res s0/s1 冻结探针（v2）

在修复 raw-selector IoU 双计数、并将负对照升级为每 ROI 的二维空间 roll 后，完成
`--feature-source=sam_highres` 的全新 v2 运行。运行时 provenance guard 确认来源是
`sam2_backbone_fpn_pre_RSSAM2PAFPN`：image `256×64×64`、s0 `256×256×256`、s1
`256×128×128`。独立审计复核了 520 train / 130 val 无 ID 重叠、130 条唯一 image records，
以及三臂一致的 writes/errors（234,112 / 164,402）。

| selector | recall | precision | fixed-10% signed-flip coarse IoU |
|---|---:|---:|---:|
| raw-only | 0.41874 | 0.29405 | 0.77461 |
| aligned native s0+s1 | 0.41069 | 0.28840 | 0.77359 |
| spatially rolled s0+s1 | 0.37792 | 0.26539 | 0.76790 |

aligned high-res − raw 的 recall/precision/coarse-IoU CI 为
`[-0.01534,+0.00180]`、`[-0.01118,+0.00114]`、`[-0.00252,+0.00067]`，均未通过；
aligned high-res − roll 三项 CI 均为正。**gate = fail。** s0/s1 包含可读的空间信息，但
它仍不能超过 raw coarse 定位带外错误的能力。停止“native s0/s1 直接供给 pre-PromptEncoder
adaptive support”的路线，不训练对应模块。此停止结论仅覆盖冻结 A0、线性 readout、accepted-S4
complement 与 10% 固定预算，不能外推为所有非线性 SAM2 high-res 模块都无效。
