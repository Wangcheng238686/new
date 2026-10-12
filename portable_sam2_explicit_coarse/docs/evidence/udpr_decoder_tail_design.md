# UDPR：不确定性感知解码尾部点细化器

> 本文档保留 A3/UDPR-v1 的历史实施契约。针对其“低置信但正确像素也被无条件写回”
> 的诊断修正，A4 的论文级模块定义、损失、接线和停止条件见
> [`dcr_dual_confidence_refinement_design.md`](dcr_dual_confidence_refinement_design.md)。

## 定位

UDPR（Uncertainty-Guided Decoder-tail Point Refiner）是 A0 的独立输出端模块，位于
SAM2 MaskDecoder 的原生 mask logits 后、最终 mask loss/后处理前。它不新增 prompt，
不读取 coarse canvas/P2，不重跑 decoder，也不改变 detector 的框、类别或排序。

输入为每个 RoI 的 native logit `z`、SAM2 `output_upscaling` 特征 `U` 和已选 mask
token `t`。selector 固定为 native raster 网格上 `TopK smallest |z|`；相同 selector
用于训练和推理，且不读 GT。对选中点 `p`，MLP 消费 `[U(p), t, z(p), x, y, |z(p)|]`
并输出受 `delta_logit_max*tanh` 限制的残差，仅 scatter 回这些点。

首版是 `K=64`、hidden=128、class-agnostic、full-image 输出的单点写回版本。末层零
初始化，因此刚接入严格恒等 A0；没有额外零 gate，以免首步梯度为零导致模块死亡。

## 训练与初始化契约

- 机制筛选可用 NWPU A0 `best_model_epoch36.pth` 使用 `--init-from --allow-cross-arch-init`
  热启动；loader 强制检查 A0→UDPR 时 missing keys **只能**是
  `roi_head.mask_head.decoder_tail_refiner.*`。该热启动不属于 A 系列可比主结果。
- `--train-decoder-tail-only` 会冻结完整 A0（含检测器、PromptEncoder、dense gate、
  decoder）；仅 UDPR 参数可训练，optimizer 必须只出现 `decoder_tail` group。
- A 系列 `a3` 是 PBM/A0 的**单阶段**对照：不设置 `INIT_FROM`、不传
  `--train-decoder-tail-only`，仅开启零初始化的 UDPR-K64；A0 的其他训练参数、fi
  契约、数据、seed、100ep cosine 均保持不变。runner 以
  `DECODER_TAIL_TRAIN_ONLY=1` 明确区分上述两种模式。
- GT 只来自训练正 RoI 的现有 `get_full_image_targets`，作为 selected-point BCE target；
  推理期不做 IoU 匹配、更不使用 GT。point BCE 按跨 DDP 的正/负 selected 点计数分别
  归一后等权组合，避免背景点数主导。
- 正式可报告实验应比较 `A0`、`A0+UDPR-K64`（必要时才加 K256）；Tail 不应被记为
  P2/dense 的增益。现有 Tail Oracle 是 GT 纠错上限，不是训练效果承诺。

## 验收

1. UDPR disabled 与历史 A0 前向逐位一致；启用但新模块零初始化时输出同样恒等。
2. K=0/zero-init smoke、检测框/类别/score hash 与 A0 相同。
3. `TAIL/*` 日志显示 selector、残差和 point BCE 有限；**tail-only 热启动**的首个非零
   LR update 后仅 decoder_tail 参数改变。A3 单阶段臂则应按 A0 的正常参数组更新，不能
   错把完整网络更新判为接线泄漏。
4. 训练后以完整 validation records 对 A0 做 paired bootstrap；不可只报单点 mAP。

## 实施验收（2026-09-07）

- `scripts/smoke/test_decoder_tail_refiner.py` 通过：zero-init 恒等、稳定 tie-break、末层可获
  非零 BCE 梯度。
- `scripts/smoke/check_udpr_a0_equivalence.py` 在 A0 NWPU validation 首 batch 通过：A0 strict
  load 后与 zero-init UDPR 的完整 prediction SHA256 相同
  (`350e6da8662024c5ac2f4bdaae6d2f576ac34f553f26029dac5adc920dc6a961`)。
- 4-GPU 一 batch 热启动 smoke 通过，日志在仓库外
  `/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/smoke/udpr_k64_ddp_one_step/run.log`：
  A0 loaded=665，shape mismatch/unexpected/migrated 均为 0，只缺 6 个新 tail keys；仅
  `decoder_tail` 38,217 参数可训练，`TAIL/selected_count=4096`、point BCE 有限，首次
  non-zero-LR update 后 DDP max deviation 为 0。该 smoke 不报告精度。
