# NWPU P2-v2 开发协议 100ep：A0 / A1 / A2e 训练结果与结论

日期：2026-09-07。范围：本机完成的三臂完整训练（A0/A1/A2e），各 100 epoch、
每 epoch 全量验证。设计文档见 [p2_v2_redesign_design.md](p2_v2_redesign_design.md) §3；
矩阵背景见 [nwpu_fi_matrix_plan.md](nwpu_fi_matrix_plan.md)。

状态：本文记录已完成的训练事实与机制遥测；**A2（默认包络 R1）在另一台机器训练中，
结果回填后 2×2 网格才可定稿**。A1/A2e 的 best/last 完整 COCO 推理尚未跑（需空 GPU）。

## 1. 公共协议（三臂一致）

- 数据：NWPU VHR-10，520 train / 130 val，全量、无子集（选择集=报告集，val==test 按用户裁决）。
- 训练：100ep 压缩余弦（total_optimizer_steps=6500），warmup 100 步，lr 5e-4，wd 0.05，
  4 GPU × batch1 × accum2（有效 batch 8），AMP，EMA 跟踪但选优用 raw，seed 44。
- 空间契约（fi）：`FINAL_MASK_COORDINATE_MODE=full_image` +
  `FINAL_MASK_LOSS_MODE=roi_balanced_dice`（绑定，非可调超参）。
  三臂运行签名一致：`debug_mask_target_fill=0.0177~0.0179`（legacy roi_local 为 0.5623），
  roi_balanced 三段遥测齐全，推理 `mask_postprocess=full_image_resize`。
- 增强：hflip 0.5 + vflip 0.5 + 多尺度抖动 0.5@value（896–1152 五档，先抖动再定回 1024）；
  rot90 本轮裁决不开（设计文档 §0 audit P2-1），三臂统一。
- 入口：`scripts/ablations/vhr10_p2v2_dev.sh <arm>`（单一参数化入口，防漂移；
  六臂构建已由 `scripts/smoke/verify_p2v2_arms.py` 全部断言通过）。

## 2. 三臂定义（相对 A0 的唯一差异轴）

| 臂 | P2 | loss 模式 | β | Δ_max（cap） | 其余 |
|---|---|---|---|---|---|
| A0 | off | — | — | — | PBM（points_box_dense），D5-B 配方：α0.5 可学习 global_sigmoid 门、PE mask_downscaling 解冻（4684 参数，lr_mult 0.1） |
| A1 | on | boundary（原版 P2） | 0.10 | 0.50（±0.05） | = A0 |
| A2e | on | correction_keep（R1） | 0.30 | 1.00（±0.30，sat_thr 0.297） | margin 1.0，λ=0.1，w_aux 0.05 |

架构指纹（日志实测）：A0 `9489ea70…`，A1 `efa0a623…`，A2e `775f2a00…`
（A2 默认臂 `0a4335bc…`，另一台机器）。臂间差异已验证恰好落在设计轴上
（p→pb: mode；pb→a0: mode+md；a0→a1: enabled；a1→a2: loss/margin/keep；a2→a2e: β+Δ）。

## 3. 主结果（segm/mAP，每 epoch 全量 130 图验证）

| 臂 | best（epoch） | E100 | 平台期 E91–100（均值±std） | best−A0 | 平台−A0 | 训练时长（日志跨度） |
|---|---|---|---|---|---|---|
| **A0** | **0.6669**（E36） | 0.6464 | **0.6467**±0.0011 | — | — | 3h55m |
| A1 | 0.6530（E32） | 0.6425 | 0.6433±0.0005 | **−0.0139** | −0.0034 | 4h22m（OOM 前次尝试已隔离，不计） |
| A2e | 0.6590（E32） | 0.6367 | 0.6369±0.0014 | **−0.0079** | −0.0098 | 4h27m |

Top-5（选择集口径，供噪声带参考）：
- A0：E36 0.6669 / E41 0.6587 / E74 0.6559 / E45 0.6549 / E67 0.6543
- A1：E32 0.6530 / E66 0.6528 / E67 0.6496 / E74 0.6492 / E52 0.6485
- A2e：E32 0.6590 / E67 0.6546 / E33 0.6540 / E52 0.6531 / E74 0.6528

A0 锚点对照：best 0.6669 已超 legacy 400ep Full 0.6476 与 ft200 0.6617（约 1/4~1/6 预算）；
三臂 best 都出现在 E32–E41 早段，平台期差距（A1 −0.003 / A2e −0.010）比 best 差距更能说明问题，
因为 130 图选择集上 best 单点含选择噪声（A0 自身 E36 与 E41 差 0.008）。

## 4. A0 完整 COCO 指标（新契约推理，best 与 last）

| 口径 | segm/mAP | mAP_50 | mAP_75 | mAP_m | mAP_l | AR@100 | bbox/mAP |
|---|---|---|---|---|---|---|---|
| best（E36） | 0.6669 | 0.9487 | 0.7133 | 0.5429 | 0.7240 | 0.7078 | 0.6764 |
| last（E100） | 0.6464 | 0.9170 | 0.6907 | 0.4748 | 0.7144 | 0.6832 | **0.7089** |

bbox 与 segm 双时间尺度：segm 平台在 E60–80（E91–100 均值 0.6467），
bbox 到 E100 仍在涨（E100 0.7089 > best 轮 0.6764）。100ep 下 bbox 欠收敛，
这直接影响"最终矩阵用多长 schedule"的决策（nwpu_fi_matrix_plan D1 两层协议待定稿）。

## 5. A2e 机制遥测（R1 + 扩大包络）

训练全程 `[DEBUG-P2BR]` 分 epoch 汇总（饱和阈值 0.297 = 0.99×β×Δ_max）：

| epoch | 支持内饱和率 | 有效 ROI | 备注 |
|---|---|---|---|
| 1 | 0% | 61 | 冷启动 |
| 10 | 28.3% | 151 | |
| 20 | **49.2%**（峰值） | 154 | |
| 32 | 38.0% | 174 | best 轮 |
| 50 | 38.5% | 99 | |
| 80 | 29.5% | 171 | |
| 100 | 26.7% | 184 | 收敛态 |

- 饱和率走势是"中程冲高后回落"：E20 峰值 49%，末段稳定在 23–27%，
  **修正幅度没有钉死在新 cap 上**——扩大 3 倍的包络（±0.05→±0.30）不是后期的绑定约束。
- P2 损失绝对量全程近似恒定（≈0.055），占总损失比例从 5.7%（E5）升到 22.8%（E100），
  原因是其他损失衰减而 P2 不衰减；R1 两部分中 corrective 主导（≈0.054），
  keep 部分始终 ≈0.001（margin=1.0 下 keep 区几乎不产生违规）。
- 中程（E41）观察记录：当时饱和率 43%，且 refinement 增益为负（dice −0.0021）——
  更大的修正没有兑换成更好的 mask。

## 6. 结论

1. **A1：原版 P2（boundary 监督）在新消费端下无增益**——best −0.0139、平台 −0.0034，
   与 WHU D 系列方向一致，NWPU 全量 100ep 复现。
2. **A2e：R1 监督 + 3 倍包络仍无增益**——best −0.0079、平台 −0.0098。
   叠加遥测（饱和率不钉死、P2 损失不衰减但兑换不成 mAP、refinement 增益为负），
   **"包络/参数是瓶颈"的假说被否定：给更大的修正预算，模型学会了用，
   但用出来的修正不改善最终 mask**。
3. **P2-as-logit-residual-correction 这一角色基本判死**：2×2 网格
   （监督方式 boundary/R1 × cap 0.05/0.30）本机三格（A0/A1/A2e）全部为负，
   且负向不随监督方式和包络大小翻转。等另一台机器的 A2（默认包络）回填最后一格定稿。
4. **A0 是当前最优配方**（PBM + D5-B：α0.5 可学习门 + PE mask_downscaling 解冻 + P2 off），
   100ep 即超 legacy 长训锚点。后续候选方向：R3（从 P2 特征直接渲染 dense 画布，
   走 canvas 通道而非 logit 残差）——已有 WHU 冻结诊断 GT-canvas@α1.0 +0.0154 的通道价值证据，
   NWPU 侧前置测试（canvas 准确率探针，blend r=0/0.5/1.0）已排队待 GPU 空闲执行。
5. 下一个矩阵读数依赖 P/PB 基线臂（box 增益 = PB−P，dense 包增益 = A0−PB），
   已具备（入口/构建断言通过），按用户指示等空闲再挂。

## 7. 工件索引

| 项 | 路径 |
|---|---|
| A0 ckpt | `ablations/vhr10_p2v2_dev100_a0_pbm_d5b_tr1.0_va1.0/`（best_model.pth=E36, last_model_epoch100.pth） |
| A1 ckpt | `ablations/vhr10_p2v2_dev100_a1_p2v1_tr1.0_va1.0/`（best=E32, last=E100） |
| A2e ckpt | `ablations/vhr10_p2v2_dev100_a2e_r1_esc030_tr1.0_va1.0/`（best=E32, last=E100） |
| 训练日志 | `logs/ablations/vhr10_p2v2_dev100_{a0_pbm_d5b,a1_p2v1,a2e_r1_esc030}_tr1.0_va1.0_*.log` |
| A0 推理 | 上述 A0 目录 `inference_{best,last}_vhr10/run_manifest.json` |
| 隔离工件 | `*_a1_p2v1_crashed_oom_partial`（A1 OOM 前次尝试）、`*_a2_r1_*_stopped_for_a2e_partial`（本地 A2 让位 A2e） |
| 臂构建断言 | `scripts/smoke/verify_p2v2_arms.py`（六臂全过，指纹与真实 run 逐位一致） |

## 8. 判读注意（caveats）

- best 单点含 130 图选择噪声；臂间比较以平台期（E91–100 均值）为主、best 为辅，两口径本文同报。
- A2e 的"饱和率"是支持区内逐像素 |delta|≥0.297 的池化比例（分子/分母分别求和后再除），
  不是 ROI 均值的均值。
- A1/A2e 尚无 best/last 完整 COCO 推理；本文 A1/A2e 只有训练期验证指标，
  与 A0 第 4 节的推理口径（同一契约、同 130 图）可比性待推理补齐后确认。
- 本系列保存 `last_model_epoch100.pth`（SAVE_LAST_MODEL=1），不存在 D 系列 E15 未存档问题。
- 单次训练、单 seed（44）；小差距的严格置信需训练重复，本文结论以"多臂同向 +
  机制遥测互证"支撑，不依赖单臂单点差。
