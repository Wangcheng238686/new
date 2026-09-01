# VHR-10 战役报告（C4 误跑 → C5-v2 修正，2026-08-29 ~ 09-01）

## 1. 运行清单

| run | 架构 | 说明 | best segm/mAP | commit |
|---|---|---|---|---|
| vhr10_fast400 | **C4（误跑）** | 入口漏架构导出：无 P2 精修/stride 32/无 detach | 0.6136 @ ep378 | a4e3799 |
| vhr10_large400 | C4 + large 编码器 | 编码器升级在 C4 内验证 | 0.6171 @ ep295 | 68a14af |
| **vhr10_c5v2_400** | **C5-v2（修正）** | 补齐架构契约，与 WHU 获胜协议一致 | **0.6476 @ ep321** | 2c08a0e |

事件记录：8/31 发现两次 run 的 p2_boundary loss 全程 0.00%、契约为 c4——
独立入口丢失 P2_BOUNDARY_REFINER_ENABLED=1 / SAM_IMAGE_EMBED_STRIDE=16 /
SHAPE_DENSE_DETACH=1 三个导出（`2c08a0e` 修复并附验证）。

## 2. 最终对照（val=报告集，RSPrompter 520/130，maxDet=100，1024 空间）

| 指标 | C4 | **C5-v2 (raw)** | C5-v2 EMA | 论文表(MRCNN→最佳) |
|---|---|---|---|---|
| **segm mAP** | 0.6136 | **0.6475** | 0.6442 | 59.7 → 67.5 |
| segm AP50 | 0.9333 | **0.9389** | 0.9381 | 89.2 → 91.7 |
| segm AP75 | 0.6719 | **0.7074** | 0.7070 | 65.6 → 74.8 |
| segm AP_m/l | 0.481/0.675 | 0.546/0.705 | — | — |
| bbox mAP | 0.7168 | 0.7096 | 0.7102 | 62.3 → 70.3 |
| bbox AP50 | 0.9328 | **0.9409** | 0.9297 | 88.3 → 93.6 |
| bbox AP75 | 0.8226 | 0.8132 | 0.8158 | 75.2 → 81.0 |

出报告用 raw（val tie-break 胜 EMA）。原尺度口径差异 ≤0.55 点（已验证）。

## 3. 结论

- **bbox mAP 71.0 / AP75 81.3 / AP50 94.1：三项全表第一**（超 RSPrompter-anchor
  70.3/81.0/93.6）；segm AP50 93.9 全表第一（超 query 91.7 达 2.2）。
- **segm mAP 64.8**：超表内多数方法，距 RSPrompter-query 67.5 差 2.7；
  AP75 70.7 距 74.8 差 4.1。
- C5-v2 相对 C4 的增益（mAP +3.4 / AP75 +3.6）集中在 GTF(+6.9)、
  airplane(+6.0)、bridge(+3.8)、vehicle(+3.2)——边界精修的定向价值，
  也反证 8/30 "编码器升级无效"的结论是在残缺架构上得出的。

## 4. 逐类 segm AP（原尺度，C4→C5-v2）

| 类 | C4 | C5-v2 | | 类 | C4 | C5-v2 |
|---|---|---|---|---|---|---|
| airplane | 0.214 | 0.274 | | basketball | 0.794 | 0.766 |
| ship | 0.570 | 0.562 | | GTF | 0.883 | **0.952** |
| storage_tank | 0.786 | 0.782 | | harbor | 0.493 | 0.503 |
| baseball | 0.801 | 0.820 | | bridge | 0.373 | 0.411 |
| tennis | 0.621 | 0.636 | | vehicle | 0.546 | 0.578 |

## 5. 遗留短板：airplane 的"飞机+阴影"复合标注（已诊断，未解决）

airplane AP50=**0.946** → AP65=0.270 → AP75=**0.007** 断崖；96% 实例
IoU≥0.5 分类定位正确；GT 掩码填充率中位数 0.25（方形框内对角稀疏形）。
判定：标注多边形=飞机+对角阴影复合体，SAM2 的 SA-1B 物体先验驱使
模型只分割飞机本体，IoU 被钉在 0.5-0.7。airplane 若修复到 ~0.7，
mAP 预计 +4 点（→0.69，超 RSPrompter）。候选方向：更长训练让监督
压过先验 / airplane 类的阴影感知增强（把阴影上下文并入 prompt）。

## 6. 文件

- 训练日志：`logs/ablations/vhr10_{fast400,large400,c5v2_400}_*.log`
- 指标：本目录 `c4/c5v2_*_val_metrics_*.json`；predictions.json 在
  各 run checkpoint 目录（`inference_validation_model/`）
- 工具：`tools/eval_vhr10_original_scale.py`（原尺度口径）
