# SURFBEV-P1-V2-H0 独立实现与接线审计（修复复审）

**审计人：**独立代理（未参与实现）  
**审计日期：**2026-10-12  
**审计对象：**`docs/SURFBEV-P1-V2-H0_preregistration.md`、
`portable_sam2_explicit_coarse/scripts/ue/probe_surfacebev_p1_h0.py`、
`portable_sam2_explicit_coarse/rsprompter/surfacebev.py`，以及为验证标签隔离而读的
`data/ue_raw_pair_dataset.py`。  
**审计方法：**逐行静态审计；另对已登记 checkpoint 计算 SHA256 并只读解析 metadata。
本审计不运行 H0，故不把实现审计通过表述为 H0 的运行结果通过。

## VERDICT: PASS — H0 实现/接线可启动；`H0_PASS` 仍须由全量运行工件逐门判定

上版审计的 B1（标签读取）和 B2（B0 权重身份未锁定）均已关闭。v2/test 隔离、
严格 B0 last load、被动预测哈希 parity、correct/sham 原始输入不变量、几何回投、
train-only 高度 bin 与遥测字段均已按预注册接线。没有发现会阻止正式 H0 的实现问题。

## 上版阻塞项复核

| 原发现 | 复审证据 | 状态 |
|---|---|---|
| B1：H0 实际读取 `satellite_instance_id_clean.png` | `UERawPairDataset` 新增 `load_targets`；只有其为真才读取 ID raster / 构造 `gt_*`（`ue_raw_pair_dataset.py:101-122`）。H0 的 train/val 均明确传 `load_targets=False`，且对每个取出的 item 断言不存在 `gt_*`（`probe_surfacebev_p1_h0.py:201-228`）。 | 已关闭 |
| B2：checkpoint 未锁死为 EXP-0006 last | H0 同时检查精确绝对路径、basename、文件 SHA256、zero-based epoch=99、run tag、seed=44、epochs=100 和 v2 split（`probe...py:35-37,180-192`）。实际文件 SHA256 为 `fc3ead…65145`，与预注册及常量一致；只读解析实际 checkpoint 得到 `epoch=99`、`seed=44`、`epochs=100`、v2 split 和登记 run tag。 | 已关闭 |

## 预注册门逐项接线核对

1. **v2 与 test 隔离：通过。** 只构造 train/val dataset（`probe...py:201-202`）；
   dataset 对 accepted manifest、v2 split SHA、v2 schema 和 694/100/151 场数量强校验
   （`ue_raw_pair_dataset.py:34-75`）。没有 test dataset 或 test scene asset 读取路径。
2. **B0 / 被动采集 parity：通过。** B0 用 strict state load（`probe...py:196-199`）。
   每个 val scene 在采集前后执行标准 `predict`，对 bbox、score、label、mask 分字段
   哈希；任何变化即失败；提取 P3 必须有限且为 128²（`:68-104`）。
3. **UAV encoder 与标签/预测隔离：通过。** 独立 encoder 设为 `eval()`、
   `requires_grad_(False)`，descriptor 在 inference mode 下获得（`:107-125`）。
   `rasterize_surfacebev` 的签名和实现只接收 descriptor、DeviceZ、K/C2W/W2C、height
   edges 和输出格，不接收 label、B0 prediction 或 WorldPosition（`surfacebev.py:12-99`）。
4. **几何及 12-bin SurfaceBEV：通过。** DeviceZ 按 RGB radix 解为 camera-Z；经 UAV
   K/C2W 回投 world，卫星 K/W2C 重投 P3 格，并以卫星 camera-Z z-buffer 保留表面
   （`surfacebev.py:24-52`）。C2W/W2C 用 float64 互逆关系核验（`probe...py:128-145`）。
   height edges 只在读 val 前、由完整 694 train 的 0.5/99.5 分位生成，low/high 均有限
   且严格有序（`:203-223`），产生 12 个 bin。
5. **correct/sham 等容量和原始输入不变量：通过。** correct/sham 使用同一个 `desc`；
   correct 返回后和 sham 返回后均复核 descriptor hash；每场显式断言两臂
   `input_points` 一致，并把两侧 hash/计数写进 `per_scene.jsonl`
   （`probe...py:229-253`）。sham 只绕 satellite camera XY 中心旋转 90°，再完整重投影
   与 z-buffer（`surfacebev.py:32-44`）；in-FOV/winner 数不要求相等，避免破坏阴性
   几何对照的意义。
6. **遥测与 verdict 语义：通过。** 每场保存 input/in-FOV/zbuffer point 数、covered
   cells、non-missing bins，以及有效 bin 的 independent-view count 和 in-FOV candidate
   的 z-buffer residual `p0/p50/p90/p100`（`surfacebev.py:87-99`、
   `probe...py:243-253`）。`H0_PASS` 只会在非 smoke、694 train height pass 和 100 val
   rows 全完成时生成；否则明确为 `SMOKE_ONLY`（`:203-205,255-265`）。

## 发现清单

- **无阻塞发现。**
- **运行前提醒（非缺陷）：** 该报告只说明程序接线满足预注册。实际正式运行后仍必须
  检查 `summary.json` 为 `H0_PASS`、100 条逐场记录齐全、每场 correct/sham 均有
  z-buffer surface point、遥测无 NaN/负计数/越界，并归档输出目录。只有这些运行时门
  同时成立，才允许另立 P1-V2 线性读出；H0 本身仍不能证明任何 segm/bbox AP 增益。

