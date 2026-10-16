# DUAL-WARMSTART-SCREEN 结论（2026-10-17 夜间筛查链）

对照预注册 `docs/DUAL-WARMSTART-SCREEN-20261017_preregistration.md`（四要素与判定门原文）。
臂：BEV-READOUT-P0（探针）→ WARM-R1 → WARM-R1G → WARM-R1B（各 30ep 筛查，seed 44，
init-from EXP-0009 + 冻结基座）。代码提交 `a8dff49`（实现）+ `9f9e6df`（launch 修复）；
实现审计见 `docs/reviews/DUAL-WARMSTART-SCREEN-20261017_implementation_review.md`。

## 1. BEV-READOUT-P0（探针，2026-10-17 01:11 收线）

冻结定标 BEV（EXP-0014 口径，64×64）→ 3 层 MLP 读出 GT 建筑并集 mask；
train 60 / val 40（与 DUAL-REDUNDANCY 同口径），seed 44。
产物：`probes/BEV-READOUT-20261017/readout.json`、`probes/BEV-READOUT-run.log`。

| 通道组 | val cell-IoU@0.5 | cell-AP |
| --- | --- | --- |
| 常数先验（恒 1，=覆盖率） | 0.5515 | — |
| desc_0_256（语义描述子） | 0.6709 | 0.873 |
| **sham_perm_desc（通道内置换）** | **0.6696** | 0.872 |
| zmean_256（高度单通道） | 0.8580 | 0.903 |
| **stats_256_262（6 几何/统计通道）** | **0.8954** | **0.977** |
| all_262（全部） | 0.8719 | 0.969 |
| sham_scene_shift（场景错配） | 0.4150 | — |

预注册对比：
- desc − zmean = **−0.187 [−0.198, −0.176]**（描述子显著差于高度通道）
- zmean − const = **+0.306 [+0.287, +0.327]**（高度通道显著高于底率）
- sham 场景错配崩溃至 0.415（< all_262 − 0.10）→ 探针有效

**落判：R3-NULL**（预注册判读：描述子无独立信号但 zmean 有 → 收益上限≈高度通道，
WARM-R1B 上调为链内主判读臂）。

机制读数（超出预注册字面、供归因）：
1. **描述子的 0.671 ≈ 其置换对照 0.670**：MLP 只从描述子块读到"边际统计"
   （与覆盖率相关的平均激活），未读到通道身份级场景信号；语义货物近乎空载，
   与 DUAL-REDUNDANCY 的"独特但不对题"（残差空间均匀 1.19）定量互洽。
2. **统计通道组是全场最优（0.895）且全部通道混入反而稀释（0.872）**：可提取的
   任务信号集中在 zmean/zstd/vcount/zres/flags 六个几何通道。
3. 推论：BEV 融合路线内的收益上限由高度/覆盖几何承载——这正是 R1B
   buildingness 门消费的通道组；R1G（开度）与 R1（纯 warm 基准）保留作为
   动力学对照。

## 2. WARM-R1（冻结基座纯加性读出，30ep）

- 启动 2026-10-17 01:11，契约行 `WARMSCREEN_CONTRACT arm=r1 commit=9f9e6df dirty=0 …`
  （`launcher_logs/WARM-R1-ddp4.log`）；checkpoint 目录
  `surfacebev_v2_512warm_r1_scr30_ddp4_s44_WARM-20261017`。
- （收线后填写：last segm/bbox、on/off 逐场对比、判定）

## 3. WARM-R1G（gate_init 0.001 非零初始参与，30ep）

（收线后填写）

## 4. WARM-R1B（buildingness 空间门 + 辅助监督，30ep）

（收线后填写）

## 判定门原文（引用预注册）

SCREEN-PASS：on−off 双指标 CI95 下界均 >0 且 on−single segm 点估计 >0；
SCREEN-EXT（一次性）：on−off ∈ (0,+0.3) 且任一 CI 含 0 → 续训至 60ep 再判；
SCREEN-FAIL：on−off 双指标 CI 均含 0；护栏：on−single < −0.5 且 CI 上界 < 0。
