# DISTILL-P0 探针预注册（临时号 DISTILL-P0-20261016）

第一级阶梯（可学性探针套件）。上位设计：`docs/EXP-0018_distill_design_discussion.md`
（两轮对抗评审定稿的四门体系）。本文件把门体系转正为可执行预注册。探针级：
只读 + 短训（G2t），无正式 wrapper，产物 voided-grade，不消耗 EXP-ID，
不与 0017 抢卡（单卡共租，OOM 杀探针不杀 run）。

## 问题

训练期跨视角蒸馏的教师信号（冻结 SAM2 stride-4 描述子 [:256]、或归一化
zmean 高度通道）在卫星单流表示上是否存在**内容特异、几何锁定、机制可学**
的可塑价值——分四门裁决，并给出教师路由（G3）数据。

## 教师定义（两版并测）

- `desc`：`rasterize_surfacebev` 单 bin 64×64 的 [0:256] 描述子，
  covered-cell 白化（train 694 场逐通道 μ/σ，从缓存统计，JSON+sha 登记）；
- `height`：同 BEV 的 zmean 通道（cm→白化，1 通道）——DEPTH 先验的正参照。
- mask M：未定标 BEV 的 missing 通道（count>0）。
- sham90 对照：`rasterize_surfacebev(sham90=True)`，确定性变换。
- **教师缓存**：冻结教师逐场景落盘（train+val × true/sham，fp16），
  全部后续计算从缓存读——教师冻结故缓存精确。

## 四门（判据先于运行）

- **G1 结构门**：S = E_val|T̂−μ_spatial| / E_val|T̂−c_med|（covered cells；
  μ_spatial=train 逐 cell 通道均值图，c_med=train 逐通道中位数）。
  desc 门：S_desc < τ₁ = min(0.7, S_height+0.1)。杀纯噪声（S→1）。
- **G2f 冻结差分门**：ridge（λ 5 折 CV）从 0009 last P3（512d/cell）预测
  白化教师；统计量 val 逐场 R²(true)−R²(sham)（R² 对 desc 取 256 通道均
  值），scene-paired percentile bootstrap 10,000 draws、seed 20261014，
  **CI 下界 > 0**。杀"冻结表示下无几何锁定内容"。不过不杀方向（让 G2t
  裁决）。
- **G2t 可学性差分门（权威）**：从 0009 last 出发、任务损失置 0、仅
  L_distill=masked L1（λ=1）、AdamW lr 5e-4（warmup 100 步）、batch 4、
  fp32、单卡、5 epoch；g=Conv2d(512,256,1)（height 版 g=Conv2d(512,1,1)）
  kaiming、wd=0，g 初始化前重置 RNG（四臂同 g 初值）。度量为 ep0/ep5 的
  val（40 场固定子集）逐场 R²(P3→T̂)；kill 量 = 逐场 ΔR²_true−ΔR²_sham
  的 bootstrap CI 下界 > **0.01**。四臂 = {desc, height} × {true, sham}。
  G2t 不过 → kill 该教师；两教师均不过 → 训练期蒸馏用法死亡。
- **G3 教师路由（数据采集，同 G2t 运行）**：冻结差分与可塑性数据 +
  （可选）误差轴读出留给第二级前裁决；本预注册不设 G3 门线，只采集
  两教师的全部四门量。

## 统计与口径

- bootstrap：val 场重采样、10,000 draws、seed 20261014、percentile CI95。
- R² 定义：逐场 covered cells 池化，1 − SSres/SStot（SStot 用该场该通道
  方差）；desc 为 256 通道均值；height 为单通道。
- 白化统计与缓存 hash 在附录封存后再跑门（顺序：harvest → 封存 → 门）。

## 判读边界

- 探针级结论，非 AP 主张；G2t 为微调可塑性下界（0009 last 最收敛点，
  保守方向）；40 场子集为 G2t 效率选择（G1/G2f 用全 100 场）。
- GPU 共租风险：若 OOM 影响正在运行的正式 run，杀探针（约定已在设计
  纪要 §6）。
- 产物：`/data2/wangcheng/checkpoint/surfbev_b0/probes/DISTILL-P0-20261016/`
  （缓存、白化 JSON、各门 JSON、G2t 四臂 log+R² 曲线）。

## 附则（harvest 后封存，2026-10-16 晚）

- 教师缓存：`probes/DISTILL-P0-20261016/cache/`，MANIFEST sha256
  `9ee0db703495838e56ae16f0173bd46cc58b3d7489a47b7c44ad97bc7bf62a4b`
  （train 694 × {true,sham} + val 100 × {true,sham}）；
- 白化 JSON：`cache/whitening.json` sha256
  `cb305b1cd3ee940680097f4bec0fcd6d0737d47b70363a60fb5da421a0245a21`
  （**勘误登记**：harvest 脚本首版把 sham 遍历也累计进统计（μ 约 2×
  偏置），已修脚本并从 train_true 缓存精确重算——缓存文件本身不受影响；
  desc σ 四分位 [0.0066, 0.0142, 0.0367]，与冗余探针 rms 交叉一致）；
- 0009 P3 身份：`identity_single.json` = ue-whu-a3sgm-512-v1（build_baseline
  严格加载）；
- G1 预读（运行后追加判读注记的依据）：S_desc=1.010、S_height=0.986 ——
  UE 无数据集级空间模板（无 SCVT4 式中心先验泄漏通道），G1 在此体制下
  对"噪声 vs 场景特异内容"无区分力（两者均给 S≈1），按设计层级交由
  G2f/G2t 裁决；G1 字面 fail 不作为 kill 证据，此判读边界在结论文档
  如实登记。
