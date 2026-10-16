# DUAL-WARMSTART-SCREEN 预注册（2026-10-17 夜间筛查链）

路线决议（PI 2026-10-17）：**不换轨道**，保留 BEV 栅格注入路线，按"代价/收益分解"
（`docs/DUAL-AB-LADDER-20261016_conclusion.md`）攻击两项：
off−single 代价（−0.35~−1.65，联合训练拖累）与 on−off 收益（+0.38~+1.59，真实但小）。
本链为**热启动筛查级**（治理规程"模块叠加型"路径；冻结基座=EXP-0009 方法基线权重，
模块零初始化，missing-keys 恰为新增模块键），不占用 EXP-ID；过筛查门后另册申请全量。

四项（严格顺序执行，前项判负不阻塞后项，但判读须引用前项结果）：

| # | 臂名（临时编号） | 类型 | 单变量（相对 WARM-R1） |
| --- | --- | --- | --- |
| 1 | BEV-READOUT-P0 | 冻结诊断探针（无模型训练） | — |
| 2 | WARM-R1 | 热启动筛查 30ep | 基准臂：0015 配方逐字复用 + trainer 侧 init-from(0009)+冻结基座 |
| 3a | WARM-R1G | 热启动筛查 30ep | gate_init 0.0→0.01（非零初始参与） |
| 3b | WARM-R1B | 热启动筛查 30ep | +bev_head（buildingness 空间门 + 辅助监督） |

## 固定协议（全部臂共享，单变量纪律）

- 数据：冻结快照 `/data2/wangcheng/dataset_frozen/ue_raw_hero_uav_pairset_20261014_frozen`
  （pair manifest `f15cd8f5…`，v2 split `17f49430…`，train 694 / val 100 / test 151，test 不开封）
- 模型配方：EXP-0015 基座逐字（`configs/ue_whu_a3sgm_512_dualv2_l0_lora.py`，
  身份 `ue-whu-a3sgm-512-dualv2-l0-lora-v1`，L0(P2) 单层注入 + UAV LoRA + fusion 免 wd）
- 训练：seed 44、DDP4、每卡 batch 1、累积 2、30 epoch、lr 5e-4、wd 0.05、warmup 100、
  amp、EMA 关、无早停、last 口径
- warm-start：`--init-from EXP-0009 last_model_epoch100.pth`（跨架构显式豁免）+
  `--freeze-base`（仅 `uav_encoder.` / `proj` / `gate` / `bev_head` 前缀可训，其余
  requires_grad=False 且强制 eval 模式）；UAV-off 前向按构造逐位等于 0009
- 评估：val 100 逐场 COCO AP（`per_scene_ap` 同 ladder 口径），on/off 两遍；
  对照单流行 `probes/DUAL-AB-LADDER-20261016/rows_single.jsonl`（EXP-0009 既有逐场行，不重训）

## 四要素

### 1. BEV-READOUT-P0（探针）

- **主指标**：val 40 场 cell-IoU@0.5（64×64 格，GT 并集 mask area 降采样>0.5 二值）：
  描述子块（0:256）读出 IoU(desc) 减 zmean 单通道读出 IoU(zmean)，逐场配对。
- **伴随指标**（≥2 维度）：(a) 常数先验基线 IoU(const)（恒输出训练集平均覆盖率）；
  (b) cell-AP；(c) sham 对照两枚——描述子通道内置换（IoU(perm)）与场景错配
  （IoU(shuf)，BEV 与 GT 错一格配对）。
- **选择口径**：train 60 / val 40（split 顺序前 N，与 DUAL-REDUNDANCY 探针同口径同 seed 44）；
  读出器固定 3 层 MLP（输入组通道→256→256→1，BCE，AdamW 1e-3，≤200 epoch，早停于
  val loss 平台）；无任何检测模型训练。
- **判定门**：
  - **R3-PASS**：IoU(desc) − IoU(zmean) ≥ +0.05 且逐场配对 percentile bootstrap
    （10k，seed 20261014）CI95 下界 > 0 → 描述子承载高度之外的建筑物信号，
    整合是瓶颈，WARM 链按计划执行；
  - **R3-CEIL**：IoU(zmean) − IoU(const) < +0.03 → 连高度通道都读不出任务信号，
    当前 BEV 货物天花板极低，WARM 链照跑但预期下调，路线内下一步指向货物升级；
  - **R3-NULL**：介于两者之间（desc 无独立信号但 zmean 有）→ 收益上限≈高度通道，
    WARM-R1B（buildingness 门）权重上调为链内主判读臂。
- **判读边界**：读出质量是融合增益的必要非充分证据；sham 未崩溃（IoU(shuf) 未显著
  低于真配对）则全部结论作废；探针结论不替代 AP 级结局验证。

### 2–4. WARM-R1 / WARM-R1G / WARM-R1B（筛查臂）

- **主指标**：val 100 逐场配对 bootstrap（10k，seed 20261014）：
  dual(on) − single(EXP-0009) 的 segm 与 bbox 点估计与 CI95；
- **伴随指标**：(a) on−off 消融（同 checkpoint 双遍，CI95）；(b) 门遥测
  （gate_mean/std/absmax、delta_p3_ratio、bev_coverage，末 5 epoch 均值）；
  (c) 冻结有效性核验（off 前向与 0009 逐位一致的哈希断言 + 训练日志参数组审计行）。
- **选择口径**：last_model_epoch30.pth（唯一口径；无 best/平台条款）。
- **判定门**：
  - **SCREEN-PASS（升级全量）**：on−off 双指标 CI95 下界均 > 0，且 on−single
    的 segm 点估计 > 0；
  - **SCREEN-EXT（一次性延长）**：on−off ∈ (0,+0.3) 且任一 CI 含 0 → 该臂从
    last 续训至 60ep 再判一次（唯一一次延长，预注册于此）；
  - **SCREEN-FAIL**：on−off 双指标 CI 均含 0（热启动下通路仍学不开）→ 该臂关账；
  - **护栏（停查实现）**：on−single < −0.5 且 CI 上界 < 0（冻结被破坏或门发散）。
- **判读边界**：30ep 筛查对"融合模块从零学跨视角映射"的收敛时长敏感，SCREEN-FAIL
  不等同于全量 FAIL（故设 EXT 条款）；筛查数字不进论文口径，不更新阶梯终值表。

## 风险与防护（预注册）

1. gate_init 的量纲依据与**护栏触发记录（2026-10-17 00:37，启动前）**：初版
   gate_init=0.01（0015 末代门 std 0.0055 的 ~2×，k=2 gate-sweep 证据）。2-batch
   冒烟实测 step-0 delta_p3_ratio=2.047 > 1.0，**护栏按条款终止该取值**（未定标
   BEV × 0.01 门注入为基座范数 2 倍）。修订（启动前、判定门不变）：gate_init
   降至 0.001，复测 step-0 delta_p3_ratio=0.205（0015 平衡态 0.489 之内）。
   冒烟证据：`/tmp/warm_smoke_r1g.log`（终版）与本节记录。
2. bev_head 输入归一化：head 内部载入 0014 谱系 harvest 定标（frozen buffer，
   sha `f7c3c5fb…`）；proj 通路保持 0015 配方不动（单变量纪律不破坏）。
3. 冻结子模块的 LoRA dropout 噪声：冻结模块强制 eval 模式（train() 后复位），
   防止卫星侧 train/eval 失配。
4. 与 EXP-0017 的关系：0017 是延迟解冻（弱形式），本链是冻结基座（强形式）；
   0017 的拉通落判另行归档，不改动本链预注册条款。

## 工件

- 探针：`scripts/ue/probe_bev_readout.py` → `probes/BEV-READOUT-20261017/`
- 筛查：`scripts/ue/run_warm_screen.sh {r1|r1g|r1b}`、
  `scripts/ue/eval_warm_screen.py` → `probes/WARM-SCREEN-20261017/`
- 训练日志：`/data2/wangcheng/checkpoint/surfbev_b0/launcher_logs/WARM-*.log`
- checkpoint：`/data2/wangcheng/checkpoint/surfbev_b0/surfacebev_v2_512warm_*`
