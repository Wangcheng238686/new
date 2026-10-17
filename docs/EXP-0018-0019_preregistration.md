# EXP-0018/0019 预注册：WARM-R1G 全量双 seed（2026-10-17）

依据：DUAL-WARMSTART-SCREEN 筛查链结论
（`docs/DUAL-WARMSTART-SCREEN-20261017_conclusion.md` 第 6 节建议第 1 条）。
R1G 配方（EXP-0015 基座 + gate_init 0.001 + warm 冻结）是现有证据下最有
把握过 DUAL_GAIN 门的接法：12 个筛查净差全正、R1G@60 bbox +0.147 为谱系
最强、门开度仅 0.085（平衡态 0.489 的 17%）——收益项缺完整预算而非信息。

## 臂定义

| EXP-ID | 臂 | seed | GPU | 世界大小 |
| --- | --- | --- | --- | --- |
| EXP-0018 | SURFBEV-warm-r1g-full-DDP2-v2-noaug | 44 | 0,1 | DDP2 |
| EXP-0019 | SURFBEV-warm-r1g-full-DDP2-v2-noaug | 45 | 2,3 | DDP2 |

- 配置：`configs/ue_whu_a3sgm_512_warm_r1g.py`（身份
  `ue-whu-a3sgm-512-warm-r1g-v1`，指纹与筛查臂逐位一致——同配置复用）
- warm 初始化：`--init-from EXP-0009 last_model_epoch100.pth` +
  `--freeze-base`（trainable = 前缀匹配 ∩ 构建时可训 = 0015 LoRA 精确集
  1,492,528 + fusion 135,168，cdc9e822 语义）
- 训练：**100 epoch 单余弦无重启**、每卡 batch 1、累积 4、**global batch 8
  与阶梯 DDP4 臂逐值一致**、optimizer steps/epoch 87 一致、lr 5e-4、wd 0.05
  （fusion 免 wd）、warmup 100、amp、EMA 关、无早停、无随机增强
- 数据：冻结快照（pair `f15cd8f5…`、split `17f49430…`、train 694 / val 100）；
  **test 至两 seed last 产生前不开封**

## DDP2 协议注记（登记为与阶梯的已知偏差）

双 seed 并行受四卡限制，每 seed 取 2 卡（batch 1×累积 4×2 卡 = global 8，
与 DDP4 臂的 4×1×2=8 逐值一致；每 epoch optimizer steps 同为 87）。差异仅
在 shard 组成（2 分片 vs 4 分片）与 rank0 验证不变。选择理由：用户要求双
seed 并行；两 seed 同用 DDP2，臂间完全可比；对锚点（EXP-0009，DDP4）的
配对比较口径不变。

## 四要素

1. **主指标**：val 100 逐场 COCO AP 配对 bootstrap（10k，seed 20261014）
   dual(on) − single（EXP-0009 既有 rows，唯一单流锚点，seed 44）：
   每 seed 各报 segm/bbox Δ 与 CI95；另报**双 seed 合并**（scene×seed
   配对池，200 行）。
2. **伴随指标**（≥2 维度）：(a) UAV-off 消融（on−off，parity 构造下 = 净差，
   仍独立实测一遍）；(b) 门遥测（gate_mean/std/absmax、delta_p3_ratio、
   bev_coverage，末 5ep 均值 vs 0015 平衡态 0.489）；(c) 训练期冻结有效性
   （参数组审计行 + parity 抽查 3 场逐位）。
3. **选择口径**：last_model_epoch100.pth（唯一；无 best/平台条款——筛查期
   延长劣化教训：best 尖峰不作证据）。
4. **判定门**（沿用阶梯原文，两 seed 各判 + 合并判）：
   - **DUAL_GAIN**：双指标 CI95 下界均 >0 且幅度 ≥ +0.5 点；
   - **DUAL_PARITY**：双指标 CI 均含 0 且点估计 ≥ −0.2（方向保持、量级不足）；
   - **DUAL_HARM**：任一指标 CI 上界 < 0；
   - 中间态（一指标过线一指标含 0）按八臂拉通惯例逐项报告。
   - seed 间一致性条款：两 seed 点估计符号不一致 → 判读降级为"seed 敏感"，
     不得择优报告。
5. **test开封**：两 last 齐备后对两 seed 各跑一次标准推理（沿用 ladder 的
   test 协议），结果仅作最终报告，不参与判定门。

## 与筛查臂的关系

- 不从筛查 checkpoint 续训（重启调度教训）；从 0009 warm 初始化全新跑。
- 预算：每 seed ~9.5h（DDP2 100ep），并行一次收线。
- 若 DUAL_GAIN 达成：双 seed + 筛查链构成完整证据链，test 开封后可进
  论文口径；若 DUAL_PARITY：收益上限回到货物宽度（R3 判读），货物升级
  （统计通道加权）为下一变量。
