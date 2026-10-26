# TRAIN-ONOFF-20261023：train 侧 on−off 判决性检查

**问题**：九次消费失败的各双流臂，其 UAV 通路是"在训练分布上学到了东西、被
train→val 分布错配杀死"（H-DIST，分布过拟合），还是"从未学成"（H-OPT，
优化失败）？判定依据 = 同一检查点在 **train split** 上的 on−off 配对差，
与已知 val on−off（≈0）对照。

**背景**：两代数据集的 split 粒度错配实测（2026-10-23）：旧快照
train/val/test 实例密度 55.6/27.9/63.8、small 占比 48%/26%/50%；新归档
58.9/32.6/98.0、51%/24%/72%。val 系统性最粗。历次 A/B 均在 val 判定，
train 侧 on−off 从未测量——本检查补上这一格。

## 预注册（2026-10-23，先于任何评估运行）

- **对象**：EXP-0020（WARM-R1C 全量 100ep，seed 44，
  `surfacebev_v2_512warm_r1c_full100_ddp4_s44_EXP-0020/last_model_epoch100.pth`）。
  已知 val 侧参照：on−single segm +0.03 [−0.10,+0.16] / bbox +0.09 [0.00,+0.20]
  （warm 冻结下 off≡single 逐位成立，on−off 与之相同）。
- **评估**：`UERawPairDataset(SNAP, 'train', load_uav=True)`，**前 200 条
  records（split manifest 顺序，确定性子集）**；on / off（`uav_eval_off`）
  两模式逐场 COCO AP（`probe_dual_ab_ladder.per_scene_ap` 同口径）。
- **统计**：train on−off 逐场配对 bootstrap 10k、seed 20261014（谱系一致）。
  附带 segm_s/m/l、bbox_s/m/l 同口径（探索性）。
- **判定门**（单位：点）：
  - **H-DIST 成立**：train on−off **segm 点估计 ≥ +0.30 且 CI95 下界 > 0**
    ——通路在训练分布上有真实效应，val 零效应归因分布错配；
  - **H-OPT 成立**：train on−off **双指标 CI95 均含 0 且 segm 点估计 < +0.15**
    ——通路在训练分布上也未被使用，"从未学成"结论坐实；
  - **中间态**：其余情形——报告 train/val 效应比，部分归因。
- **复现条款**：若主对象落 H-DIST，追加 EXP-0018（R1G 全量）同口径复现；
  落 H-OPT 或中间态则不追加。
- **边界**：本检查为 eval 期证据，不区分"通路输出正确但不被消费"与
  "通路输出本身无用"（该区分已由 ATTR 阶梯/机制门覆盖）；200 场为 train
  的确定性子集，不代表全 694 场（敏感性：若落中间态，报告与前 400 场
  外推的稳健性说明）。

## 结果（2026-10-23 16:0x 收线，~11 分钟，GPU3 共租推理，无事故）

对象 EXP-0020（R1C 全量 100ep，s44），train 前 200 场（确定性子集），
on/off 两模式逐场 COCO AP，配对 bootstrap 10k/seed 20261014。
工件：`probes/TRAIN-ONOFF-20261023/{rows_train_onoff_r1c_full_s44_on,off}.jsonl、
summary_train_onoff_r1c_full_s44.json、identity_*.json`；日志
`probes/TRAIN-ONOFF-run.log`；脚本 `scripts/ue/probe_train_onoff.py`。

| 对比（点） | train[:200] Δ [CI95] | （参照）val Δ [CI95] |
| --- | --- | --- |
| on−off segm | **+0.003 [−0.05, +0.06]** | +0.03 [−0.10, +0.16] |
| on−off bbox | +0.029 [−0.01, +0.08] | +0.09 [0.00, +0.20] |
| on−off segm_s（探索） | +0.072 [−0.06, +0.22] | — |
| on−off bbox_s（探索） | +0.106 [−0.03, +0.27] | — |

绝对值参照：train on/off segm 0.8575/0.8575、bbox 0.8681/0.8678——
off≡single（冻结基座逐位成立），train 绝对 AP 比 val 高约 8 点
（基座自身的训练集记忆，与通路无关）。train 效应**不大于** val 效应
（segm 甚至更小）——若分布过拟合成立应看到 train ≫ val。

## 落判（按预注册门）

- H-DIST（分布过拟合）：**不成立**——要求 train segm ≥ +0.30 且 CI 下界>0，
  实测 +0.003、CI 含 0；
- **H-OPT（从未学成）：成立**——双指标 CI95 均含 0 且 segm 点估计 +0.003
  < +0.15。UAV 通路在**其被优化的训练分布上**同样零贡献。
- 复现条款不触发（H-OPT 落判不追加 EXP-0018）。

**结论**：九次消费失败的性质是**优化层的"从未学成"**，不是"学到了、
被 train→val 错配杀死"。分布错配作为臂失败主因的假说被本检查关闭；
其真实角色收窄为此前确认的三个次要环节——长训回吐（基座自身过拟合
train 粒度）、val 稀释小目标信号（bbox_s 方向在 train/val 两侧一致为正
但均不显著）、val→test 绝对数漂移需 per-size 口径解读。

**对路线的含义**：
1. "分布修复救通路"（粒度平衡采样/重加权）作为双流补救手段**降级**——
   通路在任何分布上都未被使用，先解决消费问题才轮到分布问题；
2. 正在跑的 adapter d 系列的判定仍建议带 per-size 报告（split 三方
   粒度错配是协议事实，与本落判独立成立）；
3. 本检查同时给了一个免费副产品：冻结基座在 train[:200] 的逐场锚点
   行（off 行），后续任何 train 侧评估可直接复用。

边界：eval 期证据、单检查点（预注册主对象）、200/694 子集；不区分
"输出正确但不被消费"与"输出本身无用"（该区分由 ATTR 阶梯既有结论覆盖：
C0−C2=0，两者兼有）。test 集未触碰。
