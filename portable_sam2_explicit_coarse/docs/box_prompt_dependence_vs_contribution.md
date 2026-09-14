# Box token 的依赖性与净贡献：观察、因果验证与训练期 dropout 方案依据

日期：2026-09-09。范围：matrix300 P/PB 臂（进行中）+ prompt_switch 干预探针。
本文为"训练期 box token dropout"方案提供证据支撑，并固定机制解释。

状态：PB 臂 E271–300 平台窗口待跑完回填（预计 2026-09-09 ~17:15）；其余数字均已定版。

## 1. 摘要

- 从头训练比较（净贡献）：PB（points+box）对 P（points）的 segm 领先约 **+0.010**（窗口均值，
  多数窗口为正），bbox 持平——box token 有**小而真实**的增益。
- 冻结权重干预（依赖性）：在 PB best 权重上推理时移除 box token，segm 从 0.6530 掉到
  0.6173（**−0.0357**）——decoder 对 box token 是**重度依赖**。
- 依赖（−0.036）≫ 贡献（+0.010），机制解释：box 信息与 RoI 裁剪几何高度冗余，PB 的 decoder
  把几何判断"外包"给显式 box token，裁剪内容推断路径训练不足；P 臂被迫把该路径练满。
- 推论（下一步方案）：训练期对 box token 做 dropout（classifier-free guidance 式条件丢弃），
  强制两条路径同时激活，预期 ≥ 现 PB 绝对值，且 drop_box 退化幅度收窄本身就是方案的机制验收指标。

## 2. 观察：matrix300 P vs PB（净贡献）

协议：NWPU 520/130 全量、300ep 单余弦（19500 步）、有效 batch 8、seed 44、fi 契约、
分片验证（每 5ep）、num=256、P2 off；唯一差异 = EXPLICIT_PROMPT_MODE（points vs points_box）。
采样量对齐（pos_rois ≈ 58 两臂相同），遥测可比。

逐窗口 segm/mAP（PB−P，30ep 窗口）：

| 窗口 | E31–60 | E61–90 | E91–120 | E121–150 | E151–180 | E181–210 |
|---|---|---|---|---|---|---|
| Δ | +0.0111 | +0.0142 | −0.0020 | +0.0096 | +0.0103 | +0.0018 |

对齐 best（两臂同在 E155）：**PB 0.6530 vs P 0.6412（+0.012）**；
bbox best：PB 0.7276 vs P 0.7197（best 口径略高，窗口口径待回填）。

E271–300 平台（终判，2026-09-09 18:10 回填）：**P segm=0.6258 / bbox=0.7161；
PB segm=0.6395±0.0006 / bbox=0.7216±0.0009（E300=0.6395/0.7215）→ 平台差
segm +0.0137 / bbox +0.0055**——box 增益双指标确认，segm 幅度远超 +0.005 判据线。

各选择依据 checkpoint 的完整 val COCO（`infer_from_checkpoint --split validation`，
**四臂 × 4 口径 = 16/16**：P/PB 于 2026-09-09，A0/A3 于 2026-09-10）：

| 臂 | ckpt | segm/mAP | segm/AP50 | segm/AP75 | bbox/mAP | bbox/AP50 | bbox/AP75 | segm/AR100 | bbox/AR100 |
|---|---|---|---|---|---|---|---|---|---|
| P | bestS(E155) | 0.6413 | 0.9264 | 0.6949 | 0.7024 | 0.9361 | 0.8142 | 0.6831 | 0.7533 |
| P | bestB(E250) | 0.6245 | 0.9205 | 0.6643 | 0.7197 | 0.9298 | 0.8188 | 0.6636 | 0.7633 |
| P | bestC(E280) | 0.6299 | 0.9256 | 0.6741 | 0.7181 | 0.9360 | 0.8189 | 0.6689 | 0.7607 |
| P | last(E300) | 0.6247 | 0.9218 | 0.6576 | 0.7161 | 0.9339 | 0.8145 | 0.6642 | 0.7588 |
| PB | bestS=bestC(E155) | 0.6530 | 0.9354 | 0.7165 | 0.7174 | 0.9352 | 0.8168 | 0.6939 | 0.7614 |
| PB | bestB(E235) | 0.6411 | 0.9290 | 0.6742 | 0.7276 | 0.9416 | 0.8139 | 0.6782 | 0.7649 |
| PB | last(E300) | 0.6392 | 0.9233 | 0.6855 | 0.7215 | 0.9336 | 0.8255 | 0.6750 | 0.7596 |
| A0 | bestS(E105) | 0.6446 | 0.9301 | 0.6918 | 0.6963 | 0.9467 | 0.8229 | 0.6831 | 0.7405 |
| A0 | bestB=bestC(E265) | 0.6355 | 0.9362 | 0.6769 | 0.7284 | 0.9443 | 0.8105 | 0.6757 | 0.7694 |
| A0 | last(E300) | 0.6331 | 0.9312 | 0.6671 | 0.7218 | 0.9370 | 0.8041 | 0.6753 | 0.7649 |
| A3 | bestS(E45) | 0.6930 | 0.9444 | 0.7796 | 0.6508 | 0.9333 | 0.7795 | 0.7332 | 0.7044 |
| A3 | bestB(E265) | 0.6686 | 0.9432 | 0.7289 | 0.7240 | 0.9383 | 0.8314 | 0.7032 | 0.7628 |
| A3 | bestC(E220) | 0.6781 | 0.9540 | 0.7401 | 0.7206 | 0.9394 | 0.8403 | 0.7123 | 0.7618 |
| A3 | last(E300) | 0.6696 | 0.9497 | 0.7295 | 0.7190 | 0.9386 | 0.8280 | 0.7046 | 0.7591 |
| R3 | bestS(E70) | 0.6433 | 0.9363 | 0.6780 | 0.6847 | 0.9414 | 0.8002 | 0.6846 | 0.7314 |
| R3 | bestB(E250) | 0.6248 | 0.9083 | 0.6554 | 0.7131 | 0.9197 | 0.7996 | 0.6674 | 0.7549 |
| R3 | bestC(E290) | 0.6284 | 0.9096 | 0.6720 | 0.7123 | 0.9184 | 0.8036 | 0.6685 | 0.7537 |
| R3 | last(E300) | 0.6271 | 0.9105 | 0.6709 | 0.7127 | 0.9181 | 0.8022 | 0.6676 | 0.7544 |
| A3-s45 | bestS(E75) | 0.6808 | 0.9572 | 0.7556 | 0.6936 | 0.9470 | 0.8229 | 0.7166 | 0.7355 |
| A3-s45 | bestB=bestC(E270) | 0.6677 | 0.9397 | 0.7259 | 0.7068 | 0.9266 | 0.7959 | 0.7046 | 0.7515 |
| A3-s45 | last(E300) | 0.6659 | 0.9361 | 0.7254 | 0.7046 | 0.9238 | 0.7998 | 0.7022 | 0.7472 |
| A0-sg | bestS(E155) | 0.6592 | 0.9349 | 0.7152 | 0.7183 | 0.9448 | 0.8239 | 0.6976 | 0.7574 |
| A0-sg | bestB=bestC(E225) | 0.6513 | 0.9307 | 0.7029 | 0.7376 | 0.9380 | 0.8469 | 0.6934 | 0.7805 |
| A0-sg | last(E300) | 0.6483 | 0.9342 | 0.6931 | 0.7348 | 0.9399 | 0.8477 | 0.6879 | 0.7736 |

**A0-sg（dense 梯度隔离）：dense 家族首次正增益**（2026-09-12，另一 AI 助手实现，本机
matrix300 seed44 完赛 + 四权重推理 14:54）。与 A0 的唯一差异 = dense 源 stopgrad 隔离
（final-mask 梯度不再回传 coarse head，后者专职喂 2P2N 点）。平台 E271-300：
**segm 0.6475 / bbox 0.7346**（vs A0 0.6342/0.7208 → **+0.0133/+0.0138**，双指标同向；
vs PB 0.6395/0.7216 → **+0.0080/+0.0130**，dense 包首次反超 PB）。轨迹：早期 −0.02
（隔离后适应）→ E60 起稳定领先 → E211-240 +0.019 无衰减。bbox/AP75 last 口径 0.8477
为全表新高（超 PB bestB 的 0.8255）。dense 叙事从阴性附录升级为"梯度隔离"正增益行；
a0_sg_densecapres256（隔离+容量头，2026-09-12 12:14 起）在跑，出数后定最优 dense 基座
→ 新 A3 变体隔离 UDPR 干净增益。

A0-sg 推理产物：`ablations/vhr10_p2v2_matrix300_a0_sg_tr1.0_va1.0/inference_a0_sg_
{best_model_epoch155,best_bbox_model_epoch225,best_composite_model_epoch225,last_model_epoch300}_vhr10/`。

**A3sg（A0-sg + UDPR）：六臂矩阵终版**（2026-09-13 07:35 完赛 + 四权重推理）。
与 a3 的唯一差异 = `SHAPE_DENSE_DETACH=1`（dense 源梯度隔离）；UDPR decoder-tail 配置
完全相同（K=64/hid=128/loss_w=1.0/delta_max=2.0）。指纹 `1cd99dde`。四权重全指标：

| A3sg | segm/mAP | segm/AP50 | segm/AP75 | bbox/mAP | bbox/AP50 | bbox/AP75 | segm/AR100 | bbox/AR100 |
|---|---|---|---|---|---|---|---|---|
| bestS=bestC(E35) | **0.6981** | 0.9481 | **0.7755** | 0.6879 | 0.9437 | 0.7832 | **0.7375** | 0.7367 |
| bestB(E285) | 0.6605 | 0.9203 | 0.7109 | **0.7135** | 0.9346 | **0.8055** | 0.6925 | 0.7509 |
| last(E300) | 0.6605 | 0.9279 | 0.7104 | 0.7130 | 0.9369 | 0.8042 | 0.6922 | 0.7506 |

平台 E286-300：**segm 0.6596±0.0017 / bbox 0.7130**。关键差值：
**UDPR 干净增益 = a3sg − a0_sg = +0.0121**（论文 UDPR 行）；
**完整方法总增益 = a3sg − PB = +0.0201**（论文 Full 行）；
a3sg − a3 = −0.0093（梯度隔离对 Full 有轻微代价，因 UDPR 的协同路径也被隔离）。
bestS(E35) 0.6981/s_75 0.7755 为全矩阵最高单点（错峰模式再现，last 口径为报告主口径）。

产物：`ablations/vhr10_p2v2_matrix300_a3sg_pbm_sg_udprk64_tr1.0_va1.0/inference_a3sg_
{best_model_epoch35,best_bbox_model_epoch285,best_composite_model_epoch35,last_model_epoch300}_vhr10/`。

**六臂矩阵终版（平台 segm / bbox，2026-09-13 定版）**：

| 臂 | segm | bbox | 单变量增量 |
|---|---|---|---|
| P | 0.6258 | 0.7161 | — |
| PB（+box） | 0.6395 | 0.7216 | +0.0137 |
| A0-sg（+dense 隔离） | 0.6475 | 0.7346 | +0.0080 |
| **A3sg（+UDPR）** | **0.6596** | **0.7130** | **+0.0121** |
| *(参照) A0（dense 无隔离）* | *0.6342* | *0.7208* | *−0.005 vs PB* |
| *(参照) A3（旧基座+UDPR）* | *0.6689* | *0.7190* | *+0.0347 vs A0* |

**论文推荐消融链**：P → PB(+box +0.014) → A0-sg(+梯度隔离 +0.008) → A3sg(+UDPR +0.012)
= **Full +0.020 vs PB**——单调递增、每级单变量干净。dense 无隔离（−0.005）、容量路线
（R3 −0.013、densecapres E145 掉队停跑）作分析附录。

**论文链 image-paired bootstrap CI（2026-09-13 补齐，E300 last 对 last，500 次重采样
seed44，130 图十类合同）**：

| 对比 | baseline | treatment | Δ 点估计 | 95% CI | P(Δ>0) |
|---|---:|---:|---:|---|---:|
| A0-sg vs PB（隔离） | 0.6392 | 0.6483 | +0.0091 | [−0.0113, +0.0342] | 0.834 |
| A3sg vs A0-sg（UDPR） | 0.6483 | 0.6605 | +0.0122 | [−0.0224, +0.0380] | 0.758 |
| **A3sg vs PB（Full）** | 0.6392 | 0.6605 | **+0.0214** | **[+0.0012, +0.0418]** | 0.978 |

判读边界：① last 口径点估计与平台口径互相印证（+0.0091/+0.0122/+0.0214 vs
+0.0080/+0.0121/+0.0201）；② **总增益 Full−PB 的 CI 下界为正，图像级显著**；③
两级单步增益的 CI 跨零——130 图重采样只量化 val 抽样噪声，不覆盖训练 seed/权重选择，
训练级稳定性以双 seed 证据为准（A3 双 seed 平台差 −0.0037、A0 双 seed 噪声尺度
0.008），论文中两种不确定性须分开表述，不得以跨零 CI 单独否定单步增益。工具复现
契约：三组 baseline/treatment 点估计均逐位复现各 inference run 自身 metrics.json
（PB last 0.6392、a0_sg last 0.6483、a3sg last 0.6605）；GT/images.json 复用
`dfbr_a0sg_e225_full_20260913/p0` 的标准导出，图像集合恒等校验 = manifest
processed_image_ids。工件：`diagnostics/matrix_chain_bootstrap_20260913/`
（含三份 bootstrap JSON 与三臂 dt_records 物化目录）。

**A3SGT（UDPR 尾部完全梯度隔离 / tailstop，2026-09-14 完赛 + 四权重推理）**：
预注册问题 = v1(a3sg) 的 segm +0.012 是否来自 tail 部署期写入（相对 a0_sg 的
bbox −0.022/composite −0.005 是否可以用梯度隔离消除）。设计/判定门见
`docs/a3sgt_tail_stop_design.md`；相对 a3sg 唯一模型差异 =
`decoder_tail_refiner_cfg.mode=residual_v1_stop`（tail 输入全 detach + 训练期
full-mask loss 走未细化 z，基座目标与 A0-SG 逐位同构）。

| 权重 | segm | AP50 | AP75 | bbox | bAP50 | bAP75 | sAR100 | comp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bestS=bestC(E225) | 0.6560 | 0.9388 | 0.7023 | 0.7240 | 0.9476 | 0.8273 | 0.6942 | 0.6900 |
| bestB(E250) | 0.6439 | 0.9405 | 0.6883 | 0.7274 | 0.9412 | 0.8304 | 0.6826 | 0.6857 |
| last(E300) | 0.6437 | 0.9374 | 0.6842 | 0.7225 | 0.9400 | 0.8294 | 0.6817 | 0.6831 |

**预注册门判定（last 与 bestC 双口径均 FAIL）**：segm 门（≥a0_sg+0.005）：last
−0.0046、bestC +0.0047（差 0.0003 未过线，且 bestS 0.6560 < a0_sg bestS 0.6592）；
bbox 门（±0.005）：last −0.0123、bestC −0.0136（超出，量级在单 seed 轨迹噪声
~0.01 内——基座目标同构但 RNG 流被 tail 初始化移位）；composite：last 0.6831 vs
a0_sg 0.6915（−0.0084）、a3sg 0.6868。对 a3sg（last）：segm −0.0168、bbox +0.0095。

**结论（UDPR 谱系按预注册关账）**：完全隔离下 tail 写入无部署期价值——机制遥测
（E275）：flip 1.47% / destroy 1.34%（净≈0）、delta_abs 0.81（v1 的 2.6 倍）、
点级 BCE 0.407（优于 v1 的 0.420）——tail 学会了"安全地不做有用的事"：点级目标
拟合良好、写入自我收敛到中性，选中错误仅 ~9.5% 翻转（cap 内可达 89%）。两代接法
互补证伪：**v1 的 segm 增益来自梯度通道的基座协同（伴随 bbox −0.022），完全隔离
则增益消失（bbox −0.012 属轨迹噪声）**——UDPR 的价值在训练期梯度、不在部署期
残差，且该梯度增益与 bbox 代价捆绑不可分。A0-SG 基座上不存在 composite 正增益的
UDPR 接法。产物：`ablations/vhr10_p2v2_matrix300_a3sgt_pbm_sg_udprk64ts_tr1.0_va1.0/
inference_{best,best_bbox,best_composite,last}_vhr10/`。

（可选收尾证据，未跑：a3sgt E300 上的 GT 授权 / cap 内最优写冻结 oracle 探针，
用于量化"任何写入策略"的价值上界，作为论文负结果分析的封顶。）

**A3SGT tail 写入策略 oracle 探针（2026-09-14，已跑毕 + 两轮独立审计 PASS）**：
探针 `inference/probes/tail_write_oracle_probe.py`，a3sgt last E300 冻结权重、
130 图、detector hash 七 cell 全同、p0 与生产推理逐位一致（0.6437，
`b0512558…`）。GT 仅在冻结前向之后用于重标定写入；sham 对照同码反路。

| cell（vs no_write=0.6375） | 干预 | Δ | image-paired CI95 | P(Δ>0) |
|---|---|---:|---|---:|
| p0 | 部署的已学写入 | +0.0062 | [+0.0018, +0.0119] | 0.998 |
| gt_auth | 已学 δ 仅写 GT 符号错误点 | +0.0090 | [+0.0054, +0.0144] | 1.000 |
| inv_auth（sham） | 已学 δ 仅写正确点 | −0.0032 | [−0.0054, +0.0000] | 0.026 |
| gt_cap（审计前） | clip(0.1·y−z0,±2) 全选点 | +0.0011 | [−0.0019, +0.0048] | 0.762 |
| **gt_cap_thr（审计后修正）** | 阈值锚定 target=b±m，仅错误侧写 | **+0.0267** | **[+0.0206, +0.0323]** | 1.000 |

关键发现与审计修正（两轮审计：实现 PASS + 复核 PASS）：

1. **已学写入显著有用**（+0.0062，CI 下界 +0.0018）。此前"写入价值归零"的判断
   系把基座单 seed 轨迹差（no_write 0.6375 vs a0_sg last 0.6483 = −0.011，
   噪声尺度内偏深）错算到写入头上；a3sgt−a0_sg = 轨迹 −0.011 + 写入 +0.006 ≈
   −0.005，分解闭合。
2. **授权不是主要缺口**：完美授权余量 gt_auth−p0 = +0.0028，CI [−0.0003,+0.0050]
   跨零（按 logit-0 符号定义的授权；边界感知 gate 未测但按 3 为次要）。
3. **主要缺口是写入目标标定**：该模型部署二值化阈值为 `mask_thr_binary=0.4`
   （logit 边界 **−0.4055**，非 0）。锚在 logit 0 的 margin 写（gt_cap、以及
   历史上的 A5 crossing loss 与 v1 BCE 的决策面）与部署决策规则错配——gt_cap
   实测删不掉假阳性区域、把 56% 正确背景点拉入掩膜（净增 39.5k px）。阈值锚定
   + 仅错误侧写的 gt_cap_thr 达 **+0.0267（CI 下界 +0.0206）**（0.6375→0.6642，
   AP75 +0.0508；flip 7222、destroy 0、写入 18287/45696——措辞按审计：
   "边界+margin 目标的错误侧"点，非"错误点"）。这是**固定约束下已展示的
   oracle 可达值（oracle 价值的下界）**，不是全策略上界。
4. 含义：0.6642 已超 a0_sg last（0.6483）与 a3sg last（0.6605），且 bbox 经
   detector hash 不变量结构性不动——**tailstop 容器内，写入策略的可行价值空间
   为 +0.006（已学）→ +0.027（阈值锚定 oracle）**。谱系不关账。
5. 证据导出的最小改动方向（未实施，需立项）：点损失从 BCE 换为**部署阈值锚定
   的单侧 margin**（target = logit(0.4/0.6) ± margin，正确侧恒零损失），在
   tailstop 容器上训练（A5 的失稳通道已被关闭；其 margin 锚在 0 的缺陷由本探针
   定位）。预注册门建议：segm ≥ a0_sg+0.005 且 bbox ±0.005（沿用 a3sgt 门）。

工件：`diagnostics/a3sgt_tail_write_oracle_full_20260914/`（首轮 6 cell + 4
bootstrap）、`diagnostics/a3sgt_tail_write_oracle_thr_20260914/`（含 gt_cap_thr
的 7 cell 复跑——五个共享 cell 与首轮逐位一致——+ 3 bootstrap）。

**A3 双 seed 同口径对照（复现细节，2026-09-11 21:08 推理批）**：
bestS：0.6930(E45)/0.6808(E75)（差 −0.012，尖峰换位）；last：0.6696/0.6659（−0.0037）；
last 的 AP75 0.7295/0.7254（−0.004，AP75 集中机制复现）；bbox bestS 0.6508/0.6936（seed44
的尖峰轮 bbox 异常低，seed45 正常——尖峰轮的 bbox 波动本身不可复现，进一步支持选择伪影解释）。

**矩阵终版（平台 E271–300 segm / bbox，2026-09-11 定版，含 R3 与 A3 双 seed）**：

| 臂 | segm | bbox |
|---|---|---|
| P | 0.6258 | 0.7161 |
| A0（PBM） | 0.6342 | 0.7208 |
| PB | 0.6395 | 0.7216 |
| R3（PB+渲染器画布） | 0.6262 | 0.7101 |
| A3 seed44 | 0.6689±0.0010 | 0.7190 |
| **A3 seed45** | **0.6661±0.0005** | 0.7048 |

A3 双 seed：segm 0.6675±0.0014（复现差 −0.0028，远小于 0.0083 跨 seed 标尺），对 PB
+0.0266/+0.0294（3.2~3.5× 宣称线）——头牌复现成立；bbox 跨 seed 摆幅较大（0.7190→0.7048，
−0.014），双 seed 均值 0.7119。R3 为 dense 第二实现的全量阴性（−0.013 vs PB），详见
r3_canvas_oracle_matrix300.md 的结案章。

**四级矩阵平台读数（E271–300 segm / bbox，2026-09-10 定版）**：
P 0.6258/0.7161 < A0 0.6342/0.7208 < PB 0.6395/0.7216 **< A3 0.6689±0.0010/0.7190**。

要点：(1) 推理与训练验证一致（bestS 自检 ✓，A0 bestC=bestB 同轮 E265 数值逐位相同 ✓）；
(2) **A3（PBM+decoder-tail）对 A0 +0.0347、对 PB +0.0294（last 口径 +0.0365）**——远超
0.008 宣称线，是全矩阵最大单模块增益；bbox 全臂落在 0.716–0.722 持平带（A3 −0.002 相对
A0，噪声内）；(3) **A3 的增益集中在 AP75**：last 口径 s_75 0.7295 vs A0 0.6671（**+0.062**）、
vs PB 0.6855（+0.044）；bestC(E220) 的 bbox/AP75 0.8403 为全表最高——decoder-tail 与 box
token 同构（边界精度敏感），量级放大 3 倍；(4) A3 bestS(E45) 是极端错峰形态（segm 尖峰轮
bbox 掉到 0.6508），选择伪影的又一实证，论文主口径坚持 last；(5) P/PB 段原要点保留：
AP50 段臂间接近、AP75 段差为 mAP 差两倍；单臂内错峰依旧。

推理产物索引（每目录含 `metrics.json` / `predictions.json` / `run_manifest.json`，
同步协议 evidence 项；批量日志 `logs/ablations/matrix_ckpt_evals_v3.log` 与
`logs/ablations/a3a0_ckpt_evals_*.log`）：

```
/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/
├── vhr10_p2v2_matrix300_p_point_tr1.0_va1.0/inference_p_point_{best_model_epoch155,
│   best_bbox_model_epoch250,best_composite_model_epoch280,last_model_epoch300}_vhr10/
├── vhr10_p2v2_matrix300_pb_tr1.0_va1.0/inference_pb_{best_model_epoch155,
│   best_bbox_model_epoch235,best_composite_model_epoch155,last_model_epoch300}_vhr10/
├── vhr10_p2v2_matrix300_a0_pbm_d5b_tr1.0_va1.0/inference_a0_pbm_d5b_{best_model_epoch105,
│   best_bbox_model_epoch265,best_composite_model_epoch265,last_model_epoch300}_vhr10/
└── vhr10_p2v2_matrix300_a3_pbm_udprk64_tr1.0_va1.0/inference_a3_pbm_udprk64_{best_model_epoch45,
    best_bbox_model_epoch265,best_composite_model_epoch220,last_model_epoch300}_vhr10/

R3 同名目录下 `inference_pb_r3_{best_model_epoch70,best_bbox_model_epoch250,
best_composite_model_epoch290,last_model_epoch300}_vhr10/`（2026-09-11 链式批补齐，
链日志 `logs/ablations/chain_r3evals_v2_*.log`）。
A3 seed45 目录下 `inference_a3seed45_{best_model_epoch75,best_bbox_model_epoch270,
best_composite_model_epoch270,last_model_epoch300}_vhr10/`（2026-09-11 21:08 批，
evidence 协议义务项）。
```

训练侧遥测（239 个对齐 epoch，同采样量同监督）：
`debug_final_mask_roi_dice` 差 ±0.0005（零）；`loss_mask` 差 −0.001~−0.004，
大头在早期（E31–60 的 −0.0040）。**训练时（正例多为 GT/高 IoU 框）box 不改善 mask 拟合**——
此时 box 信息与 RoI 裁剪完全冗余。

## 3. 因果验证：prompt_switch 探针（依赖性）

工件：`refactor_golden/prompt_switch_pb_best.json` +
`logs/ablations/prompt_switch_pb_gpu3_*.log`。权重：matrix300 PB `best_model_epoch155.pth`。

| 干预 | segm/mAP | Δ vs baseline |
|---|---|---|
| baseline（points+box，如训练） | 0.6530 | —（与 E155 训练验证值逐位一致，口径自检 ✓） |
| **drop_box**（推理时置空 box token） | **0.6173** | **−0.0357** |
| drop_dense | 0.6530 | 0（PB 无 dense 通路，阴性对照 ✓） |
| drop_both | 0.6173 | = drop_box（box 为唯一附加模态 ✓） |

顺序关系：P（0.6412，协同适应的无 box 模型）**>** PB-drop_box（0.6173，被抽走输入的带 box
模型）——补偿路径学出的效果优于"跛脚的完整模型"，是"信息大部分可替代"的直接印迹。

平行证据（WHU，D6 探针，dev100 前身）：drop_box **−0.1139**、drop_dense −0.0007——
"强条件信号被重度依赖"的模式跨数据集复现。

## 4. 机制解释

两条携带相同几何信息的路径：
1. **裁剪内容推断路**（始终可用）：`mask_roi_extractor` RoIAlign 按框裁出特征窗口 = decoder
   的 image embedding；窗口边界即 box 几何，窗口内前景/背景内容暴露物体在窗内的真实位置与
   范围。decoder 由内容推断 extent。
2. **box token 显式路**（仅 PB）：SAM2 PE `_embed_boxes` 生成 2 个角 token（专属类型嵌入
   `point_embeddings[2]/[3]`），把同一几何零成本显式给出。

训练时梯度流向最省力的证据源：PB 的 decoder 把几何锚定外包给 box token（训练遥测平坦、
正例为 GT 框时两路完全冗余），裁剪内容推断路训练不足（"没练满"）；P 臂没有显式路，被迫把
内容推断路练满（300ep 足够）。推理时全部 mask 从 **RPN 框**（非 GT）解码，box token 的
"窗口内确切几何"才显出不可替代的价值——这解释了训练遥测平坦与推理干预大掉并存。

## 5. 方案：训练期 box token dropout——**已被否决（2026-09-11，预登记判罚）**

否决依据：[a0_prompt_dependence_dense_vs_box.md](a0_prompt_dependence_dense_vs_box.md)
（A0 best/last 双权重四模式探针 + 三个 image-paired CI 全不跨零）。落判：
E300 drop_dense = −0.0175（CI [−0.0306, −0.0046]）落在预登记"−0.01 量级"分支——
**dense 瓶颈在内容侧不在用量侧**，"dropout 解锁用量"路径证伪；E105 的 drop_dense
−0.0639（2× drop_box）进一步证明用量早已拉满。dropout 若将来再做，只能以独立的
提示鲁棒性实验立项，不得挂 dense 叙事。本节原文保留仅为历史动机记录。

**设计**：对每个正 RoI 以概率 p（建议起点 0.2，扫 {0.1, 0.2, 0.3}）在**构造 sparse prompt 时
不拼接该 RoI 的 box 角 token**（输入层丢弃，无新参数；实现位置：mask head 的
`explicit_use_box_prompt` 分支按 RoI 采样；与 PE 冻结、fi 契约、分片验证均无交互）。
推理时 box token 照常使用（或两口径都评）。

**预期**：
- decoder 双路同时激活 → 推理带 box 的绝对值 ≥ 现 PB；上限来自"裁剪路练满 + box 零成本显式"的互补；
- **机制验收指标（预登记）**：训练后的 drop_box 退化应从 −0.0357 显著收窄（趋近 0 = 双路等价练满；
  收窄到 −0.01 量级 = 部分练满）。该指标本身即 dropout 是否生效的遥测。

**验证协议**：dev100 短程 A/B（pb vs pb+boxdrop，4.5h/臂）→ 平台期差 ≥ +0.005 判正（本机
同轨重复噪声 ~0.003–0.005）；判正后升 matrix300 入正式矩阵。风险：drop 掉 box 的 RoI 训练
信号变难 → 损失上升/收敛变慢（dev100 可观察）；对矩阵单调性叙事无破坏（新行为 PB 的增强行）。

**纪律提醒**（本日校准结论）：一切评测与调参判断必须在本机做相对差——跨机（机器 B）
同臂系统偏移实测 +0.016~0.021，会淹没 +0.01 量级的臂间效应。

## 6. 工件索引

| 项 | 路径 |
|---|---|
| P/PB 训练日志 | `logs/ablations/vhr10_p2v2_matrix300_{p_point,pb}_tr1.0_va1.0_20260909_*.log` |
| PB best 权重 | `ablations/vhr10_p2v2_matrix300_pb_tr1.0_va1.0/best_model_epoch155.pth` |
| 干预探针 | `inference/probes/prompt_switch_probe.py`；输出 `refactor_golden/prompt_switch_pb_best.json` |
| WHU 平行证据 | D6 探针（drop_box −0.1139，见 p2_dense 系列记录） |
| 机器偏移校准 | 本地 A2 0.6420/0.6513 vs 机器 B A2 0.6578/0.6718（平台/best） |
| 裁剪路代码位 | `rsprompter/sam2_mask_head_helpers.py`（sparse 构造）；`sam2/prompt_encoder.py:123`（`_embed_boxes`） |

## 7. 判读注意

- PB 的 +0.010 贡献量级与 E271–300 平台终判待回填（本文 §2 空位）；若平台差落在
  ±0.005 内，"小而真实"降级为"边缘"，但依赖性证据（−0.036）不受影响，方案 §5 依然成立。
- drop_box 是分布偏移干预（模型没见过无 box 输入），−0.036 是依赖性上界口径，不是信息量。
- 全部结论基于单 seed 单跑 + 干预探针；机内噪声带引用 D5-B 复跑（best 差 0.0045）与
  本地 A2 复跑（平台 ±0.0017 段内 std）。
