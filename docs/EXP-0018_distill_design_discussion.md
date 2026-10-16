# EXP-0018 设计讨论纪要：训练期跨视角蒸馏臂（两轮独立评审后定稿）

2026-10-16。主笔（会话 A）与独立评审子代理（会话 B，读过全部代码/证据/
登记表）两轮对抗性讨论后的合成设计。**本文件是预注册草案的输入，不是
预注册本身**；立项需按治理规程另走四要素预注册 → P0 探针 → 审计 → 拉起。

## 0. 编号与谱系（B 的否决级修正，已核实）

EXP-0015/0016/0017 已被并行会话发号（0015 持平、0016 HARM、0017 待审
计放行）。本臂 = **EXP-0018**，臂名候选 `SURFBEV-snap512distill-sealed
-DDP4-v2-noaug`，路由 = 基座/训练动力学型（B 型：训练期损失，冻结基座
无从验证）。A/B 锚点 EXP-0009（逐场 0.7764/0.7533）。

**动机链（修正后）**：0015 分解显示 on−off 双显著（segm +0.38 [+0.06,+0.70]、
bbox +0.34 [+0.02,+0.67]）但 off−single 训练代价（−0.35/−0.58）精确抵消
收益——UAV 信号可被训练期利用，但推理期双架构的联合训练代价吃掉净收益。
蒸馏臂把 UAV 用法整体移入训练期（零推理期代价、零推理期依赖），是对该
分解的直接回应。注意引用边界（B 修正）：0015 证明的是 eval 期条件性使用，
不是"读出 NULL 信号产生训练期增益"；冻结读出边际对训练期可用性既不必要
也不充分——这正支撑 P0 门的设计（见 §2）。

## 1. 机制分析（B 的贡献，定稿采纳）

L1 蒸馏不动点 = 条件中位数 m(x) = med(T̂|x)。教师分解
T̂ = μ_spatial(cell) + ε(x) + ν：
- ν（输入无关残差）：条件期望为零，只贡献梯度噪声与 clip 预算占用；
- μ_spatial（数据集空间模板）：固定偏置拉力 = 中心先验通道（SCVT4 泄漏
  同构）；sham90 臂同样可学，差分为零；
- **ε(x)（输入一致的场景特异内容）= 内容特异性塑形价值的全部来源**。

推论：(i) 自蒸馏式价值不被 §2 的门排除（其教师天然纯 ε）；(ii) 任何正
结果必须过 sham90 差分才能主张"UAV 信息增益"（正则/模板两条替代通道
均 sham-可复现）。

## 2. P0 门体系 v2（两轮辩论的核心产出）

- **G1 结构门（地板，杀纯噪声）**：S = E_val|T̂−μ_spatial| / E_val|T̂−c_med|
  < τ₁ = min(0.7, S_height+0.1)。A 原门 (i) 有命名反转（S 越小越模板化，
  杀噪声放模板），降级为地板；纯模板情形由 G2 差分防。
- **G2f 冻结差分门（廉价前置）**：predictor=0009 末代 P3（512d/cell，
  ridge、5 折 CV 选 λ），predictand=白化教师；统计量 val R²(true)−R²(sham)
  scene-paired bootstrap（10k、seed 20261014）CI 下界>0。承认这是读出的
  配对差分形式（A 原门 (ii) 的实质），作为前置而非权威。
- **G2t 可学性差分门（权威 kill，新增 P0-d）**：从 0009 last checkpoint
  出发、任务损失置 0、仅 L_distill（λ=1），真/sham 两臂各训 5 epoch
  （单臂 ≈22 min，合计 <1 h，探针级脚本、voided-grade）；kill 量 =
  ΔR²(P3→T̂, ep0→ep5)_true − ΔR²_sham 的 CI 下界 >0.01。语义：表示在
  真实训练机制下能否向教师移动且超过 sham。G2f 不过但 G2t 过 → 放行
  （这正是"冻结读出测不出、训练能学出"论点的兑现口）；G2t 不过 → kill
  该教师。附带产出：E1 的经验可学底线（真臂 loss 平台）。
- **G3 教师路由器**：误差轴 (SAT+TEACH)−SAT ΔAUC 配对比较 height vs
  descriptor；height 显著占优 → 主臂教师切归一化 zmean；两者皆 ≤0 →
  仍可启动，但登记"任务对齐先验低"预期 + §4 sham 升格条款。
- **kill 语义（A 提议、B 接受）**：门杀的是教师（descriptor 或 height），
  不是方向；"训练期蒸馏"死亡仅当两个教师都过不了 G2t。
- **如实摊开（B 的立场，A 接受）**：按现有证据（DEPTH +0.0213 vs P4
  描述子负边际），G3 大概率路由到**高度教师**——本讨论的"外观蒸馏"
  原案很可能降为备选。这与 DECOMP 方向决议 #1 的原排序一致。

## 3. 臂规格（定稿要点；细节以预注册为准）

- **教师**：冻结 SAM2 base_plus，`max(fpn[-4:], key=area)`（stride-4、
  256ch、128²）→ `rasterize_surfacebev` 单 bin 到 64×64；**只取 [:256]
  描述子**（cm 通道归高度臂、vcount/missing/visible 是飞行模式统计，
  蒸它们=注入位置先验）；**covered-cell 白化（μ,σ）重新 harvest**
  （train 694 场全量、JSON+sha256 登记；不复用 bev_scale——其为加性
  注入设计，含零 cell 偏置与 floor 截断）；mask M 从未定标 BEV 的
  missing 通道取（count>0）。
- **学生 g 头**：`Conv2d(512,256,1)`，131,328 参数（<0.02%，登记），
  **无归一化**（A 草案的 "LN" 是规格错误——nn.LayerNorm 在 [B,C,H,W]
  上沿 W 归一化，语义错；白化在目标侧完成）；kaiming 初始化（禁零初始
  化）；**wd 豁免**（扩 `FUSION_ROOTS` 加 `distill_g`，沿 0015 先例；
  否则复刻"门被 wd 拉回零"的静默关闭失败模式）；遥测 `g_weight_norm`。
- **损失**：`L_distill = λ·Σ_ij M_ij·mean_c|g(P3)_ij − T̂_ij| / mean(M)`，
  P3=x[1]（neck 输出，与注入谱系同格，探针直接复用）；逐 cell 等权首版；
  **λ 由公式定**：白化后初始 L≈λ，冒烟核验 loss_distill 占总损失份额
  ∈[2%,10%，超带则启动前一次性改 λ 并登记，不事后扫参；λ=0.1 起点。
  归约：覆盖 cell 均值 → DDP all-reduce 均值；AMP：教师光栅化+L1 全程
  fp32 岛，P3 出 autocast 后 `.float()`。
- **装配**：模型侧 `RSPrompterAnchorDistill(RSPrompterAnchor)`，
  `loss()` 内 `_fetch_uav()` + fp32 岛教师前向 + 追加 `loss_distill`
  进返回 dict——trainer 零改动（loops.py 对 loss dict 求和、loss_meter
  自动记录）；eval 期 predict 路径与单流逐位等价，UE_LOAD_UAV=1 时
  val 载入 UAV 但不消费（wrapper 契约 echo 钉死该选择）。
- **组合变量登记**（如实，不称"唯一变量"）：损失 + g + g 的 wd 豁免 +
  RNG 流偏移 + UE_LOAD_UAV；harness diff 证明键集恰为
  `{uav_encoder, distill_*}`。
- **训练协议**：与 0009/0010 逐项一致（v2 快照、seed 44、100 epoch、
  DDP4 batch1×accum2、lr 5e-4、wd 0.05（g 豁免）、warmup 100、无增强、
  amp 1、ema 0、禁早停、last 判定、test 不开封）。

## 4. 判定体系（含本臂专属参与度算子）

- 主指标：EXP-0018 − 0009 的 val segm/bbox AP 差（逐场 scene-paired
  bootstrap，10k、seed 20261014）。
- 参与度算子（对应注入臂的 open_eff）：
  **E1** 底线比 = 末段 loss_distill ÷ P0-d 真臂经验平台（fallback：
  解析常数图底线）；**E2** 对齐度 = val 覆盖 cell 的 mean cos(g(P3),T̂)，
  对初始化与 sham 臂各报；**E3** 可重建性迁移 = WHY-NULL 探针 A 协议在
  本臂末代 P3 上复测 R²（对照 0009 的 0.01）。
- 四判定：`DISTILL_GAIN`（双 CI 下界>0 且 ≥+0.5 点，且 sham90 臂不重现
  同幅度——否则降级 `REGULARIZER_GAIN`）；`DISTILL_ENGAGE_NULL`
  （无增益 且 E1≈1 且 E3 抬升<0.05）；`DISTILL_USED_BUT_NULL`（E1<1、
  E3≥0.05 但无增益）；`DISTILL_HARM`（任一 CI 上界<0，即停谱系叠加，
  禁同臂连续改参重跑）。
- **sham90 全量臂**：教师级光栅化旋转（确定性变换，非 RNG 事件），同
  seed 同 g 初始化、首梯度步起分叉；**升格为任何正结果主张前的强制
  前置**（G3 低先验路由下尤其如此）；若 GAIN 且 sham 不重现 → 两臂同批
  升第二 seed 复验。
- **小目标防护条款**：headline 过门但 AP_small 分层 CI 上界<0 且幅度
  ≥−1.0 → 判 `PARTIAL_GAIN_WITH_SMALL_HARM`，必须并列报告（SCVT4
  小建筑 −9.2 与 R0 62% 小实例不可见）。
- 期望登记（效应形态）：R0 包络下预期为 bbox 侧大建筑增益 + segm 侧
  小幅边界带增益；+0.5 双门处于包络边缘，单侧结果的解释空间预先关闭。

## 5. 因子表（预登记不并发）

F1 余弦损失（E1≈1 幅度塌缩时）；F2 边界带加权（E2 抬升但 headline 持平）；
F3 P2 接入点（128²，教师原生分辨率；小目标分层为负时）；F4 高度教师
（G3 路由触发时升主臂）；F5 冻结正交 Q 投影（容量混杂排除）；F6 教师
stride-8 层；F7 延迟启动（早期爬坡受损签名时）。

## 6. 排程与预算

- 0017 未收线前不抢四卡；P0 探针（只读 + P0-d 短训 <1h）绑最空闲单卡
  并行，约定"探针 OOM 影响 run 则杀探针不杀 run"；P0 产物 voided-grade
  探针区，不进正式 checkpoint 目录。
- 全量臂（若 P0 过门）：约 7–8 h（参照 0014 节奏）；sham 臂条件执行。

## 7. 讨论分歧记录（供审计追溯）

- A 初稿错误（B 抓出）：编号冲突、教师通道未闭合、bev_scale 复用不当、
  LN 规格错误、λ 无量纲、判定缺参与度算子、缺 sham 触发条款与小目标
  条款、"唯一变量"表述失实。
- A 门 (i) 命名反转与 sham 同值问题、A 对 0015 的过度引用（B 修正）。
- 唯一实质分歧（P0 门语义 kill vs re-route）：第二轮收敛为 §2 的 v2
  门体系；B 的 L1 条件中位数论证为收敛提供了机制基础。
