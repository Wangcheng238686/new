# D3/D4 设计文档 v3：P2 组合升级与画布符号置信变换（待审核）

状态：**设计稿 v3，未实施**。v1 因 Gaussian 语义/接线/混杂/判据错误被否；
v2 的 csig 数学解释、机制迁移表述、决策树、机制门槛与证据表述经审计修正，
见附录 B。分支 `machine2/fast-whu150`；工作区含未提交修复（runner 以
git_dirty + diff_sha16 溯源）。

---

## 0. 背景与目标（审计后的谨慎表述）

四臂 full_image 新协议（10%/15ep dev）+ 旧契约四臂已完成：

| 臂 | 新契约 best | 旧契约同角色 |
|---|---|---|
| D1fi 对照 (Point+Box) | 0.6144 | 0.5814 |
| D1fi Mask (+dense) | 0.6090 | 0.5824 |
| D2fi Mask | 0.6133 | 0.5768 |
| D2fi Full (+P2) | 0.6069 | 0.5832 |

**证据表述边界（不可越界使用）**：

1. **"+0.03"是协议组合效果**：新协议同时改变 final-mask 坐标
   （roi_local→full_image）与 loss（standard→roi_balanced_dice）。
   完整 2×2 中 `roi_local+roi_balanced_dice` 被 guard 禁止（缺一角）；
   `full_image+standard` 一角**可以跑但因历史塌缩与成本未排期**——
   因此坐标单因归因是"未做"而非"不可做"。当前只能表述为：
   协议组合（坐标+loss）整体 +0.024~+0.037。
2. **dense 无增益是现象，冗余是假说**。两协议下 dense 均无正增益
   （−0.0054 / +0.001）；但点是有损摘要，画布仍可能携带轮廓、置信度、
   拓扑信息，冗余机制未被证明。
3. **噪声水平未定量化**：重复对仅各一对（best 差 0.0043~0.0056）；
   两 Mask 臂 ep6 相差 0.0608，中期轨迹差异大。±0.01 量级结论不可
   凭单次运行。
4. **点位 oracle CI 的范围**：+0.0059，CI95 [+0.0032, +0.0065]，度量
   旧 D2 checkpoint 上 GT 质量点替换的收益——是点质量价值的上限性
   证据，**不是** fi 协议任何模块的增益，不支持任何模块"带 +0.006
   进矩阵"。图像 bootstrap 不能替代训练重复，也不能消除同一验证集
   反复选型的偏差。
5. **跨协议符号翻转（P2：+0.0064→−0.0064；dense：+0.001→−0.0054）
   不能证明噪声主导**——也可能存在协议×模块交互效应；两模块的效应
   量级均处于未定量的不确定范围内。

用户目标：矩阵四行单调（Mask 行须有真增益）。v3 提供两个旋钮级
（不加模块）dev 臂：**D3-combo**（P2 包络+监督联合升级，组合配方、
无归因力）与 **D4-csig**（画布变换为有界符号置信表示，单配置因素）。

---

## 1. 证据基线

- P2 遥测（D2fi Full，ep15 终值）：`saturated_in_support=10.30%`
  （总体上升：中期 0→3.3→6.1→9.65→10.70，后期 10.70→10.52→10.30
  非单调）；`delta_abs=0.0125`；`refinement gain: dice −0.000137,
  boundary_f1 +0.000239`（全程 ≈0，P2 未真正改善 coarse 边界）。
- 配对差：dense = **−0.0054**；P2 = **−0.0064**（旧契约 +0.001 /
  +0.0064；符号翻转的解释见第 0 节第 5 条）。
- 残差公式（`p2_boundary_refiner.py:202-203`）：
  `delta = beta × delta_logit_max × tanh(head) × search_support`。
- **辅助 loss 的梯度边界**（`:245`）：`refined_for_boundary =
  raw_logits.detach() + delta_logits`——辅助梯度**只进 P2 自身参数**
  （经 delta_logits），不直接回传 coarse 头；coarse 只受最终任务路径
  的间接影响。

---

## 2. D3-combo：P2 包络 + 监督联合升级（组合配方，无归因力）

### 2.1 改动

| 旋钮 | 现值 | D3 值 | 倍数 |
|---|---|---|---|
| `P2_BOUNDARY_REFINER_BETA` | 0.10 | 0.30 | ┐ 包络 0.05→0.30（6×） |
| `P2_BOUNDARY_REFINER_DELTA_LOGIT_MAX` | 0.50 | 1.00 | ┘ |
| `P2_BOUNDARY_REFINER_LOSS_WEIGHT` | 0.05 | 0.20 | 监督 4× |

- **组合配方声明**：近初始化处 **P2 头自身的辅助梯度**理论上放大约
  6×4=24×（受非线性与全局裁剪约束）；该放大**不直接回传 coarse 头**
  （见第 1 节梯度边界），对 coarse 的影响只经最终任务路径间接发生。
  本臂**不能区分**"幅度不够"还是"监督不足"——胜则作配方采用，
  负则两因皆弃（归因留待 2.5 可选隔离臂）。
- **支持域警示**：帽子上不改变 search band（2/4）与 coverage gate；
  带外错误与被拒 ROI 不因升级获得修复能力。

### 2.2 协议与入口

单臂 Full；对照复用 D2fi Mask 0.6133 与保守 Full 0.6069。协议 = D 系列
dev 同款（2 GPU、batch1×accum4、10%/10%、seed44、15ep、val@3、EMA off、
full_image+roi_balanced_dice）。wrapper：
`scripts/ablations/whu_fi_d3_p2esc_10p_2gpu.sh`，
`RUN_TAG=whu_d3fi_full_p2esc030_d100_10p_2gpu`，GPU 2,3。

### 2.3 判据

> 饱和度（含新帽下的饱和）**只是描述性上下文**：输出逼近上限不说明
> 修正方向正确；新帽下不饱和也不能反推旧帽是否够用（模型实际需要
> 0.15 时在 0.30 帽下不饱和，恰说明旧帽不足）。饱和度不参与判定。

- **机制判据**："是否稳定改善边界"——refinement gain 的 boundary_f1
  与 dice 在后半程（ep9-15）**持续、一致地为正**；**+0.01 为期望目标
  （非硬门槛）**：达到即"边界机制明确"；未达 +0.01 但持续为正 →
  记"弱机制信号"；不升反降 → 机制不成立。
- **效果判据**：best mAP vs 0.6133（对照）与 0.6069（保守 Full）。
- **效果改善而机制目标未达**的标签：**"效果改善、边界机制解释不足"**
  ——不是"配方无效"。
- **残差分布观察**：现有日志只有 `delta_abs`（均值型）字段，**无
  分位数遥测**；|delta| 分位数需离线探针（一次性脚本：加载 checkpoint
  对 val 若干 batch 前向记录 |delta| 分布，对照旧帽 0.05）——实施时
  作为小工具补齐，不得写成现成日志字段。

### 2.4 风险

- 点从 **refined** logits 挖出：学坏的大帽子同时污染画布与点（前向
  路径，与梯度无关）。
- 影响面仅启用 P2 的 Full 行。

---

## 3. D4-csig：画布有界符号置信变换（单配置因素实验）

### 3.1 表示的数学事实（经前向+梯度实测核对）

`transform_coarse_prompt`（`dense_prompt_utils.py:47-54`）默认
gamma=strength=1.0 时 `prompt = s·|s|`，`s = 2·sigmoid(logit)−1`：

| 输入 logit | 输出 | 对输入梯度 |
|---|---|---|
| 0 | 0 | 0 |
| 0.1 | 0.0025 | 0.0498 |
| 1 | 0.2136 | 0.3634 |
| 4 | 0.9293 | 0.0681 |

- 该表示**相对压低低置信（近边界）区域的幅值**，高置信区绝对输出
  仍更大且有界（±1）。
- **零交叉不是新性质**：raw logits 的零点本来也对应概率 0.5。
- **梯度事实**：csig 在零点梯度为 0、logit=0.1 处约 0.05；raw 在未
  clamp 时梯度为 1。即变换同时改变**前向表示**与**回传梯度的空间
  分布**——边界附近的任务梯度可能显著衰减。

### 3.2 假说（按上述事实重写）

**检验"有界、置信度加权的非线性表示是否更适合冻结的 PromptEncoder"**，
对照未校准灰度 logit 场。**不得预先称为"边界增强"**——它也可能把本就
弱的边界信号进一步压小（该风险显式列入判读）。若画布信息本身冗余，
任何表示变换都不会转正——D4 检验表示形式，不创造信息。
detach 保持 0（无 gaussian 的强制约束、无梯度通路截断），但如上所述
"detach 不变"**不等于**梯度效应不变。

### 3.3 协议、入口与接线修正

- **wrapper 修正**：`whu_p2_matrix_common.sh:12` 硬编码
  `raw_logits`，实施时改为
  `export SHAPE_DENSE_TRANSFORM="${SHAPE_DENSE_TRANSFORM:-raw_logits}"`。
- **架构 ID**：confidence_signed 落入 known map，与现 Mask 行同 ID
  `r1_c4_pafpn_coarse_points_box_dense_emb64`（fingerprint 不同；
  ID 复用按 a0 先例，靠 fingerprint+run tag 区分）。
- wrapper：`scripts/ablations/whu_fi_d4_csig_10p_2gpu.sh`，
  `RUN_TAG=whu_d4fi_mask_csig_10p_2gpu`，GPU 0,1。
- **DRY_RUN 断言清单（逐行 grep）**：`shape_dense_transform=
  confidence_signed`、`shape_dense_detach=0`、
  `final_mask_coordinate_mode=full_image`、
  `final_mask_loss_mode=roi_balanced_dice`、
  `ablation_id=r1_c4_pafpn_coarse_points_box_dense_emb64`、
  `run_tag=whu_d4fi_mask_csig_10p_2gpu`；PREFLIGHT 构建后断言
  `head.dense_prompt_cfg.transform == "confidence_signed"` 且
  `detach_input == False`。

### 3.4 判据

- 主判据：best mAP vs 对照 0.6144 与 D2fi Mask 0.6133。
- **best > 0.6144**：候选配方（dense 转正信号）；**= 持平（含
  0.6144）**：中性，不称转正。
- 机制旁证：`DENSE/canvas_*` 分布、`applied_delta_ratio`、gate
  `residual_alpha`；**梯度旁证**（因表示改变回传梯度分布）：
  dense 参数组的 final-mask 梯度范数与参数更新量（prompt pathway
  行）须与 raw 版本对比——边界附近梯度衰减是否伤害 dense 通路学习。

### 3.5 风险

- 表示不创造信息：若冗余假说为真则不涨（检验本身）。
- 边界附近梯度衰减可能使 dense 通路学习变慢甚至受损。
- 首轮固定默认 gamma/strength=1.0，不扫参。

---

## 4. 矩阵归一化约束

- 若 D4-csig 胜出并复跑确认：**Full 行必须在相同 csig 配置下重跑**，
  不得以"csig Mask + raw Full"构成只消融 P2 的相邻行。
- 表示变换同时改变前向与反向：Full 重跑时**不能只复核 refinement
  gain**，须同时复核 P2 的 final-mask 梯度、参数更新量、refinement
  gain 与最终 AP；D3 在 raw 下的学习行为**不保证**在 csig 下复现，
  v2 的"机制结论可迁移"表述删除。
- 若未来任何臂启用 detach=1：final-mask→dense→P2 的任务梯度被切断，
  一切 raw+detach0 的机制结论不可迁移。

---

## 5. 决策树（两阶段：单次→候选；复跑→采用）

统一动作语义：**单次胜出 = 候选配方；同配置关键配对复跑确认 = 正式
采用**。受投稿时间约束可以停止投入失败方向，但**"停止投入"与
"证明设计无效"是两个结论**，后者需要复跑或 CI。

| 观测 | 动作 |
|---|---|
| D3 机制稳定改善（后半程 gain 持续为正）且 best > 0.6133 | 候选配方；同配置复跑确认后，矩阵 Full 行采用；（可选）envelope-only 隔离臂归因 |
| D3 "效果改善、机制解释不足"（AP 升、gain 未达期望目标） | 标注该标签；是否进矩阵=用户裁决（效果可用但机制故事不完整） |
| D3 gain 不升反降 / best < 0.605 | 停止投入该配方；**不等于证明 P2 设计无效**。矩阵 P2 行默认**不进**（当前证据 −0.0064，保守 P2 无稳定收益证据），保留与否=用户裁决 |
| D4 best > 0.6144 | 候选配方；同配置复跑确认后 Mask 行采用 csig；Full 行按第 4 节同配置重跑 |
| D4 = 持平（含 0.6144）或 ∈ (0.609, 0.6144) | 中性：dense 转正未成立（复跑/CI 可选）；用户裁决 (a) 矩阵缩行 (b) 学习型画布模块大动作 (c) 作为冗余/中性证据行 |
| D4 < 0.609 | 该配置本次未获益；停止投入 csig（非证伪）；Mask 行维持 raw_logits |

---

## 6. 实施清单（审核通过后执行）

1. `whu_p2_matrix_common.sh:12` 改可覆盖（`:-raw_logits`）。
2. 写 `whu_fi_d3_p2esc_10p_2gpu.sh`、`whu_fi_d4_csig_10p_2gpu.sh`
   （overlay + 行 wrapper；D4 无需预设 ABLATION_ID）。
3. `bash -n` + DRY_RUN（按 3.3 断言清单逐行核验）+ PREFLIGHT_MODEL
   （构建后断言 transform/detach）。
4. D4 挂 0,1；D3 挂 2,3（均空闲），并行，各 ~2.5h。
5. |delta| 分位数离线探针小工具（见 2.3）——实施时补齐，不冒充日志字段。
6. 完成后出对比表（结论按第 5 节两阶段语义），更新架构/README 文档，
   给矩阵建议。

---

## 附录 A：v1 → v2 更正（摘要）

Gaussian 语义废弃；common 覆盖修正；D3 标注组合配方；饱和判据撤回；
detach/矩阵归一化约束新增；冗余/归因/噪声/oracle 表述降级。

## 附录 B：v2 → v3 更正（本轮审计对照）

| 审计项 | v2 问题 | v3 更正 |
|---|---|---|
| csig 数学解释 | "抑制高置信区""零交叉是新边界性质" | 实测表更正：压低低置信幅值、高置信区输出仍更大；零交叉非新性质；假说重写为"有界置信加权表示是否更适合冻结 PE"（§3.1-3.2） |
| detach 不变≠梯度不变 | "D3 机制结论原则上可迁移" | 删除；写明 csig 同时改变前向与反向（零点梯度 0）；Full 重跑复核清单扩为梯度/更新/gain/AP（§3.1、§4） |
| 决策树 | 持平称转正；D3 失败→保守 P2 进矩阵；<0.609 称证伪 | 两阶段动作语义统一（候选→复跑→采用）；P2 行默认不进矩阵；"停止投入"≠"证明无效"（§5） |
| 机制门槛 | boundary_f1 +0.01 硬门槛（≈末轮 42 倍）无依据 | +0.01 改期望目标；机制判据=后半程稳定改善；未达目标但 AP 升= "效果改善、机制解释不足"（§2.3） |
| \|delta\| 分位数 | 写成现成日志字段 | 明确需离线探针小工具，列入实施清单第 5 项（§2.3、§6） |
| 证据表述 | "三臂"未更新；饱和"单调"；符号翻转证噪声；"无法交叉消融"；24× 回传 coarse | 四臂；总体上升非单调（后期回落）；交互效应可能；可跑未跑≠不可区分；24× 仅 P2 头自身（aux loss 用 raw.detach()）（§0、§1、§2.1、§2.4） |
