# Prompt 消费鲁棒化与 P2 边界提示联合改造：现状、决策门与实施指南

日期：2026-09-04
状态：历史候选与当前短程开发决策并存的指导文档；不代表下述新模块已经落地或获得指标收益。
适用主线：`portable_sam2_explicit_coarse/`，当前分支 `machine2/fast-whu150`。

## 1. 文档目的

本指南把两个互相依赖的问题收敛为一条可验证的实施路线：

1. **Prompt 消费鲁棒化**：让可训练的 SAM2 MaskDecoder 对同语义、同 token 数的点位变化
   不再产生分布外负响应，并能利用指向真实 FP/FN 区域的纠错提示；
2. **P2 边界提示改进**：利用 stride-4 P2 信息改进 coarse 派生的边界提示，重点改善
   `N1` 外环负点，而不是继续依赖已被证明几乎无效的 dense/logit 微扰通路。

最终目标不是让某个辅助 loss 下降，而是建立可观测的因果链：

```text
P2/边界信息变好
→ refined point 真实移动且语义正确
→ MaskDecoder 正确消费该移动
→ P2 on 相对 P2 off 的最终 segm 指标稳定提高
```

任何阶段只证明链条中的一段，都不能宣称“P2 角色复活”。

## 当前执行决定（2026-09-04，优先级最高）

本轮目标已收敛为：**在不新增主模块的前提下，先让现有的 Point、Box、dense prompt 与
P2BoundaryRefiner 按四行消融矩阵产生预期的逐步净收益；随后才冻结设计并进行正式实验。**

```text
Point → Point + Box → Point + Box + Mask → Point + Box + Mask + P2 (Full)
```

这不是要求在一次短程试验中“碰巧”得到递增 mAP，而是要求每一项新增信息在进入正式矩阵前
已经通过独立、受控的开发验证。旧 C5-v2 WHU full-150 的结论是：dense 与 P2BR 都有实际前向
扰动，但 dense 没有显示净指标收益，且 P2 refined coarse 不优于 raw coarse；此外该旧实验使用
`SHAPE_DENSE_DETACH=1`。因此它不能作为新 Full（`SHAPE_DENSE_DETACH=0`）的最终裁决，
但足以否定“直接启动四行 150 epoch 正式消融”的前提。

### D1：先使 dense prompt 成为正贡献

只比较现有第二、第三行：

```text
Point + Box  vs.  Point + Box + Mask
```

固定当前实现、PAFPN、stride-16、ROI-local 坐标、SAM2 初始化、优化器、数据增强和 seed；dense
行必须使用已有的 raw-logit prompt 且 `SHAPE_DENSE_DETACH=0`。短程开发使用固定 WHU 图像级
10% train / 10% validation（`SUBSET_SEED=44`）、4 GPU AMP、15 epoch、每 3 epoch validation、
EMA 完全关闭。该阶段只允许调整**已有** dense 控制量，例如初始 dense gate、raw-logit temperature
和既有 loss 权重；不得新增 PromptEncoder、decoder 或新的 dense 分支。

通过条件是 dense 具有可复现的正向最终分割趋势，且日志同时确认 `applied_delta_ratio>0`。若 dense
不优于 Point+Box，则先使其注入更保守或更可校准；不得把无增益的 dense 行写入正式矩阵期待它
自行变好。选定的 dense 参数一经确定即冻结。

### D2：在 dense 已获益后再使 P2BR 成为正贡献

只比较已冻结的第三、第四行：

```text
Point + Box + Mask  vs.  Point + Box + Mask + P2 (Full)
```

二者共享 D1 冻结的全部参数与短程协议；Full 继续使用 `SHAPE_DENSE_DETACH=0`。该阶段只允许调整
现有 P2BR 控制量：`beta`、`delta_logit_max`、boundary auxiliary loss weight 及既有投影/中间通道
宽度；不得引入 P2PointRefiner、额外 token 或第二个 P2 分支。

Full 的最低机制验收是 raw→refined 的 coarse Dice、IoU 和 boundary-F1 不再系统性变差，P2 support/
delta 统计非零且不以大面积饱和为代价；最终分割趋势必须优于已冻结的 Mask 行。若失败，P2BR 不进入
正式论文主矩阵，而不是在矩阵中保留一个无效的“Full”。选定的 P2BR 参数随后冻结。

### 正式四行消融的进入门

只有 D1、D2 均通过，才从同一初始化独立启动 Point、Point+Box、Point+Box+Mask、Full 四行 WHU
全量 150 epoch 实验。四行除逐行增加的 prompt/P2 开关外不得改变任何参数或训练协议；这时预期的
递增结果才可归因于设计，而非在正式消融中调参。短程 10%/10% 仅用于开发选型，不作为论文最终
数字；正式结论仍以全量 WHU 的预注册协议为准。

本节覆盖本文件后续“新增 PromptRobustifier / P2PointRefiner”的近期实施优先级：这些内容保留为
历史候选和后续备选，当前不得据此新增模块或启动其训练脚本。

## 2. 已确认现状

### 2.1 当前有效主干

- C5-v2 使用 PAFPN、stride-16/64×64 SAM image embedding、显式 coarse、2P2N、box、
  dense prompt 和 P2BoundaryRefiner；
- PromptEncoder 从 SAM2 checkpoint 严格加载并冻结，但其前向保持可微；
- MaskDecoder、ShapePriorInjector 和 P2 refiner 可训练；
- refined coarse 是点挖掘和 dense canvas 的共同来源；
- 2P2N 坐标选择为 hard/stop-gradient，final-mask loss 不能通过离散坐标回传到 coarse；
- dense prompt 理论可微，但现有推理消融显示其对最终输出近乎无贡献。

### 2.2 当前 P2BoundaryRefiner 的真实行为

当前实现位于 `rsprompter/p2_boundary_refiner.py`：

- detached PAFPN P2 经 `1×1 + GN + GELU` 投影；
-用同一 proposal 做 aligned avg RoIAlign 到 `32×32`；
- 以 `roi_feature - avg3(roi_feature)` 作为高频特征；
- raw coarse 经平滑、0.5 阈值和形态学运算生成 hard search band；
- search band 覆盖率超过 50% 时整 ROI 关闭；
- residual 为 `0.20 × 2.0 × tanh(.)`，最终绝对上限 `0.4 logit`；
- boundary auxiliary 在 raw search support 内使用二值 BCE；
- GT 只进入 loss，不进入 forward support。

该实现的坐标、梯度隔离、训练/推理一致性和零初始化契约是正确的。问题是修改能力过弱：

- 已归档 coarse 指标中 raw/refined Dice 仅约第四位变化，boundary-F1 约第三位变化；
- WHU fast-150 同一 checkpoint 推理消融中，P2 on/off 的 segm/mAP 为
  `0.7492684 / 0.7492672`，可视为无差异；
- 已有带内 logit 修正 oracle 即使放宽到强修正，最终收益也只有约 `+0.0027 mAP`；
- dense prompt 关闭不降，说明 logit refinement 的可微下游通路没有被 MaskDecoder 有效消费。

因此，**不得直接把“扩大旧 P2-BRR 网络/残差”作为默认主路线**。它只保留为低成本机制
对照；主路线应优先改进 coarse 派生的点提示及其消费端。

### 2.3 当前 prompt jitter 的真实行为

提交 `dbb8772` 已加入：

- box jitter：概率 0.3、尺度 0.05，IoU≥0.5 回退保护；同一 jittered RoI 用于
  RoIAlign、prompt 和 target crop；
- point jitter：固定概率 0.5、半径 1 coarse cell，只扰动 P1/P2；
- 训练期启用、推理期关闭，默认配置仍为 off。

现实现能训练 box/正点局部稳定性，但不能支撑完整目标：

- 没有 epoch 退火；
- `(0,0)` offset 也计入 jitter 数；
- fallback 正点没有扰动后的显式语义检查；
- N1/N2 完全不动，而 P2 目标主要依赖 N1；
- E0/E1 追加 4P4N/4N 改变 token 数，与固定 2P2N 位移训练不匹配；
- 随机同标签扰动容易训练局部不变性，不保证“更正确的位置带来更好输出”。

### 2.4 两个目标必须分离

本轮只解决：

> 固定 2P2N 语义和 token 数下，训练消费端利用 coarse/P2 派生的同语义位置校准。

以下内容不进入本轮主实验：

- 追加 4P4N/4N 的二轮迭代提示；
- 可变 token 数训练；
- Gumbel 选点器；
- 对 hard argmax/形态学挖点做 STE 或软质心梯度手术；
- 用 GT 控制推理期 forward support。

E0/E1 只能作为未来“可变点数消费训练”的独立路线，不能作为本轮固定 2P2N 的主杀线。

## 3. 联合方案总览

本文件定义的是**带决策门的路线**，而不是预先批准的联合模块。Prompt 消费鲁棒化可先独立实施；
`P2PointRefiner` 只有通过第 10 节的目标域、真实-P2 相对伪-P2 信息门后，才允许接线进联合训练。
在此之前不得把该方案命名为 C5-v3，也不得在论文中宣称其有效。

```text
Mask RoI feature ───────────────→ ShapePriorInjector ─→ raw coarse 64×64
                                                        │
PAFPN P2 / 可选原生 stride-4 feature ─→ [条件式 P2PointRefiner] │
proposal RoI ────────────────────────────────────────────┤
                                                        ↓
                                   ShapePointMiner → base 2P2N
                                                        ↓
                         training-only PromptRobustifier/Teacher
                         ├─ clean
                         ├─ semantic local jitter
                         └─ FP/FN corrective replacement
                                                        ↓
                                         fixed-cardinality 2P2N
                                                        ↓
                         frozen PromptEncoder → trainable MaskDecoder → mask
```

核心原则：

1. **保持语言不变**：槽位始终为 P1、P2、N1、N2，标签和 token 数不变；
2. **GT 只作训练 teacher/合法性保护**，推理仍只用模型预测和 P2；
3. **先证明目标域 P2 含有超出 coarse 的信息，再证明 decoder 会消费，最后才看 mAP**；
4. **旧 P2 logit residual 默认关闭**，避免同时改变 coarse、dense 和点而无法归因；
5. 所有新行为进入 `cfg.model` 和 checkpoint 指纹，默认关闭以兼容旧 checkpoint。

## 4. 模块 A：语义感知 Prompt 消费训练

### 4.1 新配置契约

在 `mask_head` 中新增完整、可持久化的配置：

```python
prompt_robust_cfg=dict(
    enabled=False,
    mode="fixed_2p2n_semantic",
    clean_ratio=0.50,
    local_jitter_ratio=0.25,
    corrective_ratio=0.25,
    jitter_slots=("p1", "p2", "n1"),
    radius_cells=1,
    max_resample_attempts=8,
    semantic_guard="pred_and_gt",
    exclude_zero_offset=True,
    warmup_fraction=0.05,
    hold_end_fraction=0.60,
    anneal_end_fraction=0.95,
    start_strength=1.0,
    end_strength=0.10,
    final_clean_epochs=20,
)
```

要求：

- `enabled=False` 时数值和 RNG 消耗均保持旧路径；
- 构造函数严格拒绝未知 key、负概率、比例和不合法 schedule fraction；
- 三种 mode 的比例和为 1；
- 当前 optimizer update 与总 update 预算由已有训练状态显式传入，不读取全局环境状态；
- 影响 forward 的解析值必须保存在 checkpoint `model_config`。

### 4.2 GT target 前移但不泄漏推理

当前 coarse GT target 在 mask forward 后才生成。语义保护和纠错 teacher 需要在点编码前
看到与 jittered proposal 对齐的 ROI-local GT mask。

实施方式：

1. `models.py::mask_loss()` 在最终 `prompt_rois` 确定后，调用一次
   `mask_head.get_coarse_targets(..., prompt_pos_priors=prompt_pos_priors)`；
2. 通过 `_mask_forward(..., prompt_gt_masks=...)` 传到 MaskHead；
3. 训练时 `prompt_gt_masks` 为 `[N,1,64,64]`，推理始终为 `None`；
4. 后续 coarse loss 和 P2 auxiliary 复用同一 target，禁止重复 crop 产生坐标差异；
5. GT tensor 必须 detach，不允许成为可学习输入特征。

这里的 GT 只用于选择/验证训练提示，与常见的点击模拟训练相同；推理不读取 GT。

### 4.3 三种训练样本

每个 ROI 独立抽取一种模式，但同一 ROI 的四个槽位保持一致处理记录。

#### Clean（默认 50%）

P2 分支关闭时，完全使用 ShapePointMiner 原始 2P2N；P2 分支获批并开启时，使用模型生成的
`p2_refined_coords`，但不施加训练期 augmentation。该比例不能被退火到 0，用于保持干净推理口径。

#### Semantic local jitter（默认 25%）

- offset 从半径 1 的八邻域抽样，明确排除 `(0,0)`；
- P1/P2 合法条件：扰动点位于 GT foreground；若配置为 `pred_and_gt`，还须位于
  refined coarse foreground；
- N1 合法条件：位于 GT background，且仍处于 refined coarse 的外环候选区域；
- N2 本轮默认不抖动；若未来开启，必须保持在 GT background 与 safe background；
- 最多重采样 8 次，失败则回退原坐标并记录原因；
- 不允许仅做 canvas clamp 后继续使用语义翻转的点。

#### Corrective replacement（默认 25%）

根据 `refined_or_raw_coarse.detach()` 与 GT coarse target 构造：

```text
FN = GT foreground ∩ predicted background
FP = GT background ∩ predicted foreground
```

- 若 FP 边界区域非空，用其中的高置信/大平台候选替换 N1，标签仍为 0；
- 若 FN 内部区域非空，用其中的安全内部候选替换 P2，标签仍为 1；
- P1 保留稳定主体锚点，N2 保留远背景锚点；
- 候选要满足与其他槽位的最小距离；
- 找不到合法纠错候选时保持原点，不伪造点；
- teacher 选择全程 `no_grad`，不尝试穿过 hard selection 反传。

该模式与纯随机 jitter 的区别是：坐标变化与真实错误方向相关，MaskDecoder 若利用提示就能
降低最终 mask loss，从而训练“响应性”而非单纯“不变性”。

### 4.4 退火

以固定 update 预算的比例表示，避免 10% 子集因 epoch 变短而改变课程：

- 前 5% updates：clean warm-up；
- 5%–60%：按配置使用 50/25/25 混合；
- 60%–95%：local/corrective 总强度线性降至初始的 10%；
- 最后 5%：clean calibration，默认只保留 5% 语义扰动或完全关闭；
- validation、EMA validation 和独立推理始终关闭 augmentation。

退火建议作用于“选择 augmented mode 的概率”，不改变 1-cell 整数半径。快速验证的所有臂必须
使用同一个总 optimizer-update 预算；全量确认可复用同一比例，而不复制某个固定 epoch 数。

### 4.5 box jitter 定位

保留现有 box jitter 实现，但它是 proposal/crop augmentation，不作为“点消费已打开”的证据。
正式联合实验前必须先做 point-only 与 box-only 的短程归因。box jitter 不参与 corrective
teacher 的坐标定义歧义：GT target、RoI feature、coarse、点映射和 mask target 必须继续使用
同一个 jittered proposal。

## 5. 模块 B：P2 边界提示校准

### 5.1 默认决策：P2 点位校准被降级为条件路线

旧 logit 带内修正 oracle 的最终上限约 `+0.0027 mAP`；miner-on-GT 的同语义 2P2N oracle
约 `+0.0082~+0.0111`。后者只说明“若给出更好的点，decoder 有空间”，**并不证明 P2 能从
coarse 之外预测这些点**。现有跟踪实验 `logs/learn_probe/fit_report.json` 在 76,257 个样本上仅将
GT-miner 单点坐标误差从 21.335 px 降至 20.668 px（3.125%）；而单一 GT-miner 坐标也可能不是
唯一的有效点。因此当前证据不支持直接实现 P2PointRefiner。

默认优先级为：

1. 先独立实施并验证 `PromptRobustifier`；
2. 先完成第 10 节 Phase 0--1 的目标域 P2 信息探针；
3. 仅当真实 P2 相对所有伪 P2/无 P2 对照存在可靠增益时，才实现本节后续的 P2PointRefiner；
4. P2-BRR v2 不进入长训，也不以扩大网络替代信息证明。

### 5.2 P2PointRefiner 输入与输出

新增 `rsprompter/p2_point_refiner.py`：

```text
输入
  p2_feature     [B,C,H/4,W/4]，detach
  prompt_rois    [N,5]
  coarse_logits  [N,1,64,64]，detach 仅作 cue
  local_yx       [N,4,2]
  labels         [N,4]

输出
  offsets_image  [N,4,2]
  refined_coords [N,4,2]
  per-slot gate/confidence
```

推理语义：miner 决定点的角色，P2PointRefiner 只做有界校准，不增删点、不改 label、不交换槽位。

### 5.3 推荐结构

第一版控制复杂度：

1. detached P2 做 `1×1 Conv + GN + GELU`，512→64；
2. 对 proposal 做 aligned RoIAlign；候选为 `64×64` 直接对齐版和 `32×32` 兼容版，先用
   离线点误差选择，不凭直觉决定；
3. 在 miner local point 周围用 `grid_sample` 提取中心及 `3×3` 邻域特征；
4. 拼接以下 cue：
   - P2 point/patch feature；
   - coarse probability、uncertainty、Sobel dx/dy、局部梯度幅值；
   - 到 coarse boundary 的有符号/无符号距离；
   - slot/label embedding；
   - box width/height 与点的归一化 ROI 坐标；
5. slot-conditioned MLP 输出 `(dx,dy)` 与 gate；
6. 最后一层零初始化，初始严格 `offset=0`；
7. 图像像素位移上限：

   ```text
   max_offset = clip(0.10 × min(box_w, box_h), 1 px, 8 px)
   offset = gate × tanh(raw_offset) × max_offset
   ```

原设计的上限 16 px 对 WHU 小目标偏大，第一版先用 8 px；只有 oracle offset sweep 证明需要时
才放宽。

### 5.4 条件实施时的 teacher：候选集合和冻结 decoder 效用优先

禁止把同一 `ShapePointMiner` 的单个 GT 坐标当作唯一回归标签。条件实施时，先在 GT 定义的
合法候选集合中（P1/P2 前景内部、N1 真边界外的 FP 纠错环、N2 安全背景）采样同语义候选；再用
冻结的 PromptEncoder/MaskDecoder 对每个候选做配对前向，以 ROI mask loss/IoU 改善选出 teacher
集合或最小代价候选。GT miner 坐标只能作为候选之一和诊断指标。

仍可使用以下 GT logits 产生候选中心：

```python
gt_logits = where(gt_mask >= 0.5, +8.0, -8.0)
gt_local_yx, gt_labels = miner_teacher(gt_logits, ...)
```

匹配规则：

- 正点集合 `{P1,P2}` 内做 2×2 最小总代价匹配；
- 负点集合 `{N1,N2}` 不默认交换角色：N1 是边界外环，N2 是安全背景；优先按角色匹配，
  仅在 teacher 槽无效时回退；
- 无效 teacher slot 不计回归 loss；
- 预测 slot 无效时不凭空创建 token；
- GT 只产生训练期候选/监督，不进入推理 forward。

损失：

```text
L_set   = 到有效 teacher 候选集合的最小 SmoothL1
L_utility = 冻结 decoder 的候选效用蒸馏（仅在候选排序有间隔时）
L_sem   = 校准后 P/N 语义违规惩罚
L_mag   = 非纠错样本上的 offset L1 正则
L_total = w_set L_set + w_utility L_utility + 0.05 L_sem + 0.001 L_mag
```

`w_set` 与 `w_utility` 由冻结探针的尺度确定；初始权重只作实施默认，必须通过短程机制实验确认，
不得直接视为最终超参。每槽报告集合距离、候选效用和语义正确率，禁止只报告平均欧氏误差。

N1 可使用 1.5 倍回归权重，但必须同时报告每槽误差，防止总均值被 P1/P2 主导。

### 5.5 与 PromptRobustifier 的顺序

推荐顺序：

```text
ShapePointMiner base points
→ P2PointRefiner 产生模型预测的校准点
→ training-only PromptRobustifier 选择 clean/local/corrective
→ PromptEncoder
```

理由：

- P2PointRefiner 的 on/off 可以在 augmentation 前明确比较；
- corrective teacher 能覆盖 P2 尚未学会的错误样本；
- 后期退火后训练输入逐渐回到真实 P2PointRefiner 分布；
- 推理时只保留前两步，不存在 augmentation。

必须额外保留两个坐标：`base_coords` 与 `p2_refined_coords`，禁止 augmentation 覆盖后丢失
P2 自身统计。

### 5.6 旧 P2BoundaryRefiner 的处理

- 不删除旧实现和旧配置，保证 checkpoint 可重建；
- 新配置明确设置 `p2_boundary_refiner_cfg.enabled=False`；
- 禁止默认同时启用 logit refiner 与 point refiner；若用于研究，配置校验要求显式
  `allow_dual_p2_refiners=True`；
- architecture ID 至少区分 `p2br0/1` 与 `p2pr0/1`，完整参数继续进入 SHA256 fingerprint；
- 老 checkpoint 缺少 point-refiner 配置时默认 off。

## 6. 竞争路线 B0：P2-BRR v2 机制试验

该路线不作为默认长训。既有 oracle 与 P2 on/off 结果已经显示其上游能力和下游消费都很弱，
不值得先工程化一个 v2。它仅保留为冻结 checkpoint 上的可选 oracle 对照，用来解释 Phase 0 的
目标域结果；不得新增网络、不得进入 400-epoch 试验。

## 7. 代码实施清单

### 7.1 新增文件

- `rsprompter/prompt_robustifier.py`
  - 语义局部 jitter；
  - FP/FN corrective replacement；
  - epoch schedule；
  - stats；
- `rsprompter/p2_point_refiner.py`
  - **仅在 Phase 1 信息门通过后新增**：P2 点位校准、候选集合 teacher 与 stats；
- `configs/whu1024_prompt_p2_v3.py`
  - 从 WHU C5-v2 配置继承；
  - 旧 P2-BRR off、PromptRobustifier on；P2PointRefiner 默认 off，只有获批实验显式开启；
- `scripts/ablations/whu1024_prompt_p2_a0_clean_10p.sh`、
  `whu1024_prompt_p2_a1_local_10p.sh`、`whu1024_prompt_p2_a2_corrective_10p.sh`、
  `whu1024_prompt_p2_a3_joint_10p.sh`
  - 四个固定 PromptRobustifier 训练臂；每个 wrapper 固定本臂唯一的训练 mode，不接受用环境变量
    偷换模块组合；
- `scripts/ablations/whu1024_prompt_p2_oracle_10p.sh`
  - Phase 0 的 raw-logit、GT-miner 和 matched-shift oracle；
- `scripts/ablations/whu1024_prompt_p2_probe_10p.sh`
  - Phase 1 的 collect/fit/eval，固定 real/coarse-only/sham 分支和输出布局；
- `scripts/ablations/whu1024_prompt_p2_eval_10p.sh`
  - 给定 checkpoint 的 10% validation 配对评估，支持显式 P2-on/P2-off 标签；
- `scripts/ablations/whu1024_prompt_p2_full.sh`
  - 仅在快速机制结论成立且用户另行批准后使用；
- 对应纯张量单测文件，避免只在 commit message 记录测试。

### 7.2 修改文件

- `rsprompter/models.py`
  - coarse target 前移并复用；
  - 将 `prompt_gt_masks` 传入 mask forward；
  - 汇总新 stats 和 loss；
- `rsprompter/models_sam2.py`
  - 严格解析 `prompt_robust_cfg`，以及条件式 `p2_point_refiner_cfg`；
  - 接入 robustifier；信息门通过后才接入 point refiner；
  - 保存 base/P2/augmented 三套诊断坐标；
- `rsprompter/shape_prior.py`
  - 保持 canonical miner 本身确定性；
  - 删除或弃用当前内嵌 point jitter，避免两套 augmentation 重复生效；
- `rsprompter/architecture_contract.py`
  - 新 architecture ID；
- `train/train_rsprompter_fusion.py`
  - PromptRobustifier 新参数/epoch 传递；获批后再加 P2 refiner 参数组；
  - current epoch 传递；
  - DDP 聚合机制指标；
  - 获批后可选 `--train-p2-point-refiner-only`；
- `inference/infer_from_checkpoint.py`、`inference/oracle_p2_probe.py`、
  `inference/probe_p2_point_learnability.py`
  - 统一增加与训练 DataLoader 同语义的图像级 `--subset-ratio`、`--subset-seed`；禁止以
    `--max-batches` 冒充 10% validation 子集；
  - 输出 manifest 必须记录解析后的 image-id 列表或其 SHA256、子集比例、seed、checkpoint SHA256、
    P2 mode、命令与脚本路径；获批后的 `infer_from_checkpoint.py` 还须支持
    `--disable-p2-point-refiner` 配对消融并记录 on/off；
- `scripts/smoke_test_components.py`
  - 新模块数值、梯度、语义和兼容性回归；
- `scripts/ablations/validate_ablation_contract.py`、`smoke_all.sh`、`README.md`；
- `PROJECT_ARCHITECTURE.md`；实现阶段再新增/更新 `DEBUG_FIELDS.md`。

## 8. 必需可观测字段

新增字段必须先写入 `DEBUG_FIELDS.md`，并明确 latest-forward、rank-local sum/count 和
DDP epoch ratio 的区别。至少包括：

### PromptRobustifier

- `PROMPT_ROBUST/roi_count`
- `PROMPT_ROBUST/clean_count`
- `PROMPT_ROBUST/local_jitter_count`
- `PROMPT_ROBUST/corrective_count`
- `PROMPT_ROBUST/actual_moved_count`（排除 `(0,0)`）
- `PROMPT_ROBUST/semantic_reject_count`
- `PROMPT_ROBUST/fallback_count`
- `PROMPT_ROBUST/p1/p2/n1_moved_count`
- `PROMPT_ROBUST/fp_candidate_count`、`fn_candidate_count`
- `PROMPT_ROBUST/effective_strength`

### 条件式 P2PointRefiner

- `P2PR/valid_slot_count`
- `P2PR/mean_abs_offset_px`
- `P2PR/p1/p2/n1/n2_error_before_sum`
- `P2PR/p1/p2/n1/n2_error_after_sum`
- `P2PR/error_count`（所有误差必须配 denominator）
- `P2PR/semantic_violation_before/after_count`
- `P2PR/actual_changed_slot_count`
- `P2PR/gate_mean`
- `P2PR/saturated_offset_count`

### 因果链指标

- raw→refined coarse boundary-F1（若启用 logit 对照）；
- base→P2 候选集合距离与冻结 decoder 配对效用；
- P2 on/off mask logit L1 与二值 flip ratio；
- P2 on/off 配对 segm/mAP；
- clean/augmented final-mask loss 分开统计。

## 9. 单测与 smoke 验收

### 9.1 PromptRobustifier 单测

- `enabled=False` bitwise 等价且不额外消耗 RNG；
- eval 模式 bitwise 等价；
- offset 永不为 `(0,0)`；
- P 点不离开合法前景、N 点不进入前景；
- thin/empty/full mask 能回退且无越界；
- invalid slot 保持 invalid 且坐标不动；
- 各模式比例在大样本统计上合理；
- epoch schedule 边界值正确；
- seed 固定时可复现。

### 9.2 条件式 P2PointRefiner 单测

- 零初始化严格 `offset=0`；
- offset 不超过每 ROI 上限；
- invalid slot 不动；
- P2 detach，refiner 参数有梯度；
- PromptEncoder/MaskDecoder final loss 能回传到 point refiner；
- teacher 候选集合为空时 loss 为有限 0；
- 正点集合匹配对交换不敏感；
- N1/N2 角色不被静默交换；
- on/off checkpoint round-trip strict，无 missing/unexpected key；
- 老 checkpoint 自动解析为 point refiner off。

### 9.3 端到端 smoke

单卡至少 50 train iterations + 20 validation images：

- 无 NaN/Inf；
- `actual_moved_count>0`；
- `N1 corrective_count>0`；
- 若本 smoke 显式启用且已获信息门批准，point refiner 参数 grad norm 非零；
- semantic violation after 为 0；
- P2 on/off 输出至少有非零差异；
- validation 关闭所有 training-only augmentation。

## 10. 分阶段实验矩阵与杀线

### 运行可追溯性契约（适用于全部阶段）

训练、oracle/probe、checkpoint 验证均**只能**经本节列出的 `.sh` wrapper 启动，禁止直接调用
`python` 或 `torchrun`。wrapper 统一 `set -Eeuo pipefail`、加载 `scripts/load_environment.sh`，并将
固定的 `TRAIN_SUBSET_RATIO=0.1`、`VAL_SUBSET_RATIO=0.1`、`SUBSET_SEED=44`、实验 ID、初始化
checkpoint、总 optimizer-update 预算、P2 开关与输出目录写入启动快照/manifest。

每个 wrapper 只能透传运行时无关架构的 `--master-port`、`RUN_SUFFIX`、`LOG_FILE`、`CHECKPOINT_DIR`
等隔离参数；任何改变模型或实验臂语义的值必须由独立 wrapper 固定。输出目录已存在时默认失败，
只有显式 `ALLOW_RERUN=1` 才允许创建带新 run-id 的重跑目录，绝不覆盖已有证据。

### Phase 0：WHU 10%/10% 基线与 oracle（先于代码扩展）

本轮目标域固定为 WHU。所有快速验证固定使用图像级、确定性抽取的 10% train 和 10% validation：
`TRAIN_SUBSET_RATIO=0.1`、`VAL_SUBSET_RATIO=0.1`、`SUBSET_SEED=44`（validation 按既有规则使用
`44+10000`）。同一对 10% 子集、初始化、阈值和更新预算必须用于全部对照；不得为不同模块重抽样。
该 validation 子集是机制 dev，不是最终性能声明依据。Phase 0 只允许通过
`whu1024_prompt_p2_oracle_10p.sh` 启动，输出由 wrapper 生成唯一 run-id 与 manifest。

在同一 WHU 起始 checkpoint 上跑三类配对 oracle：raw-logit 强修正、GT-miner 同卡点，以及
P1/P2/N1 各自的 matched ±1-cell 位移。记录每槽语义和冻结 decoder 的 ROI loss/IoU 响应。所有比较
报告按 image bootstrap 的 95% CI；`+0.003 mAP` 是历史目标而非本阶段杀线。若点位 oracle 本身没有
稳定的 decoder 效用，P2 路线立即停止，只保留 PromptRobustifier。

### Phase 1：P2 信息探针（缓存、伪对照、无需接入主训练）

在固定 WHU 10% train 上缓存 `coarse cue + base points + P2`，在固定 WHU 10% validation 上比较，
以相同轻量预测器和相同参数量运行，并只经 `whu1024_prompt_p2_probe_10p.sh` 启动：

| 输入 | 目的 |
|---|---|
| coarse cues only | 基本可学习性 |
| real PAFPN P2（或可用 native stride-4） | 候选真实信息 |
| 跨 ROI 打乱 P2 或空间打乱 P2 | 排除额外容量、位置先验和 coarse 泄漏 |

teacher 使用第 5.4 节候选集合；主指标是预测点相对 base 点的**冻结 decoder 配对效用**，次指标是
到有效候选集合的距离，而非某个 GT-miner 点的欧氏误差。相同预算下，只有 real P2 相对最佳 sham 的
效用差的 image-bootstrap 95% CI 下界大于 0，且至少覆盖 matched-oracle 效用缺口的 10%，才允许实现
P2PointRefiner。否则记录“未证明 P2 增量信息”，停止该分支。

### Phase 2：WHU 10%/10% Prompt 消费端短训归因

先用相同初始化、固定 WHU 10%/10% 子集和**按 optimizer update 对齐的固定短预算**完成不含 P2 的
四臂微筛：A0 clean、A1 local-only、A2 corrective-only、A3 local+corrective；对应训练及验证必须分别由
`whu1024_prompt_p2_a{0..3}_*_10p.sh` 与 `whu1024_prompt_p2_eval_10p.sh` 启动。不可按每臂 best epoch
选优。先分别确认 local 与 corrective 的作用，才允许称其组合有效。box jitter 后置，
仅在 A3 明确优于 A0 后，另与 A3 配对比较。

若且仅若 Phase 1 通过，再追加 B0（A3 + 无 P2）、B1（A3 + real P2 refiner）、B2（A3 + sham-P2
refiner）。B1 必须在冻结推理、同一阈值和配对样本上优于 B0 与 B2，才能归因给 P2；仅作同一
checkpoint 的 P2 on/off 只能证明模块依赖，不能替代独立训练对照。

### Phase 3：WHU 全量确认（快速筛选后另行批准）

10%/10% 快速实验只能决定“值得继续”或“停止”，不能形成最终 P2 性能结论。只有 Phase 2 已预注册
获胜，才建议启动独立的 WHU 全量 train / 全量 validation 确认；该步骤需要另行批准，不随本轮快速验证
自动启动。届时使用相同训练预算、初始化规则和评估口径，并用第二独立 seed 复现；报告固定末期、
预先定义的 last-k 均值或 AULC，避免以最佳 epoch 选择获益，所有主差值附 image-bootstrap CI/seed 方差。

只有全量确认中 clean 指标不劣、matched corrective probe 为正、且 real P2 独立训练比较的 CI 支持正增益，
才允许写 P2 结论。E0/E1 追加点只作旁证。

## 11. 失败定位决策树

```text
目标域 matched-oracle 是否有稳定的 decoder 效用？
├─ 否 → 当前固定 2P2N 无足够点位空间；只做 PromptRobustifier
└─ 是
   ↓
real P2 是否显著优于 coarse-only 与最佳 sham-P2？
├─ 否 → P2 增量信息未获证明；停止 P2 refiner，不以加容量补救
└─ 是
   ↓
点是否真实移动、语义正确，且提高冻结 decoder 候选效用？
├─ 否 → teacher/set loss、gate、匹配或保护实现问题
└─ 是
   ↓
独立训练的 real-P2 是否优于 no-P2 和 sham-P2？
├─ 否 → 消费链未打开或增益不可归因；不进长训
└─ 是 → 才能进入锁盒长训与论文归因
```

若 P2-BRR oracle 的 refined boundary 指标改善但 final mask 不变，不再继续加大 residual；这表示
dense/logit 消费通路无效，不能据此继续投入。

## 12. 实施顺序

1. 固定 WHU 10% train / 10% validation 子集、`SUBSET_SEED=44` 与短程 update 预算；
2. 跑 Phase 0 WHU oracle；未见空间则只实施 PromptRobustifier；
3. 将现有内嵌 positive jitter 重构为默认关闭的 `PromptRobustifier`，前移并复用 GT coarse target；
4. 完成其单测、统计和 50-iteration smoke，执行 Phase 2 的 A0--A3 微筛；
5. 仅当 Phase 0 有空间时跑 Phase 1 的缓存 real/sham P2 信息探针；
6. 仅当 Phase 1 通过时实现 P2PointRefiner 与候选集合 teacher，再跑 B0--B2；
7. 记录快速筛选结论；只有用户另行批准才启动 WHU 全量确认；
8. 完成后更新 `PROJECT_ARCHITECTURE.md`、`DEBUG_FIELDS.md`、消融 README 和论文声明。

## 13. 提交与回滚边界

建议拆成可回滚提交：

1. `refactor(prompt): semantic robustifier and target plumbing`
2. `feat(prompt): matched fixed-2p2n corrective curriculum`
3. `test(prompt): mechanism smoke and checkpoint contracts`
4. `feat(p2): gated bounded point calibration refiner`（仅信息门通过后）
5. `docs(prompt-p2): experiment matrix and operational fields`

不得把训练日志、checkpoint、`__pycache__` 或现有无关 `main5.tex` 修改纳入这些提交。

## 14. 结果声明边界

实施完成但训练前，只能声明“联合机制已接线并通过数值/梯度 smoke”。

Phase 1 通过后，只能声明“在目标域和伪特征对照下，P2 含有 coarse 之外、可带来冻结 decoder
候选效用的增量信息”。

Phase 2 通过后，才能声明“MaskDecoder 能消费 P2 校准点，且短程配对结果为正”。

只有 Phase 3 在 WHU 全量协议和独立 seed 上通过，才能声明：

> P2 边界信息通过同语义点位校准与共适应的 prompt 消费端，提高了最终实例分割性能。

## 15. 子代理审查与对抗收敛记录

审查分为“P2 可行性”与“因果/实验设计”两条独立立场。P2 审查指出现有离线点学习报告仅有
3.125% 的坐标误差改善，不能将点校准直接列为主线；因果审查进一步指出 GT-miner 单点不唯一、
真实 P2 可能只是在获得额外容量、WHU fast-150 的结果不能替代确定的 WHU 快速子集协议、以及验证集
不可被反复换子集用于选择。两者的共同裁决不是废弃 P2 假设，而是先把它降为可证伪条件分支。

收敛后的不可协商项如下：目标域 oracle 先行；teacher 改为集合/冻结 decoder 效用；必须有
coarse-only、sham-P2 与 independently-trained no-P2 对照；固定 10%/10% 机制 dev 与未来全量确认
分离；主差值给出 image-bootstrap CI 和 seed 方差。因协作线程额度已满，无法启动第二轮实时互评；本记录只陈述
已返回的两份独立审查及其基于现有日志的收敛，不把未发生的对话伪称为结论。
