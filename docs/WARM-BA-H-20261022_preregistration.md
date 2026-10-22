# WARM-BA-H 预注册：实例边界裁决臂（设计 1，V1 数据集版，2026-10-22 启动前）

依据：`docs/DUAL-NEWERA-DIRECTION-20261021.md` §四（WARM-BA 骨架）、
`docs/S2DBNET-READING-20261022.md` §三（S²DB-Net 吸收设计 1）、
PI 指令（2026-10-22）：**V1 数据集 + V1 单流基座**（暂不用 v3）。

## 固定契约

- **数据**：`CitySample_SatUAV_TrainMinimal_v1_20261021`（train 2142 /
  val 459 / test 459）。裁决器训练 = train split；四重门判定 = val 459；
  test 仅在全部 val 门落判后作一次收口读数（沿既有 PI 授权 peek 模式，
  cap250+md250、通道序修复口径）。
- **基座（不可再训练）**：EXP-0025（v1 hflip 臂）`best_model.pth`
  （epoch 71，val segm 0.8690；台账锚 `docs/SINGLESTREAM-0024-0027-METRICS-20261022.md`
  §3/§4）。推理协议：cap250 + maxDets=250 + 通道序修复（187c29e/645aa40）。
- **供给**：逐界面几何特征——界面 = NMS 后相邻预测实例对间的
  ZMEAN-CONT 式边界带（dilate(A,3)∩dilate(B,3)∩~(A∪B)，两侧带
  = 各自 ∩ dilate(strip,2)，双侧有效覆盖 ≥0.6）；zmean 面来自 EXR
  深度 + 精确相机的 z-buffer 均值（probe_zmean_cont_v2 同款数学）。
  特征：dz=|两侧 zmean 均值差|、两侧 p25/p50/p75、带内 zmean、带
  missing 率、界面长度、两侧预测分数；对照特征 dgray（同界面卫星
  灰度差）。
- **消费**：**post-hoc 裁决**——训练轻量裁决器（logistic 回归，
  sklearn 默认 + 类平衡；禁 GBDT 以防过拟合小样本）输出"该界面两
  预测应否合并"；仅 **merge 动作**（本臂单一动作；split 动作登记为
  变体，当前不实现）。基座与推理管线其余部分逐位不动，off≡single
  按构造。
- **监督**：train split 上预测对与 GT 的匹配关系——两侧预测最优匹配
  到**同一 GT 实例**（IoU≥0.5 口径，双侧各自最优且相同）→ 标签
  "应合并"；其余 → "保持"。标签与特征收割脚本一次性冻结（hash 归档）。

## 四重门（val 459，全部先于任何 test 读数）

0. **前置门（先于一切）**：R0-new 错误分解（基座在 val 的预测 vs GT）：
   "拆分型界面错误"（相邻预测对匹配同一 GT）占场均预测实例数 ≥3%，
   且 ≥总错误质量（漏检+虚警+错配）的 10%——不过则方向在 v1 上不
   启动，落判记录（该结局同样有效）。
1. **供给门**：裁决器 5 折 CV（train split 内）AUC ≥0.60 且折间
   配对 CI95 下界 >0.5；dgray-only 对照 AUC 同时报告。
2. **净门**：裁决后 vs 基线，val 459 逐场 segm/bbox AP 配对 bootstrap
   （10k draws，seed 20261014）CI95 下界 >0 且点估计 >0；伴随指标
   B-AP 7px（严格边界口径）与合并错误计数变化。
3. **归因门（双门，缺一不可）**：(a) **外观消融门**——仅 dgray 特征
   的对照裁决器在 val 上的净增益必须显著低于含高度特征版（配对差
   CI95 上界 <0 或点估计差 ≥1/2 净增益）；(b) **错配门**——换错
   场景 zmean（C1 式倒序配对）后裁决必须劣化（裁决后(on)−裁决后(C1)
   配对 CI95 下界 >0）。
4. **签名**：净增益集中于多实例场景（场景 GT 数 > 中位数）与
   strict-7px 边界口径，而非 AP50。

## 预注册解读条款

- 全部过门 → WARM-BA-h 在 v1 成立；v3 复跑为独立臂（同预注册换数据
  契约即可）。
- 前置门/供给门失败 → 方向在 v1 不成立，记录数字；不启动 test。
- 净门过但归因门败（外观可替代）→ 判 **APPEARANCE-SHORTCUT**：增益
  真实但不属于 UAV，方向按"无双流价值"落判。
- 决策阈值（合并概率 >0.5）在 train split 内选定冻结；任何门失败
  不允许回调特征/阈值重跑（新臂需新预注册）。

## 工件

- 代码：`scripts/ue/warm_ba_h_decompose.py`（前置门）、
  `scripts/ue/warm_ba_h_adjudicate.py`（特征收割/训练/裁决/评估）
- 输出：`/data2/wangcheng/checkpoint/surfbev_b0/probes/WARM-BA-H-20261022/`
- 本预注册提交于任何实现代码之前。
