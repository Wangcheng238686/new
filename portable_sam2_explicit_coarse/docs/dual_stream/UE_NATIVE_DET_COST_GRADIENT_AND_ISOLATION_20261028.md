# Native full100 C 臂检测分支负收益：量化、梯度图与隔离方案（2026-10-28）

主代理对正在本机四卡共租运行的 `ue_g_ubr_native_full100_20261028`（A_nohead / B_zero /
C_correct，100 轮，进行至 E59/E60）做的中期只读分析，外加"C 对纯单流能否稳定超越"的设计
判断。全部数字来自训练树 val 459 逐轮 metrics 与旁挂检测遥测，**不是 test、无配对 bootstrap、
单 seed 44、训练未完成**；任何结论以此边界为限。

## 一、bbox 负收益的量化事实

三臂共同初始化、同预算同 LR 日程；差异只在 native 尾部（A 无头 / B 头在但 RGB+几何证据置零 /
C 正确证据），上游 G（detail 变体，含正确 UAV RGB）与实例头三臂相同
（`study_contract.json` limitations 自述 "not a pure satellite comparison"）。

逐轮 `bbox/mAP` 差（E1–59，val 459）：

| 比较 | 为负轮数 | 全程均值 | 后期均值(E45–59) | 后期按尺寸 |
|---|---|---|---|---|
| C−B | 33/59 | −0.41pp | −0.20pp | 中 −0.44 / 大 −0.40 / 小 +0.35 |
| C−A | 23/59 | +0.08pp | +0.21pp | 全为小幅正值 |
| B−A | 44/59 为正 | — | +0.41pp | — |

segm 同源对照：C−B 后期 +1.38pp（小 +2.18 / 中 +1.27 / 大 +1.11），全程稳定为正。当前
val-best（按 segm/mAP 选权）：A .8295(E54) / B .8344(E55) / C .8297(E56)。

与 E10 短研究（`UE_NATIVE_TAIL_ATTRIBUTION_RESULTS_20261028.md`，test、LR 完整衰减）的关系：
短研究 C−A bbox −1.33pp 区间明确为负、**B−A 也 −1.19pp**、C−B −0.14pp 跨 0。即短预算下
负收益主要是"加头"的结构代价；该代价在 100 轮日程中消失（本 run B−A 转正 +0.4pp），持续
存在的只有 **C−B 约 −0.2~−0.4pp 的内容性小代价**。短研究是衰减完成的 10 轮，与本 run 的
E10 不可直接比较。

## 二、遥测分解：损失不在框回归，在排序

旁挂检测遥测（`telemetry/detection/`，每轮 459 图全量，CPU 只读）后期 E45–59 均值，
combined 视图（最终 dt 候选，score≥.25 / IoU≥.5）：

| 臂 | TP | FN | 背景FP占比 | 小/中/大召回 | TP框误差 |
|---|---|---|---|---|---|
| A | 14785 | 195 | .306 | .9564/.9965/.9978 | \|dx\|1.01px \|logwr\|.0399 |
| B | 14782 | 198 | .314 | .9569/.9958/.9992 | \|dx\|1.00px \|logwr\|.0391 |
| C | 14798 | 182 | **.341** | .9605/.9961/.9993 | \|dx\|1.03px \|logwr\|.0401 |

三条判定性事实：

1. **TP 框回归质量三臂逐项相同**（中心 ~1px、log 宽高 ~.04）——框没有回归得更差；
2. **召回不降反升**（C 小目标召回最高、FN 最少）——不是漏检问题;
3. **超额损失全部是背景假阳性**：C 的背景 FP 绝对数较 A/B 多约 +1.5~2.2k（中大型预测框上
   最明显：大框背景 FP 802 vs A 385），定位型 FP 反而更少。**original_path（纯卫星前缀路径）
   三臂几乎逐项相同**——卫星检测路径本身未被破坏，分歧集中在 combined（含 UAV 补充候选与
   分数重编码）一侧。

结论指纹：同框质量、同/更好召回、更多背景候选越过分数线 = **分数排序/校准小幅退化**，
而非定位或召回问题。逐 epoch 序列见 root 下各 `validation_epoch*/metrics.json`；
遥测原始 JSON 在 `telemetry/detection/<arm>/epoch*.json`。

## 三、当前梯度流动路径（含行号）

模型：`CompactJoint`（`rsprompter/uav/compact_joint.py:32`）= 卫星 `RSPrompterAnchor` +
`CandidateDetailInteraction`（**detail 变体**，`scripts/probes/ue_ubr_native_runtime_20261027.py:39`
`install_g(...,'detail',...)`）+ native 尾部（`install_native_refiner`，同文件 :40）。
损失分组：D=rpn_cls+rpn_bbox+cls+bbox；M=主 512² mask loss（作用在 refined logits 上）+
coarse .1；I=loss_uav_instances（center/offset/logwh）。

```text
前向（三臂结构相同）：
五视角RGB → rgb_encoder(P0 LoRA)+rgb_project → selected(候选读取, candidate_detail_interaction.py:58-62)
    → reduce_rgb('detail'=valid均值, :68-70) → aligned 128²×32
        ├─(NativeEvidenceG 缓存 native_aligned_rgb, 不 detach, ubr_native_integration.py:34)
        └→ rgb_fusion(u, aligned) → u128 (:97)
几何5(测量,数据侧无梯度) → CNN a/b/c/d/up → u ──► block4 双向交叉注意力 interact()
                                        instance_interaction_probe.py:113-154
                                        updated_sat = base+delta (internal=True 写回)
卫星Hiera+LoRA ◄─ block1 写回 base+geometry_delta (detail 变体激活, candidate_detail_interaction.py:89-100)
    → PAFPN → RPN/RoI 检测头 (D)  ──► cspm mask head → SAM decoder 一次解码
                                        │ logits + 捕获 act2
                                        ▼ (B/C) NativeDecoderShim (ubr_native_integration.py:76-115)
                                     sample_roi_evidence(native_aligned_rgb 保图, 几何5数据侧)
                                     NativeRoIRefiner: [LN(act2), LN(rgb32), geo5, σ(logitsᵈᵉᵗ)]
                                     → zero-init delta → refined = logits+delta (M)
实例头 heat/reg (I) ← head_fusion(high_u, head_satellite(updated_sat))  ← 刻意穿过卫星写回分支
                                        (instance_interaction_probe.py:141-142 注释)
```

关键梯度事实：

- **已被 detach 的边界**：RoI 框/image_ids/采样位置（`native_roi_refiner.py:71,96,112`）、RGB
  有效性（:93）、refiner 输入的 logits 概率（:168）。
- **唯一保留计算图的证据端口是 `native_aligned_rgb`**——实施文档明写"证据缓存随
  encode/clear 更新，不 detach"，是显式设计决定而非疏漏。
- 各损失到达的共享参数：D→检测头→FPN→主干（途经两处写回输出，故 D 也训练交互/detail 参数）；
  M→①logits 支路→decoder→主干（三臂皆有）②delta 支路→refiner→act2→decoder→主干（B/C 有）
  ③**→roi_rgb→grid_sample→native_aligned_rgb→reduce_rgb→selected→rgb_encoder（仅 C）**；
  I→几何CNN/rgb_fusion/UAV编码器/主干（经 updated_sat，刻意设计）。
- **C 独有耦合一句话**：M 经尾部 RGB 端口训练 UAV RGB 编码器，而该编码器输出 `aligned`
  位于两条检测前向路径上游——(i) `rgb_fusion→u128→block4 写回→updated_sat→全部检测头`；
  (ii) `context→实例头→补充候选的框与分数`。这是纯梯度期耦合（前向结构三臂相同），与遥测
  指纹吻合：original_path 不变、超额背景 FP 只出现在 combined。

**注意**：这是"接线+遥测指纹"的推断，不是实测。四卡 det/mask 梯度余弦遥测
（`UE_NATIVE_TASK_TELEMETRY_20261028.md`）已部署但因共租显存不足未出数（telemetry 目录
只有源码无梯度产物）；该文档自身也警告负余弦不能单独证明 AP 下降由此导致。

## 四、隔离方案（按优先级）

- **方案 A（首选）：端口级 detach**，对应单流 a3sgm 的 `SHAPE_DENSE_DETACH` 先例
  （`rsprompter/cspm/dense_prompt_utils.py:21-32`：dense canvas 输入从 coarse 计算图
  detach，保前向内容切梯度边界）。本处一行：
  `ubr_native_integration.py:34` 改为 `self.native_aligned_rgb = aligned.detach().T.reshape(1,32,128,128)`。
  前向逐位不变；几何通道无需处理（数据侧无梯度）。预注册 **C_detach 臂**（与 B/C 同
  common_init 同日程），判据事先写死：(i) bbox C_detach−B 是否归零（验机制）；(ii) mask
  C_detach−B 保留多少（隔离代价）。两 outcome 都是干净归因。
- **方案 B（更彻底）**：`ubr_native_integration.py:104` 再加 `head.native_refiner(feature.detach(),...)`，
  让 6625 参数头成为纯后置修正器；B 需配对处理。信息损失大于 A（E10 只证明消费端内容
  依赖 −9.19pp，未测 decoder 侧适应贡献）。
- **方案 C（测量先行，零训练成本）**：GPU 空闲后先跑已排队的四卡梯度 VJP 遥测（E1/E10/E25/E50
  保留检查点都在）。预期若本机制成立：C 的 M 梯度出现在 RGB alignment/fusion 分组而 B 近零；
  D/M 在 shared_representation 余弦 C 更负或范数比更大。PCGrad/权重再平衡必须等此测量
  （治理协议明文"不自动调整 loss 权重/PCGrad/解冻范围"）。
- **方案 D（反向思路）**：不隔离而给检测分支自己的证据入口（staged U 臂动机；attrition
  分析早已表明 mask-only 入口救不回检测漏失）。S 臂先回答"代价是否 UAV 特有"。

执行顺序（治理合规）：① 梯度遥测门 → ② C_detach 短预算探针（E10 同 LR 日程，可与短研究
直接对话，新 worktree、不动 full100 冻结源）→ ③ 通过后并入 staged 矩阵。

## 五、C 对纯单流能否稳定超越（设计判断）

**判断：mask AP 上会超过、但预期是"薄而稳"（+0.5~1.5pp），bbox 可能打平或略负；"稳定"
目前只对几何内容成立（双 seed CI 正：真−常量 +0.24/+0.18pp），RGB 内容只正过一次
（internal +0.194pp CI 正）。**

- 澄清：**C 与纯单流至今没有真正比过**——A/B/C 全带上游 G+实例头；最接近的证据是四臂
  联合训练 E67（SAT 85.50 / RGBD 86.99，RGBD−SAT 26/27 轮正、均值 +1.02pp），但其常量
  对照 86.08 几乎复制该增益（RGBD−常量 +0.11pp、16/27 轮正）。判决实验是 staged **S 臂**
  （纯卫星 UBR，同预算同日程），这正是路线图把 S 排在阶段一第一位的原因。
- 支持面：信息论（单张卫星图无法消解高度/遮挡/粘连歧义）；三入口多点受益（G 早期注入/
  实例补充头/native 尾部）；几何内容差双种子区间全正；收益落在 Boundary AP1% +5.8pp——
  与跨视角深度证据的理论优势位置自洽；补充候选是单流结构性缺失的召回通道（覆盖 85.89%
  vs 常量 61.80%，补回 GT 811 vs 476）。
- 风险面：**"强基座弱内容差"是本线反复出现的模式**（常量/同容量对照屡次压缩毛增益）；
  S+同预算可能吃掉大部分结构收益，C−S 压缩到内容差堆叠；检测分支 −0.2~−0.4pp 代价未解；
  单 seed；v1 test 已开发曝光（最终结论需新保留集 + 双种子）。
- 推翻条件：S 臂 100 轮后若 C−S mask < +0.2pp 且 CI 跨 0 → 双流叙事不成立，退回
  "深度引导边界精修"窄主张；若 C−S bbox 为负 → 如实保留并缩窄表述。

## 六、边界与未定项

- 全部为训练中途 val（E59 快照）；test 一次性评测在 owner 队列（pending），此前不改任何
  已排队合同。
- 逐轮波动 ±1~2pp，远大于 −0.2~−0.4pp 均值效应；无配对 bootstrap。
- 梯度耦合机制未经四卡遥测实测验证。
- 分尺寸结论（中大型承担损失）与背景 FP 尺寸分布方向一致但未做逐臂配对，不作因果宣称。

## 来源

- root：`/data/wangcheng/checkpoint/ue_g_ubr_native_full100_20261028`（owner 在跑；
  `study_contract.json`、`*/validation_epoch*/metrics.json`、`telemetry/detection/`）。
- 代码：本仓库 `rsprompter/uav/{native_roi_refiner,ubr_native_integration,candidate_detail_interaction,instance_interaction_probe,compact_joint}.py`、
  `scripts/probes/ue_ubr_native_runtime_20261027.py`（与冻结运行源 SHA 一致）。
- 前置文档：`UE_NATIVE_TAIL_ATTRIBUTION_RESULTS_20261028.md`、
  `UE_UBR_NATIVE_ROI_IMPLEMENTATION_20261027.md`、`UE_NATIVE_TASK_TELEMETRY_20261028.md`、
  `UE_COMPACT_EXPERIMENT_ROADMAP_20261028.md`、`a3sgm_architecture_guide.md`（sg 先例）。
