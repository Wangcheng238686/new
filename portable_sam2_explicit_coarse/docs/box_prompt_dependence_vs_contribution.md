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

## 5. 方案：训练期 box token dropout（提案，未实施）

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
