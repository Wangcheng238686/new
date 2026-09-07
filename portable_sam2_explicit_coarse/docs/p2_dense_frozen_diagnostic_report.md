# P2/dense 冻结权重诊断报告（供 Codex 复查）

日期：2026-09-06（v1 诊断 09-05 深夜，v2 修复与网格 09-06 凌晨）
分支：`machine2/fast-whu150`（HEAD 624165c2 + 未提交工作区）
诊断对象：`whu_d3fi_full_p2esc030_d100_10p_2gpu_tr0.1_va0.1/best_model_epoch12.pth`
（D3：P2 beta=0.30×Δ=1.00、aux loss 0.20、raw 画布、detach=0、
full_image + roi_balanced_dice、10%/15ep dev 协议，best 0.6155）

工件清单（均未提交，供审查）：
- 探针：`inference/probes/p2_dense_diagnostic.py`（v2）
- 全量网格：`refactor_golden/d3_diagv2_full.json` / `.log`
- 子集网格 + Pass A：`refactor_golden/d3_diagv2_subset.json` / `.log`
- 冒烟：`d3_diagv2_smoke.json`、`/tmp/diag_smoke2.json`
- v1 遗留（**已作废**，仅存档）：`d3_diagnostic_full.json`、`d3_diagnostic_smoke.json`
- 残差分布探针（另一工具，独立于本报告主体）：
  `inference/probes/delta_quantile_probe.py` + `d3fi_full_delta_quantiles.json`

---

## 0. 背景与问题链

D3（P2 包络升级）/D4（csig 画布）六臂完成后，我方曾据 v1 诊断宣称
"P2 是置信度增强器""消费端聋了""学习型画布无意义"。用户复核后否决，
指出 v1 三处实质缺陷（§1）。本报告 = v1 撤回 + v2 修复后的完整重测。

## 1. v1 三处缺陷（全部成立，相应结论撤回）

| # | v1 缺陷 | 后果 | v2 修复 |
|---|---|---|---|
| 1 | 计分循环 `if len(prediction["scores"])` 滤掉零检测图 | 漏计漏检，"遍历全验证集"≠"完整计分" | 所有图（含零检测）进入 COCO 评估；记录 image ids / GT 总数 / 有检测图数 |
| 2 | GT 用全图实例**并集**（每 ROI 同一张占据图） | 检验的是"全图建筑占据提示"，非实例级；Pass A 的正确/错误像素定义失真 | 每 ROI 按 max box-IoU≥0.5 匹配**对应实例**；未匹配 ROI 单独统计，干预时保留学习画布（显式策略） |
| 3 | amplify 判据 `正确 & |delta|≥0.5cap`，无方向 | logit 3→2.8 也被计为"增强"；correct_mean_delta 混合前景/背景符号 | `toward = delta×(2×GT−1)` 方向化；增强/削弱/大幅增强/大幅削弱分开报；修复数与破坏数带各自分母 |

撤回的 v1 结论：**"82.9% 置信度增强"**（并集伪影）、**"GT 完美画布
Δ≈0 ⇒ 消费端聋"**（并集画布非实例信息 + 门控未动）、**"学习型画布
无意义"**（依据前两条，不成立）。

## 2. v2 探针设计要点

- **加载**：与 `infer_from_checkpoint` 同源（`_load_checkpoint →
  _resolve_model_config → _register_and_build → _select_state_dict →
  _load_model_state`，strict；missing/unexpected 非空即抛错）。
- **env 重放**：source `phase0/d2_full_env.sh` 后覆盖 fi 四变量
  （FINAL_MASK_COORDINATE_MODE/LOSS_MODE + P2 beta/delta/loss_weight），
  解决嵌入指纹-契约恢复问题。
- **Pass A**（refiner 行为）：`register_forward_hook(with_kwargs=True)` 取
  `prompt_rois/raw_logits/delta_logits/search_support`；每 ROI 匹配实例
  GT，裁剪→nearest→64×64 coarse 网格作 target；流式累计像素计数
  （方向化，见 §1 修复 3）。未匹配 ROI 单独累计（含其 |delta| 均值）。
- **网格 Pass**（消费端定位）：canvas ∈ {learned, matched-instance-GT}
  × alpha ∈ {current≈0.11, 0.5, 1.0} + dense_off。
  - 画布干预：`prompt_encoder` 的 forward **pre**-hook 改写 `masks`
    kwarg——此时 masks 已是**学习画布**（head 先算好再传 PE），因此
    未匹配 ROI 原样保留学习画布（无需重算）；实例占据图 ±logit_span
    （默认 4.0）二值画布，全图帧，PE mask 分辨率（256×256）。
  - alpha 干预：monkeypatch `head._effective_shape_dense_alpha` 返回
    常数（α=0 即 base_dense，画布断路）。
  - **点/框/检测不动**：只改 masks kwarg 与门控；检测端（box/score/NMS）
    不经过 mask head 反馈，因此各格子的检测完全一致——mAP 差异纯来自
    mask 形状（segm 排序用 detector score，排序亦不变）。
  - 遥测：dense embedding 范数（mask_decoder pre-hook args[3]）、
    decoder low-res logit |·| 均值（post-hook）、最终掩码 vs 基线格的
    变化率与 IoU（256×256 area 下采样后 packbits 对比）。
- **实现事故记录**（诚实披露）：v2 开发中修过 4 个 bug——refiner/head
  hook 的 kwargs 签名（2 处）、掩码差分基线字典键类型（`(id,)` vs `id`，
  导致 changed 恒 0，已修并验证）、Pass A 的 ROI 计数（记的是调用次数，
  已修；像素级比例不受影响，子集 JSON 中 matched/unmatched_rois 为
  修复前的调用计数，勿引用）。

## 3. 自洽性验证

1. `gt_instance_a0.0` 与 `dense_off` 是同一状态的两条独立代码路径：
   子集 mAP 完全一致（0.6127）、掩码变化率完全一致（0.313）。
2. 基线格自比较 changed=0.000。
3. 全量基线 0.6253 与 v1 全量基线（当时计分口径缺陷对基线格影响仅
   在零检测图遗漏）一致。
4. 学习画布格 canvas 值域 −5.5~7（raw logit）；GT 画布 ±4 在同量级。

## 4. 结果

### 4.1 Pass A（实例匹配 + 方向化；子集 50 batch，修复前 ROI 计数口径）

| 指标 | 值 |
|---|---|
| ROI→实例匹配率（网格口径，全量） | 81%（14721 注入 / 3392 保留） |
| 正确像素：toward（朝 GT 增强） | 24.0% |
| 正确像素：away（背离 GT 削弱） | **76.0%**，mean toward-delta **−0.140**（帽 0.30） |
| 正确像素被破坏（翻转成错） | 1.14% |
| 错误像素：支持域覆盖 | 29.4% |
| 带内错误：方向朝 GT | 83.4% |
| 带内错误：实际翻转 | 8.4%（占全部错误 2.46%） |
| 未匹配 ROI 的 |delta| 均值 | 0.237（≈帽值的 79%） |

画像：**方向感知的置信度压缩 + 微弱纠错**。削弱已正确像素、把带内
错误朝 GT 推但翻转率低；净效应微正（2.46% 修复 vs 1.14% 破坏，分母
不同，仅定性）。注：匹配用推理期 max-IoU，与训练期 sampler 指派
不完全等同（§6-L2）。

### 4.2 消费端定位网格

**全量（627 图，314 batch）**：

| 冻结权重干预 | segm/mAP | vs 基线 | vs 基线平均 mask IoU |
|---|---|---|---|
| 基线（learned, α≈0.11） | 0.625336 | — | 自比较 |
| learned, α=1.0 | 0.626174 | +0.0008 | 0.973159 |
| **GT instance, α=1.0** | **0.640702** | **+0.015366** | 0.969163 |
| dense off | 0.624285 | −0.001051 | 0.994273 |

同为 α=1.0 时，GT 相对 learned +0.014528。"80% 掩码变化"表示变化广泛
（平均 IoU 0.97，非每实例剧变）；本表为配对干预差，**不得除以训练
重复差来构造"几倍噪声带"的表述**（重复差不能定义分布）。

**子集（100 图）**：

| 画布 \ α | ≈0.11 | 0.5 | 1.0 |
|---|---|---|---|
| learned | 0.6132（基线） | 0.6154 | 0.6145 |
| GT instance | 0.6138 | 0.6209 | **0.6267** |
| dense off | 0.6127 | — | — |

全量没有 GT@当前 α 格——"内容与强度存在交互"来自子集，是**线索**
（lead），不构成"两个独立故障已确证"；冻结 PE 是**待检验假说**，不是
已定位的根因。

### 4.3 读数

1. **decoder 对 dense 有响应**：α=1.0 时 80% 最终掩码发生变化
   （平均 IoU 0.973）、dense_off 亦有 35% 变化（平均 IoU 0.994）——
   "变化广泛"不等于"每实例大幅变化"。
2. **内容×强度交互（子集线索 + 全量 α=1 配对）**：只开门
   （learned α=1.0）+0.0008；只换内容（GT@α≈0.11，仅子集）≈+0.0006；
   同时上 +0.0154（全量配对）。两独立故障的分解**未在单臂全量格子中
   完成交叉验证**。
3. 当前 dense 整体 ≈ 中性（off −0.0011），与 D 系列训练结论一致。

### 4.4 Pass A 的正确引用方式（修订）

按当前匹配/栅格化口径，子集原始像素计数：**纠正 37,323 / 破坏 27,398 /
净纠正 +9,925**。此前"2.46% vs 1.14%"两个百分比分母不同，**不可并列
比较**；像素净正亦**不可**直接解释为 AP 净正（AP 是实例级、经 NMS 与
匹配的量）。子集 JSON 的 ROI 计数为旧口径（调用次数），不可引用。

## 5. 结论分级（证据边界）

- **确证**（冻结权重上的直接测量）：v1 三缺陷已修复；decoder 对 dense
  高度响应；学习画布内容在门开大后无增益；实例准确画布 + 门开大有
  +0.0154（全量配对）。
- **线索（未完成全量交叉验证）**：内容与强度存在交互——"门控静音 ×
  内容不准"的两故障分解来自子集 + 全量 α=1 格；冻结 PE 是待检验假说
  而非已定位根因。
- **假说（待短训检验）**：门初值、PE mask_downscaling 适配（需
  PROMPT_ENCODER_LR_MULT≠0）、画布内容质量可在训练中共同适配，
  兑现部分 +0.015 的空间。
- **不主张**：GT 画布可部署（不可，诊断道具）；推理期 α 抬升可作
  训练配置（分布偏移）；+0.0154 = 严格上限或可训练收益承诺；冻结
  推理干预差与训练重复差做"倍数"比较（L5）；像素净纠错等于 AP 净
  收益（4.4）；"与 v1 基线数值相同"不等价于标准推理入口一致性验证
  （未做，L9）。

## 6. 残余局限（诚实清单，供复查重点）

| # | 局限 | 影响 |
|---|---|---|
| L1 | 网格无逐图记录导出 | +0.0154 无 bootstrap CI（可加导出后补算） |
| L2 | ROI→GT 匹配 = 推理期 max-IoU≥0.5，与训练期 sampler 指派非严格同一 | Pass A 的正确/错误分类在指派不一致的 ROI 上有噪声；网格注入同理 |
| L3 | Pass A target 裁剪用 int() 截断量化，训练 target 用 floor/ceil | 边界 1px 级差异，影响边界像素计数 |
| L4 | GT 画布为 ±4 二值图，与学习画布的平滑 logit 分布不同 | 干预本身引入分布偏移（已声明；α=1.0 下仍为正增益说明 decoder 对此鲁棒，但不排除 ±4 恰好有利） |
| L5 | +0.0154 是冻结权重推理干预的配对差，与训练重复差非同一统计口径 | 不得做倍数比较或据此定义噪声带 |
| L6 | 掩码差分在 256×256 area 下采样 | 亚 256 变化不可见 |
| L7 | 探针加载全量 validation split（非 dev 10% 子集） | 与训练期验证口径不同（诊断目的可接受） |
| L8 | alpha monkeypatch 返回 CPU 常数张量 | 依赖 torch 0-dim CPU 张量与 CUDA 运算的广播语义（当前版本成立） |
| L9 | 未做标准推理入口的独立数值复现（inference/infer_from_checkpoint 全量对照） | "与 v1 基线相同"不能替代该验证；后续可补 |

## 7. 复现命令

```bash
cd portable_sam2_explicit_coarse
bash -c '
source /data/wangcheng/checkpoint/portable_sam2_explicit_coarse/refactor_golden/phase0/d2_full_env.sh
export FINAL_MASK_COORDINATE_MODE="full_image" FINAL_MASK_LOSS_MODE="roi_balanced_dice"
export P2_BOUNDARY_REFINER_BETA="0.30" P2_BOUNDARY_REFINER_DELTA_LOGIT_MAX="1.00" P2_BOUNDARY_REFINER_LOSS_WEIGHT="0.20"
export CUDA_VISIBLE_DEVICES="0"
python inference/probes/p2_dense_diagnostic.py \
  --checkpoint <d3fi best_model.pth> --weights model \
  [--max-batches N] [--skip-refiner-pass] [--grid-alphas 1.0] \
  --output <out.json>
'
```

## 8. 下一步：D5 两臂最小短训（按 Codex 复审定稿，**待用户授权启动**）

见 `docs/d5_review_handoff.md` §3-§6。要点：Mask 行（points_box_dense、
raw_logits、detach=0、**P2 关闭**）两臂——
- **D5-A（门控对照）**：`SHAPE_DENSE_ALPHA_INIT=0.5`（可学习
  global_sigmoid 门的**初值**，非 fixed），PE 冻结如旧；
- **D5-B（dense 适配）**：D5-A + `PROMPT_ENCODER_TRAIN_MASK_DOWNSCALING=1`
  + `PROMPT_ENCODER_LR_MULT=0.1`（解冻 4,684 参数的读取端卷积，
  基础 LR 5e-4×0.1=5e-5）。
不得借用 densefix 配置（其 gate=fixed 违反可学习门要求）。判读：
D5-A vs 历史 Mask（0.6090/0.6133）筛门控初值作用；D5-B vs D5-A 筛
PE 适配作用；候选须超 Point+Box 0.6144 且经配对复跑确认。
本报告 §5 的"+0.015 空间"是动机参考，非收益承诺。
