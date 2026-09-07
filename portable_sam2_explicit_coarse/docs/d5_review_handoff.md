# 阶段诊断复审与 D5 交接意见

日期：2026-09-06。接收方：后续负责实施的 agent。

状态：本文件记录 Codex 已完成的审查及建议实施方向；D5 尚未由本次工作实现或启动。
用户本次要求是将意见写入独立 MD 通信，不是授权自动训练、提交或推送。
接收方开始工作前应重新检查工作区和用户最新授权，不把文档中的待办当作已执行结果。

## 1. 审查对象与结论

- 阶段报告：[p2_dense_frozen_diagnostic_report.md](p2_dense_frozen_diagnostic_report.md)。
- 探针：[p2_dense_diagnostic.py](../inference/probes/p2_dense_diagnostic.py)。
- 全量工件：`/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/refactor_golden/d3_diagv2_full.json`。
- 子集工件：同目录 `d3_diagv2_subset.json`。
- 被诊断权重：D3fi `best_model_epoch12.pth`，raw dense、P2 cap=0.30、aux=0.20。

结论：v2 已修复 v1 的零检测图漏计、GT 实例并集、残差增强不分方向三项主要问题。
现有证据足以推进最小短训筛选，不需要再次推翻重做整套诊断。
下一步优先测试已有 dense 路径的门控初值与 mask_downscaling 训练适配；
暂不新增学习型画布网络，不继续增大 P2 残差。

## 2. 已核实的证据与边界

全量四格均记录 627 图、627 唯一图像、15,926 GT、533 张有检测图。
探针代码将零检测图一并送入 COCO 评估；未匹配 ROI 保留学习画布。

| 冻结权重干预 | segm/mAP | 与基线平均 mask IoU |
|---|---:|---:|
| learned，当前 alpha | 0.625335997 | 自比较基线 |
| learned，alpha=1 | 0.626174291 | 0.973159 |
| matched-instance-GT，alpha=1 | 0.640702000 | 0.969163 |
| dense off | 0.624284541 | 0.994273 |

同为 alpha=1，GT 相对 learned 提高约 0.01453；GT@1 相对原基线提高约 0.01537。
这说明 dense 不是死通路，decoder 能利用该实例级诊断提示；单纯推理放大学习画布收益很小。

必须保留以下限制：

1. 全量没有 GT@当前 alpha 格，只有子集数据；内容与强度存在交互是线索，
   不是已经确证“两个独立故障”。冻结 PE 是待检验假说，不是已定位根因。
2. “80% mask 变化”表示变化广泛，不表示每个实例变化巨大；同时报告上表平均 IoU。
3. GT 注入仅匹配 IoU>=0.5 的 ROI，画布为 ±4 二值提示，存在匹配和分布偏移限制。
   +0.01537 不是严格性能上限，更不是训练后可兑现的承诺。
4. 不将冻结推理干预差除以训练重复差，宣称“几倍噪声带”；重复差不能定义 ±0.006 分布。
5. 本次核对的是现有代码与 JSON，未重跑完整推理，也未完成标准推理入口的独立数值复现。
   原报告“与 v1 baseline 相同”不能替代标准入口一致性验证。

Pass A 子集原始计数：纠正 37,323 像素、破坏 27,398 像素，净纠正 9,925。
可以说“按当前匹配/栅格化口径净像素纠错为正”，不能比较分母不同的
2.46% 与 1.14% 来论证，也不能把像素净正直接解释为 AP 净正。
子集 JSON 的 ROI 数仍是旧计数口径，不能引用；int 裁剪与训练 target 的边界差异也仍存在。

以上措辞建议同步修正到原报告；本次交接未覆盖原报告，以保留作者原始记录。

## 3. D5：先在 Mask 行做两臂，P2 均关闭

目标：先使 Point+Box→Point+Box+Mask 出现稳定增益，再恢复 Mask/Full 配对。
两臂不是新模块设计，也不包含 GT 画布训练。

| 配置 | D5-A：门控对照 | D5-B：dense 适配 |
|---|---|---|
| explicit_prompt_mode | points_box_dense | points_box_dense |
| SHAPE_DENSE_TRANSFORM | raw_logits | raw_logits |
| SHAPE_DENSE_DETACH | 0 | 0 |
| gate 模式 | global_sigmoid，可学习 | 同左 |
| SHAPE_DENSE_ALPHA_INIT | 0.5 | 0.5 |
| PROMPT_ENCODER_TRAIN_MASK_DOWNSCALING | 0 | 1 |
| PROMPT_ENCODER_LR_MULT | 0 | 0.1 |
| P2_BOUNDARY_REFINER_ENABLED | 0 | 0 |

注意 alpha=0.5 是可学习门的初值，不是 fixed=0.5。
不得盲目使用 densefix 配置而顺带将 gate 改为 fixed。

共同协议：两卡、每卡 batch=1、accum=4、有效 batch=8、AMP=1、
WHU train/val=10%/10%、seed=44、15 epoch、每 3 epoch 验证、val batch=2、
EMA 全关、full_image+roi_balanced_dice、dense temperature=1。
LR=5e-4、weight decay=0.05、warmup=100 optimizer steps；增强及其余超参保持 D 系列。
两臂使用相同的基础初始化来源和训练流程；不得单臂从 D3 best 续训再与原始 Mask 从头训练比较。
显式设置 RUN_IN_BACKGROUND=0；独立 run tag/目录；两臂若并行须不重叠 GPU 且端口不同。

建议新入口名（尚未创建）：

- `scripts/ablations/whu_fi_d5a_mask_gate050_10p_2gpu.sh`
- `scripts/ablations/whu_fi_d5b_mask_gate050_pe010_10p_2gpu.sh`

## 4. 关键实施风险：解冻开关与遥测不能只看名字

mask head 支持独立解冻 `prompt_encoder.mask_downscaling`，不必解冻点/框 embedding。
但必须验证环境变量经过当前选用的配置实际写入
`cfg.model.roi_head.mask_head.prompt_encoder_cfg.train_mask_downscaling`。
只 export 环境变量不等于配置已经接通；若须补布线，应保留旧默认行为与旧权重兼容。

当前 trainer 的 dense pathway 参数组仅含 ShapePriorInjector 与 dense gate，
**不含 PE mask_downscaling**。现有 dense 梯度/更新非零不能证明新解冻参数在学习。
需要独立的 PE-mask-downscaling 检查或监控，不得静默改变旧 dense 字段的统计定义。

验收至少包括：

- 模型构建后核对全部 trainable 参数名：只额外解冻预期 mask_downscaling，点/框 embedding 保持冻结。
- 参数进入 optimizer，B 臂 PE 实际基础 LR=5e-5（乘 warmup/schedule 后按当步值判断），无遗漏/重复参数。
- 仅最终 `loss_mask` 对该参数组的梯度有限且非零；不是只看 total loss。
- 首次非零 LR 更新与 epoch 前后参数实际变化；DDP 同步审计通过。
- 若新增字段：明确 parameter-group L2/分母/DDP 聚合/采样范围，并同步 `DEBUG_FIELDS.md`。
- 不把开头 warmup 的零学习率步骤视为更新失败。

## 5. 验证与交付清单

在用户另行授权实施后，由接收 agent 按以下顺序完成：

1. 检查 git status；保留用户未提交改动，尤其 `main5.tex`、`.zcode/`、历史日志与权重。
2. 实现两臂薄入口、必要配置布线与独立 PE 更新验证；不新增模型模块。
3. bash -n / Python 语法检查；DRY_RUN 完整协议差分；PREFLIGHT 真构建。
   除预期 trainability/LR 及派生指纹、run tag 等外，A/B 配置一致。
4. 小型真训练 smoke 验证梯度、更新和 DDP；记录新模型配置/架构契约，不能绕过校验。
5. 新 checkpoint 严格构建+加载，确认配置重建保留训练适配字段；检查参数 missing/unexpected。
6. 更新 PROJECT_ARCHITECTURE、脚本 README；仅实际新增遥测时更新 DEBUG_FIELDS。
7. 交付修改清单、配置差分、测试证据、未验证事项和可复现启动命令。
   正式短训/全量训练的启动遵守用户最新授权；本 MD 本身不授权启动、commit 或 push。

## 6. 结果判读与后续分支

- D5-A 对原 raw Mask：筛选提高门控初值的作用；历史参考 best 为 D1fi Mask 0.6090 / D2fi Mask 0.6133。
- D5-B 对 D5-A：筛选额外解冻 PE 的作用；候选仍需超过 Point+Box 0.6144，并复跑关键配对确认。
- 主口径仍为 best segm/mAP，同时报告末轮与全部验证时点；不事后换选权重口径找正收益。
- 候选胜出不等于正式采用；小差距需重复训练证据，图像 bootstrap 不能替代训练重复。
- 机制旁证包括 gate、实际 dense 注入、PE 独立梯度/更新、coarse 质量。
  D5 为 P2-off，不运行要求 P2-enabled 的 Pass A 并假称 D5 的 P2 机制结果。
- dense 候选成立后，固定同一表示、门控与 PE 训练策略，再做 Mask/Full 配对。
- 若失败，仅说明该适配配方未获益；不自动改造新画布网络，不宣判 dense/P2 永远无效。

交接结论：先把现有 dense 路径的训练适配测清楚；P2 暂停调参，架构扩张暂缓。
