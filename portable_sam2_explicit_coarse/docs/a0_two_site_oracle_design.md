# A0 双位置冻结权重 Oracle 与候选模块设计（v1，待实施）

日期：2026-09-07。状态：**仅设计；尚未新增模型、配置、训练任务或论文结论。**

## 0. 决策目标与边界

以 NWPU A0 的最佳权重（`points_box_dense`、D5-B、P2-off）为共同、冻结的
起点，回答两个不同的问题：

| 路线 | 干预位置 | 要证伪的问题 | 若 Oracle 正向，才允许考虑的候选 |
|---|---|---|---|
| C（canvas） | `ShapePriorInjector` 输出的全图 mask canvas，**进入 PromptEncoder 前** | 更准确的空间先验能否被当前 PromptEncoder + MaskDecoder 消费 | 图像特征/粗 mask 到 canvas 的小型可学习 renderer |
| T（tail） | SAM2 MaskDecoder 已生成的原生 `256×256` 单 mask logits，**在输出 token hypernetwork 后、最终 full-image loss/后处理前** | 不改变 prompt 内容时，decoder 尾端的可部署式局部纠错是否有收益空间 | PointRend-style uncertainty point tail refiner |

两条路线共同固定：检测框、分类/检测得分、2P2N 点、box token、A0 backbone、
PromptEncoder、原 SAM2 MaskDecoder、数据 split 和 COCO 计分。每张验证图（包括
零检测图）都必须评分。任何干预若改变检测输出，立即失败，不报告 mAP。

**GT 的唯一角色**是 Oracle 中生成反事实答案，或后续候选模块的训练 target。
GT 绝不作为候选模块验证/测试期的输入，也不用于不确定点选择。

## 1. 共同实验协议

- checkpoint：优先 A0 的正式 `best_model.pth`；记录其 checkpoint SHA256、epoch、
  `model_config` hash、权重键 hash 和原始完整 validation mAP。
- 数据：NWPU validation 全 130 图；使用 checkpoint 内嵌的 VHR-10 10 类契约；
  不通过文件名或外部环境猜配置。
- 实例匹配：预测 proposal 只能匹配**同类别** GT，再取最大 box IoU，阈值固定
  `0.5`；未匹配 proposal 保持原输出，单独计数，不能借 GT 修正。若另跑 class-agnostic
  格，只能标为宽松 ceiling，不能用作训练 target 或方法上限。
- 输出不变量：逐图 SHA256 记录 `image_id,bboxes,scores,labels`；所有干预格必须等于
  无 hook 标准推理的 detector hash。基准格还必须逐位等于标准推理的完整 prediction hash。
- 排名不变量：T-Oracle 启动前必须从 A0 内嵌 `model_config` 断言 `quality_head` 未启用；
  否则 decoder-return 后改 logits 会进入 quality head 并改变 `mask_scores`/排序。首轮不把
  “mask 修正 + 排名变化”混为一个干预；该断言不成立时停止 T，而不是静默继续。
- 计分：共用 `utils/coco_eval_utils.py`，记录 segm mAP、mAP50、mAP75、AR@100，
  并导出每图 `dt_records/gt_records/images/run_manifest`，使修复后的
  `bootstrap_paired_map.py` 能作 paired bootstrap；不以单一 best mAP 作唯一判断。
- 选择纪律：Oracle 可以预先定义多个预算格以描出上界曲线，但**不得**在 validation
  上从中挑一个再称其为最终方法超参。训练候选前，预算由计算成本和训练集曲线确定；
  最终报告还须独立复跑/CI。

## 2. 路线 C：canvas 内容 × 消费强度 Oracle

### 2.1 已有工具与最小先行格

`inference/probes/canvas_accuracy_probe.py` 已实现这一路线：包装实际调用的
`ShapePriorInjector.forward_roi`，在 PromptEncoder 的 `masks` 输入处，将**匹配
实例**的已交付 canvas 替换为 GT occupancy logits；未匹配实例保留 learned canvas。

先跑最小二维格，不做 blend 插值扫参：

| 格 | canvas | dense gate | 目的 |
|---|---|---|---|
| C0 | learned | checkpoint current | 标准基线，必须逐位复现 |
| C1 | **support-matched GT**：只在原 proposal bbox paste 覆盖的区域替换，框外保持 learned canvas 的原值 | checkpoint current | 原几何支持域内的内容改善 |
| C2 | learned | 强制 `alpha=1` | 单独测消费幅度改变/分布偏移 |
| C3 | support-matched GT | 强制 `alpha=1` | renderer 可对照的“内容×读取”联合上限 |
| C4 | **full-image GT**（仅匹配实例） | 强制 `alpha=1` | 放宽 bbox 支持域后的不可部署 ceiling，不可作为 ROI renderer 的可达上限 |

运行形式（A0 完成且 GPU 空闲后）：

```bash
CUDA_VISIBLE_DEVICES=<one_gpu> "$PYTHON" inference/probes/canvas_accuracy_probe.py \
  --checkpoint <A0_RUN_DIR>/best_model.pth --device cuda:0 \
  --blends 0 1 --gate-alphas current 1.0 \
  --output <DIAGNOSTIC_DIR>/a0_canvas_oracle.json
```

原 probe 的 full-image GT 替换会改变原 learned canvas 的 bbox paste 支持域；因此它只能
承担 C4，不可再把它称为 ROI renderer 的上限。必须先实现 C1/C3 的 support-matched
替换、records/manifest 导出和 label-aware matching，才可作正式判读。

`C3-C0` 是“原支持域内准确内容加全开读取”的联合上限，不能归因给 renderer，也不能
直接当作可训练模型预期收益。`C1-C0` 接近零而 `C3-C0` 为正，只能说明内容与读取强度
存在联合交互；须连同 C2 和 C4 判读，不能单独断言 gate 是唯一瓶颈。C1/C3 均接近零才
停止 ROI CanvasRenderer；C4 明显正而 C3 为零时，结论是 bbox 支持域/几何才是限制，
不是“画布内容无用”。

### 2.2 只有 C 路线过门时的训练候选

候选 `CanvasRenderer` 输入仅为 A0 已有的每 ROI 图像/FPN 特征、proposal geometry 与
raw coarse logits，输出全图 `256×256` canvas residual/logits；不能输入 GT、匹配 ID、
验证期实例 mask 或未来 P2 的 GT support。第一版不接 P2，避免把 renderer 效果与 P2
质量混淆。

训练从 A0 `--init-from` 热启动：冻结 detector、backbone、原 `ShapePriorInjector`、
PromptEncoder 和原 SAM2 decoder；只训练 `CanvasRenderer`、既有 dense gate 以及（若
被纳入共同候选定义）PE `mask_downscaling`。renderer 以 train split 的 matched GT
instance canvas 做 BCE+Dice 监督，最终 mask loss 仍保留。验证期使用 renderer 预测。

## 3. 路线 T：decoder-tail 的 PointRend-style Oracle

### 3.1 为什么不是“再加若干点”

历史 E0/E1 已表明，在已共适应的 decoder 上把额外 sparse point token 再送回 decoder
会伤害结果；因此 T 路线**不增加 sparse token、不进行二次 decoder 调用**。它只读取
原 SAM2 单 mask token 已生成的 native-grid mask logits，并在输出端局部替换不确定点的
答案。这是 PointRend 的“在不确定点做高分辨率分类”思想，不是提示挖点的复刻。

SAM2 的实际计算顺序是：`transformer -> upscaled_embedding -> mask-token
hypernetwork -> [N,1,256,256] logits`。T 插入点固定为最后一项之后，随后仍沿用 A0 的
full-image resize、`roi_balanced_dice` 训练 target、COCO 后处理。它不会触碰 detector
或 prompt 构造。

### 3.2 T-Oracle：只回答“可部署选择规则覆盖的错误值不值得修”

对每个预测 ROI：

1. 先完整运行 A0，得到 native `256×256` logits `z`；
2. 用**只依赖 `z` 的固定规则**选择候选点：`abs(z)` 最小的 top-K 点；禁止借 GT、
   IoU、coarse 或最终错误图来选择坐标。首轮不增加第二 selector，以免在 validation
   上从多个选择器中挑胜者；
3. 仅为匹配 IoU≥0.5 的 ROI，将这些坐标处的 `z` 替为 GT 签名的常数 `±L`
   （默认 `L=8`，应大于阈值而非根据验证结果调节）；未匹配 ROI、未选点均原样保留；
4. 继续标准 A0 full-image postprocess 和 COCO scoring。

预注册的预算曲线为 `K∈{0,64,256,1024}`、固定 `L=8`。输出还必须报告：匹配率、
选点中真实错误比例、错误覆盖率、正/负错误拆分、每 ROI 实际选点数、改动像素数以及
输出 mask 相对 C0 的 IoU 分布。

这不是“GT 画出完整正确 mask”的无意义上界：它限制 GT 只能回答部署时可由不确定度
规则选到的少量位置。若 T-Oracle 仍无正向变化，PointRend tail 没有可利用的点预算，
停止 T 路线；若随 K 明显上升，才证明值得学习一个点分类器。

为分离“选点覆盖不足”与“tail 本身无价值”，可附加 **诊断专用、不可部署** 的
`T-error-oracle`：在同一 K 预算下由 GT 错误点选坐标后再注入 GT 标签。它只能说明
选择器的理论损失；不可与 T-Oracle 或训练候选比较，更不可写入论文主结果。

### 3.3 T-Oracle 的实现形态与验证

新增独立的 `inference/probes/decoder_tail_oracle_probe.py`，在 mask-head **紧接 decoder
return** 的位置只替换 selected native-grid logits。`RSSAM2MaskDecoderWrapper` 当前只
返回 logits、IoU 和 mask token，未暴露 `upscaled_embedding`；因此训练候选之前必须先做
一个**不改 A0 行为的 API/hook feasibility spike**，验证能可靠取到该张量并以标准 A0
逐位前向等价、严格 checkpoint load 通过。T-Oracle 本身只需 logits，仍可先实现。其实现必须：

- 先无 hook 标准推理，再执行 `K=0`；两者完整输出 hash 必须逐位相等；
- 对所有 K 断言 detector hash 不变；
- 读取 A0 的 `quality_head` 开关并执行 §1 的硬失败断言；不得通过重算/修改 quality
  score 把排名变化伪装成 point tail 收益；
- 复用训练 `get_full_image_targets(..., nearest 256)` 的唯一 target 映射实现；并做
  native-grid 单点 impulse 经 full-image postprocess 的坐标单测，避免 ROI crop、bbox
  paste 或额外 resize 制造坐标假象；
- 所有 130 图进入 evaluator，零检测图不得过滤；保存 records 与 run manifest；
- 单元测试覆盖 selector 无 GT 依赖、K=0 identity、未匹配 ROI 不变、边界/图像边缘坐标、
  单点和空预测；
- 不修改训练模型代码、architecture ID 或 checkpoint，故 probe 失败可直接删除且不会
  污染 A0/A1/A2e。

## 4. 只有 T 路线过门时的可学习候选

候选名暂定 `DecoderTailPointRefiner`，是新的架构，必须有新 `cfg.model` 开关和新
architecture fingerprint；不能把它伪装成 A0 或原 Full。

对每个由当前预测 `abs(z)` 选出的 point，head 输入为：

`[upscaled_embedding at point (32d), selected mask-token embedding (256d), z,
 normalized x/y]`。

MLP 输出 refined point logit；它替换该点的 `z`。训练时 A0 主体冻结，head 在 520 张
训练图上以**同一无 GT 的 top-K selector**学习 point BCE（可对前景/背景平衡），并以
固定 K 的输出参与最终 mask loss。首版不混入随机点，以免引入 selector 分布偏移；若
后续需要探索点，必须预注册比例并分别记录两类点的 loss。验证 selector 必须仍是无 GT
的固定 top-K。第一版不训练 detector、不改 box score、不解冻整个 SAM2 MaskDecoder。

在 native 256 grid 的 1 点修正过于稀疏时，候选可使用固定的局部 bilinear scatter
（半径和权重预注册）将 point logit 写回邻域；不得在验证期按 GT 自适应膨胀。这个版本
仍属于 decoder tail，而非 canvas/P2；若采用，它在消融矩阵中需单列，不能宣称为原始
PBM/Full 的 mask 或 P2 效应。

## 5. 双路线门与资源次序

| 条件 | 动作 |
|---|---|
| C1/C3 对 C0 均没有超过 paired-bootstrap 噪声带的正向信号 | 不实现 ROI CanvasRenderer |
| C3 正向，且 C1/C2/C4 能定位到内容、读取或支持域 | 先实现最小 CanvasRenderer short run |
| T 在任一预注册 K 下无正向信号 | 不实现 DecoderTailPointRefiner |
| T 随 K 明显正向，且 T-error-oracle 显示 selector 不已是主要限制 | 实现冻结 A0 的 tail-head short run |
| 两者都正向 | 先做训练参数更少、仅输出端的 T；C 保持独立，绝不把二者同时加入首轮 |
| 两者都失败 | 停止新增 mask/P2 模块；保留 A0/PB/PBM 的现有结论，转向正式矩阵与论文证据整理 |

“正向”不预先写成单次 mAP 阈值：至少需要完整 validation 上与 A0 同预测记录的 paired
bootstrap CI 不跨零，且候选短训独立复跑方向一致。Oracle 的正向只是启动训练的必要条件，
不是论文效用的充分条件。

## 6. 本轮工程清单（按顺序）

1. 先修订 canvas probe：support-matched C1/C3、full-image C4 明确分离、label-aware
   matching、每图 bootstrap records/manifest 与完整指标；再在 A0 best 上完成 C0–C4。
2. 仅新增、审查并 smoke `decoder_tail_oracle_probe.py`；在同一 A0 权重完成 T 的
   `K={0,64,256,1024}` 曲线与可选诊断 `T-error-oracle`。
   K 是**每 ROI**预算，必须额外报告每图总改点数/比例，防止 ROI 数差异伪造预算比较。
3. 为 T 候选先完成 `upscaled_embedding` 的 API/hook feasibility spike；未通过不得宣称
   该候选的 32d 输入已接线。
4. 汇总两条 Oracle 的记录、逐图 paired bootstrap 和运行时/显存；写一页裁决报告。
5. 只有一个路线通过时，才新增对应模型、配置、训练日志字段、短训 wrapper 与新架构 ID；
   先 `--build-only`/严格加载/1 epoch smoke，再挂 100ep 候选。

在步骤 1–3 前，**不得**将 GT canvas、GT point 标签或任何 hook 写入训练主线；A1/A2e
也不作为这两个通用路线的主判据。A0 是唯一不含 P2 的干净共同底座。
