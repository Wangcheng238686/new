# A0 Signed EDT-Confidence Dense Canvas：冻结门与实施规格

**状态：冻结 probe 已实现并完成；严格进入门失败，停止 A0-D/A0-E 训练。**

本文定义 A0 的下一条 dense 表示候选。它只借鉴 SAMRefiner 的“从不可靠 coarse
mask 中挖掘鲁棒提示”原则，不宣称复现 SAMRefiner，也不将 R3 作为该候选的输入源。
任何实现、实验命名、论文表述和结论均以本文的边界为准。

## 1. 问题与决策

matrix300 的单 seed E300 结果表明，R3 不是更优 dense source：R3 last
segm/bbox mAP 为 `0.6273/0.7127`，低于 A0 的 `0.6330/0.7218`；同口径
learned canvas IoU 也为 `0.7614 < 0.7821`（A0）。因此停止把 R3 renderer
作为 dense-source 主线。

本候选只回答一个更窄的问题：**对同一个 A0 coarse source，PromptEncoder 是否更
适合接收“保留形状、降低边界权重的 signed EDT-confidence canvas”，而非 raw
ROI logits？**

不新增损失。coarse 继续只受已有 `BCE+Dice` 监督；SAM2 final-mask loss、dense
gate、PromptEncoder mask-downscaling、decoder 和 detector 均沿用 A0 的既有定义。

## 2. SAMRefiner 参考：论文、源码与不可照抄之处

参考：

- 论文：<https://arxiv.org/html/2502.06756>
- 官方主流程：<https://github.com/linyq2117/SAMRefiner/blob/main/sam_refiner.py>
- 官方 prompt 工具：<https://github.com/linyq2117/SAMRefiner/blob/main/utils.py>

SAMRefiner 从同一个 hard coarse mask 一起挖掘 distance-guided points、可选
context-aware box 和 mask prompt；每轮选取 SAM 预测 IoU 最大的 mask 后重新挖掘。
我们**不**迁移 CEBox（会改变 box 变量），也不迁移多轮 refinement（会改变 decoder
调用与输入分布）。A0 的 points、detector box 和一次 decoder forward 保持不变。

论文 Eq.(3) 写成以最深前景点为中心的径向 Gaussian；官方代码实际生成的是
foreground 内的 **EDT-depth confidence**：

\[
G(p)=\exp\left[-\frac{(D(p)-D_{\max})^2}{|M|/\gamma}\right],\quad p\in M,
\]

其中 \(M\) 是 hard foreground、\(D\) 是到最近背景的距离。它会保留细长和非凸目标
的内部中心脊，而不是把目标压缩成一个圆形中心。主候选采用这一**官方代码语义**，
不采用单中心径向 Gaussian。

官方的 dense 输入也不是“正负两侧均按高斯衰减”：foreground 为正的 depth-weighted
response，background/padding 为恒定负值。因此本文使用的准确称呼是
**shape-preserving signed EDT-confidence canvas**，不称“双侧 signed Gaussian”。

## 3. 候选张量定义

输入为 A0 `ShapePriorInjector` 的 ROI-local logits
\(z\in\mathbb{R}^{N\times1\times H\times W}\)，默认 \(H=W=64\)。所有实例独立处理。

1. 先显式停止梯度：`z_detached = z.detach()`；
2. `M = sigmoid(z_detached) >= tau`，默认 `tau=0.5`；
3. 对每个非空 `M` 计算 exact Euclidean EDT `D`，并令 `A = M.sum()`、
   `Dmax = D.max()`；
4. 在 foreground 内计算
   `G = exp(- (D - Dmax)^2 / (A / gamma))`；
5. 构造 ROI-local signed canvas：

```text
Q(p) = +omega * G(p),  if M(p) = 1
     = -omega,         if M(p) = 0
```

6. 仅将 `Q` resize/paste 到 A0 原有 proposal-box 对应的 PromptEncoder mask canvas；
   **proposal box 外保持 A0 的 `outside_fill_logit=0`**，不可在全图生成正 Gaussian
   长尾；
7. 继续走原 `_forward_dense_embeddings`：现有 no-mask base、box support、global
   gate 与 SAM2 MaskDecoder 均不变；
8. 空 foreground ROI 必须逐实例退回原 no-mask base（等价现有 invalid-dense 处理），
   不允许产生全负或 NaN canvas。

`omega` 与 `gamma` 不是从 SAMRefiner 直接继承的常数。其 `omega=15/30` 服务于
冻结 SAM 的 mask-input logit 范围；A0 的 PromptEncoder mask-downscaling 已解冻，
必须先在冻结 probe 中预注册小范围扫描后锁定，禁止依据 val 结果继续扩展扫参。

## 4. 为什么必须 detach，以及它不表示冻结 ShapePrior

hard threshold、exact EDT、`argmax`/面积等操作不具稳定可用梯度。若不 detach，
final-mask gradient 对 `z` 要么为零、要么依赖不透明近似，不应伪装为端到端梯度。

detach 后的训练链为：

```text
coarse BCE+Dice  ───────────────────────────────────────→ ShapePrior
final-mask loss → PromptEncoder / dense gate / SAM2 decoder
```

ShapePrior 仍由原 coarse supervision 优化，并未冻结。代价是 A0 raw 默认具有的
`final-mask → raw coarse` 梯度路径被切断。因此训练期不能把 `A0` 与本候选直接称为
“仅 dense 表示变化”；必须采用第 6 节的梯度控制臂。

若未来希望保留该最终任务梯度，应另行提出并验证 soft/differentiable EDT 近似；它是
另一个模块，不能称为本文 hard excavation 的实现。

## 5. 冻结推理门：先证伪，后训练

共同底座为 A0-last checkpoint。探针只改 PromptEncoder 之前的 dense canvas；固定
coarse source、2P2N points、detector boxes、PromptEncoder、gate、decoder、检测输出和
COCO 评分契约。无梯度前向下，detach 本身不构成变量，故可严格比较表示。

最低四格：

| 格 | canvas | 用途 |
|---|---|---|
| R | A0 raw logits（production parity） | 基线与 hash 不变量。 |
| B | hard signed binary：foreground `+omega`、ROI background `-omega` | 分离二值化/负背景的影响。 |
| E | 本文 signed EDT-confidence `Q` | 主候选，仅降低边界正证据。 |
| C | 现存 `gaussian_edt` 的全图正 center-area Gaussian | 反例；量化形状坍缩和框外正长尾，不得作为主候选。 |

每格必须记录：

- strict unhooked parity：R 与标准 A0 的每图预测 hash、检测字段、mAP 完全一致；
- 检测不变量：B/E/C 不改变 bbox、label、score、proposal 顺序；
- canvas 的正/负/零比例、ROI 内外非零比例、空 ROI 比例、`omega/gamma`；
- PromptEncoder 后 `dense_pe-base` 的 norm、cosine、box 内 applied delta；
- matched/unmatched proposal 分层，以及 image-paired ten-class bootstrap CI；
- GT canvas oracle：对同一匹配 GT coarse 分别生成 raw 与 E canvas。若 `GT-E`
  明确低于 `GT-raw`，说明表示本身删掉了对 decoder 有用的形状信息，停止该路线。

主候选进入训练的必要门：E 相对 R 的 image-paired mAP 改善 95% CI 下界大于 0，且
GT-E 不低于 GT-raw。单个总 mAP、单 seed、或仅 canvas IoU 改善均不足以过门。

### 2026-09-11 完整冻结裁决

在 A0 matrix300 last、NWPU validation 130 图、`tau=0.5, omega=8, gamma=4` 下，
standard 与 R 的完整输出逐位相同，所有 cell 的 detector hash 相同。E-R 为 -0.000864，
500 次 image-paired CI 为 `[-0.002590,+0.001470]`（`P(Δ>0)=0.29`），故上述严格门失败。
GT-EDT−GT-raw 为 -0.000035，CI `[-0.000320,+0.000261]`：EDT 没有可测的 GT shape
保真损失，但也未带来冻结收益。B 对照和 C 的负结果不能挽救主门；尤其不得把 E-B 的 CI
错标为 B-R。结论只停止此 hard signed-EDT package，不等同“dense 通道没有内容上限”。
工件与后续总体路线见 `docs/a0_dense_route_verdict_20260911.md`。

## 6. 训练矩阵：优化路径与表示严格分解

冻结门通过后才新增训练条目；P/PB/A0/A3 和既有 R3 结果绝不重跑或改写。

| 臂 | dense representation | final-mask→ShapePrior | 作用 |
|---|---|---:|---|
| A0 | raw logits | 有 | 已存在基线。 |
| A0-D | raw logits | 无（detach） | 仅量化切断梯度的影响。 |
| A0-E | signed EDT-confidence | 无（detach） | A0-D→A0-E 才是纯表示比较。 |

三臂继承相同的 A0 matrix300 协议：数据、epoch、seed、PE mask-downscaling 解冻、
optimizer、gate 初始化、loss、评估与 checkpoint 选择规则均相同。A0-E 不加 BCE、
boundary、distillation、negative 或 gate loss。

若资源只允许新增一个训练臂，可以运行 A0-E 作为**hard-prompt excavation package**
候选，但论文和实验表不得声称它是 raw dense 的单变量替代；缺失 A0-D 时无法分离
表示收益与梯度切断。

## 7. 实现边界与验收

实现应新增独立 transform 名，例如 `signed_edt_confidence`，不得重定义既有
`gaussian_edt` 的历史含义。建议修改范围仅限：

1. `rsprompter/dense_prompt_utils.py`：逐实例 hard mask、exact EDT、shape-preserving
   signed canvas 与 empty fallback；
2. `sam2_mask_head_helpers.py`：通过既有 `_shape_prior_to_prompt_mask` 选择该
   transform，并维持 proposal-local paste；
3. 配置/runner：将 transform、detach、`omega/gamma/tau` 写入 `cfg.model`、architecture
   contract、checkpoint；新增 A0-D/A0-E arm 时默认 matrix300 队列保持不变；
4. `scripts/smoke/test_dense_prompt_utils.py`：验证符号、ROI 外零值、细长/非凸 shape
   保留、empty no-mask、gamma 方向、不可导契约；
5. 新冻结 probe：实现第 5 节四格、hash 不变量、records 与 paired bootstrap。

不要：

- 复用现有全图正 `gaussian_edt` 并改名为 signed；
- 让正 Gaussian 在 proposal 外进入 PromptEncoder；
- 把 CEBox、多轮 SAM refinement、R3、P2 或 UDPR 混入 A0-E；
- 将 SAMRefiner 的论文径向公式、官方 EDT-depth 代码和本项目 ROI paste 说成“逐位复现”；
- 将 Oracle 的 GT 干预或冻结结果写成训练模型收益；
- 在未通过冻结门前启动 300 epoch 训练。

## 8. 论文表述边界

通过所有训练与多 seed 验证前，只能称其为“候选 dense prompt representation”。
若 A0-D→A0-E 稳定为正，论文可声明：在固定 A0 coarse source、sparse prompts、box
geometry 与 decoder 下，shape-preserving signed EDT-confidence encoding 改善了 dense
prompt consumption。不能声明“复现 SAMRefiner”、不能声明“EDT 生成了额外视觉信息”，
也不能把 R3 的负结果包装为该模块收益。
