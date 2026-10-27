# UBR 激进提速方案：复用原生 RoI 解码，直接消费 UAV 证据

日期：2026-10-27。状态：设计草案，尚未实施。本文件记录侧会话提出的下一版方案，不修改当前训练、模型代码或已固定实验合同。

## 1. 目标与核心选择

将 UBR 改为原掩码解码器内部的轻量证据融合头：复用本次 SAM2 解码结果，在 native RoI 网格内读取 UAV 外观和深度几何，生成掩码残差，由主 mask loss 直接监督。

优先同时解决三个问题：

1. 避免完整预测池上下文和 active 实例的重复解码。
2. 避免逐实例执行 512² 整图卷积。
3. 统一主任务与 UBR 的训练实例、坐标及监督目标，减少独立 loss 和接线状态。

这是模型与监督合同的调整，不宣称与现有全图 UBR 非零输出或训练轨迹等价。

## 2. 完整结构

```text
卫星 + 现有 G 上游
    │
    ├── 检测路径、UAV 实例发现头及其 loss：保留
    │
    └── 原 SAM2 mask decoder：同一组 RoI 只解码一次
                     │
              native 特征 32ch + mask logits
                     │
           ┌─────────┴─────────┐
           │                   │
      同 RoI 读取 UAV 特征   同 RoI 读取深度几何/支持
           └─────────┬─────────┘
                     │
             native RoI 轻量融合头
                     │
              refined native logits
                     │
             sigmoid → 原 bbox paste
```

训练直接使用原 mask 分支已经采样的正 RoI、GT assignment 及本次 decoder 特征。推理直接复用原预测 RoI 的同一次解码结果。新增特征捕获必须保持实例顺序和 decoder 输出合同，避免额外解码或读取过期缓存。

## 3. 删除的路径

- 不再为 UBR 邻居上下文额外解码完整 250 候选池。
- 不再为 UBR 重新解码最多 64 个匹配正预测。
- 不再将每个实例的 32ch 特征贴至 512² 后做空间卷积。
- 删除 UBR 的 SDT、邻居掩码及相关 CPU/GPU 往返。
- 删除独立的 UBR 预测匹配训练人口；沿用原正 RoI 的监督分配。
- 删除独立 UBR 边界带 loss。

保留原检测与补充候选策略、实例发现头及其直接监督。共享候选分类确认等后续工作不混入本轮尾部改动。

## 4. UAV 证据与坐标

首版复用现有 G 路径的卫星坐标 UAV 对齐特征，不新增跨视角注意力网络。

| 输入 | 通道 | 职责 |
|---|---:|---|
| native decoder feature | 32 | 当前实例的语义与形状状态 |
| 对齐 UAV feature | 32 | 补充外观及已有跨视角聚合证据 |
| 深度几何与支持标记 | 5 | 高度变化、几何形状及有效性 |
| 当前掩码概率 | 1 | 提供已有预测状态，首版建议 detach |

以当前约 128² 的 native 网格为准，实际尺寸由 decoder 输出核定。对 bbox `(x0, y0, x1, y1)`，令 `w=x1-x0`、`h=y1-y0`，网格位置对应卫星坐标：

\[
x=x_0+(j+0.5)\frac{w}{W},\qquad
y=y_0+(i+0.5)\frac{h}{H}.
\]

按此位置采样 UAV 对齐特征和几何，显式处理源特征 stride、像素中心、插值及越界规则；不能把整图特征直接 resize 后当作对应 RoI。该处理沿用历史 native 输出按 RoI 解释的合同，不声称 decoder 特征曾被物理裁剪。

无有效支持的位置屏蔽 UAV/几何证据，仍允许卫星特征驱动修正。支持标记本身不得由 GT 生成。原始多视角特征的聚合压缩仍然存在，本方案不宣称恢复全部 RGB-D 信息。

## 5. 轻量修正头

```text
concat(decoder32, UAV32, geometry5, probability1)  # 70ch
                    ↓
               1×1 Conv → 32
                    ↓
           3 个轻量空间块，保持 native 分辨率
                    ↓
               1×1 Conv → 1
                    ↓
           z_final = z_native + Δz
```

- 末层零初始化，初始输出保持原 native logits，部署继续执行原概率贴图。
- 不增加 alpha、独立质量头、复杂门控或新监督头。
- 首版空间块可采用深度可分离卷积；其实际 GPU 延迟必须实测，低 FLOPs 不自动代表低耗时。
- 不使用 SDT 或边界带写入约束，修正作用于 native RoI 掩码。
- 首先测试正常批量前反传是否可承受，再决定是否启用局部 checkpoint，避免默认叠加重算。

## 6. 监督与 loss

UBR 输出仍处于 native RoI 坐标，可以接受原实例 mask 监督：

\[
L=L_{\mathrm{det}}+L_{\mathrm{mask}}(z_{\mathrm{final}})
  +0.1L_{\mathrm{coarse}}+L_{\mathrm{UAV实例}}.
\]

这里沿用现有 mask loss 类型，不借此次提速额外增加边界 loss。保留实例头及现有内部尺寸项权重；UDPR 不再串联，独立 UBR `.1` 边界带项删除。

建议最终 mask 监督显式设为 native 分辨率，避免仍缩至 28² 后损失细节。GT 应使用正确的 RoI 裁剪目标，不是将全图 GT 直接 resize。该监督变化必须同样应用到所有匹配参考组，并明确损失分母、空正例及 DDP 行为，不能将更高分辨率监督的收益归为 UAV 收益。

主 mask loss 直接训练修正头、原 decoder 及可微的 UAV 消费路径；无需新增 UAV 或 BEV 标签。

## 7. 提速来源与能力边界

主要提速来源是：复用解码、消除 SDT/邻居上下文、空间计算从每实例 512² 降至约 128²，以及缩减修正头宽度与深度。仅空间面积从 512² 到 128² 为原来的 1/16；这不是整个模型的加速承诺。最终须报告完整成功更新和完整推理的耗时及显存。

输出继续沿用 bbox paste，框外增长受限。本模块优先争取 Mask AP、AP75 和轮廓质量；漏检、候选拒伪及 bbox 改善仍由检测/实例发现路径负责，不能预先保证这些指标提升。

单、双流可以共用相同修正头结构，单流不提供 UAV 证据。证明 UAV 内容有效仍需匹配容量、监督与预算的对照；上游已有 G 时，尾部端口归零不能称为纯卫星模型。

## 8. 首轮最小验证矩阵

三组保持相同上游 G、初始化、native 监督分辨率、训练预算、候选与评价协议：

| 组别 | 新修正头 | 尾部 UAV/几何证据 | 判断目的 |
|---|---|---|---|
| A | 无 | 不读取 | 相同监督下的参考 |
| B | 有 | 全零 | 新增修正容量的贡献 |
| C | 有 | 正确配对、有效性屏蔽 | 真实证据相对匹配容量的净贡献 |

先通过真实模型坐标/零初始化恒等、梯度可达、空 rank、完整更新资源及恢复检查，再启动训练。零初始化恒等应对同权重、同 RoI、同解码批次比较；不能拿旧 UDPR 精修结果作为恒等目标。

统一报告完整目标的成功更新秒数、图像吞吐、端到端推理延迟和显存；计时须控制 GPU 共租竞争、候选/正 RoI 数量与精度。最终报告 Mask AP/AP75、Boundary AP 及 bbox/召回，配合正确/常量/错配内容诊断。速度收益、容量收益与 UAV 内容收益分别判断。

本文件不授权停止、重启或修改现有实验，也不将设计预期写成已实现或已有 AP 增益。

## 9. 本会话确认的首版合同（2026-10-27，覆盖前文可选项）

用户已同意主代理最近两轮收敛，并要求子代理实现、主代理审核。本次只实施独立native模块/接线和真实四卡工程门，不自动重启10轮或100轮实验；当前三组已按用户要求停止，产物保留。

1. RGB取G `_early_inject`里`reduce_rgb`的原返回值，即进入`rgb_fusion`前的aligned32；保留同一可微张量，不重新encode或聚合，不用`context[0]`几何混合图。RGB支持来自`index>=0`的`valid.any(1)`，不借用geometry support。多视角平均/候选截断的信息压缩仍存在。
2. RGB消费格是floorUV/4候选桶，中心`4j+2`，不是旧CNN/Hiera的名义格原点；geometry是floorUV像素栅格，中心`j+.5`。RoI中心映射到同一卫星edge坐标后分别采样；RGB有效性加权bilinear、geometry nearest并自有support门；无证据仅置零证据，卫星可修正。源/target坐标、stride/origin显式。
3. Native decoder32和sampledRGB32分别无affine channel LayerNorm；几何沿既定量纲height/100、dxdy/10并clip±5、viewcount/5、support0/1，不另自适应放大。当前prob detach，70ch→1x1/32→三个DW3x3+PW1x1/GN8/GELU残差块→零初始化1x1/1→z+delta。无alpha/SDT/neighbor/band约束/新loss；不强制给无支持处delta清零。
4. 实际UE主mask目标经运行核为512²，而非历史28²。修正仍在native约128²，沿旧插值至512²主CrossEntropyLoss（weight1、原GT RoI crop、原分母）直接监督；不把整个图GT resize，不另切native目标。coarse64/.1、检测和实例头及内部尺寸项不改。UDPR和旧globalUBR均不实例化/串联。
5. A无头/B有头但尾部RGB与geometry全零/C同头正确证据；上游都为相同G正确RGB-D，不是纯卫星；B/C头同初始化，shared模块同初始化。C>B只检尾部RGB-D整体贡献；固定CP端口正确/常量/错配为内容依赖诊断，真实RGB必要性仍须匹配常量训练对照。
6. 工程门：同权重同RoI同chunk下零头raw/consumer binary/boxes/scores精确恒等、每组SAMdecoder只调用一次、主loss仅原有keys、非零头可经主maskloss反传decoder/UAVLoRA、空rank新参数连零及副本SHA、完整成功update/250推理资源、恢复源/状态。零初始化首步只out有梯度不判中间层断路，非零/更新后再核。工程秒数不冒充AP增益。

7. 数值验收补充：新轻量头整体FP32（输入、LN/Conv/GN/GELU、delta），最后转回原logits dtype相加，原decoder/上游仍AMP。真实门低scale32的隐藏梯度下溢已定位；正式512回退256可通过，但采用局部FP32降低此依赖，不改变监督分母或初始化。
