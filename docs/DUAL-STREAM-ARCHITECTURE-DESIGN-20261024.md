# 双流架构设计文档：从发现到联合训练最终形态（2026-10-24）

本文件汇总：边界弱点量化 → 十四次消费失败根因 → 唯一可行方向（几何候选）→ BRR 边界精修设计 → 从头联合训练的完整架构与损失精简。提交主代理审核。

---

## 一、纯卫星模型的边界弱点（量化）

基线：EXP-0025 v1 best ep71，test 459 场，cap250+maxDets250。

| 物体尺寸 | 整体 AP | 边界 AP(7px) | 塌落率 |
|---|---|---|---|
| 大 | 0.921 | **0.532** | **42.3%** |
| 中 | 0.825 | 0.668 | 19.1% |
| 小 | 0.671 | 0.638 | 4.9% |

**弱点集中在大建筑轮廓**：标准 IoU 被大面积稀释，错误高度集中在 7px 以上的结构性偏移——被圆化的转角、被抹掉的凹槽/内院、整段轮廓偏移、阴影侧过冲、粘连邻居。这些是**多像素问题**，不是亚像素问题。

---

## 二、十四次消费失败的证据与根因

### 2.1 探针舰队判决（本机 26 训练 = 13 变体 × real/constant 双臂）

| 探针 | 变量 | 内容差(correct − constant) |
|---|---|---|
| D1 宽通道 128 | 内部维度 32→128 | **−0.16pp ✗** |
| D2 四头 4×8 | 头数 1→4 | **−0.12pp ✗** |
| D3 多尺度源 | P0→P0+P2 | **−0.19pp ✗** |
| D4 候选 8→10 | 跨视角截断 | **−0.09pp ✗** |
| D5 深度步长 2→1 | 锚点密度 | **−0.07pp ✗** |
| D6 几何只进 K | V 纯内容 | **−0.22pp ✗** |
| P1 BN+ReLU 供给 | SCVT4 归一化 | **−0.02pp ✗** |
| P2 α=0.1 起步 | SCVT4 非零参与 | **−0.16pp ✗** |
| P3 高度门控 V | SCVT4 高度先验 | **−0.12pp ✗** |
| P4 几何查询嵌入 | SCVT4 几何路由 | **−0.12pp ✗** |
| N1 边界监督辅助头 | 梯度拓扑修复 | **−0.12pp ✗** |
| N2 注入后移 post-act2 | 线性通路 | **−0.09pp ✗** |
| N3 可学温度 τ | 注意力锐度 | **−0.08pp ✗** |

**全部 13 个变体内容差为负**——把 UAV RGB 换成常量，模型反而更好。优化器在训练中学到的是依赖常量统计结构而非真实图像内容。

### 2.2 四重根因（全部已被证据锁定）

| 根因 | 证据 | 适用范围 |
|---|---|---|
| **(a) stride-4 分辨率天花板** | SAM2 mask head 是 1×1 conv 打在 stride-4 特征上，输出 4px 网格 logits 后双线性 ×4 上采样。网格之间的"亚像素"全是插值，不是证据 | 所有特征层注入方案 |
| **(b) 梯度饥饿** | 收敛冻结头上，mask loss 对新增输入方向的梯度 ≈ 0。13 个变体换了前向算子的每个自由度，全部无效 | 所有冻结基座端到端方案 |
| **(c) 跨视角亚像素配准损耗** | z-buffer 前景边缘膨胀 1–3px + stride-4 采样误差 | 所有跨视角特征采样 |
| **(d) 信息冗余** | 卫星图已把大建筑**粗轮廓位置**看清（整体 AP 0.92）；缺失增量是多像素结构细节（7–20px 偏差），特征层融合路径的步长量化把它吞掉 | 所有密集特征融合 |

**核心教训**：无论注入算子多宽/多少头/什么源/什么几何路由/什么归一化/什么注入位置/什么温度——在冻结收敛基座上的特征层注入**系统性不可学**。

---

## 三、唯一可行方向：几何实例候选补充（已跨机复现）

### 3.1 机制

```
5×UAV深度+相机 → z-buffer投影 → 5ch几何图512² → 小CNN(203K) → 实例中心/框/objectness
                                                                      ↓
强SAT前缀(已保留) ──→ 去重+NMS ──→ 冻结掩码路径解码新增实例 ──→ 最终实例集
```

**绕开全部四个根因**：
- 不做特征融合（离散框坐标接口，无 stride-4 天花花板）
- 不依赖冻结 RPN/分类器（自身 objectness + 直接实例监督）
- 不做亚像素配准（框坐标只需整像素精度）
- 不与卫星特征竞争（信息冗余不适用——找的是卫星漏掉的实例）

### 3.2 复现结果（本机）

| 比较 | Mask AP | bbox AP | Boundary 2% | Boundary 1% |
|---|---|---|---|---|
| **净增益(vs 强SAT)** | **+0.4672** [+0.297,+0.616] | +0.359 [+0.246,+0.535] | +0.522 [+0.334,+0.611] | +0.471 [+0.304,+0.551] |
| **内容差(vs 常量)** | **+0.2468** [+0.108,+0.350] | +0.244 [+0.149,+0.414] | +0.258 [+0.116,+0.349] | +0.344 [+0.106,+0.332] |

全部 CI 下界 > 0；数值复现 ±0.002pp；训练复现（同 init 独立重训）一致。

### 3.3 增益来源

- 新增实例（高密度图新增 811 个 IoU≥.5 匹配 vs 常量 476）
- 集中在 >100 实例图（688/811）和小/中实例（451/359/1）
- **旧实例掩码完全不变**——边界增益来自新实例的正确边界，不是修正旧边界

---

## 四、BRR 边界精修设计（对抗讨论产出，待探针验证）

### 4.1 为什么特征层注入修不了已有边界

四个根因叠加：stride-4 天花板 + 梯度饥饿 + 配准损耗 + 信息冗余。13 个变体证据覆盖了全部算子自由度。

### 4.2 BRR（Boundary Re-classification Refiner）

**核心思想**：不做 offset 回归（拓扑病态 + 均值回归），做**边界带内逐像素重分类**。

- **输入**（逐实例，box 外扩 32px，native 512²，全卷积）：
  1. frozen soft mask logits（1ch）
  2. UAV 几何图 5ch（height/dx/dy/view-count/support）
  3. signed distance transform（1ch，编码当前边界位置）
  4. 邻居实例 mask 之和（1ch，处理粘连）
  5. 卫星 RGB（3ch，消融项，默认关——阴影是原始误差来源之一）
- **输出**：1ch 逐像素 logit，只在 |SDT| ≤ k 带内生效（k 消融自 {6,8,12,16}）
- **带内硬约束**解决两个经典死法：
  - 拓扑病态：重分类天然能挖凹槽、删突起（warp 做不到）
  - 均值回归："带内保持原判"是平凡可学的 copy 策略，58% 已对的实例学会不动
- **网络**：6-8 个 conv block，32-64ch，~200K 参数
- **损失**：带内 focal BCE + Dice，逐实例等权
- **训练数据纪律**：输入必须用冻结模型在 train 集**真实推理输出**（不是 GT 加噪）
- **content delta 在输出层测量**——13 个 adapter 的核心教训
- **采纳线**：B-AP_l(7px) ≥ +1.0pp 且 AP_l 回退 ≤0.1pp 且输出层 content delta ≥0.5pp

### 4.3 第 0 步（零训练风险泄压阀）

- **误差分类学**：对过不了 7px 的大实例，统计误差幅度直方图 + 拓扑类型占比（圆角/缺凹槽/偏移/阴影/粘连）+ 与密度相关
- **零训练 snap 探针**：frozen mask 边界沿法向 ±5px 搜索高度梯度极大值做贪心吸附。如果 +1pp B-AP_l → headroom 存在；如果 0 → 几何图本身是瓶颈，先修 z-buffer

### 4.4 与 UDPR 的关系

| 维度 | UDPR | BRR |
|---|---|---|
| 位置 | 解码器内（梯度耦合主干） | 解码器外→训练内（从头联合时梯度可回流） |
| 信息 | 仅卫星 stride-4 特征 | 卫星输出 + **UAV 几何** |
| 分辨率 | stride-4 (128²) | **native 512²** |
| 密度 | 64 稀疏点 | 带内逐像素 |
| 当前效果 | 边界仍 42% 塌落 | 待验证 |

**探针阶段共存不冲突**（串行管线，BRR 输入已含 UDPR 修正）。
**从头联合训练时 BRR 替代 UDPR**（同功能、更好信息、更高分辨率）。

---

## 五、从头联合训练最终形态

### 5.1 初始化

```
SAM2 backbone (Hiera+FPN)  ← 加载 sam2_hiera_base_plus.pt 预训练
RPN / RoI bbox / 几何头 / BRR ← 随机初始化
SAM2 mask decoder           ← SAM2 预训练权重
```

### 5.2 架构

```
卫星 RGB ──→ SAM2 backbone 🎯 ──→ features ──→ RPN 🎯 ──→ proposals
                                                          │
UAV 深度+相机 ──→ 几何图(纯数学) ──→ 几何头 🎯 ──→ 新增候选
                                                          │
                   ┌──────────────────────────────────────┘
                   ↓
             SAM2 mask decoder 🎯 → mask logits
                   │                    (不再有 UDPR)
                   ↓
             BRR 🎯 (训练内, native res, +UAV几何)
                   │
                   ↓
             refined mask → loss_mask + loss_brr
```

### 5.3 损失集——7 项（vs 当前单流 8 项 + 新增 2 项 = 10 项）

| # | 损失 | 来源 | 权重 |
|---|---|---|---|
| 1 | rpn_loss_cls | anchor 分类 | 1.0 |
| 2 | rpn_loss_bbox | anchor 回归 | 1.0 |
| 3 | loss_cls (RoI) | RoI 前景判别 | 1.0 |
| 4 | loss_bbox (RoI) | 框精修 | 1.0 |
| 5 | loss_mask | 掩码主任务 | 1.0 |
| 6 | loss_geometry | UAV 实例发现（focal+L1） | λ_geo (从0.1起调) |
| 7 | loss_brr | 边界精修（带内BCE+Dice） | λ_brr (从0.1起调) |

**删除（vs 当前单流）：**
- ~~loss_decoder_tail (UDPR)~~ → 被 BRR 替代（同功能，更好信息，更高分辨率）
- ~~loss_shape_prior (coarse)~~ → SAM2 预训练初始化已提供掩码先验，从头训练时无需额外教粗形状

### 5.4 关键设计决策

| 决策 | 理由 |
|---|---|
| BRR 在训练内（非后处理） | 梯度可回流 decoder，教 decoder 产出更适合精修的 mask；UDPR 做不到这一点 |
| 删 UDPR 不删 BRR | BRR 有 UAV 几何（双流的核心价值）；UDPR 只有卫星信息（已被证明不够） |
| 删 coarse shape_prior | SAM2 init 已会做通用分割；从头训练时主掩码损失足够引导 |
| 保留 loss_cls (RoI) | 从头训练时需要学前景/背景判别（不像 E73 已收敛时饱和） |
| 几何头损失权重独立 | λ_geo 从 0.1 起调，避免多任务初期几何损失淹没检测损失 |

### 5.5 训练预算与排程

- 100ep，同当前单流协议（DDP4、batch 8、AdamW、cosine）
- backbone lr 1e-5（SAM2 预训练微调），head lr 1e-4（新组件）
- Kill 门：val segm/mAP 在 ep10 前 < 0.5 → 停（几何分支干扰检测）
- 对照：同预算纯卫星从头训练（作为公平基线）
- 验收：双指标(净增益 CI>0) + 边界指标(B-AP_l 改善) + 内容差(常量控制)

---

## 六、当前单流训练的真实结构（审核参考）

当前 E73 模型一次 `model.loss()` 产生 8 个损失项：

```
rpn_loss_cls + rpn_loss_bbox          ← RPN anchor 级
loss_cls + loss_bbox (RoI)            ← RoI 检测
loss_mask                             ← 最终掩码 (SAM2 prompt-based, 512²)
loss_decoder_tail (UDPR)              ← 64 点 stride-4 边界残差 (权重1.0)
loss_shape_prior (coarse)             ← 32² 粗掩码 BCE+Dice (权重0.1)
[内部 shape point mining + dense prompt]
```

训练日志（ep100 实测占比）：det 39.2% / final_mask 52.5% / shape 8.4%。

全部损失通过共享 backbone 反传。SAM2 mask decoder 以框为 prompt（非标准 Mask R-CNN），UDPR 在其 stride-4 输出上选最不确定的 64 个 logit 位置做有界残差写入。

---

## 七、工件索引

- 边界评估：`probes/CITYSAMPLE-BOUNDARY-AP-20261021/`
- 探针舰队（26 训练 + 评估）：`/data2/wangcheng/remote-lthpc/runs/{d1..n3}_{real,const}/`
- 几何方向复现：`docs/GEO-DIRECTION-REPRO-20261024.md`
- RGB 增量预注册：`docs/GEO-RGB-INCREMENT-20261024_preregistration.md`
- 远端结论：lthpc `docs/dual_stream/UE_UAV_EFFECTIVE_DIRECTION_VERDICT_20261024.md`
- 架构文档：lthpc `docs/dual_stream/UE_GEOMETRY_INSTANCE_SUPPLEMENT_ARCHITECTURE_20261024.md`
- SCVT4 参考：`docs/SCVT4-FORENSICS-20261016_conclusion.md`、`/data2/wangcheng/remote-lthpc/scvt4_ref/`
- S²DB-Net 精读：`docs/S2DBNET-READING-20261022.md`
