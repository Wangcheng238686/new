# 双流架构设计文档：从发现到联合训练最终形态（2026-10-24）

> **已被取代**：本文档的 §四 UBR 设计、§五 最终形态与损失精简已由 `docs/UBR-UNIFIED-ARCHITECTURE-DESIGN-20261026.md` 修订定案（UDPR→UBR 换装、损失账本、四风险裁决、实验阶梯）；§一/§二/§三 证据章继续有效。

本文件汇总：边界弱点量化 → 十四次消费失败根因 → 唯一可行方向（几何候选）→ UBR 统一边界精修设计（BRR 与 UDPR 的统一）→ 从头联合训练的完整架构与损失精简。提交主代理审核。

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

## 四、UBR 统一边界精修设计（替代 UDPR 和 BRR，单流/双流共用）

### 4.1 设计动机

单流和双流模型不能在最后挂不同的边界精修模块——否则代码、接口、训练管线全部分裂。
正确做法是**一个统一模块，UAV 几何作为可选输入端口**：
- 单流 = UBR(UAV=off)
- 双流 = UBR(UAV=on) + 几何头

### 4.2 与 UDPR 的功能对比

| 维度 | UDPR（被替代） | UBR（统一替代者） |
|---|---|---|
| 选择方式 | \|logit\| 最小的 64 个点（解码器"最不确定"位置） | \|SDT\| ≤ k 空间带（掩码二值化边界的几何邻域） |
| 输入 | 每点：上采样特征 32ch + mask token 256d + 位置 4d = 292d 向量 | 每像素：解码器特征 32ch + 掩码 1ch + SDT 1ch + 邻居 1ch + UAV几何 5ch = 40ch 图像块 |
| 网络 | LayerNorm → Linear(292→128) → GELU → Linear(128→1)——**逐点 MLP** | 6-8 个 conv block, 32-64ch——**稠密全卷积 CNN** |
| 输出 | 每点一个**有界残差 ±2.0**，加回原 logit | 带内每像素**全新二分类 logit**，替换原掩码 |
| 写入方式 | `logit[i] += Δ_i`（只能推高/推低） | `mask[band] = UBR[band]`（可完全重定义边界，含拓扑变化） |
| 分辨率 | stride-4 网格上的 64 点（上采样后每点覆盖 4×4 块） | **native 512²** 逐像素 |
| 信息源 | **仅卫星**（解码器内部的上采样特征 + mask token） | **卫星解码器特征 + 掩码状态 + 邻居 + UAV 几何（可选）** |
| 空间连贯性 | 64 个独立点，无邻域信息 | CNN 稠密预测，天然连贯 |
| 覆盖范围 | 只选解码器不确定的位置（自信但错误的位置看不到） | 处理边界带内**全部**像素，不管确不确定 |
| 损失 | margin-anchored 单侧 BCE | 带内 focal BCE + Dice，逐实例等权 |

### 4.3 UDPR 修不了但 UBR 能修的四类错误

1. **自信但错误**：解码器 logit 已饱和（很确定），但边界位置偏了 7-20px。UDPR 不选这些位置（\|logit\| 大）；UBR 处理整条边界带。
2. **拓扑变化**：挖凹槽、删假突起——需要边界形态改变而非微扰。UDPR 的残差 `logit += Δ` 做不到；UBR 的重分类天然支持。
3. **亚步长精度**：stride-4 网格上的 64 点覆盖不了 1-2px 的精确边界定位。UBR 在 native 512² 上操作。
4. **需要外部证据**：UDPR 只能用卫星特征；UBR 可引入 UAV 几何（高度不连续 = 真实建筑边界的物理信号）。

### 4.4 UBR 架构

```
                    单流模型                    双流模型
                 ┌─────────────┐            ┌─────────────┐
                 │ 卫星 backbone │            │ 卫星 backbone │
                 └──────┬──────┘            └──────┬──────┘
                        │                          │
                 ┌──────┴──────┐            ┌──────┴──────┐    ┌────────────┐
                 │ 检测+掩码解码│            │ 检测+掩码解码│    │ 几何头 🎯   │←─ UAV深度/相机
                 └──────┬──────┘            └──────┬──────┘    │ (实例发现)  │
                        │                          │           └─────┬──────┘
                        │                          │                 │ 新增候选
                        ▼                          ▼                 │
                 ┌─────────────────────────────────────────┐        │
                 │        UBR 统一边界精修器 🎯              │◄───────┘
                 │                                         │
                 │  输入:                                   │
                 │  ① 解码器特征 32ch (stride-4→上采样512²) │ ← 始终有
                 │  ② 当前掩码 logits 1ch (上采样后)        │ ← 始终有
                 │  ③ SDT 1ch                              │ ← 始终有
                 │  ④ 邻居掩码之和 1ch                      │ ← 始终有
                 │  ⑤ UAV几何 5ch                           │ ← 双流时有,单流时置零
                 │                                         │
                 │  网络: 稠密CNN ~200K, native 512²       │
                 │  输出: 边界带内逐像素二分类               │
                 │  损失: 带内focal BCE + Dice              │
                 └─────────────────────────────────────────┘
```

### 4.5 可选端口实现

```python
class UnifiedBoundaryRefiner(nn.Module):
    """Single architecture for both single-stream and dual-stream models.
    UAV geometry is an optional input port: zero-filled when absent."""

    def __init__(self, base_ch=35, uav_ch=5, hidden=48):
        # base_ch = 32(decoder features) + 1(logits) + 1(SDT) + 1(neighbor)
        # uav_ch  = 5 (height, dx, dy, view_count, support)
        self.input_proj = nn.Conv2d(base_ch + uav_ch, hidden, 3, padding=1)
        self.blocks = ...  # 6-8 conv blocks, GroupNorm + SiLU

    def forward(self, decoder_feat, mask_logits, sdt, neighbors, uav_geom=None):
        if uav_geom is None:
            uav_geom = torch.zeros(B, 5, 512, 512, device=decoder_feat.device)
        x = torch.cat([decoder_feat, mask_logits, sdt, neighbors, uav_geom], dim=1)
        return self.blocks(self.input_proj(x))  # band-limited binary logit
```

### 4.6 端口行为矩阵

| 场景 | ⑤号端口状态 | UBR 行为 |
|---|---|---|
| 单流训练/推理 | 全零填充 | CNN 学会用 ①–④ 做卫星-only 边界精修 |
| 双流训练（real 臂） | 真实几何图 | CNN 学会联合使用 ①–⑤ |
| 双流训练（constant 对照臂） | 常量均值填充 | 内容差实验的对照条件 |
| 双流推理（UAV 不可用） | 全零填充 | 优雅降级为单流行为 |

### 4.7 Modality dropout（训练策略）

从头联合训练时，对 ⑤ 号端口以概率 p 随机置零 UAV 通道。同时实现：
1. 模型学会在 UAV 不可用时优雅降级
2. constant 对照实验天然就是 dropout 的极端情况
3. 防止模型过度依赖 UAV 通道而丢失卫星能力

### 4.8 第 0 步（零训练风险泄压阀，先于任何 UBR 训练）

- **误差分类学**：对过不了 7px 的大实例，统计误差幅度直方图 + 拓扑类型占比（圆角/缺凹槽/偏移/阴影/粘连）+ 与密度相关
- **零训练 snap 探针**：frozen mask 边界沿法向 ±5px 搜索高度梯度极大值做贪心吸附。如果 +1pp B-AP_l → headroom 存在；如果 0 → 几何图本身是瓶颈，先修 z-buffer

### 4.9 探针阶段验收线（预注册）

- B-AP_l(7px) ≥ +1.0pp
- AP_l 回退 ≤ 0.1pp
- 输出层 content delta（UBR(真实几何) − UBR(常量几何)）≥ +0.5pp

## 五、从头联合训练最终形态

### 5.1 初始化

```
SAM2 backbone (Hiera+FPN)  ← 加载 sam2_hiera_base_plus.pt 预训练
RPN / RoI bbox / 几何头 / UBR ← 随机初始化
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
             UBR 🎯 (训练内, native res, +UAV几何)
                   │
                   ↓
             refined mask → loss_mask + loss_ubr
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
| 7 | loss_ubr | 边界精修（带内BCE+Dice） | λ_ubr (从0.1起调) |

**删除（vs 当前单流）：**
- ~~loss_decoder_tail (UDPR)~~ → 被 UBR 替代（同功能，更好信息，更高分辨率）
- ~~loss_shape_prior (coarse)~~ → SAM2 预训练初始化已提供掩码先验，从头训练时无需额外教粗形状

### 5.4 关键设计决策

| 决策 | 理由 |
|---|---|
| UBR 在训练内（非后处理） | 梯度可回流 decoder，教 decoder 产出更适合精修的 mask；UDPR 做不到这一点 |
| 删 UDPR，UBR 统一替代 | UBR 有 UAV 几何（双流的核心价值）；UDPR 只有卫星信息（已被证明不够） |
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
