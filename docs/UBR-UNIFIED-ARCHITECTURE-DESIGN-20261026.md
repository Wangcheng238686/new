# UBR 统一架构设计：UDPR→UBR 换装与单/双流统一（2026-10-26）

本文档取代 `DUAL-STREAM-ARCHITECTURE-DESIGN-20261024.md`（其 §一/§二/§三 证据章继续有效，§四 BRR 与 §五 损失精简按本文档修订）。核心决议：**UDPR 在单流与双流中全线出局，由统一边界精修器 UBR（Unified Boundary Refiner）替代；单/双流成为同一架构的两种端口状态，dual−single 差异收敛为 {端口⑤ UAV 几何, loss_geometry} 两个可枚举项。** 全部裁决基于三方证据：本机接口证据、VHR-10 会议消融、远端 compact_joint 从头训练实证。

---

## 一、证据基座

### 1.1 本机接口证据（唯一 CI 级正面方向，跨机复现）

几何实例候选接口（UAV 深度+相机 → z-buffer 5ch 几何图 512² → 小 CNN 头 ~203K → 离散框候选 → 冻结卫星掩码路径解码）：

| 比较 | Mask AP | bbox AP | Boundary2% | Boundary1% |
|---|---|---|---|---|
| 净增益 vs 强SAT前缀 | **+0.4672** [+0.297,+0.616] | +0.359 | +0.522 | +0.471 |
| 内容差 vs 常量 | **+0.2468** [+0.108,+0.350] | +0.244 | +0.258 | +0.344 |

热启动联合 +0.1355 [+0.007,+0.353]（Boundary1% +0.56）。增益全部来自新增实例的正确边界，旧实例掩码逐位不变——**已量化最大弱点（大建筑边界塌落 42.3%、B-AP_l 0.532）尚无人用 UAV 解决，这是 UBR 的目标空地。**

### 1.2 VHR-10 会议消融（prompt 模态与 UDPR 的承重实测）

| 行 | Bbox AP / AP50 / AP75 | Mask AP / AP50 / AP75 |
|---|---|---|
| P | 71.81 / 93.60 / 81.89 | 63.15 / 92.56 / 67.73 |
| B（仅框） | 72.14 / 94.34 / 82.24 | 65.25 / 94.12 / 69.02 |
| P+B | 71.74 / 93.52 / 81.68 | 66.94 / 93.93 / 72.73 |
| P+B+M | 73.76 / 93.80 / 84.69 | 66.35 / 93.66 / 71.61 |
| P+B+M+UDR | 73.43 / 93.96 / 84.16 | **68.28 / 94.53 / 74.61** |

三条结论：模态提示（P+M）相对仅框 **+1.10 Mask AP**，是会议论文的研究项目本身，**栈保留**；UDPR 贡献 **+1.93 Mask AP / +2.99 AP75**，是实测承重结构——换装 UBR 不是清理边角料，是替换一个有署名贡献的模块，**再封版门必须盯 AP75**；RSPrompter MLP sparse-token 路线否决（legacy 已刻意移除、范式切换使会议指标与双流证据全部失效、无优势证据）。

### 1.3 远端 compact_joint 从头训练实证（lthpc，codex/uav-bev-dual）

- **共同基础初始化（仅通用 SAM2，任务权重零热启动）从头训练可达 E73 水平**：卫星臂 ep91–98 val segm/mAP 稳定 0.861–0.864；rgbd/constant_rgb ep59–66 达 0.856–0.873；reference_plane 0.838–0.854（偏低属预期方向）。四臂无塌陷、无发散。
- **打分带结构免疫**：合并集前缀走 r(s)=0.5+0.5/(1−log s)（严格单调、>0.5），候选封顶 0.25·s（≤0.25）——两个分数带永不交错，合并排序对 RoI 分数漂移在构造上不敏感。100ep 从头训练带着未改打分收敛，是 r(s) 保留的直接实证。
- **候选不进训练期掩码损失**：`CompactJoint.loss` = 卫星 `loss` + `loss_uav_instances`（直接中心框损失，权重 1.0）；候选只在推理期进入合并集。随机几何头的早期垃圾只脏 val 尾部：匹配 epoch 对比 ep1 dual 比卫星低 1.2pp，ep10 反超 +1.1pp，自愈。
- **插槽余量**：前缀均 142.8 实例/图、候选均接收 107.2/图，459 图全部顶满 250 上限——候选带插槽充足，"前缀膨胀饿死候选"残余风险当前不成立。
- **注意**：远端臂保留 UDPR+shape_prior+原损失集，与统一架构不同构——它们是**参考不是对照**；但其训练器（初始化合同、打分、损失结构、100ep 预算收敛性）是统一架构最终形态的**宿主实证**。

---

## 二、UBR 统一边界精修器

### 2.1 与 UDPR 的功能对比

| 维度 | UDPR（出局） | UBR |
|---|---|---|
| 选择方式 | \|logit\| 最小 64 点 | \|SDT\| ≤ k 空间带（k∈{6,8,12,16} 消融） |
| 输入 | 逐点 292d 向量（卫星 only） | 逐像素 40ch 图像块（卫星+可选UAV） |
| 网络 | 逐点 MLP | 6–8 conv block（32–64ch，GroupNorm+SiLU），~200K |
| 输出/写入 | 有界残差 ±2.0 加回 logit | 带内全新二分类 logit，`mask[band]=UBR[band]`（可重定义边界） |
| 分辨率 | stride-4 网格 64 点 | native 512² |
| 损失 | margin-anchored 单侧 BCE | 带内 focal BCE + Dice，逐实例等权 |

UDPR 修不了而 UBR 能修的四类错误：自信但错误（饱和 logit 不被选中）、拓扑变化（挖凹槽/删突起）、亚步长精度、需要外部证据（UAV 高度不连续 = 建筑边界物理信号）。带内"保持原判"是平凡可学策略，规避均值回归。

### 2.2 输入端口与行为矩阵

| # | 输入 | 通道 | 状态 |
|---|---|---|---|
| ① | 解码器特征（stride-4 上采样 512²） | 32 | 始终有 |
| ② | 当前掩码 logits（上采样后） | 1 | 始终有 |
| ③ | SDT | 1 | 始终有 |
| ④ | 邻居实例掩码之和（推理用精修前并行掩码） | 1 | 始终有 |
| ⑤ | UAV 几何（height/dx/dy/viewcount/support） | 5 | 见下 |

| 场景 | ⑤ 状态 | 语义 |
|---|---|---|
| 单流训练/推理 | 全零 | 卫星-only 精修 |
| 双流 real 臂 | 真实几何图 | 联合使用 ①–⑤ |
| 双流 constant 臂 | **support=1 处填 train-only 逐通道均值、洞处保持零** | 内容门对照（support 通道分布与 real 臂一致） |
| 双流推理 UAV 不可用 | 全零 | 优雅降级为单流行为 |

带内输出做**最大连通域保留**（防拓扑分裂）；UBR 作用于**全部解码实例**（含几何候选新增实例，同一解码路径）。**v1 不做 modality dropout**（零/常量/真实三态已覆盖两极端，dropout 留作期刊鲁棒性章节增强项）。

### 2.3 关键力学事实

输入①②到 native 512² 是插值上采样；带内唯一 native 分辨率的新证据只有⑤（③④为当前掩码派生量）。因此 **UBR(off) 相对 UDPR 的优势纯来自力学（带内重分类+空间连贯），UBR(on) 才增加分辨率层级的证据**——内容差理应集中在大实例边界，与 §1.1 弱点量化对齐。

---

## 三、统一架构定义

### 3.1 骨架（单/双流逐位同构）

```
卫星 RGB → SAM2 backbone → RPN → RoI → 框+点+dense 三模态 prompt（points_box_dense，
  冻结 PromptEncoder，dense 通道 detach_input=True，shape point mining no_grad——
  梯度防火墙原样保留）→ SAM2 mask decoder（无 UDPR）
                                        ↓
       单流: UBR(⑤=0) ←────────────────┤
       双流: UBR(⑤=几何图) ←───────────┤   几何头 🎯 ← UAV深度/相机（loss_geometry）
                                        ↓
                              refined mask → loss_mask + loss_ubr
```

### 3.2 UDPR 出局的工程含义

`decoder_tail_refiner` 在新封版配置中**不实例化（删代码，非运行时开关）**；`loss_decoder_tail` 从两个模型的损失字典永久消失；再封版时铸造新架构指纹。冻结参照（EXP-0025 ep71、E73）内部含 UDPR，仅作证据源与对照锚点，非架构成分。

### 3.3 损失账本

| 模型 | 损失集 | 项数 |
|---|---|---|
| 统一单流 UBR(off) | rpn_cls + rpn_bbox + roi_cls + roi_bbox + loss_mask + loss_shape_prior + loss_ubr | 7 |
| 统一双流 UBR(on) | 同左 + loss_geometry | 8 |

dual−single = {端口⑤, loss_geometry}。λ_geo 锁定远端实证的 **1.0**（center_loss 权重，本机线的 0.1 不再引入）；λ_ubr 从 0.1 起调（阶梯既有项，非新增）。**loss_shape_prior 保留**：coarse 头的唯一监督（dense 通道 sg-detach、mining no_grad 使下游梯度不存在，删损失留通道 = 给冻结 PE 喂随机 logits）；其删除与 points/dense 栈绑定，归入永不默认执行的 box-only 臂（20261024 文档 §5.3 的删除条目就此撤销）。

---

## 四、四风险裁决（最小改动，全部落定）

| 风险 | 裁决 | 依据 | 哨兵（骑现有产物） |
|---|---|---|---|
| R1 双分数语义 | **不动**：r(s)/0.25 带打分原样 | 带状分离对分数漂移结构免疫 + 远端 0.862 实证；改动反而破坏冻结期配对可比性与跨机复现契约 | eval 加 ~2 行带不变量断言；离线监控 records.json 的 `original_path` 均长 >200 与 `geometry_accepted` →0 |
| R2 早期垃圾 prompt | **接受，零改动** | 候选不进训练期损失，只脏推理合并集尾部，ep10 内自愈（ep1 −1.2pp / ep10 +1.1pp） | 已落盘 epoch*.json：同 epoch dual ≥ satellite −0.5pp（ep≥10 起） |
| R3 可微 prompt | **v1 绝不打开** | 离散框接口是唯一 CI 阳性资产与优化防火墙；远端可微特征路径 14 个负内容差反证 | 程序规则：任何开梯度提案必须自带预注册内容门 |
| R4 prompt 栈凝练 | **栈逐位保留，box-only 默认永不** | VHR-10 实测栈值 +1.10/+3.03；会议贡献线延续 | trainer source_hashes 契约断言（栈被改动即失败） |

**明确不做清单**：可学分数校准、prompt source embedding、候选 warmup 门、软框/STE-NMS、删 shape_prior/points/dense、改 r(s)/0.25/250/maxDets/FP32 任何一位。净新增训练超参 0、净新增臂 0、净新增代码 ≤2 行 eval 断言。"打开 UAV 分支学习"的唯一含义 = loss_geometry 联合训练（mask 梯度不经候选框回流几何头）。

---

## 五、实验阶梯（预注册门，逐级放行）

| 级 | 内容 | 验收门 | 成本 |
|---|---|---|---|
| 门0 | 误差分类学 + 零训练 snap 探针（frozen 边界沿法向 ±5px 吸附高度梯度极大值） | ΔB-AP_l ≥ +1pp → headroom 存在；≈0 → 先修 z-buffer，§二暂停 | ≈0 |
| 级1 | 固化 EXP-0025 ep71 + E73（归档 + 冻结全量 test：mAP + Boundary AP + AP75） | 产物落盘、SHA 绑定 | ≈0 |
| 级2 | 加法探针：冻结旧封版(含UDPR)真实 train 输出上独立训练 UBR，real/constant 双条件 | B-AP_l(7px) ≥ +1.0pp 且 AP_l 回退 ≤0.1pp 且输出层内容差 ≥ +0.5pp | 接口头同量级 |
| 级3 | 再封版：UBR(off) 单流从头 100ep 同协议（宿主=远端验证配方） | 配对 bootstrap CI 含 0（**segm/bbox mAP + AP75 双盯**：UDPR 的 +2.99 AP75 不许丢） | 1×100ep |
| 级4 | 最终形态：从头联合三臂（统一单流 / UBR-on 常量 / UBR-on 真实），宿主=远端 from-scratch 训练器谱系 | 内容门（real−constant CI>0）+ 净增益（real−单流 CI>0）+ B-AP_l 改善 | 3×100ep |

级3 通过 → 统一架构成为新封版；不通过 → 回退"UDPR 保留 + UBR 双流专用"分裂构型（设计反对的形态，作为兜底记录在案）。级4 收线后 QG（远端 candidate_detail）结果汇合：QG 内容差为正 → 最终形态为"候选+QG+UBR"三层；为负 → "候选+UBR"两层。

---

## 六、期刊叙事（单一动机纪律）

全文只回答一个问题：**UAV 信息对卫星侧实例分割有没有帮助。**骨架四段：动机（卫星弱点量化 + UAV 互补证据）→ 仪器（统一架构，⑤端口为 UAV 信息唯一入口）→ 实验（内容门矩阵 + 配对 bootstrap）→ 归因（增益分解）。**常量臂是该问题的正面证明机制**（证明增益来自信息而非参数/容量），居主表 C 位；负结果舰队只作内部记录与 rebuttal 弹药，不进正文叙事，设计依据一句话带过。会议→期刊连续性：会议为显式提示模态研究（栈被继承，VHR-10 指标不复跑），期刊在同一骨架上以 UBR 升级 UDR 并开 UAV 端口；凝练主张由 UBR 统一承担（−UDPR +UBR，单/双流同构），不动模态栈。执行纪律（实施前过骨架检查门）见工作区根 `AGENTS.md`。

---

## 七、与远端线的合并界面

- 远端 compact_joint 四臂与 candidate_detail 十臂**继续按其原合同跑完，不中途改动**；其结果按"参考"吸收。
- 统一架构级3/级4 若沿用远端训练器，须注意其 S' 特征可微交互属于其谱系选择，与本文档"候选框路径离散"是两件事，不得作为打开 R3 的许可。
- 本文档应同步至远端工作区，UBR 作为合并载荷的完整规格。

---

## 八、工件索引

- 前版文档（§一/二/三 证据章有效）：`docs/DUAL-STREAM-ARCHITECTURE-DESIGN-20261024.md`
- 接口复现：`docs/GEO-DIRECTION-REPRO-20261024.md`；RGB 否证：`docs/GEO-RGB-INCREMENT-20261024_*.md`
- VHR-10 消融配置：`configs/ablations/vhr10_{point_only,box_only,point_box,point_box_dense}.py`
- prompt 栈与防火墙：`rsprompter/cspm/{mask_head,dense_prompt_utils,shape_point_miner}.py`（生产 `detach_input: True`）
- 远端实证：lthpc `scripts/probes/ue_{compact_joint_run,instance_interaction}_20261024.py`、`rsprompter/uav/compact_joint.py`、`/data/wangcheng/checkpoint/ue_compact_joint_20261024/*/epoch*.json`
- 打分带实现：`scripts/probes/ue_joint_eval_20261024.py`（本机）、远端 `inference_outputs`（逐位同构）
