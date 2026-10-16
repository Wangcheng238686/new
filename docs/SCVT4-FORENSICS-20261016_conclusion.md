# SCVT4 远端调研（2026-10-16）：旧项目"+UAV 更好"的证据结构与机制

调研对象：`wangcheng@10.171.103.175:/home/wangcheng/project/SCVT4`
（`portable_sam2_fusion` 与 `portable_sam2_fusion_large`，即本项目
`legacy_baseline` 的直系前身）。只读取证：configs/models 源码、训练日志、
`inference_results/*/eval_results.json`、修复/改进文档。

## 一、"+UAV 更好"的两层证据

### 表层叙事（未受控，夸大）

项目文档 `docs/dual_branch_improvement_proposal.md` 的基线表：
单分支 59.40/51.68（bbox/segm）vs 双分支 62.71/58.06。逐文件核对：
"单分支"= `metrics_satS_only_20260214_215309`（2026-02-14，SAM2
base_plus、30 epoch、**SAM2 mask head 尚处 bug 修复期**——见
`修复总结.md`：loss_mask 恒 ln2、维度错配、high_res_features 报错等
均在 2–3 月间修复）；"双分支"= 3 月初的 sat-drone v1（70 epoch、
修复后的模型栈）。**跨 3 周模型栈代差 + epoch 预算差**，增益大部分
是时间混淆。

### 受控层（同期同栈同数据同预算，真实但有限）

`portable_sam2_fusion_large`：两侧同为 576 train 场景、50 epoch、
sam2_large 栈、3 月下旬同期评估：

| 模型 | bbox/mAP | segm/mAP | bbox s/m/l |
| --- | --- | --- | --- |
| 单流 v12 uniform-lr（03-30 重评） | 0.6885 | 0.6996 | .532/.637/.728 |
| 融合 sat-drone-large v1（03-14） | 0.7120 | 0.6994 | — |
| 融合 3views（03-30） | **0.7150** | 0.7015 | .440/.651/.769 |
| 融合 4views（03-30） | 0.7115 | — | — |
| 融合 official-depth 变体（03-13） | 0.6840 | — | — |

**受控结论：bbox +2.4~+2.7 真实增益；segm 持平。尺度分解：大建筑
bbox +4.1、中 +1.4、小建筑 −9.2（segm 大 +0.8/小 −2.1 相消）。**
视角数 3>4（5views 未见更优）；深度源用 DepthAnything 优于 official
depth（−2.8）。

口径 caveat：旧项目未做 drone-off 评估（消融基础设施存在但无产物）；
单 seed；val 集口径未从工件完全确认（segm 两侧持平暗示同集）。

## 二、那边的 UAV 分支为什么能拿到 bbox 增益（机制）

`configs/rsprompter_sat_drone_sam2_v1.py` + `models/` + `uav/`：

1. **真实数据**：卫星图 + 每场景真实航拍目录（COLMAP `camera_params.json`
   配准，采样 3–15 视角，512²）。真实立面/阴影/纹理信息，高建筑在斜视
   中可见——增益恰好落在大建筑（+4.1）。
2. **UAV 编码器完全可训**：独立 SAM2(-large) backbone、`freeze=False`、
   lr_mult=1.0——编码器向任务适配，而非用冻结通用特征。
3. **融合形态避开我们踩过的全部坑**：
   - BEV 投影先过 **BN+ReLU**（量纲问题设计上不存在；对照本项目
     EXP-0014 诊断的 62× 范数失衡）；
   - **交叉注意力**（softmax 归一化、内容路由）而非加性残差；
   - **fusion_weight 初始 0.3**（训练从第 1 步就被迫使用通路——无
     零初始化门控的 modality-collapse 逃逸问题）；
   - **高度门**：DepthAnything 深度先验 → height_encoder → 建筑掩码
     加权注意力（把 UAV 信息预先对齐到"哪里是建筑"的任务结构）。
4. **辅助监督**：loss_spatial_align（0.03x 量级）+ 高度方差损失
   （建筑/地面双峰先验）——把 BEV 表征往任务几何上拉。
5. LSS 风格 cross-view attention 建 80×80 BEV + 场景嵌入（32d/场景）。

## 三、与本项项目证据链的互洽（关键收获）

旧项目受控增益的形态与我们 UE 上全部诊断自洽，且互相解释：

| 旧项目观察 | 本项目对应证据 |
| --- | --- |
| 增益全在大建筑（+4.1），小建筑 −9.2 | R0：62% 小实例两视角几何不可见；z=22 屋顶可查询率 38%>z=0 的 19.9%——只有高/大建筑有 UAV 可兑现信号 |
| segm 持平（增益只在 bbox） | R0：掩码质量通道（+11.94 oracle）与 UAV 无关，由卫星分辨率主导 |
| 编码器可训才有效（冻结版即早期 crude 版亦弱） | 0012（唯一解冻臂）最接近单流且唯一建立真实结构化依赖（WHY-NULL 门扫） |
| BN 归一化 + 0.3 初始权重 + 注意力路由 | EXP-0014：量纲修复后仍无效——我们的用法还缺"初始即使用"与"内容路由"；WHY-NULL：优化动力学关闭弱模态通路 |
| 高度门聚焦建筑区 | DEPTH 探针：高度标记 SAT 出错边界带（79/79 场正）——同一信息的任务对齐用法 |
| 迁移密封时明确排除 UAV 特征（MIGRATION_MANIFEST） | 非证伪，是密封简化决策；受控增益（bbox +2.7）当时未被记录 |

**综合回答"为什么那边加了 UAV 更好"**：(1) 表层 +7~12 点主要是跨代际
混淆；(2) 真实增益为 bbox +2.7，集中于大建筑，来自"真实高分辨率航拍
信息 × 完全可训编码器 × 归一化+注意力+初始即用的融合 × 高度先验任务
对齐"四个条件的合取——每一项都恰好是本项目 UE 实验里缺失或后被证明
必要的；(3) 该增益形态（bbox-only、大建筑-only）与我们 R0/DEPTH/
WHY-NULL 的机制诊断完全一致，不构成对本项目结论的反例。

## 四、对本项目的可迁移启示（记录，不立项）

若未来在 UE 上重启双流，旧项目指明的杠杆组合是：可训（或重 LoRA）
UAV 编码器 + BN/注意力形态融合 + 非零初始参与 + 高度先验门控/辅助
监督；且预期收益带 = 大建筑 bbox，须同时防护小建筑回退（旧项目 −9.2）。

## 工件

远端：`SCVT4/portable_sam2_fusion{,_large}/{configs,models,uav,logs,
inference_results}`、`docs/dual_branch_improvement_proposal.md`、
`修复总结.md`、`QUICK_START_ABLATIONS.sh`；本地副本 `/tmp/scvt4/`。
判读边界：远端工件只读未改动；无训练/评估运行；引用数字均为既有
eval_results.json 原值。

## 五、附录（同日追问）：COLMAP 相对位姿 + DepthAnything 相对深度的利用是否合理

逐路径核查（`uav/cross_view_attention.py`、`uav/bev_embedding.py`、
`uav/depth_aware_bev.py`、`data/satellite_drone_dataset.py`；`_large`
项目同源抽查确认）：

**主通路（BEV 特征）——形式上合理，因为它不消费度量几何。**
数据集对 COLMAP 内外参原样透传（仅 resize/flip 修正，无尺度归一、无
与卫星帧配准）。但 CrossViewAttention（simple-bev 血统）只把相机用于
位置编码：光线方向（I_inv）+ 相机中心/朝向（E_inv）→ 编码归一化后进
注意力；BEV 是纯学习画布（`h_meters=None` ⇒ `V_inv=I` 且可训 +
逐格偏移可训 + 场景级 4×4 对齐矩阵可训）。场景内 COLMAP 相对几何自洽
+ 对应关系由注意力学出 ⇒ 规范不变性使相对位姿"够用"。残留缺陷：全局
单一 V_inv 无法吸收各场景 COLMAP 尺度差异，位置编码带场景相关噪声
（退化、不致命）。

**高度图通路——名义合理，实际是伪几何。**
`_compute_height_map`：自训深度头（ReLU、仅方差/双峰损失约束，无度量
监督）× COLMAP 内参反投影 → 取世界系 Y 作"高度"。两处不成立：
(1) 深度尺度与 COLMAP 重建尺度互不相关，相乘无量纲意义；COLMAP 世界
系 Y 轴为规范任意方向，不保证对齐重力；(2) `_aggregate_height_to_bev`
把各视角**图像平面**高度图直接 bilinear resize 到 80×80 再跨视角
max——没有任何投影几何。故"height_map"实为各视角高点马赛克，非地理
高度场。下游消费端是可学习的 height_encoder/sigmoid 门，只需一致可学
的先验——方差损失把它推向粗糙的"建筑性"图，功能上退化但不崩。
佐证：官方度量深度变体反而更差（0.684 vs 0.712），与本链不消费度量
正确性的判断相容；`multiview_depth_consistency`（本可约束跨视角几何
一致）在配置中关闭。

**DepthAnything 特征用法合理**：冻结 DINOv2 特征（384→64d）注入注意力
key/val——纯外观先验，与相对性无冲突。

**对增益形态的再解读**：正因为该 BEV 是"注意力软化的无人机语义包"
而非几何产品，其增益表现为大建筑 bbox（+4.1）而非需要几何对齐的精
细掩码增益（segm 持平）。光谱两端对照：SCVT4 把几何全部换成可学软先
验（容忍了相对性、牺牲了几何正确性）；本项目 UE 链路是强几何弱学习
（真度量相机契约 + 逐位投影 + 零初始化门）。两端证据合并指向同一设
计原则：度量几何做锚 + 归一化/注意力/非零初始参与把信息任务形态化。
