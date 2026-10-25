# 多视角无人机影像 → 卫星视角"无损"聚合方法调研

**主题**：如何把同一场景的多个 oblique UAV 视角聚合到 satellite-view 表征中，且**不丢失信息**（面向 building instance detection / segmentation）
**日期**：2026-10-25
**检索方式**：WebSearch（Google/Bing 检索 + 出版方页面交叉确认）

---

## 0. 我们的设定与基线的"信息损失清单"

设定：5 个 oblique UAV 视角（yaw 0/72/144/216/288°，512×512 RGB + camera-space depth EXR，厘米级 metric world coords，已知 intrinsics/extrinsics），对应同一 512×512 satellite patch。下游为 CenterNet-style 小 conv head（提出 instance candidates，与 RPN 互补），并可能向 frozen SAM2 检测器注入 feature。

当前 baseline 聚合：把所有视角投到 satellite pixel grid，**z-buffer 只保留每个 pixel 最近的表面**，再压成 4 个统计通道（height map、dx/dy height gradient、view-count）。

明确丢弃的信息：
- **(a) 所有被遮挡表面**（multi-height 结构：塔楼、屋檐 overhang、双层屋顶）——z-buffer 只留第一层；
- **(b) 全部 RGB appearance**——只剩几何标量；
- **(c) per-view evidence**——5 个视角变成一个 count，谁看见了什么、置信度多少全部丢失。

注意一个经典陷阱：true-orthophoto 文献早已指出 **z-buffer 对"高瘦建筑"存在 false visibility 问题**（Oliveira et al., "Occlusion Detection by Height Gradient for True Orthophoto Generation", ISPRS Archives 2013；Gharibi & Yun, "True Orthophoto Generation from Aerial Frame Images", Remote Sensing 2018）——即我们的 baseline 不只是丢信息，还会把"遮挡"误判为"可见"。这是所有 candidate 方案必须修的 bug。

以下按与我们的契合度**从高到低**排列五个 family（1-5 节），第 6 节专门整理"更丰富的聚合**不**帮忙"的已发表证据，第 7 节为对比表，第 8 节为 top-3 推荐。

---

## 1. Family：Multi-height / Multi-level BEV（高度分桶 occupancy）

**核心机制**：把每条 camera sight-line 沿垂直/深度轴离散成 K 个 bin（LSS-style frustum lifting），把 feature splat 进 `512×512×K` 的 voxel/height-bin 网格后再沿高度方向 flatten（sum 或 learned pooling）——flatten 之前垂直结构是完整保留的。

关键工作：

| 工作 | venue+year | 一句话机制 |
|---|---|---|
| Lift, Splat, Shoot (LSS) | ECCV 2020 | 每 pixel 预测 categorical depth distribution（canonical 设定 D=41 bins, 4–44 m），把 context feature 外积后 splat 到 BEV，多视角在 BEV 网格内自然求和融合 |
| CaDDN | CVPR 2021 | 同样的 categorical depth-distribution lifting 进 voxel grid，供 3D 检测；给出 depth-bin 数量 D 的消融 |
| SurroundOcc | ICCV 2023 | 用 multi-scale **3D deformable attention** 直接把 2D feature 逐级 lift 到 3D voxel occupancy，不经过平坦 BEV 中间层，明确以"保留垂直结构（overhang、异形车顶）"为卖点 |
| Occ3D benchmark | arXiv 2023 | 定义"每个 voxel 的 occupancy state + semantic label"任务与大规模 benchmark，论证 BEV segmentation 会把垂直维度压掉 |
| GaussianBeV | arXiv 2024 (2407.14108) | 用连续 3D Gaussians 替代离散 bin 做 BEV lifting，垂直位置不量化 |
| FB-BEV | ICCV 2023 | 显式比较/混合 lifting-based（LSS）与 attention-based（BEVFormer）两条 view-transformation 路线并取长补短 |

**相对 z-buffer 4 通道保留了什么**：所有 K 个高度层的表面（overhang/塔楼第二表面不丢）、每层的 feature（可以携带 RGB semantic）、多视角在每层内的叠加权重（等效 per-view evidence 的 soft 计数）。
**我们规模下的开销**：`512×512×K×C`。K=8、C=64、fp16 → 约 268 MB activation（K=8、C=32 → 134 MB；C=256 fp32 会到 ~2 GB，不可取——**C 必须压到 32–64 或在 stride-2 网格上做**）。BEV pooling/splat 本身是 scatter-add，5 个视角 × 262 k pixel 的点是 trivial 计算。若只存 K 层 height/occupancy 标量不带 feature（K×512×512×4B，K=8 即 8.4 MB），几乎零成本。
**frozen-backbone 兼容性**：**好**。我们已有精确 depth EXR，因此**不需要** LSS 的深度分布预测——lifting 的几何部分是确定性 warp，可以 training-free 地把 frozen RGB encoder 的 feature splat 进 height-bin 网格；只需训练一个小 3D conv / 高度 pooling head。LSS/CaDDN 原文需要端到端训练是因为它们要从 RGB 猜深度；我们已经知道答案。
**契合度评分：★★★★★（最高）**——与我们的差别仅是"z-buffer 单层 + 4 标量通道"→"K 层 + per-bin feature"，工程上是基线的直接泛化。

---

## 2. Family：Multi-plane Representations（MPI / LDI / plane-sweep cost volume / occupancy）

**核心机制**：不按 z-buffer 取最近表面，而是沿参考视线（我们的情形：satellite pixel 的垂直线）维护**一叠有序平面/多层深度 pixel**，或按 plane-sweep 构建跨视角对齐的 cost volume。

关键工作：

| 工作 | venue+year | 一句话机制 |
|---|---|---|
| Layered Depth Images (LDI) | Shade et al., SIGGRAPH 1998 | 每条视线存多个深度 pixel（不止第一个表面），warped 到新视角时逐层 composite——正是"z-buffer 只留一层"的原始解法 |
| Single-View View Synthesis with MPI | Tucker & Snavely, ECCV 2020 | 从单图预测一叠 fronto-parallel RGBA 平面（MPI），novel view 由 per-layer warp + alpha compositing 渲染；文中明确 MPI 是 LDI 的平面化变体 |
| Tiled Multiplane Images | Khan et al., CVPR 2023 | 把 MPI 分 tile 化，用 RGBA 层叠建模复杂遮挡外观，使 3D photography 实用化 |
| MVSNet | Yao et al., ECCV 2018 | plane-sweep homography warp 后，用**variance metric** 把任意数量 source view 的 feature 压进一个 cost volume——"per-view evidence" 以特征方差形式显式保留 |
| Learning-based MVS survey | arXiv 2024 (2408.15235) | 系统梳理 feature→cost volume→regularization→depth regression 全家桶，含 cost-volume 内存优化（coarse-to-fine、KD-tree 稀疏化） |
| PAC-MVSNet | Sensors 2025 | 在 cost volume 构造中加入 perspective-aware 卷积与 metadata 增强，代表 2025 年 cost-volume 路线的延续 |

**相对 z-buffer 4 通道保留了什么**：LDI/MPI 保留每条视线的**全部表面层**（含 alpha/软可见性）；plane-sweep cost volume 保留**每个深度平面上各视角特征的一致性证据**（variance/photometric residual），即显式的 per-view disagreement 信号——这是发现"卫星图与无人机图不一致/遮挡边界"的天然特征。
**我们规模下的开销**：cost volume `D×512×512×C` 与 Family 1 同阶（D=8–16、C=32–64、fp16 → 134–537 MB）；MVS 系的 3D UNet regularization 是主要算力项（我们只有 5 个视角、单 patch，比 nuScenes 6 路小得多）。LDI 版本更省：只存 K 层 `(z, feature)`，K=2–4 层时 ~67–134 MB。
**frozen-backbone 兼容性**：**中-好**。LDI/multi-layer 网格的构建是纯几何（training-free）；cost volume + 3D UNet 需要 training，但 2D feature extractor 可 frozen（MVSNet 的 2D encoder 本来就浅）。
**契合度评分：★★★★☆**——LDI 是对我们"multi-height 结构丢失"问题最对症的表征；cost volume 的 variance 通道还免费提供 view-conflict 信号（对 modality misalignment 敏感）。

---

## 3. Family：Cross-view Feature Fusion with Learned Attention（跨视角注意力聚合）

**核心机制**：在目标视角（这里是 satellite grid）布 query，用 cross-attention（通常 deformable、geometry-restricted）从 5 个 UAV 视角的 feature map 里采样聚合——融合是学出来的，几何只用于限制采样位置。

关键工作：

| 工作 | venue+year | 一句话机制 |
|---|---|---|
| BEVFormer | ECCV 2022（期刊版 TPAMI 2025） | BEV query 把参考点投影到各相机，spatial cross-attention 用 deformable attention 只在相关位置采样，多视角特征被 query 显式聚合 |
| Cross-View Transformer (CVT, Brady et al.) | ICRA 2022 | ground-view 与 map-view 之间按极线几何做 cross-attention，实现实时 map-view semantic segmentation |
| Multi-View Attentive Contextualization (MAC) | CVPR 2024 | 在 BEVFormer/DFA3D 的 spatial cross-attention 内部对多视角上下文做 cluster contextualization，改进跨视角聚合质量 |
| TransGeo | CVPR 2022 | 纯 Transformer 做 ground↔aerial cross-view geo-localization，attention 自主选择信息区域 |
| Geometry-Guided Cross-View Transformer | ICCV 2023 | ground-to-satellite 定位中用几何引导的 cross-view transformer 提升 3-DoF 位姿精度 |
| TransFG | CVPR 2024 | transformer-based feature aggregation + gradient guidance 做 UAV↔satellite 跨视角 geo-localization |
| Cross-View UAV Geo-Localization with Precision-Focused matching | arXiv 2025 (2510.22582) | 2025 年 UAV→卫星库匹配的最新精度导向工作 |
| End-to-End Multiview Fusion for Building Mapping from Aerial Images | Chen et al., Information Fusion 2024 | **与我们最像的遥感侧工作**：多视角航空影像 end-to-end 特征融合 + piecewise affine projection 做稳健 image-to-object 映射，输出 building mapping |

**相对 z-buffer 4 通道保留了什么**：全部 RGB appearance（feature 语义）、soft occlusion（attention 权重隐式学到"从这个 grid cell 该信哪个视角"）、视角间外观差异（attention 分布本身）。但**不天然保留多层表面**——除非 query 网格本身是多高度的（与 Family 1 组合）。
**我们规模下的开销**：deformable cross-attention 极便宜：262 k query × 4 points × 5 views × C=64 ≈ 数 GFLOPs 量级；5 张 512²×C=64 的 feature map fp16 约 168 MB（stride-2 后 42 MB）。**没有 dense cost volume**，是三个"重量级"方案里最省的。
**frozen-backbone 兼容性**：**中**。2D backbone（含 SAM2 内的 image encoder）可 frozen，但 fusion module 必须训练，且**没有 task gradient 之外的正则时容易出现"attention 退化成平均"**（见第 6 节负面结果）。BEVFormer/MAC 原文均为端到端训练。
**契合度评分：★★★★☆**——机制上最贴合"向 frozen SAM2 注入 feature"的下游形态；风险是任务对齐不足时学不出有效的跨视角权重。

---

## 4. Family：Depth-aware / Geometry-gated Appearance Aggregation（几何门控的外观聚合）

**核心机制**：不把几何与外观二选一，而是**沿已知几何把 RGB/semantic feature 携带过去**（paint/splat），并用 per-view 置信度/方差做加权池化。

关键工作：

| 工作 | venue+year | 一句话机制 |
|---|---|---|
| PointPainting | Vora et al., CVPR 2020 | 把 LiDAR 点投影到图像分割输出上，给每个点 append semantic score——"几何载体 + 外观涂装"范式的鼻祖，检测器本体不动 |
| BEVFusion (MIT) | ICRA 2023 | 在统一 BEV 空间融合 camera 与 LiDAR feature，高效 BEV pooling 沿预测深度分布 splat camera feature；论文明确主张该表征"同时保留几何与语义信息" |
| Lift-Attend-Splat | CVPRW 2024 | splat 前用 attention 融合 camera+LiDAR，并研究 depth supervision 对 splat 质量的作用 |
| Dense Projection Fusion | Sensors/PMC 2024 (PMC11461888) | 用 sparse depth 辅助 LSS 生成图像 BEV feature 再与 LiDAR BEV 融合——与我们"depth EXR 引导 RGB splat"结构同构 |
| QMF: Provable Dynamic Fusion for Low-Quality Multimodal Data | ICML 2023 | 用 energy-based uncertainty 表征每个模态质量，推理时动态加权，给出优于 static fusion 的泛化界 |
| Reliability-Driven LiDAR-Camera Fusion | arXiv 2025 (2502.01856) | 按 modality 可靠性门控融合，处理一个模态退化时的性能崩塌 |

**相对 z-buffer 4 通道保留了什么**：RGB appearance 全保留（被 paint 到几何载体上）、per-view 置信度显式保留（visibility/quality 权重）、几何通道不受损（PointPainting 是 **append** 而非替换——它的关键设计教训）。
**我们规模下的开销**：最低的一类。5 视角 × 262 k pixel 的 scatter-gather + 加权求和，毫秒级；内存即 5 张 feature map（见 Family 3）。
**frozen-backbone 兼容性**：**最好**。PointPainting 范式明确支持"detector 原样、backbone 原样"，只加通道/加权；QMF 式 per-view 权重可用深度残差等 training-free 代理（深度一致性即可视性代理），也可小数据学出。
**契合度评分：★★★★☆**——是"最小改动、最低风险拿回 RGB 与 per-view evidence"的路径，可与 Family 1/2 的多层网格叠加。

---

## 5. Family：Building-footprint / Reconstruction-specific（建筑测绘专用管线）

**核心机制**：photogrammetry 社区对"多视角航拍 → 无遮挡 overhead 产品"有成体系的解法（true orthophoto + nDSM 双通道喂检测器），以及 off-nadir 专用的 roof→footprint offset 学习。

关键工作：

| 工作 | venue+year | 一句话机制 |
|---|---|---|
| True Orthophoto Generation from Aerial Frame Images | Gharibi & Yun, Remote Sensing 2018 | DSM + z-buffer 遮挡检测（统计每 DSM cell 投影到多少 image pixel），遮挡区从其他视角补偿合成 true ortho |
| Occlusion Detection by Height Gradient for True Orthophoto Generation | Oliveira et al., ISPRS Archives 2013 | 用 height gradient 检测遮挡，指出 z-buffer 对高瘦建筑的 **false visibility** 缺陷 |
| Deep Learning Based Building Footprint Extraction from VHR True Orthophotos and nDSM | Büyükdemircioğlu & Kada, ISPRS Annals 2022 | 给检测器的通道组合实证：true ortho RGB + nDSM 高度层显著优于普通 ortho（后者含 building lean、tree/shadow 遮挡） |
| 3D Instance Segmentation of MVS Buildings | arXiv 2021 (2112.09902) | 每个视角先做 **RGBH（RGB+heightmap 4 通道）** roof instance segmentation，再做多视角 3D 融合——"每视角带高度做实例分割、后端融合"的遥感式答案 |
| Extracting Polygonal Footprints in Off-nadir Images | arXiv 2024 (2408.08645) | roof segmentation + roof-to-footprint offset prediction，处理 off-nadir 位移 |
| SpaceNet 4 Off-Nadir Challenge | 2018 | 多 off-nadir 视角建筑 footprint 提取的 benchmark，系统展示了视角/遮挡对检测的影响 |
| A Deep Learning Method for Building Height Estimation | Cao et al., Remote Sensing of Environment 2021 | 多视角卫星影像端到端建筑高度估计——多视角是垂直信息的主要来源 |
| IM2ELEVATION | Liu et al., Remote Sensing 2020 | 单张 overhead 图 → elevation map 的端到端网络（说明 height 通道可从 RGB 学，但多视角输入更可靠） |
| MAVREC: Can Multi-view Improve Aerial Visual Perception? | Dutta et al., CVPR 2024 | 实证多视角聚合对 aerial detection 的增益（正面结果：多视角确有互补价值） |

**相对 z-buffer 4 通道保留了什么**：true-ortho 管线保留**无遮挡 RGB 外观**（遮挡区从其他视角补偿——即利用了我们 5 个 yaw 的互补性）并显式输出**遮挡 mask**（免费获得"哪里 z-buffer 不可信"通道）；RGBH 路线把 height 作为第 4 通道与 RGB 同网格喂网络。但它们大多仍是 **单层（最顶层表面）** 表征——对塔楼/overhang 的第二表面同样不保留（这是测绘产品定义决定的）。
**我们规模下的开销**：true ortho 合成 = 每视角一次 z-buffer 投影 + 遮挡补偿 warp，**training-free、毫秒级**；offset/height 学习头是小头。
**frozen-backbone 兼容性**：**最好**（几何部分零训练；学习头独立小模块）。
**契合度评分：★★★☆☆**——直接可用的工程经验（通道配方：true-ortho RGB + nDSM + 遮挡 mask），但表征天花板低（单层），适合作为 baseline 增强与消融对照，而非终态。

---

## 6. 已发表的负面结果 / 消融：什么时候"更丰富的聚合"**不**帮忙

这正是我们观察到 modality collapse / weak task alignment 的文献对照：

1. **Modality collapse（融合塌缩到单一模态）**：Agarwal et al., "What Makes for Robust Multi-Modal Object Detectors?", ECCV 2022 —— 系统展示 fusion detector 过度依赖单一模态（LiDAR），另一模态退化时性能崩塌；提出 multi-modal dropout 训练缓解。**对应我们的风险**：如果 task gradient 只奖励几何（height 对建筑检测已很强），RGB 通道会被忽略/塌缩。
2. **低质量模态下 static fusion 可证更差**：Zhang et al., "Provable Dynamic Fusion for Low-Quality Multimodal Data (QMF)", ICML 2023 —— static fusion 在一个模态质量低时劣于单模态，需要 uncertainty 驱动的动态加权。**教训**：与其无差别拼接 5 视角 RGB，不如 per-view 质量加权。
3. **质量感知融合何时是假象**："When Does Quality-Aware Multimodal Fusion Matter?", arXiv 2026 (2606.26473) —— 用 permutation 诊断发现很多方法的 reliability 分数在推理时其实没被用到。**教训**：加了 per-view 权重要验证其因果作用（我们的消融应包含 weight-permutation 检验）。
4. **表征丰富度不如监督/对齐质量**：Yang et al., "Camera-view supervision for bird's-eye-view semantic segmentation", 2024 (PMC11604745) —— 对 lifting 过程施加更好的 camera-view 监督带来 ~2.6% BEV IoU 提升，而堆 depth-bin 数量的收益远小于此。LSS (ECCV 2020) 与 CaDDN (CVPR 2021) 的 bin 数消融同结论：bin 数翻倍收益边际，监督与对齐才是大头。
5. **fusion 鲁棒性依然脆弱**：Mohan et al., "Progressive Multi-Modal Fusion for Robust 3D Object Detection", PMLR v270 (2025) —— 即便 2025 年的融合方案，在模态退化下仍显著掉点，需要显式 progressive 结构。
6. **正面参照**：MAVREC (CVPR 2024) 证明多视角对 aerial perception 确有互补增益——问题不在"要不要多视角"，而在**聚合表征与任务梯度是否对齐**。

**综合结论**：文献一致指向——(i) 更丰富的表征本身不自动涨点，**任务对齐的监督/门控**才是关键变量；(ii) 无门控的静态拼接在部分视角低质量时会更差；(iii) 必须保留一个"几何-only"通路的消融，防止外观通道塌缩成噪声。

---

## 7. 对比总表

| Family | 保留（vs z-buffer 4ch） | 我们规模的开销 (5 views, 512², C=64) | frozen-backbone 兼容 | 预期契合度 |
|---|---|---|---|---|
| 1. Multi-height BEV（height-bin grid） | 全部高度层表面 + per-bin feature + 层内多视角叠加 | K=8: ~268 MB fp16 activation；splat 本身毫秒级；C 需 ≤64 | 好（几何确定，depth 已知，无需 LSS 深度预测） | ★★★★★ |
| 2. MPI/LDI + plane-sweep cost volume | 全部表面层（LDI）+ 每深度平面跨视角一致性（variance） | cost volume 同 Family 1；+3D UNet 正则化算力；LDI K=2–4 更省 | 中-好（LDI training-free；cost volume 头需训练，2D encoder 可 frozen） | ★★★★☆ |
| 3. Cross-view deformable attention（BEVFormer/CVT 式） | 全部 RGB 语义 + soft occlusion + 学习到的视角权重（但默认单层） | 最省：数 GFLOPs；5 张 feature map ~168 MB fp16（stride-2 后 42 MB） | 中（fusion 模块必须训练；2D/SAM2 encoder 可 frozen） | ★★★★☆ |
| 4. Geometry-gated appearance（paint/splat + per-view 加权） | RGB appearance + per-view 置信度；几何通道不受损（append 式） | 最低：scatter-gather 毫秒级，内存同 feature map | 最好（PointPainting/QMF 范式明确支持 frozen detector） | ★★★★☆ |
| 5. True-ortho / RGBH / offset（测绘管线） | 无遮挡 RGB（true ortho）+ 遮挡 mask + nDSM；仍是单层 | training-free 合成，毫秒级 | 最好 | ★★★☆☆ |
| 基线（z-buffer 4ch） | — | ~4 MB | 是 | 对照 |

---

## 8. Top-3 推荐

### 推荐 1（首选）：Depth-gated multi-height splatting——"已知深度版 LSS/LDI"（Family 1+2+4 合体）

用现成的 depth EXR 把每个 UAV 视角的 RGB feature（frozen encoder，或一个 3–4 层小 conv）沿已知几何 splat 到 satellite grid 的 **K=4–8 个 height bin**，每 bin 存 `(feature, view-count)`，另加每 pixel 的跨视角**特征方差通道**（MVSNet variance metric 的散度版）。它同时修复基线的三处信息损失：(a) 多层表面（overhang/塔楼）；(b) RGB appearance（feature 随几何走）；(c) per-view evidence（count + variance）。关键优势是**工程上是现有 baseline 的直接泛化**（同一套投影代码，只把"z-buffer 取最近 + 4 统计"换成"K 层 scatter-add"），几何部分零训练、training-free 可先验证表征质量，CenterNet head 只需把输入通道从 4 换成 K×(C+2)+4；对 SAM2 注入则取高度维 max/attention-pooled 后的 512²×C 网格。文献支撑：LSS (ECCV 2020) 的 bin 离散化 + MVSNet (ECCV 2018) 的 variance 聚合 + BEVFusion (ICRA 2023) 的"几何与语义同空间保留"论证；C≤64、fp16 时 activation ~268 MB，完全可承受。风险与对策见第 6 节——务必保留几何-only 消融通路并对 RGB 通路加 dropout（ECCV 2022）。

### 推荐 2：Satellite-query geometry-restricted deformable cross-attention（Family 3）

在 satellite 512² grid 上布 query，每个 query 只沿其垂直线上的 K 个 3D 参考点投影到 5 个视角 feature map 做 deformable cross-attention（BEVFormer spatial cross-attention 的"目标视角=卫星、源视角=UAV"特例）。相比推荐 1，它把"多层聚合权重"交给任务梯度学习，天然产出可向 frozen SAM2 prompt/feature 接口注入的对齐表征，且是五个方案中算力/显存最省的（无 dense volume）。前提是检测任务本身要能给 fusion module 足够梯度——按第 6 节文献，任务对齐弱时该路线最容易退化成"attention 平均"。建议作为推荐 1 跑通后的第二阶段（用推荐 1 的输出作 query 初始化可缓解冷启动）。文献支撑：BEVFormer (ECCV 2022/TPAMI 2025)、CVT (ICRA 2022)、MAC (CVPR 2024)、以及同构的 Chen et al. (Information Fusion 2024) 多视角航拍建筑映射。

### 推荐 3：True-ortho 风格 layered composite 通道包（Family 5）——低风险立即可做

不动学习框架，先把 baseline 的 4 通道扩成约 8–10 通道的 training-free 包：true-ortho RGB 合成（遮挡区按 5 个 yaw 的互补性补偿，Gharibi & Yun 2018）、遮挡/可见性 mask（Oliveira 2013 的 height-gradient 法，同时修 z-buffer false-visibility bug）、第二表面 height map（LDI 第 2 层标量）、per-view color variance（view-conflict 信号）、nDSM。这直接对应测绘界实证有效的通道配方（true ortho + nDSM，ISPRS Annals 2022；RGBH，arXiv 2112.09902），改动仅是数据管线 + head 输入通道数，一周内可出消融结论——即使全部通道无增益，也为推荐 1/2 提供了干净的对照基线与 modality-collapse 证据（对应第 6 节）。

---

## 参考文献（按 family 汇总）

- LSS: Philion & Fidler, "Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs by Implicitly Unprojecting to 3D", ECCV 2020
- CaDDN: Reading et al., "Categorical Depth Distribution Network for Monocular 3D Object Detection", CVPR 2021
- SurroundOcc: Wei et al., "SurroundOcc: Multi-Camera 3D Occupancy Prediction for Autonomous Driving", ICCV 2023
- Occ3D: Tian et al., "Occ3D: A Large-Scale 3D Occupancy Prediction Benchmark for Autonomous Driving", arXiv 2023
- GaussianBeV: arXiv 2024 (2407.14108)
- FB-BEV: Li et al., "FB-BEV: Bev Representation from Forward-Backward View Transformations", ICCV 2023
- SDG-OCC: Duan et al., CVPR 2025
- LDI: Shade et al., "Layered Depth Images", SIGGRAPH 1998
- MPI: Tucker & Snavely, "Single-View View Synthesis with Multiplane Images", ECCV 2020；Khan et al., "Tiled Multiplane Images for Practical 3D Photography", CVPR 2023
- MVSNet: Yao et al., "MVSNet: Depth Inference for Unstructured Multi-view Stereo", ECCV 2018
- MVS survey: "Learning-based Multi-View Stereo: A Survey", arXiv 2024 (2408.15235)；PAC-MVSNet, Sensors 2025
- BEVFormer: Li et al., ECCV 2022（期刊版 IEEE TPAMI 2025）
- CVT: Brady et al., "Cross-View Transformers for Real-Time Map-View Semantic Segmentation", ICRA 2022
- MAC: Liu et al., "Multi-View Attentive Contextualization for Multi-View 3D Object Detection", CVPR 2024
- TransGeo: Zhu et al., "Transformer Is All You Need for Cross-view Image Geo-localization", CVPR 2022
- SAFA: Yang et al., "Spatial-Aware Feature Aggregation for Cross-View Geo-Localization", ICCV 2021
- Geometry-Guided CVT: Shi et al., ICCV 2023
- TransFG: CVPR 2024；Precision-Focused CVGL, arXiv 2025 (2510.22582)；Fan et al., Sensors 2024
- Chen et al., "End-to-End Multiview Fusion for Building Mapping from Aerial Images", Information Fusion 2024 (102498)
- PointPainting: Vora et al., CVPR 2020
- BEVFusion: Liu et al., "BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation", ICRA 2023
- Lift-Attend-Splat: Gunn et al., CVPRW 2024；Dense Projection Fusion, PMC11461888, 2024
- QMF: Zhang et al., "Provable Dynamic Fusion for Low-Quality Multimodal Data", ICML 2023
- Reliability-Driven LiDAR-Camera Fusion, arXiv 2025 (2502.01856)
- True ortho: Gharibi & Yun, Remote Sensing 2018；Oliveira et al., ISPRS Archives 2013；Braun, "Aspects on True-Orthophoto Production", phowo03
- True ortho + nDSM: Büyükdemircioğlu & Kada, ISPRS Annals 2022
- MVS Buildings: "3D Instance Segmentation of MVS Buildings", arXiv 2021 (2112.09902)
- Off-nadir: SpaceNet 4, 2018；"Extracting Polygonal Footprints in Off-nadir Images", arXiv 2024 (2408.08645)
- Height: Cao et al., Remote Sensing of Environment 2021；IM2ELEVATION, Remote Sensing 2020
- MAVREC: Dutta et al., "Can Multi-view Improve Aerial Visual Perception?", CVPR 2024
- 负面结果: Agarwal et al., "What Makes for Robust Multi-Modal Object Detectors?", ECCV 2022；QMF, ICML 2023；"When Does Quality-Aware Multimodal Fusion Matter?", arXiv 2026 (2606.26473)；Mohan et al., PMLR v270 2025；Yang et al., camera-view supervision for BEV segmentation, 2024 (PMC11604745)
