# 五视角 UAV→BEV 的 CVT 路线调研

日期：2026-10-09。聚焦“无需把 UAV 特征硬反投影并 splat 到 BEV”的替代路线。数据集版本问题暂不进入本文件的设计取舍。目标仍是 UAV RGB-D 增益卫星图上的单类别建筑实例分割，BEV 是中间特征。

## 文献路线

| 路线 | 构造 BEV 的机制 | 与本任务的关系 |
| --- | --- | --- |
| [Cross-view Transformers, CVPR 2022](https://arxiv.org/abs/2205.02833)（[代码](https://github.com/bradyz/cross_view_transformers)） | BEV query 用相机 K/E 感知的射线位置编码，cross-attention 从多视角图像特征中读取证据；视角映射隐式学习 | 最贴近旧项目。能够融合五视角，且不要求逐像素深度；但建筑高大、视角重叠稀疏时，纯方向编码可能把同一射线上的不同深度表面混淆 |
| [BEVFormer, ECCV 2022](https://arxiv.org/abs/2203.17270)（[代码](https://github.com/fundamentalvision/bevformer)） | BEV query 的 3D 参考点投到多相机图像，再通过稀疏空间注意力读取附近特征 | 证明“查询图像构造 BEV”可服务检测与地图分割；原方法的固定多高度参考点和时序模块不是本任务必需，可只借鉴局部采样 |
| [BEVSegFormer, WACV 2023](https://openaccess.thecvf.com/content/WACV2023/papers/Peng_BEVSegFormer_Birds_Eye_View_Semantic_Segmentation_From_Arbitrary_Camera_Rigs_WACV_2023_paper.pdf) | BEV query 学习各视图的图像参考点，用 deformable cross-attention 聚合；无需 K/E | 说明可以进一步弱化显式几何；但我们的逐视角 K/E 已知，完全舍弃几何会增加标定学习负担 |
| [LaRa, CoRL 2023](https://proceedings.mlr.press/v205/bartoccioni23a.html)（[代码](https://github.com/valeoai/LaRa)） | 图像→紧凑 latent→BEV 的两段 cross-attention | 避免每个 BEV 格与所有图像 token 做密集注意力；但 latent 的世界坐标/可见性解释较弱，作为效率对照更合适 |
| [DepthFormer, 2022](https://arxiv.org/abs/2211.04188) | 把深度放入位置编码与跨模态注意力以改善 RGB-D 语义分割 | 支持“深度条件化注意力”，但其输入/输出同视角，不能直接证明跨视角对齐 |

这些工作验证了查询式 BEV 表征的可行性。它们主要评估车辆/地图语义任务，尚不能推断卫星建筑 mask AP 的收益。

## 旧项目代码的真实行为

- [`cross_view_attention.py`](/home/wangcheng2021/project/compare/SCVT4/portable_sam2_fusion/uav/cross_view_attention.py) 用 UAV 射线编码、世界 XY BEV query、五视图 image keys/values 做 cross-attention。这是 CVT 式隐式视角映射，并非深度点 splat。
- [`depth_aware_bev.py`](/home/wangcheng2021/project/compare/SCVT4/portable_sam2_fusion/uav/depth_aware_bev.py) 先从 RGB **估计**深度，再将深度卷积嵌入加到 image key；后面另算 height map。它没有把新数据中的实测 camera-Z 用于 BEV query 与图像 token 的空间候选/可见性判定。代码中的深度图缩放还使用普通双线性插值，需要重新处理深度边界和无效像素。
- [`bev_embedding.py`](/home/wangcheng2021/project/compare/SCVT4/portable_sam2_fusion/uav/bev_embedding.py) 让世界网格变换 `V_inv` 和每格 offset 可学习。若该 BEV 要与逐点卫星 K/E 对齐，必须固定世界 XY 定义，或同步使用实际变形后的坐标；不能把学到的网格仍当固定米制网格。
- 原版 CVT 论文采用较低分辨率 BEV query 再上采样；五张 512² UAV 图若直接对 128² BEV 做全局注意力，计算量大。仅按每视图 32² image tokens 估算，128²×(5×32²)≈8400 万个 query-key 配对/头/层；局部或稀疏采样应优先考虑。

## 三个可比较的 UAV→BEV 方案

**A. 原味 CVT。**固定米制世界 XY query + UAV K/E 射线编码 + RGB 特征跨视角注意力；深度作为 key/value 附加通道或位置编码。它是旧路线的可复现对照。它能让模型通过学习处理立面/屋顶语义，但缺少强可见性约束。

**B. 深度约束的 CVT（建议主候选）。**保持“BEV query 从 UAV 图像读特征”的主结构。对每个有效图像 token，用真实 camera-Z 和 K/E 计算观测世界 3D 位置，仅用来构造候选邻域、相对位置编码和注意力偏置；特征仍由 attention 按内容与几何共同决定。可用

`score(q,p) = content(q,p) + b(x_p-x_q, y_p-y_q, z_p, view_angle, depth_confidence)`。

每个 query 只访问有有效深度且落在局部世界邻域的 top-k token；邻域半径要容纳立面至屋顶的水平位移和标定误差，不要把点硬指定到唯一 BEV 格。保留有效视角数、最近距离、跨视图一致性作为 confidence。无局部证据处可保持 latent 上下文，但融合到卫星时的可信度应为零或很低。

**C. 硬几何 splat。**真实深度点特征直接聚合到世界 XY 栅格，作为低成本对照。它的坐标更可解释，但垂直立面、屋顶及相邻建筑可能在同格混杂。A/B/C 使用相同 UAV 图像编码器、相同卫星 FPN 接口和训练预算比较，才能判断“学习式 BEV”是否比确定性构图带来实例收益。

## BEV→卫星与实例头

查询式 BEV 仍须有稳定的世界 XY 定义。先用卫星 K/E 把完整 FPN 目标格映射到 BEV 的参考 XY，再局部读取 BEV 特征，融合的权重由 coverage/depth consistency 控制。目标卫星图不裁剪，UAV 视野与卫星视野无需相同。对高楼屋顶，地面参考 XY 可能有视差；可让卫星 FPN query 在几何邻域内做小范围 deformable attention，或用可靠观测高度修正参考位置。

先用同一卫星实例头（RPN、bbox、mask）测试 A/B/C。主指标 mask AP/AP50/AP75、bbox AP、proposal recall 和 GT-box mask 质量；另记录有效 UAV 覆盖、注意力落点与真实深度位置偏差、错配 UAV 响应。若 B 优于 A，说明真实深度有助于注意力选点；若 B 优于 C，说明软选择与多视角语义提供硬 splat 之外的价值。

## 建议

保留 CVT 作为正式候选，而非只把它当历史实现。第一轮做 A（原味 CVT）与 B（真实深度约束 CVT），并用 C（深度 splat）作几何基线。B 是当前数据最值得投入的主线：几何负责缩小可读区域、排除无效观测，注意力负责选择哪张 UAV、哪片屋顶或立面特征对卫星实例有用。后续再判断是否需要 BEVFormer 式高度参考或 LaRa 式 latent 压缩。
