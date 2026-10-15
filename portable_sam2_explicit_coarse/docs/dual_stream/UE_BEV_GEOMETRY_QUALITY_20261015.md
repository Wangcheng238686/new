# UAV BEV 几何与局部形状留存：固定六图 CPU 审计

2026-10-15，基于当前开发子集，工作树HEAD 2f06d40。未改生产源码/配置/模型参数，未运行GPU、拟合或AP。本路重点是原始深度→生产单表面几何，不复做相机roundtrip与全局±4px旧探针。

## 已证与未证

1. **未发现这六图的几何cache失配。** 固定hash选择1218、950、306、1216、948、505，读取30个EXR和对应相机；调用当前原始build_geometry重新生成，全部payload数组与现有cache逐值相同。耗时34.68秒。manifest SHA为495841b2174242c87661e6b40dba2ddcca13d64b13798116efc8c933c679ca59，全部depth/camera/source/cache/protocol SHA见summary.json。
2. **已选中高度桶内部的高度平均不是大位移来源。** 在有效高度、共同finite/positive-depth、原始点位于卫星图像内、且点属于生产所选最高合格bin的237784个深度点上，保留原xy、仅将z改为参考表面均值，卫星投影位移均值0.009646px、p90 0.02475px、p99 0.08263px，没有>1px点。这是内部几何近似量，不是外部标定真值。
3. **一个XY格包含多个高度层是实际存在的，不能把“有支持”当成唯一同表面观测。** 六图60975个观测格（每图的格相互区分）中15166格包含多个2m深度高度bin，5050格有多个各≥3点的合格bin。在所有有效高度、原始点位于卫星图像内的384632点（包括非选中bin）中，替换参考z位移p90 3.88788px、p99 15.61078px，9.7316%>4px。这不说明这些点应被投给屋顶：立面、地面和遮挡均可能构成合法场景上下文；这里没有UAV实例/roof标签。
4. **候选仍允许明显的高度上下文混合。** 34050个最终参与细像素投影的BEV格共855882个候选票，79.7297%满足|anchor_z-reference_z|≤2m，15.8981%的anchor xy在当前4m格内。这些是未加attention权重的候选分布，不是约20.27%的实际attention错误，更不是语义污染证明。上下文不必全部来自本格。
5. **4m网格和投影池化限制细节表达，尚未量化其最终损失。** 选中bin卫星内点替换xy为格中心时位移均值2.21070px、p99 3.82433px，但生产投影用整个4m四边形覆盖像素，不是只把点塞中心。因此这些数值仅描述格内空间尺度，不能当成真实投影偏移/边界误差，不能据此宣布增大网格必然改善。

**目前更具体的可疑环节是“同token/候选上下文到单格表征如何保留局部形状”，而不是这六图的缓存错误或已选bin内平均高度。** 单表面格无法显式区分格内子区域；这由结构可知，但没有证明它就是弱互补的主因。保持当前单表面策略，无证据要求退回多高度BEV。

## 原始计数与分母

| 层级 | 计数 |
|---|---:|
| stride4有限正深度点，5view重复点未去重 |491520|
| BEV内深度点 |481624|
| BEV外深度点 |9896|
| BEV内且其格height_valid的点 |431153|
| BEV内但格高度无效的点 |50471|
| 属于最高合格bin的点 |263776|
| 有效高度格且有候选的点 |431153|
| 有效高度格且其格产生最终投影的点 |382254|
| 唯一观测格（图像内唯一，跨图不去世界位置重） |60975|
| 有效高度格且有候选格 |38741|
| 产生最终投影的格 |34050|

不能把60975个格当60975个独立建筑，也不能把491520个点当独立样本；不同view可能观察同一物理区域。没有GT使用、没有统计显著性或工程性能门。

## 生产路径与来源行号

以工作树`/home/wangcheng2021/project/new-uav-dual/portable_sam2_explicit_coarse/`为根：

- `rsprompter/uav/geometry.py:15–33`：生产stride4中心采样，单位米，五view反投影；:41–43仅observed & height_valid & candidate格进入投影。
- `rsprompter/uav/surfaces.py:7–16`：side512m/grid128给4m格，reference xy位于格中心；:21–33最高满足≥3点的2m高度bin取均值，非多表面输出。
- `rsprompter/uav/candidates.py:35–46`：stride8 source token内median和90th高度各给一个anchor，但token ID相同、几何多anchor不增加视觉分辨率；:70–80 core偏好为xy本格且dz≤2m，8m邻域中其余高度只是有限软惩罚，每view8个唯一token。
- `rsprompter/uav/cvt.py:50–61`：key=token key+anchor position，value直接为token value；高度关系通过可学习relative bias，不是硬排除，所选桶无法保证attention只采该表面。候选相容比例不是实际attention。
- `rsprompter/uav/footprints.py:19–26`：将4m水平quad按参考z投影；:33–47细像素多边形覆盖，不是简单中心splat；:56–62近深度2m竞争与重叠归一。一个格内被保留部分像素仍共享同一BEV向量。
- `rsprompter/uav/projection.py:20–27,44–55`：先按卫星stride整数分桶，再mass*quality加权平均，无法凭插值产生BEV格内新的视觉证据。

代码行号用于人工复核，精确源码SHA由summary.json保留，防止后续文件演进混淆。

## 既有训练遥测线索（不属于本次冻结模型测试）

已核`docs/DEBUG_FIELDS.md:70–103`和`train/ue_telemetry.py:65–123`。日志`/tmp/ue_uav_experiments/20261014_ad_seed42/dual/train_EXP-0006_pid745647.log`的E72 `[UE-EPOCH]`原始训练forward统计：最后CVT层sampled_queries平均205.6923、height_compatible_mass .5083171、local_mass .1385732、height_bias_m 9.873304、xy_bias_m 1.867232。每项finite_forwards=624，跨rank有限值之和/有限forward数，为forward等权，包含DistributedSampler重复，不含validation，不是checkpoint E72 EMA验证统计。生产者`uav/diagnostics.py:15–31`固定最多16×16采样，仅过滤has-candidate；没有height_valid或最终可投影过滤，参考高度可能无效。bias为加权signed offset后abs/norm，存在抵消，不能当外部对应误差。mass近似FP32重算、最后层、head均值；不能由此推出一半传入SAM的信息高度错误，也不能把训练趋势当机制因果。

本次六图投影子集候选height相容79.73%并不与训练attention50.83%矛盾：数据/参数状态/采样子集/加权方式均不同，不能直接相减或断言学习恶化。

## 下一步最小、可区分的冻结检查（尚未执行）

若另两路排除8²/32维读出限制后仍需定位，优先利用**现有支持有效投影query子集**测实际attention在core/height-compatible候选的加权质量，并按原始局部边界证据比较，不新增性能训练。需独立预注册：当前EMA、固定样本、训练和验证选择隔离、只观测不干预。不应只将height不相容权重全部关掉就宣称去污染；立面context可有价值。

若要验证4m格内形状损失，需要把原始深度与已有UAV RGB特征保留到亚格局部读出作为**诊断分辨率对照**，与现BEV同预算训练读出对照；这会涉及新特征提取/容量控制，必须主代理另审预算，不能用本次中心位移结果直接改gridsize。无需回退多高度BEV。

外部RGB-depth配准、真实相机约定、屋顶/立面实例可见真值仍待可辨识对应点/额外标签核验。无此真值时，不把卫星GT贴给UAV raw点当独立标签来评估“屋顶保留率”。

## 工件

- `/tmp/ue_bev_geometry_quality/PREREGISTRATION.md`：执行前固定协议。
- `audit.py`：唯一六图原始几何审计，`run.log` exit0。
- `summary.json`：每图hash/全部cache逐数组exact/全部计数/位移；`raw_geometry_*.npz`：原点所属格、视角、bin选择、共同集合和位移，用于主代理复算。
- `summarize.py`、`aggregate.json`：点等权汇总，CPU约0.5秒，无拟合。
- 此前轻量cache统计`/tmp/ue_bev_geometry_quality_cache.py/.json`已经生成，但与error_strata一路重叠，**不作为本报告新证据**，由主代理决定是否丢弃/复用。
