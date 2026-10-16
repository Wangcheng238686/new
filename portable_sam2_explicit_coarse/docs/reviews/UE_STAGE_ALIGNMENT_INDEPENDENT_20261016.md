# UAV→BEV→卫星→RoI独立审查（2026-10-16）

审查者为新子代理；只读HEAD 5bc45b9。无代码/配置/训练修改、无GPU/模型前向/拟合。全文中的源码路径相对 `/home/wangcheng2021/project/new-uav-dual/portable_sam2_explicit_coarse`；SAM2路径另列。读取指定七份报告及实际生产源码、探针脚本、JSON和六图raw_geometry NPZ。额外做一次5.7秒CPU独立视角depth互投。未运行未闭环的full-scale探针。

## VERDICT

**没有确认导致净收益小的坐标bug；不能宣称对齐已获外部真值验证。当前更有解释力的是局部表示受限/聚合后空间细节不足与异质性，加上模型共同适应，而不是已证明的系统缩放、正负轴错误、最高桶内部高度平均错误或大规模attention选错。**

必须把三种东西分开：几何坐标的一致性、投影内容属于哪个可见表面、投影特征还有多少实例边界信息。已有检查对第一项较强，对后两项仍有限；信息在编码/聚合中变弱可以表现为最终读出收益弱，不能直接归为融合器拒绝消费。

## 已证实bug

本次未发现新的生产bug。下面的结构事实和名义feature-center差均不应直接标成bug。尤其不应据此直接给UAV或卫星补±1.5像素。

## 已核排除项及范围

1. **六图几何缓存不是旧合同失配。** 原报告全数组重建exact；本次读取summary核全六图exact=True，并独立从六份NPZ重算所选高度桶且卫星内共同点：n=237784，z替换均值位移mean=.009645817px，p99=.082629352px，max=.277010905px。排除该子集已选桶内部均值造成数像素偏移。不能排除选错表面/其余桶/外部标定。xy替换格中心mean=2.210699546px、p99=3.824325076px；生产为quad覆盖，故这不是实际误差。
2. **代码单位/矩阵/行列未见矛盾。** cameras.py:34–43只将两矩阵translation cm→m，旋转不缩放；:46–68局部化、反投影和正投影代数正确。surfaces.py:9–16为x列/y行；projection.py:22–24为卫星像素y行/x列。build_geometry不是把BEV resize到卫星整幅，footprints.py:23–47实际使用卫星透视相机。既有153图roundtrip只能排除代数矛盾。
3. **±4px残差平移不支持一个统一平移即可解决。** 153图四方向无一过预注册门，左右/上下不一致。它平移的是SAM融合有效残差，同时固定原coverage，非相机重新标定；不能排除缩放、旋转、局部视差、来源token中心、亚像素误差，或经共同训练补偿后的误差。
4. **候选上下文混合不等于实际attention错误。** geometry.py:41–43投影要求observed & height_valid & available；有候选不表示观测支持。有效投影query实际高度相容mass约.796，均匀同query约.804；joint实际反而高于uniform。旧训练约.508来自不同集合，不能当一半有效投影内容错误。六图tail关联无稳定额外退化关系，不能据此优先硬筛高度尾部。
5. **简单候选均值没有恢复强信号。** 实际stage_shape_summary.json：C−S64改善.000182947、C−CX .000598295；U−S64 −.000509908、U−UX .000100496，四主门全FAIL；非线性C优于U .000692855，95%区间[.000573324,.000829413]。排除“当前CVT明显劣于这个等权替代、只换平均即可恢复”的证据路线，不能排除CVT丢失原始信息，因为U也已按token/候选/投影聚合。

## 新增CPU原始深度互投：比roundtrip多了什么

脚本 `check.py`，结果 `crossview_depth.json`，均在本报告目录。固定既有六图1218/950/306/1216/948/505；30个实际相机depth metadata位置与camera_to_world的camera center逐值一致（max差0m），由fx反推FOV与SceneCapture记录max差9.2756e−7度。实际metadata为SCS_DEVICE_DEPTH、post_rgb_pie、warmup4、controlled_plane_distance_cm=0。后者不是控制平面验证通过的证明。

独立view depth互投：源图stride16，四边留8px；投影到其它view后nearest取depth；两端局部3×3深度range<1m，排掉明显深度边界。120个有向view对共同得到14559点：11111(76.317%)绝对差≤1m，11188(76.846%)≤2m，3369点预测在目标深度后方>2m、仅2点在前方>2m。大部分不一致符合遮挡的方向，额外支持不同视角相机/深度的米级一致性，而不仅是同一矩阵互逆。

局限：不能把后方点全部判定合法遮挡；没有真实可见性或表面标签。未建立RGB边缘或卫星像素独立对应；nearest/smooth筛选会低估边界问题；点和view不是独立样本；没有CI或性能门。该检查不排除半像素中心、RGB与depth时间/材质差异、卫星相机偏差；也不能把76%称为准确率。没有进一步扫参数/选择有利子集。

原始导出代码还提供独立来源检查：当前共享workspace的 `citysample_render_v10_4_dual_view_pair_dataset.py:852–888` 明确UE right,-up,forward→CV x,y,z；:785设置cx=width/2；:5194–5234写SceneCapture depth来源、FOV、pose与capture phase。`decode_ue_device_depth.py:18–26`实现near_clip/(1-packed)的camera-Z解码。它们说明生成意图，不能替代实际渲染验证。RGB来自MRQ、depth来自SceneCapture，不同管线本身仍应有外部图像核对。

## 风险：表示与对齐语义

### 1. 同token多几何anchor无法分离屋顶/立面视觉内容（解释力最高）

candidates.py:35–46在每个8×8原图块取median/p90高度anchor，两个anchor共享同token；:78–80又按token去重。cvt.py:50–61的value直接来自该token，几何只影响key/bias。即使anchor落在屋顶，value也可能含立面、背景和非局部attention上下文。两个几何anchor不是两份分开的视觉证据。dual_detector.py:85–92还把stride16 semantic上采样与stride8相加；源特征不是原始8×8局部颜色向量。

surfaces.py:21–33只选最高合格2m桶；每个4m格只输出一个z和一个BEV向量。六图多bin格15166/60975；多个合格bin5050/60975。多bin不等于多个建筑，但单格表达无法显式给格内子区域不同UAV值。真正有价值的边界在此之前或此处变弱，是合理首要嫌疑，尚未用原始source token/亚格对照定位。

### 2. footprint/z-buffer只过滤代理几何，不保证可见实例正确

footprints.py:23–26将一个水平4m quad按单z投影，:56–62只在“已支持的代理quad”间按camera-Z+2m竞争、归一；不是与完整卫星真实depth竞争。没有UAV观测的前景遮挡物不会自动进入这套z-buffer。没有屋顶/立面实例标签，因此coverage是几何覆盖而非正确roof证据率。少量真实观测也会外推为整个quad。高coverage不意味着高细节、高可见性或高语义纯度。

projection.py:25–54质量加权均值后，细卫星像素投进stride16网格；即使几何正确，多个格/表面会变成一个均值，coverage只是占据比例，quality不是语义置信。单纯把输出插值放大无法逆转这些混合。

### 3. feature-center约定有未闭环审计点，未证明偏移bug

SAM2源 `/home/wangcheng2021/project/new-uav-dual/sam2/sam2/modeling/backbones/utils.py:68–86` PatchEmbed k7/s4/p3；hieradet.py:108–112以2×maxpool降采样。按算子几何支持中心计算，s4名义中心0+4i、s8为2+8i、s16为6+16i。注意力与maxpool实际最大位置不固定，所以该名义中心不是物理对应真值。

- UAV源端：anchor为8i…8i+7块中的真实挑点，块均值中心3.5+8i，和名义s8中心差1.5个**UAV输入像素**；actual anchor本来就可位于块内任意像素，不能强制其等于中心。s16→s8 align_corners=False与上述名义中心可以一致，因此此插值本身没有找到错误。
- 卫星目标端：floor(pixel/stride)分桶的离散块均值中心是s8的3.5+8i/s16的7.5+16i，和Hiera名义中心也差1.5个**卫星输入像素**。这是另一端的约定比较，不是同一个误差；不能相加或抵消。卫星neck还含卷积/上采样，SAM/FPN并非简单点格。

这里的1.5来自整数像素坐标定义；若用连续像素面积边界中心，又涉及0.5px约定。因此不应事后补±1.5px。必须用独立已知场景点/多视角边缘与坐标算子小测试区分惯例和真错位。

### 4. SAM全图输入与roi_local输出是学习转换，非显式几何裁剪

mask_head.py:378、394–409把整图embedding与high重复给每ROI，随后decoder输出直接作为roi_local map；mask_head_targets.py:25–48用proposal crop GT；mask_head.py:677–689按bbox paste。由源码确认这里没有将image embedding crop为相应ROI再给decoder。因此“卫星位置上注入的UAV残差”到“ROI规范化mask坐标”的转换需网络学会，不应把两坐标当同一网格。该路径是单/双流共享设计，未证它导致二者差异，更不是可立即修复的训练/推理不一致bug。GTdense可消费只证明这一条件路径存在，不证明已有UAV局部信息完整或自然映到ROI边界。

## 未知与最小可区分检查（建议，未执行）

1. **外部几何先于再训**：固定少量分散场景对应点/屋顶边缘，从UAV RGB/depth到卫星RGB独立核验；显式区分屋顶/立面/地面与遮挡，报告signed dx/dy随半径、z、view的变化。常量偏移、径向缩放、轴翻转、视差的残差形状不同。不要把同矩阵roundtrip当外部点真值，也不要将卫星GT反投影标签当独立UAV标签。
2. **投影尺度**：现有fine/coarse探针仅6图smoke：实际manifest complete=False、6记录，roi_aligned=True、scale=.25、16²输出；未有281图或拟合收益结论。资源/授权允许后按原冻结协议完成，可区分stride16 pooling是否限制可读性，但不回答source stride8或4m单格先前损失。
3. **聚合前后信息保留**：若仍需查，需同目标、同支持、同通道/读出预算，比较原source token可见表面局部证据、CVT、投影后表示，保留亚格信息；原uniform对照已经聚合，不能替代此检查。不推荐继续无界加头宽度/seed。
4. **ROI坐标因素**：先对已有GTdense因子探针按ROI大小/绝对位置/目标中心偏差描述，不直接改full_image训练或新训P2。能区分与目标尺度/坐标映射相关的困难，但仍是观察性。

## 解释力排序

1. 局部边界信息在source token混合→单4m格→stride16均值中受限，与现阶段微弱可读增量最一致；定位具体损失阶段仍未知。
2. UAV路径与卫星/SAM共同适应，可强依赖而小净收益；correct−zero/shift约7pp不能换算新增信息贡献。
3. 覆盖/截断/尺寸异质性及背景代价，有真实分层证据，但不足以用简单coverage gate解决。
4. 真外部局部配准/可见表面偏差与feature中心约定，尚有未知；目前缺乏系统错误的阳性证据。
5. 高度尾部attention、桶内z均值、缓存失配、统一±4px偏移作为主要解释，现证据优先级较低；不等于所有情形均排除。
