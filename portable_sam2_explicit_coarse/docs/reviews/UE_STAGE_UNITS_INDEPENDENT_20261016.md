# 独立量纲与空间尺度审查（2026-10-16）

对象：`/home/wangcheng2021/project/new-uav-dual/portable_sam2_explicit_coarse`，实际HEAD `5bc45b91a1c6f772cf0490cb31ad05565cb5748d`。本审查只读源码、原始camera/EXR/packed PNG；没有改源码、cache、训练或模型，没有GPU调用。自行抽样与计算脚本 `/tmp/ue_units_independent_probe.py`，原始数值 `/tmp/ue_units_independent_probe.json`。manifest SHA `495841b2174242c87661e6b40dba2ddcca13d64b13798116efc8c933c679ca59`，从全部manifest点按 SHA256(`units-independent-20261016:<id>`)最小值选3图：431、28、653，全部5view。并非独立测试性能结果。

## VERDICT

**未确认100倍cm/m错误、把range当camera-Z、512输入未更新内参、stride16误当stride8等实现错误。实际更有证据的是空间尺度并不支持“UAV提供比卫星更细的纹理”这一预期，加上单格/单token/stride16投影造成空间压缩。** 这能解释为什么多视角丰富而局部形状互补仍弱，但目前仍不是已证实的AP主因。外部真实RGB-depth与跨视角配准精度仍未知，不能由E×T≈I或同源project/backproject roundtrip宣布准确。

## 实际尺度（新独立原始抽样）

|量|实际值/范围|含义|
|---|---|---|
|UAV EXR raw camera-Z|26437.14–71687.74 cm（15view全部有限正像素）|正确除100得到264.371–716.877m，不是26437m，也不是264m量级的欧氏range|
|UAV相机|fx=fy=787.9111328125；512²；cx=cy=256|原始图像直接读取，无resize|
|UAV外参中心相对manifest origin|水平轨道半径300m；高度334.709、365.617、400、438.633、482.535m|5view并不只改变yaw；高程也变化|
|卫星相机|fx=fy=2844.4444444444443；中心世界Z=2000m；512²|近似垂直向下；三图地面4m沿x/y均投影5.6888889px|
|卫星近地GSD|2000/2844.444=0.703125m/px|512px约360m地面幅宽；BEV side512m不是卫星幅宽512m|
|BEV格|512/128=4m|近地5.69卫星px，屋顶更接近相机时略大|
|卫星融合stride8/16|8/16px≈5.625/11.25m近地|128²BEV不是最终SAM32²空间分辨率|
|UAV stride8水平面局部跨距|u向各view p10–p90总体包络3.05–6.69m；v向3.45–9.89m|在每个stride8中心采样点，以真实深度恢复高度，然后交同高度水平面测u+8/v+8的XY跨距；不是建筑真值、不适用于垂直立面|
|UAV stride16同法|u向6.10–13.38m；v向6.85–19.56m|与stride8相加的semantic信息来自更大物理区域；不是凭上采样恢复细节|
|camera-Z→range比例|采样p50=1.03309，p90=1.06357，最大1.09754|若真把range当Z会有明显位置误差；本consumer没有这样做|

UAV原始单像素在水平面约0.38–1.24m的局部跨度包络，与卫星0.70m并非普遍更精细；进入模型可用于几何投票的是stride8 token，已到约3–10m。多视角仍可能提供遮挡、立面、不同外观信息，不能因尺度表就断言UAV没有价值。

## 已确认实现行为与排除项

1. **cm↔m一致，抽样未见100倍错误（排除项，P0排查完成）。** `data/ue_geometry_cache.py:19–27`读EXR并除100；`uav/cameras.py:38–39`外参平移各除100；`cameras.py:45–56`origin直接按米平移。生产converter `Content/Python/decode_ue_device_depth_exr.cpp:43–52`输出cm，`finalize_uav_depth_backfill.py:81–84`传near clip。新独立将15个原始packed PNG按10cm/(1-packed)解码，与对应EXR全部像素比较，最大绝对差0.00389796cm、median ratio约0.999999998；是float32量级。当前世界Z抽样约−3.49至166.57m、相机高度335–483m，量级合理。该检查只验证编码/解码链和样本数值，不能验证深度像素与RGB对象真实对应。

2. **camera-Z consumer与生产者相符（排除项，P0排查完成）。** converter公式是逆投影camera-forward Z；生产脚本 `citysample_render_v10_4_dual_view_pair_dataset.py:5194–5199,5218–5225`为SCS_DEVICE_DEPTH/perspective/camera_space_z；`cameras.py:52–56`使用[(u−cx)Z/fx,(v−cy)Z/fy,Z]，没有错误再除ray norm。packed与EXR一致进一步排除此抽样的重复range修正，但两者同源，不能单靠它证明UE真实近裁面等标定。最小外部检查是保留独立已知世界坐标/距离的离轴平面，不仅中心平面。

3. **输入resize/intrinsic同步错误未发现（排除项）。** `data/ue_rgbd_dataset.py:80–90,104–108,119–124`强制512²，scale_factor=(1,1)，不做resize/flip；`backbone/sam2_vision.py:494–510`原图直接encoder。`dual_detector.py:84–95`真实取pyramid[1] stride8并把pyramid[2] stride16双线性上采样相加，`cvt.py:157–159`严格检查(5,256,64,64)；`candidates.py:10–11,43–45`也用64² token编码。没有256/512或32/64网格错配证据。

4. **confidence列的单位一致（排除项）。** `surfaces.py:38–40`分别view_count/5、upper_count/all_count、sigma_z/height_bin；`candidates.py:84`nearest_xy/radius；`footprints.py:63–72`输入均为无量纲数，质量在[0,1]。`cvt.py:55–61,79–82,170–173`分别以radius8m、height10m、side/2=256m、height100m归一化。未见拿米直接当0–1概率的单位错误。但这里是手工质量分数，不是校准概率，不能把0.8称80%几何正确概率。

5. **投影足迹质量/面积归一化一致（排除项）。** `footprints.py:56–62`细像素竞争之后每像素mass总和1；`projection.py:25–26,52–54`按mass×quality归一化视觉feature，quality均值=加权质量/原mass，coverage=raw_mass/(sx×sy)。这样不会因stride16包含256pixel就让feature强度扩大256倍；feature幅值与覆盖、质量分开携带。不是漏除面积或feature单位100倍错误。

## 风险（不是已证实bug）

### P1：多级空间压缩、物理尺度与“局部边界”需求不匹配

`surfaces.py:8–16,21–33`4m格仅一参考高度，`candidates.py:35–46`一个stride8 token最多两个几何anchor却共享同一视觉token，`cvt.py:52–53`value不随anchor改变；`footprints.py:20–26`整个4m水平quad共享一个BEV向量；`projection.py:22–27,44–55`最终按stride8/16合并。4m格已经约5.69卫星pixel，SAM stride16再合并到约11.25m近地单元。亚格屋顶边缘、两个高度或不同实例不能由同格单向量显式区分。UAV stride8 token本身约3–10m，是此次独立核查对“原始源是否足够细”最直接的量化。

机制：几何投影的细像素支撑精度与视觉信息的细分辨率不是同一回事。细polygon和正确相机可以精确说明哪个5.7px足迹有支持，却不能创造足迹内部5.7个独立视觉值；mask需要细边界，source+BEV+projection的聚合可能使互补只表现为上下文。

最小判别：完成已经预注册但尚未正式完成的同CVT stride16↔stride4投影读出对照，先检验投影池化，不能重新扫网格/高度/seed；若无变化，仍不能排除原stride8源/单格压缩。已有stage形状试验候选均值与CVT都弱，不能定位损失发生于CVT，因为前者也先聚合、共用stride16和32维压缩。

### P1：coverage只做valid阈值，微量支持可以获得完整归一化残差幅度

`projection.py:52–54`局部feature按支持mass归一化，`fusion.py:21–33`coverage进入learned gate，而实际残差乘`coverage>0`，不乘coverage本身。例如stride16只1个像素支持、quality=.8时，coverage=1/256=.00390625，confidence=.8；若gate同为1，与256个全支持像素的显式乘子同为alpha×.8。该设计是可读源码确定的，**没有证据说明当前模型在低coverage位置实际产生错误放大**，不能直接称bug。

机制：训练可学会利用coverage，但不保证gate已经学会可靠性；小足迹边缘覆盖与整格高置信支持可能获得相近残差强度，加上GroupNorm与learned value不具有概率意义，支持率95%不等于95%卫星pixel有UAV精确信息。

最小判别：冻结现有checkpoint，按ROI/target-cell coverage分桶看gate、applied/satellite norm、错误变化，优先复用现有缓存。不要立即改乘coverage，因为这会同时改变训练过的幅值。

### P2：pixel center与token锚点分区不能靠roundtrip确认真实采样对齐

原生产相机 `citysample_render...py:780–785`设cx=cy=256，consumer以整数u/v反投影（`geometry.py:23–26`和`candidates.py:43–44`）。UE raster sample center可能对应n+.5的屏幕坐标，是否与导出K的像素索引约定严格一致没有新的外部真值。即使有半像素不一致，投影与反投影同时使用同K必然roundtrip自洽。当前范围半UAVpixel对应camera横向约.17–.45m，通常远小于4m格，但可能影响高度桶边缘或卫星subpixel配准；不能据此定性为主要收益缺失原因。

另外vendor `../sam2/sam2/modeling/backbones/utils.py:70–86`首层conv kernel7/stride4/pad3，`hieradet.py:108–111`2×2 q-pooling；stride8 nominal receptive center起点约2px，token几何分区0..7中心3.5px。token不是严格对应独占8×8块的独立局部观测，还有attention/FPN上下文。这个约1.5pixel的名义差不是已证实“错了1.5px”：anchors被有意放到同块不同可见表面，视觉token也有重叠和长程感受野。stride16语义经align_corners=False上采样在此nominal中心关系下基本一致，没有足够证据提出把align_corners改True。

最小判别：用渲染独立已知3D小目标/角点测RGB、packed depth边缘与相机投影的残差方向及随像素位置的变化，同时报告0、.5两种像素中心假设；不能拿同源GT投影回自己作为真值。优先核实际目标，不重做纯roundtrip。

### P2：RGB与depth分开render，pose同一不保证像素物体同一

生产脚本 `citysample_render...py:5142–5146,5194–5208,5226`depth SceneCapture使用相同actor pose和配置FOV，capture phase可以post_rgb_pie。RGB和depth不同render pass、抗锯齿/LOD/streaming时刻可能不同；没有本次证据说明存在系统误差，也没有本次独立特征对应真值排除。最小检查是固定少量可辨别屋顶边缘与立面角点跨RGB-depth重投影，保留人工/独立渲染真值，报告残差分布和遮挡掩码；不从语义颜色弱差直接推导几何不准。

## 激活量纲、ROI与未消费边界

`fusion.py:10–13`UAV残差先Conv+GroupNorm，`cvt.py:145–146`source也归一化。alpha不是“UAV占总信息比例”，norm后的residual与原satellite有不同尺度；实际作用应看applied/satellite norm和干预。既有E80 SAM residual_ratio=.9457、gate_supported=.9893、alpha=.2221（报告`UE_AD_RESULTS_20261015.md:99–112`，raw训练forward统计，并非E72 EMA验证）说明不能简单用alpha=.22就断言小幅注入，或用初始.05推断未消费。若父级未核DEBUG_FIELDS，此处仅给生产公式与报告口径，不作重新日志因果分析。

RoI输入从`anchor_roi_head.py:130–137`使用像素框送标准multilevel extractor；`anchor_detector.py:93–123`以真实image/stride16挑SAM embedding，未发现框坐标米↔pixel错用。高分辨率high_res_features仍由单流super.extract_feat返回，`dual_detector.py:67,148`只直接融合FPN(s8/s16)及SAM embedding，不直接融合high-res支路或x[0] stride4。这可能改变UAV信息抵达小ROI的机会，具体ROI level分布和gradient消费由专门消费链审查判定，不能仅凭“传入SAM”断言所有ROI都用了UAV。

## 优先级建议

P0单位错配无需盲改：本轮未见证实bug。P1先利用已冻结证据区分stride16池化与source/BEV尺度压缩，并用小范围既有特征分桶核coverage实际幅度；P2外部配准仅以独立对应点/已知3D目标判断。不要把4m→更细网格、提高alpha、修改depth语义或全局xy平移作为已证明正确的修复。
