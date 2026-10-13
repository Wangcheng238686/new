# SURFBEV-D0-PlaneCal-R1：UAV DeviceZ 绝对深度标定

**状态：PASS。** 对生产同源 `SCS_DEVICE_DEPTH → RTF_RGBA8 → packed PNG` 路径，已验证
解码

```text
packed = R + G/255 + B/65025        # R,G,B 先归一化到 [0,1]
DeviceZ = 1 - packed
camera_Z_cm = 10 / DeviceZ
```

可恢复 camera-space forward-Z，单位为 UE cm。该结论只放行 SurfaceBEV 的**几何数据
准备**；它不声称任何检测或分割 AP 增益，也不授权融合训练。

## 预注册口径

- **临时编号：** `SURFBEV-D0-PlaneCal-R1`，只读标定；无模型、无训练、无标签修改。
- **生产一致性：** CitySample producer
  `citysample_render_v10_4_dual_view_pair_dataset.py` 的 `SceneCapture2D`，
  `SCS_DEVICE_DEPTH`、`RTF_RGBA8`、512²、FOV 36°、warmup=4、post-RGB-PIE capture。
  每个 run 在相机 optical axis 正前方生成不透明、覆盖完整 FOV 的薄立方体平面。
- **标定距离：** 10/30/60/100/150 m。10m 在 GPU2，30m GPU2，60m GPU0，100m GPU1，
  150m GPU3；每项均单场景、单 yaw。GPU 只改变执行资源，不参与数值比较。
- **主门：** 每个距离中心 80% 像素的 camera-Z 中位相对误差 `<=1%`，任一距离
  `<=2%`；每帧要求平面覆盖（深度有效且空间常量）。
- **伴随：** output status 的 capture 成功性、平面元数据距离、全图 min/mean/max 一致。

## 结果

| 真值距离 | GPU | 解码 camera-Z | 绝对误差 | 相对误差 | 覆盖 / 门 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 10m | 2 | 999.414cm | −0.586cm | −0.0586% | 全 512² 常量，PASS |
| 30m | 2 | 2999.417cm | −0.583cm | −0.0194% | 全 512² 常量，PASS |
| 60m | 0 | 5999.267cm | −0.733cm | −0.0122% | 全 512² 常量，PASS |
| 100m | 1 | 9999.623cm | −0.377cm | −0.00377% | 全 512² 常量，PASS |
| 150m | 3 | 14998.982cm | −1.018cm | −0.00678% | 全 512² 常量，PASS |

最大相对误差是 10m 的 `0.0586%`，显著低于每点 `1%` 与任一点 `2%` 的预注册门。
所有点均满足 capture 成功、深度文件存在、平面距离元数据一致；故 D0 的**绝对深度
解码门通过**。

10m 第一次尝试产生了有效深度帧并同样给出上述数值，但 RGB 的自然场景质量门因平面
造成的低纹理画面而拒绝该 sample。它不是深度 capture 失败。随后四个正式测点均关闭了
这个与深度标定无关的 RGB 外观质量门；该改变不会触及 SceneCapture depth 的 source、
render target、FOV、warmup 或 decode。10m 深度仍以 producer metadata 和全图常量值
交叉核验，计入标定表。

## 判读边界

**能证明：** 当前数据中的 packed DeviceZ 可按上式还原为相机前向 Z（cm）；结合已经
通过的 K/W2C/C2W 契约，后续可以将 UAV 像素回投到 world 坐标并做 z-buffer SurfaceBEV。

**不能证明：** 它不验证卫星标签与 UAV 表面内容存在任务相关性，不能替代 P1 的
correct/sham 信息门，也不能当作 UAV 提升 mask/bbox AP 的结论。平面只测 optical-axis
深度尺度，不取代后续全图重投影和遮挡过滤检查。

## 工件

- 10m: `/data1/wangcheng/dataset/SURFBEV_D0_PlaneCal_20261011/10m_retry1/`
- 30m: `/data1/wangcheng/dataset/SURFBEV_D0_PlaneCal_20261011/30m/`
- 60m: `/data1/wangcheng/dataset/SURFBEV_D0_PlaneCal_20261011/60m/`
- 100m: `/data1/wangcheng/dataset/SURFBEV_D0_PlaneCal_20261011/100m/`
- 150m: `/data1/wangcheng/dataset/SURFBEV_D0_PlaneCal_20261011/150m/`
- decode implementation: `/data1/wangcheng/ue_experiments/uav_realism_v1/Content/Python/decode_ue_device_depth.py`
- producer implementation: `/data1/wangcheng/ue_experiments/uav_realism_v1/Content/Python/citysample_render_v10_4_dual_view_pair_dataset.py`

## 独立审计

独立审计结论为 **VERDICT = PASS**。审计人直接逐像素重算五张原始 PNG，确认全部
512² 像素有限且为常量平面；其独立数值与上表一致，最大相对误差 `0.058616%`。
审计同时核对标定与生产 UAV metadata：均为 `SCS_DEVICE_DEPTH`、`RTF_RGBA8`、相同
packing/decode、post-RGB-PIE、warmup=4 和约 36° FOV。并行 GPU 仅为资源分配，不进入
相机或解码公式，未发现口径污染。

非阻断项：生产目录没有冻结 producer/decode 脚本的不可变 hash；后续 raw reader 与
SurfaceBEV manifest 必须将二者的 SHA256 同数据 manifest 一并写入。该标定不替代
跨视角 z-buffer 或 correct/sham 检验。
