# SURFBEV-20261011：用 UAV 深度证明卫星检测/分割增益的受控方案

## 目标与适用数据

**目标：**在 `ue_raw_hero_uav_pairset_20261011` 的 1,200 个严格配对场景上，证明在
同一卫星检测/实例分割基座下，正确使用 UAV 信息能使卫星侧的**完整独立测试集** mask AP
和检测 AP 获得可重复收益。

本方案不使用 Unreal `WorldPosition`。UAV 的三维表面点只由可部署的输入导出：每像素
camera-space depth、内参 K、外参 `world_to_camera_cv`。此前被否定的是 RGB+固定 z=0
平面 DBEV；这里检验的是**深度解算后的可见表面 BEV**，不是旧路线的换名重跑。

## 路由与临时编号

这是“新基座 + 模块叠加”路线：新原始标签/数据契约需要先训练卫星基座，然后才允许冻结
探针、热启动筛查、全量训练。临时编号依次为 `SURFBEV-D0`、`SURFBEV-B0`、
`SURFBEV-P1`、`SURFBEV-H1`；未进入全量前不发 EXP-ID。

## D0：深度与标签读取契约（阻断门）

`depth_devicez_packed.0000.png` 的 producer 元数据明确要求先把 DeviceZ 线性化为 UE cm
camera-Z，并要求 controlled-plane validation。`depth_camera_z.0000.exr` 虽存在，当前
没有独立生产语义承诺，故在验证前也不得直接称作米制/线性深度。

1. 以单独 UE 受控平面渲染，在已知距离 10m、30m、60m、100m、150m 各采一帧；固定
   与生产相同的 FOV、分辨率和 DeviceZ packing。解码 `R+G/255+B/65025` 后，以 UE
   projection 的 `ConvertFromDeviceZ` 线性化。
2. 接受条件：五个距离的中心 80% 像素中位相对误差均 `<=1%`，任一距离 `<=2%`；相机
   Z 与由 K/E 回投后再投影的像素误差 median `<=0.5px`。否则 D0_FAIL，禁止构建
   Surface-BEV，转为补采规范化 `camera_z` 或 world-depth。
3. 同时实现 raw reader：以 `satellite_instance_id_clean.png` 的局部 ID 产生实例 mask/
   bbox，验证 ID 与 `instances.json` 一一对应；五路 UAV 的 RGB、深度、K、W2C 均读入。
   读写均只按 `accepted_pairs.jsonl`。每一输入文件、解码参数、标定结果和 reader hash
   写入 manifest。

## B0：卫星单流锚点

在 D0_PASS 后，以现有卫星实例分割基座训练卫星单流 B0。数据切分按 `gx,gy` 的空间块，
绝不随机按 scene 切分：使用 3×3 原始 cell 的 supercell 作为不可拆分 group，先固定一组
连续空间 holdout 为 test，并从其余 group 固定 val；切分清单、每 split 图像数、实例数和
building-area 分布在训练前归档。UAV 文件在 B0 训练及测试均不可读取。

- 主锚点：test COCO `segm AP@[.50:.95]`。
- 伴随：`bbox AP@[.50:.95]`、segm AP75、bbox AP75、按实例面积三分位的 segm AP。
- 选择口径：每 seed 固定 last epoch；两 seed 的均值为 B0 参考，禁止按测试集挑 epoch。

## P1：冻结 Surface-BEV 信息门

对冻结 B0，建立目标卫星特征格上的可见表面特征，训练集拟合、val 选定一次、test 一次
封存：

1. 对 UAV 每个低分辨率特征像素，以标定后的 camera-Z、K 和 `camera_to_world_cv` 回投：
   `X_w = T_cw [z(u-cx)/fx, z(v-cy)/fy, z, 1]^T`。
2. 将点投到卫星 360m 覆盖范围的 XY pillar；高度轴为 12 个 bin，边界只由**训练集**
   表面高度的 0.5/99.5 百分位固定。每 bin 聚合 UAV RGB encoder feature、height mean/
   variance、view count 与 z-buffer consistency。
3. z-buffer：点重投回每一路 UAV，只有观测深度差小于 D0 标定噪声阈值的点可贡献；这
   排除被遮挡立面和错误表面。每 cell 输出有效 view 数与残差，缺失值显式为零并带 mask。
4. 三个冻结读出臂：`SAT`、`SAT+SurfaceBEV_correct`、`SAT+SurfaceBEV_sham`。sham 将
   同一场景所有 UAV 的 W2C 统一绕场景中心转 90°，并完整重做回投/z-buffer/pillar；模型
   容量、样本、标签完全相同。

P1 主指标为 test 每场 roof-boundary-F1 的 correct−SAT paired bootstrap 95% CI（10,000
draws，预注册 seed）；伴随为 mask error-cell AUROC、实例 mask-IoU 预测相关、
correct−sham CI、surface coverage 与 depth residual。`P1_PASS` 要求所有只读不变量
成立，correct−SAT 与 correct−sham 的 CI 下界都 `>0`。它只证明可消费信息，不可声称 AP
收益。

## H1：热启动筛查（唯一允许的首个融合模型）

若 P1_PASS，B0 detector、satellite backbone、RPN 和 ROI encoder 冻结；只训练 Surface-
BEV encoder、12-bin height attention 与 P2/P3 residual fusion。P1 的训练集读出权重初始化
投影层；残差门初始化为非零 `0.1`，以避免零门令内部层无梯度。没有 P1_PASS 不得启动 H1。

对照与唯一变量：

| 臂 | UAV 输入 | 作用 |
| --- | --- | --- |
| B0 | 无 | 卫星基线 |
| H1-RGB | 五路 RGB、固定平面 | 区分“额外 RGB” |
| H1-Surface | RGB + 经 D0 标定的 Surface-BEV | 主实验 |
| H1-Surface-sham | 训练后测试时 90°几何错配 | 因果/几何诊断，不用于挑权重 |

H1 主门：相对 B0，完整 test 的 `segm AP` paired scene bootstrap 95% CI 下界 `>0`，且
`bbox AP` 不低于 B0 `0.002`；同时 correct-sham 的 segm AP 差 `>0`，P2/P3 融合范数非零、
所有 legacy output/state 哈希不变量通过。伴随报 segm/bbox AP50、AP75、面积三分位 AP、
boundary-F1、误差分位和 surface coverage。失败即关闭该接线，不以某个小目标分桶的偶然
增益替代主门。

## 全量证明标准

只有 H1_PASS 才允许两 seed 全量。正式结论需满足：两 seed 的 last checkpoint 都相对
同 seed B0 有正的 test segm AP 效应；合并 scene-paired bootstrap CI 下界 `>0`；bbox AP
不降超过 `0.002`；test-time UAV 几何 sham 显著破坏收益。完整报告必须给 B0、RGB、
Surface、sham 四臂以及所有四权重的标准推理数字。

## 不能做的事

- 不把 `WorldPosition` 当模型输入，也不以 UE oracle 结果声明现实可部署收益。
- D0 未验证前，不把 packed DeviceZ 或未签名的 EXR 当线性深度。
- 不随机拆分相邻空间 scene，不用 test 选高度 bin、阈值、epoch 或融合位置。
- 不把冻结 readout、coverage 或 boundary telemetry 误称为实例 AP 结论。
