# SURFBEV-P1-V2-H0：冻结 SurfaceBEV 采集与接线门

## 定位

这是 `EXP-0006` 完成后的第一个**冻结、只读**探针，不训练任何网络，不读取
v2 test 的 RGB、UAV、标签或相机文件。它不是 P1 信息读出本身，更不是 AP 实验；
它只回答：在无泄露 v2 的 train/val 场景中，DeviceZ + K/C2W/W2C 是否能稳定地
生成与 B0 实际 P3（128²）同格的多高度 SurfaceBEV，且被动采集不会改写 B0 输出。

## 不可变输入

- B0：仅允许 `EXP-0006` seed 44 的
  `/data1/wangcheng/checkpoint/surfbev_b0/surfacebev_b0_v2_raw1024_noaug_ddp4_s44_EXP-0006/last_model_epoch100.pth`
  （SHA256 `fc3ead0e9490ffd9249cdbc4ce2ee9556bed5252efc3e475dfcc0f2aace65145`）。
  checkpoint 内的零基 epoch 必须为 99（即第 100 epoch），并同时核验 run tag、seed、
  epoch 总数和 v2 split；不使用 best 或中间权重。
- 数据：`accepted_pairs.jsonl` SHA256
  `f15cd8f5b8d10a69d721546d6cfa1f2cce58af2861d57b1357c5e9ff1e1fe62a`；v2 split
  SHA256 `17f49430b15037c8da25c63a1bf7fe4d510cf5ff153263d2feba62c4393441d5`。
- 场景：694 train 场只用于冻结高度边界；100 val 场用于完整采集/不变量检查。
  151 test 场在 H0 和后续开发阶段均封存。
- UAV：独立 SAM2 Base+ encoder，`eval()` / `no_grad()`；DeviceZ 按 D0-R1 的
  `camera_Z_cm = 10 / (1 - packed)` 解码。不得输入 WorldPosition、卫星标签、
  B0 预测或任何 test 数据。H0 专用 reader 不得打开
  `satellite_instance_id_clean.png`，也不得构造 `gt_*` 字段。

## 实施与预注册门

每一个 UAV 特征像素以 camera-Z、K、C2W 回投到世界坐标，再以 satellite W2C
投到 B0 实际 P3=128² cell。每个 cell 建 12 个仅由 train 表面高度
0.5/99.5 分位确定的 bin；每 bin 聚合冻结 descriptor、world-Z 均值/方差、
有效 view 数、卫星视角 z-buffer 残差和缺失标记。`sham` 将同一表面点绕卫星
相机 XY 中心旋转 90°后，完整重做投影与 z-buffer。

H0_PASS 必须同时满足：

1. 每个 val scene 的 B0 正常前向在采集前后逐字段哈希一致；采集函数不接触
   detector state、标签或预测。
2. correct/sham 每场的原始 UAV descriptor 哈希、原始候选点数与模型容量相同；
   输出 SurfaceBEV 的空间格与 B0 P3 均为 128²。
3. 100 个 val scene 全完成，所有 DeviceZ、K/C2W/W2C、P3 feature 与 SurfaceBEV
   均有限；每场 correct 与 sham 均至少保留一个 z-buffer 表面点，且 correct 的
   覆盖率、view-count、z-buffer residual 不出现 NaN/负计数/超出图像格的异常。
4. 高度边界只来自 694 train 场，严格递增且全部有限；val 不参与分位数估计。

遥测只报告每场 `input_points`、`in_fov_points`、`zbuffer_points`、covered cells、
非零 bin、view count 和 z-buffer residual 的分位数。它能证明接线与数据几何可用；
不能证明 UAV 有任务信息，更不能证明 segm/bbox AP 增益。

## 后续唯一分支

H0_PASS 后，另立 P1-V2 信息读出预注册：train 拟合等容量线性读出、val 一次确定
正则/阈值、test 一次封存，并含 SAT / correct / sham、boundary-F1、error-AUROC 与
mask-IoU 伴随门。H0_FAIL 则先修几何/接线，不得训练融合模型。
