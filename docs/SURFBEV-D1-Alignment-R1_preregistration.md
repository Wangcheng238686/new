# SURFBEV-D1-Alignment-R1：UAV 表面—卫星 roof 标签几何对齐探针

**性质：**只读数据探针；不读模型特征、不训练、不报告检测或分割 AP。

## 问题

DeviceZ 已经通过绝对 camera-Z 标定。D1 只检验一个必要条件：将真实 UAV 表面点投到
卫星相机平面、按卫星视角 z-buffer 后，是否比同一表面点的 90° XY 旋转 sham 更贴近
卫星的 visible roof-top instance raster。

## 不可变输入与处理

- 仅使用 `ue_raw_hero_uav_pairset_20261011/accepted_pairs.jsonl`（SHA256
  `f15cd8f5b8d10a69d721546d6cfa1f2cce58af2861d57b1357c5e9ff1e1fe62a`）。
- 对每个 UAV pixel 解码 `z_cm=10/DeviceZ`，以 UAV `K,C2W` 回投，再以该 scene 的
  satellite `K,W2C` 投影。UAV 像素固定 stride=4；不以结果调 stride。
- 在卫星像素上按最小 satellite camera-Z 保留表面（z-buffer）；点占用以 2-pixel
  方形膨胀后与 strict raster 比较。
- sham：每个 UAV 回投点绕卫星 camera center 的 world XY 旋转 +90°，Z 不变，然后
  完整重投影和 z-buffer。它保留深度分布、点数、相机和标签，唯独破坏位置对应。
- 用 `(gx,gy)` 的 81 个 cell 作分层：每 cell 按 scene 名排序，取前三个 accepted scene；
  所以最多 243 场景。不能用结果替换场景。

## 指标、口径与门

- **主指标：**每 scene 的 `instance-supported ratio`：每个 raster 可见实例中，至少有
  两个 strict-instance 像素被 2px 膨胀后的 z-buffer surface occupancy 覆盖的实例比例。报告 correct−sham 的
  scene-paired bootstrap 95% percentile CI，10,000 draws、seed `20261011`。
- **伴随：**roof-pixel coverage（strict foreground 被 2px surface occupancy 覆盖比例）和
  surface precision（2px surface occupancy 落入 strict roof foreground 的比例），同样报告
  correct−sham CI；另报有效点数、z-buffer 保留点数、visible instance 数。
- **选择口径：**预定 81-cell 分层样本的一次扫描，无调参、无 best。
- **PASS：**无数据/矩阵/标签错误；主指标 correct−sham 95% CI 下界 `>0`；两个伴随
  指标的 correct−sham CI 下界也均 `>0`。否则 D1_FAIL，停止 SurfaceBEV 模型设计。

## 判读边界

PASS 只表示 UAV 的已标定表面在空间上更对齐卫星 roof 标签，因而值得进入卫星单流
基座和冻结信息探针。它不证明 UAV RGB 内容有额外语义，也不代表任何 bbox/mask AP
提升。D1 使用卫星标签仅作审计，不会作为之后 UAV 分支的监督或输入。
