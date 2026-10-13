# SURFBEV-P1-V2：冻结 SurfaceBEV 信息读出门

## 路由与输入

H0_PASS 后的模块叠加型冻结信息门；不训练 detector、UAV encoder 或融合模块。唯一
卫星锚点为 EXP-0006 seed-44 的 `last_model_epoch100.pth`（H0 登记 SHA）。数据固定为
v2 split 694/100/151；H0 冻结的 12 个 world-Z bin 边界原样复用。UAV 输入只可为
RGB、DeviceZ、K、C2W/W2C；禁止 WorldPosition、卫星标签或 B0 预测进入 SurfaceBEV。

## 读出与标签隔离

每场分别被动 harvest B0 P3、correct SurfaceBEV 和 90°-XY-sham SurfaceBEV。correct
与 sham 共用同一 frozen UAV descriptor、原始候选数、随机投影、训练 cell、线性模型
维度和优化器；差异只能是世界点的 90°旋转后重投影/z-buffer。

卫星 strict ID 仅在 readout 标签生成阶段读取：roof-boundary target 是 1024² roof union
经固定 2px morphological boundary 后 max-pool 到 B0 P3 128²；它绝不参与任何 UAV feature
或几何计算。三臂都用固定随机正交投影到 64 维、`StandardScaler + LogisticRegression`
（class-balanced, LBFGS, max_iter=2000）：`SAT` 只投影 P3；`correct`/`sham` 先拼接同维
SAT 与各自 SurfaceBEV，再投影到相同 64 维。每个 train scene 固定抽最多 256 个正和
256 个负 cell；随机 seed 20261012。

## 四要素和封存流程

- 主指标：每场 P3 roof-boundary F1。伴随一是 B0 预测 mask 与 strict roof boundary 的
  固定 2px error-map 的 cell AUROC（分数越高越能定位 B0 边界错误）；伴随二是每个 B0
  预测实例与其 best-IoU strict 实例的 IoU 的预测 Spearman 相关。实例臂将同一 64 维
  cell field 在该预测 mask 内作平均，再以相同 L2 的 Ridge 读出；无匹配预测的 IoU=0。
  三项均测 frozen feature 的可读取信息，均非 detector AP。
- 选择：只在 val 一次从 `C={0.01,0.1,1,10}` 和 F1 threshold `{.1,…,.9}` 选择同一
  三臂共有的 C/threshold（以 SAT+correct 的 val F1 最大，平局取小 C/小阈值）；test
  只运行一次且不再调参。
- 显著性：test 151 场 scene-paired percentile bootstrap 10,000 draws、seed 20261012。
- P1_PASS：完整性/哈希/标签隔离/coverage 无异常，并且 test 上 `correct−SAT`、
  `correct−sham` 的 boundary-F1 95% CI 下界均 >0；两项伴随指标的两组比较 CI 下界也
  均 >0。否则 P1_FAIL，关闭该接线，不进入 H1。

P1 的测试集开封是一次性的；若实现/审计失败，必须在读取 test 前修复重审。P1 通过也
仅放行 H1 热启动筛查，不能声称 mask/bbox AP 收益。

## R1：可恢复执行实现（2026-10-12，test 前修复）

首次正式执行完成 train/val 的 794 场冻结 harvest 后，在读出拟合开始前非正常退出，
未读取任何 test scene，且没有生成结果工件。本修复不改变读出特征、三个实验臂、训练
样本上限、C/threshold 搜索、指标或判定门；只把执行拆为两个可审计阶段：

1. `harvest` 仅读取 train/val，并逐场原子落盘冻结读出特征和几何遥测；它拒绝
   `--allow-test`。每个 train scene 的正负 cell 抽样改为由 `(SEED, scene_index,
   target_index)` 独立确定，以保证中断恢复时同一 scene 的样本不变。
2. `evaluate --allow-test` 先从落盘 train/val 特征完成拟合和唯一的 val 选择，再创建
   写入所选 `C/threshold` 的不可变 test lock，之后才读取 test。test 场景也逐场落盘，
   仅允许在相同 lock 下恢复未完成场景；不得修改任何选择或重开 test。

cache contract 锁定 B0 SHA、H0 SHA、height-bin 边界、split、随机种子、容量和源文件
哈希。R1 在 test 前须完成独立接线复审；R1 没有重置或复用首次异常执行的任何内存数据。

### R1a：审计阻断修复（2026-10-12，仍未开 test）

R1 独立审计发现 `uint8` mask 矩阵乘法会使实例交集面积按 256 溢出，故实例 IoU 的
Ridge/Spearman 伴随门无效；现改用 `int32` 计算交并面积。审计还发现 contract 未实际
比较源码哈希；现改为 fail-closed 的完整字典比较，并锁定 P1 runner、SurfaceBEV
rasterizer、raw-pair reader、H0 helper 和 checkpoint inference helper 的 SHA256。此前
R1 smoke/cache 不得复用；R1a 必须以新目录重新 harvest train/val、复审后才能开 test。
