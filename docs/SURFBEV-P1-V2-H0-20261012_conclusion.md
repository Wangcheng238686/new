# SURFBEV-P1-V2-H0：冻结 SurfaceBEV 采集与接线结论

**结论：H0_PASS。** 这只是 SurfaceBEV 的无标签几何/接线准入，不是 UAV 提升
卫星检测或实例分割 AP 的证据，也不允许跳过 P1 直接训练 H1。

## 预注册门的落判

- **锚点身份：PASS。** 只使用 EXP-0006 seed 44 的 epoch-100 last checkpoint；
  SHA256 为 `fc3ead0e9490ffd9249cdbc4ce2ee9556bed5252efc3e475dfcc0f2aace65145`，
  checkpoint 零基 epoch=99、run tag、seed=44、epochs=100 与 v2 split 均匹配。
- **数据隔离：PASS。** 694 train 场仅用于高度分位数，100 val 场全部采集；H0 专用
  reader 不打开实例 ID、未构造任何 `gt_*` 字段；151 test 场未读取。
- **被动 parity：PASS。** 每一个 val 场景都在 P3 harvest 前后运行 B0 标准前向，
  bbox、score、label、mask 四字段哈希逐字段一致；B0 实际 P3 均为 512×128×128。
- **correct/sham：PASS。** 每场原始 descriptor 均为 5×256×128×128，原始候选点数
  固定 81,920，descriptor hash 在 correct 与 90° XY sham 间一致。旋转后才重投影和
  z-buffer，因此允许两臂的 in-FOV / winner 数不同。
- **有限性与几何遥测：PASS。** 所有 100 场的 SurfaceBEV、相机审计、P3 均有限，
  correct 和 sham 每场均有非零 winner/covered cell。12 个高度 bin 只由 694 train
  场 world-Z 的 0.5/99.5 分位冻结，边界为 −355.20 至 24,405.48 cm。

## 遥测摘要（每场，min / median / p90 / max）

| 条件 | z-buffer winner 点 | 覆盖 P3 cell | 覆盖率 | mean residual（cm） |
| --- | ---: | ---: | ---: | ---: |
| correct | 8,227 / 14,977 / 16,029 / 16,535 | 6,977 / 13,571 / 14,561 / 15,076 | 42.6% / 82.8% / 88.9% / 92.0% | 521 / 1,664 / 3,313 / 3,742 |
| 90° sham | 8,209 / 14,986 / 16,054 / 16,566 | 6,964 / 13,572 / 14,570 / 15,081 | 42.5% / 82.8% / 88.9% / 92.1% | 517 / 1,655 / 3,302 / 3,730 |

残差是所有进入卫星视野点相对同 cell 最近表面的 camera-Z 距离，在 winner 筛选前
统计；因此它不是恒为零的伪遥测。每个非缺失 P3×高度 bin 的有效视角数 p50/p90
均为 1（最大为 2）：几何上存在五路输入，但在当前 128² cell 与高度分桶下，局部
聚合通常由单一可见视角贡献。这不是失败，却说明后续 P1 必须实际检验
`correct > sham` 的可读取信息，不能以“多视角覆盖”替代该证据。

## 判读边界与下一步

H0 证明 DeviceZ、K/C2W/W2C、卫星 z-buffer 与 B0 P3 对齐的只读数据路径可用，且
correct/sham 对照不会污染 B0。它**不能**证明 SurfaceBEV 包含卫星 P3 之外的任务
信息，更不能证明 segm AP 或 bbox AP 增益。

唯一允许的下一步是 P1-V2：训练集拟合等容量的 `SAT`、`SAT+SurfaceBEV-correct`、
`SAT+SurfaceBEV-sham` 冻结线性读出，val 仅一次选正则/阈值，再对封存 test 做一次
预注册的 boundary-F1 / boundary-error AUROC / mask-IoU 相关检验。P1 未通过不得进入
H1。

## 工件与审计

- 完整 summary：`/data1/wangcheng/checkpoint/surfbev_b0/probes/SURFBEV-P1-V2-H0-full-20261012/summary.json`
- 逐场 telemetry 与哈希：`/data1/wangcheng/checkpoint/surfbev_b0/probes/SURFBEV-P1-V2-H0-full-20261012/per_scene.jsonl`
- [预注册](SURFBEV-P1-V2-H0_preregistration.md)
- [实现独立审计（PASS）](reviews/SURFBEV-P1-V2-H0_independent_audit.md)
