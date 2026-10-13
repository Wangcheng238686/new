# SURFBEV-D1-Alignment-R1 独立审计

**审计结论：AUDIT PASS；实验门结论：可信的 D1_FAIL。**

审计人独立核对了预注册、实现和输出工件。

- manifest SHA、81 个空间 cell、每 cell 固定字典序前三场景均一致；243/243 场景完成，无错误。
- `DeviceZ -> camera-Z -> C2W -> satellite W2C` 与行向量变换一致；矩阵互逆抽查通过。
- correct 与 sham 唯一区别是表面点绕 satellite camera 的 world-XY 中心作 +90 度旋转；两者之后使用完全相同的投影、卫星 z-buffer 和 2px occupancy 膨胀。
- 三个指标及 paired bootstrap（10,000 draws，seed 20261011 加指标名偏移）独立复算，与 `summary.json` 一致；脚本可编译，`git diff --check` 通过。

预注册要求三个 `correct - sham` 95% CI 下界均大于 0。实际结果为：

| 指标 | 均值差 | 95% CI | 门 |
| --- | ---: | ---: | --- |
| instance-supported ratio | +0.003501 | [-0.000765, +0.007832] | 未过 |
| roof-pixel coverage | +0.026963 | [+0.022677, +0.031135] | 过 |
| surface precision | +0.017173 | [+0.014575, +0.019723] | 过 |

`instance-supported ratio` 在两臂均接近 1（correct 0.981760；sham 0.978259），存在明显天花板效应。因此不得用两项伴随指标替代主门，也不得据此进入模型训练。该失败不否定 DeviceZ 标定，亦不等价于 UAV 不可能带来 AP 增益；它只否定 D1 的严格三门准入。
