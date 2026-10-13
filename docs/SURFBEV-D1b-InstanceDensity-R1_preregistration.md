# SURFBEV-D1b-InstanceDensity-R1：按实例的 UAV SurfaceBEV 密度探针

**性质：**D1 之后的新只读数据探针。它不重开或改变 D1 的失败裁决，不训练且不报告 AP。

## 问题与理由

D1 的二值主指标“一个实例是否有至少两个支持点”在 correct 和 sham 均接近 98%，出现天花板。
D1b 改问与 feature-map 输入更贴近的问题：正确几何是否为**每个可见卫星实例提供更多表面覆盖面积**。
它仍采用完全相同的 243 场景、DeviceZ 解码、投影、卫星 z-buffer、2px occupancy splat 和
+90-degree XY sham；因此没有重抽样或调参。

## 指标、口径与门

- **主指标：**scene 内对每个 strict visible instance 求
  `covered strict pixels / all strict pixels`，再对实例作等权平均（`instance_macro_coverage`）。
  比较 correct−sham，并给 scene-paired bootstrap 95% percentile CI（10,000 draws，seed
  `20261012`）。大建筑不能主导该指标。
- **伴随一：**每 scene 的实例覆盖率 25% 分位数（`instance_coverage_q25`），同样
  correct−sham CI。它检查较弱覆盖的一组实例是否也受益。
- **伴随二：**strict roof-pixel coverage，作为与 D1 同定义的总体覆盖一致性检查。
- **选择口径：**一次固定扫描；不按结果更改 sample、stride、splat、分位数或统计方式。
- **PASS：**无错误；主指标及两个伴随的 correct−sham 95% CI 下界均 `>0`。否则 D1b_FAIL。

## 判读边界

PASS 仅说明经标定的 UAV 表面有按实例可利用的空间覆盖密度，允许进入随后独立注册的冻结模型
信息探针；它既不覆盖 D1_FAIL，也不是 AP 增益证据。卫星标签仅用于本审计，不作为 UAV 分支
输入或监督。
