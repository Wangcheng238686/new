# SURFBEV-D1b-InstanceDensity-R1 结论

## 裁决：D1b_PASS（可信的数据几何准入）

该只读探针不训练、不读取模型特征、不报告 AP。它沿用 D1 的固定输入和空间分层：1200-pair
manifest（SHA256 `f15cd8f5b8d10a69d721546d6cfa1f2cce58af2861d57b1357c5e9ff1e1fe62a`）、81 个
cell、每 cell 字典序前三个 scene，共 243 个；243/243 完成，无错误。

问题是：若将 UAV 的 DeviceZ 表面投到卫星平面，正确几何相对同点数、同深度分布、同相机、仅
旋转 world XY +90 度的 sham，是否给每个卫星 visible roof-top 实例提供更多空间覆盖。

| 指标（correct - sham） | correct | sham | 差值 | paired bootstrap 95% CI |
| --- | ---: | ---: | ---: | ---: |
| instance macro coverage（主） | 0.889254 | 0.863395 | +0.025859 | [+0.019620, +0.031946] |
| instance coverage Q25 | 0.887995 | 0.832938 | +0.055056 | [+0.043715, +0.066216] |
| roof-pixel coverage | 0.927970 | 0.901007 | +0.026963 | [+0.022680, +0.031041] |

预注册的三项 CI 下界均严格大于零，故通过。直白说，正确 UAV 几何平均多覆盖每栋楼约 **2.6 个
百分点**；对该场景中覆盖较弱的四分之一实例，优势约 **5.5 个百分点**。这说明 UAV 的 SurfaceBEV
不是只“碰到”屋顶，而是提供了按实例可利用的空间密度。

## 与 D1 的关系

D1 仍为可信的 FAIL：它的二值主指标在两臂都约 98%，因饱和而未能判出差异。D1b 不修改 D1 的
门、样本或结论；它是一次新预注册、用连续覆盖度回答不同问题的探针。两个结果合起来的诚实表述是：
**几何密度优势成立，但 D1 未能证明“每个实例至少有两点支持”的二值优势。**

## 准入与边界

独立审计为 AUDIT PASS，见 `docs/reviews/SURFBEV-D1b-InstanceDensity-R1_independent_audit.md`。
因此允许下一步设计一个冻结模型信息探针：将预计算、卫星平面 z-buffer 的 UAV SurfaceBEV 特征
注入卫星单流，并以正确几何、90 度 sham、零注入三臂检查机制与既有 detector 字段不受污染。

它仍不能证明检测或实例分割 AP 增益；训练必须等模型结构、可重复 feature contract、预注册的 AP
门和独立接线审计齐备后才能启动。
