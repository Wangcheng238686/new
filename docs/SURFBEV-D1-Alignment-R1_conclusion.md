# SURFBEV-D1-Alignment-R1 结论

## 裁决：D1_FAIL（可信负结果）

这是只读数据几何探针，未训练、未读取模型特征，也不产生 AP 结论。它固定使用 1200-pair
manifest（SHA256 `f15cd8f5b8d10a69d721546d6cfa1f2cce58af2861d57b1357c5e9ff1e1fe62a`）中 81 个
空间 cell 的字典序前三个 scene，共 243 个 scene；全部完成且无错误。

| 指标（correct - 90-degree sham） | correct | sham | 差值 | paired bootstrap 95% CI |
| --- | ---: | ---: | ---: | ---: |
| instance-supported ratio（主） | 0.981760 | 0.978259 | +0.003501 | [-0.000765, +0.007832] |
| roof-pixel coverage | 0.927970 | 0.901007 | +0.026963 | [+0.022677, +0.031135] |
| surface precision | 0.504268 | 0.487094 | +0.017173 | [+0.014575, +0.019723] |

预注册门要求三项 CI 下界都严格大于零。主指标不满足，故 D1_FAIL；不允许以两个正向伴随
指标将它改判为通过。

## 能证明与不能证明

能证明：在固定表面点数（每 scene 81,920）、近乎相同卫星画面内点数（correct 70,140.7；sham
70,110.2）和同一 z-buffer 下，正确的 UAV 几何令卫星 roof 像素覆盖和表面纯度稳定增加。

不能证明：每个实例都获得了稳定额外支持，更不能证明 bbox 或 mask AP 会提高。原因不是几何无效，
而是主指标只问“是否有至少两个点”，两臂均约 98%，已经饱和。

独立审计记录见 `docs/reviews/SURFBEV-D1-Alignment-R1_independent_audit.md`；审计结论为
**AUDIT PASS，D1_FAIL 可信**。

## 后续

关闭 D1，不进入模型设计/训练。新立项的 D1b 必须用不饱和的、按实例宏平均的连续覆盖度检验
SurfaceBEV 是否具有可供模型利用的空间密度；其标准独立于 D1，不能追溯改写本结论。
