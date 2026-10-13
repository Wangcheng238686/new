# SURFBEV-D1b-InstanceDensity-R1 独立审计

**审计结论：AUDIT PASS；实验门结论：D1b_PASS 可信。**

- 243/243 scene 完成，覆盖固定 81 个空间 cell、每 cell 固定字典序前三个 scene；无错误、无重复，manifest SHA 一致。
- 两臂每场景均为 81,920 个 UAV 表面点。唯一变化是绕 satellite camera 的 world-XY 地面中心作 +90 度旋转；卫星光轴严格竖直向下，旋转中心定义正确。
- 五个跨空间位置的原始数据端到端重算，correct/sham 指标与 telemetry 均和落盘记录一致。
- paired bootstrap（10,000 draws、seed 20261012、scene 成对重采样）独立复算一致：

| 指标 | correct | sham | 差值 | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| instance macro coverage | 0.8893 | 0.8634 | +0.0259 | [+0.0196, +0.0319] |
| instance coverage Q25 | 0.8880 | 0.8329 | +0.0551 | [+0.0437, +0.0662] |
| roof-pixel coverage | 0.9280 | 0.9010 | +0.0270 | [+0.0227, +0.0310] |

三项 CI 下界均大于 0，故 D1b_PASS 成立。D1 原始输出仍独立保留为 FAIL；D1b 是对 D1
二值“至少两点命中”天花板的独立连续密度检验，不能覆盖或反转 D1_FAIL。

该审计允许进入后续独立注册的冻结模型信息探针，不能推断任何 bbox/mask AP 增益。
