# 根因结论与可行方向（2026-10-24）：WARM-RR 召回救援臂首证

## 一、根因（量化闭环）

基座（EXP-0017 ep73）test 459 场全量错误分解（`r0new_test.json`）：
误差总质量 6866 = **miss 5429（79%）** + false 1319（19%）+ 界面型 118（1.7%）；
miss 的 98% 是小实例（5317 个；中 110、大 0）。VIS-MISS 探针
（`vis_miss_test.json`）：被漏小实例 **96.7% 在 UAV 几何覆盖内**
（90.3% 覆盖 ≥0.9）——旧数据"62% 小实例不可见"结论在新数据不成立。

**三层根因**：
1. **主因：误差构成与 UAV 能力错配**——端到端注入家族（十次失败）瞄准的
   边界/界面类误差只占 1.7%；误差主体是整实例漏检，端到端特征注入
   结构上无法补一个"不存在的实例"。
2. **次因：冻结决策头的消费免疫**（十次实证，含直接监督 n1）。
3. **供给从未缺席**：PIXEL-CLS/EVI-DECODER/n1-aux 三证内容可解码。

## 二、可行方向：WARM-RR（决策级召回救援）

**设计**（UAV 三模态全用）：UAV RGB → EVI-DECODER v2 证据图（外观）；
UAV 深度 → 候选反投影高度通道（≥3m 门）；内外参 → 卫星帧投影 +
多视角门（≥2 视角）。三合一门 → 连通域提案 → 仅在基座无预测覆盖处
注入新实例（基座逐位不动）。阈值/分数在 train 250 场校准后冻结
（thr 0.8 / score 0.3 / hthr 0.03 / views≥2）。

**结果（test 459，同口径基座对照）**：

| | 基座 | WARM-RR | 常量对照 |
| --- | --- | --- | --- |
| segm AP | 0.6455 | 0.6458 (+0.03pt) | 0.6455（+0） |
| **AP50** | 0.8636 | **0.8708（+0.72pt）** | 0.8636（提案数 0） |
| AP75 / AP_s | 0.7560 / 0.5728 | 0.7539 / 0.5712 | — |
| train 校准集 | 0.8907 | 0.8933（+0.26pt 主AP） | — |

- **首个外观依赖的正向信号**：常量 RGB 下提案数为 0、零增益——增益
  确属 UAV 内容（十次注入臂从未做到这一点）。
- 主 AP 未转化的瓶颈明确：提案掩码 4px 块状量化限制 IoU<0.75——
  已登记的下一杠杆 = 证据图双线性上采样精化提案掩码 + 分数排序校准。

## 三、工具与工件

- `scripts/ue/r0new_decompose.py`、`probe_vis_miss.py`、`warm_rr.py`、
  `warm_rr_calib.py`
- `/data2/wangcheng/remote-lthpc/{r0new_test.json, vis_miss_test.json,
  warm_rr/{params.json, calib_grid.json, test_result.json,
  control_result.json, calib_scenes.pt}}`
- 基座预测记录：`evals/base_disabled/disabled/records.json`
