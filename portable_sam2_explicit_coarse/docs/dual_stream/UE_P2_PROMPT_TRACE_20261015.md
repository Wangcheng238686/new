# P2到粗掩码、提示与SAM的冻结追踪

日期2026-10-15；EXP-0006/0007。当前开发子集153验证图、4496次建筑出现、237栋建筑。没有训练、参数更新、生产接线或遥测修改，没有新增AP/RPN/bbox计算。实施、smoke与正式结果独立审核均PASS。

## 1. 结论

**P2变化能传到SAM解码器；本轮没有产生稳定更好的粗掩码或最终mask。** 因而不能声称“有用的P2信号已经形成，只是被SAM压制”。复用P3融合器的冻结跨尺度迁移限制仍然成立，不能据此否定适配后P2融合的价值。

本轮明确了两个结构限制：

1. FPN P2经RoI、粗掩码、点/密集提示影响SAM；它不是SAM解码器直接使用的高分辨率图像特征。仅在FPN P2加UAV，不等同于向SAM高分辨率视觉输入补充UAV细节。
2. 197个小目标在32²网格没有框内支持cell，形状密集提示被完全消去；占小目标197/881=22.36%，占全部4.38%。这是实际提示支持限制，不是UAV几何覆盖率，也尚未证明是双流缺少净收益的主因。

## 2. 协议与回放验收

[预注册](probes/results_20261015/P2_PROMPT_TRACE_PREREGISTRATION.md)、[完整汇总](probes/results_20261015/p2_prompt_trace_summary.json)、[运行manifest](probes/results_20261015/p2_prompt_trace_run_manifest.json)、[逐图不变量](probes/results_20261015/p2_prompt_trace_invariance_checks.json)、[独立审核全文](../reviews/UE_P2_PROMPT_TRACE_20261015.md)。

固定E72 strict EMA、原正确P3/P4与SAM融合、GT共同框和顺序。复现上轮四组：baseline、原生正确P2 .25（主比较）、原生供体P2 .25、原生正确P2 1（压力对照）。供体沿用donor_a、目标深度/相机不变。只改变mask路径的FPN x[0]。本轮不重复上采样或完整检测。

forward hooks只读、返回None，按生产chunk32拼接，捕获RoI、coarse64²、真实点、canvas128²、PE dense32²、decoder实际sparse/dense、tail前后logits和正式paste mask。GT仅用于离线质量诊断，不作为模型输入。原始中间张量不落盘。

- 修复版三图smoke：每臂hook开/关最终mask逐像素相同。
- 正式153图：四组共17984个GT终点IoU逐项精确等于上轮P2探针；839个非P2 GT的RoI/coarse/paste保持精确不变。
- 42图实际覆盖多chunk；跨chunk顺序、boxes、数量、参数version、特征hash、RGB/几何恢复与hook移除验收通过。
- 正式不变量中的observer_smoke_pixel_exact=false表示未重复关闭hook推理；其验收由单独三图smoke承担，不能当作正式失败。
- 首次smoke因未显式设置mask_roi_chunk_size属性而在manifest前退出；修为生产同款getattr(...,32)默认，没有改模型或数学对照。失败日志与修复后日志均保留。

## 3. P2变化到达哪里

以下只看3657个直接读取P2的GT，连续阶段按逐实例RMS>1e−6计数；末行按实际paste二值mask变化。不同阶段维度/尺度不同，计数不能当信息保真率。

| 阶段 | 正确 .25 | 供体 .25 | 正确 1 |
|---|---:|---:|---:|
| RoI特征 | 3473 | 3473 | 3473 |
| 粗掩码logits | 3473 | 3473 | 3473 |
| 提示canvas | 3473 | 3473 | 3473 |
| PE原始dense | 3444 | 3449 | 3460 |
| 实际sparse提示 | 1282 | 1362 | 2579 |
| 实际dense提示 | 3204 | 3227 | 3303 |
| SAM raw logits | 3397 | 3396 | 3425 |
| tail后logits | 3397 | 3396 | 3425 |
| 最终paste mask | 2092 | 2153 | 2933 |

正确.25下，有2191个粗掩码发生数值变化但点提示不变；点挖掘的离散选择可以保持不变，同时dense路径仍传播。没有实例出现sparse/dense均不变但最终paste改变。仅凭观测重叠不能分离dense、点与tail各自的因果贡献。

## 4. 变化是否有益

粗IoU以正式get_coarse_targets生成64²诊断目标、logit0阈值计算；最终IoU以正式paste后的512²二值mask计算。两者口径不同，不直接比较绝对数值高低来归因解码器。

| 对照 | 粗掩码IoU | 最终GT框IoU |
|---|---:|---:|
| baseline | .870204814 | .841768525 |
| 正确 .25 | .870201575 | .841736646 |
| 供体 .25 | .870170995 | .841807435 |
| 正确 1 | .870082056 | .841842899 |

预定主比较正确.25−baseline：

- 粗IoU差−.00000324，1000次配对图像sum/count bootstrap 95%区间[−.00004626,+.00003585]。
- 最终IoU差−.00003188，同口径区间[−.00013429,+.00008172]。
- 最终IoU改善>.002有243次、退化>.002有265次；粗IoU对应122/113次。全量分组和建筑等权结果见JSON。
- 正确−供体粗IoU仅+.00003058，最终IoU−.00007079；均是描述性结果，未证明有效内容优势。
- 强度1变化更大，但粗IoU差−.00012276、最终差+.00007437，区间均跨0；不能后验改选为成功方案。

本轮bootstrap seed1750，与上轮P2探针使用的重采样不同，区间会略不同；所有逐GT终点完全一致。区间不涵盖空间相关、验证集选检查点与单seed训练不确定性，不是独立test推断。

## 5. 点提示与小目标密集提示

按P2 GT、各槽有效数作分母：baseline两个正点正确率99.289%/99.207%，两个负点61.236%/87.490%；正确.25为99.289%/99.207%与61.181%/87.583%。这些只是按64² GT诊断目标采样的标签一致性，不是点提示因果收益或全图边界精度。无效点不计错，不能用全GT×4作分母。

密集提示在32²上按框内cell中心裁切。197个support=0实例全部属于small（框面积<1024），所有臂实际送入decoder的dense均精确等于no-mask基底。正确.25下其中121个RoI/coarse改变、47个点提示改变、22个最终mask改变：**dense关闭并不意味着整条提示路径关闭**，点仍可传递变化。

该比例基于GT共同框，正式预测框下的比例可能不同。这是当前形状提示的框支持规则，与BEV投影有效范围无关。小目标更难、dense支持为空与质量下降不能自动构成因果链。后续若验证支持规则需独立冻结干预，不直接改生产默认。

## 6. 接线与训练动力边界

实际路径：

```text
卫星backbone → FPN P2 → RoI → coarse64²
                              ├→ point miner → sparse prompts
                              └→ detach → canvas128² → PE → box support32² → dense prompts
卫星backbone高分辨率特征 ───────────────────────────────────→ SAM decoder
卫星SAM stride16 embedding + UAV融合 ──────────────────────→ SAM decoder
```

检查点/源码核定：shape融合roi_only，dense detach_input=True，point miner内部no_grad/detach，PromptEncoder冻结；coarse auxiliary权重固定.1，shape-dense alpha=.32814547（不是UAV融合gate）。因此最终mask loss不沿coarse生成提示的路线反传，coarse仍有辅助监督，SAM图像embedding仍有直接分割梯度。

这些是继承单流的既有接线事实。本轮没有反传，没有量化梯度或证明训练瓶颈，更不能据此直接解除detach重训。dense final/PE变化范数比均值约.239也不能解释成“只保留24%有用信息”。

## 7. 下一步判断

本轮关闭“是否仅仅变化没传到decoder”的疑问：整体传播存在，主要缺口是尚无稳定质量改善，以及当前路径是提示介导而非P2直接视觉细节。

若继续低成本定位，优先单独预注册冻结的GT粗形状提示oracle：保持SAM视觉、框和权重不变，测试更准确的形状提示是否还有可利用空间。它只测条件上界、存在提示分布迁移，不能证明UAV能生成这些提示或正式净收益。小目标支持修正可作为之后独立对照；不同时改多处，不自动扩到全FPN或启动训练。

## 8. 复现与归档

原始工件：`/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis/p2_prompt_trace/`。本轮脚本精确副本归档于`scripts/analysis/ue_p2_prompt_trace.py`及`ue_p2_prompt_trace_summary.py`，保留运行时固定路径（含/tmp helper）；它们是实验快照，不是通用生产入口。重放需恢复原数据/检查点与已归档ue_rgb_content_probe helper到预期路径。

源代码SHA和数据/供体/预注册SHA见manifest；完整逐GT标量/点记录与日志留在原工件目录，轻量汇总、manifest、不变量与审核入git。没有新增生产遥测字段，DEBUG_FIELDS定义保持不变。
