# UE首轮实验最终独立审核

日期2026-10-15。以下两名审核者未参与本轮结果脚本编写，独立只读检查原始日志、checkpoint、推理输出及bootstrap实现。原始工件根 `/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis/`。

## implementation_training_review 回复全文

**VERDICT：PASS，可据此写回开发实验分析；须保留队列管理异常说明。** 本次只读审核未发现训练未完成、配对身份不一致或AMP数值失败的证据。

- **完成与实际预算**：直接读取两臂最终checkpoint，均为E80、history 80轮、Adam step 6240、`ue_optimizer_updates=6240`。实际627 train、153 validation；DistributedSampler `drop_last=True`，每轮处理624张、每rank156个microbatch、78次更新，共6240次。不能沿用原计划628样本、79次/轮。
- **配对口径**：两臂同manifest哈希、几何签名、通用SAM2权重哈希、clean commit、seed42、有效batch8及80轮预算。checkpoint记录的训练参数仅config与输出目录不同，数据合同仅`load_uav`不同。未使用resume或UE单流热启动。
- **AMP稳定性**：160个epoch runtime记录齐全，每轮跨rank尝试数312，两臂全部AMP skip=0；scale正常由512升至4096。最终Adam状态支持6240次成功更新，而非仅尝试计数。
- **EMA口径**：两臂decay=.999、每步更新，第5轮起验证使用EMA，最终与所选best checkpoint均记录`evaluation_weights=ema`且独立EMA存在。E5指标骤降与raw→EMA切换一致，不能解释为训练崩溃；E1–4与E5以后不是同一评测权重口径。
- **选模结果**：best-val-segm选中dual E72：segm .58495797、bbox .61845387；single E66：segm .58300044、bbox .61996746。差值为segm **+0.1958个百分点**、bbox **−0.1514个百分点**。不能用独立bbox-best替换这些同checkpoint伴随指标。
- **末期走势**：最后5轮segm均值dual .58318、single .58078，差 **+0.240个百分点**；bbox均值 .61620、.61566，差 **+0.054个百分点**。E80 segm .5819/.5808，bbox .6152/.6153。两臂已进入平台并略有回落，不支持“持续提高”或“所有指标均改善”。
- **单流TCPStore异常**：首个heartbeat监视警告约08:17:12，发生在E67后，早于最终保存约30分钟，因此**不是仅shutdown阶段警告**。之后E68–80仍有完整跨rank遥测、验证及有效最终checkpoint；可确认训练产物完成，但无充分证据声称正常退出码0。队列`single running`及遗留error应标注为stale管理状态，不能据此推断仍在训练。
- **结论边界**：当前是同一validation、单seed的开发比较。结果为小幅segm差异、选中checkpoint的bbox略低；不能据此形成统计显著性、独立test或论文总体增益结论。

## implementation_integration_review 回复全文

VERDICT：PASS。未发现新 bootstrap 的 P1/P2 实现错误；结论必须限定为此次固定模型、固定验证集的探索性比较。

四组均严格加载 EMA，完整覆盖相同 153 张图、4496 个 GT；评分均为 detector `scores`，阈值为 0。双流使用 E72，独立单流使用 E66，checkpoint 内 epoch 字段为从零计数。

bootstrap 方法与当前协议一致：

- 先逐图计算 COCO 匹配，再对同一组图像索引进行配对重采样；重复图像占独立位置，不会被 ID 去重。
- 官方 `accumulate` 只消费匹配状态，重复缓存中的原 annotation ID 不影响累计 TP/FP。
- 使用 all area、maxDets=100 和默认 IoU/recall 阈值；GT bbox area 沿用项目现有行为。
- 四组全量点估计均与既有指标精确一致；包含重复图的八位置样本，与直接复制记录、重新 evaluate 的结果精确一致。
- 最终 500 次重采样已完成。正差比例是 bootstrap 描述量，不能解释为优于对照的概率或正式 p 值。

以 AP 百分点表示：

| 比较 | 分割差值 | 探索性 95% percentile 区间 |
|---|---:|---:|
| correct − single | +0.196 | [−0.201, +0.494] |
| correct − zero | +7.530 | [+6.639, +8.057] |
| correct − shift | +7.166 | [+6.374, +7.533] |

允许的结论：此次双流模型的分割输出明显依赖 UAV 融合路径，并对 BEV 特征与原几何投影的错位敏感。`zero` 是整条 UAV 融合旁路；`shift` 是 BEV 循环平移 16 格且保留几何。两者均为训练后干预，不能分别证明深度、注意力或某一融合位置的独立收益。

尚不能声称双流优于独立单流：分割差值很小且区间跨零，bbox AP 还低约 0.151 个百分点。同一验证集用于 checkpoint 选择，且仅单 seed、空间相邻样本；图像 bootstrap 不覆盖选择偏差、seed 波动或空间相关性。本次未修改文件。

## 主代理闭环

上述限制与管理异常已写入结果报告，未修改训练产物或旧run_state。图表将raw/EMA验证切换断开，不把epoch视为独立样本；机器布局审计与300dpi检查通过；人工读图发现初版遥测y上限裁掉SAM曲线后已修正并复查，数据未改。权重、日志、大图只在工件目录，库内仅结果文档与轻量摘要。
