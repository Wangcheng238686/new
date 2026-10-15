# UE UAV双流首轮完整实验分析

日期：2026-10-15。EXP-0006 dual / EXP-0007 single；代码 `af76c8ec13a324bacd62ab6b9f8a95b2ccfaa392`。

## 结论

**已证明当前模型会使用对齐的UAV辅助路径；尚未证明相对独立单流有可靠净收益。**
双流best分割AP100仅高0.196个百分点，伴随bbox低0.151个百分点；探索性配对区间跨零。
关闭或错位UAV使同一双流模型分割AP下降约7个百分点，但独立单流仍几乎追平正常双流。
这是强依赖与小净收益共存，不能把干预掉分冒充新增信息贡献。

## 1. 训练与证据身份

- 两臂80轮完成；最终checkpoint均epoch80、history80、所有Adam step6240，独立EMA存在。
- 同manifest SHA256 `495841b2174242c87661e6b40dba2ddcca13d64b13798116efc8c933c679ca59`；627train/153validation。
  1016坏标签已明确排除；两臂共享原空间边界、跨集合global建筑ID交集0。
- 同通用SAM2权重/几何合同/seed42/4卡batch1/累积2/80轮。从generic独立联合初始化，无UE热启动或resume。
- 每rank156mini-batch、每轮78update，训练sampler每轮丢3样本；总6240成功Adam更新。两臂AMP skip均0。
- checkpoint参数差仅config/输出目录，数据差仅load_uav；runtime/pretrained/评测合同相同。
- 两臂E1–4验证用raw，E5–80用EMA(decay.999)，曲线在切换处断开。E5骤降已有同checkpoint raw/EMA完整对照解释。
- dual选best-segm E72，single选E66；四组推理都strict加载EMA，完整覆盖同153图/4496GT，detector score排名、阈值0，主AP100，AP300另报。

### 进程管理异常

双流队列记录退出码0。单流TCPStore heartbeat警告首次约08:17(E67后)，早于最终保存30分钟，
不能写成仅退出时噪声；但E68–80跨rank训练统计、完整验证和可读最终checkpoint均已确认。
队列PID已不存在、GPU空闲，而run_state仍single running并带旧失败error，属于过期管理记录。
本报告保留原记录，不伪造单流正常退出码0。另存completion_audit.json记录“产物完成、正常退出未证实”。

## 2. 预注册主比较与伴随指标

下表是各臂**分割最佳**checkpoint，同一checkpoint上的伴随指标；不是另选bbox-best。
AP为0–1，差值单位为百分点。

| 指标 | single E66 | dual E72 | dual−single（百分点） |
|---|---:|---:|---:|
| segm AP100 | 0.583000 | 0.584958 | +0.196 |
| bbox AP100 | 0.619967 | 0.618454 | −0.151 |
| segm AP50 | 0.872009 | 0.880271 | +0.826 |
| segm AP75 | 0.700363 | 0.703363 | +0.300 |
| segm AP small | 0.465966 | 0.465962 | −0.0005 |
| segm AP medium | 0.671750 | 0.666108 | −0.564 |
| segm AP large | 0.493805 | 0.511606 | +1.780 |
| segm AR100 | 0.643505 | 0.643950 | +0.044 |

GT面积桶沿用本项目既有 **bbox面积** 分桶，不冒称mask面积分桶：small881、medium2510、large1105。
大目标改善可作为下一步诊断线索，未做分层多重检验/种子复验，不能立即宣称特定场景稳定收益。
独立bbox-best分别为dual E74 .6191、single E67 .6201，仅附报。

### 末期与收敛

| 口径 | single segm | dual segm | 差（百分点） | single bbox | dual bbox |
|---|---:|---:|---:|---:|---:|
| 最后5轮均值 | .58078 | .58318 | +.240 | .61564 | .61621 |
| E80 | .5808 | .5819 | +.110 | .6153 | .6152 |

双流EMA分割达到.50/.55/.58分别为E30/E41/E60，单流为E32/E43/E63；前期优势更大，末期收敛到相近平台。
这与收敛速度差一致，但EMA模型曲线还受平均滞后影响，不能据此前期差值宣称最终性能收益。
两臂60轮后进入平台，best在66–72附近，80轮末略回落。没有证据支持仅延长训练就能扩大净收益。

153张图的独立逐图AP比较：82图提升、71图下降；逐图AP均值不是全数据COCO AP，不混用。

## 3. 固定双流检查点干预

固定dual E72 EMA，权重不更新，非热启动；所有153图相同。

| 模式 | segm AP100 | bbox AP100 | segm AP75 |
|---|---:|---:|---:|
| correct | .584958 | .618454 | .703363 |
| zero | .509657 | .616496 | .571179 |
| shift | .513302 | .614876 | .582057 |

- zero：绕过整个UAV编码/BEV/融合，卫星模型仍是协同训练的双流卫星分支；不等于独立单流。
- shift：`bev.roll((16,16))`，两个坐标方向各64m，循环平移；原几何投影/置信不变，是强失配干预，不是真实标定噪声模拟。
- correct−zero：segm +7.530个百分点，bbox +0.196；correct−shift：segm +7.166，bbox +0.358。
  下降主要体现在分割及AP75，检测框总体变化小。与SAM入口较强消费一致，但不构成SAM独立归因。
- 模型对辅助路径及其空间对应敏感。因干预是分布变化，也改变了位置query特征与几何的对应，不能单独证明增益来自UAV RGB新增视觉细节。
  深度、RGB、坐标先验、注意力和各融合入口的贡献仍未拆分。

## 4. 不确定性

500次image-paired percentile bootstrap，seed44，使用完整153图的COCO逐图匹配缓存、官方accumulate。
完整点估计与四组评测精确一致；8位置含重复图样本与重新复制记录并evaluate精确一致，避免重复图ID被去重。

| 差值 | 点估计（百分点） | 探索性95%区间（百分点） |
|---|---:|---:|
| correct−single | +0.196 | [−0.201, +0.494] |
| correct−zero | +7.530 | [+6.639, +8.057] |
| correct−shift | +7.166 | [+6.374, +7.533] |

此为选模后的validation探索：不覆盖best选择偏差、单seed训练波动或空间相邻样本相关性。
不将bootstrap正差比例解释为优于单流的概率或正式p值；不将此区间当独立test/论文显著性。

## 5. 遥测机制

口径：raw训练模型，所有rank每字段有限forward均值；dual每字段每轮分母624。不是EMA验证权重遥测。

| 字段 | E1 | E21 | E80 |
|---|---:|---:|---:|
| 真实观测BEV全格比例 | .6946 | .6942 | .6945 |
| 有效代表高度BEV全格比例 | .4367 | .4365 | .4367 |
| FPN s8实际残差/卫星L2 | .0477 | .1083 | .1211 |
| FPN s16同上 | .0486 | .0987 | .1015 |
| SAM同上 | .1323 | .7364 | .9457 |
| SAM支持区gate | .7009 | .9898 | .9893 |
| SAM alpha | .0518 | .1673 | .2221 |
| attention本格mass | .1450 | .1379 | .1385 |
| attention高度相容mass | .5140 | .5101 | .5083 |

几何支持/质量稳定符合缓存设计预期，目标卫星网格支持率FPNs8约91.5%、s16/SAM约95.4%；
与BEV height_valid不同分母，不相互矛盾。梯度日志确认UAV/CVT/FPN/SAM持续有有限梯度。
SAM gate近全开、alpha增长、残差接近卫星范数，说明其已经不再只是小扰动；不表示94.6%的信息或AP来自UAV。
FPN gate最终约.868/.846，仍有抑制。均值不足以证明门控有空间选择性或能识别错位。

attention没有随训练变得明显更局部/高度一致。约13.9%本格、50.8%高度相容，保留大量上下文。
该统计是最后层有候选query的采样，不只限height_valid投影格；加权XYZ偏差有抵消，不是GT几何误差。
不能据此直接认定CVT失败，也不能声称已经学出更精确跨视角对应；无均匀候选基准或GT对应。

## 6. 成本

实际训练过程wall time（含每轮验证/存盘）：dual约10小时51分，single约3小时31分，约3.08倍。
两臂先后运行，dual阶段存在渲染/IO共用，不能把此比值当纯模型吞吐或FLOPs对比。
最大rank训练allocated峰值：dual7777MiB，single5294MiB（约+47%）；不包括UE渲染，reserved不混算。
trainable参数约51.34M vs43.84M（约+17%），表明此系统比较包含容量差，未证明新增视觉信息的独立归因。
以当前约0.2个百分点净差来看，收益尚不匹配额外成本。

## 7. 下一步建议（优先顺序）

1. **先查净收益为何小，而不是继续增加模型复杂度。** 用现有per-image证据挑改善/退化图，重点比较大目标收益与中目标损失，检查SAM mask边界、框质量、投影支持/高度混杂。candidate支持不是可见性真值。
2. **检查SAM入口的过强协同依赖假设。** 这是待验证解释，不是已确认bug；首选已有fpn-only/sam-only训练臂与相同协议对照，评估检测入口与mask入口能否各自创造净收益，避免只强化替代卫星特征。保持通用联合初始化，不热启动UE单流。
3. 若要主张新增RGB信息，可先补保持几何不变的RGB干预诊断，再按论文需要做RGB/几何控制；推理干预只能说明依赖，归因仍需重训控制。骨干冻结和深度大矩阵消融按既定约定后置。
4. 有持续明确的开发收益后再做多seed与独立空间test。当前不应为+0.2个百分点直接宣称成功，亦不能以此单次小差否定UAV数据集或研究设想。
5. 保留本轮EMA配方结果；下一轮在启动前明确固定raw/EMA选模口径或适配update预算的EMA策略，不事后挑最好权重。修复队列生命周期记录是独立工程事项，不覆盖本轮证据。

## 8. 工件与复现

完整输出根：`/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis/`。

- `dual_correct/dual_zero/dual_shift/single_best/`：metrics/run_manifest/predictions/GT/DT/images，含零检测图执行身份证据。
- `summary.json`、`bootstrap.json`、`bootstrap_samples.json`、`per_image_ap.json`、`training_parsed.json`、`checkpoint_audit.json`、`final_checkpoints.json`。
- `convergence.png/.pdf`、`telemetry.png/.pdf`及灰度图；机器layout检查空，300dpi strict检查通过，人工复核通过。
  初版遥测图自动y上限截断SAM曲线，已发现并修正为覆盖全数据范围，未改数据。
- 可复现分析脚本同目录 `analysis_scripts/`；profile.txt明确160行、每臂80轮，无缺失；epoch不视为独立统计样本。
- 审核全文：[最终独立审核](../reviews/UE_FINAL_RESULTS_20261015.md)。版本化轻量摘要在 `probes/results_20261015/`，日志/权重/大图不入库。
