# Native检测／掩码梯度和框误差遥测

用户要求观察det/mask梯度范数、共享参数余弦、bbox中心/宽高误差及FP/FN。当前全100轮ABC冻结运行树已经E36–37；采用独立sidecar，不修改原105份源码、不重启/热改训练、不增加loss/门控/自动选模。

## 文件与启动

| 文件 | 职责 |
|---|---|
| scripts/analysis/ue_native_gradient_math_20261028.py | 严格loss分区、参数覆盖分组、固定坐标任务方向统计 |
| scripts/analysis/ue_native_gradient_telemetry_20261028.py | 四卡固定8图checkpoint VJP、总梯度/重复校准、状态不变审计 |
| scripts/analysis/ue_native_detection_telemetry_20261028.py | CPU读取完整验证records，bbox误差与阈值化TP/FP/FN |
| scripts/analysis/ue_native_telemetry_owner_20261028.py | 独立源快照、最多2 CPU workers、最多1四卡梯度组、资源准入与低频观察 |
| scripts/experiments/ue_native_telemetry.sh | 一键start/status入口 |
| tests/test_ue_native_gradient_math.py | 4项均值梯度、零范数、union/交集、loss分区校验 |
| tests/test_ue_native_detection_telemetry.py | 7项匹配/分解/阈值/尺寸/误差/空图校验 |

```bash
./scripts/experiments/ue_native_telemetry.sh start
./scripts/experiments/ue_native_telemetry.sh status
```

产物在full100 root的`telemetry/`，各arm独立；observer_source保存本次观察器字节，observer_manifest绑定SHA。重复start对活owner幂等，旧失败owner需要显式审查，不覆盖数据。

## 梯度定义

频率：每臂保留E1/10/25/50/75/100六个检查点，各固定同一8张训练图（ds.samples等距索引）；4rank各2张，逐图seed=44000+index，train mode、无增强、FP32。**不是实时随机AMP更新梯度，也不是AdamW有效步长。**固定诊断采用FP32避免把半精度下溢误判为无梯度或冲突。

同一个前向的原已加权loss：

- D=loss_rpn_cls+loss_rpn_bbox+loss_cls+loss_bbox；
- M=loss_mask+存在时的loss_shape_prior（保留原coarse .1，不重乘）；
- I=loss_uav_instances，单列范数，不混入检测（尺寸内部权重保持）。

未知loss key拒绝执行，不静默漏项。autograd.grad不写param.grad，不step/scaler更新，不恢复optimizer状态用于诊断。每图梯度/8后跨4rankSUM，先形成global8均值，再计算L2、点积、cos和M/D范数比；不得先平均rank范数/余弦。

固定参数全集分组：satellite LoRA/其他encoder、FPN、UAV LoRA/其他encoder、geometry encoder、RGB alignment/fusion、interaction/writeback、det heads、mask decoder、native refiner、mask/prompt其余、unassigned。layout逐张量列名/shape/元素数，全部可训练参数覆盖；shared_representation是预注册encoder/FPN/UAV表征集合，不包含专属det或mask头。固定union中None按0。另附joint_connected_only交集结果，仅补充，不代替union。

每组计both/det-only/mask-only/neither的张量及元素数；零范数cos=null/status undefined，不记0或负冲突；native refiner只有mask梯度是结构预期。实例辅助范数单列，避免将其训练动力错归det/mask。额外8图逐图shared_direction用于揭示均值抵消，范数是单图未除8值，不与global8混用。

每图恢复CP buffers后固定seed；结束参数/buffer SHA一致、所有param.grad None、optimizer state空、CP与训练源码SHA不变。首个local图额外校验sum(D/M/I)梯度与总loss梯度，以及同seed重复前向梯度；校准4张额外前向不纳入主global8。FP32容差relativeL2<1e-4或maxabs<1e-5，非有限失败，不产生零值结论。

GPU C/E25作为首次真实模型/峰值门，之前仅CPU math与代码审查通过。sidecar cap14336MiB/card+512上下文+1024余量，四卡都满足15872MiB空闲才准入；最多一组。该cap是保守上限，不是已实测峰值。当前共租训练余量不足时排队；OOM仅标诊断失败，停止后续GPU诊断，不停止原训练或把结果记0。

## 验证框误差与误检／漏检

频率：每个已完成epoch，全部459 val图；CPU最多2进程，新epoch优先、历史补齐。不读test诊断后续优化，不产生AP选权条件。

预固定score>=.05/.25/.5，bbox IoU>=.5/.75；各阈值、combined/original_path独立统计，score稳定降序、同分保持输入顺序，GT一对一，tie取最早GT索引。每行公开images/GT/predictions/TP/FP/FN；TP+FP=预测数，TP+FN=GT数。

- FP互斥：duplicate=maxGTIoU>=当前匹配阈值；localization=.1<=maxGTIoU<阈值；background代理=maxGTIoU<.1（空GT按0）。不是语义或标注完整性认证。
- FN：匹配未覆盖GT；另报score筛选后及筛选前全部已输出候选的best bboxIoU分桶：无候选/<.1/.1–.5/.5–.75/>=.75。后者是已输出最多250候选，不是原始RPN召回。
- TP中心误差=(predcenter−GTcenter)，dx/dy px及dx/GTwidth、dy/GTheight；宽高signed px、log(predwidth/GTwidth)、log(predheight/GTheight)。每字段count=TP，报signed均值、绝对均值/P50/P90。跨臂TP匹配集合可能不同，均值不能单独作为定位改进因果。
- GT尺寸召回按GT mask area（<1024 / 1024–9216 / >=9216），FP尺寸另按预测bbox area；两个分母明确分开。

combined里prefix分数经过trusted_score重编码，original_path保留原分数。跨view相同score阈值并非同一校准操作点；不能把两个view的计数直接相减当UAV追加候选的因果贡献。跨arm应在同view/同阈值下比较。

每份输入records/complete/arm contract与观察器源SHA绑定；只读小JSON输出，不塞进每步训练日志，不参与loss/门控/选权。A/E1真实459图及全部11项CPU测试已验收；新梯度GPU真模型门须等待资源后执行。

## 解释边界

负余弦提示固定训练函数的局部方向冲突，不能单独证明bbox AP下降由此导致；正的global8均值余弦也不能证明所有样本无冲突。需结合norm比例、分组、逐图方向、中心/尺度/FPFN走势及原AP共同判断，不自动调整loss权重/PCGrad/解冻范围。短探针C相对B的bbox差区间跨0，不能预设所有点差都来自冲突。

## 部署状态

观察器提交`1aff788`，owner1642915，已实际启动并回填验证epoch；部署审计时9份CPU报告完成，无失败，重复start幂等。原full105运行源逐份SHA核对无改动，ABC继续训练。root_enable_receipt.json绑定11项CPUtests、A/E1真实459验收及运行状态。

四卡梯度真模型门尚未执行：审计时每卡仅约1312–1445MiB空闲，低于15872MiB准入。E1/E10/E25等保留CP已经排队，不占用/打断训练。尚无实际梯度数值，不能根据该部署认定已发现梯度冲突。
