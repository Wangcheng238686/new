# Native尾部固定E10归因分析

## 问题和固定输入

三组10轮2670次更新及459测试已完成：Mask AP250 A .6141469755、B .6088747783、C .6484947488；bbox A .6840373321、B .6721352452、C .6706915423。原始点估计不代替配对区间。所有分析固定既有E10，不拟合、不改结构/预算、不选权。

## 三类检查

1. `scripts/analysis/ue_ubr_native_paired_20261028.py`：核98源当前/冻结/原commit SHA、E10CP、输入顺序和GT，三组combined及original_path，C-A/C-B/B-A；Mask/bbox/BoundaryAP2%与1%，同图配对500次seed20261021，名义95%分位区间。复用冻结COCO matcher和快速累积，每个evaluator前三次校验原累积误差<1e-12。边界使用min(maskIoU,boundaryIoU)，半径按图像对角线比例，512²对应14/7px。
2. `scripts/analysis/ue_ubr_native_attribution_20261028.py --action interventions`：C权重FP32，正确尾部/全零尾部/循环供体错配尾部。尾部为G pre-rgb_fusion aligned32、独立RGB支持及geometry5。供体是测试序列(index+229)%459，按其自身卫星query提取；供体包也更换支持掩码，故检验联合内容/位置/支持一致性，不是RGB-only或物理反事实。目标上游G、候选实例头、框/score保持原样。正确推理逐图框/score/label/RLE必须精确复现原C正式记录。
3. 同脚本 `--action fixed_boxes --arm A_nohead/B_zero`：两模型使用C原框/score/label及原consumer分段/chunk顺序，自己的卫星/G/decoder特征，重新解码掩码。C正确记录作为同框C。固定C权重的零/错配也用同样框与chunk；框/score身份强校验，掩码仍原sigmoid/paste/阈值。跨模型比较排除候选/score变化，但不能分离各自联合训练后上游参数变化与新头本身。

四卡推理，CPU两进程最多并行，独立目录`ue_g_ubr_native_short_20261027/analysis_native_20261028`；所有新代码均为观察分析，训练98源不改。`ue_ubr_native_fixed_box_queue_20261028.py`仅一次性依赖调度：459图C框分段完整落盘后A/B各四卡并行（不必等rank0三组COCO统计，避免GPU闲置），不重训。新分析/预测分别绑定来源SHA、固定CP和state不变检查。

区间只刻画这批图像重采样，不证明跨训练种子稳定性；测试集已暴露，不当独立新基准。上游G存在于所有三臂，不能把A/B称为纯卫星。结论须分为跨模型净差异、固定权重内容依赖、固定框mask质量归因。

`ue_ubr_native_attribution_queue_20261028.py`将后续四组固定记录配对与两组像素纠错串入最多两CPU进程队列，在既有架构bootstrap完成后执行，不占训练预算。干预供体匹配始终事先固定，不能依据结果换供体/阈值。

## 首批审计状态（中间结果，不是完整归因结论）

C干预459图的correct raw正式框/score/label/RLE全部精确复现，固定sourcechunk已落segments；A/B已各四卡启动同C框解码。CPU配对首先完成original_path MaskAP500重采样：C-A .03594855，名义95% [.03154095,.03674571]；C-B .03767172，[.03453350,.04000989]；B-A −.00172317，[−.00569043,−.00039462]。这是原检测路径，不是完整combined，也不是纯卫星对照。

`scripts/analysis/ue_ubr_native_fixed_box_errors_20261028.py`用同一C bbox/score贪心IoU≥.5一对一GT匹配（不读mask定匹配）。四区域：boundary=dilate14−erode14、interior=erode14、background=~dilate14且不在全图GT并集、other_instance=~dilate14且在GT并集，互斥完备。分母为匹配实例像素（跨实例可重复），未匹配GT/预测单列；给纠正/新增错误/净纠错和均值IoU，不代替AP。

后续正确/zero/donor Mask/BoundaryAP与固定框结果未完成前，不下尾部内容归因结论。后台队列独立落盘，不需要手工再发射；若任一子任务失败保留产物并停新准入，需显式审查。
