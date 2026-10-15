# 固定共同框与检测阶段定位

日期2026-10-15。延续典型案例310/312/952，共42个GT；固定dual E72 EMA、single E66 EMA。
**不训练、不热启动、不更新权重，也不把oracle诊断作为正式AP。**

## 1. 本轮定位结论

- 所选310/312实例的原mask变化主要随**框条件**变化，同框后两模型非常接近。
- 952同框后双流仍有约.036–.055的单目标IoU优势，但原.337的总差大部分也随框变化。
- 所选目标的RPN及bbox回归后候选中存在几何较好的框；分类score驱动的NMS没有保留几何最佳框。
- 此现象也出现在单流，不是UAV独有问题，未发现NMS实现违反既有规则的证据。
- 三个事后筛选案例不能说明这是全部4496实例或总体AP小收益的主因；优先扩展全验证集候选质量/筛选统计，再决定修改模型。

## 2. 共同框协议与接线校验

对三图的每个GT分别定义3套共同框：GT框、single_pred框、dual_pred框。后两套通过此前top100最大mask-IoU的GT辅助关联确定，允许重复候选，属于oracle配对，不是可部署的proposal策略。

每套框原样传给single、dual correct、dual zero、dual shift。所有36组结果均核对共同框bytes SHA256一致。
绕过RPN/bbox分类/回归，直接调用既有 `roi_head.predict_mask`，保留extra PE、RoI特征、CSPM提示、SAM、UDPR、阈值和bbox paste。
输入label固定为唯一类别，score设1；不比较分类能力或检测召回。GT只提供框与评估目标，不注入GT mask提示。

12次原框回放（3图×4模式）覆盖原先全部历史预测mask，最大XOR像素均0；证明诊断接口能逐像素复现正常mask通路。
模型eval+inference_mode、严格重建EMA，没有optimizer；未使用autocast，与正式独立推理一致。

## 3. 选定建筑的交叉结果

数值为同一GT的二值mask IoU，非AP。

| 实例 | 单流+单流框 | 双流+单流框 | 单流+双流框 | 双流+双流框 | 单流+GT框 | 双流+GT框 |
|---|---:|---:|---:|---:|---:|---:|
| 310 / global668000 | .8014 | .7855 | .6376 | .6378 | .7834 | .7819 |
| 312 / global668000 | .6290 | .6308 | .7335 | .7464 | .7428 | .7532 |
| 952 / global241000 | .5176 | .5606 | .8184 | .8541 | .8240 | .8788 |

### 310

两模型放在双流框下都约.638，单流并不能靠自身mask能力补回被框条件限制的部分。
改成单流框，两者升至约.79–.80；GT框下几乎相同。因此当前选例的大幅退化主要随框条件改变，
而非可直接归为“UAV融合使SAM掩码能力大幅下降”。同框小差仍存在，不能说mask完全等价。

### 312

给单流换双流框，mask从.629升至.734，接近双流.746；两模型在单流框下都约.63。
与310同栋楼，改善/退化都主要跟框走。GT框不保证任一模型最好：框变动还会改变RoI特征和挖掘提示。

### 952

单流仅换双流框，.518→.818；双流同框约.854。存在小幅同框能力差，但原先巨大的single→dual差不能全算UAV带来的mask特征改善。
GT框下dual .879比single .824高约.055，是此例支持继续检查mask互补性的线索，不是全验证集收益结论。

**框条件包括RoI特征提取、coarse/CSPM提示与最终bbox paste。当前控制不能将差异进一步拆成纯paste效果。**

[310共同框图](/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis/fixed_box_probe/common_box_310.png) · [312共同框图](/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis/fixed_box_probe/common_box_312.png) · [952共同框图](/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis/fixed_box_probe/common_box_952.png)

## 4. 所有42个GT的同框结果

按实例等权平均mask IoU，不按3张图等权，也不作AP或泛化估计。

| 共同框来源 | single | dual correct | dual zero | dual shift |
|---|---:|---:|---:|---:|
| GT | .8031 | .8136 | .7081 | .7475 |
| single_pred | .7379 | .7500 | .6737 | .7045 |
| dual_pred | .7451 | .7544 | .6779 | .7074 |

同框后dual相对独立single仍只有约.009–.012的平均优势；移除/错位UAV的损失明显更大，说明mask路径依赖仍存在，且不能仅由最终检测框变化解释。
这与先前“强依赖但净收益小”一致；不能据42个挑选实例宣称稳定改善或统计显著。

## 5. 进一步定位检测框：RPN→回归候选→最终筛选

使用同一模型、输入与原生产配置。RPN用原FPN，bbox head正确注入extra PE；`rcnn_test_cfg=None`提取未NMS的解码框，另调用原test_cfg得到最终框。
每图/每模式最终框与score均与已有完整推理产物**逐值完全一致**。

下表为与选中GT几何IoU最佳的候选；各列可能对应不同候选，**不能串成同一个proposal的回归轨迹**。
全部RPN每图1000，最终框16–35等，未达到模型300输出上限。

| 模型/实例 | RPN最佳IoU | score阈值后、NMS前最佳IoU | 该候选分类score | 最终最佳框IoU |
|---|---:|---:|---:|---:|
| single 310 | .8851 | .8745 | .3212 | .7746 |
| dual 310 | .9080 | .8787 | .9266 | .6242 |
| single 312 | .8099 | .8821 | .2361 | .6562 |
| dual 312 | .8735 | .8925 | .9991 | .7357 |
| single 952 | .6471 | .7709 | .8054 | .5299 |
| dual 952 | .6860 | .8648 | .4465 | .8111 |

所选六个score>0.05的“阈值后最佳几何框”均未出现在最终框集合。有分数更高、已保留的框与其IoU超过NMS阈值.5，例如：

- **dual310**：几何最佳框GT IoU .8787 / score .9266；最终保留框GT IoU .6242 / score .999982，与前者重叠.6090。
- **single312**：候选GT IoU .8821 / score .2361；保留框GT IoU .6562 / score .999997，彼此重叠.7248。
- **single952**：候选GT IoU .7709 / score .8054；保留框GT IoU .5299 / score .996802，彼此重叠.6550。

这支持定位为**分类score与定位质量不一致时，现有score-driven NMS丢掉较好几何候选**。
不是“没有好proposal”；也不是“NMS代码算错”。表中best IoU由GT定义，模型推理时并不知道它。
最佳bbox IoU并不保证最佳mask或AP；尚未检验改选这些候选后的完整检测/分割收益。
多个更高分保留框可能与同一候选重叠，记录保留所有满足条件者，不把任意一个都称为唯一直接抑制者。
若进一步要证明具体贪心次序，应重放NMS保留/抑制轨迹；本轮没有替换NMS实现。

[NMS前后候选图](/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis/fixed_box_probe/nms_candidates.png)

## 6. 对融合设计的当前含义

- 不应仅据SAM gate≈1就先改gate：选中310的主要大差已能通过框条件复现，同框mask没有同等幅度退化。
- 辅助mask路径仍有价值线索（952、42GT同框均值），但尚不能认定新增RGB/深度/某一入口的独立贡献。
- 两模型都存在框质量排序损失。UAV可能影响候选/分数，但当前结果未拆开RPN、bbox回归、分类训练或NMS各自的训练根因。
- 先前直接考虑FPN-only/SAM-only重训仍可作为后续方案；当前更经济的下一步是固定权重做**全153图候选质量/分类分数/NMS损失审计**，确认问题规模，再决定是否修改排序或融合入口。
- 不只调NMS阈值让这3例变好。放松NMS可能增加重复框/FP并降低AP，需要完整验证集、冻结指标和阴性案例共同检验。

## 7. 未执行与边界

尚未对完整验证集做共同框oracle或RPN阶段分解；尚未做共享proposal上的bbox head交叉，以隔离训练后的回归与分类能力；未调整阈值或重训。
本轮只针对三张事后极端样本/42GT诊断，不能外推总体收益或“全局瓶颈已确诊”。

## 8. 工件与复现

完整目录：/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis/fixed_box_probe/

- single.json/dual.json：36组固定共同框、IoU、输出RLE、框hash与12次原框逐像素回放。
- summary.json/fixed_boxes.csv：数值；single_detector.json/dual_detector.json/detector_selected.json：阶段候选与重叠保留框。
- common_box_ID.png/.pdf、nms_candidates.png/.pdf：300dpi工程诊断图；layout QA与人工查看通过。
- ue_fixed_box_probe.py、ue_detector_stage_probe.py、ue_fixed_box_summary.py、ue_fixed_box_figures.py、ue_nms_figures.py：复现脚本。
- 图表数字均为GT关联mask/box IoU，未伪装为正式AP；未更改原推理产物或训练检查点。

独立复核全文：[共同框与检测阶段审核](../reviews/UE_FIXED_BOX_20261015.md)。两部分均PASS。
