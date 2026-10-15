# UAV 双流开发入口

开发工作树：`/home/wangcheng2021/project/new-uav-dual`。
分支：`codex/uav-bev-dual`。
创建基点：`95026ae677dd0a8d77a3ee4c5fb68abeb4ea01c1`，来自 `/home/wangcheng2021/project/new` 的当前单流代码。

## 已确认的设计

五视角 UAV RGB-D → 独立深度约束 CVT → 单层世界坐标 BEV → 高度/置信辅助对齐 → 卫星 P3/P4 与 SAM2 stride16 图像特征两个融合入口。

卫星与 UAV 使用独立 LoRA，五个 UAV 视角共享 UAV 编码器。正式双流直接从通用 SAM2 预训练权重联合训练，不依赖训练好的 UE 单流检查点。深度在训练与推理中都是必需输入，只有卫星实例标签作为任务监督。卫星参与 BEV 构建的路线保留为后续尝试。

## 开发依据

- [实现结构与训练/推理入口](IMPLEMENTATION_AND_RUNBOOK.md)
- [独立实现审核与修复证据](INDEPENDENT_IMPLEMENTATION_REVIEW.md)

- [三方对抗复核与必须闭环的契约](ADVERSARIAL_DESIGN_REVIEW_20261014.md)

- [设计与梯度路径](NEW_PROMPTMINER_UAV_DUAL_STREAM_DESIGN_20261014.md)
- [工程核定与实施顺序](UAV_DUAL_ENGINEERING_SPEC_20261014.md)
- [工程参数 JSON](probes/uav_dual_engineering_20261014/engineering_config.json)
- [CVT 路线调研](UAV_CVT_BEV_ROUTE_RESEARCH_20261009.md)
- [深度格式与相机契约](UE_UAV_DEPTH_RENDERING_AND_MODEL_LABEL_GUIDE.md)

首版默认：512² 图像、512m×512m/128²/256ch BEV、stride8+stride16 UAV 特征、8m 邻域/每视角最多8候选、2层8-head注意力。具体几何聚合与融合初值的限制以工程核定文档为准。

## 证据边界

`probes/uav_dual_engineering_20261014/` 是原工程核定快照，仅支持早期尺寸、对齐和接线判断。正式实现的验收证据在 `probes/implementation_20261014/`：真实样本前向、反向、optimizer update、空支持 DDP、EMA 续训、严格重建和重复推理均已通过；三方独立审核通过。上述验收未验证 AP 增益。

原审计脚本与运行上下文保存在：
`/data/wangcheng/ue/CitySample/CitySample_5.5/data/Content/Python/audit_uav_dual_engineering.py`
及该目录的 `docs/probes/uav_dual_engineering_20261014/`。其中资源脚本引用原单流路径，历史报告不能等价于本工作树正式模型的运行结果。

## 实施顺序

1. UE RGB-D dataset、collate、相机/局部坐标 payload 与 512 单流配置。
2. 多尺度 UAV 编码、局部 CVT 与有效性/高度辅助信息。
3. BEV → 完整卫星网格投影及两个置信门控入口。
4. 自描述双流 checkpoint、训练/验证/推理数据链路。
5. 真实样本的完整资源检查及单流/双流训练对照。

首版模块、训练/推理接入与开发验收已完成，独立代码审核通过。当前开发子集的首轮单流/双流对照训练已完成；完整生产集最终冻结和独立 test 划分仍需在数据生产完成后核对。

## 首轮实验结果

2026-10-15：两臂80轮完成，best-segm双流.584958/单流.583000；correct/zero/shift完整验证和探索性配对bootstrap已完成。[完整分析](UE_AD_RESULTS_20261015.md)：消费路径/对齐依赖明确，尚无可靠净收益证据。

[典型改善与退化案例](UE_CASE_AUDIT_20261015.md)：7例全景/裁剪/几何支持；同栋668000的310/312成对对照优先固定框诊断。

[固定共同框与检测阶段诊断](UE_FIXED_BOX_DIAGNOSIS_20261015.md)：310/312/952的42GT条件性诊断，所选实例差异主要随框条件变化；NMS前存在较好几何候选但排序未保留，尚未推广全验证集。

[当前开发子集完整验证审计](UE_DEV_SUBSET_AUDIT_20261015.md)：153图/4496次出现/237栋建筑；NMS问题两臂共有，GT共同框下双流微降，修正此前42GT的小样本印象。下一步优先固定几何、只干预UAV RGB；该实验尚未执行。

[仅替换UAV RGB的完整验证](UE_RGB_CONTENT_20261015.md)：几何固定，两套供体使segm AP下降约3.4–3.5pp、bbox近似不变；共同框mask同步下降。证明配对RGB敏感性，尚未证明超过单流的有效互补信息。

[FPN/SAM入口冻结定位](UE_FUSION_ROUTE_20261015.md)：两套供体的四组合完整验证，RGB替换mask敏感性主要沿SAM embedding；尚不证明FPN无用或SAM有bug。后续优先冻结残差强度/对齐诊断，不自动训练。

[SAM残差强度冻结探针](UE_SAM_STRENGTH_20261015.md)：正确RGB-D下原强度优于0/.5两种减弱，半强度segm AP下降2.07pp，不支持统一注入过强；不改默认配置，后续优先空间选择/几何对应诊断。

[空间对齐与局部质量冻结诊断](UE_SPATIAL_LOCAL_20261015.md)：±4px四方向及对称平滑未达改善门，内部相机代数一致但非外部标定证明；低支持组不能直接等同注入污染。优先评估P2候选，不自动全层融合或训练。

[P2冻结迁移探针](UE_P2_TRANSFER_20261015.md)：P2为当前最高分辨率stride4；复用P3融合器的五组固定框153图测试基本持平，主/内容/细网格支持门未过，不扩全层或训练。负结果不否定适配后P2融合价值。

[P2→提示→SAM冻结追踪](UE_P2_PROMPT_TRACE_20261015.md)：153图四组精确回放，变化可达decoder但粗掩码/最终IoU未稳定改善；197/881小目标在32² dense框支持为空。P2是提示介导入口，未直接给SAM高分辨率视觉特征注入UAV；无训练。

[GT形状提示冻结诊断](UE_GT_PROMPT_ORACLE_20261015.md)：153图固定GT框，原预测→GT64联合点/dense提示使最终IoU .841769→.860831，条件改善门通过。提示通路仍有可利用空间，尚不证明UAV可达或detach重训收益；无训练/AP。

[GT点/dense四组合冻结定位](UE_GT_PROMPT_FACTORIAL_20261015.md)：153图，点only IoU+.004540（正信号但未达.005幅度门），denseonly+.015619通过门；提示来源/对角回放及197例dense空支持阴性通过。优先检查dense构造质量，不推断UAV可达或解除detach收益。

[三路并行排查综合结论](UE_PARALLEL_DIAGNOSIS_20261015.md)：等±4幅度GT形状仍有+.016068最终IoU空间；现有UAV读出仅微弱正信号，两工程幅度门FAIL；dense反传源码可达但P2无显式UAV输入。暂缓性能训练，优先核表征/局部形状信息保留。分项：[dense形状/幅度](UE_DENSE_SHAPE_CONFIDENCE_20261015.md)、[UAV互补](UE_UAV_COMPLEMENTARITY_20261015.md)、[梯度工程](UE_GRADIENT_DESIGN_AUDIT_20261015.md)。

[BEV形状修正偏弱的三路定位](UE_BEV_WEAKNESS_DIAGNOSIS_20261015.md)：无过强ridge/全坍塌证据，信号更集中高错误/GTdense空间大目标，小目标有支持后仍弱；固定六图几何cache精确重建、所选bin均值位移小，未定位明确对齐bug。下一候选是有效投影query实际attention来源，尚未执行。
