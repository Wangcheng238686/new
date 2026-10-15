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

首版模块、训练/推理接入与开发验收已完成，独立代码审核通过。正式单流/双流对照训练尚未启动；最终数据冻结和独立 test 划分仍需在数据生产完成后核对。

## 首轮实验结果

2026-10-15：两臂80轮完成，best-segm双流.584958/单流.583000；correct/zero/shift完整验证和探索性配对bootstrap已完成。[完整分析](UE_AD_RESULTS_20261015.md)：消费路径/对齐依赖明确，尚无可靠净收益证据。

[典型改善与退化案例](UE_CASE_AUDIT_20261015.md)：7例全景/裁剪/几何支持；同栋668000的310/312成对对照优先固定框诊断。
