# 双流独立实现审核

日期：2026-10-14。三名全新子代理独立只读审核实现，未参与编码；发现由主代理修复后分别复核。该审核与先前设计对抗讨论分开。

## 判决

三方代码复核均为 **PASS**，原 P1/P2 已关闭。全量清单重建、最终重复推理、AMP诊断回归和最终DDP空支持验证均已完成。PASS 的范围是已定首版实现和开发 smoke，不代表 AP 增益。

| 审核者 | 审核范围 | 初次发现 | 复核结论 |
|---|---|---|---|
| implementation_geometry_review | 相机/深度、候选、支持/置信、足迹/投影、空间划分 | P2：floor 四角包络遗漏相机 apex | apex 并集修复；独立检查4096射线×11位置，共45056点均在包络内；PASS |
| implementation_training_review | 参数组/裁剪、冻结、指标/选模、EMA与续训、四臂与污染 | P2：resume未比较选模/早停身份；P1：EMA验证后保存shadow与raw optimizer配对 | 变更身份均拒绝；UE保存前restore raw，独立保存EMA；E1状态分离及E2严格续训证据；PASS |
| implementation_integration_review | 多模态数据、归一、batch顺序、融合/roi_local、strict rebuild与路径迁移 | 未发现发布阻碍；指出非严格加载/缺hash的合同硬化边界 | UE禁止非严格加载、要求hash、记录新manifest身份、自动选择验证权重；PASS |

## 修复与证据

- **视锥包络**：下倾相机的近处高表面可位于floor截面外；现在使用 apex 与floor四角的XY并集。低于floor的有效深度拒绝，缓存key包含floor。修复后的清单必须重建，不能拿旧包络冒充新隔离结果。
- **续训身份**：manifest、几何签名/参数、floor、类别、评测cap、选模/早停、有效batch、seed、裁剪、EMA等核心语义严格核验。epochs可延长；同权重身份的路径可迁移。
- **续训快照**：UE保留当前实际代码/路径/epochs目标，记录父checkpoint和沿用checkpoint scheduler状态的策略。warmup/LR倍率变更拒绝；最终续训产物明确记录epochs2、新输出目录和Adam step2，训练反方复验PASS。
- **EMA**：UE在所有rank保存前restore真实训练参数，模型与Adam状态对应；best/last/周期checkpoint单独保存EMA，记录optimizer update计数与该轮评测使用的权重。推理auto消费记录；旧单流保存顺序保留。
- **输入/重建**：五RGB-D文件完整且相机合同正确；缺文件失败，真实几何无支持零残差。UE要求完整pretrain/manifest hash并strict加载。显式新评测清单的hash写入run manifest。
- **前向确定性**：原CUDA原子汇总在两次预测中造成39/300输出微差（框最大0.0033px，score约1.8e-6，并有少量掩码变化）。改为稳定目标排序、segment求和及唯一目标写入后，两次新进程GPU预测JSON的SHA256完全一致；未声称反向逐位确定。
- **诊断不能改变训练**：收尾新增注意力遥测曾因no-grad autocast缓存共享relative MLP权重而导致checkpoint重算图不一致。诊断调用已关闭autocast/cache；AMP+checkpoint启停诊断的两份同权重模型输出相同、所有梯度key相同且最大梯度差0，relative MLP有非零梯度。最终完整GPU模型和空支持DDP已复验，集成/训练反方再次PASS。

实测报告位于 `/tmp/uav_dual_dev_preflight`：

- `geometry_static.json`：平面、候选唯一/屋顶优先、空支持、粗格不代表遮挡、细像素近表面竞争、apex包络及floor拒绝。
- `arms.json`：四臂实际构建、配置身份、独立参数组恰好覆盖一次、受控环境污染攻击。
- `model_gradients_resources.json`、`model_coarse_route.json`：真实几何/门控的梯度、参数更新、300个真实GT框重复构成的掩码RoI压测。
- `ddp_empty_support_rank0.json`、`ddp_empty_support_rank1.json`：rank1清空支持后仍进入loss，两rank一步更新后逐参数相同。
- `ema_checkpoint.json`、`ema_resume.json`：E1 raw/EMA分离，EMA LoRA B为raw的0.1（decay0.9）；E2恢复Adam到step2及EMA更新关系。
- `inference_roundtrip.json`：默认auto使用EMA并strict加载，两次GPU预测SHA均为 `b9714a3aa9dc36c24994923fc935a8cab371da7465952d2e8c7b92c4a17ec1a4`。
- `attention_diagnostics_regression.json`：AMP+checkpoint下诊断启停输出及全部梯度一致。
- `final_resume_identity.json`：最终snapshot准确记录当前执行配置、epoch目标及原调度器延续策略。
- `final_model_check.json`、`final_ddp_empty_support_rank*.json`：最终分段归约和诊断缓存修复后的完整更新、四条loss路径、300RoI、双rank空支持复验。
- 新空间清单 `/tmp/uav_dual_dev_data/frozen_verified/manifest.json`：全量5×1057深度下界检查，628 train/153 validation，边界或共享建筑排除276点，跨集合global建筑ID重叠0。这是开发train/validation，不冒充独立test。
- 对齐图 `/tmp/uav_dual_dev_data/alignment/alignment.png`（300dpi）与PDF：观测/有效高度/候选支持、实际投影足迹、代表高度、ground-reference修正、置信；6792格CSV剖析、机器布局检查与人工读图通过。ground-vs-height只表示投影修正量，不是准确率。
- `current_runtime_worktree.json`、`current_runtime_reference.json`：未改原仓库与工作树在同环境3个历史案例、6张图的预测hash完全一致。

历史 forward golden 在源代码变更前已经不匹配；未修改 golden，以当前环境原仓库独立参考作回归证据。8个旧检查点严格加载通过。

## 验证边界

模型验收每rank batch1、一次DDP update；未声称覆盖任意batch、多步DDP或正式80轮训练。coarse监督仅在RoI使用融合P3/P4时给UAV梯度：一张样本64个RoI均在P2，另一张63个P2、1个P3后coarse→UAV/CVT梯度非零。检测、final-mask与UDPR均有非零路径；不要求全部LoRA张量每步非零。

资源梯度检查包含多次保留图的独立反向，约9.0GiB峰值不能当成标准每步训练显存。空支持DDP正常一步更新约5.5GiB峰值（见各rank报告）；300 RoI阶段allocated约2.6GiB，reserved包含此前训练缓存，不能混作纯推理峰值。

单层高度、整格足迹、缺少卫星真实遮挡观测仍是已接受近似。正确/zero/shift属于依赖诊断；正式A/D及空间独立test才能评价收益。数据仍在生产，本次清单仅用于开发验收，不替代最终冻结的数据集与实验划分。
