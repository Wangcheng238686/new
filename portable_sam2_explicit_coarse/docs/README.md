# docs/ 目录索引

入口三件：`codebase_guide.md`（读代码从这里开始）→ `a3sgm_architecture_guide.md`
（机制全景，行号对应清理前 add7482）→ `DEBUG_FIELDS.md`（训练遥测字段字典）。

## 现行文档（docs/ 根）

| 文件 | 内容 |
|---|---|
| `codebase_guide.md` | 当前代码唯一权威地图：架构、阅读路线、接口清单、编码规范 |
| `a3sgm_architecture_guide.md` | a3sgm 机制详解（前向 Step1-11、损失、梯度通路、2P2N 附录） |
| `cleanup_rounds.md` | 2026-10 a3sgm-only 清理的 14 轮变更清单与等价性验证记录 |
| `DEBUG_FIELDS.md` | 训练日志遥测字段字典（DENSE/GATE/SP/TAIL） |
| `EXPERIMENT_GOVERNANCE.md` / `ABLATION_ENGINEERING_PROTOCOL.md` | 实验治理与消融工程协议 |
| `WHU1024_EXPERIMENT_TRACKING.md` / `NWPU_VHR10_EXPERIMENT_TRACKING.md` | 两数据集实验追踪总表（历史记录，论文数字出处） |
| `EXP_REGISTRY.md` | EXP-ID 实验登记册 |
| `machine_sync_protocol.md` | lthpc / machine2 双机同步协议 |
| `nwpu_gap_analysis_20260919.md` | NWPU 失分解剖与抠分清单（v2） |

## 证据文档（evidence/，论文与决策出处）

| 文件 | 出处 |
|---|---|
| `box_prompt_dependence_vs_contribution.md` | matrix300 全矩阵读数（论文消融行与 a3sgm 冠军判定） |
| `p2v2_matrix300_four_arm_machine2_results.md` | A3 家族四臂对照（a3sgm 证据链） |
| `k_selection_preregistration.md` + `k_sensitivity_full_coco.md` + `k_sensitivity_results.md` | 论文 K 敏感性表的预注册与数据 |
| `a0_dense_route_verdict_20260911.md` | dense 通路裁决（PBM 路线采纳依据） |
| `a0_prompt_dependence_dense_vs_box.md` | 提示模态依赖审计 |
| `nwpu_fi_matrix_plan.md` | NWPU full_image 消融矩阵协议决策 |
| `udpr_decoder_tail_design.md` | UDPR/A3 历史实施契约 |
| `whu1024_a3_full_run_protocol.md` | WHU 全量 A3 制式协议（WHU 行出处） |

`archive/`（p2 早期设计）、`reviews/`（EXP 审计）为历史归档，原样保留。
历史臂的完整文档与代码见清理前提交 add7482。

## UAV 双流开发

[dual_stream/README.md](dual_stream/README.md) 收录本工作树的设计决策、工程核定、参数和测量快照。
