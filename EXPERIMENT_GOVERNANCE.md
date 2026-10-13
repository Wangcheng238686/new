# 实验管控规范（收敛版 v1.0）

> 形态：双层。本文档是**原则层**（人读，判定标准与红线）；执行层是 skill
> `experiment-governance`（AI 接到实验类任务时加载的操作规程）。两层同源，
> 冲突时以本文档为准。skill 以双拷贝维护（`~/.codex/skills/` 与
> `~/.zcode/skills/`），修改须两侧同步（2026-10-13 决议）。
>
> 适用：模型训练、消融、策略迭代类实验。产出背景：NWPU 消融矩阵项目实操
> （2026-09），每条规则都有对应的事故或教训，标注为 ⚠。

## 0. 前提：角色模型

单人研究者 + AI 助手协作，无 PR receiver。传统流程中的"评审"由**独立子代理
复核**承担；复核人不参与被审材料的设计与开发。跨机实验以单一**规范机**的
口径为准，跨机直接比较无效。

## 1. 算力分级准入（分型双路由）

所有新增实验方向按改动类型分流，立项时声明路由并说明理由，禁止跳级：

**A. 模块叠加型**（在既有基座上加模块，冻结基座可验证其价值）：
```
冻结探针（oracle/干预，分钟级） → 热启动筛查（冻结基座单训模块，小时级）
→ 全量训练（正式预算）
```
每级设有预注册判定门；门未过 → 暂停，复盘假设后重新探针，不得进入下一级。

**B. 基座/训练动力学型**（改动对象就是基座训练本身，冻结基座无从验证）⚠：
```
冻结诊断探针 → 直接全量
```
（例：梯度隔离类改动改的就是训练动力学，热启动层对它无定义。）

失败是正式结果：负结果须**证据先行关账**（冻结证伪/判定门落判之后才停），
归档与正结果同级。探针多次不过 → 终止该方向。

## 2. 指标四要素预注册（所有层级强制）⚠

启动前写死，禁止跑完再定标准挑结论：

1. **主指标** + **伴随指标**（禁止单指标报喜——"segm +0.012 藏着 bbox
   −0.022"是一次真实事故）；
2. **选择口径**（last / 平台窗口 / best，预先指定；未指定口径的 best 尖峰
   是选择伪影，不得作为结论依据）；
3. **显著性方法**（image-paired bootstrap CI / 双 seed / 效应主张线，
   预先约定并区分其覆盖的不确定性类型）。

## 3. 遥测三件套 ⚠

遥测方案在编码前预定义，且必须同步交付：

1. **判读边界**：每个字段写明能证明什么、不能证明什么、DDP 聚合口径；
2. **标定校验**：遥测声称度量的决策面/配置必须与部署配置交叉验证
   （计数器锚 logit-0 而部署判定线在 −0.4055 的误读是一次真实事故）；
3. **机制遥测不得替代结局验证**：训练侧字段只作机制证据，结论以完整
   validation + 推理为准。

## 4. 独立子代理复核（节点触发）

三个强制节点，复核记录**全文落盘**（`docs/reviews/`，含 VERDICT 行与
发现清单，结论 MD 引用其路径）：

1. **全量拉起前**：实施+接线审计（臂隔离、指纹回归、单变量键集）；
2. **探针首次定版时**：审实现（对照 cell、sham、不变量）；
3. **结论写回论文口径前**：审结果（数字复现、预注册门执行、无越界表述）。

日常探针迭代与筛查自验（smoke + 不变量）即可，不触发复核。

## 5. 训练启动与实验臂隔离

1. 一切训练经统一 wrapper 启动；契约 echo 自描述：git commit/dirty、全部
   超参、架构指纹、协议、运行标签；
2. 臂 = 配置（env 组合），不是代码分支。臂间隔离由检查保证 ⚠：新臂接入后
   必须跑**全臂 harness**——遗留臂指纹逐一不变 + 新旧臂 resolved config 的
   两两 diff 恰好等于预期键集。"单变量"是检查出来的，不是声明出来的；
3. checkpoint 自描述（内嵌 config snapshot），推理入口按快照重建。

## 6. Git 管控

1. **单分支为体**；`exp/` 分支仅用于协议级大改（改协议/大重构/依赖升级），
   评审后合入；
2. **脏代码分级** ⚠：探针/筛查允许 dirty 启动（wrapper 记录 commit+dirty）；
   **全量 run 一律先 commit，无例外**——论文口径数字的复现性必须钉死在
   具体 commit 上；
3. commit 信息：英文 conventional 格式（`type(scope): subject`），单次提交
   单一主题；进入全量的提交追加 ` | EXP-XXXX`；关键数字可中文或英文
   （2026-10-13 决议，与仓库既有实践对齐）；
4. **EXP-ID**：进入全量时在 `docs/EXP_REGISTRY.md` 登记发号（每全量 run
   一个，全局唯一，`EXP-XXXX`，只发不复用）；commit message 携带，并在该
   commit 上打 tag。臂名继续承担谱系语义，探针/筛查以臂名+日期引用；
5. 禁止 force push 覆盖已用于实验的 commit；
6. 资产：权重/探针工件不进 git，放数据盘目录并在结论 MD 登记路径；配置
   身份 = env 解析 + checkpoint 快照 + 契约 echo 三重保障，不设独立配置目录。

## 7. 终止、早停与归档

- **早停纪律** ⚠：平台判定成立（多窗口无新高 + 遥测走平）后才停；best
  checkpoint 已存盘才杀；按 PID 树杀进程，防自匹配与孤儿双拉；
- 中途发现代码 bug → 直接终止，不继续烧算力；
- **归档包**（每实验，含探针/筛查/全量）：结论 MD（预注册四要素的判定
  原文与落判）、复核记录落盘件、遥测日志路径、工件/checkpoint 路径、
  EXP-ID 关联。

## 8. 阶段决议记录（双流阶段启动，2026-10-13）

1. **命名**：双流融合工作流用新前缀 `DUAL-*`（探针/筛查/预注册/结论，
   与 SURFBEV- 数据线、MHBEV- 旧基座线区分）；EXP-ID 全局编号继续在
   `docs/EXP_REGISTRY.md` 只发不复用；
2. **基线 test 先开**：双流首个全量拉起前，先按冻结协议开封 EXP-0006
   （v2 split 单流锚点）的 test，建立同口径单流基线数字；
3. **控制臂重跑**：双流全量实验的卫星单流控制臂用同一 wrapper/代码路径
   重跑（同 seed/超参/协议），臂间可比性由契约 echo 与全臂 harness 保证，
   不直接沿用历史 run 数字；
4. **立项记录外迁**：冻结立项记录不再进本文档附录，新记录直接落
   `docs/governance_records/<EXP-ID>_立项记录.md`，本文档只留原则层与索引；
5. **commit 规范**：见第 6.3 条；
6. **skill 双拷贝**：见文首形态说明。

## 附：明确删除项及理由

- GPU 共租纪律 → 属硬件运维而非实验规范，留在项目实操约定中；
- PR 评审流程 → 无 receiver，由节点触发的子代理复核替代；
- DVC/对象存储、`config/experiments/` 目录制 → 与臂 env 体系重叠，
  由自描述 checkpoint + 契约 echo 覆盖其职责。

---

## 附录索引（2026-10-13 迁出）

全部立项记录已按实验 ID 迁至 `docs/governance_records/`，除来源行外内容
原文未改。原附录字母存在两次编号事故（R、S 各出现两次）；历史文件
（含 `docs/reviews/` 中 "Appendix X at commit ..." 字样）不改写，按
下表解析：

| 原附录 | 实验 ID | 文件 |
| --- | --- | --- |
| A | MHBEV-ROI-20260914 | `governance_records/MHBEV-ROI-20260914_立项记录.md` |
| B | MHBEV-DIAG-20260915 | `governance_records/MHBEV-DIAG-20260915_立项记录.md` |
| C | MHBEV-DIAG-20260915-B1 | `governance_records/MHBEV-DIAG-20260915-B1_立项记录.md` |
| D | MHBEV-DIAG-20260915-B2 | `governance_records/MHBEV-DIAG-20260915-B2_立项记录.md` |
| E | MHBEV-DIAG-20260915-B3 | `governance_records/MHBEV-DIAG-20260915-B3_立项记录.md` |
| F | MHBEV-DIAG-20260915-C1 | `governance_records/MHBEV-DIAG-20260915-C1_立项记录.md` |
| G | MHBEV-DIAG-20260917-D1 | `governance_records/MHBEV-DIAG-20260917-D1_立项记录.md` |
| H | MHBEV-DIAG-20260918-D2 | `governance_records/MHBEV-DIAG-20260918-D2_立项记录.md` |
| I | MHBEV-DIAG-20260918-D3 | `governance_records/MHBEV-DIAG-20260918-D3_立项记录.md` |
| J | MHBEV-FUS-20260919 | `governance_records/MHBEV-FUS-20260919_立项记录.md` |
| K | MHBEV-FUS-20260919-E5 | `governance_records/MHBEV-FUS-20260919-E5_立项记录.md` |
| L | MHBEV-FUS-20260919-H1 | `governance_records/MHBEV-FUS-20260919-H1_立项记录.md` |
| M | MHBEV-FUS-20260919-P6 | `governance_records/MHBEV-FUS-20260919-P6_立项记录.md` |
| N | MHBEV-FUS-20260920-P7 | `governance_records/MHBEV-FUS-20260920-P7_立项记录.md` |
| O | MHBEV-FUS-20260920-P8 | `governance_records/MHBEV-FUS-20260920-P8_立项记录.md` |
| P | MHBEV-FUS-20260920-P9 | `governance_records/MHBEV-FUS-20260920-P9_立项记录.md` |
| Q | MHBEV-FUS-20260920-P10 | `governance_records/MHBEV-FUS-20260920-P10_立项记录.md` |
| R（第一次） | MHBEV-FUS-20260920-P11 | `governance_records/MHBEV-FUS-20260920-P11_立项记录.md` |
| S（第一次） | MHBEV-FUS-20260921-P12 | `governance_records/MHBEV-FUS-20260921-P12_立项记录.md` |
| R（第二次，编号事故） | MHBEV-FUS-20260919-H2 | `governance_records/MHBEV-FUS-20260919-H2_立项记录.md` |
| S（第二次，编号事故） | MHBEV-FUS-20260919-R1/R2 | `governance_records/MHBEV-FUS-20260919-R1R2_立项记录.md` |
| T | MHBEV-FUS-20260921-T1 | `governance_records/MHBEV-FUS-20260921-T1_立项记录.md` |
| U | MHBEV-FUS-20260921-U1 | `governance_records/MHBEV-FUS-20260921-U1_立项记录.md` |
| V | MHBEV-FUS-20260929-SZAB | `governance_records/MHBEV-FUS-20260929-SZAB_立项记录.md` |
