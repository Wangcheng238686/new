# AGENTS.md — 本工作区执行纪律

## 论文骨架（单一叙事动机，2026-10-26 定案）

**全文只回答一个问题：UAV 信息对卫星侧实例分割有没有帮助。**

骨架四段，一切实施挂载于此：

1. **动机**：卫星单流弱点量化（大建筑边界塌落 42.3%、B-AP_l 0.532、高密度图漏检）+ UAV 互补证据（近距观测、高度不连续、遮挡关系）。
2. **仪器**：统一架构——单/双流逐位同构，UBR ⑤端口为 UAV 信息唯一入口，dual−single = {端口⑤, loss_geometry}；架构决策以一句话设计理由带过，不展开历史考证。
3. **实验**：内容门矩阵（统一单流 / UBR-on 常量 / UBR-on 真实）+ 配对 bootstrap CI。
4. **归因**：增益分解（新增实例 vs 边界精修；尺寸/密度分层）。

## 实施前过骨架（每次动作的检查门）

任何实施动作（实验立项、探针、训练、评估、代码改动、文档撰写）启动前依次自检：

1. **服务性**：该动作是否直接服务于回答上述单一问题？不直接服务的不做。
2. **判据校准**：验收判据是否直接校准于该问题（内容门 real−constant、配对 CI、B-AP/AP75）？"顺便证明某方法不行"的支线目标一律砍掉。
3. **叙事纪律**：负结果不进论文叙事。**常量臂例外**——它是"信息（而非参数/容量）有帮助"的正面证明机制，居主表 C 位。负结果舰队只作内部记录与 rebuttal 弹药；正文设计依据一句话 + 附录指针。
4. **证据连续性**：不得破坏会议→期刊证据链（prompt 模态栈逐位保留、UDPR→UBR 走再封版门、VHR-10 已报指标不复跑、冻结期配对可比性与跨机复现契约不动）。

## 实验实施规范（本工作区硬规程）

权威版本：仓库根 `EXPERIMENT_GOVERNANCE.md`（原则与红线）与 experiment-governance skill（操作清单）；本节为绑定本工作区的浓缩版，冲突时以原则文档为准。

### 立项与预注册
- 新实验先判路由并写进立项记录：**模块叠加型**（冻结基座可验证）→ 探针 → 热启动筛查 → 全量；**基座/训练动力学型** → 冻结诊断探针 → 直接全量（须说明热启动为何无定义）。
- **预注册四要素**先于任何编码/启动，写进设计文档：主指标 + 伴随指标（≥2 维度，禁单指标报喜）｜选择口径（last / 固定E / best 指定其一）｜显著性方法（配对 bootstrap 次数与种子固定；本项目惯用：逐图配对 500 draws seed 20261021，场景配对 10,000 draws seed 20261014）｜各级判定门的具体不等式。
- 立项发临时编号（臂名+日期）；**EXP-ID 于进入全量时**在 `docs/EXP_REGISTRY.md` 发号登记。

### 探针纪律
- 探针结论可作门判据的前提（缺一不得判定）：**sham/阴性对照**（matched 常量控制，support 掩码与分布对齐）、**parity/哈希不变量**（被动 hook 与标准前向逐位一致）、**detector 字段不受干预污染**。
- 遥测三件套：判读边界（能证/不能证/DDP 口径）、标定校验（锚点 vs 部署配置交叉验证）、机制遥测不替代结局验证。
- 首次定版拉独立子代理审实现；审后改动需复审关键分支。

### 全量训练
- 拉卡前置五条（全部满足才启）：① 代码已 commit（message 携带 EXP-ID，在该 commit 打 tag）；② EXP_REGISTRY 发号登记；③ 全臂 harness 通过（遗留臂指纹逐一不变 + 新旧臂 config diff 恰为预期键集）；④ 独立子代理完成实施+接线审计（VERDICT: PASS，报告落盘 `docs/reviews/`）；⑤ wrapper 契约 echo 核对（commit/超参/指纹/协议/标签齐全）。
- 训练中：遥测定期巡检（趋势 + 标定一致性）；异常（NaN、遥测不动、依赖字段漂移）按 PID 树终止，不烧无效算力。
- **test 隔离**：test 全程不读，收线时才做全量 test 与落判；不因测试结果回调结构/轮次/判定标准。
- 收尾：标准推理出数 → 按预注册四要素落判 → 结论 MD（对照表 + 判定原文 + 工件路径）→ 子代理审结果 → 归档。

### 归档与红线
- 归档包（每实验，失败同级）：结论 MD、复核落盘件、遥测日志路径、checkpoint 路径（数据盘登记、不进 git）、EXP-ID 关联；负结果证据先行关账（冻结证伪/门落判后才停）。
- 红线速查：跳级禁止；事后改判定/挑口径禁止（best 尖峰不是证据）；单指标报喜禁止；机制遥测当结论禁止；全量 dirty 启动禁止（无豁免）；未过全臂 harness 拉全量禁止；force push 覆盖已用于实验的 commit 禁止；复核记录只留会话不落盘禁止。
- 本工作区叠加纪律：双流类实验主判据 = **内容门（real−constant 配对 CI）**；r(s)/0.25 分数带与评估协议契约逐位不动；跨机复现的按位/容差契约预先声明。

## Git 管理规约（2026-10-26 定）

### 副本拓扑（事实）
- 本机 worktree：`/home/wangcheng/project/new/.worktrees/dual-stream-dev`，分支 `codex/dual-stream-dev`，**权威开发分支**。
- 远端 lthpc：`/home/wangcheng2021/project/new-uav-dual`，分支 `codex/uav-bev-dual`（另一会话工作区，同 origin）。
- 运行树：`/data2/wangcheng/remote-lthpc/portable_sam2_explicit_coarse`，**非 git 仓库**，仅放运行产物与运行副本。
- 共同 origin：`git@github.com:Wangcheng238686/new.git`，主分支 `single`。会议论文锚点 tag：`paper-icassp2026-terminal`。

### 运行树纪律（最重要）
- 任何新探针/训练/评估脚本的源文件**先落 worktree 并 commit，再同步到运行树**；运行树不是源码的家。
- 全量启动前 wrapper 断言运行树脚本与 git blob SHA 一致（把"全量 dirty 启动禁止"从口头变成哈希契约）。
- 存量未入库脚本（`ue_joint_geo_sat_*.py`、`ue_joint_eval_*.py`、rgb 缓存构建器等）择机补录并补注来源。

### 分支与推送
- 功能分支 `codex/<topic>`；两线合并（dual-stream-dev ↔ uav-bev-dual）必须显式 merge commit + 子代理审冲突，禁止 cherry-pick 拼贴历史。
- **每个工作段结束 push 到 origin 同名分支**（含远端会话：`codex/uav-bev-dual` 尚未推过 origin，属备份风险）。
- 实验全量 run 绑定单一 commit，tag 命名统一 `exp/EXP-XXXX`（存量裸 `EXP-XXXX` tag 不改，新 tag 用统一风格）。

### 提交纪律
- Conventional 风格 `type(scope): 摘要`（docs/feat/fix/artifacts/chore × exp/uav/design/governance）；实验相关提交必须携带 EXP-ID 或臂名；一次提交一个意图（源码/文档/工件登记分开）。
- 不进 git：checkpoint、日志、缓存、数据集——路径登记进结论 MD/EXP_REGISTRY；归档动作时顺检 .gitignore。

### 红线（git 专项）
- force push 已用于实验的 commit/tag 禁止；运行树与 git 不一致时禁止拉全量；未 push 的工作段结束禁止开新的全量 run（防单机丢失）。

## 当前架构定案速查（防上下文丢失）

- 权威设计文档：`docs/UBR-UNIFIED-ARCHITECTURE-DESIGN-20261026.md`（取代 20261024 版，旧文档顶部有指针）。
- UDPR 全线出局（删代码，非运行时开关）；UBR 统一单/双流；prompt 栈 points_box_dense 逐位保留（含冻结 PromptEncoder、dense sg-detach、mining no_grad 三层防火墙）；loss_shape_prior 保留；λ_geo=1.0；r(s)/0.25 分数带不动；离散框接口不开可微（明确不做清单见设计文档 §四）。
- 实验阶梯：门0 零训练 snap 探针 → 级1 固化双参照（EXP-0025 ep71 + E73）→ 级2 加法探针（冻结旧封版+UBR 串联）→ 级3 再封版（UBR(off) 从头 100ep，配对 CI 含 0，mAP+AP75 双盯）→ 级4 最终形态三臂内容门。
- 远端（lthpc，codex/uav-bev-dual）各臂为**参考非对照**（其保留 UDPR+shape_prior）；其训练配方（共同初始化、打分带、损失结构、100ep 收敛性）是级3/级4 的宿主实证；QG 内容门结果决定最终形态"候选+UBR"两层还是"候选+QG+UBR"三层。
- RSPrompter MLP prompt、box-only 简化、可微 prompt、modality dropout（v1）均已否决/推迟，理由与触发条件见设计文档 §四。
