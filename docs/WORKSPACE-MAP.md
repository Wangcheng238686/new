# 工作区总地图（WORKSPACE-MAP，2026-10-28）

本文件是本 worktree（`/home/wangcheng/project/new/.worktrees/dual-stream-dev`，分支 `codex/dual-stream-dev`）的主索引：什么在用、什么已归档、数据盘工件在哪、怎么启动实验。
**本地图只做索引，不做搬动**——所有路径一律保持原名（在跑任务按路径导入，改名将破坏断点续训）。
权威规程见仓库根 `AGENTS.md` 与 `EXPERIMENT_GOVERNANCE.md`；实验台账见 `docs/EXP_REGISTRY.md`（本地图以其为骨架）。

---

## 一、正在使用（active）

### 1.1 三个在跑/排队的臂（级3 再封版 + 级4 最终形态）

| 臂 | 状态（2026-10-28 01:58 实测） | 证据 |
| --- | --- | --- |
| **EXP-0028-R2**（UBR-Reseal-v1.1 单流，seed44） | **在跑**：owner PID 2972162，ep 21/100（约 12.5min/ep），输出根 `ue_ubr_reseal_r2_20261027/` | `ps aux` + launch log；cwd = 本 worktree 的 `portable_sam2_explicit_coarse/` |
| **EXP-0029**（UBR 双流·真实臂②，seed44） | **在跑**：owner PID 3006086，ep 7/100，输出根 `ue_ubr_dual_real_20261027/` | 同上；与 R2 共租四卡（每卡 2 训练进程上限） |
| **EXP-0030**（UBR 双流+RGB 臂） | **冻结待发射**（PI 指令，OOM 风险，等口令）；实现 `bb23069`、审计 PASS、锚点缓存已绑定 | `docs/UBR-DUAL-RGB-LAUNCH-SPEC-20261028.md`；发射时仍需打 tag `exp/EXP-0030`（见 §四） |
| （预留）EXP-0029 臂③常量 / EXP-0030 constant | 排队/预留：臂③按 PI 指令后续手动发射；RGB constant 仅在 RGB_WINS 判决后 | wrapper 头注释 + 级4 预注册 |

两个在跑进程的导入链（**这些路径一个都不能动**）：`scripts/probes/ue_ubr_reseal_20261027.py`、`ue_ubr_reseal_dual_20261027.py` → `rsprompter/uav/{ubr,ubr_train_integration,ubr_dual_integration}.py` → `train/train_rsprompter_fusion.py` + `configs/ablations/ue_ubr_{single,dual}_512.py`，全部从本 worktree 的 `portable_sam2_explicit_coarse/` 解析（`/proc/<pid>/cwd` 实证）。

### 1.2 活跃代码（portable_sam2_explicit_coarse/ 内）

- **UBR 生产模块**（`rsprompter/uav/`）：`ubr.py`（统一边界精修器，v1.1 = 分桶/半精度/分片 val 执行层 + 32ch×6 统一容量 67,457 参数）、`ubr_train_integration.py`（v1.1 单流接线）、`ubr_dual_integration.py`（双流接线，extends 不 fork）、`ubr_dual_rgb_integration.py` + `rgb_p0_encoder.py` + `rgb_geometry_cache.py`（EXP-0030 +RGB 路径）。
- **宿主训练器**：`train/train_rsprompter_fusion.py`（两个在跑臂的实际训练循环）。
- **配置**：`configs/ablations/ue_ubr_single_512.py`（EXP-0028-R2）、`ue_ubr_dual_512.py`（EXP-0029）、`ue_ubr_dual_rgb_512.py`（EXP-0030）。
- **证据层脚本**（`scripts/probes/`）：`ue_ubr_reseal_20261027.py`、`ue_ubr_reseal_dual_20261027.py`（在跑 owner）、`ue_ubr_dual_rgb_real_20261028.py`（EXP-0030 runner）、门禁三件套 `ue_ubr_v11_gates_20261027.py` / `ue_ubr_dual_gates_20261027.py` / `ue_ubr_dual_rgb_gates_20261028.py`、全臂 harness `ue_ubr_reseal_harness_20261027.py`、探针阶梯 `ue_ubr_snap_20261026.py` / `ue_ubr_addprobe_20261026.py` / `ue_ubr_addprobe_cache_20261026.py`。
- **发射 wrapper**（`scripts/ue/`）：`run_ubr_reseal_ddp4.sh`、`run_ubr_dual_real.sh` → `run_ubr_dual_ddp4.sh`、`run_ubr_dual_rgb_real.sh`（唯一合法发射路径，见 §四）。
- 周边：`ue/`（UE 外围适配器）、`data/`、`utils/`、`inference/`、`tools/`。

### 1.3 合同谱系（wrapper contract lineage）

遗留 echo 版（`run_exp0007_ddp4.sh` … `run_exp0027_ddp4.sh`、`run_warm_screen.sh`：exp_id+commit 回显）→ **RESEAL 合同 v1/v2**（`run_ubr_reseal_ddp4.sh`，经 `cfe47fa`/`ae8f1fb`/`97721da` 演化：tag 前缀匹配、runner CLI 对齐、合同落 sideline 文件）→ **v3**（`run_ubr_dual_ddp4.sh`：+arm、几何缓存/常量 SHA、共租每卡 2 进程上限）→ **v4**（`run_ubr_dual_rgb_real.sh`：+RGB 内容、3060 锚点缓存计数、UBR_RGB_DESCRIPTOR_MEAN（constant 臂）、排队守卫"双跑不加第三"）。

### 1.4 权威文档（active 权威层，docs/ 内）

| 文档 | 角色 |
| --- | --- |
| `AGENTS.md`（仓库根） | 执行纪律总纲：论文骨架、实验规程、git 规约、代码规范、沟通规范 |
| `EXPERIMENT_GOVERNANCE.md`（仓库根） | 实验治理原则权威版 |
| `docs/EXP_REGISTRY.md` | **实验台账（权威）**：EXP-0001…EXP-0030 全部全量登记 |
| `docs/UBR-UNIFIED-ARCHITECTURE-DESIGN-20261026.md` | **权威架构设计**（取代 20261024 版：UDPR 全线出局、单/双流统一、UBR ⑤端口唯一 UAV 入口、实验阶梯） |
| `docs/UBR-V1.1-PERF-AND-RELAUNCH-20261027.md` | UBR v1.1 显存/速度优化与 R2 重启方案（EXP-0028-R2 的出生证） |
| `docs/UBR-RESEAL-20261027_preregistration.md` | **现行预注册**：级3 再封版四要素 + **级4 双流臂附录**（2026-10-27 深夜，共租纪律与内容门判据） |
| `docs/UBR-DUAL-RGB-LAUNCH-SPEC-20261028.md` | EXP-0030 +RGB 臂启动规格 |
| `docs/UBR-PROBE-PHASE-20261026_preregistration.md` + `_progress.md` | 探针阶梯（门0/级1/级2）预注册与滚动记录（级2 PASS 数值在此） |
| `docs/reviews/UBR-RESEAL-20261027_audit.md`、`UBR-RESEAL-R2_audit.md`、`UBR-DUAL-20261027_audit.md`、`UBR-DUAL-RGB-20261028_audit.md` | 四份在册审计（全 PASS），wrapper fail-closed 依赖 |
| `docs/ops/OPS-AGENT.md` + `docs/ops/session-log.md` | OPS 子代理 charter 与任务日志（远端巡检基线 OPS-001 在内） |
| `docs/SINGLESTREAM-0024-0027-METRICS-20261022.md` | 现纪元单流四臂指标台账（EXP-0025 = 级3 对照锚的数字来源） |

### 1.5 活跃输入（数据/缓存，两个在跑臂直接读）

- 数据集：`/data1/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021`（**test 全程密封**）
- 几何缓存：`/data2/wangcheng/remote-lthpc/cache/ue_v1_native_geometry`（native512 3060 景）
- 几何常量：`/data2/wangcheng/remote-lthpc/checkpoint/ue_geometry_mechanism_20261022/common_constants.json`
- SAM2 权重：`/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt`（`SAM2_CKPT`）
- Python：`/data2/wangcheng/envs/cvt2/bin/python`
- EXP-0030 追加：FP64 锚点缓存 `/data2/wangcheng/remote-lthpc/cache/ue_v1_s2db_geometry_v2`（3060 npz / 8.4GB）

---

## 二、历史结论（archived verdicts，一行一条 + 全记录所在）

> 台账本体永远是 `docs/EXP_REGISTRY.md`；下列为"已经落判、不再驱动当前动作"的结论谱系。负结果舰队按骨架纪律只作内部记录与 rebuttal 弹药。

**UBR 探针阶梯（2026-10-26/27，当前阶梯的地基）**
- 门0 SNAP v1/v2 第三结局：真实高度梯度带信息（real−sham **+5.66** CI[+5.39,+5.92]）但逐像素位移机制破坏性（real−nosnap −15.19）→ 门0 FAIL 却放行级2（学习型空间连贯消费者必要性被量化）。全记录：`docs/UBR-PROBE-PHASE-20261026_progress.md`。**接棒者**：级2。
- 级1 固化双参照：EXP-0025 v1best(ep71, SHA 2fcae2ff, val b7 AP_l 0.5307) + EXP-0017 E73(dd9106f4, 0.5236)，弱点签名跨封版一致。全记录：同上 progress + `frozen_refs/`（见 §三）。**接棒者**：级2/级3 对照系。
- 级2 加法探针 **PASS**（val：real b7 AP_l 68.83 vs orig 53.07，real−constant **+4.75 CI[+4.54,+4.98]**；test：real−constant AP_l **+4.70 CI[+3.87,+5.57]**，小目标 AP_s +3.85）。全记录：progress 文档级2 节 + `ue_ubr_probe_20261026/`。**接棒者**：级3 EXP-0028-R2（在跑）。

**接口时代（2026-10-24/25，几何候选方向确立）**
- 跨机复现成功：远端 3090 "几何实例候选"方向在本机两层复现（净增益 vs 强SAT前缀 **+0.4672** [+0.297,+0.616]；内容差 vs 常量 +0.2468）。全记录：`docs/GEO-DIRECTION-REPRO-20261024.md`。**接棒者**：进入 20261026 统一架构设计 §一证据基座。
- RGB 增量判负（当时口径）：RGB 内容门 +0.049 CI 含 0 FAIL，几何-only 反而显著略优 → 该时代 RGB 出局。全记录：`docs/GEO-RGB-INCREMENT-20261024_conclusion.md`。**接棒者**：RGB 仅以远端唯一 test 阳性形态（+0.89）于 EXP-0030 回归——非复活旧结论。
- 热启动联合首跑：+0.1355 [+0.007,+0.353]（Boundary1% +0.56），增量全部来自新增实例正确边界。全记录：`docs/JOINT-TRAINING-FIRST-RUN-20261024.md`。**接棒者**：同上进设计文档证据基座。
- 从头联合全量预算（远端宿主实证）：`docs/JOINT-FULLBUDGET-20261025_launch.md`；远端 compact_joint 四臂与 candidate_detail 矩阵进展见 `docs/ops/session-log.md` OPS-001（satellite 已收 segm .8607；rgbd−const 双臂 ep71 差 ±0.002-0.004 未分离，待 100ep+CI）。远端各臂**参考非对照**（AGENTS.md 定案）。**接棒者**：其 RGB 阳性形态 = EXP-0030 依据。

**旧双流舰队（2026-10-14~17，旧数据时代）**
- 13 变体舰队（26 训练 = 13 变体 × real/constant）：**全部 13 个内容差为负**——冻结基座端到端消费全军覆没（十四次消费失败）。全记录：`docs/DUAL-STREAM-ARCHITECTURE-DESIGN-20261024.md` §二。
- EXP-0010–0017 谱系关账：P3/P2/P2P3/LoRA/embedding/延迟解冻逐臂 DUAL_HARM 或 NULL（EXP-0017 后"旧数据时代双流谱系全部关账"）。全记录：`docs/EXP_REGISTRY.md` 各行 + `docs/DUAL-AB-LADDER-20261016_conclusion.md`。**接棒者**：WARM 时代筛查链。
- 量纲缺陷根因：BEV 语义通道贡献 0.04% → EXP-0014 定标修复验证。全记录：`docs/DUAL-DELTA-DECOMP-20261016_conclusion.md`。**接棒者**：warm 配方（gate_init/wd 豁免）。

**WARM 时代（2026-10-17~24）**
- WARM 筛查链（BEV-READOUT→R1→R1G→R1B）与 13+ 筛查变体（R1E/R1R/R1F/R1Q/R1FB…）：全记录 `docs/DUAL-WARMSTART-SCREEN-20261017_conclusion.md`、`docs/DUAL-AXIS-SCREEN-20261018_preregistration.md`（含决策树与终局条款）。
- EXP-0018–0023 全量六臂：中间态（bbox 合并池 CI>0 谱系首次，+0.06~+0.10 未近 GAIN 门；R1C≈R1G 通路塌缩为静态偏置）。全记录：`docs/EXP-0018-0019_conclusion.md`、`docs/EXP-0020-0021_conclusion.md`、注册表 0022/0023 行。**接棒者**：TRAIN-ONOFF 判决性检查。
- TRAIN-ONOFF-20261023：旧基座 on−off≈0（train 侧同样学不到）→ 消费失败定性，转向几何候选方向。全记录：`docs/TRAIN-ONOFF-20261023_conclusion.md`。
- WARM-RR 根因闭环：误差质量 79% 是漏检、漏小实例 96.7% 在 UAV 几何覆盖内；v1 零训练救援 AP50 +0.72。全记录：`docs/WARM-RR-20261024_rootcause_and_direction.md` + `docs/WARM-BA-PROBES-20261023_conclusion.md`。⚠ `docs/WARM-RR-V2_preregistration.md` 已写、正式臂未见结论文件（见"存疑清单"）。

**更早纪元（2026-09 ~ 10-21）**
- MHBEV 时代（12 份 docs + `docs/governance_records/` 24 份立项）：B0–D3/E5/SZAB 诊断与融合探针舰队，旧 MHBEV 基线。**接棒者**：SURFBEV 迁移。
- SURFBEV 时代（40 份 docs）：数据/切分/掩码 QA、B0 与 P1 训练（EXP-0001–0005，v1 切含跨集重叠不作独立泛化证据）、EXP-0005 于 ep58 定向终止。全记录：注册表 + `docs/CLEANUP-20261013.md`。
- 数据完整性事故（2026-10-14）：EXP-0007 训练窗内 49 train/15 val 场景 GT 被修复管线改写 → 锚点作废、冻结快照重训替代。全记录：`docs/DATA-INTEGRITY-20261014_incident.md`。
- 新纪元数据定型：CITYSAMPLE 切分审计与 v1→v3 漂移治理（v1 val−test 漂 −16 → v3 ≤5 点条款达标）。全记录：`docs/CITYSAMPLE-SPLIT-AUDIT-20261021.md`、`docs/VALTEST-DRIFT-20261021.md`。EXP-0024–0027 单流四臂：增强方向三口径一致（+0.8~+1.2），EXP-0025 成为级3 对照锚。
- 远端判决文档 `UE_UAV_EFFECTIVE_DIRECTION_VERDICT_20261024.md`：本 worktree 内**只有引用**（3 处），原件在远端会话线（codex/uav-bev-dual）——见"存疑清单"。

---

## 三、数据盘工件地图（**只登记，一律不动**）

### 3.1 `/data2/wangcheng/remote-lthpc/checkpoint/`

| 目录 | 归属 | 性质 |
| --- | --- | --- |
| `ue_ubr_reseal_r2_20261027/` | **EXP-0028-R2 在跑输出根** | 活跃（勿写入/勿移）；旁挂 `.launch_contract.txt` + `_launch.log` |
| `ue_ubr_dual_real_20261027/` | **EXP-0029 在跑输出根** | 活跃；同上旁挂件 |
| `ue_ubr_dual_real_20261027_smoke_runner/` | EXP-0029 冒烟 | 活跃关联（发射前冒烟） |
| `ue_ubr_dual_gates_20261027/`（d1–d3）、`ue_ubr_v11_gates_20261027/`（G1–G4 + val 对齐）、`ue_ubr_dual_rgb_gates_20261028/`（G1–G4） | v1.1 / 级4 / RGB 门禁工件 | 活跃证据（审计引用） |
| `ue_ubr_reseal_20261027/`（+ 4 个 smoke 目录、harness_report.json、launch/launch2.log） | EXP-0028 v1 **死臂**（ep1–13） | 归档对照（registry：作废臂留作对照，never written again） |
| `ue_ubr_probe_20261026/` | 探针阶梯根：`frozen_refs/`（EXP-0025 v1best ep71 + EXP-0017 E73 + SHA256SUMS + manifest）、`cache/{train,val,test,smoke}*`（级2 加法探针缓存）、`snap*/`、`arm_{real,constant}/`、`parity/`、`analysis/test_shards + test_report.json`（test 分片） | 级1/级2 证据 + 级3 对照参照（**冻结参照，活跃资产**） |
| `ue_minimal_single_20261021_EXP-0017/` | EXP-0017（E73 宿主 = frozen ref 来源） | 死臂归档 + 参照来源 |
| `ue_geometry_mechanism_20261022/` | 几何机制根：`common_constants.json`（**EXP-0029/0030 活跃输入**）、`common_init.json/pth`、FINAL_REPORT、gate 子目录 | 混合：常量/共同初始化活跃，其余历史 |
| `ue_joint_scratch100_20261025/`、`ue_sat_scratch100_20261025/` | 远端从头联合/单星 宿主实证（100ep） | 历史结论工件（参考非对照） |
| `ue_joint_scratch100_BROKEN_fullparams`、`ue_sat_scratch100_BROKEN_fullparams` | 早期全参 BROKEN 跑 | 死档 |
| `ue_joint_scratch_20261024/`、`ue_sat_scratch_baseline_20261024/`、`ue_joint_geo_sat_20261024/`、`ue_joint_eval_20261024/` | 接口时代首跑与评估 | 历史 |
| `ue_geometry_{proposal,conditional,rgb}*_20261024/`（7 个）、`geo_rgb_cis/`、`geo_repro_cis/`、`ue_rpn_geometry_20261023/` | 接口时代探针舰队 | 历史 |

`/data2/wangcheng/remote-lthpc/cache/`：`ue_v1_native_geometry`（**活跃**，native512 几何缓存 3060 景）、`ue_v1_s2db_geometry_v2`（**活跃**，FP64 锚点 3060 npz，EXP-0030 绑定）、`ue_v1_rgb_geometry`（RGB 时代缓存）、`ue_s2db_anchor_manifest_v2.jsonl`。

### 3.2 `/data2/wangcheng/checkpoint/`

- `citysample_v1/citysample_v1_512single_defaug_ddp4_s44_EXP-0025/` —— **旧封版（级3 对照锚）**：v1best ep71（frozen-ref 来源）+ last（收线主对照；registry 备注曾待补录 ep98–100，收线前需内容完整性核验）。
- `citysample_v1/citysample_v1_512single_noaug_ddp4_s44_EXP-0024/` + `launcher_logs/` —— EXP-0024 对照臂归档。
- `surfbev_b0/` —— 旧纪元 B0 checkpoint（EXP-0001/2/5/6…）与 `probes/` 探针工件海：**`CITYSAMPLE-BOUNDARY-AP-20261021/`**（边界 AP 工件，42.3% 弱点量化与 §一证据来源）、TRAIN-ONOFF-20261023、WARM-SCREEN/EVAL、DUAL-AB/DELTA/GATESWEEP/REDUNDANCY、PIXEL-CLS、DISTILL-P0、EVI-DECODER、ZMEAN-CONT、AGG-DIAG-20261025 等。全历史。

### 3.3 数据集与其他

- `/data1/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021` —— **现行活跃数据集**（test 密封）；`..._v3_20261021` + `CitySample_SatUAV_Split_Spatial_v3_20261021` —— v3 纪元（EXP-0026/0027）。
- `/data2/wangcheng/dataset_frozen/ue_raw_hero_uav_pairset_20261014_frozen`（+ manifest）—— 冻结快照纪元（EXP-0009–0023）。
- `/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt` —— SAM2 基座权重（`SAM2_CKPT`）。
- 仓库内只读旧物：`legacy_baseline/`（封存实现 + SOURCE_SHA256SUMS）、`main*.tex/pdf` 与 `采样示例图.pdf`（论文旧稿）、`models/depth_anything_v2`、`artifacts/launcher_logs`、`portable_sam2_explicit_coarse/{outputs,work_dirs,logs}`、`docs/whu_surfacebev_migration/`（迁移验收记录）、`docs/research/`（AGG 综述/诊断，20261025）。

### 存疑清单（flagged，未擅自定论）

1. `docs/WARM-RR-V2_preregistration.md` 有预注册、未见对应结论 MD 或 EXP-ID —— 正式臂是否发射过不确定（脚本 `warm_rr_v2a.py` 存在）。
2. `UE_UAV_EFFECTIVE_DIRECTION_VERDICT_20261024.md`（远端判决原件）不在本 worktree，仅被 3 份文档引用；原件应在远端线 `codex/uav-bev-dual` 的会话工作区。
3. `UBR-RESEAL-R2_audit.md` 在册，但 wrapper `run_ubr_reseal_ddp4.sh` 的 `--audit-verdict` 与 PASS 断言指向旧名 `UBR-RESEAL-20261027_audit.md`（与在跑进程实参一致）——按现状登记，不改脚本。
4. 注册表 EXP-0025 行的"ep98–100 last 权重补录"备注与 SINGLESTREAM 台账的 test 数字并存：级3 收线前对照锚取哪个文件需主代理按预注册裁定（预注册写明 last ep100 为 `last_model_epoch100.pth`）。

---

## 四、训练入口速查（runbook）

**唯一合法发射路径 = wrapper**（`portable_sam2_explicit_coarse/scripts/ue/` 下）。禁止直接调用 `scripts/probes/ue_*.py`：wrapper 承担 fail-closed 五查（树洁 / HEAD 带 `exp/EXP-XXXX` tag / 审计文件 `VERDICT: PASS` / 输出根不存在 / 共租每卡 <2 训练进程），并把合同 echo 落到 `${OUT_ROOT}.launch_contract.txt`。公共环境 wrapper 自带：`PY=/data2/wangcheng/envs/cvt2/bin/python`、`SAM2_CKPT=/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt`、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`、`TRAIN_FLIP_PROB=0`（3090 协议）；当前在跑作业的卡面为 `CUDA_VISIBLE_DEVICES=0,1,2,3`。

| 实验 | 发射命令（在 `portable_sam2_explicit_coarse/` 下） | 前置门 | 状态 |
| --- | --- | --- | --- |
| **EXP-0028-R2** 级3 单流再封版 | `bash scripts/ue/run_ubr_reseal_ddp4.sh` | tag `exp/EXP-0028(-R2)`@HEAD、审计 `UBR-RESEAL-20261027_audit.md` PASS、树洁、out-root 不存在 | **在跑**（PID 2972162，ep21/100）；收线对照 = EXP-0025 last（先做 736c1532 锚点内容完整性核验），判定 RESEAL_ACCEPT/NULL/FAIL 三分支见预注册 |
| **EXP-0029 臂②** 双流真实 | `bash scripts/ue/run_ubr_dual_real.sh`（= `run_ubr_dual_ddp4.sh real`） | tag `exp/EXP-0029`、审计 `UBR-DUAL-20261027_audit.md` PASS、几何缓存存在、每卡 <2 训练进程 | **在跑**（PID 3006086，ep7/100） |
| EXP-0029 臂③ 双流常量 | `bash scripts/ue/run_ubr_dual_ddp4.sh constant` | 同上；PI 排队指令解除后手动发射 | 排队（内容门的 matched 常量对照） |
| **EXP-0030** 双流+RGB real | `bash scripts/ue/run_ubr_dual_rgb_real.sh real` | **等 PI 发射口令**；补打 tag `exp/EXP-0030`（当前缺，wrapper 会 fail-closed）；锚点缓存 3060 npz；排队守卫：须 0028-R2 或 0029 之一收线 | **冻结待发射** |
| EXP-0030 constant（预留第四臂） | `UBR_RGB_DESCRIPTOR_MEAN=<train-only 均值 json> bash scripts/ue/run_ubr_dual_rgb_real.sh constant` | 仅在 RGB_WINS 判决后 | 预留 |
| EXP-0001…0027（历史全量） | `run_exp0007_ddp4.sh` … `run_exp0027_ddp4.sh`、`run_exp0018_19_full.sh` 等 | **不要重启**——纪元已关账，数据/权重已归档 | done/terminated（台账为准） |

监控：launch log 在 `${OUT_ROOT}_launch.log`（tail 即可）；遥测 `epoch_telemetry.jsonl`；远端/本机巡检按 `docs/ops/OPS-AGENT.md` 走 OPS 代理，基线见 `docs/ops/session-log.md` OPS-001。

*本地图由 2026-10-28 组织通行成（OPS-002）；此后结构性变化请随段更新本表与 §一。*
