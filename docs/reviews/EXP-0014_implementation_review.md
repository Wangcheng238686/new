# EXP-0014 实施+接线独立审计（BEV 逐通道定标融合臂 dualnorm）

- 审计日期：2026-10-16（全量启动前）
- 审计对象：worktree `codex/dual-stream-dev` 未提交改动（`ue/dual.py`、`ue/contract.py` 修改；`configs/ue_whu_a3sgm_512_dualnorm.py`、`scripts/ue/harvest_bev_stats_exp0014.py`、`scripts/ue/run_exp0014_ddp4.sh`、`docs/EXP-0014_preregistration.md` 新增）
- 预注册：`docs/EXP-0014_preregistration.md`；诊断依据：`docs/DUAL-DELTA-DECOMP-20261016_conclusion.md`
- 独立性声明：审计过程未修改任何被审计文件；唯一写入为本报告。所有结论均由本机重跑命令验证，不采信声明文本。
- 环境：`/data2/wangcheng/envs/cvt2/bin/python`，`SAM2_CKPT=/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt`。

## 1. 单变量原则

**命令**：`diff configs/ue_whu_a3sgm_512_dual.py configs/ue_whu_a3sgm_512_dualnorm.py`

**结果**：差异恰为 docstring 替换（1–11 行）+ 一行插入（第 20 行）：

```
 'bev_stats': '/data2/wangcheng/checkpoint/surfbev_b0/bev_stats/exp0014_bev_scale_p3.json',
```

模型字典其余 200+ 键逐字节相同（含 fusion 字段 `target_level=1`、uav_encoder、batch_augments、train/test_cfg、`sam_image_embedding_stride`）。

**指纹 harness**（按 wrapper 同款命令，对 `ue.contract.BASELINE_CONFIGS` 全部 7 条逐一执行）：

```
ue-whu-a3sgm-v1            5d5f78ec947f...  resolves->自身
ue-whu-a3sgm-512-v1        5305b3155914...  resolves->自身
ue-whu-a3sgm-512-dual-v1   4ccb42b87f8e...  resolves->自身   （=EXP-0010 wrapper 钉值）
ue-whu-a3sgm-512-dualv2-v1 d26f1f97ed9c...  resolves->自身
ue-whu-a3sgm-512-duallora  50f6b0979a47...  resolves->自身
ue-whu-a3sgm-512-dualv2-l0 277a95dd4d00...  resolves->自身
ue-whu-a3sgm-512-dualnorm  473d1ede426287dfbe2562135430524fa86c875960ab98f8bb89523493c92a99  resolves->自身
```

六条既有 baseline 前缀逐一不变；新身份可解析且指纹 = wrapper 钉值 `473d1ede…c92a99`。**通过**。

## 2. 默认路径不变（bev_stats=None）

**命令**：真实构建两臂模型（`MODELS.build(Config.fromfile(...).model)`，加载 SAM2 权重），输出：

```
OLD arm: hasattr(bev_scale) = False | getattr probe = None | state_dict keys w/ bev_scale: []
OLD arm: gate init all-zero: True | telemetry init keys: [bev_coverage, delta_p3_ratio, gate_absmax, gate_mean, gate_std]
NEW arm: bev_scale buffer shape (1,262,1,1) fp32 | in state_dict: True | requires_grad: False
         | equals JSON scale reshaped: True
state_dict key diff old->new: ['bev_scale']   new->old: []
```

- (a) None 路径不创建属性，`getattr(self,'bev_scale',None)` 探测返回 None（`ue/dual.py` L167）；
- (b) 两臂 state_dict 键差恰为 `bev_scale` —— 旧臂 checkpoint strict-load 不受影响；
- (c) buffer 注册于 `__init__`（L119-124），应用点在 `_uav_bev` 内 rasterize（L163-166）之后、返回（L170）之前，proj 在 `extract_feat` L185；`_uav_bev` 调用点（L184）位于 `torch.autocast('cuda', enabled=False)` 岛内（L178）——即 fp32、AMP 之外；`.to(bev.device)` 不降精度；
- (d) 用新臂真实模块数值验证 gate=0 等价性：`proj(bev')` 有限（‖·‖≈837，O(1) 输入），`gate⊙proj` **逐位为 0**，`base+delta` 与 `base` **逐位相等**（IEEE 0×finite=0，定标后无 inf/NaN 风险）。第 0 步单流等价性保持。

附带验证：DDP `broadcast_buffers` 同步该 buffer 无害（各 rank 从同一 JSON 构建，逐位一致）；buffer 不入 optimizer 参数组（requires_grad=False）。**通过**。

## 3. harvest 保真

**方法**：逐行对照 `scripts/ue/harvest_bev_stats_exp0014.py` 与 `ue/dual.py::_uav_bev` 冻结分支，并做一次**真实等价性重放**（train split 第 0 场，CPU，两路径同设备）：

- encoder 构建：harvest `Config.fromfile('configs/ue_whu_a3sgm_512_dual.py')` 的 `model.uav_encoder`（env 注入 SAM2_CKPT 后与训练同一权重文件）；实测 `model.uav_encoder` 与 harvest 构建 encoder 的 state_dict **逐参数相等**（`all torch.equal`）；
- 归一化常数、`max(fpn[-4:], key=area)` + `.float()` desc 选择、`_k_and_w2c`（直接 `from ue.dual import`）、`edges=[-1e9,1e9]` 单 bin、`HW=(64,64)`、`devicez_rgb`/相机张量：逐行一致；collate 侧 `uav_batch` 的 `[0]` 剥层与原始 dataset item 为同一张量（`data/loader.py` L102-109：`torch.stack` 单样本 + 相机保持 per-sample list）；`rasterize_surfacebev` 默认 `with_grad=False`，与训练臂 `with_grad=self.uav_encoder_lora=False` 相同；
- P3 网格实测：512 输入 `extract_feat` 输出 level1 = `(1,512,64,64)`，与 `HW=(64,64)` 一致；
- **等价性重放**：harvest 循环重算的 `bev` 乘以 scale 后与真实 `m._uav_bev(uav, (64,64))` 输出**逐位相等（max abs diff = 0.0）**；该场定标后 rms：desc≈0.09–0.40、zmean 1.13、zres 1.07——五个数量级失衡收敛到一个数量级，与诊断/预注册叙事一致；
- split：`UERawPairDataset(SNAP,'train',True,'surfbev_b0_split_v2.json',image_size=(512,512))`——构造器校验 manifest 元数据后 `self.records` 仅保留 `split=='train'`（v2 期望 694），`__getitem__` 只开 train 场景文件；**无 val/test 样本泄露**；
- rms 口径：`sq += bev.pow(2).sum(dim=(1,2))`、`n_cells += 64*64`——含未覆盖零单元，分母 694×4096，与预注册公式逐字一致；`scale = 1.0/rms.clamp_min(0.05)` 即 `s=1/max(rms,0.05)`，floor 语义一致。

**通过**。

## 4. stats 产物

**命令**：`sha256sum` + JSON 解析。

- sha256 = `f7c3c5fba3ff0ea67ad26dde44660180dcf436a56062149ef0bdd907f9a1ad3b`（= 预注册附录 = wrapper L49 钉值，三处一致；harvest 日志末行亦打印同值）；
- `len(rms)=len(scale)=262`；`n_scenes=694`、`hw=[64,64]`、`floor=0.05`、`schema=exp0014-bev-scale-v1`；
- 抽验恒等式：`max_i |scale[i]·max(rms[i],0.05) − 1| = 5.5e-8`（fp32 JSON 往返容差内）；214/262 通道被 floor 截断（×20），定标后 rms 范围 [0.067, 1]；
- 预注册附录引用数字与产物一致（见 §7 勘注）。

**通过**。

## 5. wrapper 守卫族

**命令**：`diff scripts/ue/run_exp0010_ddp4.sh scripts/ue/run_exp0014_ddp4.sh` + 逐行阅读。

守卫族与 0010 逐项同构，差异恰为：注释、EXP-ID 门（`EXP-0014`）、RUN_TAG（`…dualnorm…EXP-0014`）、新增 `BEV_STATS` sha 守卫（L48-51）、三处 config 路径（指纹/身份/exec）、契约 echo 前缀 `EXP14_CONTRACT` 及新增字段 `levels=P3-only bev_stats_sha256`。其余逐字节相同：

- manifest 哈希 `f15cd8f5…`、split 哈希 `17f49430…`、快照 payload 指纹 python 段（逐字符同 0010）、架构指纹守卫（`473d1ede…`，已实测命中）、身份守卫（`ue-whu-a3sgm-512-dualnorm-v1`，已实测解析）、干净树、输出目录不存在、无参 guard、`B0_PREFLIGHT_ONLY` 旁路；
- 训练参数与 0010 逐项一致：`epochs 100 / batch 1 / accum 2 / lr 5e-4 / wd 0.05 / warmup 100 / seed 44 / val-batch 1 / val-every 1 / compute-val-loss 0 / amp 1 / ema 0`，`SAVE_LAST_MODEL=1`、`UE_LOAD_UAV=1`，exec 行唯一差异 `--config …dualnorm.py`；
- **实弹测试**：`B0_EXP_ID=EXP-0014 bash scripts/ue/run_exp0014_ddp4.sh` → `EXP-0014 full training requires a clean Git worktree`，exit 2（dirty 守卫生效）；
- `WORK_DIR=/data2/.../surfacebev_v2_512dualnorm_sealed_ddp4_s44_EXP-0014` **不存在**（未启动）。

**通过**。

## 6. 预注册四要素与遥测

- 主指标：val 100 场 segm/bbox AP@[.50:.95] 的 EXP-0014−EXP-0009 差 + 机制对照 EXP-0014−EXP-0010（定标净效应）+ 伴随指标（AP50/75、面积分层、UAV-off 消融、开度全套）——完备；
- 口径：last epoch（第 100），best 仅参考——明确；
- 显著性：scene-paired percentile bootstrap 10,000 draws、seed 20261014、单 seed 边界如实标注——明确；
- 判定门：DUAL_GAIN / DUAL_ENGAGE_NULL / DUAL_USED_BUT_NULL / DUAL_HARM 四门预注册 + 非门预测登记（ENGAGE_NULL 或 USED_BUT_NULL）——完备；
- v2.3 有效开度算子替换勘误：先登记于 `docs/DUAL-DELTA-DECOMP-20261016_conclusion.md`（mtime 2026-10-16 00:57），EXP-0014 预注册 §"遥测与判定算子（v2.3 勘误，先于本臂启动）"采纳（open_L = Σ|gate⊙proj(bev')|/Σ|proj(bev')|，参与 ⇔ ≥0.20，legacy δ_L 同报对照、不追溯已启动臂）；本臂尚未启动 ⇒ 勘误先于启动成立；
- 遥测键不变：`ue/dual.py` L190-197 单层时 `tag=''`（无 `_L` 后缀），diff 未触碰遥测；smoke 日志实测 `gate_mean=0 gate_std=0 gate_absmax=0 delta_p3_ratio=0 bev_coverage=0.9238 gate_steps=2`——单层无后缀。**通过**。

## 7. 冒烟证据

`/data2/wangcheng/checkpoint/surfbev_b0/probes/EXP0014-smoke.log`（DDP4，2026-10-16 01:23:26–01:24:02）：

- 4 rank Gloo 互连、seed 44、`fingerprint=473d1ede…`（配置/身份）、Dataset size 6（≈1% 子集）、2 epoch；
- optimizer 分组 `other … first=gate last=proj.bias`（gate+proj 在优化器内）；
- 遥测键无后缀；epoch 1/2 内 `gate_absmax=0, delta_p3_ratio=0`——与 EXP-0012 已放行冒烟同签名（查证 `EXP0012-smoke.log` epoch 1 遥测同为全 0，warmup 100 + AMP 首步跳步伪影），`bev_coverage=0.92` 证明 UAV 前向实际运行；"第 0 步等价性"由 §2(d) 的逐位数值验证补强。

`EXP0014-smoke2.log`（单 rank，01:25:26–01:27:35，无 Gloo/Rank 行）：Dataset size 20、`warmup: 10 optimizer steps`、3 epoch：

```
gate_absmax:     3.49e-4 → 2.62e-3 → 4.45e-3
delta_p3_ratio:  8.03e-5 → 5.24e-4 → 8.16e-4   （单调上升）
```

与预注册附录描述（"3.5e-4→2.6e-3→4.4e-3，delta_p3_ratio 8e-5→8e-4 单调上升"）一致——归一化通路梯度存活的直接数值验证。两份冒烟均在 scratch/probes 目录（非正式产物，voided）。**通过**。

## 红线检查

1. **无跳级**：本臂属 EXP-0008 谱系（从头联合训练 A/B）；PI 级热启动豁免已登记于 `docs/EXP-0008_preregistration.md`（"PI 级豁免（跳过筛查层直接双臂全量），在此登记"），EXP-0014 预注册 §路由判定继承并如实标注基座动力学型风险。合规。
2. **预注册先于编码启动**：工件时间链 coherent——probe 00:53 → 诊断结论(含 v2.3 勘误) 00:57 → harvest 脚本 01:08 → stats 产物 01:16 → wrapper(钉 sha) 01:22 → dual.py 定稿 01:23:10 → 冒烟 01:23–01:27 → 预注册附录补登 01:27:49 → 本审计。全量未启动（WORK_DIR 不存在）。
3. **审计先于启动**：本报告即启动前审计；启动被 dirty-tree 守卫阻塞直至 commit。
4. **无 dirty 全量启动路径**：wrapper L25-27 干净树守卫实测触发（exit 2）。

## 勘注（非阻塞）

1. **中位数口径**：预注册附录"desc[0:256] rms 中位 0.01422"与 harvest 日志一致（torch.median 偶数长度取下中位，`%.4g` → 0.01422）；若以 statistics.median（两中位平均）重算为 0.014234——非数值差异，仅约定不同，附录忠实转录日志。
2. **`bev_stats` 路径进入指纹原文**：`architecture_contract._PATH_KEYS` 仅归一化 `checkpoint/checkpoint_path/ckpt_path`，`bev_stats` 绝对路径原样入指纹与 config snapshot。本机上确定性无误（已验证），跨机复现指纹需同路径；与谱系既有做法（环境注入路径归一化）略不一致，可在谱系下次契约演进时纳入归一化——**本次不得改动**（单变量冻结）。
3. **启动 residual risk（沿袭 0010–0013 模式）**：wrapper 哈希守卫覆盖 config/身份/stats，但不哈希 `ue/dual.py` 源文件本身；干净树守卫保证启动 commit 即含本次审计过的文件——启动 commit 不得再改动这些文件，否则本审计失效。
4. dual.py mtime 01:23:10 晚于 wrapper 01:22（冒烟前最后一次定稿）；冒烟与本次全部实测均基于最终版本，指纹/身份不受影响（指纹仅哈希 config）。

## 结论

七项核查全部通过；四条红线全部满足；勘注四条均不阻塞。单变量原则在 config、代码路径、state_dict、遥测、训练参数五个层面均得到独立复核；harvest 与训练路径在真实场景上逐位等价；stats 产物三处 sha 钉死一致；冒烟证据与预注册附录吻合。

VERDICT: PASS
