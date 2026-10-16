# EXP-0017 launch-gate audit + EXP-0015/0016 八臂拉通加遗结果审计（2026-10-16）

独立审计人复核。审计对象两件：

- **TASK A（launch gate）**：EXP-0017 实施提交 `d6d6288`（HEAD，clean worktree）。
- **TASK B（archival gate）**：`docs/DUAL-AB-LADDER-20261016_conclusion.md` 追加节
  "EXP-0015/0016 并入八臂拉通"，底层数据
  `/data2/wangcheng/checkpoint/surfbev_b0/probes/DUAL-AB-LADDER-20261016/summary.json`
  （由 `portable_sam2_explicit_coarse/scripts/ue/probe_dual_ab_ladder.py` 重算，
  ARMS 八臂，delta_operator 含 v2.3 `open_eff` 与 `E` 位）。

---

## TASK A — EXP-0017 实施审计（launch gate）

### A.1 提交触及面

`git show --stat d6d6288` 恰好 5 个文件、全部为新增行（152 insertions, 0 deletions）：

1. `portable_sam2_explicit_coarse/train/loops.py`（+24）
2. `portable_sam2_explicit_coarse/train/cli_args.py`（+2）
3. `portable_sam2_explicit_coarse/scripts/ue/run_exp0017_ddp4.sh`（+68，新文件）
4. `docs/EXP-0017_preregistration.md`（+57，新文件）
5. `docs/EXP_REGISTRY.md`（+1，EXP-0017 行）

与要求清单一致。**通过。**

### A.2 loops.py 差异分析

新增代码共两块，逐条核对：

**(a) 钉零触发条件与执行点**（loops.py:70-80, 248-251）：

- `unlock = int(getattr(args, "fusion_unlock_epoch", 0) or 0)`——flag 缺失（getattr 默认）或
  传 0/None 时 `unlock == 0`。
- 收集与钉零仅在 `unlock > 0 and epoch_number <= unlock` 分支发生（epoch_number 1-indexed：
  unlock=50 ⇒ 第 1–50 epoch 含被钉，第 51 epoch 起自由，与预注册"第 1–50 epoch（含）值钉零"
  语义一致）。
- `core = model.module if hasattr(model, "module") else model`——取 DDP unwrap 后的模型。
- 钉零位于 `scaler.step(optimizer)` + `scaler.update()` 之后、`ema.update(model)` 之前，
  处于 optimizer-step 块内（`if (batch_idx + 1) % grad_accum == 0 or ...`），每个 optimizer
  step 执行一次；`with torch.no_grad(): p.zero_()` 为纯值写入。EMA 与 DDP 同步审计看到的都
  是钉零后的值，次序无害。

**(b) 默认路径与父提交（e420195）行为逐位一致**：两个 diff hunk 的全部新语句中，`unlock==0`
时实际执行的只有一次 Python 属性读取/整数转换、`fusion_gate_params = []` 赋值和两个恒为
False 的分支判断 + 一个恒为 False 的 `if fusion_gate_params:`；无任何张量算子、无 RNG 消耗、
无日志。`cli_args.py` 仅新增 default=0 的 argparse 旗标。张量数值路径与父提交 bit-identical。

**(c) 匹配器安全性**：`n == "gate" or n.startswith("gate.") or n == "gate_e"`。全库 `self.gate`
赋值仅存在于 `ue/dual.py` 的 `RSPrompterAnchorDual.__init__`：单 level（本配置
`target_levels: [0]`）时 `self.gate = nn.Parameter(...)`（名字恰为 `gate`）、多 level 时
`nn.ParameterDict`（名字 `gate.<t>`，被 `startswith("gate.")` 覆盖）、embed 臂
`self.gate_e = nn.Parameter(...)`（恰为 `gate_e`）。父类 `RSPrompterAnchor`
（rsprompter/detection/anchor_detector.py）全文无 `gate` 属性；其余参数全部位于子模块路径
前缀之下——卫星侧 LoRA 在 `backbone.encoder.base_model.model...`（冒烟日志 optimizer 组
首张量 `backbone.encoder.base_model.model.trunk.patch_embed.proj.lora_A.default.weight`），
UAV LoRA 在 `uav_encoder.encoder.base_model.model...`，掩码解码器/shape_prior 在
`roi_head.mask_head...`。顶层无任何名为 `gate`/`gate.*`/`gate_e` 的非融合参数，匹配器
不可能误伤。冒烟日志融合组恰 3 张量（`first=gate last=proj.bias`，
135,168 = 262×512 + 512 + 512）为旁证。

### A.3 wrapper 对比（run_exp0017_ddp4.sh vs run_exp0015_ddp4.sh）

`diff` 全部差异且仅限：

- 头注释（EXP-0017 描述，含身份复用先例说明）；
- `B0_EXP_ID` 守卫及各错误消息中的 EXP-0015→0017 文案；
- `RUN_TAG`：`..._dualv2l0lora_sealed_...EXP-0015` → `..._dualv2l0lora_fk50_sealed_...EXP-0017`（fk50 stem）；
- 契约 echo：前缀 `EXP0015_CONTRACT`→`EXP0017_CONTRACT`、exp_id、run_tag，第二行末尾新增
  ` fusion_unlock_epoch=50`（其余逐字符同 0015：fusion_no_wd=1 保留）；
- trainer 调用新增 `--fusion-unlock-epoch 50`。

守卫全部原样：EXP-ID 守卫、零位置参数守卫、clean worktree 守卫、WORK_DIR 不存在守卫、
accepted_pairs/split/快照指纹、架构指纹 `5c7e1afc…`、baseline id
`ue-whu-a3sgm-512-dualv2-l0-lora-v1`（与 0015 相同——身份复用）。config 路径
`configs/ue_whu_a3sgm_512_dualv2_l0_lora.py`、seed 44、epochs/lr/wd/accum 等协议逐字同 0015。
预注册文档第 13–14 行明文引用 EXP-0007/0009 身份复用先例（wrapper 头注释亦引）。**通过。**

### A.4 微跑证据（/tmp/fk50_smoke.log，3-epoch、unlock=2 缩比）

三条预注册机械核验线全部在日志中兑现：

1. **锁定期遥测恒零**：
   - Epoch 1：`gate_mean=0 gate_std=0 gate_absmax=0 delta_p3_ratio=0 bev_coverage=0.887337 gate_steps=8`
   - Epoch 2：`gate_mean=0 gate_std=0 gate_absmax=0 delta_p3_ratio=0 bev_coverage=0.875099 gate_steps=8`
   （bev_coverage>0 且 gate_steps>0 ⇒ 前向确实带了 UAV 支路而注入恰为 0.0，非"支路没跑"。）
2. **解锁行**：`fusion gate unlock takes effect at epoch 3`（第 182 行；另有 epoch-1 的
   `fusion gates pinned to zero through epoch 2 (1 tensors)`，张量数=1 与单 gate 配置一致）。
3. **解锁后即刻活动**：Epoch 3 `gate_absmax=0.000633439`、`delta_p3_ratio=0.103726`，均 >0。

优化器分组：`Optimizer group fusion params=135168 tensors=3 lr=5.000e-04 wd=0.000e+00
first=gate last=proj.bias`——wd=0、恰 3 张量（gate/proj.weight/proj.bias），与 0015 配方
`--fusion-no-wd 1` 一致；日志中架构指纹 `5c7e1afc…` 与 wrapper 守卫值一致（同一 config）。
**通过。**

### A.5 放行条件

- **输出目录**：`/data2/wangcheng/checkpoint/surfbev_b0/surfacebev_v2_512dualv2l0lora_fk50_sealed_ddp4_s44_EXP-0017`
  不存在（surfbev_b0 下无任何 fk50/EXP-0017 条目）。✓
- **GPU**：无本谱系任何进程在跑（无 `train_rsprompter_fusion`/`torch.distributed.run`/
  `run_exp00*`）。注意（非阻塞，操作层面）：4 卡当前各被 1 个本账号（wangcheng）的
  UnrealEditor CitySample 离屏渲染（10–13 GB）与 shibaotian 的训练（约 13 GB）共占，
  每卡余约 22–25 GB（卡总 48 GB）；冒烟微跑（18:41，与 UE 渲染共存的时段）已正常完成，
  但正式 4 卡 DDP 启动前建议确认显存余量可容纳全程。按审计条款字面（"无我方实验进程"）判通过。
- **预注册完整性**（docs/EXP-0017_preregistration.md）：四要素（主指标/选择口径 last-epoch-100/
  显著性 seed=20261014 10k 抽样/判定门）齐；预注册解读条款 4 条（主预期 DUAL_GAIN、惰性复发、
  代价持续、DUAL_HARM 停谱系）齐；机械核验线（锁定期注入比恒 0、两条 rank0 日志行、解锁后 >0）齐。✓
- **操作提示**：本审计报告文件入库提交前 worktree 为 dirty 态，会触发 wrapper 的 clean-worktree
  守卫——放行后需先提交本报告再启动。

---

## TASK B — EXP-0015/0016 并入八臂拉通（archival gate）

### B.1 行文件与身份

- `rows_dual_l0lora.jsonl` / `rows_dual_l0lora_off.jsonl` / `rows_dual_embed.jsonl` /
  `rows_dual_embed_off.jsonl` 各恰 100 行，scene_id 顺序与 `rows_single.jsonl` 完全一致
  （summarize 侧亦有同名断言，本审计独立比对通过）。
- `identity_dual_l0lora.json`：`baseline_id = ue-whu-a3sgm-512-dualv2-l0-lora-v1`（= wrapper
  守卫值）、`checkpoint_sha256 = 7ad098a35af6a0c103f31532f6859d68513bf02fd1d658f171e8a6392adf68d6`；
  `identity_dual_embed.json`：`baseline_id = ue-whu-a3sgm-512-dualembed-v1`、
  `checkpoint_sha256 = e57948565c4b501b5f1c9f9df2e33fc65f6ef57322b54b1d9a0cd74b40bb8e5e`。
- `sha256sum` 实测：EXP-0015 `last_model_epoch100.pth` = `7ad098a3…`，EXP-0016 =
  `e5794856…`，逐字符匹配。✓

### B.2 独立重算（同 seed 20261014、10,000 draws、逐对照重置种子的 percentile bootstrap）

用与 `probe_dual_ab_ladder.py::boot()` 相同的算子（`np.random.default_rng(20261014)`、
`r.integers(0, n, (10000, n))`、`np.quantile [.025,.975]`）从 rows 文件独立重算，
**全部与 summary.json 全精度一致**，换算成点数后：

| 对照 | segm | bbox |
| --- | --- | --- |
| dual_l0lora − single | +0.02 [−0.63, +0.64] | −0.25 [−0.88, +0.37] |
| dual_embed − single | −0.62 [−1.19, −0.05] | −0.75 [−1.30, −0.21] |
| l0lora on − off | +0.38 [+0.06, +0.70] | +0.34 [+0.02, +0.67] |
| embed on − off | +0.26 [+0.12, +0.40] | **恰 0.0，CI [0,0]** |

**注入位纯度核验**：`rows_dual_embed.jsonl` 与 `rows_dual_embed_off.jsonl` 的逐场 bbox 值
逐位相同（100/100 全等，`n_pred` 亦全等；segm 逐场不同——符合"embedding 只达掩码合成"）。
on−off bbox 恒 0 是逐位精确的，非舍入产物。✓

### B.3 dual_vs_dual

- `dual_l0lora − dual_p2`：segm +0.73 [+0.11, +1.38]（CI 不含 0，显著）、bbox +0.54
  [−0.11, +1.20]（含 0）。与 summary.json 及重算一致。✓
- `dual_embed − dual_p3`：segm +0.32 [−0.35, +1.02]、bbox +0.03 [−0.65, +0.69]，
  两 CI 均含 0。✓

### B.4 结论文档数字交叉核对

- 加遗表逐场 on/off 均值：0015 on 0.7766/0.7509、off 0.7729/0.7475；0016 on 0.7703/0.7459、
  off 0.7677/0.7459——与 summary.json means（0.776612/0.750865、0.772853/0.747461、
  0.770258/0.745863、0.767664/0.745863）一致。✓
- Δ 与 on−off 列：与 summary.json 及本审计重算一致（见 B.2/B.3 表）。✓
- 判定：0015 双指标 vs single CI 均含 0（segm 点估计 +0.02 精确持平）——"双指标无显著差异"
  成立（summary.json 机械落判 DUAL_ENGAGE_NULL：open_eff 0.006<0.20 且 delta_L 8.7e-5<0.20）；
  0016 两指标 CI 上界均 <0 ⇒ DUAL_HARM，与 summary.json verdicts 一致。✓
- 训练日志终值：EXP-0015-ddp4.log 末行 `Epoch 100/100 | … segm/mAP=0.7576 bbox/mAP=0.7335`；
  EXP-0016-ddp4.log 末行 `Epoch 100/100 | … segm/mAP=0.7557 bbox/mAP=0.7334`——与加遗表
  "全局 last" 及 EXP_REGISTRY 行一致。✓
- 末代遥测引用：0015 `Epoch 100 DUAL telemetry: … gate_std=0.00553824 delta_p3_ratio=0.488765 …`
  （文档引 0.489、门 std 0.0055）；0016 `Epoch 100 … delta_p3_ratio_E=0.0649016`（文档引
  0.065）且 ep1 `delta_p3_ratio_E=0.443361`（文档引 0.443）；open_eff：0015 0.00603→0.006、
  0016 0.002486→0.0025；legacy delta_L 两臂 8.7e-5 / 1.1e-4 ≈1e-4。全部一致。✓
- 八臂总表逐场均值：single 0.7764/0.7533、0012 0.7751/0.7527、0011 0.7702/0.7471、
  0013 0.7693/0.7454、0010 0.7670/0.7458、0014 0.7674/0.7446——与 summary.json means 一致。✓

**非阻塞性笔误/措辞备注**（不影响任何判定）：

1. 机制读数 1 的"训练期代价（off−single = −0.35/−0.58）"精确值为 −0.358/−0.586（按四舍五入
   应作 −0.36/−0.59，文档向零截断）；EXP-0017 预注册沿引同值。量级与结论不变。
2. 八臂总表的">"链与所引两列数字非严格单调（0016 0.7703 排在 0011 0.7702 之后、0014 0.7674
   排在 0010 0.7670 之后）；差异 ≤0.0004 且均在噪声内，排序口径未注明。
3. "单流 0.7764/0.7533 仍为天花板"：segm 点估计上 0015（0.7766）微高于单流（+0.02，
   CI [−0.63,+0.64] 不显著）；"天花板"只在不显著意义上成立，加遗表自身已如实给出该 CI。

### B.5 归档日志完整性

- `/data2/…/launcher_logs/EXP-0015-ddp4.log`（5506 行）与 `EXP-0016-ddp4.log`（5529 行）均含
  `Saved final checkpoint: …/last_model_epoch100.pth` 且末行即 `Epoch 100/100 …`（完整收线日志）。
- 仓库副本 `artifacts/launcher_logs/EXP-0015-ddp4.log` sha256 =
  `11cd0e38a1ce7040a325a50e98dd6f7b47b7d334274d578794589a278682dd74`、
  `EXP-0016-ddp4.log` = `2399add000c39d22475fbb11016d676222901079ab8fc10dc79996d4cd2687f3`，
  与 /data2 原件逐字符相同。✓

---

## 结论

TASK A：EXP-0017 为 EXP-0015 配方上的干净单变量（`--fusion-unlock-epoch 50`），默认路径与
父提交逐位一致，匹配器只及融合门控，微跑三核验线齐证，输出目录未占用，无我方实验进程；
GPU 存在共占（本账号 UE 渲染 + 第三方训练，每卡余 22–25 GB），属操作提示而非协议缺陷。

TASK B：加遗节全部载荷数字（Δ、on−off、dual_vs_dual、逐场均值、全局 last、末代遥测、
open_eff/delta_L）与 summary.json、训练日志及本审计独立重算一致；embed on−off bbox 逐位
恒 0 属实；归档日志完整且 sha256 相符。仅 3 处非阻塞笔误（截断舍入、总表排序口径、
"天花板"措辞）。

TASK A VERDICT: PASS

TASK B VERDICT: PASS
