# SURFBEV-P1-V2-R1 可恢复执行独立审计

**审计对象：** `portable_sam2_explicit_coarse/scripts/ue/probe_surfacebev_p1.py` 与
`docs/SURFBEV-P1-V2_preregistration.md` 的 R1 条目；同时静态检查其直接依赖的
`rsprompter/surfacebev.py` 和 raw-pair reader。

**审计范围：** 只读静态审计与小型 NumPy dtype 复现；未运行 `--allow-test`，未读取任何
test scene，未改动实现。

## 结论

**VERDICT: FAIL — 不得启动 `evaluate --allow-test`，不得开封 test。**

R1 的阶段切分本身大体正确：`harvest` 明确拒绝 `--allow-test`，且仅构造 train/val
dataset；`evaluate` 在 `fit` 和完整 val 遍历后才调用 `harvest_test`。原子 `.npz` 分片和按
`(SEED, scene_index, target_index)` 的抽样也能使正常中断后的 train/val 恢复保持逐场样本
不变。

但存在一项会直接破坏预注册第二项伴随指标的数值错误，另有一项会允许改码后静默复用旧
cache 的契约漏洞。这两项都必须在 test 前修复、重新审计。由于尚未读取 test，修复仍可在
不重置 test 的前提下进行；但受影响的 train/val cache 不能被直接沿用。

## 已通过的检查

### 1. train/val 阶段不打开 test：通过

- `run_harvest` 首先拒绝 `a.allow_test`（脚本第 88–90 行）。
- 该阶段只构造 `('train', 'val')` 两个 `UERawPairDataset`，没有 test dataset 或 test
  `__getitem__` 路径（第 90 行）。
- 每场特征落盘先写同目录临时文件，再 `os.replace`（第 26–29、87 行）；正常进程中断不
  会留下被当作完成场景使用的半文件。

### 2. val 选择在 test 原始场景读取前：通过

- `run_eval` 先要求完整的 `train_val_complete.json`（第 145–148 行）。
- 它完整遍历预注册的 `C` 与阈值，在 `vf` 上选择唯一的 `(C, threshold)`（第 149–154
  行），此时尚未构造 test dataset。
- 选择值先写入 `test_opened_once.json`，其后 `harvest_test` 才构造 test dataset 并访问
  `ds[i]`（第 129–137 行）。同一 lock 下只允许继续未完成的 test 分片。

### 3. 标签隔离、三臂容量和几何输入：通过（静态）

- `rasterize_surfacebev` 的参数不含卫星标签或 B0 预测；标签仅在 `pack_scene` 中、两套
  SurfaceBEV 均构建完成后生成读出 target（第 75–86 行；`surfacebev.py` 第 13–18 行）。
- correct 与 sham 调用共用同一个 `desc`、相机和深度张量；唯一调用差异是 `sham90=True`
  （第 96、137 行）。二者输入点数相等和有限性也被 fail-closed 检查（第 97–99、138–140
  行）。
- 三臂最终均是 64 维，train boundary/error 样本均由同一 scene/target 的最多
  256 正、256 负 cell 构成（第 60–65、77–84 行）。P3/SurfaceBEV 的 row-major 形状检查
  仍在 preflight（第 34–35 行）。

### 4. 判定门和 bootstrap：未见 R1 改动

`CS`、阈值集合、151-scene paired bootstrap 的 10,000 draws/固定 seed，以及 six-way
`correct−SAT` / `correct−sham90` 的 CI 下界全大于零的 PASS 条件，均与预注册一致
（第 16、23、149–159 行；预注册第 30–36 行）。此项本身通过，但下述 IoU target 错误会
使其中的 `rho` 伴随门无效，不能据此作 P1 判定。

## 阻断项 A：实例 IoU target 因 uint8 矩阵乘法溢出

**严重性：阻断。**

`iiou` 将展平的 1024² bool mask 转为 `np.uint8`，随后以 `a @ b.T` 计算交集
（脚本第 71–74 行）。NumPy 的 `uint8 @ uint8` 保持 `uint8` 输出：任意面积超过 255 的
交集会按 256 取模，而非得到真实像素计数。审计中以 1×1,048,576 的全 1 uint8 向量复现：
输出 dtype 为 `uint8` 且点积为 `0`。

因此 `qi`、`y_iou` 和 train/val/test 的实例质量 target 全部错误；Ridge 和 Spearman
不能代表预注册的“best-IoU strict 实例 IoU”。这既不是仅遥测问题，也不只是某一场景的
边缘情况。

**必须修复：** 在矩阵乘法前转换为至少 `int32`/`int64` 或 `float32`，并以一个面积
大于 255 的合成 mask 加入 fail-closed 单元/运行前检查。该改动改变现有 cache 中的
`qi/y_iou`，故必须使用新的 cache/output 目录重新 harvest train/val；此前异常运行和 R1
cache 都不得混用。修复后须在 test 前复审。

## 阻断项 B：cache 的 source hash 被有意忽略，不能防止语义漂移

**严重性：阻断。**

R1 文档声称 cache contract 锁定“源文件哈希”（预注册第 54–55 行），`make_contract` 也
写入 `source_sha256`（脚本第 38 行）。但实际比较循环对该字段显式跳过（第 43–44 行）。
所以在 harvest 与 evaluate 之间，只要 B0/H0/height 边界等不变，修改本脚本仍可静默复用
旧分片；更不用说 `surfacebev.py`、`ue_raw_pair_dataset.py`、H0 helper 的改动根本不在该
hash 内。这样既可能改变 train/test 的 feature 语义，也可能让 `test` 分片在修改后的
实现下继续 harvest，违反 R1 的“相同 lock 下恢复”所需的实现不变量。

**必须修复：** cache contract 应 fail-closed 比较全部代码/配置指纹；至少包括本脚本、
`surfacebev.py`、raw reader 与被导入的 H0 helper 的 SHA（或采用经审计的单一 commit SHA），
并在任何不匹配时拒绝恢复。已有不完整 cache 不应通过“忽略 hash”来继续。修复后重新
harvest 至新目录并复审。

## 非阻断但应补强的恢复审计项

1. 已存在的 `.npz` 只按存在性跳过（第 94–95、135–136 行），未检查分片包含预期 key、
   shape、dtype、scene_id 或逐片内容 hash。原子写能覆盖中断，不覆盖磁盘损坏、手工替换或
   错目录复用。建议每次恢复时校验这些结构，至少将 `scene_id`、contract hash 和 feature
   shape 写入并检查。
2. `train_val_complete.json` 仅证明写入时的 dataset 长度，不证明 train/val 目录中恰有
   694/100 个完整、正确索引的分片；`fit` 会默默使用 glob 到的任何数目。应在 evaluate
   前显式要求索引集合恰为 `0000..0693` 和 `0000..0099`，且每片通过上述结构校验。
3. `test_opened_once.json` 可防止选择值变更，但 `evaluate` 完成后仍可再次执行同一评估并
   重读缓存。指标是确定的且不再调参时风险较小；仍建议把完整 summary/per-scene 的 hash
   与 completed 状态写入 lock，后续命令只允许验证/读取，不允许覆盖结果。

## 本轮审计动作记录

- 阅读 R1 预注册、实现、SurfaceBEV 构建器和 raw reader；未运行数据集或模型前向。
- 执行 `python -m py_compile`：通过。
- 仅在内存中执行 NumPy dtype 微复现，确认 `uint8 @ uint8` 的模 256 溢出；未访问 test。
- 未编辑实现文件，未创建或修改 P1 cache/结果工件。

## 放行条件

在 **A、B 均修复** 后，使用新的未污染 output/cache 目录完成 train/val harvest；再由独立
审计确认：IoU 合成测试、分片索引/shape/scene_id、完整契约 hash、train/val 后才打开 test、
以及 correct/sham 容量与标签隔离均通过。届时才可将 verdict 改为 PASS 并运行一次
`evaluate --allow-test`。

---

## R1a 修复复审（2026-10-12，仍未开 test）

**VERDICT: PASS — 允许以新的 output/cache 目录运行 R1a `harvest`；在该 harvest 完成后，
允许一次 `evaluate --allow-test`。**

本节仅撤销上述两个阻断项；不将 P1 结果预先判为 PASS，也不改变仍建议补强的分片结构校验。
本复审没有执行模型前向、dataset 读取或带 `--allow-test` 的命令。

### A. IoU dtype 修复：通过

- `iiou` 现将 prediction 与 strict-GT masks 转为 `np.int32` 再相乘（runner 第 76 行）。
  1024² mask 的最大交集为 1,048,576，位于 int32 安全范围内；面积和、并集也远小于
  int32 上限。
- 用 1×1,048,576 的全 1 `int32` mask 作独立内存复现，得到 dtype `int32`、交集
  `1,048,576`、IoU `1.0`，不再是旧版 uint8 的模 256 值。
- 该修复只改变 `qi/y_iou` 的数值正确性，不改变 primary F1、error AUROC、三臂容量、
  C/threshold 选择或 bootstrap/P1_PASS 不等式。R1a 文档也明确此前 cache 不得复用，
  因而不会把旧的错误 IoU target 混入新的训练/验证读取。

### B. 完整实现指纹及 fail-closed contract：通过

- `implementation_fingerprints()` 现在指纹 runner、SurfaceBEV rasterizer、raw-pair
  reader、H0 helper 和 checkpoint inference helper（第 38–40 行）；五个文件均存在，且
  relative path 相对 `portable_sam2_explicit_coarse` 可解析。
- `cache_contract()` 不再跳过 source 字段，而是要求落盘合同与本次完整 dict 严格相等
  （第 43–46 行）。因此其中任一列出的实现文件、B0/H0 SHA、height 边界、split、seed、
  arms、维度或容量改变，都会在恢复或 evaluate 前 fail-closed。
- R1a 预注册（新增节）如实登记这两个修复、明确旧 cache 不可复用，并仍要求新目录重跑
  train/val 后才可开 test；这与代码行为一致。

### 复核其余关键门：保持通过

- harvest 仍在第 91 行拒绝 `--allow-test`，只构造 train/val；evaluate 仍先完成 train/val
  cache、所有 val C/threshold 拟合和选择，随后才写 test lock 并构造 test dataset
  （第 148–156 行）。
- `correct`/`sham90` 仍共用同一 UAV descriptor、depth、相机和固定投影容量；标签仍只在
  两个 SurfaceBEV 完成后进入 readout target。缓存切分没有改变这些调用或随机抽样 seed。
- R1a 没有改动预注册六个 paired-bootstrap 差值、10,000 draws、seed 或所有 CI 下界大于零
  的 P1_PASS 门。

### 运行条件与遗留建议

1. **必须使用从未写入过 R1/R1a 分片的新 output/cache 目录。** 不得删除、覆盖或复用旧
   cache 来绕过合同不匹配；旧目录保留作失败执行审计证据。
2. 完成 harvest 后，启动 evaluate 前应确认 `train_val_complete.json` 存在、cache contract
   等于当前指纹、且 train/val 分片数为 694/100。原子分片足以处理普通中断；此前提出的
   逐片 key/shape/scene_id 重验仍是推荐的韧性补强，但不再构成本次 test 前的阻断项。
3. test 打开后仅允许同一 lock 下恢复未完成 test 分片；任何实现、合同或选择值变化均须
   停止，不得重新开封或调参。

---

## R1b `run_p1_supervised.py` 外层监督器复审（2026-10-13，未开 test）

**VERDICT: FAIL — 该 supervisor 不得用于 `evaluate --allow-test`，直至其自身被纳入
不可变实现合同，或改为不 monkey-patch runner 的纯进程/日志包装器。**

审计对象为 `scripts/ue/run_p1_supervised.py`。未运行 worker、P1 runner、dataset 或任何
带 `--allow-test` 的命令。

### 已核实：参数、调用顺序与返回/异常传播（通过）

- 预期外层调用形式为：`run_p1_supervised.py <supervisor-flag> <status-output-dir>
  <原始 P1 参数...>`。supervisor 将 `sys.argv[3:]` 原样传给 child；worker 将其复原为
  `probe_surfacebev_p1.py` 的 `sys.argv`（supervisor 第 33–37 行、worker 第 30–31 行）。
  因而 `--phase evaluate --output-dir ... --allow-test` 等 runner 参数没有被删改。
- 每个 wrapper 仅在函数调用前后打印时间戳；它以默认实参绑定原函数、传回原函数的返回值，
  不捕获或吞掉异常（第 18–29 行）。worker 的非零退出码被 supervisor 写入 runtime 记录并
  以同一正数（或 signal 映射）退出（第 39–47 行）。
- 由于 wrapper 没有重排 `run_eval` 内部调用，未在其自身引入提前 test 读取：仍是 runner 的
  `cache_contract` → train/val `fit`/val selection → `harvest_test` 顺序。对 `harvest_test`
  的日志包装不会提前调用它。
- 所谓 `start_new_session=True`、`stdin=DEVNULL`、stdout 重定向到日志文件并不在该脚本的
  内层 `Popen` 中；若由外层启动 supervisor 时提供，child 会继承该会话与 file descriptors。
  这不是协议语义变化，但启动命令应被归档，不能把这些能力误归因于第 37 行的 Popen。

### 阻断：supervisor 改写了 runner 函数，却未被 cache contract 指纹覆盖

R1a 的严格合同只指纹 P1 runner、SurfaceBEV、reader、H0 helper 和 inference helper。
supervisor 在运行时以 `setattr(p1, name, logged)` 替换 `preflight`、`cache_contract`、
`trainxy`、`fit`、`evaluate`、`harvest_test` 与 `run_eval`（第 18–29 行），即它不是一个
纯粹的 stdout/进程包装器，而是直接介入了受审计的 protocol control-flow。可是其自身 SHA
不在 P1 `implementation_fingerprints()` 或任何独立 fail-closed sidecar contract 中。

后续修改 supervisor 可静默改变参数、异常、选择或 test 调用时机，而既有 P1 cache 仍会通过
R1a 合同；这不满足“cache SHA 严格检查继续有效”的要求。因此即使当前代码表面上只记录日志，
也不能用于开封 test。

**修复任选其一：**

1. 最稳妥：删去 monkey-patch，supervisor 只以子进程方式运行未改写的 runner，并仅通过
   stdout/stderr 与 PID/exit-code 记录进度。这样它不属于 P1 特征/评估语义，R1a runner
   contract 保持充分。
2. 若必须 monkey-patch：将该文件 SHA 纳入 runner 的 `implementation_fingerprints()`，并以
   新的未污染 cache/output 重新 harvest；或在**harvest 与 evaluate 都通过同一 supervisor**
   时写入并严格比较一个包含 supervisor SHA 的独立 sidecar contract。任一实现变化均应在
   test 前 fail-closed。

### 非阻断启动韧性问题

- `status-output-dir` 与传给 runner 的 `--output-dir` 没有一致性校验；它们不同会让
  `evaluation_runtime.json` 出现在不含 cache/summary 的目录，造成错误的运维判断。建议
  wrapper 显式解析并要求二者相同，且在启动前确认状态目录存在。
- supervisor 在第一次 `status` 写入失败时会异常退出，而 child 继续运行，反而失去监督；
  应在 spawn 前检查/创建状态目录，并把写状态失败作为可观察的 fail-closed 终止策略。
- 外层 detach 启动命令须记录 `argv`、启动时刻、supervisor/worker PID 和 stdout/stderr 的
  绝对日志路径。否则“脱离会话后仍在运行”的证据不可复核。

在完成上述阻断修复并进行一次针对 supervisor 的复审前，R1a 的“允许 evaluate”放行仅适用于
直接调用经过指纹审计的 P1 runner，不适用于当前 monkey-patch supervisor。

---

## R1c 纯进程 supervisor 修复复审（2026-10-13，未启动评估）

**VERDICT: PASS — 允许使用当前 `run_p1_supervised.py` 启动一次 P1 `evaluate --allow-test`。**

R1b 的 monkey-patch 阻断已移除：现行 supervisor 不 import P1 模块、不定义 worker、不
`setattr`、不替换 runner 任何函数。它仅以 `subprocess.Popen([python, -u,
probe_surfacebev_p1.py, *原始参数])` 启动原始 runner（第 17–26 行），所以 P1 的
implementation fingerprints 与 cache contract 仍由未修改的 runner 自身完整执行；supervisor
不再是 feature、标签、选择、test 开封或统计语义的一部分。

### 复审通过项

- 参数透传：`arguments = sys.argv[3:]` 被原样附在 runner argv 后；未添加、删除或修改
  `--phase`、`--allow-test` 等 P1 参数。
- 输出绑定：启动前要求 wrapper 指定的状态目录与 runner `--output-dir` 完全相同，否则
  fail-closed；之后才创建目录及状态文件（第 14–23 行）。
- 运行状态：启动前记录 argv 与 started_at；每十秒记录 supervisor/worker PID、退出码、
  test lock 是否存在以及已落盘 test `.npz` 数。`glob` 仅读目录项，不调用 dataset 或读取
  test scene 内容，故不构成提前开封。
- 状态写失败：捕获后 `terminate()` 并 `wait()` child，再抛错，不会留下无人监管的 runner
  （第 34–39 行）。非零 worker 退出码继续由 supervisor 原样（signal 规范映射）传出。
- `py_compile` 通过；本复审未启动 P1、未读取 dataset，未运行任何 `--allow-test` 命令。

外层负责的 `start_new_session=True`、`stdin=DEVNULL`、stdout/stderr 日志重定向仍应在实际
启动 supervisor 的 Popen 中显式登记；它们不是本 wrapper 第 26 行的职责。只要外层不改写
传给该 wrapper 的 argv，这种会话脱离不改变 P1 协议。
