# MHBEV-ROI-20260914 — A1 frozen-probe implementation review

## A1-Frozen-R2 正式产物最终复核（2026-09-14）

**复核对象。**
`outputs/MHBEV-ROI-20260914/A1-Frozen-R2-fullval/a1_frozen_probe.json`，按
`EXPERIMENT_GOVERNANCE.md` 的 A1-Frozen-R2 预注册门执行。

**结果。** 工件完整覆盖 259 个 val scene（259 parity、259 calibration，三种
correct/XY+80m/scene-permuted telemetry 各 259）。259/259 zero-gate prediction
parity 相等；三 sham 非零、候选数相同、hard-mask 为零和 tile-centre 五视角门均通过。
FP64 W2C 最大自洽残差为 **1.877197064e-09 cm**，小于 `1e-6 cm`；共同有效点最大
FP32–FP64 UV 差为 **0.000249597 px**，小于 `0.5 px`。

R2 边界审计也通过：共有 85 个 view-query 的 FP32/FP64 raw valid 差异，但
`valid_mismatch_outside_boundary=0`、`interior_valid_equal` 在 259/259 场均为真；
即每个差异都在预注册的 `<=0.5 px` 四边 boundary band 内。所有 P2 query 的
`>=2-view` support mask 均逐位一致（`support_false=0`），因此不存在 boundary
舍入传播为实际 candidate hard-mask 决策变化的 failure。边界带总计 222,222 个
view-query，mismatch 数量已随工件归档，只作数值边界行为证据，不作语义增益证据。

**VERDICT: PASS — 冻结门通过，可放行 A1 热启动筛查。** 这只授权 Appendix A
规定的下一阶段：冻结单流基座、仅训练新增 `drone_guidance` 模块、按两个 seed 和
last-5-epoch 指标门筛查。它**不**授权全量训练；全量仍须在热启动门通过后，履行
commit / EXP-ID / 全臂 harness 等独立准入要求。

---

## A1-Frozen-R2 fixed-smoke 复核（2026-09-14）

**复核对象。** 修正后的 `probe_a1_frozen.py` 与
`outputs/MHBEV-ROI-20260914/A1-Frozen-R2-smoke-fixed/a1_frozen_probe.json`。

**实现：通过。** boundary distance 现在严格实现为
`min(abs(u), abs(u-512), abs(v), abs(v-512))`，并以 FP32 或 FP64 的该距离
`<=0.5px` 定义 band。这与 R2 的“到任意画面边缘的距离”一致：远离画面的外部点
不再因带符号坐标为负而被错误豁免。该改动只用于 calibration/audit；`project()`
的 FP32 valid 逻辑、P2 查询、采样、support/hard-mask 与 detector 接线路径均未变。
脚本语法检查通过。

**fixed smoke：通过。** 两场中 `boundary_band_view_queries=858`（先前错误谓词为
55,621），没有 valid mismatch，也没有 band 外 mismatch；interior valid 和全图
two-view support 均逐位一致。全部既有门及 `fp32_fp64_projection_r2` 为 `true`，
并继续满足 zero-gate parity、三 sham/hard-mask、FP64 W2C 与 tile-centre 门。

**VERDICT: 可启动 R2 完整 259 场冻结探针；仍不可启动热启动。** fixed smoke 已
消除了此前 boundary predicate 的实现阻塞，且没有改变已预注册的门。正式产物仍需
在 259 场上重跑并由独立复核确认所有 R2/既有 gate 通过后，才可判定热启动放行。

---

## A1-Frozen-R2 预注册、实现与 smoke 复核（2026-09-14）

**复核对象。** `EXPERIMENT_GOVERNANCE.md` 的 A1-Frozen-R2 条目、
`portable_sam2_explicit_coarse/scripts/ue/probe_a1_frozen.py`，以及
`outputs/MHBEV-ROI-20260914/A1-Frozen-R2-smoke/a1_frozen_probe.json`。

**预注册与部署隔离：原则上正确。** R2 事先明确保留运行时 FP32 `project()`、360-m
P2 点、采样、`>=2` support/hard mask 和模型；FP64 继续只生成 audit 字段。R2 的
five-part gate（边界 band 外 valid 相等、全部 P2 support 相等、所有 mismatch 在
band 内、共同有效 UV `<=0.5px`、以及既有 W2C/parity/sham/mask 门）也都被写入，
没有以 R1 的失败回改部署路径。两场 smoke 的现有门和 R2 汇总门均显示 `true`。

**阻塞实现问题。** boundary-band 的实现不是预注册所称“投影到任意画面边缘的
距离 `<=0.5px`”。代码以
`min(u, 512-u, v, 512-v) <= 0.5` 判断；该量在任一画面外的点为负数，故例如
`u=-100` 或 `u=700` 会被当作 band 内，虽然其到最近边缘的距离是 100/188 px。
结果 smoke 已报告每景 `55,621` 个 band view-query，且该谓词会把全部远离画面的
无效区纳入豁免范围。这会削弱“所有 valid mismatch 只能发生在 0.5px 边界带”的
R2 门；正确的距离应针对四条边使用绝对距离（例如
`min(abs(u), abs(u-512), abs(v), abs(v-512))`），再与 0.5 比较。

`two_view_support_equal` 的全图逐位比较、`valid_mismatch_outside_boundary` 与
共同有效 UV 的记录形式本身是合适的；问题仅在 boundary membership 谓词过宽。因为
smoke 没有 valid mismatch，它不能暴露该实现偏差，不能作为正式运行前的豁免证据。

**VERDICT: HOLD / 暂不可启动 R2 正式 259 场探针。** 先修正 boundary-distance
谓词，使其精确实现已预注册的 `<=0.5px` 定义，并以同一两场 smoke 重跑确认；这属
实现纠正而非更改门条件。之后再由独立复核确认，方可启动 259 场。仍不可启动热启动。

---

## A1-Frozen-R1 正式产物复核（2026-09-14）

**复核对象。**
`outputs/MHBEV-ROI-20260914/A1-Frozen-R1-fullval/a1_frozen_probe.json`。

**通过部分。** 工件完整覆盖 259 个 scene：parity records、calibration records 与
三种 telemetry 均为 259；259/259 prediction parity 通过。原有 bypass、三条件
nonzero、candidate count equality、hard mask 与 tile-center 五视角门均为 `true`。
R1 的 FP64 W2C 自洽门也通过，最大残差为 **1.877197064e-09 cm**（小于
`1e-6 cm`）；共同有效点最大 FP32–FP64 像素差为 **0.000249597 px**（小于
`0.5 px`）。

**失败部分（决定性）。** `fp32_fp64_projection=false`。R1 预注册要求对同一 FP32
P2 点的 valid mask **完全相同**，而正式 JSON 中 `fp32_fp64_valid_equal=false` 有
**85/259 场**（本审查以 JSON 的 259 条 calibration record 结构化计数；与口头所称
89 场不一致，但不影响失败判定）。这些是临界像素的 valid/in-frame 判定差异；很小的
共同有效点 UV 差不可以替代或豁免“mask exact equality”条件。

**VERDICT: HOLD / 不放行 A1 热启动。** R1 条目写明 valid-mask 完全相同、任一门
失败即停止；故本正式结果必须按失败归档，不能以 `max_uv <0.5px` 事后放宽解释。
若继续该方向，应另行诊断边界临界点与运行时 FP32 valid 判定，并在任何重跑前新建、
明确预注册的审计/容差规则；不得改写当前 R1 门或将本工件视为通过。

---

## A1-Frozen-R1 预注册与 smoke 复核（2026-09-14）

**复核对象。** `EXPERIMENT_GOVERNANCE.md` 的 A1-Frozen-R1 条目、修订后的
`portable_sam2_explicit_coarse/scripts/ue/probe_a1_frozen.py`，及
`outputs/MHBEV-ROI-20260914/A1-Frozen-R1-smoke/a1_frozen_probe.json`。

**预注册与运行时契约：通过。** R1 明确作为重新立项的数值标定审计，只替换审计
精度；运行时 K/E、360-m 映射、由 `GeometrySparseBEV._ground_points()` 生成的
FP32 P2 点、`project()` 的 FP32 计算与 autocast-disabled 均未改动。`project_fp64()`
没有进入 feature sampling、candidate、zero-gate 或 detector forward，只产生
calibration 字段；因此 FP64 是 audit-only，不会悄然改变部署路径。R1 也在运行前
写死了三项新条件：`E64 @ inv(E64)` 最大残差 `<1e-6 cm`、同一 FP32 查询点的
valid mask 完全一致、共同有效点最大投影差 `<=0.5 px`，并保留完整 259 场
parity/sham/hard-mask 的重跑要求。

**两场 smoke：通过。** `002002` 与 `002003` 的所有既有门和新增门均为 `true`：
逐预测 parity、三种条件非零/候选数一致、hard mask、tile-center 五视角、FP64 W2C
自洽及 FP32–FP64 投影一致性。新 W2C 残差为 `4.37e-11` 与 `8.55e-11 cm`；共同
有效点最大像素差约 `1.92e-4` / `1.88e-4 px`，远低于 0.5 px。此前 FP32 inverse
残差超门可被归因于在大尺度 cm 坐标上进行 FP32 逆运算的审计数值消减，而非 R1
运行时投影被提升为 FP64。

**限定。** `E64` 是将 dataset 已加载的 FP32 K/E 转为 double 后审计，故它验证
“部署所用 FP32 矩阵的 FP64 自洽性”，并不额外验证原始 JSON 数值在未量化前的
精度；这与本轮“不得改变运行时 FP32 契约”的目标一致。若未来要主张原始 metadata
精度，需另立 raw-JSON FP64 数据审计，不能混入本门。

**VERDICT: 可启动 R1 正式 259 场冻结探针；尚不可启动热启动。** R1 的新阈值、
条件和 smoke 已预注册并通过，正式运行是其预注册的下一步。正式 JSON 必须再次
满足全 259 场的所有旧门和两项新 FP64 门，并经独立结果复核后，才能决定 A1
热启动是否放行。

---

## 正式产物复核（A1-Frozen-fullval；2026-09-14）

**复核对象。**
`portable_sam2_explicit_coarse/outputs/MHBEV-ROI-20260914/A1-Frozen-fullval/a1_frozen_probe.json`。

**覆盖与已通过门。** 正式 JSON 的 `scene_ids`、parity records 与三组 telemetry
均覆盖 259 个 val 场景；259/259 个 `prediction_equal=true`，因此全 val 的
`a0_bypass_bitwise=true`。三种 correct / XY+80m / scene-permuted 条件均报告非零
feature、统一 65,536 candidate、`hard_mask_zero=true`；正式 gate 中
`all_conditions_feature_nonzero`、`candidate_counts_equal`、`hard_mask_zero` 与
`tile_center_five_view` 也均为 `true`。这些项目通过，且 scene-permuted / shifted
sham 均未进入 detector，未发现由 proposal 数变化造成的假象。

**失败门（决定性）。** `w2c_camera_center=false`。预注册/实现采用的阈值是最大
残差 `<0.1 cm`，而最大值是 scene `002795` 的 **1.51953125 cm**。这不是唯一的
超阈场景：`002004` (0.125)、`002005` (0.1875)、`002809` (0.10546875)、`003010`
(0.517578125) 也超过 0.1 cm。尽管绝对量相对于 360 m tile 很小，事后不能以
“数值误差”放宽已写入的门；且 Appendix A.4 明确规定任何 W2C 标定失败即停止。

**VERDICT: HOLD / 不放行热启动。** 正式全 val 已消除了先前仅两场的覆盖不足，
但 W2C 机械门失败足以阻止 A1→热启动。应先将该负结果归档并单独诊断这五场的
浮点求逆/metadata 精度与可接受标定容差；如要改变阈值或计算口径，必须在新的
修订立项/探针前预注册，而不能修改此轮结果的解释。

---

## 修订复核（A1-Frozen-v2-smoke；2026-09-14）

**复核对象。** 修订后的
`portable_sam2_explicit_coarse/scripts/ue/probe_a1_frozen.py`，以及 GPU2 两场产物
`portable_sam2_explicit_coarse/outputs/MHBEV-ROI-20260914/A1-Frozen-v2-smoke/a1_frozen_probe.json`。

**Smoke 机械门：通过（仅限所运行的两场）。** 产物对 `002002`、`002003` 的
`a0_bypass_bitwise`、`all_conditions_feature_nonzero`、`candidate_counts_equal`、
`hard_mask_zero`、`w2c_camera_center`、`tile_center_five_view` 均为 `true`。两场的
四个预测字段（bbox、score、label、mask）逐字节哈希相等；P2 候选残差非零（均值约
0.0112）而 `gamma_p2=0.0`，故真实 hook 内的 `P2 + gamma * residual` 仍严格恒等。
这足以证明该 smoke 的零门旁路没有污染 detector。

**对原阻塞项的复核。**

- `encode_p2()` 现已按线上 `DualStreamRSPrompterAnchor.extract_feat()` 使用 RGB
  `[0,1]` 的 ImageNet mean/std 标准化，再调用 frozen UAV encoder；原先输入分布
  不一致的问题已修复。
- `a1_candidate()` 以 `valid.sum(0) >= 2` 形成实际 support mask，平均多视角特征后
  乘入该 mask；每个 correct、XY+80m 和 scene-permuted 条件都断言 `<2` view 的
  P2 候选严格为零。由于全无效是 `<2` 的子集，该实现也满足其 residual 为零的
  代数要求。
- hook 现在构造 256×256 的 P2 candidate、显式通道适配并在 neck 输出的 P2 上做
  零标量 gate 接入；shape 不符或零门非逐位恒等会直接报错。这是有效的 A1
  zero-gate wiring smoke（仍是探针实现，不替代后续训练臂的正式模块接线审计）。
- 每景已硬校验 K/E 为 `(5,3,3)/(5,4,4)`、UAV FPN 为 5×256 通道，并要求 tile
  center 的五路投影均有限且入框。产物中中心均约为 `(256,256)`；W2C camera-center
  最大残差为 0.0391 / 0.0313 cm，低于预设 0.1 cm 门。
- 三种条件的候选数均为 65,536，且都有非零 masked feature。scene-permuted 使用
  下一场 UAV map、保留当前 K/E/查询点；XY+80m 仅改 world-X。两种 sham 不会改变
  satellite/detector 输入。shift 的有效覆盖较低是预期几何后果，不是候选数变化。

**授权结论：仍不可授权 A1 热启动。** 该产物仅覆盖 2/259 val scenes；虽然
`--parity-scenes=0` 在本次等价于覆盖全部 `--scenes=2`，但 Appendix A.4 的正式
A0/A1 frozen gate 需要完整 UE val 的逐预测 bypass parity，且几何/掩蔽/sham 门也应
在预注册的完整范围上落判。因此必须在 GPU2 运行 `--scenes 259 --parity-scenes 0`
并归档完整 JSON，随后再做一次针对正式产物的复核，方可决定是否授权热启动。

---

**Scope.** Independent review of `portable_sam2_explicit_coarse/scripts/ue/probe_a1_frozen.py` against Appendix A of `EXPERIMENT_GOVERNANCE.md`. Reviewed without changing the probe/model code. Syntax check passed with `/data2/wangcheng/envs/cvt2/bin/python -m py_compile`.

## VERDICT: HOLD — do not treat the frozen gate as passed or start A1 hot-start

The script has a sensible read-only skeleton, but it does not yet verify several pre-registered invariants. The issues below must be fixed and the probe rerun before a gate decision.

## What is sound

- The detector is rebuilt strictly from the anchor checkpoint, run in `eval()`/`inference_mode()`, and the only detector-side intervention is a forward hook returning `None`. It compares hashes for bboxes, scores, labels, and masks. Thus, **for the sampled scenes**, it is a valid passive-observation parity check; no UAV tensor is supplied to detector `predict`.
- The world grid reuses `GeometrySparseBEV._ground_points`, retaining the 360-m satellite affine and centimetre conversion. Projection explicitly uses FP32 with autocast disabled and `E @ [camera_center,1] = [0,0,0,1]`, consistent with W2C column-vector metadata.
- Both shams leave the satellite input and detector call unchanged. `XY+80m` shifts only queried world X; scene permutation keeps current K/E and query coordinates while taking the next scene's frozen UAV map. Candidate cardinality is recorded as the same 256² in all conditions.

## Blocking findings

1. **Online UAV preprocessing is not reproduced (P0).** `encode_p2()` calls the UAV SAM encoder directly on RGB `[0,1]` data (lines 135–143, 191), whereas the deployed dual-stream path normalizes using ImageNet mean/std before that encoder (`rsprompter/models.py:288–291`). Consequently all feature magnitude/cosine/nonzero telemetry is from a different frozen encoder input distribution. Apply exactly the deployed normalization before every frozen UAV encode, then rerun.

2. **The required two-view hard mask is never applied or tested (P0).** The probe computes `valid.sum(0) >= 2` only as a ratio (lines 218/239). It samples features at all positions, including 0/1-view positions, and has neither a candidate feature mask nor an asserted zero residual for invalid positions. Appendix A.2 says fewer than two valid views must be hard-masked and all-invalid UAV residual must be strictly zero; Appendix A.4 requires stopping on mask failure. Implement the actual mask in the sampled candidate tensor and assert: (a) every `<2` candidate is zero after masking, (b) all-invalid scene/condition residual is zero, and (c) the telemetry denominator is explicit.

3. **The implemented parity is only a no-op hook, not an A1 zero-gate integration test (P0).** A hook that returns `None` can establish that observing `detector.neck` does not alter outputs, but it cannot establish that the proposed P2 residual module, its gate, and its branch wiring preserve anchor behavior at `gamma=0`. The script never instantiates an A1 branch/gate and does not log `gamma_p2` or residual/base norm. Before opening the A1 training gate, add a frozen A1 module/path with a zero-initialized injection and compare it against A0. If implementation is deliberately deferred, label this artifact “geometry/passive-hook diagnostic”, not “A1 frozen gate”.

4. **Parity coverage is insufficient for the stated A0 gate (P1).** Default parity is four scenes (`--parity-scenes=4`), while the pre-registered gate says A0 bypass must agree with the anchor per prediction and the data contract is full UE val (259 scenes). Run the bitwise field parity on all 259 val images (the geometry/content diagnostic may stay at a fixed prefix if runtime requires it), record the list/count, and fail on any mismatch. Also make an empty `parity` list fail rather than allowing `all([])==True`.

5. **Required gate results for both sham conditions are absent (P1).** The summary only gates `correct_feature_nonzero` (lines 255–258). Appendix A.4 now requires nonzero candidate UAV feature in *all three* conditions. Add per-condition `all(feature_abs_mean>0)` gates plus a candidate count equality assertion across correct/shift/permuted. Current same-count records are useful evidence but not an enforced gate.

6. **Calibration validation is incomplete (P1).** W2C camera-centre residual is good, but no assert checks `K==(5,3,3)`, `E==(5,4,4)`, five maps/views, or that tile-centre projections are valid/finite. `tile_center_valid_views` is merely logged. Validate those shapes and explicitly assert finite, positive-depth, in-frame tile-centre results according to the existing `diagnose_multihigh_geometry.py` convention; record the exact accepted criterion. This is needed to make the 360-m mapping gate auditable.

## Non-blocking notes

- The shifted sham may legitimately have different valid/two-view coverage near tile borders. This does not violate “same candidate count”; do not force equal coverage. Report it so affinity statistics are not overinterpreted.
- Pairwise cosine is appropriately marked descriptive-only. It is not a valid frozen semantic-effectiveness criterion, particularly across oblique UAV views; it must not be used to authorize training.
- `passive_p2_hash` casts to float before hashing, so it is not a raw-bit P2 hash. Output instance-field hashes are the actual parity evidence; either rename this field or hash native contiguous bytes.
- Fix the `all([])` vacuous pass in every gate and include the immutable checkpoint SHA/config fingerprint in the artifact for archival traceability.

## Gate disposition

**A1 frozen probe: not passed.** No training/hot-start authorization follows from the current script. After the six blocking items are addressed, rerun the corrected probe on GPU2, archive its JSON outside git, and request a fresh review of the run artifact before deciding the Appendix A.4 gate.
