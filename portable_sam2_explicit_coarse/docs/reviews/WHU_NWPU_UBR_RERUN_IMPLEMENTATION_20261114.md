# WHU/NWPU 单流重跑（UDPR→native UBR）实施+接线独立审计 2026-11-14

- 审计对象：分支 `machine2/fast-whu150`，commit `6b0d4684`（协议预登记）与
  `96b194b6`（native UBR 移植与接线）。审计时工作树干净（仅未跟踪 `.worktrees/`）。
- 关联：EXP-0046（WHU fast150）/ EXP-0047（NWPU matrix300），预登记
  `docs/evidence/whu_nwpu_ubr_rerun_protocol_20261114.md`。
- 方式：只读静态审查 + CPU 实测探针（一次性临时脚本，放在仓库外的 /tmp，审计
  结束已删除，未留存；GPU 全被占用故未做 GPU/AMP/DDP 实跑）。探针内容：两个 delta
  config 各自 CPU build → 记录 `_mask_forward` 输出 → 安装 → 比对恒等、键布局、
  优化器分组、aux-clip 分组、strict round-trip、`_empty_rank_losses`。路径均相对
  `portable_sam2_explicit_coarse/`。
- 未覆盖（如实声明）：4-rank DDP smoke、AMP 下真实 GradScaler 步、checkpoint→
  推理 CLI 全链路 round-trip 均**未实跑**，只做了代码路径论证与 CPU 等价验证。

## 复审记录（第 2 轮，commit 483957f9）

初审发现 F1（阻断）与 C1/C3 后，实施者提交 `483957f9` 修复。复审于同日完成，
逐项复验（含运行验证）：

- **F1 已修复并复验通过**。`rsprompter/native_ubr_integration.py:68-78` 恢复双流
  源侧（abc51d74）语义：`losses, telemetry = super()._empty_rank_losses(x)`，仅在
  `losses["loss_mask"]` 上加 refiner 零图项，返回二元组（与源侧逐行等价，仅少了
  不适用的 `dual_target` 分支）；模块头注 `:22-25` 改为 tuple 契约描述。实测
  （两个 delta config）：`_empty_rank_losses` 返回 `(dict, dict)`，`loss_mask`
  可 backward 且有限，refiner 22/22 参数梯度物化（全零、有限）；调用方
  `anchor_roi_head.py:159-160` 的解包路径吻合。回归测试
  `test_empty_rank_losses_tuple_contract_and_refiner_graph`（真实 build + install
  路径）加入后旧代码形态会直接 TypeError，测试有效。
- **C1 已修复并复验通过**。`train/optimizer.py:59-65` 在 IoU 头组匹配中补充
  shimmed 路径 `roi_head.mask_head.mask_decoder.base.decoder.iou_prediction_head`。
  实测（两个 delta config、wrapper 实际 lr-mult 组合）：安装后 `iou_head` 组仍为
  132,612 参数，安装前后除 refiner（+6,625 落 `other` 组，lr 5e-4/wd 0.05）外
  **无任何参数换组**（逐参数 id 比对 MOVED=NONE）；未安装模型不受影响
  （`mask_decoder.base.` 命名只可能来自 shim）。原“日后调
  `--mask-decoder-lr-mult` 连带改 IoU 头”的隐患消除。
- **C3 部分修复**：新增 `test_shim_context_guards`（NativeMaskHead 无 shim /
  shim 无 call_context 均拒）与 F1 回归测试；单测 12/12 实跑通过。仍缺（非阻断，
  见第 7 项）：模型级安装恒等前向、strict round-trip、推理 marker 真值表的仓库
  内测试。
- 修复 commit 只触及上述 3 个文件；`native_ubr_refiner.py` sha256 复核仍为
  `aa057e3e…`（与 ed8bd4c6 源一致）；安装后恒等（两 config `torch.equal`）、
  strict round-trip 复测通过。
- 残留（化妆性，非阻断）：宿主 `anchor_roi_head.py:219` 的 `-> dict` 陈旧注解
  仍未改（修复 commit 只在移植侧注明）；建议顺手更正但不影响运行。

**复审结论：无新增阻断项。总 VERDICT 改判为 PASS（见文末 GPU smoke 补记中的
待完成事项——WHU fp32 GPU smoke 与 round-trip 按用户裁定延后，不作为代码阻断）。**

## 总 VERDICT: PASS

（初审 2026-11-14 为 FAIL，阻断项 F1；已由 commit `483957f9` 修复并经本轮复审
复验通过后改判。）

### F1（已修复，commit 483957f9）空正样本 rank 上 `NativeRoIHead._empty_rank_losses` 必崩

- `rsprompter/detection/anchor_roi_head.py:219` 的签名注解写 `-> dict`，但函数体
  实际返回二元组：`:252` `return losses, telemetry`；调用方 `:160`
  `return self._empty_rank_losses(x)` 经 `mask_loss` 被按
  `mask_losses, mask_telemetry` 解包（`:384-386` 附近 `losses.update(mask_losses)` /
  `self._loss_telemetry.update(mask_telemetry)`）。
- 移植代码 `rsprompter/native_ubr_integration.py:66-73` 把返回值当 dict：
  `for key in list(losses)` 迭代到的是两个 dict 本身，`losses[key]` 即
  `tuple[dict]`。模块头注 `:22-23`“returns a plain dict in this package”与事实不符
  （被错误的类型注解误导）；源侧 `ubr_native_integration.py`（abc51d74）原本就是
  `losses,telemetry=super()._empty_rank_losses(x)` 再 `return losses,telemetry`，
  new 侧宿主契约与源侧相同，这处“适配”是多余且错误的。
- 实测复现（两个 delta config 各一次，CPU build + install 后直接调用）：
  `TypeError: tuple indices must be integers or slices, not dict`
  （`native_ubr_integration.py:71`）。
- 影响：任一 rank 的 micro-batch 出现 0 个正样本 proposal（`anchor_roi_head.py:159`）
  即整个 DDP 训练崩溃。WHU 为 4 rank × batch 1，单图无正样本即触发；触发频率我
  没有量化（未查数据集是否过滤空标注图），但这是 150/300 epoch 全量 run 中的潜在
  中途崩溃点，且现有 10 个单测全绿并未覆盖它。
- 修复方向（供实施者，不在本审计内改）：按源侧写法解包
  `losses, telemetry = super()._empty_rank_losses(x)`，只给 `losses["loss_mask"]`
  加零图项后 `return losses, telemetry`；并加一条在真实 build 的模型上调用
  `_empty_rank_losses` 的测试（断言返回二元组、`loss_mask` 可 backward 且
  refiner 参数拿到零梯度）。注：训练器 DDP 用 `find_unused_parameters=True`
  （`train/train_rsprompter_fusion.py:222-229`），所以零图项本身只是防御性的，
  但覆盖方法不能崩。
- **修复状态：已由 commit 483957f9 按上述方向修复**（详见文首“复审记录”），本审
  计复验通过：两 config 实测返回 `(dict, dict)`、`loss_mask` backward 成功、
  refiner 22/22 参数梯度物化且有限；回归测试入仓（12/12 绿）。

## 逐项结论

### 1. 移植保真 — PASS

- **refiner 字节一致**：`rsprompter/native_ubr_refiner.py` sha256
  `aa057e3edb7c2bc1a9c82501bf30bf192e662ed4832742cbc4ad8c76ca627170`，与
  `git show ed8bd4c6:portable_sam2_explicit_coarse/rsprompter/uav/native_roi_refiner.py`
  的 sha256 完全相同（自行计算）。
- **integration 差异逐处核对**（对 abc51d74 版 `diff -u`）：
  - 删除 `native_model_config`（512 mask_size 硬断言）、`NativeEvidenceG`、
    `NativeDualMaskHeadMixin`、`sample_roi_evidence` 调用、`detach_*` /
    `dual_target` / `posthoc_evidence_grad` / telemetry 分支：均为 UAV 证据与
    双流专属路径，预登记“不移植”清单已列。
  - 保留且语义不变：act2 hook 挂在 `self.base.decoder.output_upscaling[-1]`
    （`:109`）、`len(result) != 3` 断言（`:112`）、lattice 形状校验（`:117`）、
    `head.native_refiner(feature, logits, None, None)`（`:130`）。源侧
    `zero_evidence=True` 且各 detach 旗标为默认 False 时走的正是
    `native_refiner(feature, logits, rgb=None, geometry=None)`，等价。
  - 形状校验由 `len(boxes)` 改为 `n = logits.shape[0]`，并新增
    `len(boxes) != n` 检查（`:121`）：`multimask_output=False` 下两者相等
    （实测两 config 均为 False），不放宽约束。
  - 源侧 `boxes is None or image_ids is None` 硬报错被放宽为可空：零证据形态不
    消费 boxes/ids，且唯一调用点恒传二者（见第 2 项），无语义风险。
- **`capture_upscaled_embedding=False` 属性**（`:92`）：唯一读取者是
  `verify_ubr_ready`（`:159`）与宿主 `mask_head.py:258`（构造期写入 wrapper 的
  kwargs，不读 shim）。安装后宿主 forward 以 `self.decoder_tail_enabled`
  （`mask_head.py:411`）而非该属性决定 3/4 元组解包，故属性不参与数值路径。
  无风险。
- **`verify_ubr_ready` 顺序改动**（`:152` 先判 Already installed）：只改变重复
  安装时抛哪条错，四个条件仍全部在触碰模型前检查。无风险。
- **安装器 RNG**：`fork_rng` + `manual_seed(2080)`；非 CUDA 时 `devices=[]`
  （`:173`）。实测安装前后全局 RNG 状态逐位相同，6625 参数、22 个 state 键。

### 2. 宿主接口适配 — PASS（F1 已于 483957f9 修复并复验）

- **call_context**：`anchor_roi_head.py:131-141` 以关键字传
  `roi_img_ids=rois[:, 0]`、`boxes=rois[:, 1:]`，`NativeMaskHead.forward`
  `kwargs.get(...)`（`native_ubr_integration.py:54`）吻合；返回
  `MaskHeadOutput` 原样透传。
- **唯一调用点**：`grep 'mask_head('` 在 `rsprompter/ train/ inference/` 仅命中
  `anchor_roi_head.py:131`（`_mask_forward`），其调用者为 `mask_loss:189` 与
  `predict_mask:417/427`。`shape_prior_injector.py:123` 的 `self.mask_decoder`
  是 injector 自己的 `SmallMaskDecoder`，与 shim 无关。
- **UDPR else 分支**：`mask_head.py:445-447` 在 `decoder_tail_enabled=False` 时按
  3 元组解包，shim 返回 `(refined, iou, tokens)` 吻合。
- **full_image 与 roi_local 同构**：`mask_head.py:325-450` 的 forward 不含任何
  `final_mask_coordinate_mode` 分支；该模式只在 `mask_head_targets.py:307`
  （目标构造）、`mask_head.py:677`（预测后处理）、`anchor_roi_head.py:267`
  （UDPR 尾损失，已关停）与 `mask_head_init.py:86`（loss 配置校验）出现。实测两
  config 的 decoder 输出均为 `(N, 1, 256, 256)`、`native_hw=[256, 256]`。refined
  logits 作为 `mask_preds` 进入 `loss_and_target`（`anchor_roi_head.py:197-203`），
  full_image 下目标尺寸取自 `mask_preds.shape[-2:]`。
- **零初始化恒等（实测）**：两 config 各自在同一输入、同一 3 个 RoI 上，安装前后
  `_mask_forward` 输出 `torch.equal` 为 True；把 `out.bias` 置 0.25 后输出差恰为
  0.25（最大偏差 1.2e-7），证明 refiner 真在数值路径上；反传后 `out.weight/bias`
  梯度非零，隐藏层梯度在零初始化时为 0（符合预期）。
- **`_empty_rank_losses`**：见 F1。回答清单原问：float-only guard 并没有机会
  “破坏非 float 条目”，因为在那之前就对元组做了下标访问而崩溃。

### 3. 训练器接线 — PASS（代码路径论证；DDP/AMP 未实跑）

- **安装点**：`train/train_rsprompter_fusion.py:179-180` build + `to(device)` →
  `:182-199` 安装 → `:222` DDP → `:398` `_build_optimizer` → `:477` EMA →
  `:520` resume → `:577` clip 分组。顺序符合要求。
- **segm_score 检查**：`:201-209` 用 `getattr` 读 `segm_score_mode/key`，
  `head.__class__` 重赋不改实例属性，检查照常通过。
- **resume**：`_load_checkpoint`（`train/checkpointing.py:110-153`）在安装之后执行，
  默认 `strict=True`；键布局为 `mask_decoder.base.*` + `native_refiner.*`，实测
  “新 build + install 后 strict 加载”成功、“未安装时 strict 加载”报 missing/
  unexpected。漏传 `--native-ubr` 续训会响亮失败而非静默错配。
- **EMA**：`train/ema.py:19-20` 基于 `state_dict()` 枚举，在安装后构造，自然包含
  refiner 与 `base.*` 键；`_frozen_keys` 按 `named_parameters` 的
  `requires_grad` 判定，refiner 可训故正常做滑动平均。
- **AMP**：`native_ubr_refiner.py:164-170` 的 `autocast(enabled=False)` 块与
  `delta.to(logits.dtype)` 随字节拷贝保留；训练器只在外层用
  `scaler.scale(loss).backward()` / `unscale_` / `step`（`train/loops.py:193,
  217-226`），不触及头内精度。逻辑上契约不变，但我没有在 AMP 下实跑。

### 4. optimizer — PASS（C1 已于 483957f9 修复并复验）

- **aux-clip**：`train/optimizer.py:209-215` 加入前缀；`_partition_aux_clip_params`
  用 `_get_model_core` 后的名字做 `startswith`。实测 aux 组恰为 1 组 6625 参数、
  base 663 个张量。WHU 无 `--decoupled-aux-clip` 时 `decoupled_clip_partition`
  为 None，走 `loops.py:224` 全局 clip，前缀表不被读取；NWPU 走
  `loops.py:219-222`，base 与 refiner 各 0.5。
- **参数组**：refiner 落 `other` 组，实测 `(lr=5e-4, wd=0.05)`，两 wrapper 的
  `other-lr-mult` 均为 1.0（WHU 显式、NWPU 取 CLI 默认 `cli_args.py:94`）。
- **`--decoder-tail-lr-mult 1.0`**：无 `decoder_tail_refiner` 参数时
  `params_decoder_tail` 为空，`optimizer.py:145` 的 `if` 跳过，安全 no-op。
- **C1（已修复，commit 483957f9）**：shim 把 decoder 键改名为
  `mask_decoder.base.decoder.*` 后，`optimizer.py` 原 `:59` 的子串
  `roi_head.mask_head.mask_decoder.decoder.iou_prediction_head` 不再命中，IoU 头
  6 个张量（132,612 参数）从 `iou_head` 组落入 `mask_decoder` 组（初审实测）。
  修复后（`train/optimizer.py:59-65` 补 shimmed 路径）复测：安装前后除 refiner
  外无参数换组，`iou_head` 组 132,612 保持；未安装路径不受影响。

### 5. 推理 marker 分支 — PASS，附 CONCERN C2

- `inference/infer_from_checkpoint.py:186-208`：marker =
  `training_args.native_ubr` 或 `--native-ubr`；与所选权重中是否含
  `.native_refiner.` 键交叉校验。真值表：T/T 安装；T/F 与 F/T 均抛
  `RuntimeError`；F/F 不进入任何分支。
- 顺序：`_register_and_build`（`:181`）→ 安装 → `_load_model_state`（`:229`），
  默认 strict。`--weights ema` 时 `_select_state_dict` 取 EMA 状态做键检测，
  EMA 状态同样含 refiner 键。
- 默认路径零行为变化：F/F 时新增代码只多做一次 `snapshot.get` 与一次纯函数
  `_select_state_dict`（同一调用在 `:231` 本就会执行，异常类型与时机等价）。
- 闭环：快照里的 `model_config` 是安装前的 `cfg.model.to_dict()`
  （`train/config_snapshot.py:249`），不含 UBR；`training_args` 取自
  `vars(args)`（`:105`）含 `native_ubr=True`，续训时保存的快照覆盖当前快照
  （`train_rsprompter_fusion.py:557`）也保留该字段。闭环成立，但未跑 CLI 实证。
- **C2（CONCERN）**：架构指纹（`architecture_contract`）对 UBR 不可见——UBR 臂
  与“同 delta config 但不装尾”的模型指纹相同，`assert_checkpoint_architecture`
  无法区分二者，区分完全依赖 strict 加载与 `training_args` marker。另有三处绕过
  marker 分支直接 `_register_and_build` / `MODELS.build` 的工具
  （`scripts/baselines/dump_coarse_teaser.py:88`、
  `scripts/smoke/check_a3sgm_forward_equivalence.py:125`、
  `scripts/baselines/whu_test_mmdet.py:103`）对 UBR checkpoint 会在 strict 加载处
  失败；不是本次协议的评测入口（`scripts/infer_{whu,nwpu}_checkpoint.sh` 均走
  `infer_from_checkpoint.py`），仅提示。

### 6. config / wrapper 保真 — PASS

- **delta config**：两文件只覆盖 `backbone.checkpoint_path`、
  `backbone.init_cfg.checkpoint`、`mask_head.sam2_mask_decoder.checkpoint_path`
  与 `decoder_tail_refiner_cfg=dict(_delete_=True, enabled=False)`。前三者键名均
  在 `rsprompter/architecture_contract.py:13` 的
  `_PATH_KEYS = {"checkpoint", "checkpoint_path", "ckpt_path"}` 内，被归一化为
  `<EXTERNAL_CHECKPOINT>`（`:32-33`）。测试
  `test_delta_fingerprint_touches_only_udpr_key` 把论文的 UDPR cfg 还原进 delta 后
  指纹等于论文指纹，实跑通过——证明差异恰为该键。
- **wrapper 逐行 diff**（对 `whu1024_a3sgm_roilocal_full_fast.sh` /
  `vhr10_a3sgm_matrix300.sh`）：除头部注释外，差异恰为 `RUN_TAG`、
  `master_port`（32633→32634 / 32808→32809）、config 路径（NWPU 为
  `ABLATION_CONFIG` 默认值）、新增 `--native-ubr`。没有其他超参被改动。
  `RUN_TAG` 与预登记一致。
- 说明：`AMP/BATCH_SIZE/GRAD_ACCUM_STEPS/NPROC_PER_NODE` 来自环境装载
  （`scripts/load_environment.sh:30-32`），与基线脚本同源，不在 diff 内；WHU 的
  fp32 取决于启动时环境里的 `AMP=0`，wrapper 契约 echo 时需核对。

### 7. 测试质量 — CONCERN（C3，部分修复；不阻断）

实跑 `tests/test_native_ubr_single.py`（483957f9 后）：**12 passed**（CPU，含
heavy）。无伪测试，断言都对着真实对象。

- 已覆盖：6625 参数与零初始化（refiner 级）、安装回执与守卫、RNG 局部性、aux-clip
  分组、两 config 真实 build + 键布局、论文指纹 parity（`5d5f78ec…`/`ed7a422d…`）、
  delta 仅触及 UDPR 键；483957f9 新增 shim 上下文守卫（`test_shim_context_guards`）
  与空正样本元组契约 + refiner 收梯度 backward 实证
  （`test_empty_rank_losses_tuple_contract_and_refiner_graph`，真实 build + install
  路径，旧代码形态会在此直接 TypeError——回归有效）。
- 仍缺（非阻断，建议在 GPU smoke 前后顺手补进仓库）：
  1. **没有任何测试让数据流过 `NativeMaskHead.forward` → `NativeDecoderShim.forward`
     全链**——预登记探针门的“安装前后同 batch 前向逐位一致，两坐标契约分支各测”
     目前只有 refiner 级与审计外部证据（本轮复审已在仓库外复测两 config
     `torch.equal` 通过，但仓库内无对应测试）。
  2. “保存→新 build+install→strict 加载”round-trip 与推理 marker 真值表测试。
  3. 优化器分组断言（refiner 落 `other`、IoU 头组保持）。

### 8. 风险盲区 — 见下

- **F1**：如上。
- **C1 / C2 / C3**：如上。
- **val 路径**：`predict_mask`（`anchor_roi_head.py:388-441`）经 `_mask_forward`
  调头，boxes/roi_img_ids 始终带上；分块推理是顺序多次调用，每次进出都清
  `call_context`，不触发“not reentrant”。
- **DDP**：`find_unused_parameters=True`，refiner 在有正样本的 rank 上收梯度，无
  正样本 rank 上不需要零图项也不会报 unused 错；因此 F1 的修复不改变梯度语义。
- **`--freeze-bn`**（C4，CONCERN）：`_set_norm_eval`（`train/optimizer.py:194-201`）
  会把所有 `GroupNorm` 置 eval 并冻结其参数，且在安装之后执行
  （`train_rsprompter_fusion.py:218-219`），会冻结 refiner 的 3 个 GN8。两 wrapper
  均未传该旗标，当前不触发；若日后叠加需注意。
- **`torch.compile`**：训练器内未见使用（`grep` 无命中），不涉及。
- **frozen_modules**：`anchor_detector.py:46` 仅在 `decoder_freeze` 时登记原
  decoder 对象，shim 包裹的是同一对象，引用仍有效。
- **`_log_trainable_state`**（`train/telemetry.py:81`）：对 shim 调
  `parameters()`，统计含 `base`，refiner 不在其中（挂在 head 上）；日志里没有
  refiner 单独一行，仅观测性问题。
- **流程项（非本两 commit 的缺陷）**：`docs/EXP_REGISTRY.md` 中尚无 EXP-0046/0047
  条目；按规程属全量启动前置项，拉卡前需补登记并打 tag。

## 复审结论与遗留事项

- F1（483957f9）修复后复审通过，无新增阻断项，总 VERDICT 改判 **PASS**。
- 仍开放（非阻断）：
  - C2：架构指纹对 UBR 不可见（区分靠 strict 加载 + `training_args` marker）；
    三个绕过 marker 分支的工具脚本对 UBR checkpoint 会 strict 失败（非本次评测入口）。
  - C3 剩余缺口：模型级安装恒等前向、round-trip、marker 真值表、优化器分组断言
    未入仓库（复审已在仓库外复测通过恒等与 round-trip）。
  - C4：`--freeze-bn`（安装后执行）会冻结 refiner 的 GroupNorm；两 wrapper 均未
    传，当前不触发。
  - 化妆项：宿主 `anchor_roi_head.py:219` 的 `-> dict` 陈旧注解建议顺手更正。
  - 流程项：`docs/EXP_REGISTRY.md` 尚无 EXP-0046/0047 条目；按规程属全量启动前置
    （发号 + commit message 携带 EXP-ID + 打 tag），拉卡前需补。

## GPU smoke 补记（2026-11-14，实施者报告；审计未复现，待共租卡空位）

- **NWPU AMP 2-rank GPU smoke**（共租卡）已推进至首个 `mask_loss` 的 refiner
  前向：安装回执 full_image / 6625 参数 / 22 键、DDP backward、aux 组恰为
  22 张量（6625 参数）均实证。
- 中断原因为**共租卡物理显存不足**（约 9G free vs 1024 负载），非代码问题；
  判定为环境容量限制，不构成对本实现的证伪。
- **GPU smoke 补完（含 WHU fp32 分支与 checkpoint→推理 rebuild→strict 加载
  round-trip）按用户裁定延后**：等双流臂（I-bidir E22、no20-IT E17）跑完后在
  空出的卡上执行。此项为预登记探针门的组成部分，全量拉卡前必须完成——延后是
  排程决定，不是豁免。
