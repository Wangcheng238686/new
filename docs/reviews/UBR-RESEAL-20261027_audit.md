# UBR-RESEAL-20261027 独立预发射审计（实施 + 接线）

- 审计对象：commit `6ae7310`（4 文件：`configs/ablations/ue_ubr_single_512.py`、
  `rsprompter/uav/ubr_train_integration.py`、`scripts/probes/ue_ubr_reseal_20261027.py`、
  `scripts/probes/ue_ubr_reseal_harness_20261027.py`），对照
  `docs/UBR-RESEAL-20261027_preregistration.md` 与 AGENTS.md 拉卡前置五条。
- 审计方式：只读代码审读 + harness 独立重跑一次（GPU3，`CUDA_VISIBLE_DEVICES=3`，
  torch 2.8.0+cu128，报告写 `/tmp/ubr_reseal_harness_audit_rerun.json`，未覆盖原报告）+
  只读 smoke/harness 工件 + 数据集统计与 CPU 微基准。未修改任何文件、未启动训练。
- 审计日期：2026-10-26（实施提交 2026-10-27 13:48 +0800 落库后）。

## VERDICT: PASS

实施与接线忠于预注册；harness 七项检查非空洞且独立重跑 7/7 PASS；正向数据流审计
未发现正确性阻断项。下列发现均为发射前置的运维/治理事项，不构成 FAIL，但 Chief
须在拉卡前逐条处置（见文末清单）。

## 一、harness 独立重跑结果（7/7 PASS）

| 检查 | 结果 | 关键细节 |
| --- | --- | --- |
| config_diff_exact | PASS | added=8（ubr_integration_cfg 全键）/ removed=8（decoder_tail_refiner_cfg 全键）/ changed=0；与封版 `configs/ue_whu_a3sgm_512.py` 的扁平叶路径 diff 恰为预注册键集 |
| old_seal_untouched | PASS | frozen_refs SHA256SUMS 两文件（e73 dd9106f4…、v1best 2fcae2ff…）重算一致；EXP-0025 `best_model_epoch72.pth`=2fcae2ff…、`last_model_epoch100.pth`=736c1532… 未漂移 |
| dataset_sha | PASS | 7 个 SHA pin + label_revision review99_v3_user_approved_20261021 + 2142/459/459（126214/14980/44988 实例）+ 512×512 |
| launch_preconds | PASS | 输出根目录不存在；runner PROTOCOL 与封版逐项相等（seed44/b1/a2/lr5e-4/wd.05/warmup100/100ep/flip.5/vflip0/ms0） |
| model_build_ctor | PASS | 163,537 参数（笔算复核：input_proj 17,328 + 7×ConvBlock 20,880 + out 49）；全 trainable；out 层零初始化；state_dict 无 decoder_tail_refiner 键；config 内所有 ckpt 路径 == SAM2 ctor 权重；no-mask embedding 契约成立 |
| fwd_bwd_finite | PASS | loss dict 含 loss_ubr=0.0605、不含 loss_decoder_tail；全损失有限；零初始化下 out.weight/out.bias 梯度非零非空；扰动读出后 loss_ubr 单独 backward 使 decoder `output_upscaling` 6 个参数张量获非零梯度（①–②回流路径实证）；ubr_instances=64、band_nonempty_frac=1.0 |
| predict_refine | PASS | 零初始化 changed_px=0（精确复制）；读出置偏（bias=5）后 changed 64,104 ≤ band 68,060，且集成内逐实例断言改动 ⊆ 带 |

七项检查均实测其所声称的内容，非空洞；其中 old-anchor 的 frozen_refs 校验以
SHA256SUMS 内路径重算哈希，且 2fcae2ff/736c1532 同时是代码内硬编码常量交叉钉死，
不存在"改清单文件即可通过"的空洞通道。

## 二、正向数据流审计（任务重点项）

### (a) 可微 `_do_paste_mask` / AMP / DDP
- dtype：`z.float()` 后 sigmoid+grid_sample 全链 fp32；`band_loss` 内 `BCEWithLogits`
  前显式 `.float()`（autocast 安全）；eval 侧 UBR 前向自带 `autocast(enabled=is_cuda)`，
  与训练侧（trainer autocast）数值口径一致。未见精度陷阱。
- 图保留：stash 在 mask_loss 入口/出口 `_release()`，计算图由损失张量自身持有，
  生命周期正确；mask head 前向每次 loss 恰触发一次（roi_head 单次 `_mask_forward`，
  miner 不重复前向），`z.shape[0]==n_pos` 断言成立（smoke 40 个场景-epoch 实证）。
- DDP：trainer 本就以 `find_unused_parameters=True` 包裹（PE 冻结所需）；
  n_pos==0 时 loss_ubr 返回与全部 UBR 参数相连的零张量（`sum(p.sum())*0`），
  全空带极端例（band_loss 退化为 new_zeros，UBR 参数当步 unused）由
  find_unused=True 容忍。UBR 在 `__init__` 内注册为子模块 → DDP 桶、optimizer、
  checkpoint、state_dict 自动涵盖（smoke ckpt 实测含 32 个 `ubr_refiner.*` 键，
  resume 目录 ep3 证明断点续跑含 UBR 严格加载）。
- paste 一致性：mmdet 3.3.0 roi_local 路径与本模块重贴用同一 grid_sample 数学
  （skip_empty 仅改变采样窗，逐像素位级一致）；eval 无 autocast，z 为 fp32，
  parity 断言在真实 val batch 上通过。
- 命名注意：UBR 输入通道 "mask_logits" 实际喂的是 sigmoid 概率图（train/predict
  两侧一致，且与 level-2 探针约定相同）——命名误导但语义一致，非缺陷。

### (b) GT 邻居装配
逐图分组（pos_counts 偏移）正确；`pos_assigned_gt_inds` 索引本图 GT 正确；
`nbr_full = 全部 GT 之和 − 自身 GT` 的自排除正确；numpy uint8 求和提升 uint64
无回绕。GT 掩码为 BitmapMasks（gt_samples 实证），`hasattr(masks,"masks")` 分支
正确。注意：训练用 GT 邻居 / 推理用预测邻居是**文档化的口径选择**（config
docstring + 设计文档），与 level-2 冻结串联探针（全预测邻居）不同——已在预注册
范围内，作观察项不作缺陷。

### (c) 逐实例 window/SDT 开销（CPU）与显存
- EXP-0025 遥测 `debug_mask_pos_rois=64.0`：每步正样本恒为 64（采样上限）。
- 本机实测 scipy EDT（含正负变换）1.97ms@150²、3.51ms@200²、6.05ms@260²、
  8.57ms@310²；GT bbox p50=39px/p90=91px/p99=151px → 逐实例窗 p50≈103px，
  64 实例 EDT+装配 ≈ 0.13–0.20s/步。
- EXP-0025 基线（同机 4090 共租）：训练 ~559s/536 步 ≈ 1.04s/步，val 459 约
  70s + COCO 15s，全程 ~640s/epoch。
- 估计训练步开销 +15–25%（< 30% 阈值，不 flag 但接近；EDT 为损失路径上的
  单线程 CPU 串行段，共租 CPU 争用可能上推）。epoch 级（含 val 逐实例重分类
  ~0.1s/图·早期最多 0.25s/图）估计 +20–30%，100ep 总时长约 +4–5h。
- 显存：`_pad_batch` 将 64 个窗补齐到批内最大窗（数据集上限 ≈311px），
  UBR 激活估计 **+3.5–4.5GB（典型）至 ~8–9GB（p99 大建筑批）**/rank。
  smoke 在空闲 GPU3（48GB）上通过；当前 GPU0–2 被共租占 36.5GB/卡（余
  ~12.5GB）——**拉卡前必须按宿主核实每 rank 余量 ≥ 基线峰值 + ~10GB**
  （或等共租释放/换 lthpc 宿主，预注册允许）。harness 未测显存，是覆盖缺口。

### (d) 零初始化 UBR + λ=0.1 的梯度尺度
loss_ubr(0) = 0.1×(focal 0.173 + dice ≈0.43) ≈ 0.060，与 harness 0.0605 及
smoke 0.062–0.075 一致；对总损失占比 ~2%（loss_mask 2.43、rpn_cls 0.70、cls
0.65），逐实例等权归一化密集图；零初始化读出使上游梯度为 0（中性起步，同
level-2 性质），out.* 自第一步即获非零梯度（harness 断言实证）。UBR 参数入
optimizer 默认 "other" 组（lr 5e-4、wd 0.05），无前缀误撞。量级合理。

## 三、Runner 与 EXP-0025 argv 逐位比对

与 `scripts/ue/run_exp0025_ddp4.sh` 的 torchrun 调用相比，trainer argv 仅含
注册差异：`--config`（新臂）、`--run-tag`/`--checkpoint-dir`（新目录）、
`--epochs`/`--train-subset-ratio`/`--val-subset-ratio`/`--num-workers`（仅 smoke
传入；全量 num_workers=4 与 trainer 默认一致，为无操作差异）、可选
`--resume-from`。其余（seed/b1/a2/lr/wd/warmup/patience 0/val b1/val-every 1/
compute-val-loss 0/amp 1/ema 0/ema-eval 0/--dataset whu/--image-size 512 512）
逐字一致；env（SAM2_CKPT、WHU_*、TRAIN_FLIP/VFLIP/MS、SAVE_LAST_MODEL）与
wrapper 逐字一致。**git diff `EXP-0025..6ae7310` 限于 train/、封版 config、
cspm/mask_head、detection/、data/loader.py 为空**——封版训练器代码位级未动。
审计门（`--audit-verdict` 必含 VERDICT: PASS）、输出根不存在断言、resume 路径
（目录存在 + ckpt 文件 + 严格加载含 UBR 键）均在。test 不可达（runner 无 test
路径；`create_test_loader` 仅收线时独立使用，trainer 不调用）。

## 四、发现（编号，无 FAIL 级）

1. **[发射风险-高] GPU 余量未验证**：UBR 新增激活估计 3.5–9GB/rank（见二(c)）；
   当前 GPU0–2 各被共租占用 36.5/49GB。拉卡前须实测/核实宿主余量，或在
   lthpc（compact 收线后）发射。harness 无显存检查项（建议未来 harness 加
   `torch.cuda.max_memory_allocated` 记录）。
2. **[治理缺口-中] 运行树 SHA 契约未实现**：`/data2/wangcheng/remote-lthpc/
   portable_sam2_explicit_coarse` 为陈旧异源树（无 `configs/ue_whu_a3sgm_512.py`、
   无 4 个新文件、trainer 属旧 dual 支系）。runner 未实现 AGENTS.md 运行树纪律
   的"脚本 blob SHA == git"断言；且在非 git 目录运行时 `git status`/`rev-parse`
   门与 RESEAL_CONTRACT 的 commit 字段**空洞通过/为空**。缓解：从 worktree
   （恰在 6ae7310，git 门生效）发射；若须在运行树发射，先全量同步 6ae7310 并
   由 wrapper echo 人工核对 commit。
3. **[运维-低] worktree 未洁**：`docs/ops/` 未跟踪（本审计文件亦将加入）。
   从 worktree 发射会被 clean-tree 断言拒绝（fail-closed 方向正确）——发射前
   须一并 commit（顺带满足五条①的 message 携带 EXP-ID 要求）。
4. **[开销-观察] 步时 +15–25%、epoch +20–30% 估计**（二(c)），低于 30% flag
   线但无裕量；建议发射后头两 epoch 盯 `TRAIN/TIME`（SEGMENT_TIMING=1）与
   epoch 墙钟，若 >30% 记录并接受（协议不可中途改）。
5. **[口径-观察] 邻居通道 train=GT / predict=pred** 与 level-2 全预测口径不同，
   属预注册文档化选择；收线归因时勿混用两套口径的机制遥测。
6. **[命名-观察] UBR 输入 "mask_logits" 实为概率图**（train/predict/level-2
   三处一致），建议后续 PR 只改注释不改数值。

## 五、Chief 拉卡前清单（AGENTS.md 五条对照）

- [ ] ① 代码 commit 携带 EXP-ID 并打 tag `exp/EXP-XXXX`：6ae7310 已含臂名但
      EXP-ID 未发；发号后需一个携带 EXP-ID 的 commit（可与 docs/ops/、本审计
      文件、registry 同 commit）并在其上打 tag。
- [ ] ② `docs/EXP_REGISTRY.md` 发号登记 UBR-RESEAL-20261027（当前无条目）。
- [ ] ③ 全臂 harness 通过：已有（原报告 + 本审计独立重跑均 7/7 PASS）。
- [ ] ④ 本审计 VERDICT: PASS 落盘（本文件）。
- [ ] ⑤ wrapper 契约 echo 核对：发射时核对 RESEAL_CONTRACT 的
      config/protocol/dataset/out_root/audit_verdict/commit 字段齐全（注意发现 2：
      commit 为空即说明在非 git 树发射，须停止并核对）。
- [ ] 宿主/GPU 决策：按发现 1 核实 4090 共租余量或改 lthpc（预注册允许），
      wrapper 声明宿主；单卡 b1×a8 降级备案须 wrapper 声明后方可使用。
- [ ] 推送确认：origin/codex/dual-stream-dev 已在 6ae7310（核实通过）；发号
      commit 后再推一次。

## 六、审计限制

- 未运行训练（含 smoke 复跑）；运行时行为证据取自 smoke/resize 工件、一次
  harness 重跑与 EXP-0025 历史日志外推。
- 显存为解析估计（激活张量逐项推算），非实测峰值；开销为微基准+遥测外推。
- 未复核 level-2 三门证据本身（冻结诊断级证据按预注册采信）。
