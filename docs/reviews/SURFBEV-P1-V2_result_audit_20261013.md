# SURFBEV-P1-V2 R1a 结果独立审计（2026-10-13）

**审计对象：**
`/data1/wangcheng/checkpoint/surfbev_b0/probes/SURFBEV-P1-V2-R1a-20261012/`

**审计范围：** 只读检查 `summary.json`、`per_scene.jsonl`、cache 合同/分片/遥测/test lock/
runtime 和归档源码指纹；仅从已封存的 151 场 `per_scene.jsonl` 重放预注册 bootstrap。未拟合
读出器、未改选择参数、未执行 runner、未读取原始 test scene，未改动实现。

## 最终判定

**VERDICT: RESULT AUDIT PASS；预注册结论为可信的 `P1_FAIL`。**

没有发现会推翻本次失败判定的完整性、test 开封顺序、缓存、遥测或统计复现问题。依预注册，
不得进入 H1，也不得按本次 test 再调 C、阈值、SurfaceBEV 表达或读出设计。

## 封存与执行完整性：通过

- `evaluation_runtime.json` 记录运行于 2026-10-13 11:18:40–13:05:26 +08:00，原始 runner
  参数为 `--phase evaluate --device cuda:0 --allow-test`，worker exit code 为 **0**。
- 缓存分片为精确连续索引：train `0000..0693`（694）、val `0000..0099`（100）、test
  `0000..0150`（151）；三个 split 的 scene ID 各自唯一且互不相交。
- `train_val_complete.json` 为 `{train:694,val:100}`；`test_opened_once.json` 已封存唯一选择
  `C=10.0, threshold=0.6`，并写明选择在 test 前完成。test cache 151 个、`per_scene.jsonl`
  151 行/151 个唯一 scene ID，且 per-scene 顺序与 test cache 顺序逐项一致。
- train cache 检出全部预期 readout key；val/test cache 检出 `yb/ye/pred/qi` 与三臂各自
  `x_*` key。没有发现 split 混入或不完整编号。

## 合同、源码及遥测：通过

- `cache/contract.json` 锁定 B0 SHA
  `fc3ead0e…ce65145`、H0 SHA `765a3e…31d86a`、v2 split、12 个 height edges、三臂、64 维和
  每类每 scene 256-cap；H0 文件 SHA 复算一致。
- 当前工作树已在 P1 完成后迁移，不能以当前文件字节直接对照四个 runner 依赖；但合同中的
  五个实现 SHA 均与归档 ref `codex/archive-dual-stream-snapshot-20261013`
  (`7659eb7a853e0178a76789a60d2303493664669d`) 的相应文件逐字节匹配。因此本次结果仍有可
  定位的源码快照；不得拿当前迁移后的文件重开或续跑此 cache。
- 遥测为 train+val 794 行、test 151 行，共 945 个唯一 scene ID。每行 descriptor shape 均为
  `[5,256,128,128]`；correct/sham 输入点相等，z-buffer points 与 covered cells 均为正，
  z-buffer residual 均有限。train/val 的 correct/sham 合并统计 covered-cell 中位数
  14,680.5（最小 6,964），test 中位数 15,070（最小 14,563）；未见 coverage 或深度残差契约
  异常。

## 统计重放：六项均逐字节一致

以预注册 seed `20261012`，从封存的 per-scene 差值进行 151-scene paired percentile bootstrap
（10,000 draws）。下表每项的 mean/CI 与 `summary.json` **完全一致**：

| 差值 | mean | 95% CI | 门 |
|---|---:|---|---|
| correct − SAT F1 | +0.007215 | [+0.005009, +0.009410] | 通过 |
| correct − sham90 F1 | +0.064310 | [+0.062302, +0.066300] | 通过 |
| correct − SAT AUROC | −0.003770 | [−0.004823, −0.002749] | **失败** |
| correct − sham90 AUROC | +0.014114 | [+0.013356, +0.014881] | 通过 |
| correct − SAT Spearman | +0.134690 | [+0.117494, +0.152533] | 通过 |
| correct − sham90 Spearman | +0.004602 | [−0.003884, +0.013357] | **失败** |

预注册 `P1_PASS` 要求上述六项的 CI 下界都严格大于零。两项失败足以、也必须判为
`P1_FAIL`；这不是因 effect size 很小而可酌情放宽的情形。

## 可以与不可以得出的结论

**可以说：** 在本次冻结特征/线性读出设置中，正确几何的 SurfaceBEV 对 roof-boundary F1
有可重复的小幅读取增益（相对 SAT +0.72 个百分点，且优于 90° sham），并改善实例 IoU
可预测性相对卫星单流。

**不可以说：** 它没有在全部预注册信息维度上提供一致增益；相对 SAT 的 boundary-error
AUROC 是显著下降的，而实例 IoU 质量的 correct 相对 sham 优势不能排除为零。因此不能称
本 SurfaceBEV 接线“通过信息门”，不能据此训练 H1，也不能声称 UAV 提升了 bbox AP、mask
AP、整体实例分割或检测。该探针也没有比较单高度与 12-height-bin，不能把失败外推为
“多高度 BEV 无用”或“UAV 无用”。

## 归档工件

- 结果：`summary.json`、`per_scene.jsonl`
- 封存/遥测：`cache/contract.json`、`cache/train_val_complete.json`、
  `cache/test_opened_once.json`、`cache/telemetry_train_val.jsonl`、
  `cache/telemetry_test.jsonl`
- 运行证据：`evaluation_runtime.json`、`evaluation_supervised_20261013.log`
- 协议：`docs/SURFBEV-P1-V2_preregistration.md`
- 实施审计：`docs/reviews/SURFBEV-P1-V2-R1_recovery_audit.md`
