# SURFBEV-P1-V2 R1a 结果（2026-10-13）

预注册门的计算结果：**P1_FAIL，不放行 H1**。三臂固定特征读出完成，不属于检测/分割
AP实验；不得用局部F1正效应替代全部伴随门，不按test回调任何读出设计。

## 完整性

运行 2026-10-13 11:18:40–13:05:26 +08:00，supervisor记录worker正常退出code=0。
train/val/test分片数694/100/151，遥测945条，test逐场结果151条、scene ID唯一。
选择 C=10、threshold=0.6，由val确定。test锁存在；未重开或按test调参。
主代理从per_scene复算全部六项scene-paired bootstrap，10,000 draws、seed20261012，
与summary逐项一致。独立结果复核见下方审计路径（以该报告最终VERDICT为准）。

## 三臂均值

| 指标 | SAT | correct | sham90 |
|---|---:|---:|---:|
| roof-boundary F1 | 0.605611 | 0.612826 | 0.548516 |
| boundary-error AUROC | 0.938272 | 0.934503 | 0.920389 |
| instance IoU prediction Spearman | 0.459198 | 0.593888 | 0.589286 |

## 预注册门逐项判定

P1_PASS要求以下六项95% CI下界全部>0，并且完整性/parity/标签隔离/coverage无异常。

| 差值 | mean | 95% CI | 通过 |
|---|---:|---|---|
| correct−SAT F1 | +0.007215 | [+0.005009,+0.009410] | 是 |
| correct−sham90 F1 | +0.064310 | [+0.062302,+0.066300] | 是 |
| correct−SAT AUROC | −0.003770 | [−0.004823,−0.002749] | 否 |
| correct−sham90 AUROC | +0.014114 | [+0.013356,+0.014881] | 是 |
| correct−SAT Spearman | +0.134690 | [+0.117494,+0.152533] | 是 |
| correct−sham90 Spearman | +0.004602 | [−0.003884,+0.013357] | 否 |

## 结论边界与下一步

这份结果支持当前冻结读出中正确SurfaceBEV有小幅的roof-boundary可读取信息增益：
F1比SAT提高0.72个百分点，且优于90°错误几何。它不支持三个信息维度全面改善。
错误定位相较SAT下降，实例质量预测的收益尚不能归因于正确几何（correct−sham CI跨0）。
不能据此声称UAV无用、SAM2无用或多高度无用；当前测试也未单独比较单高度和12-bin。
依预注册关闭当前接线、不进入H1；任何进一步假设需另行预注册，不能修改本次门或
重复利用这151场test选参。boundary F1不是mask/bbox AP。

## 工件

根目录：`/data1/wangcheng/checkpoint/surfbev_b0/probes/SURFBEV-P1-V2-R1a-20261012/`

- `summary.json`：三臂均值、CI和判定。
- `per_scene.jsonl`：151场结果。
- `cache/contract.json`：B0/H0/五个实现SHA。
- `cache/test_opened_once.json`：冻结选择锁。
- `cache/telemetry_train_val.jsonl`、`cache/telemetry_test.jsonl`：945场遥测。
- `evaluation_runtime.json`、`evaluation_supervised_20261013.log`：退出码及运行日志。
- 预注册：`docs/SURFBEV-P1-V2_preregistration.md`。
- 实现审计：`docs/reviews/SURFBEV-P1-V2-R1_recovery_audit.md`。
- 独立结果审计：`docs/reviews/SURFBEV-P1-V2_result_audit_20261013.md`。

此为冻结探针，无全量EXP-ID；锚点为EXP-0006 seed44 last。
